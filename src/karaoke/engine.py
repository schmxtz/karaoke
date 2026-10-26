"""Real-time karaoke engine: plays the stems, listens to the microphone and scores the singer.

Timing: playback and capture share one duplex stream. A mic frame is stamped
with the song position the singer was *hearing* when they sang it, i.e. the
playback position minus input + output latency (plus a user-tunable offset).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr

from karaoke.library import Song
from karaoke.pitch import MedianSmoother, NoteSnapper, PitchTrack, YinTracker, fold_octave

DEFAULT_MIC_THRESHOLD_DB = -50.0  # quieter input is treated as silence

# Mixer channels -> stem files; "guide" is the original vocals the singer replaces.
CHANNELS = {"music": "instrumental", "guide": "vocals"}


@dataclass(frozen=True)
class VoiceFrame:
    t: float  # song time in seconds
    midi: float  # detected pitch, NaN when not singing
    note: float  # `midi` snapped to a held semitone, NaN when not singing
    hit: bool


class Scorer:
    """Share of sung reference frames where the singer held the reference note.

    Lenient on purpose: the first/last SLACK frames of every phrase are not
    scored, and a hit counts for reference frames up to SLACK frames away.
    The same note in another octave counts as a hit.
    """

    SLACK = 5  # reference frames (50 ms)

    def __init__(self, reference: PitchTrack):
        self.ref = reference  # semitone notes, see segment_notes
        voiced = ~np.isnan(reference.midi)
        width = 2 * self.SLACK + 1
        self.scorable = np.convolve(voiced, np.ones(width), mode="same") >= width
        self.hits = np.zeros(len(voiced), dtype=bool)
        self.start = 0

    def reset(self, start_time: float = 0.0) -> None:
        self.hits[:] = False
        self.start = int(round(start_time / self.ref.hop))

    def add(self, t: float, note: float) -> bool:
        target = self.ref.at(t)
        if np.isnan(target) or np.isnan(note) or fold_octave(note, target) != target:
            return False
        i = int(round(t / self.ref.hop))
        self.hits[max(i - self.SLACK, 0): i + self.SLACK + 1] = True
        return True

    def score(self, now: float) -> float | None:
        """Percent of scorable frames between the start and `now` that were hit."""
        end = int(round(now / self.ref.hop)) - self.SLACK
        window = self.scorable[self.start:end]
        if not window.any():
            return None
        return 100.0 * self.hits[self.start:end][window].mean()


def pick_devices() -> tuple[int | None, int | None, int]:
    """Prefer WASAPI (a few ms latency) over the MME defaults (~90 ms)."""
    for api in sd.query_hostapis():
        if "WASAPI" in api["name"] and api["default_input_device"] >= 0 and api["default_output_device"] >= 0:
            in_dev, out_dev = api["default_input_device"], api["default_output_device"]
            sr = int(sd.query_devices(out_dev)["default_samplerate"])
            try:
                sd.check_input_settings(in_dev, channels=1, samplerate=sr, dtype="float32")
                sd.check_output_settings(out_dev, channels=2, samplerate=sr, dtype="float32")
                return in_dev, out_dev, sr
            except sd.PortAudioError:
                break
    return None, None, int(sd.query_devices(kind="output")["default_samplerate"])


class KaraokeEngine:
    def __init__(self, song: Song, devices: tuple[int | None, int | None, int] | None = None):
        """Decodes the stems (slow, fine on a worker thread). Call `start()` to open the audio devices."""
        self._fixed_devices = devices is not None
        self._in_dev, self._out_dev, self.sr = devices or pick_devices()
        self.song = song
        self.gains = {"music": 1.0, "guide": 0.0}
        self.tracks = self._load_tracks(song)
        self.latency_offset = 0.0  # seconds, tuned by the user if the voice line looks early/late
        self.output_latency = self.round_trip = 0.0

        self.scorer = Scorer(song.notes)
        self.level_db = -120.0
        self.yin: YinTracker | None = None
        self._mic_threshold_db = DEFAULT_MIC_THRESHOLD_DB
        self._voice: list[VoiceFrame] = []
        self._lock = threading.Lock()
        self._pos = 0  # next sample to play
        self._clock = (0.0, time.perf_counter())  # (song time of last callback, when it ran)
        self.playing = False
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._stream: sd.Stream | None = None

    def start(self) -> None:
        """Open the duplex stream. Call on the main thread: WASAPI fails on threads without COM."""
        try:
            self._open_stream()
        except sd.PortAudioError:
            if self._fixed_devices:
                raise
            # WASAPI device busy or misbehaving: fall back to the system defaults.
            sr = int(sd.query_devices(kind="output")["default_samplerate"])
            self.tracks = {c: soxr.resample(t, self.sr, sr) for c, t in self.tracks.items()} if sr != self.sr else self.tracks
            self._in_dev, self._out_dev, self.sr = None, None, sr
            self._open_stream()
        threading.Thread(target=self._analyse, daemon=True).start()

    def _open_stream(self) -> None:
        self.length = max(len(t) for t in self.tracks.values())
        self.yin = YinTracker(self.sr, min_level_db=self._mic_threshold_db)
        stream = sd.Stream(
            device=(self._in_dev, self._out_dev),
            samplerate=self.sr,
            blocksize=self.sr // 100,  # 10 ms: one pitch frame per block
            channels=(1, 2),
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        try:
            stream.start()
        except sd.PortAudioError:
            stream.close()
            raise
        in_latency, self.output_latency = stream.latency
        self.round_trip = in_latency + self.output_latency
        self._stream = stream

    def _load_tracks(self, song: Song) -> dict[str, np.ndarray]:
        tracks = {}
        for channel, name in CHANNELS.items():
            audio, sr = sf.read(song.stems[name], dtype="float32", always_2d=True)
            tracks[channel] = soxr.resample(audio, sr, self.sr) if sr != self.sr else audio
        return tracks

    @property
    def mic_threshold_db(self) -> float:
        """Mic level (dBFS) below which input counts as silence; lower = more sensitive."""
        return self._mic_threshold_db

    @mic_threshold_db.setter
    def mic_threshold_db(self, db: float) -> None:
        self._mic_threshold_db = db
        if self.yin:
            self.yin.min_level_db = db

    # --- transport -------------------------------------------------------

    @property
    def duration(self) -> float:
        return self.length / self.sr

    @property
    def position(self) -> float:
        """Song time currently coming out of the speakers."""
        song_time, at = self._clock
        if self.playing:
            song_time = min(song_time + time.perf_counter() - at, self._pos / self.sr)
        return max(song_time - self.output_latency, 0.0)

    def play(self) -> None:
        if self._pos >= self.length:
            self.seek(0.0)
        self._clock = (self._pos / self.sr, time.perf_counter())
        self.playing = True

    def pause(self) -> None:
        self.playing = False
        self._clock = (self._pos / self.sr, time.perf_counter())

    def seek(self, seconds: float) -> None:
        self._pos = int(np.clip(seconds, 0, self.duration) * self.sr)
        self._clock = (self._pos / self.sr, time.perf_counter())
        with self._lock:
            self._voice.clear()
        self.scorer.reset(seconds)

    def voice_since(self, t: float) -> list[VoiceFrame]:
        with self._lock:
            i = len(self._voice)
            while i > 0 and self._voice[i - 1].t >= t:
                i -= 1
            return self._voice[i:]

    def close(self) -> None:
        self.playing = False
        if self._stream:
            self._stream.close()
        self._queue.put(None)

    # --- audio thread ----------------------------------------------------

    def _callback(self, indata, outdata, frames, _time, _status) -> None:
        if not self.playing:
            outdata.fill(0)
            self._queue.put((indata[:, 0].copy(), None))  # keeps the mic meter live while paused
            return
        start = self._pos
        end = min(start + frames, self.length)
        mix = np.zeros((frames, 2), dtype=np.float32)
        for channel, track in self.tracks.items():
            gain = self.gains[channel]
            if gain:
                chunk = track[start:end]
                mix[: len(chunk)] += gain * chunk
        outdata[:] = mix
        self._pos = end
        self._clock = (start / self.sr, time.perf_counter())
        if end >= self.length:
            self.playing = False
        self._queue.put((indata[:, 0].copy(), start / self.sr))

    # --- analysis thread -------------------------------------------------

    def _analyse(self) -> None:
        frame = np.zeros(self.yin.frame_length, dtype=np.float32)
        smoother = MedianSmoother()
        snapper = NoteSnapper()
        while (item := self._queue.get()) is not None:
            block, block_time = item
            frame = np.concatenate([frame[len(block):], block])
            midi, self.level_db = self.yin.detect(frame)
            if block_time is None:
                continue
            midi = smoother.push(midi)
            # Centre of the analysed frame (minus the smoother's delay), mapped to what the singer was hearing then.
            t = block_time + (len(block) * (1 - smoother.delay) - len(frame) / 2) / self.sr - self.round_trip - self.latency_offset
            decided = [VoiceFrame(ft, fm, note, self.scorer.add(ft, note)) for ft, fm, note in snapper.push(t, midi)]
            with self._lock:
                self._voice.extend(decided)
