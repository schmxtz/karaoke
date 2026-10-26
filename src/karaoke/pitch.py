"""Pitch tracking: offline reference melody (CREPE) and live microphone pitch (YIN).

Pitches are MIDI note numbers as floats (69.0 = A4 = 440 Hz); NaN means unvoiced.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

REFERENCE_HOP = 0.01  # seconds between reference pitch frames
FMIN, FMAX = 65.0, 1100.0  # C2..C#6 covers sung vocals
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(hz):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 440.0)


def note_name(midi: float) -> str:
    n = int(round(midi))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def fold_octave(midi, target):
    """Shift `midi` by whole octaves to land nearest `target` (singing an octave off still matches)."""
    return midi - 12.0 * np.round((midi - target) / 12.0)


class PitchTrack:
    """Reference melody sampled every REFERENCE_HOP seconds."""

    def __init__(self, midi: np.ndarray, hop: float = REFERENCE_HOP):
        self.midi = midi.astype(np.float32)
        self.hop = hop

    @property
    def duration(self) -> float:
        return len(self.midi) * self.hop

    def at(self, t):
        """Reference pitch at time(s) `t` (NaN outside the song or when unvoiced)."""
        idx = np.round(np.asarray(t) / self.hop).astype(np.int64)
        valid = (idx >= 0) & (idx < len(self.midi))
        midi = np.where(valid, self.midi[np.clip(idx, 0, len(self.midi) - 1)], np.nan)
        return float(midi) if midi.ndim == 0 else midi

    def nearest_voiced(self, t: float, within: float = 1.5) -> float:
        """Reference pitch at the voiced frame closest to `t` (NaN if none within `within` seconds)."""
        lo = max(int((t - within) / self.hop), 0)
        hi = min(int((t + within) / self.hop) + 1, len(self.midi))
        voiced = np.flatnonzero(~np.isnan(self.midi[lo:hi]))
        if voiced.size == 0:
            return np.nan
        closest = voiced[np.argmin(np.abs(lo + voiced - t / self.hop))]
        return float(self.midi[lo + closest])

    def range(self) -> tuple[float, float]:
        voiced = self.midi[~np.isnan(self.midi)]
        if voiced.size == 0:
            return 48.0, 72.0
        return float(np.percentile(voiced, 2)), float(np.percentile(voiced, 98))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, midi=self.midi, hop=self.hop)

    @classmethod
    def load(cls, path: Path) -> PitchTrack:
        data = np.load(path)
        return cls(data["midi"], float(data["hop"]))


def extract_reference_pitch(vocals_path: Path, device: str | None = None) -> PitchTrack:
    """Melody of an isolated vocal stem with torchcrepe (full model, Viterbi decoding)."""
    import soundfile as sf
    import torch
    import torchcrepe

    audio, sr = sf.read(vocals_path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hop = int(round(sr * REFERENCE_HOP))
    hz, periodicity = torchcrepe.predict(
        torch.from_numpy(mono)[None],
        sr,
        hop_length=hop,
        fmin=FMIN,
        fmax=FMAX,
        model="full",
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        batch_size=1024,
        device=device,
    )
    periodicity = torchcrepe.filter.median(periodicity, 3)
    hz = torchcrepe.filter.mean(hz, 3)
    hz, periodicity = hz[0].cpu().numpy(), periodicity[0].cpu().numpy()

    # Voicing: confident pitch and audible level (CREPE reports pitch even in silence).
    n = len(hz)
    frames = np.pad(mono, (0, max(0, n * hop - len(mono))))[: n * hop].reshape(n, hop)
    level_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    loud = level_db > np.percentile(level_db, 99) - 40
    midi = np.where((periodicity > 0.4) & loud, hz_to_midi(hz), np.nan)
    return PitchTrack(_drop_short_runs(midi, min_frames=6))


def _drop_short_runs(midi: np.ndarray, min_frames: int) -> np.ndarray:
    """Unvoice voiced blips shorter than `min_frames`."""
    midi = midi.copy()
    voiced = ~np.isnan(midi)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < min_frames:
            midi[start:end] = np.nan
    return midi


class NoteSnapper:
    """Snaps a pitch stream to semitones without flicker.

    A line starts once one semitone has collected `confirm` frames; all frames up to then are
    back-filled with it, so the line begins at the real onset on the settled note instead of the
    scoop or overshoot of the first few tens of ms. Within a line, moving to another note takes
    `confirm` consecutive frames on it; shorter excursions keep the current note.
    Output lags input by up to `confirm` frames.
    """

    def __init__(self, confirm: int = 8, hysteresis: float = 0.2, max_pending: int = 30):
        self.confirm = confirm
        self.reach = 0.5 + hysteresis  # how far (semitones) a frame may stray and still be on a note
        self.max_pending = max_pending
        self.note: float | None = None
        self._candidate: float | None = None
        self._streak = 0
        self._votes: dict[float, int] = {}  # frames per semitone while a line is starting
        self._pending: list[tuple[float, float]] = []

    def push(self, t: float, midi: float) -> list[tuple[float, float, float]]:
        """Feed one frame; returns the frames decided so far as (t, midi, note), note NaN when silent."""
        if np.isnan(midi):
            out = self._release(self.note) + [(t, midi, np.nan)]
            self.note, self._candidate, self._streak = None, None, 0
            return out
        if self.note is None:
            return self._start_line(t, midi)
        if abs(midi - self.note) <= self.reach:
            out = self._release(self.note) + [(t, midi, self.note)]
            self._candidate, self._streak = None, 0
            return out

        out = []
        if self._candidate is not None and abs(midi - self._candidate) <= self.reach:
            self._streak += 1
        else:
            out = self._release(self.note)  # an excursion that didn't hold keeps the current note
            self._candidate, self._streak = float(round(midi)), 1
        self._pending.append((t, midi))
        if self._streak >= self.confirm:
            self.note, self._candidate, self._streak = self._candidate, None, 0
            out += self._release(self.note)
        elif len(self._pending) >= self.max_pending:
            out += self._release(self.note)
        return out

    def flush(self) -> list[tuple[float, float, float]]:
        return self._release(self.note)

    def _start_line(self, t: float, midi: float) -> list[tuple[float, float, float]]:
        note = float(round(midi))
        self._votes[note] = self._votes.get(note, 0) + 1
        self._pending.append((t, midi))
        if self._votes[note] >= self.confirm:
            self.note = note
            return self._release(note)
        if len(self._pending) >= self.max_pending:
            return self._release(None)  # never settled: silence rather than a guess
        return []

    def _release(self, note: float | None) -> list[tuple[float, float, float]]:
        out = [(t, m, np.nan if note is None else note) for t, m in self._pending]
        self._pending = []
        self._votes = {}
        return out


def segment_notes(track: PitchTrack, confirm: int = 8, min_note: float = 0.12) -> PitchTrack:
    """Reference melody as held semitone notes (for note bars and scoring).

    Notes shorter than `min_note` seconds (pitches passed through while sliding) are dropped.
    """
    snapper = NoteSnapper(confirm=confirm)
    notes = np.full(len(track.midi), np.nan, dtype=np.float32)
    decided = []
    for i, m in enumerate(track.midi):
        decided += snapper.push(i, float(m))
    for i, _, note in decided + snapper.flush():
        notes[int(i)] = note
    changes = np.flatnonzero(np.diff(notes) != 0) + 1  # NaN != NaN, so every unvoiced frame is a boundary too
    for start, end in zip(np.concatenate([[0], changes]), np.concatenate([changes, [len(notes)]])):
        if end - start < min_note / track.hop:
            notes[start:end] = np.nan
    return PitchTrack(notes, track.hop)


class MedianSmoother:
    """Running median over the last `size` live pitch frames.

    A frame counts as voiced only when at least `need` of the window are, so one- and
    two-frame glitches (octave jumps at note onsets) are dropped. Output lags input by `delay` frames.
    """

    def __init__(self, size: int = 5, need: int = 3):
        self._frames: deque[float] = deque([np.nan] * size, maxlen=size)
        self.need = need
        self.delay = size // 2

    def push(self, midi: float) -> float:
        self._frames.append(midi)
        voiced = sorted(m for m in self._frames if not np.isnan(m))
        # Middle element, never an average: averaging an octave pair would invent a tritone.
        return voiced[(len(voiced) - 1) // 2] if len(voiced) >= self.need else np.nan


class YinTracker:
    """Real-time YIN pitch detector for one mono stream.

    Each call to `detect` analyses the most recent `frame_length` samples.
    """

    def __init__(self, samplerate: int, fmin: float = FMIN, fmax: float = FMAX, window: float = 0.025,
                 threshold: float = 0.15, fallback_threshold: float = 0.2, min_level_db: float = -50.0):
        self.sr = samplerate
        self.tau_min = int(samplerate / fmax)
        self.tau_max = int(np.ceil(samplerate / fmin))
        self.window = int(samplerate * window)  # YIN integration window
        self.frame_length = self.window + self.tau_max
        self.threshold = threshold
        self.fallback_threshold = fallback_threshold
        self.min_level_db = min_level_db
        self._nfft = 1 << int(np.ceil(np.log2(self.frame_length + self.window)))

    def detect(self, frame: np.ndarray) -> tuple[float, float]:
        """Return (midi or NaN, level in dBFS) for a frame of `frame_length` samples."""
        x = np.asarray(frame[-self.frame_length:], dtype=np.float64)
        level_db = 10 * np.log10(np.mean(x**2) + 1e-12)
        if level_db < self.min_level_db:
            return np.nan, level_db

        w, tmax = self.window, self.tau_max
        # Difference function d(tau) = e(0) + e(tau) - 2 * r(tau), with r via FFT cross-correlation.
        spectrum = np.fft.rfft(x, self._nfft)
        r = np.fft.irfft(spectrum * np.conj(np.fft.rfft(x[:w], self._nfft)), self._nfft)[: tmax + 1]
        energy = np.concatenate([[0.0], np.cumsum(x**2)])
        e_tau = energy[w: w + tmax + 1] - energy[: tmax + 1]
        d = np.maximum(e_tau[0] + e_tau - 2 * r, 0.0)

        # Cumulative mean normalised difference.
        cmnd = np.ones_like(d)
        cumsum = np.cumsum(d[1:])
        cmnd[1:] = d[1:] * np.arange(1, tmax + 1) / np.maximum(cumsum, 1e-12)

        below = np.flatnonzero(cmnd[self.tau_min: tmax] < self.threshold)
        if below.size:
            tau = self.tau_min + below[0]
            while tau + 1 < tmax and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
        else:
            # No clear dip (breathy or rough voice): accept the best one if it is still reasonably periodic.
            tau = self.tau_min + int(np.argmin(cmnd[self.tau_min: tmax]))
            if cmnd[tau] > self.fallback_threshold:
                return np.nan, level_db
        tau = self._shortest_period(cmnd, tau)

        # Parabolic interpolation around the minimum for sub-sample accuracy.
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        return float(hz_to_midi(self.sr / (tau + shift))), level_db

    def _shortest_period(self, cmnd: np.ndarray, tau: int, margin: float = 0.1) -> int:
        """Undo subharmonic picks: the normalised difference dips at every multiple of the period and
        favours longer lags, so prefer tau/3 or tau/2 when they dip almost as deep (octave-down errors)."""
        for divisor in (3, 2):
            centre = round(tau / divisor)
            radius = max(2, centre // 20)
            lo, hi = centre - radius, centre + radius + 1
            if lo < max(self.tau_min, 1):
                continue
            j = lo + int(np.argmin(cmnd[lo:hi]))
            if lo < j < hi - 1 and cmnd[j] <= cmnd[tau] + margin:
                return j
        return tau
