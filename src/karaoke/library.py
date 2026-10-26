"""Song library: one folder per song with its stems and reference pitch."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from karaoke.lyrics import LYRICS_FILE, Lyrics, load_or_fetch
from karaoke.pitch import PitchTrack, extract_reference_pitch, segment_notes
from karaoke.separation import MANIFEST

PITCH_FILE = "melody.npz"
OBSOLETE_PITCH_FILE = "pitch.npz"  # former melody, partly taken from the lead-only stem


def default_library_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or Path.home() / ".cache"
    return Path(base) / "karaoke" / "library"


@dataclass
class Song:
    title: str
    folder: Path
    stems: dict[str, Path]
    gain: float
    pitch: PitchTrack  # continuous melody
    notes: PitchTrack  # the melody as held semitone notes
    lyrics: Lyrics | None

    @classmethod
    def load(cls, folder: Path) -> Song:
        manifest = json.loads((folder / MANIFEST).read_text())
        stems = {name: folder / f"{name}.flac" for name in manifest["stems"]}
        pitch = PitchTrack.load(folder / PITCH_FILE)
        lyrics = Lyrics.load(folder) if (folder / LYRICS_FILE).exists() else None
        return cls(Path(manifest["source"]).stem, folder, stems, manifest["gain"], pitch, segment_notes(pitch), lyrics)


def prepare_song(audio_path: Path, library_dir: Path | None = None, progress: Callable[[str], None] = print) -> Song:
    """Separate stems and extract the reference melody, reusing anything already computed."""
    audio_path = Path(audio_path)
    folder = (library_dir or default_library_dir()) / audio_path.stem

    from karaoke.separation import StemSeparator

    progress("Separating vocals (first time only, about a minute)...")
    stems = StemSeparator().separate(audio_path, folder)

    pitch_path = folder / PITCH_FILE
    if not pitch_path.exists() or pitch_path.stat().st_mtime < (folder / MANIFEST).stat().st_mtime:
        progress("Tracking the melody...")
        extract_reference_pitch(stems.vocals).save(pitch_path)
        (folder / OBSOLETE_PITCH_FILE).unlink(missing_ok=True)

    progress("Looking up lyrics...")
    melody = PitchTrack.load(pitch_path)
    try:
        load_or_fetch(folder, audio_path, melody.duration, melody)
    except (OSError, ValueError) as exc:  # offline or LRCLIB unavailable: the song works without, retried next open
        progress(f"Lyrics unavailable ({type(exc).__name__})")

    lyrics = Lyrics.load(folder) if (folder / LYRICS_FILE).exists() else None
    if lyrics and lyrics.needs_word_timing():
        progress("Timing the words...")
        try:
            time_words(lyrics, stems.vocals)
            lyrics.save(folder)
        except Exception:  # optional: lines still show without word highlighting
            traceback.print_exc()
            progress("Word timing failed, showing whole lines")
    if lyrics and lyrics.needs_furigana():
        try:
            add_furigana(lyrics)
            lyrics.save(folder)
        except Exception:  # optional: lines still show, without readings
            traceback.print_exc()

    return Song.load(folder)


def add_furigana(lyrics: Lyrics) -> None:
    from karaoke.furigana import Furigana

    furigana = Furigana()
    for line in lyrics.lines:
        line.ruby = furigana.annotate(" ".join(line.text.split()))


def time_words(lyrics: Lyrics, vocals: Path) -> None:
    """Align every line's words against the vocal stem, in windows placed by the current offset."""
    import soundfile as sf
    import torch

    from karaoke.align import WordAligner

    audio, sr = sf.read(vocals, dtype="float32", always_2d=True)
    shift = lyrics.user_offset
    aligner = WordAligner()
    timed = aligner.align_lines(audio.mean(axis=1), sr, [(l.start + shift, l.end + shift, l.text) for l in lyrics.lines])
    for line, words in zip(lyrics.lines, timed):
        line.words = words
    lyrics.words_offset = shift
    del aligner
    torch.cuda.empty_cache()
