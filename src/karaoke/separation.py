"""Split a song into vocals and instrumental with a RoFormer source-separation model.

All vocals (lead and backing) stay in one stem: lead/backing models misjudge
double-tracked lead vocals as backing, which silently loses the melody.

Levels: audio-separator normally peak-normalizes every stem on its own,
which skews the vocal/instrumental balance. Instead we separate in 32-bit
float with normalization off, then apply one shared gain to all stems so
they fit in 24-bit FLAC. The gain is recorded in stems.json; dividing by it
restores the original loudness, and the stems sum back to the source mix.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Mel-Band RoFormer by Kimberley Jensen: vocals SDR 12.60 on audio-separator's benchmark.
VOCAL_MODEL = "vocals_mel_band_roformer.ckpt"

SAMPLE_RATE = 44100
PEAK_CEILING = 0.99
MANIFEST = "stems.json"
STEMS = ("vocals", "instrumental")
OBSOLETE_STEMS = ("lead_vocals", "backing_vocals")  # from the former lead/backing split


def default_models_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or Path.home() / ".cache"
    return Path(base) / "karaoke" / "models"


@dataclass(frozen=True)
class Stems:
    instrumental: Path
    vocals: Path
    gain: float = 1.0  # stems = original level * gain


class StemSeparator:
    """Keeps the model loaded; reuse an instance across songs."""

    def __init__(self, models_dir: Path | None = None, vocal_model: str = VOCAL_MODEL, log_level: int = logging.ERROR):
        # Imported lazily: pulls in torch, which takes a few seconds.
        from audio_separator.separator import Separator

        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self.vocal_model = vocal_model
        self._separator = Separator(
            model_file_dir=str(self.models_dir),
            output_format="WAV",
            use_soundfile=True,  # writes the input's subtype, i.e. float for our float input
            sample_rate=SAMPLE_RATE,
            log_level=log_level,
        )
        # The constructor rejects thresholds above 1; models copy this value when loaded.
        self._separator.normalization_threshold = float("inf")
        self._loaded_model: str | None = None

    def separate(self, audio_path: Path, out_dir: Path, overwrite: bool = False) -> Stems:
        """Write stems for `audio_path` into `out_dir`; a previous result is reused unless `overwrite`."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite and (cached := self._cached(out_dir, audio_path)):
            return cached

        with tempfile.TemporaryDirectory(dir=out_dir, prefix=".work-") as tmp:
            work = Path(tmp)
            mix = work / "mix.wav"
            _decode_to_float_wav(audio_path, mix)
            stems = self._run(self.vocal_model, mix, work, {"vocals": "vocals", "other": "instrumental"})
            gain = _write_flac_stems(stems, out_dir)

        _save_manifest(out_dir, audio_path, self.vocal_model, gain)
        return _stems_from(out_dir, gain)

    def _cached(self, out_dir: Path, audio_path: Path) -> Stems | None:
        try:
            manifest = json.loads((out_dir / MANIFEST).read_text())
        except (OSError, ValueError):
            return None
        if manifest.get("models", {}).get("vocal") != self.vocal_model or not all((out_dir / f"{n}.flac").exists() for n in STEMS):
            return None
        if set(manifest["stems"]) != set(STEMS):
            # Made with the former lead/backing split: vocals and instrumental are still valid.
            for name in OBSOLETE_STEMS:
                (out_dir / f"{name}.flac").unlink(missing_ok=True)
            _save_manifest(out_dir, audio_path, self.vocal_model, manifest["gain"])
        return _stems_from(out_dir, manifest["gain"])

    def _run(self, model: str, audio_path: Path, work: Path, names: dict[str, str]) -> dict[str, Path]:
        if self._loaded_model != model:
            self._separator.load_model(model)  # downloads on first use
            self._loaded_model = model
        # The model instance copies output_dir at load time, so point both at this song.
        self._separator.output_dir = self._separator.model_instance.output_dir = str(work)
        self._separator.separate(str(audio_path), custom_output_names=names)
        paths = {n: work / f"{n}.wav" for n in names.values()}
        missing = [n for n, p in paths.items() if not p.exists()]
        if missing:
            raise RuntimeError(f"{model} did not produce stems {missing}")
        return paths


def _decode_to_float_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(dst)],
        check=True,
    )


def _write_flac_stems(stems: dict[str, Path], out_dir: Path) -> float:
    """Scale all stems by one gain so the loudest peak fits under PEAK_CEILING; returns that gain."""
    audio = {name: sf.read(path, dtype="float32")[0] for name, path in stems.items()}
    peak = max(float(np.abs(a).max()) for a in audio.values())
    gain = min(1.0, PEAK_CEILING / peak) if peak > 0 else 1.0
    for name, a in audio.items():
        sf.write(out_dir / f"{name}.flac", a * gain, SAMPLE_RATE, subtype="PCM_24")
    return gain


def _stems_from(out_dir: Path, gain: float) -> Stems:
    return Stems(**{n: out_dir / f"{n}.flac" for n in STEMS}, gain=gain)


def _save_manifest(out_dir: Path, source: Path, vocal_model: str, gain: float) -> None:
    manifest = {"source": source.name, "models": {"vocal": vocal_model}, "stems": list(STEMS), "gain": gain, "sample_rate": SAMPLE_RATE}
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
