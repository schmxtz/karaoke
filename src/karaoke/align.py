"""Word timings for lyric lines: CTC forced alignment of the text against the vocal stem.

Same method as WhisperX's alignment step (wav2vec2 character probabilities + a
Viterbi path through the line's characters), implemented here because WhisperX
pins an older torch. Each line is aligned on its own window, so one bad line
can't drag the rest. Words of lines that don't align confidently are spread
over the line by length instead.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# English wav2vec2 (large, self-trained on LibriLight) fine-tuned for character CTC. On the library
# songs it aligned every line (base-960h, WhisperX's default: 61/64) with higher confidence.
ALIGN_MODEL = "facebook/wav2vec2-large-960h-lv60-self"
MODEL_RATE = 16000
PAD_BEFORE, PAD_AFTER = 0.5, 0.5  # seconds of context around each LRC line
MIN_SCORE = 0.15  # mean character probability below which a line counts as not aligned


@dataclass
class Word:
    start: float
    end: float
    text: str
    score: float  # mean character probability, 0 when spread by length


def models_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or Path.home() / ".cache"
    return Path(base) / "karaoke" / "models" / "huggingface"


class WordAligner:
    def __init__(self, model_name: str = ALIGN_MODEL, device: str | None = None):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without developer mode
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # e.g. the unused training-only `masked_spec_embed` notice

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:  # use the downloaded copy without asking the Hub for updates
            self.processor, model = (cls.from_pretrained(model_name, cache_dir=models_dir(), local_files_only=True)
                                     for cls in (Wav2Vec2Processor, Wav2Vec2ForCTC))
        except OSError:  # first use: download (~1.2 GB)
            self.processor, model = (cls.from_pretrained(model_name, cache_dir=models_dir())
                                     for cls in (Wav2Vec2Processor, Wav2Vec2ForCTC))
        self.model = model.to(self.device).eval()
        vocab = self.processor.tokenizer.get_vocab()
        self.blank = self.processor.tokenizer.pad_token_id
        self.separator = vocab.get("|")
        self.letters = {ch: i for ch, i in vocab.items() if len(ch) == 1 and ch != "|"}
        self.uppercase = any(ch.isupper() for ch in self.letters)

    def align_lines(self, audio: np.ndarray, sr: int, lines: list[tuple[float, float, str]]) -> list[list[Word]]:
        """Word timings for each (start, end, text) line; `audio` is the mono vocal stem."""
        import soxr

        audio16 = soxr.resample(audio.astype(np.float32), sr, MODEL_RATE)
        return [self._align_line(audio16, start, end, text) for start, end, text in lines]

    def _align_line(self, audio16: np.ndarray, start: float, end: float, text: str) -> list[Word]:
        words = text.split()
        tokens, word_of_token = self._tokenize(words)
        a = max(int((start - PAD_BEFORE) * MODEL_RATE), 0)
        b = min(int((end + PAD_AFTER) * MODEL_RATE), len(audio16))
        if not tokens or b - a < MODEL_RATE // 4:
            return spread_words(words, start, end)

        log_probs = self._emissions(audio16[a:b])
        if log_probs.shape[0] < len(tokens):
            return spread_words(words, start, end)
        path = _viterbi(log_probs, tokens, self.blank)
        frame = (b - a) / MODEL_RATE / log_probs.shape[0]
        offset = a / MODEL_RATE

        timed: dict[int, list[tuple[int, float]]] = {}
        for token_index, t, prob in path:
            w = word_of_token[token_index]
            if w is not None:
                timed.setdefault(w, []).append((t, prob))
        aligned = []
        for i, word in enumerate(words):
            frames = timed.get(i)
            if not frames:
                aligned.append(None)  # nothing alignable (e.g. digits): filled in below
                continue
            ts = [t for t, _ in frames]
            aligned.append(Word(offset + min(ts) * frame, offset + (max(ts) + 1) * frame, word, float(np.mean([p for _, p in frames]))))

        scored = [w.score for w in aligned if w]
        if not scored or np.mean(scored) < MIN_SCORE:
            return spread_words(words, start, end)
        return _fill_gaps(aligned, words, start, end)

    def _tokenize(self, words: list[str]) -> tuple[list[int], list[int | None]]:
        tokens, owner = [], []
        for i, word in enumerate(words):
            chars = [self.letters[c] for c in (word.upper() if self.uppercase else word.lower()) if c in self.letters]
            if not chars:
                continue
            if tokens and self.separator is not None:
                tokens.append(self.separator)
                owner.append(None)
            tokens += chars
            owner += [i] * len(chars)
        return tokens, owner

    def _emissions(self, chunk: np.ndarray) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(chunk, sampling_rate=MODEL_RATE, return_tensors="pt").input_values.to(self.device)
        with torch.inference_mode():
            logits = self.model(inputs).logits[0]
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()


def _viterbi(log_probs: np.ndarray, tokens: list[int], blank: int) -> list[tuple[int, int, float]]:
    """Best monotonic path emitting `tokens` in order, blanks anywhere.

    Returns (token index, frame, probability) for every frame assigned to a token.
    """
    T, J = log_probs.shape[0], len(tokens)
    emit = log_probs[:, tokens]  # T x J
    stay = log_probs[:, blank]
    trellis = np.full((T + 1, J + 1), -np.inf)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = np.cumsum(stay)
    for t in range(T):
        trellis[t + 1, 1:] = np.maximum(trellis[t, 1:] + stay[t], trellis[t, :-1] + emit[t])
    # Backtrack from the last token at the last frame.
    path, j = [], J
    for t in range(T, 0, -1):
        if j == 0:
            break
        if trellis[t - 1, j - 1] + emit[t - 1, j - 1] >= trellis[t - 1, j] + stay[t - 1]:
            path.append((j - 1, t - 1, float(np.exp(emit[t - 1, j - 1]))))
            j -= 1
        # else: blank at this frame, the token is still j
    return path[::-1]


def spread_words(words: list[str], start: float, end: float) -> list[Word]:
    """Distribute words over [start, end] in proportion to their length."""
    if not words:
        return []
    weights = np.array([max(len(re.sub(r"\W", "", w)), 1) + 1 for w in words], dtype=float)
    edges = start + np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * (end - start)
    return [Word(float(a), float(b), w, 0.0) for a, b, w in zip(edges[:-1], edges[1:], words)]


def _fill_gaps(aligned: list[Word | None], words: list[str], start: float, end: float) -> list[Word]:
    """Give unaligned words (digits, symbols) the time between their aligned neighbours."""
    out = list(aligned)
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        lo = out[i - 1].end if i > 0 else start
        hi = out[j].start if j < len(out) else end
        out[i:j] = spread_words(words[i:j], lo, max(hi, lo + 0.1 * (j - i)))
        i = j
    return out
