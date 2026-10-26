"""Time-synced lyrics from LRCLIB (lrclib.net), aligned to the song's own vocals.

LRCLIB gives line-level timings for the released track. A local file can be a
different cut (e.g. a YouTube rip with a longer intro), so the timings are
shifted by a global offset found by matching line starts to where singing
starts in the separated vocals.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from karaoke.align import Word
from karaoke.furigana import is_japanese
from karaoke.pitch import PitchTrack

API = "https://lrclib.net/api/"
USER_AGENT = "karaoke-desktop/0.1 (https://github.com/; personal karaoke app)"
LYRICS_FILE = "lyrics.json"
RETRY_NOT_FOUND = 7 * 24 * 3600  # look again a week after a miss
MAX_DURATION_GAP = 30.0  # seconds; a bigger mismatch is a different recording
MAX_OFFSET = 30.0  # seconds searched when aligning lines to the vocals
MIN_STRUCTURE_MATCH = 0.35  # real matches score ~0.45-0.6, another song ~0.2

# Noise in YouTube-style titles: "(Lyrics)", "[Official Video]", "(Official Visualizer)", ...
_NOISE = re.compile(
    r"\s*[(\[][^)\]]*\b(lyrics?|official|video|audio|visuali[sz]er|hd|hq|4k|remaster(ed)?|explicit|clean|mv)\b[^)\]]*[)\]]",
    re.IGNORECASE,
)
_FEAT = re.compile(r"\s*[(\[]?\b(feat\.?|ft\.?|featuring)\s.*$", re.IGNORECASE)
_LRC_TIME = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


@dataclass
class LyricLine:
    start: float  # seconds in the local file, auto offset applied
    end: float
    text: str
    words: list[Word] = field(default_factory=list)  # aligned to the vocals, see Lyrics.words_offset
    ruby: list[tuple[str, str]] = field(default_factory=list)  # (text, furigana) covering the text; Japanese only


@dataclass
class Lyrics:
    lines: list[LyricLine]
    artist: str
    title: str
    source: str = "lrclib"
    source_id: int | None = None
    offset: float = 0.0  # auto-detected shift applied to the LRC timings
    user_offset: float = 0.0  # manual correction on top, set in the UI
    words_offset: float | None = None  # user_offset the word timings were aligned with (None: not yet)
    extra: dict = field(default_factory=dict)

    @property
    def word_shift(self) -> float:
        """Shift to add to word times: follows user_offset changes made since the words were aligned."""
        return self.user_offset - (self.words_offset or 0.0)

    def needs_word_timing(self) -> bool:
        return self.words_offset is None or abs(self.user_offset - self.words_offset) > 0.25

    def needs_furigana(self) -> bool:
        return not any(line.ruby for line in self.lines) and is_japanese(" ".join(line.text for line in self.lines))

    def line_at(self, t: float) -> int | None:
        """Index of the line being sung at time `t` (None before the first or in a gap)."""
        t -= self.user_offset
        for i, line in enumerate(self.lines):
            if line.start <= t < line.end:
                return i
        return None

    def next_line(self, t: float) -> int | None:
        t -= self.user_offset
        return next((i for i, line in enumerate(self.lines) if line.start > t), None)

    def save(self, folder: Path) -> None:
        (folder / LYRICS_FILE).write_text(json.dumps({"found": True, **asdict(self)}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> Lyrics | None:
        data = json.loads((folder / LYRICS_FILE).read_text(encoding="utf-8"))
        if not data.pop("found"):
            return None
        data["lines"] = [
            LyricLine(**{**line, "words": [Word(**w) for w in line.get("words", [])],
                         "ruby": [tuple(segment) for segment in line.get("ruby", [])]})
            for line in data["lines"]
        ]
        return cls(**data)


def guess_artist_title(audio_path: Path) -> tuple[str | None, str]:
    """Artist and title from the file's tags, else from an "Artist - Title (noise)" file name."""
    tags = _read_tags(audio_path)
    artist, title = tags.get("artist"), tags.get("title")
    if not title:
        name = audio_path.stem
        if " - " in name:
            artist, title = name.split(" - ", 1)
        else:
            title = name
    title = _FEAT.sub("", _NOISE.sub("", title)).strip()
    return (artist.strip() if artist else None), title


def _read_tags(audio_path: Path) -> dict[str, str]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(audio_path)],
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=30,
        ).stdout
        return {k.lower(): v for k, v in json.loads(out).get("format", {}).get("tags", {}).items()}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def fetch_lyrics(audio_path: Path, duration: float, melody: PitchTrack) -> Lyrics | None:
    """Look the song up on LRCLIB and align the line timings to the vocals; None if not found."""
    artist, title = guess_artist_title(audio_path)
    record = _lookup(artist, title, duration)
    if not record:
        return None
    lines = parse_lrc(record["syncedLyrics"])
    if not lines:
        return None
    offset, match = estimate_offset(lines, melody)
    if match < MIN_STRUCTURE_MATCH:
        return None  # the timings don't fit this recording's singing: wrong song or a different cut
    shifted = [LyricLine(l.start + offset, l.end + offset, l.text) for l in lines]
    shifted = [LyricLine(round(l.start, 3), round(_singing_end(l, melody), 3), l.text) for l in shifted]
    return Lyrics(
        shifted, record.get("artistName") or artist or "", record.get("trackName") or title,
        source_id=record.get("id"), offset=round(offset, 3),
        extra={"lrclib_duration": record.get("duration"), "offset_match": round(match, 2)},
    )


def _lookup(artist: str | None, title: str, duration: float) -> dict | None:
    if artist:
        exact = _get("get", artist_name=artist, track_name=title, duration=round(duration))
        if exact and exact.get("syncedLyrics"):
            return exact
    query = {"track_name": title, **({"artist_name": artist} if artist else {})}
    results = [
        r for r in (_get("search", **query) or [])
        if r.get("syncedLyrics") and r.get("duration")
        and _same(r.get("trackName"), title) and (not artist or _same(r.get("artistName"), artist))
    ]
    if not results:
        return None
    best = min(results, key=lambda r: abs(r["duration"] - duration))
    return best if abs(best["duration"] - duration) <= MAX_DURATION_GAP else None


def _same(found: str | None, wanted: str) -> bool:
    """Loose name match: case/punctuation-insensitive, one containing the other ("Free.99" ~ "Free 99")."""
    norm = lambda s: re.sub(r"[^0-9a-z]+", "", (s or "").lower())
    a, b = norm(found), norm(wanted)
    return bool(a and b) and (a in b or b in a)


def _get(endpoint: str, **params) -> dict | list | None:
    url = API + endpoint + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code != 503 or attempt == 2:
                raise
            time.sleep(1 + attempt)  # LRCLIB asks clients to back off under load
    return None


def parse_lrc(text: str) -> list[LyricLine]:
    """LRC "[mm:ss.xx] words" lines -> LyricLine list. Empty timed lines end the previous line."""
    stamped = []
    for raw in text.splitlines():
        times = _LRC_TIME.findall(raw)
        words = _LRC_TIME.sub("", raw).strip()
        stamped += [(int(m) * 60 + float(s), words) for m, s in times]
    stamped.sort(key=lambda x: x[0])
    lines = []
    for (start, words), (next_start, _) in zip(stamped, stamped[1:] + [(stamped[-1][0] + 6.0, "")]):
        if words:
            lines.append(LyricLine(start, next_start, words))
    return lines


def estimate_offset(lines: list[LyricLine], melody: PitchTrack, min_match: float = MIN_STRUCTURE_MATCH) -> tuple[float, float]:
    """Shift (seconds) that lines the LRC timings up with the singing in the vocals.

    1. Structure: correlate "a line is being sung" (each line start + up to 5 s) with voiced
       frames. This finds the intro, breaks and outro, but is flat over a few seconds because
       lines cover continuous singing.
    2. Within that plateau, pick the shift where line starts best fall on silence -> singing
       edges (a breath before the first word).
    Returns (offset, match) with match = the structure correlation; below `min_match` the lyrics
    likely belong to another version, so no shift is applied.
    """
    hop = melody.hop
    voiced = (~np.isnan(melody.midi)).astype(float)
    n = len(voiced)
    half_second = int(0.5 / hop)
    activity = np.convolve(voiced, np.ones(half_second) / half_second, mode="same")
    starts = np.array([line.start for line in lines])
    ends = np.array([min(line.end, line.start + 5.0) for line in lines])

    coarse = np.arange(-MAX_OFFSET, MAX_OFFSET + 1e-9, 0.05)
    match = np.full(len(coarse), -1.0)
    for i, shift in enumerate(coarse):
        predicted = np.zeros(n)
        for a, b in zip(np.round((starts + shift) / hop).astype(int), np.round((ends + shift) / hop).astype(int)):
            predicted[max(a, 0):max(min(b, n), 0)] = 1.0
        if 0 < predicted.mean() < 1:
            match[i] = np.corrcoef(predicted, activity)[0, 1]
    best_match = float(match.max())
    if best_match < min_match:
        return 0.0, best_match

    plateau = coarse[match >= best_match - 0.03]
    fine = np.arange(plateau.min() - 0.5, plateau.max() + 0.5, hop)
    csum = np.concatenate([[0.0], np.cumsum(voiced)])
    before, after = int(0.4 / hop), int(0.5 / hop)
    edges = []
    for shift in fine:
        t = np.clip(np.round((starts + shift) / hop).astype(int), 0, n)
        pre = (csum[t] - csum[np.clip(t - before, 0, n)]) / before
        post = (csum[np.clip(t + after, 0, n)] - csum[t]) / after
        edges.append(np.mean(post - pre))
    return float(fine[int(np.argmax(edges))]), best_match


def _singing_end(line: LyricLine, melody: PitchTrack, release: float = 0.4, min_length: float = 1.0) -> float:
    """End a line shortly after its last sung frame, so it doesn't stay lit through an instrumental break."""
    a, b = int(line.start / melody.hop), int(line.end / melody.hop)
    voiced = np.flatnonzero(~np.isnan(melody.midi[max(a, 0):max(b, 0)]))
    if voiced.size == 0:
        return line.end
    sung_until = (max(a, 0) + voiced[-1]) * melody.hop + release
    return min(line.end, max(sung_until, line.start + min_length))


def load_or_fetch(folder: Path, audio_path: Path, duration: float, melody: PitchTrack) -> Lyrics | None:
    """Cached lyrics for a song folder; fetches when missing (or a week after a miss)."""
    cache = folder / LYRICS_FILE
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("found") or time.time() - data.get("checked", 0) < RETRY_NOT_FOUND:
            return Lyrics.load(folder)
    lyrics = fetch_lyrics(audio_path, duration, melody)
    if lyrics:
        lyrics.save(folder)
    else:
        cache.write_text(json.dumps({"found": False, "checked": time.time()}), encoding="utf-8")
    return lyrics
