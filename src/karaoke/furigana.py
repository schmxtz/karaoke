"""Furigana for Japanese lyrics: kana readings over the kanji, from MeCab with the UniDic dictionary.

Each line becomes (segment, reading) pairs that cover its text exactly; the reading is empty
for kana, Latin and spaces. Okurigana stay outside the reading: 優しさ -> 優(やさ) しさ.
"""

from __future__ import annotations

import re

_KANJI = re.compile(r"[㐀-䶿一-鿿豈-﫿々〆ヶ]")
_KANA = re.compile(r"[ぁ-ゟァ-ヺ]")
_KANJI_RUNS = re.compile(r"([㐀-䶿一-鿿豈-﫿々〆ヶ]+)")
# 何 is なに before these particles (何を見て), なん elsewhere (何で, 何の); MeCab often picks なん.
_NANI_BEFORE = ("を", "が", "も", "に", "は", "か")


def has_kanji(text: str) -> bool:
    return bool(_KANJI.search(text))


def is_japanese(text: str) -> bool:
    """Kanji alongside kana; kanji without any kana is more likely Chinese."""
    return has_kanji(text) and bool(_KANA.search(text))


def hiragana(kana: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in kana)


class Furigana:
    def __init__(self):
        from fugashi import Tagger  # MeCab + unidic-lite, ~50 MB, loads in well under a second

        self._tagger = Tagger()

    def annotate(self, text: str) -> list[tuple[str, str]]:
        tokens = self._tagger(text)
        segments: list[tuple[str, str]] = []
        pos = 0
        for i, token in enumerate(tokens):
            surface = token.surface
            start = text.find(surface, pos)
            if start < 0:  # tokenizer normalised something: give up on the rest of the line
                break
            segments.append((text[pos:start], ""))  # whitespace between tokens
            pos = start + len(surface)
            reading = token.feature.kana if has_kanji(surface) else None
            if not reading or reading == "*":  # kana/Latin, or a word the dictionary doesn't know
                segments.append((surface, ""))
                continue
            reading = hiragana(reading)
            following = tokens[i + 1].surface if i + 1 < len(tokens) else ""
            if surface == "何" and following.startswith(_NANI_BEFORE):
                reading = "なに"
            segments += _split_okurigana(surface, reading)
        segments.append((text[pos:], ""))
        return _merge_plain(segments)


def _split_okurigana(surface: str, reading: str) -> list[tuple[str, str]]:
    """Put the reading over the kanji runs only, matching the kana around them literally."""
    parts = [p for p in _KANJI_RUNS.split(surface) if p]
    pattern = "".join("(.+)" if has_kanji(p) else re.escape(hiragana(p)) for p in parts)
    match = re.fullmatch(pattern, reading)
    if not match:  # e.g. irregular readings that don't line up: whole word, whole reading
        return [(surface, reading)]
    groups = iter(match.groups())
    return [(p, next(groups)) if has_kanji(p) else (p, "") for p in parts]


def _merge_plain(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for text, reading in segments:
        if not text:
            continue
        if not reading and merged and not merged[-1][1]:
            merged[-1] = (merged[-1][0] + text, "")
        else:
            merged.append((text, reading))
    return merged
