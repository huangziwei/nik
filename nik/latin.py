"""Read Latin-script words the Japanese way by rewriting them in katakana.

Irodori is a Japanese-only model: left as Latin letters, English words come out
garbled (schedule → すーじゅー). The TTS pipeline rewrites every Latin word after
ruby, reading overrides and number normalization have run, so a user override
such as ``Boogiepop=ブギーポップ`` always wins over anything chosen here.

A word or multi-word phrase resolves in this order:

1. ``_CURATED_READINGS`` — abbreviations, and common words the dictionary gets
   wrong (UniDic's only ``email`` is the French émail, エマイユ).
2. UniDic loanword lemmas (``スケジュール-schedule``, ``ニューヨーク-New York``),
   longest phrase first, exact spelling before a case-folded match.
3. Acronyms and other short unknown tokens are spelled out with Japanese letter
   names (SNS → エスエヌエス, ADHD → エーディーエイチディー).
4. e2k's C2K character model transliterates anything else (YouTube → ユーチューブ).

URLs and e-mail addresses are dropped: nobody wants them spelled out.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

LETTER_NAMES = {
    "a": "エー",
    "b": "ビー",
    "c": "シー",
    "d": "ディー",
    "e": "イー",
    "f": "エフ",
    "g": "ジー",
    "h": "エイチ",
    "i": "アイ",
    "j": "ジェー",
    "k": "ケー",
    "l": "エル",
    "m": "エム",
    "n": "エヌ",
    "o": "オー",
    "p": "ピー",
    "q": "キュー",
    "r": "アール",
    "s": "エス",
    "t": "ティー",
    "u": "ユー",
    "v": "ブイ",
    "w": "ダブリュー",
    "x": "エックス",
    "y": "ワイ",
    "z": "ゼット",
}

# Abbreviations whose trailing period belongs to the word (Mr. Smith, ver.2).
_ABBREVIATIONS = {
    "dr": "ドクター",
    "etc": "エトセトラ",
    "mr": "ミスター",
    "mrs": "ミセス",
    "ms": "ミズ",
    "ver": "バージョン",
    "vol": "ボリューム",
    "vs": "バーサス",
}
_CURATED_READINGS = {
    **_ABBREVIATIONS,
    "am": "エーエム",
    "pm": "ピーエム",
    "e-mail": "イーメール",
    "email": "イーメール",
    "ok": "オーケー",
    "okay": "オーケー",
    "wi-fi": "ワイファイ",
    "wifi": "ワイファイ",
}
# Two capitals are spelled out (TV, PC, IT, ID) except these words (スイッチをON).
_CAPS_WORDS = {
    "an": "アン",
    "as": "アズ",
    "at": "アット",
    "be": "ビー",
    "by": "バイ",
    "do": "ドゥー",
    "go": "ゴー",
    "he": "ヒー",
    "if": "イフ",
    "in": "イン",
    "is": "イズ",
    "me": "ミー",
    "my": "マイ",
    "no": "ノー",
    "of": "オブ",
    "oh": "オー",
    "on": "オン",
    "or": "オア",
    "so": "ソー",
    "to": "トゥー",
    "up": "アップ",
    "we": "ウィー",
}

# Fullwidth ASCII (U+FF01–U+FF5E) and the ideographic space fold one-to-one onto
# ASCII, so spans found in the folded text index the original text directly.
_WIDTH_FOLD = {cp: cp - 0xFEE0 for cp in range(0xFF01, 0xFF5F)} | {0x3000: 0x20}

_WORD = r"[A-Za-z]+(?:['’][A-Za-z]+)*"
_PHRASE_RE = re.compile(rf"{_WORD}(?:(?: |-|‐){_WORD})*")
_WORD_RE = re.compile(_WORD)
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s「」『』（）()<>、。]+"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
)
_ASCII_ROMAN_RE = re.compile(r"[IVXLCDM]{2,}")
_KATAKANA_RE = re.compile(r"[ァ-ヺー・]+")
_LEXICON_SOURCE_RE = re.compile(r"[A-Za-z][A-Za-z '.&-]*[A-Za-z.]")
# A space next to these hiragana continues the sentence (Amazon で, YouTube や).
_PARTICLE_STARTS = set("でやにをはがのともへなっかよねだ")
_MAX_PHRASE_WORDS = 4
_SHORT_TOKEN_MAX = 3
_CLIPPING_MAX_RATIO = 0.6
_LEXICON_VERSION = 2


@dataclass(frozen=True)
class _Lexicon:
    """Source spelling → katakana candidates as ``[kana, rank]``, best first."""

    exact: dict[str, list[list]]
    folded: dict[str, list[list]]


def _fold_width(text: str) -> str:
    return text.translate(_WIDTH_FOLD)


def _is_hiragana(ch: str) -> bool:
    return "\u3041" <= ch <= "\u309f"


def _is_kanji(ch: str) -> bool:
    from . import tts

    return tts._is_japanese_char(ch) and not tts._is_kana_char(ch)


def _rank_candidates(candidates: dict[str, int]) -> list[list]:
    # A clipped form next to its full form (リストラ / リストラクチュアリング,
    # ビル / ビルディング) is the colloquial Japanese word, not how the Latin
    # spelling reads, so it goes; パリ / パリス stay as real alternatives.
    kept = [
        kana
        for kana in candidates
        if not any(
            other != kana
            and other.startswith(kana)
            and len(kana) <= _CLIPPING_MAX_RATIO * len(other)
            for other in candidates
        )
    ]
    kept.sort(key=lambda kana: (candidates[kana], kana))
    return [[kana, candidates[kana]] for kana in kept]


def _build_lexicon(lex_csv: Path) -> _Lexicon:
    """Map the source spelling of UniDic loanword lemmas to their katakana."""
    exact: dict[str, dict[str, int]] = {}
    folded: dict[str, dict[str, int]] = {}
    with lex_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 12:
                continue
            kana, _, source = row[11].partition("-")
            if not source or not _KATAKANA_RE.fullmatch(kana):
                continue
            if not _LEXICON_SOURCE_RE.fullmatch(source):
                continue
            try:
                cost = int(row[3])
            except ValueError:
                continue
            bucket = exact.setdefault(source, {})
            bucket[kana] = min(cost, bucket.get(kana, cost))
            # Case-folded lookups rank common words above proper nouns, so
            # "nice" reads ナイス even though the city Nice is ニース.
            rank = cost if source.islower() else 1_000_000 + cost
            bucket = folded.setdefault(source.lower(), {})
            bucket[kana] = min(rank, bucket.get(kana, rank))
    return _Lexicon(
        exact={source: _rank_candidates(c) for source, c in exact.items()},
        folded={key: _rank_candidates(c) for key, c in folded.items()},
    )


@lru_cache(maxsize=1)
def _lexicon() -> _Lexicon:
    from . import tts

    dict_dir = tts._ensure_unidic_dir(tts._resolve_unidic_dir())
    cache_path = (
        dict_dir.parent / f"latin-katakana-{dict_dir.name}-v{_LEXICON_VERSION}.json"
    )
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return _Lexicon(exact=data["exact"], folded=data["folded"])
    lex_csv = dict_dir / "lex.csv"
    if not lex_csv.exists():
        return _Lexicon(exact={}, folded={})
    lexicon = _build_lexicon(lex_csv)
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(
            {"exact": lexicon.exact, "folded": lexicon.folded}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    tmp_path.replace(cache_path)
    return lexicon


@lru_cache(maxsize=1)
def _c2k_model() -> Any:
    from e2k import C2K

    # The default 16-symbol cap truncates long words (インターナショナリゼーション).
    return C2K(max_len=32)


@lru_cache(maxsize=1)
def _ngram_model() -> Any:
    from e2k import NGram

    return NGram()


def _c2k(word: str) -> str:
    return _c2k_model()(word.lower())


def _pronounceable(word: str) -> bool:
    return bool(_ngram_model()(word.lower()))


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def spell_out(word: str) -> str:
    return "".join(LETTER_NAMES.get(ch, "") for ch in word.lower())


def _pick(candidates: list[list], word: str) -> str:
    best_rank = candidates[0][1]
    tied = [kana for kana, rank in candidates if rank == best_rank]
    if len(tied) == 1:
        return tied[0]
    # Equal dictionary cost (Netflix: ネトフリ / ネットフリックス, Edgar: エドガー /
    # エドガール): take the reading closest to how the English spelling sounds.
    guess = _c2k(re.sub(r"[^A-Za-z]", "", word))
    return min(tied, key=lambda kana: (_edit_distance(kana, guess), kana))


def _lookup(text: str) -> str | None:
    curated = _CURATED_READINGS.get(text.lower())
    if curated:
        return curated
    lexicon = _lexicon()
    candidates = lexicon.exact.get(text) or lexicon.folded.get(text.lower())
    return _pick(candidates, text) if candidates else None


def _pluralize(reading: str) -> str:
    if reading.endswith("ト"):
        return reading[:-1] + "ツ"  # cats → キャッツ
    if reading.endswith("ド"):
        return reading[:-1] + "ズ"  # cards → カーズ
    if reading.endswith(("ク", "プ", "フ")):
        return reading + "ス"  # books → ブックス
    if reading.endswith(("ス", "ズ")):
        return reading
    return reading + "ズ"  # others → アザーズ


def _plural_reading(word: str) -> str | None:
    lowered = word.lower()
    if len(lowered) < 4 or not lowered.endswith("s") or lowered.endswith("ss"):
        return None
    if lowered.endswith("ies"):
        singular = _lookup(word[:-3] + "y")
        if singular:
            return _pluralize(singular)
    singular = _lookup(word[:-1])
    return _pluralize(singular) if singular else None


@lru_cache(maxsize=4096)
def word_reading(word: str) -> str:
    """Katakana for one Latin word (no spaces or hyphens)."""
    letters = re.sub(r"[^A-Za-z]", "", word)
    if not letters:
        return ""
    lowered = letters.lower()
    if len(letters) == 1:
        return LETTER_NAMES[lowered]
    curated = _CURATED_READINGS.get(word.lower())
    if curated:
        return curated
    caps = letters.isupper()
    if caps and len(letters) == 2:
        return _CAPS_WORDS.get(lowered) or spell_out(letters)
    reading = _lookup(word)
    if reading:
        # A capitalised short token is usually an acronym that happens to be a
        # word too (TV); keep the word only when it reads like one (WIN, THE).
        if caps and len(letters) <= _SHORT_TOKEN_MAX and not _pronounceable(letters):
            return spell_out(letters)
        return reading
    if len(letters) <= _SHORT_TOKEN_MAX:
        return spell_out(letters)  # sns, bgm, pdf
    plural = _plural_reading(word)
    if plural:
        return plural
    if not _pronounceable(letters):
        return spell_out(letters)  # ADHD, PTSD
    return _c2k(letters)


def _phrase_reading(words: list[str], separators: list[str]) -> str:
    readings: list[str] = []
    idx = 0
    while idx < len(words):
        for stop in range(min(len(words), idx + _MAX_PHRASE_WORDS), idx + 1, -1):
            phrase = words[idx] + "".join(
                sep + word
                for sep, word in zip(separators[idx : stop - 1], words[idx + 1 : stop])
            )
            reading = _lookup(phrase)
            if reading:
                readings.append(reading)
                idx = stop
                break
        else:
            word = words[idx]
            if word == "a" and len(words) > 1:
                readings.append("ア")  # the article, not the letter
            else:
                readings.append(word_reading(word))
            idx += 1
    return "・".join(reading for reading in readings if reading)


def _is_skipped_token(word: str) -> bool:
    """Tokens number normalization reads already (ASCII roman numerals II–L)."""
    if not _ASCII_ROMAN_RE.fullmatch(word):
        return False
    from . import tts

    value = tts._roman_to_int(word)
    return value is not None and value <= tts._ASCII_ROMAN_MAX_VALUE


def latin_reading_spans(text: str) -> list[dict]:
    """Latin runs of ``text`` with their katakana reading.

    Offsets index ``text`` itself (fullwidth Latin included). URLs and e-mail
    addresses come back with an empty reading: they are dropped from speech.
    """
    if not text:
        return []
    folded = _fold_width(text)
    if not re.search(r"[A-Za-z]", folded):
        return []
    spans: list[dict] = []
    taken: list[tuple[int, int]] = []
    for match in _URL_RE.finditer(folded):
        start, end = match.span()
        taken.append((start, end))
        spans.append(
            {"start": start, "end": end, "base": text[start:end], "reading": ""}
        )
    for match in _PHRASE_RE.finditer(folded):
        if any(
            match.start() < t_end and t_start < match.end() for t_start, t_end in taken
        ):
            continue
        for group in _phrase_groups(match):
            start, end = group[0][0], group[-1][1]
            words = [word for _, _, word in group]
            separators = [folded[prev[1] : nxt[0]] for prev, nxt in pairwise(group)]
            if (
                words[-1].lower() in _ABBREVIATIONS
                and end < len(folded)
                and folded[end] == "."
            ):
                end += 1
            if start > 0 and folded[start - 1] == "@":
                start -= 1  # @handle: the at sign is not spoken
            reading = _phrase_reading(words, separators)
            if reading:
                spans.append(
                    {
                        "start": start,
                        "end": end,
                        "base": text[start:end],
                        "reading": reading,
                    }
                )
    spans.sort(key=lambda span: (span["start"], span["end"]))
    return spans


def _phrase_groups(match: re.Match) -> list[list[tuple[int, int, str]]]:
    """Split a phrase match into word runs around tokens numbers already read."""
    groups: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    for word_match in _WORD_RE.finditer(match.group(0)):
        word = word_match.group(0)
        if _is_skipped_token(word):
            if current:
                groups.append(current)
            current = []
            continue
        current.append(
            (match.start() + word_match.start(), match.start() + word_match.end(), word)
        )
    if current:
        groups.append(current)
    return groups


def latin_to_katakana(text: str) -> str:
    """Rewrite every Latin word of ``text`` in katakana and drop URLs."""
    spans = latin_reading_spans(text)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    previous_end = -1
    for span in spans:
        start, end, reading = span["start"], span["end"], span["reading"]
        if start < cursor:
            continue
        head = text[cursor:start]
        if reading and cursor == previous_end and head and not head.strip():
            head = "・"  # Mr. Smith → ミスター・スミス
        elif head.endswith(" "):
            # Typographic spaces around a Latin word are not pauses (以降は LINE、
            # Twitter 上で). A space after a kanji or katakana term stays: it
            # separates a gloss (間奏 Interlude) and becomes a pause later.
            before = head.rstrip(" ")[-1:]
            if not reading or _is_hiragana(before):
                head = head.rstrip(" ")
        out.append(head)
        out.append(reading)
        cursor = previous_end = end
        if text[cursor : cursor + 1] == " ":
            after = text[cursor + 1 : cursor + 2]
            continues = after in _PARTICLE_STARTS or _is_kanji(after)
            if not reading or not after or continues:
                cursor += 1
    out.append(text[cursor:])
    return "".join(out)
