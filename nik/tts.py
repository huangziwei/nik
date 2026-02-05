from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import csv
import time
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import voice as voice_util
from .text import read_clean_text
from .voice import VoiceConfig

torch = None
Qwen3TTSModel = None
mlx_audio = None
mlx_load_model = None
fugashi = None
wordfreq = None

_KANA_TAGGER = None
_KANA_TAGGER_DIR: Optional[Path] = None
_ZH_LEXICON: Optional[set[str]] = None
_JP_FREQ_CACHE: Dict[str, float] = {}
_RUBY_BASE_READING_CACHE: Dict[str, str] = {}

_END_PUNCT = set("。！？?!…")
_MID_PUNCT = set("、，,;；:：")
_CLOSE_PUNCT = set("」』）】］〉》」』”’\"'")
_DOUBLE_QUOTE_CHARS = {"\"", "“", "”", "«", "»", "„", "‟", "❝", "❞", "＂"}
_SINGLE_QUOTE_CHARS = {"'", "‘", "’", "‚", "‛", "＇"}
_LEADING_ELISIONS = {
    "tis",
    "twas",
    "twere",
    "twill",
    "til",
    "em",
    "cause",
    "bout",
    "round",
}
_JP_QUOTE_CHARS = {
    "「",
    "」",
    "『",
    "』",
    "《",
    "》",
    "〈",
    "〉",
    "【",
    "】",
    "〔",
    "〕",
    "［",
    "］",
    "｢",
    "｣",
    "〝",
    "〞",
    "〟",
}
_JP_OPEN_QUOTES = {"「", "『", "《", "〈", "【", "〔", "［", "｢", "〝"}
_JP_CLOSE_QUOTES = {"」", "』", "》", "〉", "】", "〕", "］", "｣", "〞", "〟"}
_DASH_RUN_RE = re.compile(r"[‐‑‒–—―─━]{2,}")
_KANJI_SAFE_MARKS = {"々", "〆", "ヶ", "ヵ", "ゝ", "ゞ"}
_RUBY_YOON_LARGE = set("やゆよヤユヨ")
_RUBY_YOON_SMALL_MAP = str.maketrans("やゆよヤユヨ", "ゃゅょャュョ")
_RUBY_YOON_SMALL_KATA_MAP = str.maketrans("ヤユヨ", "ャュョ")

_SHORT_TAIL_PUNCT = "。"
_SHORT_TAIL_MAX_CHARS = 12
try:
    _JP_COMMON_ZIPF = float(os.environ.get("NIK_JP_COMMON_ZIPF", "4.0") or 4.0)
except ValueError:
    _JP_COMMON_ZIPF = 4.0

DEFAULT_TORCH_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
MIN_MLX_AUDIO_VERSION = "0.3.1"
FORCED_LANGUAGE = "Japanese"
FORCED_LANG_CODE = "ja"
READING_TEMPLATE_FILENAME = "global-reading-overrides.md"
UNIDIC_URL = "https://clrd.ninjal.ac.jp/unidic_archive/2512/unidic-cwj-202512_full.zip"
UNIDIC_DIR_ENV = "NIK_UNIDIC_DIR"
UNIDIC_URL_ENV = "NIK_UNIDIC_URL"
UNIDIC_DIRNAME = "unidic-cwj-202512_full"

_DIGIT_KANA = {
    "0": "ゼロ",
    "1": "いち",
    "2": "に",
    "3": "さん",
    "4": "よん",
    "5": "ご",
    "6": "ろく",
    "7": "なな",
    "8": "はち",
    "9": "きゅう",
}

_KANJI_DIGITS = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}

_KANJI_DIGIT_READINGS = {
    "〇": "ゼロ",
    "零": "ゼロ",
    "一": "いち",
    "二": "に",
    "三": "さん",
    "四": "よん",
    "五": "ご",
    "六": "ろく",
    "七": "なな",
    "八": "はち",
    "九": "きゅう",
    "十": "じゅう",
}

_KANJI_UNITS = ["", "十", "百", "千"]
_KANJI_BIG_UNITS = ["", "万", "億", "兆", "京"]

_MONTH_READINGS = {
    1: "いちがつ",
    2: "にがつ",
    3: "さんがつ",
    4: "しがつ",
    5: "ごがつ",
    6: "ろくがつ",
    7: "しちがつ",
    8: "はちがつ",
    9: "くがつ",
    10: "じゅうがつ",
    11: "じゅういちがつ",
    12: "じゅうにがつ",
}

_DAY_READINGS = {
    1: "ついたち",
    2: "ふつか",
    3: "みっか",
    4: "よっか",
    5: "いつか",
    6: "むいか",
    7: "なのか",
    8: "ようか",
    9: "ここのか",
    10: "とおか",
    11: "じゅういちにち",
    12: "じゅうににち",
    13: "じゅうさんにち",
    14: "じゅうよっか",
    15: "じゅうごにち",
    16: "じゅうろくにち",
    17: "じゅうしちにち",
    18: "じゅうはちにち",
    19: "じゅうきゅうにち",
    20: "はつか",
    21: "にじゅういちにち",
    22: "にじゅうににち",
    23: "にじゅうさんにち",
    24: "にじゅうよっか",
    25: "にじゅうごにち",
    26: "にじゅうろくにち",
    27: "にじゅうしちにち",
    28: "にじゅうはちにち",
    29: "にじゅうきゅうにち",
    30: "さんじゅうにち",
    31: "さんじゅういちにち",
}

_COUNTERS = [
    "番目",
    "回目",
    "人目",
    "冊目",
    "巻目",
    "章",
    "話",
    "巻",
    "頁",
    "ページ",
    "行",
    "列",
    "文字",
    "時間",
    "分間",
    "秒間",
    "年間",
    "ヶ月間",
    "か月間",
    "カ月間",
    "ヵ月間",
    "週間",
    "日間",
    "年",
    "月",
    "日",
    "時",
    "分",
    "秒",
    "人",
    "名",
    "匹",
    "羽",
    "頭",
    "本",
    "枚",
    "個",
    "台",
    "回",
    "度",
    "階",
    "着",
    "件",
    "社",
    "歳",
    "才",
    "円",
    "丁目",
    "番",
]


@dataclass(frozen=True)
class ChapterInput:
    index: int
    id: str
    title: str
    text: str
    path: Optional[str] = None


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def chapter_id_from_path(index: int, title: str, rel_path: Optional[str]) -> str:
    if rel_path:
        stem = Path(rel_path).stem
        if stem:
            return stem
    return f"{index:04d}-chapter"


def _lazy_import_tts() -> None:
    global torch, Qwen3TTSModel
    if torch is not None and Qwen3TTSModel is not None:
        return
    try:
        import torch as _torch
        from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Torch backend dependencies are missing. "
            "Install them with `uv sync --extra torch`."
        ) from exc
    torch = _torch
    Qwen3TTSModel = _Qwen3TTSModel


def _require_tts() -> None:
    _lazy_import_tts()


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _parse_version(value: str) -> tuple:
    try:  # Prefer packaging if available.
        from packaging.version import Version  # type: ignore

        return (Version(value),)
    except Exception:
        parts = []
        for token in re.split(r"[.+-]", value):
            if not token:
                continue
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
            else:
                parts.append(0)
        return tuple(parts)


def _version_at_least(value: str, minimum: str) -> bool:
    left = _parse_version(value)
    right = _parse_version(minimum)
    return left >= right


def _mlx_audio_version() -> Optional[str]:
    try:
        return importlib.metadata.version("mlx-audio")
    except Exception:
        return None


def _is_mlx_audio_compatible() -> bool:
    version = _mlx_audio_version()
    if not version:
        return False
    return _version_at_least(version, MIN_MLX_AUDIO_VERSION)


def _has_mlx_audio() -> bool:
    if importlib.util.find_spec("mlx_audio") is None:
        return False
    return _is_mlx_audio_compatible()


def _select_backend(value: Optional[str]) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in {"torch", "mlx"}:
        if normalized == "mlx" and not _is_mlx_audio_compatible():
            raise ValueError(
                f"mlx-audio>={MIN_MLX_AUDIO_VERSION} is required for MLX backend. "
                "Install it with `uv sync --prerelease=allow`."
            )
        return normalized
    if normalized in {"", "auto"}:
        if _is_apple_silicon():
            if _has_mlx_audio():
                return "mlx"
            raise ValueError(
                "MLX backend requires mlx-audio on Apple Silicon. "
                "Install it with `uv sync --prerelease=allow`."
            )
        return "torch"
    raise ValueError(f"Unsupported backend: {value}")


def _default_model_for_backend(backend: str) -> str:
    return DEFAULT_MLX_MODEL if backend == "mlx" else DEFAULT_TORCH_MODEL


def _resolve_model_name(value: Optional[str], backend: str) -> str:
    if value:
        return value
    return _default_model_for_backend(backend)


def _lazy_import_mlx_audio() -> None:
    global mlx_audio, mlx_load_model
    if mlx_audio is not None and mlx_load_model is not None:
        return
    if not _is_mlx_audio_compatible():
        raise RuntimeError(
            f"mlx-audio>={MIN_MLX_AUDIO_VERSION} is required for MLX backend. "
            "Install it with `uv sync --prerelease=allow`."
        )
    try:
        import mlx_audio as _mlx_audio
        from mlx_audio.tts.utils import load_model as _load_model
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "mlx-audio is not installed. Install it with "
            "`uv sync --prerelease=allow`."
        ) from exc
    mlx_audio = _mlx_audio
    mlx_load_model = _load_model


def _load_mlx_model(model_name: str):
    _lazy_import_mlx_audio()
    return mlx_load_model(model_name)


def _mlx_kwarg(name: str, signature: inspect.Signature) -> bool:
    return name in signature.parameters


def _normalize_lang_code(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in {"japanese", "ja", "jp"}:
        return "ja"
    if lowered in {"english", "en"}:
        return "en"
    if lowered in {"chinese", "zh", "zh-cn", "zh-hans", "mandarin"}:
        return "zh"
    if lowered in {"korean", "ko"}:
        return "ko"
    return lowered


def _generate_audio_mlx(
    model: object,
    text: str,
    voice_config: Optional[VoiceConfig] = None,
) -> Tuple[List[np.ndarray], int]:
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise RuntimeError("MLX model does not expose generate().")

    kwargs: dict[str, object] = {}
    try:
        sig = inspect.signature(generate)
    except (TypeError, ValueError):
        sig = None

    if sig and voice_config:
        if voice_config.name and _mlx_kwarg("voice", sig):
            kwargs["voice"] = voice_config.name
        lang_code = FORCED_LANG_CODE
        if _mlx_kwarg("lang_code", sig):
            kwargs["lang_code"] = lang_code
        elif _mlx_kwarg("language", sig):
            kwargs["language"] = lang_code
        if voice_config.ref_audio:
            for key in ("ref_audio", "reference_audio", "speaker_wav", "speaker_audio"):
                if _mlx_kwarg(key, sig):
                    kwargs[key] = voice_config.ref_audio
                    break
        if voice_config.ref_text:
            for key in ("ref_text", "reference_text", "prompt_text"):
                if _mlx_kwarg(key, sig):
                    kwargs[key] = voice_config.ref_text
                    break
    elif sig:
        lang_code = FORCED_LANG_CODE
        if _mlx_kwarg("lang_code", sig):
            kwargs["lang_code"] = lang_code
        elif _mlx_kwarg("language", sig):
            kwargs["language"] = lang_code

    outputs = generate(text, **kwargs)
    if isinstance(outputs, np.ndarray):
        audio = outputs
        sample_rate = getattr(outputs, "sample_rate", None)
        return [np.asarray(audio).flatten()], int(sample_rate or 24000)

    audio_parts: List[np.ndarray] = []
    sample_rate: Optional[int] = None
    for result in outputs:
        audio = getattr(result, "audio", result)
        audio_parts.append(np.asarray(audio).flatten())
        if sample_rate is None:
            sample_rate = getattr(result, "sample_rate", None)
    if not audio_parts:
        return [], int(sample_rate or 24000)
    if sample_rate is None:
        sample_rate = getattr(model, "sample_rate", None)
    return audio_parts, int(sample_rate or 24000)

def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _trim_span(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _advance_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _space_pause_split_indices(text: str) -> set[int]:
    if not text:
        return set()
    splits: set[int] = set()
    start = 0
    length = len(text)
    while start < length:
        end = text.find("\n", start)
        if end == -1:
            end = length
        line = text[start:end]
        if line:
            candidates: List[int] = []
            for idx, ch in enumerate(line):
                if ch not in (" ", "\u3000"):
                    continue
                prev = line[idx - 1] if idx > 0 else ""
                next_ch = line[idx + 1] if idx + 1 < len(line) else ""
                if _is_japanese_char(prev) and _is_japanese_char(next_ch):
                    candidates.append(idx)
            if len(candidates) == 1:
                has_punct = any(ch in _END_PUNCT or ch in _MID_PUNCT for ch in line)
                has_ascii = any(ch.isascii() and ch.isalnum() for ch in line)
                if not has_punct and not has_ascii:
                    splits.add(start + candidates[0])
        start = end + 1
    return splits


def split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    space_splits = _space_pause_split_indices(text)
    start = 0
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if i in space_splits:
            span = _trim_span(text, start, i)
            if span:
                spans.append(span)
            i += 1
            i = _advance_ws(text, i)
            start = i
            continue
        if ch == "\n":
            span = _trim_span(text, start, i)
            if span:
                spans.append(span)
            i += 1
            i = _advance_ws(text, i)
            start = i
            continue
        if ch in _END_PUNCT:
            j = i + 1
            while j < length and text[j] in _END_PUNCT:
                j += 1
            while j < length and text[j] in _CLOSE_PUNCT:
                j += 1
            span = _trim_span(text, start, j)
            if span:
                spans.append(span)
            j = _advance_ws(text, j)
            start = j
            i = j
            continue
        i += 1
    span = _trim_span(text, start, length)
    if span:
        spans.append(span)
    return spans


def _split_long_span(
    text: str, start: int, end: int, max_chars: int
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    i = start
    while i < end:
        j = min(end, i + max_chars)
        if j < end:
            split_at = -1
            for punct in _MID_PUNCT:
                idx = text.rfind(punct, i, j)
                if idx > split_at:
                    split_at = idx
            if split_at > i:
                j = split_at + 1
        span = _trim_span(text, i, j)
        if span:
            spans.append(span)
        i = _advance_ws(text, j)
    return spans


def make_chunk_spans(
    text: str, max_chars: int, chunk_mode: str = "japanese"
) -> List[Tuple[int, int]]:
    _ = chunk_mode
    spans: List[Tuple[int, int]] = []
    for sent_start, sent_end in split_sentence_spans(text):
        if max_chars > 0 and sent_end - sent_start > max_chars:
            spans.extend(_split_long_span(text, sent_start, sent_end, max_chars))
        else:
            spans.append((sent_start, sent_end))
    return spans


def make_chunks(text: str, max_chars: int, chunk_mode: str = "japanese") -> List[str]:
    spans = make_chunk_spans(text, max_chars=max_chars, chunk_mode=chunk_mode)
    return [text[start:end] for start, end in spans]


def _strip_double_quotes(text: str) -> str:
    if not text:
        return text
    return "".join(ch for ch in text if ch not in _DOUBLE_QUOTE_CHARS)


def _strip_single_quotes(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch not in _SINGLE_QUOTE_CHARS:
            out.append(ch)
            continue
        prev = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < len(text) else ""
        if (
            prev
            and next_ch
            and prev.isascii()
            and next_ch.isascii()
            and prev.isalnum()
            and next_ch.isalnum()
        ):
            out.append(ch)
            continue
        if (
            (not prev or not prev.isalnum())
            and next_ch
            and next_ch.isascii()
            and next_ch.isalpha()
        ):
            end = idx + 1
            while end < len(text) and text[end].isascii() and text[end].isalpha():
                end += 1
            word = text[idx + 1 : end].lower()
            if word in _LEADING_ELISIONS:
                out.append(ch)
                continue
        continue
    return "".join(out)


def _strip_format_chars(text: str) -> str:
    if not text:
        return text
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _strip_jp_quotes(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    length = len(text)
    for idx, ch in enumerate(text):
        if ch not in _JP_QUOTE_CHARS:
            out.append(ch)
            continue
        if ch in _JP_CLOSE_QUOTES:
            prev = text[idx - 1] if idx > 0 else ""
            j = idx + 1
            while j < length and text[j] in _JP_QUOTE_CHARS:
                j += 1
            next_ch = text[j] if j < length else ""
            if (
                prev
                and next_ch
                and _is_japanese_char(prev)
                and _is_japanese_char(next_ch)
                and prev not in _END_PUNCT
                and prev not in _MID_PUNCT
                and next_ch not in _END_PUNCT
                and next_ch not in _MID_PUNCT
            ):
                out.append("、")
        continue
    return "".join(out)


def _append_short_tail_punct(text: str) -> str:
    if not text:
        return text
    if len(text) > _SHORT_TAIL_MAX_CHARS:
        return text
    if not _has_japanese(text):
        return text
    idx = len(text) - 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    if idx < 0:
        return text
    last = text[idx]
    if last in _END_PUNCT or last in _MID_PUNCT:
        return text
    if last in _CLOSE_PUNCT:
        j = idx - 1
        while j >= 0 and text[j] in _CLOSE_PUNCT:
            j -= 1
        if j >= 0 and (text[j] in _END_PUNCT or text[j] in _MID_PUNCT):
            return text
    return text + _SHORT_TAIL_PUNCT


def _has_kanji(text: str) -> bool:
    if not text:
        return False
    ranges = (
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F),  # Extension C
        (0x2B740, 0x2B81F),  # Extension D
        (0x2B820, 0x2CEAF),  # Extension E
        (0x2CEB0, 0x2EBEF),  # Extension F
        (0x30000, 0x3134F),  # Extension G
    )
    for ch in text:
        code = ord(ch)
        for start, end in ranges:
            if start <= code <= end:
                return True
    return False


def _is_kanji_char(ch: str) -> bool:
    if not ch:
        return False
    if len(ch) != 1:
        return False
    code = ord(ch)
    ranges = (
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F),  # Extension C
        (0x2B740, 0x2B81F),  # Extension D
        (0x2B820, 0x2CEAF),  # Extension E
        (0x2CEB0, 0x2EBEF),  # Extension F
        (0x30000, 0x3134F),  # Extension G
    )
    for start, end in ranges:
        if start <= code <= end:
            return True
    return False


def _is_kanji_boundary_char(ch: str) -> bool:
    if not ch:
        return False
    if ch in _KANJI_SAFE_MARKS:
        return True
    return _is_kanji_char(ch)


def _has_kana(text: str) -> bool:
    if not text:
        return False
    return any(_is_kana_char(ch) for ch in text)


def _is_kanji_only(surface: str) -> bool:
    if not surface:
        return False
    if _has_kana(surface):
        return False
    if any(ch.isascii() and ch.isalnum() for ch in surface):
        return False
    return _has_kanji(surface)


def _has_japanese(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x309F:  # Hiragana
            return True
        if 0x30A0 <= code <= 0x30FF:  # Katakana
            return True
        if 0x31F0 <= code <= 0x31FF:  # Katakana Phonetic Extensions
            return True
        if 0xFF66 <= code <= 0xFF9D:  # Halfwidth Katakana
            return True
    return _has_kanji(text)


def _normalize_kana_style(value: Optional[str]) -> str:
    normalized = (value or "mixed").strip().lower()
    if normalized in {"off", "none", "disable", "disabled", "no"}:
        return "off"
    if normalized in {"partial", "part", "p"}:
        return "partial"
    if normalized in {"mixed", "mix", "m"}:
        return "mixed"
    if normalized in {"hiragana", "hira", "h"}:
        return "hiragana"
    if normalized in {"katakana", "kata", "k"}:
        return "katakana"
    return "mixed"


def _katakana_to_hiragana(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _hiragana_to_katakana(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _is_kana_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    if 0x3040 <= code <= 0x309F:  # Hiragana
        return True
    if 0x30A0 <= code <= 0x30FF:  # Katakana
        return True
    if 0x31F0 <= code <= 0x31FF:  # Katakana Phonetic Extensions
        return True
    if 0xFF66 <= code <= 0xFF9D:  # Halfwidth Katakana
        return True
    return False


def _is_hiragana_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return 0x3040 <= code <= 0x309F


def _is_japanese_char(ch: str) -> bool:
    if not ch:
        return False
    if _is_kana_char(ch):
        return True
    code = ord(ch)
    ranges = (
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F),  # Extension C
        (0x2B740, 0x2B81F),  # Extension D
        (0x2B820, 0x2CEAF),  # Extension E
        (0x2CEB0, 0x2EBEF),  # Extension F
        (0x30000, 0x3134F),  # Extension G
    )
    for start, end in ranges:
        if start <= code <= end:
            return True
    return False


def _split_surface_kana_sequences(text: str) -> List[str]:
    sequences: List[str] = []
    buf: List[str] = []
    for ch in text:
        if _is_kana_char(ch):
            buf.append(ch)
        else:
            if buf:
                sequences.append("".join(buf))
                buf = []
    if buf:
        sequences.append("".join(buf))
    return sequences


def _kana_char_count(text: str) -> int:
    return sum(1 for ch in text if _is_kana_char(ch))


def _trailing_kana_sequence(text: str) -> str:
    if not text:
        return ""
    idx = len(text)
    while idx > 0 and _is_kana_char(text[idx - 1]):
        idx -= 1
    return text[idx:] if idx < len(text) else ""


def _feature_value(feature: Any, names: Sequence[str]) -> str:
    if not feature:
        return ""
    for name in names:
        value = getattr(feature, name, None)
        if value and value != "*":
            return str(value)
    return ""


def _maybe_canonicalize_surface(surface: str, token: Any, reading_kata: str) -> str:
    if not surface:
        return surface
    feature = getattr(token, "feature", None)
    if not feature:
        return surface
    lemma = _feature_value(feature, ("lemma",))
    if not lemma or lemma == surface:
        return surface
    if _kana_char_count(lemma) <= _kana_char_count(surface):
        return surface
    if not reading_kata:
        return surface
    lemma_reading = _feature_value(
        feature, ("lForm", "kanaBase", "pronBase", "formBase", "form")
    )
    if not lemma_reading:
        return surface
    lemma_reading_kata = _hiragana_to_katakana(
        unicodedata.normalize("NFKC", lemma_reading)
    )
    reading_kata = _hiragana_to_katakana(unicodedata.normalize("NFKC", reading_kata))
    tail = _trailing_kana_sequence(lemma)
    if not tail:
        if reading_kata == lemma_reading_kata:
            return lemma
        return surface
    tail_kata = _hiragana_to_katakana(unicodedata.normalize("NFKC", tail))
    if not tail_kata:
        return surface
    if not lemma_reading_kata.endswith(tail_kata):
        return surface
    if len(reading_kata) < len(tail_kata):
        return surface
    if lemma_reading_kata[:-len(tail_kata)] != reading_kata[:-len(tail_kata)]:
        return surface
    new_tail_kata = reading_kata[-len(tail_kata):]
    if _is_hiragana_char(tail[0]):
        new_tail = _katakana_to_hiragana(new_tail_kata)
    else:
        new_tail = new_tail_kata
    return lemma[:-len(tail)] + new_tail


def _apply_surface_kana(reading_kata: str, surface: str) -> str:
    if not reading_kata:
        return reading_kata
    sequences = _split_surface_kana_sequences(surface)
    if not sequences:
        return reading_kata
    out: List[str] = []
    idx = 0
    for seq in sequences:
        seq_kata = _hiragana_to_katakana(unicodedata.normalize("NFKC", seq))
        if not seq_kata:
            continue
        pos = reading_kata.find(seq_kata, idx)
        if pos == -1:
            out.append(reading_kata[idx:])
            return "".join(out)
        out.append(reading_kata[idx:pos])
        out.append(seq)
        idx = pos + len(seq_kata)
    out.append(reading_kata[idx:])
    return "".join(out)


def _japanese_space_to_pause(text: str, *, allow_full_stop: bool = True) -> str:
    if not text or " " not in text:
        return text
    length = len(text)
    japanese_spaces: List[int] = []
    for idx, ch in enumerate(text):
        if ch != " ":
            continue
        prev = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < length else ""
        if _is_japanese_char(prev) and _is_japanese_char(next_ch):
            japanese_spaces.append(idx)
    if not japanese_spaces:
        return text
    has_punct = any(ch in _END_PUNCT or ch in _MID_PUNCT for ch in text)
    has_ascii = any(ch.isascii() and ch.isalnum() for ch in text)
    use_full_stop = (
        allow_full_stop
        and len(japanese_spaces) == 1
        and not has_punct
        and not has_ascii
    )
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch != " ":
            out.append(ch)
            continue
        prev = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < length else ""
        if _is_japanese_char(prev) and _is_japanese_char(next_ch):
            out.append("。" if use_full_stop else "、")
        else:
            out.append(ch)
    return "".join(out)


def _digit_seq_to_kana(seq: str) -> str:
    if not seq:
        return seq
    return "".join(_DIGIT_KANA.get(ch, ch) for ch in seq)


def _group_to_kanji(value: int) -> str:
    if value <= 0:
        return ""
    out = []
    thousands = value // 1000
    hundreds = (value // 100) % 10
    tens = (value // 10) % 10
    ones = value % 10
    digits = [thousands, hundreds, tens, ones]
    for idx, digit in enumerate(digits):
        if digit == 0:
            continue
        unit = _KANJI_UNITS[len(digits) - idx - 1]
        if digit == 1 and unit:
            out.append(unit)
        else:
            out.append(f"{_KANJI_DIGITS.get(digit, '')}{unit}")
    return "".join(out)


def _int_to_kanji(value: int) -> str:
    if value == 0:
        return "零"
    out = []
    group_index = 0
    n = value
    while n > 0 and group_index < len(_KANJI_BIG_UNITS):
        group = n % 10000
        if group:
            part = _group_to_kanji(group)
            unit = _KANJI_BIG_UNITS[group_index]
            out.append(f"{part}{unit}")
        n //= 10000
        group_index += 1
    if n > 0:
        return _digit_seq_to_kana(str(value))
    return "".join(reversed(out))


def _month_reading(value: int) -> str:
    return _MONTH_READINGS.get(value, "")


def _day_reading(value: int) -> str:
    return _DAY_READINGS.get(value, "")


def _normalize_numbers(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)

    def _safe_int(raw: str) -> Optional[int]:
        cleaned = raw.replace(",", "")
        if not cleaned.isdigit():
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None

    def _to_kanji_number(raw: str) -> str:
        cleaned = raw.replace(",", "")
        if not cleaned.isdigit():
            return raw
        if len(cleaned) > 16:
            return _digit_seq_to_kana(cleaned)
        if len(cleaned) > 1 and cleaned.startswith("0"):
            return _digit_seq_to_kana(cleaned)
        return _int_to_kanji(int(cleaned))

    def _replace_ymd(match: re.Match) -> str:
        year = _safe_int(match.group("year") or "")
        month = _safe_int(match.group("month") or "")
        day = _safe_int(match.group("day") or "")
        if year is None or month is None or day is None:
            return match.group(0)
        year_part = f"{_int_to_kanji(year)}年"
        month_part = _month_reading(month)
        day_part = _day_reading(day)
        if not month_part or not day_part:
            month_part = f"{_int_to_kanji(month)}月"
            day_part = f"{_int_to_kanji(day)}日"
        return f"{year_part}{month_part}{day_part}"

    def _replace_ym(match: re.Match) -> str:
        year = _safe_int(match.group("year") or "")
        month = _safe_int(match.group("month") or "")
        if year is None or month is None:
            return match.group(0)
        year_part = f"{_int_to_kanji(year)}年"
        month_part = _month_reading(month) or f"{_int_to_kanji(month)}月"
        return f"{year_part}{month_part}"

    def _replace_md(match: re.Match) -> str:
        month = _safe_int(match.group("month") or "")
        day = _safe_int(match.group("day") or "")
        if month is None or day is None:
            return match.group(0)
        month_part = _month_reading(month)
        day_part = _day_reading(day)
        if not month_part or not day_part:
            month_part = f"{_int_to_kanji(month)}月"
            day_part = f"{_int_to_kanji(day)}日"
        return f"{month_part}{day_part}"

    def _replace_slash_date(match: re.Match) -> str:
        a = match.group("a") or ""
        b = match.group("b") or ""
        c = match.group("c") or ""
        ai = _safe_int(a)
        bi = _safe_int(b)
        ci = _safe_int(c) if c else None
        if ai is None or bi is None:
            return match.group(0)
        if ci is not None:
            if len(a) == 4:
                year, month, day = ai, bi, ci
            elif len(c) == 4:
                year, month, day = ci, ai, bi
            else:
                year, month, day = ai, bi, ci
            year_part = f"{_int_to_kanji(year)}年"
            month_part = _month_reading(month) or f"{_int_to_kanji(month)}月"
            day_part = _day_reading(day) or f"{_int_to_kanji(day)}日"
            return f"{year_part}{month_part}{day_part}"
        if 1 <= ai <= 12 and 1 <= bi <= 31:
            month_part = _month_reading(ai) or f"{_int_to_kanji(ai)}月"
            day_part = _day_reading(bi) or f"{_int_to_kanji(bi)}日"
            return f"{month_part}{day_part}"
        return match.group(0)

    def _replace_time(match: re.Match) -> str:
        h = _safe_int(match.group("h") or "")
        m = _safe_int(match.group("m") or "")
        s = _safe_int(match.group("s") or "") if match.group("s") else None
        if h is None or m is None:
            return match.group(0)
        out = f"{_int_to_kanji(h)}時{_int_to_kanji(m)}分"
        if s is not None:
            out = f"{out}{_int_to_kanji(s)}秒"
        return out

    def _replace_decimal(match: re.Match) -> str:
        left = match.group("left") or ""
        right = match.group("right") or ""
        if not left.isdigit() or not right.isdigit():
            return match.group(0)
        left_part = _to_kanji_number(left)
        right_part = _digit_seq_to_kana(right)
        return f"{left_part}てん{right_part}"

    def _replace_percent(match: re.Match) -> str:
        num = match.group("num") or ""
        if not num:
            return match.group(0)
        parts = num.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            left = _to_kanji_number(parts[0])
            right = _digit_seq_to_kana(parts[1])
            value = f"{left}てん{right}"
        else:
            value = _to_kanji_number(num)
        return f"{value}パーセント"

    def _replace_counter(match: re.Match) -> str:
        num = match.group("num") or ""
        counter = match.group("counter") or ""
        if not num or not counter:
            return match.group(0)
        if counter == "話" and num == "1":
            return "いちわ"
        return f"{_to_kanji_number(num)}{counter}"

    def _replace_ordinal(match: re.Match) -> str:
        num = match.group("num") or ""
        suffix = match.group("suffix") or ""
        if not num:
            return match.group(0)
        return f"第{_to_kanji_number(num)}{suffix}"

    def _replace_alnum(match: re.Match) -> str:
        prefix = match.group("prefix") or ""
        num = match.group("num") or ""
        suffix = match.group("suffix") or ""
        if not num:
            return match.group(0)
        return f"{prefix}{_digit_seq_to_kana(num)}{suffix}"

    def _replace_standalone(match: re.Match) -> str:
        num = match.group("num") or ""
        if not num:
            return match.group(0)
        return _digit_seq_to_kana(num)

    def _replace_fallback(match: re.Match) -> str:
        num = match.group("num") or ""
        if not num:
            return match.group(0)
        if len(num) > 1 and num.startswith("0"):
            return _digit_seq_to_kana(num)
        return _to_kanji_number(num)

    def _replace_kanji_digit_list(match: re.Match) -> str:
        seq = match.group("seq") or ""
        if not seq:
            return match.group(0)
        out: List[str] = []
        for ch in seq:
            out.append(_KANJI_DIGIT_READINGS.get(ch, ch))
        return "".join(out)

    text = re.sub(
        r"(?<!\d)(?P<year>\d{2,4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日",
        _replace_ymd,
        text,
    )
    text = re.sub(
        r"(?<!\d)(?P<year>\d{2,4})年(?P<month>\d{1,2})月",
        _replace_ym,
        text,
    )
    text = re.sub(
        r"(?<!\d)(?P<month>\d{1,2})月(?P<day>\d{1,2})日",
        _replace_md,
        text,
    )
    text = re.sub(
        r"(?<!\d)(?P<a>\d{1,4})[/-](?P<b>\d{1,2})(?:[/-](?P<c>\d{1,2}))?(?!\d)",
        _replace_slash_date,
        text,
    )
    text = re.sub(
        r"(?<!\d)(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?(?!\d)",
        _replace_time,
        text,
    )
    text = re.sub(
        r"(?<!\d)(?P<num>\d+(?:\.\d+)?)%",
        _replace_percent,
        text,
    )
    text = re.sub(
        r"(?<!\d)(?P<left>\d+)\.(?P<right>\d+)(?!\d)",
        _replace_decimal,
        text,
    )
    counters = "|".join(sorted({re.escape(c) for c in _COUNTERS}, key=len, reverse=True))
    if counters:
        text = re.sub(
            rf"(?P<num>\d+)(?P<counter>{counters})",
            _replace_counter,
            text,
        )
        text = re.sub(
            rf"第(?P<num>\d+)(?P<suffix>{counters})?",
            _replace_ordinal,
            text,
        )
    text = re.sub(
        r"(?P<prefix>[A-Za-z]+)(?P<num>\d+)(?P<suffix>[A-Za-z]*)",
        _replace_alnum,
        text,
    )
    kanji_digit_chars = "".join(_KANJI_DIGIT_READINGS.keys())
    text = re.sub(
        rf"(?P<seq>[{kanji_digit_chars}](?:[、，,・][{kanji_digit_chars}])+)",
        _replace_kanji_digit_list,
        text,
    )
    text = re.sub(
        r"(?<![0-9A-Za-z一-龯ぁ-んァ-ヶ々〆〤])(?P<num>\d+)(?![0-9A-Za-z一-龯ぁ-んァ-ヶ々〆〤])",
        _replace_standalone,
        text,
    )
    text = re.sub(r"(?P<num>\d+)", _replace_fallback, text)
    return text


def _is_kana_reading(text: str) -> bool:
    if not text:
        return False
    allowed = {
        "ー",
        "・",
        "ゝ",
        "ゞ",
        "ヽ",
        "ヾ",
        "ヿ",
    }
    for ch in text:
        code = ord(ch)
        if ch in allowed:
            continue
        if 0x3040 <= code <= 0x309F:  # Hiragana
            continue
        if 0x30A0 <= code <= 0x30FF:  # Katakana
            continue
        if 0x31F0 <= code <= 0x31FF:  # Katakana Phonetic Extensions
            continue
        return False
    return True


def _default_unidic_dir() -> Path:
    return Path.home() / ".cache" / "nik" / UNIDIC_DIRNAME


def _resolve_unidic_dir() -> Path:
    env = os.environ.get(UNIDIC_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return _default_unidic_dir()


def _download_unidic_archive(url: str, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_name = Path(urllib.parse.urlparse(url).path).name or "unidic.zip"
    archive_path = target_dir.parent / archive_name
    if not archive_path.exists():
        sys.stderr.write(f"Downloading UniDic from {url}\n")
        with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    with tempfile.TemporaryDirectory(dir=str(target_dir.parent)) as tmp_dir:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmp_dir)
        candidates = [p.parent for p in Path(tmp_dir).rglob("dicrc")]
        if not candidates:
            raise RuntimeError("UniDic archive missing dicrc.")
        source_dir = candidates[0]
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(source_dir), str(target_dir))


def _ensure_unidic_dir(path: Path) -> Path:
    dicrc_path = path / "dicrc"
    if dicrc_path.exists():
        return path
    if os.environ.get(UNIDIC_DIR_ENV):
        raise RuntimeError(f"UniDic directory missing dicrc: {path}")
    url = os.environ.get(UNIDIC_URL_ENV) or UNIDIC_URL
    _download_unidic_archive(url, path)
    if not dicrc_path.exists():
        raise RuntimeError(f"UniDic download failed to produce dicrc at {path}")
    return path


def _ensure_mecabrc(dict_dir: Path) -> Path:
    mecabrc_path = dict_dir / "mecabrc"
    if mecabrc_path.exists():
        return mecabrc_path
    try:
        mecabrc_path.write_text(f"dicdir = {dict_dir}\n", encoding="utf-8")
        return mecabrc_path
    except Exception:
        return Path("/dev/null")


def _lazy_import_fugashi() -> None:
    global fugashi
    if fugashi is not None:
        return
    try:
        import fugashi as _fugashi
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Kana normalization requires fugashi. Install it with `uv sync`."
        ) from exc
    fugashi = _fugashi


def _lazy_import_wordfreq() -> None:
    global wordfreq
    if wordfreq is not None:
        return
    try:
        import wordfreq as _wordfreq
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Japanese frequency guard requires wordfreq. Install it with `uv sync`."
        ) from exc
    wordfreq = _wordfreq


def _jp_zipf_frequency(word: str) -> float:
    cached = _JP_FREQ_CACHE.get(word)
    if cached is not None:
        return cached
    _lazy_import_wordfreq()
    freq = float(wordfreq.zipf_frequency(word, "ja"))
    _JP_FREQ_CACHE[word] = freq
    return freq


def _is_common_japanese_word(word: str) -> bool:
    if not word:
        return False
    return _jp_zipf_frequency(word) >= _JP_COMMON_ZIPF


def _load_zh_lexicon() -> set[str]:
    global _ZH_LEXICON
    if _ZH_LEXICON is not None:
        return _ZH_LEXICON
    lex: set[str] = set()
    lex_path = Path(__file__).resolve().parent / "templates" / "modern-chinese-common-words.csv"
    if not lex_path.exists():
        raise RuntimeError(
            f"Partial kana mode requires {lex_path}. Missing file."
        )
    try:
        with lex_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = str(row.get("word") or "").strip()
                if not word:
                    continue
                lex.add(word)
    except Exception as exc:
        raise RuntimeError("Failed to load modern-chinese-common-words.csv.") from exc
    _ZH_LEXICON = lex
    return lex


def _get_kana_tagger() -> Any:
    global _KANA_TAGGER, _KANA_TAGGER_DIR
    dict_dir = _ensure_unidic_dir(_resolve_unidic_dir())
    if _KANA_TAGGER is not None and _KANA_TAGGER_DIR == dict_dir:
        return _KANA_TAGGER
    _lazy_import_fugashi()
    mecabrc = _ensure_mecabrc(dict_dir)
    _KANA_TAGGER = fugashi.Tagger(f"-r {mecabrc} -d {dict_dir}")
    _KANA_TAGGER_DIR = dict_dir
    return _KANA_TAGGER


def _extract_token_reading(token: Any) -> str:
    feature = getattr(token, "feature", None)
    for attr in ("kana", "pron", "reading"):
        value = getattr(feature, attr, None)
        if value and value != "*":
            value = str(value)
            if _is_kana_reading(value):
                return value
    if feature:
        raw = str(feature)
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            for candidate in reversed(parts):
                if candidate and candidate != "*" and _is_kana_reading(candidate):
                    return candidate
    return ""


def _extract_token_attrs(token: Any) -> Dict[str, str]:
    attrs = {"goshu": "", "pos1": "", "pos2": "", "pos3": "", "pos4": "", "type": ""}
    feature = getattr(token, "feature", None)
    if not feature:
        return attrs

    def _pick(names: Sequence[str]) -> str:
        for name in names:
            value = getattr(feature, name, None)
            if value and value != "*":
                return str(value)
        return ""

    attrs["goshu"] = _pick(("goshu",))
    attrs["pos1"] = _pick(("pos1", "pos"))
    attrs["pos2"] = _pick(("pos2",))
    attrs["pos3"] = _pick(("pos3",))
    attrs["pos4"] = _pick(("pos4",))
    attrs["type"] = _pick(("type",))

    raw = str(feature)
    if raw and "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        if not attrs["pos1"] and len(parts) > 0:
            attrs["pos1"] = parts[0]
        if not attrs["pos2"] and len(parts) > 1:
            attrs["pos2"] = parts[1]
        if not attrs["pos3"] and len(parts) > 2:
            attrs["pos3"] = parts[2]
        if not attrs["pos4"] and len(parts) > 3:
            attrs["pos4"] = parts[3]
        if not attrs["goshu"]:
            for part in parts:
                if part in {"漢", "和", "混", "外"}:
                    attrs["goshu"] = part
                    break
    return attrs


def _base_reading_kata(base: str, tagger: Any) -> str:
    base = unicodedata.normalize("NFKC", base or "")
    if not base:
        return ""
    cached = _RUBY_BASE_READING_CACHE.get(base)
    if cached is not None:
        return cached
    try:
        tokens = list(tagger(base))
    except Exception:
        _RUBY_BASE_READING_CACHE[base] = ""
        return ""
    parts: List[str] = []
    for token in tokens:
        reading = _extract_token_reading(token)
        if not reading:
            _RUBY_BASE_READING_CACHE[base] = ""
            return ""
        parts.append(_hiragana_to_katakana(reading))
    joined = "".join(parts)
    _RUBY_BASE_READING_CACHE[base] = joined
    return joined


def _normalize_ruby_reading(base: str, reading: str, tagger: Any) -> str:
    if not base or not reading:
        return reading
    if not any(ch in _RUBY_YOON_LARGE for ch in reading):
        return reading
    base_reading = _base_reading_kata(base, tagger)
    if not base_reading:
        return reading
    reading_kata = _hiragana_to_katakana(unicodedata.normalize("NFKC", reading))
    if reading_kata == base_reading:
        return reading
    smallified_kata = reading_kata.translate(_RUBY_YOON_SMALL_KATA_MAP)
    if smallified_kata != base_reading:
        return reading
    return reading.translate(_RUBY_YOON_SMALL_MAP)


def _normalize_ruby_entries(entries: Sequence[dict], tagger: Any) -> List[dict]:
    out: List[dict] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        base = str(item.get("base") or "")
        reading = str(item.get("reading") or "")
        if not base or not reading:
            out.append(item)
            continue
        if not any(ch in _RUBY_YOON_LARGE for ch in reading):
            out.append(item)
            continue
        normalized = _normalize_ruby_reading(base, reading, tagger)
        if normalized == reading:
            out.append(item)
            continue
        updated = dict(item)
        updated["reading"] = normalized
        out.append(updated)
    return out


def _maybe_normalize_ruby_entries(entries: Sequence[dict]) -> List[dict]:
    if not entries:
        return list(entries)
    needs_fix = False
    for item in entries:
        if not isinstance(item, dict):
            continue
        reading = str(item.get("reading") or "")
        if any(ch in _RUBY_YOON_LARGE for ch in reading):
            needs_fix = True
            break
    if not needs_fix:
        return list(entries)
    try:
        tagger = _get_kana_tagger()
    except Exception:
        return list(entries)
    return _normalize_ruby_entries(entries, tagger)


def _should_partial_convert(
    surface: str, attrs: Dict[str, str], zh_lexicon: set[str]
) -> bool:
    if not surface or not _has_kanji(surface):
        return False
    if _has_kana(surface):
        return False
    if any(ch in _KANJI_SAFE_MARKS for ch in surface):
        return False
    length = len(surface)
    if length <= 1:
        return False
    if surface in zh_lexicon:
        return True
    if _is_common_japanese_word(surface):
        return False
    return True


def _resolve_honorific_reading_kata(
    surface: str,
    reading_kata: str,
    attrs: Dict[str, str],
    *,
    tagger: Any,
    next_token: Optional[Any] = None,
) -> str:
    if not surface:
        return reading_kata
    if not surface.startswith("御"):
        return reading_kata
    if not reading_kata:
        return reading_kata
    reading_kata = _hiragana_to_katakana(unicodedata.normalize("NFKC", reading_kata))
    if not _is_kana_reading(reading_kata):
        return reading_kata
    if reading_kata[0] not in {"オ", "ゴ"}:
        return reading_kata
    pos1 = attrs.get("pos1") or ""
    if pos1 and pos1 != "接頭辞":
        return reading_kata

    rest_surface = ""
    if surface == "御":
        if next_token is not None:
            rest_surface = str(getattr(next_token, "surface", "") or "")
    else:
        rest_surface = surface[1:]

    goshu = ""
    if surface == "御" and next_token is not None:
        goshu = _extract_token_attrs(next_token).get("goshu", "")
    if not goshu and rest_surface:
        try:
            rest_tokens = list(tagger(rest_surface))
        except Exception:
            rest_tokens = []
        if rest_tokens:
            goshu = _extract_token_attrs(rest_tokens[0]).get("goshu", "")

    if goshu in {"漢", "外", "混"}:
        return "ゴ" + reading_kata[1:]
    if goshu == "和":
        return "オ" + reading_kata[1:]
    return reading_kata


def _should_convert_honorific_prefix(
    surface: str, reading_kata: str, attrs: Dict[str, str]
) -> bool:
    if surface != "御":
        return False
    if not reading_kata:
        return False
    reading_kata = _hiragana_to_katakana(unicodedata.normalize("NFKC", reading_kata))
    if not _is_kana_reading(reading_kata):
        return False
    pos1 = attrs.get("pos1") or ""
    if pos1 and pos1 != "接頭辞":
        return False
    return True


def _normalize_kana_with_tagger(
    text: str, tagger: Any, *, kana_style: str = "mixed", zh_lexicon: Optional[set[str]] = None
) -> str:
    if not text:
        return text
    kana_style = _normalize_kana_style(kana_style)
    out: List[str] = []
    tokens = list(tagger(text))
    run_action: List[str] = [""] * len(tokens)
    if kana_style == "partial" and tokens:
        if zh_lexicon is None:
            zh_lexicon = _load_zh_lexicon()
        idx = 0
        while idx < len(tokens):
            surface = getattr(tokens[idx], "surface", "") or ""
            if not _is_kanji_only(surface):
                idx += 1
                continue
            end = idx + 1
            while end < len(tokens):
                next_surface = getattr(tokens[end], "surface", "") or ""
                if not _is_kanji_only(next_surface):
                    break
                end += 1
            if end - idx >= 2:
                run_surface = "".join(
                    str(getattr(tokens[pos], "surface", "") or "")
                    for pos in range(idx, end)
                )
                has_numeric = False
                has_counter = False
                for pos in range(idx, end):
                    attrs = _extract_token_attrs(tokens[pos])
                    if attrs.get("pos2") == "数詞" or attrs.get("type") == "数":
                        has_numeric = True
                    if attrs.get("pos3") == "助数詞" or attrs.get("type") == "助数":
                        has_counter = True
                action = ""
                if has_numeric and has_counter:
                    action = "convert"
                elif run_surface and run_surface in (zh_lexicon or set()):
                    action = "convert"
                if action:
                    for pos in range(idx, end):
                        run_action[pos] = action
            idx = end
    for idx, token in enumerate(tokens):
        surface = getattr(token, "surface", "")
        if not surface:
            continue
        if not _has_kanji(surface):
            out.append(surface)
            continue
        reading = _extract_token_reading(token)
        if not reading or reading == "*":
            out.append(surface)
            continue
        reading_kata = _hiragana_to_katakana(reading)
        next_token = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if surface.startswith("御"):
            attrs = _extract_token_attrs(token)
            reading_kata = _resolve_honorific_reading_kata(
                surface,
                reading_kata,
                attrs,
                tagger=tagger,
                next_token=next_token,
            )
        if kana_style == "partial":
            attrs = _extract_token_attrs(token)
            action = run_action[idx]
            surface = _maybe_canonicalize_surface(surface, token, reading_kata)
            if action == "keep":
                out.append(surface)
                continue
            if action == "convert":
                out.append(_apply_surface_kana(reading_kata, surface))
                continue
            if _should_convert_honorific_prefix(surface, reading_kata, attrs):
                out.append(_apply_surface_kana(reading_kata, surface))
                continue
            if _should_partial_convert(surface, attrs, zh_lexicon or set()):
                out.append(_apply_surface_kana(reading_kata, surface))
            else:
                out.append(surface)
            continue
        if kana_style == "katakana":
            out.append(reading_kata)
        elif kana_style == "hiragana":
            out.append(_katakana_to_hiragana(reading_kata))
        else:
            out.append(_apply_surface_kana(reading_kata, surface))
    return "".join(out)


def _normalize_kana_text(text: str, *, kana_style: str = "mixed") -> str:
    kana_style = _normalize_kana_style(kana_style)
    if kana_style == "off":
        return text
    if not _has_kanji(text):
        return text
    tagger = _get_kana_tagger()
    zh_lexicon: Optional[set[str]] = None
    if kana_style == "partial":
        zh_lexicon = _load_zh_lexicon()
    parts = re.split(r"(\s+)", text)
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.isspace():
            out.append(part)
            continue
        if _has_kanji(part):
            out.append(
                _normalize_kana_with_tagger(
                    part,
                    tagger,
                    kana_style=kana_style,
                    zh_lexicon=zh_lexicon,
                )
            )
        else:
            out.append(part)
    return "".join(out)


def prepare_tts_text(text: str, *, add_short_punct: bool = False) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    has_dash_run = bool(_DASH_RUN_RE.search(text))
    text = _DASH_RUN_RE.sub(" ", text)
    text = _strip_format_chars(text)
    text = _strip_jp_quotes(text)
    text = _strip_double_quotes(text)
    text = _strip_single_quotes(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _japanese_space_to_pause(text, allow_full_stop=not has_dash_run)
    if add_short_punct:
        text = _append_short_tail_punct(text)
    return text


def _select_ruby_spans(
    chapter_id: Optional[str],
    text: str,
    ruby_data: dict,
) -> List[dict]:
    if not chapter_id:
        return []
    chapters = ruby_data.get("chapters") if isinstance(ruby_data, dict) else None
    if not isinstance(chapters, dict):
        return []
    entry = chapters.get(chapter_id)
    if not isinstance(entry, dict):
        return []
    text_hash = sha256_str(text)
    clean_hash = entry.get("clean_sha256")
    if isinstance(clean_hash, str) and clean_hash == text_hash:
        spans = entry.get("clean_spans")
        return spans if isinstance(spans, list) else []
    raw_hash = entry.get("raw_sha256")
    if isinstance(raw_hash, str) and raw_hash == text_hash:
        spans = entry.get("raw_spans")
        return spans if isinstance(spans, list) else []
    return []


def _apply_ruby_evidence(
    text: str,
    chapter_id: Optional[str],
    ruby_data: dict,
    skip_bases: Optional[set[str]] = None,
) -> str:
    if not ruby_data:
        return text
    spans = _select_ruby_spans(chapter_id, text, ruby_data)
    if spans:
        spans = _maybe_normalize_ruby_entries(spans)
        text = apply_ruby_spans(text, spans)
    ruby_global = _ruby_global_overrides(ruby_data)
    if ruby_global:
        ruby_global = _maybe_normalize_ruby_entries(ruby_global)
        if skip_bases:
            ruby_global = [
                item
                for item in ruby_global
                if str(item.get("base") or "").strip() not in skip_bases
            ]
        if ruby_global:
            text = apply_reading_overrides(text, ruby_global)
    return text


def _chunk_ruby_spans(
    chunk_span: Optional[Sequence[int]],
    chapter_spans: Sequence[dict],
) -> List[dict]:
    if not chunk_span or not chapter_spans:
        return []
    try:
        chunk_start = int(chunk_span[0])
        chunk_end = int(chunk_span[1])
    except (TypeError, ValueError, IndexError):
        return []
    if chunk_end <= chunk_start:
        return []
    spans: List[dict] = []
    for span in chapter_spans:
        if not isinstance(span, dict):
            continue
        try:
            start = int(span.get("start"))
            end = int(span.get("end"))
        except (TypeError, ValueError):
            continue
        if start < chunk_start or end > chunk_end:
            continue
        reading = str(span.get("reading") or "")
        base = str(span.get("base") or "")
        if not reading:
            continue
        spans.append(
            {
                "start": start - chunk_start,
                "end": end - chunk_start,
                "base": base,
                "reading": reading,
            }
        )
    return spans


def _apply_ruby_evidence_to_chunk(
    chunk_text: str,
    chunk_span: Optional[Sequence[int]],
    chapter_spans: Sequence[dict],
    ruby_data: dict,
    skip_bases: Optional[set[str]] = None,
) -> str:
    text = chunk_text
    spans = _chunk_ruby_spans(chunk_span, chapter_spans)
    if spans:
        spans = _maybe_normalize_ruby_entries(spans)
        text = apply_ruby_spans(text, spans)
    ruby_global = _ruby_global_overrides(ruby_data)
    if ruby_global:
        ruby_global = _maybe_normalize_ruby_entries(ruby_global)
        if skip_bases:
            ruby_global = [
                item
                for item in ruby_global
                if str(item.get("base") or "").strip() not in skip_bases
            ]
        if ruby_global:
            text = apply_reading_overrides(text, ruby_global)
    return text


def load_book_chapters(book_dir: Path) -> List[ChapterInput]:
    toc_path = book_dir / "clean" / "toc.json"
    if not toc_path.exists():
        toc_path = book_dir / "toc.json"
    if not toc_path.exists():
        raise FileNotFoundError(f"Missing toc.json at {toc_path}")

    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    entries = toc.get("chapters", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("toc.json contains no chapters.")

    chapters: List[ChapterInput] = []
    for fallback_idx, entry in enumerate(entries, start=1):
        rel = entry.get("path")
        if not rel:
            continue
        path = book_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing chapter file: {path}")

        text = read_clean_text(path)
        if not text.strip():
            continue

        index = int(entry.get("index") or fallback_idx)
        title = str(entry.get("title") or f"Chapter {index}")
        chapter_id = chapter_id_from_path(index, title, rel)

        chapters.append(
            ChapterInput(
                index=index,
                id=chapter_id,
                title=title,
                text=_normalize_text(text),
                path=rel,
            )
        )

    if not chapters:
        raise ValueError("No chapter text found.")

    return chapters


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_status(out_dir: Path, stage: str, detail: Optional[str] = None) -> None:
    payload = {"stage": stage, "updated_unix": int(time.time())}
    if detail:
        payload["detail"] = detail
    atomic_write_json(out_dir / "status.json", payload)


def _load_voice_map(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Voice map not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Voice map must be a JSON object: {path}")
    chapters = data.get("chapters", {})
    if not isinstance(chapters, dict):
        chapters = {}
    return {
        "default": data.get("default"),
        "chapters": chapters,
    }


def _normalize_reading_mode(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if not cleaned or cleaned in {"all", "default"}:
        return None
    if cleaned in {"first", "once", "single", "one"}:
        return "first"
    if cleaned in {"isolated", "isolate", "boundary", "standalone"}:
        return "isolated"
    if cleaned in {"kanji", "kanji-boundary", "kanji_boundary", "kanjiboundary"}:
        return "kanji"
    return None


_READING_REGEX_PREFIXES = ("re:", "regex:")


def _split_reading_regex_prefix(text: str) -> tuple[bool, str]:
    cleaned = str(text or "").strip()
    lowered = cleaned.lower()
    for prefix in _READING_REGEX_PREFIXES:
        if lowered.startswith(prefix):
            return True, cleaned[len(prefix) :].strip()
    return False, cleaned


def _normalize_reading_override_entry(raw: object) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    reading = str(raw.get("reading") or raw.get("kana") or "").strip()
    if not reading:
        return None
    pattern = str(raw.get("pattern") or "").strip()
    base = str(raw.get("base") or "").strip()
    if pattern:
        return {"pattern": pattern, "reading": reading}
    if base and bool(raw.get("regex")):
        return {"pattern": base, "reading": reading}
    if base:
        is_regex, pattern = _split_reading_regex_prefix(base)
        if is_regex:
            if not pattern:
                return None
            return {"pattern": pattern, "reading": reading}
        entry = {"base": base, "reading": reading}
        mode = _normalize_reading_mode(raw.get("mode") or raw.get("scope"))
        if mode:
            entry["mode"] = mode
        return entry
    return None


def _parse_reading_entries(
    raw: object, *, single_kanji_mode: str | None = None
) -> List[dict[str, str]]:
    replacements: list[dict[str, str]] = []
    if isinstance(raw, dict):
        raw_replacements = raw.get("replacements") or raw.get("entries") or []
    elif isinstance(raw, list):
        raw_replacements = raw
    else:
        raw_replacements = []
    for item in raw_replacements:
        entry: dict[str, str] | None = None
        explicit_mode = False
        if isinstance(item, dict):
            explicit_mode = "mode" in item or "scope" in item
            entry = _normalize_reading_override_entry(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            entry = _normalize_reading_override_entry(
                {"base": item[0], "reading": item[1]}
            )
        if entry:
            if (
                single_kanji_mode
                and not explicit_mode
                and "mode" not in entry
                and len(str(entry.get("base") or "")) == 1
            ):
                entry["mode"] = single_kanji_mode
            replacements.append(entry)
    return replacements


def _parse_reading_overrides_text(text: str) -> List[dict[str, str]]:
    replacements: list[dict[str, str]] = []
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        for prefix in ("- ", "* ", "• "):
            if raw.startswith(prefix):
                raw = raw[len(prefix) :].strip()
                break
        if not raw:
            continue
        if "＝" in raw:
            base, reading = raw.split("＝", 1)
        elif "=" in raw:
            base, reading = raw.split("=", 1)
        elif "\t" in raw:
            base, reading = raw.split("\t", 1)
        else:
            continue
        base = base.strip()
        reading = reading.strip()
        if not base or not reading:
            continue
        entry = _normalize_reading_override_entry({"base": base, "reading": reading})
        if entry:
            replacements.append(entry)
    return replacements


def _template_reading_overrides_path() -> Path:
    return Path(__file__).parent / "templates" / READING_TEMPLATE_FILENAME


def _split_reading_overrides_data(
    data: object,
) -> tuple[List[dict[str, str]], dict[str, List[dict[str, str]]]]:
    global_entries: List[dict[str, str]] = []
    chapters_data: object = data
    if isinstance(data, dict):
        has_known_keys = any(key in data for key in ("global", "default", "*", "chapters"))
        if "global" in data:
            global_entries = _parse_reading_entries(data.get("global"))
        if "default" in data and not global_entries:
            global_entries = _parse_reading_entries(data.get("default"))
        if "*" in data and not global_entries:
            global_entries = _parse_reading_entries(data.get("*"))
        if "chapters" in data:
            chapters_data = data.get("chapters") or {}
        elif has_known_keys:
            chapters_data = {}
    if not isinstance(chapters_data, dict):
        return global_entries, {}
    overrides: dict[str, List[dict[str, str]]] = {}
    for chapter_id, entry in chapters_data.items():
        replacements = _parse_reading_entries(entry, single_kanji_mode="kanji")
        if replacements:
            overrides[str(chapter_id)] = replacements
    return global_entries, overrides


def _load_reading_overrides(
    book_dir: Path,
    include_template: bool = True,
) -> tuple[List[dict[str, str]], dict[str, List[dict[str, str]]]]:
    global_entries: List[dict[str, str]] = []
    if include_template:
        template_path = _template_reading_overrides_path()
        if template_path.exists():
            template_text = template_path.read_text(encoding="utf-8")
            global_entries = _parse_reading_overrides_text(template_text)

    path = book_dir / "reading-overrides.json"
    if not path.exists():
        return global_entries, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    book_global, overrides = _split_reading_overrides_data(data)
    if global_entries and book_global:
        global_entries = _merge_reading_overrides(global_entries, book_global)
    elif book_global:
        global_entries = book_global
    return global_entries, overrides


def _load_ruby_data(book_dir: Path) -> dict:
    path = book_dir / "reading-overrides.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    ruby_data = data.get("ruby") or {}
    return ruby_data if isinstance(ruby_data, dict) else {}


def _ruby_global_overrides(ruby_data: dict) -> List[dict[str, str]]:
    overrides: List[dict[str, str]] = []
    items = ruby_data.get("global") if isinstance(ruby_data, dict) else None
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            base = str(item.get("base") or "").strip()
            reading = str(item.get("reading") or "").strip()
            if base and reading:
                entry = {"base": base, "reading": reading}
                if len(base) == 1:
                    entry["mode"] = "kanji"
                overrides.append(entry)
    return overrides


def _merge_reading_overrides(
    global_overrides: Sequence[dict[str, str]],
    chapter_overrides: Sequence[dict[str, str]],
) -> List[dict[str, str]]:
    if not global_overrides and not chapter_overrides:
        return []
    merged: dict[str, dict[str, str]] = {}

    def key_for(entry: dict[str, str]) -> str:
        pattern = str(entry.get("pattern") or "").strip()
        if pattern:
            return f"re:{pattern}"
        base = str(entry.get("base") or "").strip()
        return f"lit:{base}" if base else ""

    def add_items(items: Sequence[dict[str, str]]) -> None:
        for item in items:
            entry = _normalize_reading_override_entry(item)
            if not entry:
                continue
            key = key_for(entry)
            if not key:
                continue
            merged[key] = entry

    add_items(global_overrides)
    add_items(chapter_overrides)
    return list(merged.values())


def apply_reading_overrides(
    text: str, overrides: Sequence[dict[str, str]]
) -> str:
    if not overrides:
        return text
    literals: list[dict[str, str]] = []
    regex_entries: list[dict[str, str]] = []
    for item in overrides:
        entry = _normalize_reading_override_entry(item)
        if not entry:
            continue
        if entry.get("pattern"):
            regex_entries.append(entry)
        else:
            literals.append(entry)
    ordered = sorted(
        literals,
        key=lambda item: len(item.get("base", "")),
        reverse=True,
    )
    out = text
    for item in ordered:
        base = item.get("base", "")
        reading = item.get("reading", "")
        mode = _normalize_reading_mode(item.get("mode"))
        if not base or not reading:
            continue
        if mode == "isolated":
            out = _replace_isolated_kanji(out, base, reading)
        elif mode == "kanji":
            out = _replace_kanji_boundary(out, base, reading)
        elif mode == "first":
            out = out.replace(base, reading, 1)
        else:
            out = out.replace(base, reading)
    for item in regex_entries:
        pattern = str(item.get("pattern") or "")
        reading = str(item.get("reading") or "")
        if not pattern or not reading:
            continue
        try:
            out = re.sub(pattern, reading, out)
        except re.error:
            continue
    return out


def _replace_isolated_kanji(text: str, base: str, reading: str) -> str:
    if not text or not base or not reading:
        return text
    if len(base) != 1:
        return text.replace(base, reading)
    ch = base
    out: List[str] = []
    length = len(text)
    for idx, cur in enumerate(text):
        if cur != ch:
            out.append(cur)
            continue
        prev = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < length else ""
        if _is_japanese_char(prev) or _is_japanese_char(next_ch):
            out.append(cur)
        else:
            out.append(reading)
    return "".join(out)


def _replace_kanji_boundary(text: str, base: str, reading: str) -> str:
    if not text or not base or not reading:
        return text
    if len(base) != 1:
        return text.replace(base, reading)
    ch = base
    out: List[str] = []
    length = len(text)
    for idx, cur in enumerate(text):
        if cur != ch:
            out.append(cur)
            continue
        prev = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < length else ""
        if _is_kanji_boundary_char(prev) or _is_kanji_boundary_char(next_ch):
            out.append(cur)
        else:
            out.append(reading)
    return "".join(out)


def apply_ruby_spans(text: str, spans: Sequence[dict]) -> str:
    if not spans:
        return text
    out = text
    ordered = sorted(
        (span for span in spans if isinstance(span, dict)),
        key=lambda item: int(item.get("start", 0)),
        reverse=True,
    )
    for span in ordered:
        try:
            start = int(span.get("start"))
            end = int(span.get("end"))
        except (TypeError, ValueError):
            continue
        reading = str(span.get("reading") or "")
        base = str(span.get("base") or "")
        if start < 0 or end < 0 or end < start:
            continue
        if not reading:
            continue
        if end > len(out):
            continue
        if base and out[start:end] != base:
            continue
        out = out[:start] + reading + out[end:]
    return out


def _normalize_voice_id(value: Optional[str], default_voice: str) -> str:
    if value is None:
        return default_voice
    cleaned = str(value).strip()
    if not cleaned:
        return default_voice
    if cleaned.lower() == "default":
        return default_voice
    return cleaned


def write_chunk_files(
    chunks: Sequence[str], chunk_dir: Path, overwrite: bool = False
) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in chunk_dir.glob("*.txt"):
            path.unlink()

    for idx, chunk in enumerate(chunks, start=1):
        path = chunk_dir / f"{idx:06d}.txt"
        if overwrite or not path.exists():
            path.write_text(chunk.rstrip() + "\n", encoding="utf-8")

    if overwrite:
        for path in chunk_dir.glob("*.txt"):
            stem = path.stem
            if stem.isdigit() and int(stem) > len(chunks):
                path.unlink()


def _prepare_manifest(
    chapters: Sequence[ChapterInput],
    out_dir: Path,
    voice: str,
    max_chars: int,
    pad_ms: int,
    rechunk: bool,
) -> Tuple[dict, List[List[str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = out_dir / "chunks"
    manifest_path = out_dir / "manifest.json"

    if rechunk and chunk_root.exists():
        shutil.rmtree(chunk_root)

    if manifest_path.exists() and not rechunk:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("chunk_mode") != "japanese":
            raise ValueError(
                "manifest.json chunk_mode differs from requested. "
                "Run with --rechunk to regenerate."
            )
        if int(manifest.get("max_chars") or 0) != int(max_chars):
            raise ValueError(
                "manifest.json max_chars differs from requested. "
                "Run with --rechunk to regenerate."
            )
        manifest_chapters = manifest.get("chapters", [])
        if not isinstance(manifest_chapters, list) or not manifest_chapters:
            raise ValueError("manifest.json contains no chapters.")
        if len(manifest_chapters) != len(chapters):
            raise ValueError(
                "manifest.json chapter count differs from input. "
                "Run with --rechunk to regenerate."
            )
        chapter_chunks: List[List[str]] = []
        for ch_manifest, chapter in zip(manifest_chapters, chapters):
            if ch_manifest.get("text_sha256") != sha256_str(chapter.text):
                raise ValueError(
                    "manifest.json text hash differs from input. "
                    "Run with --rechunk to regenerate."
                )
            chunks = ch_manifest.get("chunks", [])
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("manifest.json missing chunks. Run with --rechunk.")
            chapter_chunks.append([str(c) for c in chunks])
        manifest["voice"] = voice
        manifest["pad_ms"] = int(pad_ms)
        atomic_write_json(manifest_path, manifest)
        return manifest, chapter_chunks

    chapter_chunks: List[List[str]] = []
    manifest_chapters: List[dict] = []
    for ch in chapters:
        spans = make_chunk_spans(ch.text, max_chars=max_chars, chunk_mode="japanese")
        chunks = [ch.text[start:end] for start, end in spans]
        if not chunks:
            raise ValueError(f"No chunks generated for chapter: {ch.id}")
        chapter_chunks.append(chunks)
        manifest_chapters.append(
            {
                "index": ch.index,
                "id": ch.id,
                "title": ch.title,
                "path": ch.path,
                "text_sha256": sha256_str(ch.text),
                "chunks": chunks,
                "chunk_spans": [[start, end] for start, end in spans],
                "durations_ms": [None] * len(chunks),
            }
        )

    manifest = {
        "created_unix": int(time.time()),
        "voice": voice,
        "max_chars": int(max_chars),
        "pad_ms": int(pad_ms),
        "chunk_mode": "japanese",
        "chapters": manifest_chapters,
    }
    atomic_write_json(manifest_path, manifest)

    for ch_entry, chunks in zip(manifest["chapters"], chapter_chunks):
        chunk_dir = chunk_root / ch_entry["id"]
        write_chunk_files(chunks, chunk_dir, overwrite=True)

    return manifest, chapter_chunks


def _resolve_dtype(value: Optional[str]) -> Optional["torch.dtype"]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key or key == "auto":
        return None
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _load_model(
    model_name: str,
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> "Qwen3TTSModel":
    _require_tts()
    kwargs: Dict[str, Any] = {}
    if device_map:
        kwargs["device_map"] = device_map
    torch_dtype = _resolve_dtype(dtype)
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
    _ensure_pad_token_id(model)
    return model


def _ensure_pad_token_id(model: "Qwen3TTSModel") -> None:
    core = getattr(model, "model", None)
    if core is None:
        return
    gen_cfg = getattr(core, "generation_config", None)
    cfg = getattr(core, "config", None)
    eos = None
    if gen_cfg is not None:
        eos = getattr(gen_cfg, "eos_token_id", None)
        if getattr(gen_cfg, "pad_token_id", None) is None and eos is not None:
            gen_cfg.pad_token_id = eos
    if cfg is not None:
        eos = eos if eos is not None else getattr(cfg, "eos_token_id", None)
        if getattr(cfg, "pad_token_id", None) is None and eos is not None:
            cfg.pad_token_id = eos


def _prepare_voice_prompt(
    model: "Qwen3TTSModel", voice: VoiceConfig
) -> Tuple[Any, str]:
    if not voice.audio_path.exists():
        raise FileNotFoundError(f"Voice audio not found: {voice.audio_path}")
    ref_text = voice.ref_text
    if not ref_text and not voice.x_vector_only_mode:
        raise ValueError("Voice config is missing ref_text.")
    language = FORCED_LANGUAGE
    kwargs: Dict[str, Any] = {
        "ref_audio": str(voice.audio_path),
        "ref_text": ref_text,
        "language": language,
        "x_vector_only_mode": voice.x_vector_only_mode,
    }
    create_fn = model.create_voice_clone_prompt
    sig = inspect.signature(create_fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    prompt = create_fn(**filtered)
    return prompt, language


def _call_with_supported_kwargs(fn, kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _generate_audio(
    model: "Qwen3TTSModel",
    text: str,
    prompt: Any,
    language: Optional[str],
) -> Tuple[List[np.ndarray], int]:
    if hasattr(model, "generate_voice_clone"):
        kwargs = {
            "text": text,
            "language": language,
            "voice_clone_prompt": prompt,
            "non_streaming_mode": True,
        }
        return _call_with_supported_kwargs(model.generate_voice_clone, kwargs)
    if hasattr(model, "generate"):
        kwargs = {"text": text, "prompt": prompt, "language": language}
        return _call_with_supported_kwargs(model.generate, kwargs)
    raise AttributeError("Qwen3TTSModel has no supported generate method.")


def _is_valid_wav(path: Path) -> bool:
    try:
        with sf.SoundFile(path) as handle:
            return handle.channels == 1 and handle.frames > 0
    except Exception:
        return False


def _wav_duration_ms(path: Path) -> int:
    with sf.SoundFile(path) as handle:
        frames = handle.frames
        rate = handle.samplerate
    if rate <= 0:
        return 0
    return int(round(frames * 1000.0 / rate))


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def build_concat_file(segment_paths: List[Path], concat_path: Path, base_dir: Path) -> None:
    lines = []
    for p in segment_paths:
        rel = p.relative_to(base_dir).as_posix()
        lines.append(f"file '{rel}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chapters_ffmeta(
    chapters: Sequence[Tuple[str, int]], ffmeta_path: Path
) -> None:
    def _ffmeta_escape(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("\n", "\\\n")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#")
        )

    out = [";FFMETADATA1"]
    t = 0
    for title, d in chapters:
        start = t
        end = t + max(int(d), 1)
        out.append("")
        out.append("[CHAPTER]")
        out.append("TIMEBASE=1/1000")
        out.append(f"START={start}")
        out.append(f"END={end}")
        out.append(f"title={_ffmeta_escape(str(title))}")
        t = end
    ffmeta_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def chunk_book(
    book_dir: Path,
    out_dir: Optional[Path] = None,
    voice: str = "voice",
    max_chars: int = 220,
    pad_ms: int = 150,
    chunk_mode: str = "japanese",
    rechunk: bool = True,
) -> dict:
    _ = chunk_mode
    if out_dir is None:
        out_dir = book_dir / "tts"
    chapters = load_book_chapters(book_dir)
    manifest, _chapter_chunks = _prepare_manifest(
        chapters=chapters,
        out_dir=out_dir,
        voice=voice,
        max_chars=max_chars,
        pad_ms=pad_ms,
        rechunk=rechunk,
    )
    return manifest


def synthesize_book(
    book_dir: Path,
    voice: VoiceConfig,
    out_dir: Optional[Path] = None,
    max_chars: int = 220,
    pad_ms: int = 150,
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    only_chapter_ids: Optional[set[str]] = None,
    backend: Optional[str] = None,
    kana_normalize: bool = True,
    kana_style: str = "mixed",
) -> int:
    chapters = load_book_chapters(book_dir)
    backend_name = _select_backend(backend)
    model_name = _resolve_model_name(model_name, backend_name)
    global_overrides, reading_overrides = _load_reading_overrides(book_dir)
    ruby_data = _load_ruby_data(book_dir)
    chapter_ruby_spans: Dict[str, List[dict]] = {}
    if ruby_data:
        for chapter in chapters:
            chapter_ruby_spans[chapter.id] = _select_ruby_spans(
                chapter.id, chapter.text, ruby_data
            )
    kana_normalize = bool(kana_normalize)
    kana_style = _normalize_kana_style(kana_style)
    if kana_style == "off":
        kana_normalize = False
    if out_dir is None:
        out_dir = book_dir / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    if base_dir is None:
        base_dir = Path.cwd()
    kana_tagger = None
    if kana_normalize:
        dict_dir = _resolve_unidic_dir()
        if not (dict_dir / "dicrc").exists():
            write_status(out_dir, "unidic", "Downloading UniDic dictionary...")
        else:
            write_status(out_dir, "unidic", "Loading UniDic dictionary...")
        try:
            kana_tagger = _get_kana_tagger()
        except RuntimeError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2

    try:
        voice_map = _load_voice_map(voice_map_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        manifest, chapter_chunks = _prepare_manifest(
            chapters=chapters,
            out_dir=out_dir,
            voice=voice.name,
            max_chars=max_chars,
            pad_ms=pad_ms,
            rechunk=rechunk,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    seg_dir = out_dir / "segments"
    manifest_path = out_dir / "manifest.json"
    concat_path = out_dir / "concat.txt"
    chapters_path = out_dir / "chapters.ffmeta"
    if rechunk and seg_dir.exists():
        shutil.rmtree(seg_dir)

    default_voice = voice.name
    if voice_map:
        default_voice = _normalize_voice_id(voice_map.get("default"), default_voice)

    chapter_voice_map: Dict[str, str] = {}
    voice_overrides: Dict[str, str] = {}
    raw_overrides = voice_map.get("chapters", {}) if voice_map else {}
    for entry in manifest.get("chapters", []):
        chapter_id = entry.get("id") or "chapter"
        raw_value = raw_overrides.get(chapter_id) if isinstance(raw_overrides, dict) else None
        selected = _normalize_voice_id(raw_value, default_voice)
        chapter_voice_map[chapter_id] = selected
        entry["voice"] = selected
        if voice_map and selected != default_voice:
            voice_overrides[chapter_id] = selected
    manifest["voice"] = default_voice
    if voice_map:
        manifest["voice_overrides"] = voice_overrides
    manifest["pad_ms"] = int(pad_ms)
    manifest["kana_normalize"] = kana_normalize
    manifest["kana_style"] = kana_style
    atomic_write_json(manifest_path, manifest)

    voice_configs: Dict[str, VoiceConfig] = {}
    try:
        for voice_id in sorted(set(chapter_voice_map.values())):
            if voice_id == voice.name:
                voice_configs[voice_id] = voice
            else:
                voice_configs[voice_id] = voice_util.resolve_voice_config(
                    voice=voice_id,
                    base_dir=base_dir,
                )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    write_status(out_dir, "cloning", "Preparing voice")

    voice_prompts: Dict[str, Any] = {}
    voice_languages: Dict[str, str] = {}
    if backend_name == "torch":
        model = _load_model(
            model_name=model_name,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        for voice_id, config in voice_configs.items():
            prompt, language = _prepare_voice_prompt(model, config)
            voice_prompts[voice_id] = prompt
            voice_languages[voice_id] = language
    else:
        try:
            model = _load_mlx_model(model_name)
        except RuntimeError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2

    write_status(out_dir, "synthesizing")

    selected_ids = set(only_chapter_ids) if only_chapter_ids else None
    selected_indices = [
        idx
        for idx, entry in enumerate(manifest["chapters"])
        if not selected_ids or (entry.get("id") or "chapter") in selected_ids
    ]
    if selected_ids and not selected_indices:
        sys.stderr.write("No matching chapters found for synthesis.\n")
        return 2

    total_chunks = sum(len(chapter_chunks[idx]) for idx in selected_indices)
    if total_chunks <= 0:
        sys.stderr.write("No chunks selected for synthesis.\n")
        return 2

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    segment_paths: List[Path] = []

    with progress:
        overall_task = progress.add_task("Total", total=total_chunks)
        chapter_task = progress.add_task("Chapter", total=0)

        for ch_entry, chunks in zip(manifest["chapters"], chapter_chunks):
            chapter_id = ch_entry.get("id") or "chapter"
            chapter_title = ch_entry.get("title") or chapter_id
            chapter_total = len(chunks)
            if selected_ids and chapter_id not in selected_ids:
                continue
            progress.update(
                chapter_task,
                total=chapter_total,
                completed=0,
                description=f"{chapter_id}: {chapter_title}",
            )

            chapter_seg_dir = seg_dir / chapter_id
            voice_id = chapter_voice_map.get(chapter_id, default_voice)
            prompt = voice_prompts.get(voice_id)
            language = voice_languages.get(voice_id)
            voice_config = voice_configs.get(voice_id)
            chapter_overrides = reading_overrides.get(chapter_id, [])
            merged_overrides = _merge_reading_overrides(
                global_overrides, chapter_overrides
            )
            override_bases = {
                str(item.get("base") or "").strip()
                for item in merged_overrides
                if isinstance(item, dict) and str(item.get("base") or "").strip()
            }

            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                seg_path = chapter_seg_dir / f"{chunk_idx:06d}.wav"
                progress.update(
                    chapter_task,
                    description=f"{chapter_id}: {chapter_title} ({chunk_idx}/{chapter_total})",
                )

                if seg_path.exists() and _is_valid_wav(seg_path):
                    segment_paths.append(seg_path)
                    dms = _wav_duration_ms(seg_path)
                    if ch_entry["durations_ms"][chunk_idx - 1] != dms:
                        ch_entry["durations_ms"][chunk_idx - 1] = dms
                        atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                if ruby_data:
                    chunk_span = None
                    span_list = ch_entry.get("chunk_spans")
                    if isinstance(span_list, list) and len(span_list) >= chunk_idx:
                        chunk_span = span_list[chunk_idx - 1]
                    ruby_spans = chapter_ruby_spans.get(chapter_id, [])
                    tts_source = _apply_ruby_evidence_to_chunk(
                        chunk_text,
                        chunk_span,
                        ruby_spans,
                        ruby_data,
                        skip_bases=override_bases,
                    )
                else:
                    tts_source = chunk_text
                tts_source = apply_reading_overrides(tts_source, merged_overrides)
                tts_source = _normalize_numbers(tts_source)
                if kana_normalize and kana_tagger and _has_kanji(tts_source):
                    try:
                        tts_source = _normalize_kana_with_tagger(
                            tts_source, kana_tagger, kana_style=kana_style
                        )
                    except Exception as exc:
                        sys.stderr.write(f"Failed kana normalization: {exc}\n")
                tts_text = prepare_tts_text(tts_source, add_short_punct=True)
                if not tts_text:
                    ch_entry["durations_ms"][chunk_idx - 1] = 0
                    atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                if backend_name == "torch":
                    wavs, sample_rate = _generate_audio(
                        model=model,
                        text=tts_text,
                        prompt=prompt,
                        language=language,
                    )
                else:
                    wavs, sample_rate = _generate_audio_mlx(
                        model=model,
                        text=tts_text,
                        voice_config=voice_config,
                    )
                if not wavs:
                    sys.stderr.write(
                        f"Skipping empty audio {chapter_id} ({chunk_idx}/{chapter_total}).\n"
                    )
                    ch_entry["durations_ms"][chunk_idx - 1] = 0
                    atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                if isinstance(wavs, np.ndarray):
                    audio = wavs.flatten()
                else:
                    audio = np.concatenate([np.asarray(w).flatten() for w in wavs])

                if pad_ms > 0:
                    pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
                    if pad_samples > 0:
                        pad = np.zeros(pad_samples, dtype=audio.dtype)
                        audio = np.concatenate([audio, pad])

                _write_wav(seg_path, audio, sample_rate)
                segment_paths.append(seg_path)

                dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
                ch_entry["durations_ms"][chunk_idx - 1] = dms
                if manifest.get("sample_rate") != sample_rate:
                    manifest["sample_rate"] = sample_rate
                atomic_write_json(manifest_path, manifest)

                progress.advance(chapter_task, 1)
                progress.advance(overall_task, 1)

    build_concat_file(segment_paths, concat_path, base_dir=out_dir)

    chapter_meta: List[Tuple[str, int]] = []
    for ch_entry in manifest["chapters"]:
        title = ch_entry.get("title") or ch_entry.get("id") or "Chapter"
        durations = ch_entry.get("durations_ms", [])
        total_ms = sum(int(d or 0) for d in durations)
        chapter_meta.append((title, total_ms))

    build_chapters_ffmeta(chapter_meta, chapters_path)
    write_status(out_dir, "done")
    return 0


def synthesize_book_sample(
    book_dir: Path,
    voice: VoiceConfig,
    out_dir: Optional[Path] = None,
    max_chars: int = 220,
    pad_ms: int = 150,
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    backend: Optional[str] = None,
    kana_normalize: bool = True,
    kana_style: str = "mixed",
) -> int:
    chapters = load_book_chapters(book_dir)
    if not chapters:
        sys.stderr.write("No chapters found for sampling.\n")
        return 2
    if out_dir is None:
        out_dir = book_dir / "tts"

    sample_id = chapters[0].id
    sample_dir = out_dir / "segments" / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        chapters_meta = manifest.get("chapters")
        if isinstance(chapters_meta, list):
            for entry in chapters_meta:
                if entry.get("id") == sample_id:
                    chunks = entry.get("chunks")
                    if isinstance(chunks, list):
                        entry["durations_ms"] = [None] * len(chunks)
                    break
            atomic_write_json(manifest_path, manifest)

    return synthesize_book(
        book_dir=book_dir,
        voice=voice,
        out_dir=out_dir,
        max_chars=max_chars,
        pad_ms=pad_ms,
        rechunk=rechunk,
        voice_map_path=voice_map_path,
        base_dir=base_dir,
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
        only_chapter_ids={sample_id},
        backend=backend,
        kana_normalize=kana_normalize,
        kana_style=kana_style,
    )


def synthesize_chunk(
    out_dir: Path,
    chapter_id: str,
    chunk_index: int,
    voice: Optional[str] = None,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    backend: Optional[str] = None,
    kana_normalize: Optional[bool] = None,
    kana_style: Optional[str] = None,
) -> dict:
    if base_dir is None:
        base_dir = Path.cwd()
    backend_name = _select_backend(backend)
    model_name = _resolve_model_name(model_name, backend_name)
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chapters = manifest.get("chapters", [])
    if not isinstance(chapters, list):
        raise ValueError("manifest.json chapters missing or invalid")

    entry = None
    for item in chapters:
        if isinstance(item, dict) and item.get("id") == chapter_id:
            entry = item
            break
    if entry is None:
        raise ValueError(f"Unknown chapter_id: {chapter_id}")

    chunks = entry.get("chunks")
    if not isinstance(chunks, list):
        chunks = []
    spans = entry.get("chunk_spans")
    if not isinstance(spans, list):
        spans = []
    chunk_count = len(chunks) or len(spans)
    if chunk_count <= 0:
        chunk_dir = out_dir / "chunks" / chapter_id
        if chunk_dir.exists():
            chunk_count = len([p for p in chunk_dir.glob("*.txt") if p.stem.isdigit()])
    if chunk_count <= 0:
        raise ValueError(f"No chunks available for chapter: {chapter_id}")
    if chunk_index >= chunk_count:
        raise ValueError(f"chunk_index out of range for {chapter_id}")

    chunk_text: Optional[str] = None
    if chunks and chunk_index < len(chunks):
        chunk_text = str(chunks[chunk_index])
    if chunk_text is None:
        chunk_path = out_dir / "chunks" / chapter_id / f"{chunk_index + 1:06d}.txt"
        if chunk_path.exists():
            chunk_text = chunk_path.read_text(encoding="utf-8").rstrip("\n")
    if chunk_text is None:
        raise ValueError(f"Chunk text missing for {chapter_id} #{chunk_index + 1}")

    durations = entry.get("durations_ms")
    if not isinstance(durations, list) or len(durations) != chunk_count:
        durations = [None] * chunk_count
        entry["durations_ms"] = durations

    default_voice = voice or entry.get("voice") or manifest.get("voice") or ""
    if not default_voice:
        raise ValueError("Voice is required for synthesis.")

    if kana_normalize is None:
        kana_normalize = bool(manifest.get("kana_normalize"))
    if kana_style is None:
        kana_style = manifest.get("kana_style")
    kana_style = _normalize_kana_style(kana_style)
    if kana_style == "off":
        kana_normalize = False

    voice_id = default_voice
    if voice_map_path:
        voice_map = _load_voice_map(voice_map_path)
        if voice_map:
            default_voice = _normalize_voice_id(voice_map.get("default"), default_voice)
            raw_voice = (
                voice_map.get("chapters", {}).get(chapter_id)
                if isinstance(voice_map.get("chapters"), dict)
                else None
            )
            voice_id = _normalize_voice_id(raw_voice, default_voice)
    else:
        voice_id = _normalize_voice_id(entry.get("voice"), default_voice)

    config = voice_util.resolve_voice_config(voice=voice_id, base_dir=base_dir)
    if backend_name == "torch":
        _require_tts()
        model = _load_model(
            model_name=model_name,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        prompt, language = _prepare_voice_prompt(model, config)
    else:
        try:
            model = _load_mlx_model(model_name)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        prompt = None
        language = None

    book_dir = out_dir.parent
    override_dir = book_dir
    if (out_dir / "reading-overrides.json").exists():
        override_dir = out_dir
    ruby_data = _load_ruby_data(override_dir)
    if ruby_data:
        ruby_spans: List[dict] = []
        rel_path = entry.get("path")
        if rel_path:
            chapter_path = (book_dir / rel_path).resolve()
            if chapter_path.exists():
                chapter_text = _normalize_text(read_clean_text(chapter_path))
                ruby_spans = _select_ruby_spans(chapter_id, chapter_text, ruby_data)
        chunk_span = None
        span_list = entry.get("chunk_spans")
        if isinstance(span_list, list) and len(span_list) > chunk_index:
            chunk_span = span_list[chunk_index]
    global_overrides, reading_overrides = _load_reading_overrides(override_dir)
    chapter_overrides = reading_overrides.get(chapter_id, [])
    merged_overrides = _merge_reading_overrides(global_overrides, chapter_overrides)
    override_bases = {
        str(item.get("base") or "").strip()
        for item in merged_overrides
        if isinstance(item, dict) and str(item.get("base") or "").strip()
    }
    if ruby_data:
        chunk_text = _apply_ruby_evidence_to_chunk(
            chunk_text,
            chunk_span,
            ruby_spans,
            ruby_data,
            skip_bases=override_bases,
        )
    tts_source = apply_reading_overrides(chunk_text, merged_overrides)
    tts_source = _normalize_numbers(tts_source)
    if kana_normalize:
        try:
            tts_source = _normalize_kana_text(tts_source, kana_style=kana_style)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
    tts_text = prepare_tts_text(tts_source, add_short_punct=True)
    seg_path = out_dir / "segments" / chapter_id / f"{chunk_index + 1:06d}.wav"
    seg_path.parent.mkdir(parents=True, exist_ok=True)

    if not tts_text:
        if seg_path.exists():
            seg_path.unlink()
        durations[chunk_index] = 0
        atomic_write_json(manifest_path, manifest)
        return {
            "status": "skipped",
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "duration_ms": 0,
        }

    if backend_name == "torch":
        wavs, sample_rate = _generate_audio(
            model=model,
            text=tts_text,
            prompt=prompt,
            language=language,
        )
    else:
        wavs, sample_rate = _generate_audio_mlx(
            model=model,
            text=tts_text,
            voice_config=config,
        )
    if not wavs:
        durations[chunk_index] = 0
        atomic_write_json(manifest_path, manifest)
        return {
            "status": "skipped",
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "duration_ms": 0,
        }

    if isinstance(wavs, np.ndarray):
        audio = wavs.flatten()
    else:
        audio = np.concatenate([np.asarray(w).flatten() for w in wavs])

    pad_ms = int(manifest.get("pad_ms") or 0)
    if pad_ms > 0:
        pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
        if pad_samples > 0:
            pad = np.zeros(pad_samples, dtype=audio.dtype)
            audio = np.concatenate([audio, pad])

    _write_wav(seg_path, audio, sample_rate)
    dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
    durations[chunk_index] = dms
    if manifest.get("sample_rate") != sample_rate:
        manifest["sample_rate"] = sample_rate
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "ok",
        "chapter_id": chapter_id,
        "chunk_index": chunk_index,
        "duration_ms": dms,
    }
