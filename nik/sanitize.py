from __future__ import annotations

import json
import difflib
import re
import shutil
import time
import warnings
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import tts as tts_util
from .text import SECTION_BREAK, normalize_section_breaks, read_clean_text

RULE_KEYS = (
    "drop_chapter_title_patterns",
    "section_cutoff_patterns",
    "remove_patterns",
)
RULES_FILENAME = "sanitize-rules.json"

DEFAULT_RULES: Dict[str, List[str]] = {
    "drop_chapter_title_patterns": [
        r"^目次$",
        r"^奥付$",
        r"^著作権$",
        r"^著作権情報$",
        r"^権利表記$",
        r"^表紙$",
        r"^カバー$",
        r"^著者紹介$",
        r"^作者紹介$",
        r"^広告$",
        r"^内容紹介$",
    ],
    "section_cutoff_patterns": [
        r"^\s*参考文献\s*$",
        r"^\s*索引\s*$",
        r"^\s*注\s*$",
        r"^\s*脚注\s*$",
        r"^\s*引用文献\s*$",
    ],
    "remove_patterns": [
        # Add custom removal patterns in sanitize-rules.json as needed.
    ],
}


def template_rules_path() -> Path:
    return Path(__file__).parent / "templates" / RULES_FILENAME


def book_rules_path(book_dir: Path) -> Path:
    return book_dir / RULES_FILENAME

_WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['\u2019\-]|\u0300-\u036F)*")
_ROMAN_NUMERAL_RE = re.compile(r"^[IVXLCDM]+$")
_DIALOGUE_START_RE = re.compile(r"^[ \t\u3000]*[「『（(【［[｢“‘〈《]")
_SMALL_CAPS_MIN_WORDS = 2
_SMALL_CAPS_MAX_WORDS = 6
_SMALL_CAPS_ACRONYMS = {
    "abc",
    "bbc",
    "cbs",
    "cia",
    "cnn",
    "dna",
    "eu",
    "fbi",
    "hiv",
    "irs",
    "mlb",
    "nba",
    "nfl",
    "nato",
    "nasa",
    "nhl",
    "rna",
    "uk",
    "un",
    "usa",
}
_SMALL_CAPS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "before",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "no",
    "nor",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
    "you",
    "your",
    "yours",
}

_RUBY_MARKER_OPEN = "\ue100"
_RUBY_MARKER_CLOSE = "\ue101"
_RUBY_MARKER_TOKEN_RE = re.compile(
    rf"{re.escape(_RUBY_MARKER_OPEN)}(\d+){re.escape(_RUBY_MARKER_OPEN)}|"
    rf"{re.escape(_RUBY_MARKER_CLOSE)}(\d+){re.escape(_RUBY_MARKER_CLOSE)}"
)
_VOWELS = set("aeiouyāīūṛṝḷḹ")


@dataclass(frozen=True)
class Ruleset:
    drop_chapter_title_patterns: List[str]
    section_cutoff_patterns: List[str]
    remove_patterns: List[str]
    source_path: Optional[Path] = None
    replace_defaults: bool = False


@dataclass
class ChapterResult:
    index: int
    title: str
    raw_path: str
    clean_path: str
    dropped: bool
    drop_reason: str
    cutoff_reason: str
    raw_chars: int
    clean_chars: int


def load_rules(
    rules_path: Optional[Path] = None,
) -> Ruleset:
    rules = deepcopy(DEFAULT_RULES)
    source_path = None
    replace_defaults = False

    if rules_path is None:
        candidate = template_rules_path()
        if candidate.exists():
            rules_path = candidate

    if rules_path is not None:
        source_path = rules_path
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        replace_defaults = bool(data.get("replace_defaults", False))
        if replace_defaults:
            rules = {key: [] for key in RULE_KEYS}
        for key in RULE_KEYS:
            if key in data:
                value = data[key]
                if not isinstance(value, list):
                    raise ValueError(f"Rules key '{key}' must be a list.")
                rules[key].extend(str(item) for item in value)

    return Ruleset(
        drop_chapter_title_patterns=rules["drop_chapter_title_patterns"],
        section_cutoff_patterns=rules["section_cutoff_patterns"],
        remove_patterns=rules["remove_patterns"],
        source_path=source_path,
        replace_defaults=replace_defaults,
    )


def format_title_chapter(metadata: dict) -> str:
    raw_title = str(metadata.get("title") or "").strip()
    title = raw_title
    subtitle = ""
    if ":" in raw_title:
        title, subtitle = [part.strip() for part in raw_title.split(":", 1)]
    year = str(metadata.get("year") or "").strip()
    authors = metadata.get("authors") or []

    headline = title or ""
    if subtitle:
        headline = f"{title}: {subtitle}" if title else subtitle
    author_line = ", ".join(a.strip() for a in authors if str(a).strip())
    if author_line:
        author_line = f"by {author_line}"

    lines: List[str] = []
    for block in (headline, year, author_line):
        if not block:
            continue
        if lines:
            lines.append("")
        lines.append(block)

    return "\n".join(lines).strip()


def compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    compiled: List[re.Pattern] = []
    for pattern in patterns:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="Possible nested set.*",
            )
            compiled.append(
                re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)
            )
    return compiled


def _has_lowercase(word: str) -> bool:
    return any(ch.islower() for ch in word)


def _is_all_caps_word(word: str) -> bool:
    has_letter = False
    for ch in word:
        if ch.isalpha():
            has_letter = True
            if not ch.isupper():
                return False
    return has_letter


def _titlecase_token(word: str) -> str:
    parts = re.split(r"([\-'\u2019])", word)
    out: List[str] = []
    for part in parts:
        if part in {"-", "'", "\u2019"}:
            out.append(part)
            continue
        if not part:
            out.append(part)
            continue
        out.append(part[0].upper() + part[1:].lower())
    return "".join(out)


def _variant_score(word: str) -> int:
    if word and word[0].isupper() and _has_lowercase(word[1:]):
        return 3
    if word.islower():
        return 2
    return 1


def _build_case_map(text: str, extra_words: Iterable[str]) -> Dict[str, str]:
    variants: Dict[str, Dict[str, int]] = {}
    for source in [text, *extra_words]:
        for match in _WORD_RE.finditer(source):
            word = match.group(0)
            if not _has_lowercase(word):
                continue
            key = word.lower()
            bucket = variants.setdefault(key, {})
            bucket[word] = bucket.get(word, 0) + 1
    case_map: Dict[str, str] = {}
    for key, bucket in variants.items():
        best = max(bucket.items(), key=lambda item: (item[1], _variant_score(item[0])))
        case_map[key] = best[0]
    return case_map


def _capitalize_mapped(word: str, is_first: bool) -> str:
    if not is_first:
        return word
    if word and word[0].isupper():
        return word
    if word.islower():
        return word.capitalize()
    return word


def _should_preserve_caps_word(word: str, lower: str) -> bool:
    if lower in _SMALL_CAPS_ACRONYMS:
        return True
    if _ROMAN_NUMERAL_RE.fullmatch(word):
        return True
    if len(word) <= 3:
        return True
    if len(word) <= 4 and not any(ch in _VOWELS for ch in lower):
        return True
    return False


def _normalize_small_caps_word(
    word: str, case_map: Dict[str, str], is_first: bool
) -> str:
    lower = word.lower()
    mapped = case_map.get(lower)
    if mapped:
        return _capitalize_mapped(mapped, is_first=is_first)
    if lower in _SMALL_CAPS_STOPWORDS:
        return lower.capitalize() if is_first else lower
    if _should_preserve_caps_word(word, lower):
        return word
    return _titlecase_token(word) if is_first else lower


def _normalize_small_caps_paragraph(
    paragraph: str, case_map: Dict[str, str]
) -> str:
    if not re.search(r"[a-z]", paragraph):
        return paragraph
    matches = list(_WORD_RE.finditer(paragraph))
    if not matches:
        return paragraph

    run: List[re.Match] = []
    for match in matches:
        word = match.group(0)
        if not _is_all_caps_word(word):
            break
        run.append(match)
        if len(run) >= _SMALL_CAPS_MAX_WORDS:
            break

    if len(run) < _SMALL_CAPS_MIN_WORDS:
        return paragraph

    next_index = len(run)
    if next_index >= len(matches):
        return paragraph
    next_word = matches[next_index].group(0)
    if not _has_lowercase(next_word):
        return paragraph

    has_signal = False
    for match in run:
        lower = match.group(0).lower()
        if lower in case_map or lower in _SMALL_CAPS_STOPWORDS:
            has_signal = True
            break
    if not has_signal:
        return paragraph

    pieces: List[str] = []
    last = 0
    for idx, match in enumerate(run):
        pieces.append(paragraph[last:match.start()])
        replacement = _normalize_small_caps_word(
            match.group(0), case_map=case_map, is_first=idx == 0
        )
        pieces.append(replacement)
        last = match.end()
    pieces.append(paragraph[last:])
    return "".join(pieces)


def normalize_small_caps(
    text: str, extra_words: Optional[Iterable[str]] = None
) -> str:
    if not text:
        return text
    case_map = _build_case_map(text, extra_words or [])
    blocks = re.split(r"\n\s*\n", text)
    normalized: List[str] = []
    for block in blocks:
        normalized.append(_normalize_small_caps_paragraph(block, case_map))
    return "\n\n".join(normalized)


def _is_all_caps_block(text: str) -> bool:
    has_letter = False
    for ch in text:
        if ch.islower():
            return False
        if ch.isupper():
            has_letter = True
    return has_letter


def _normalize_all_caps_title_word(
    word: str, case_map: Dict[str, str], is_first: bool
) -> str:
    lower = word.lower()
    mapped = case_map.get(lower)
    if mapped:
        return _capitalize_mapped(mapped, is_first=is_first)
    if lower in _SMALL_CAPS_STOPWORDS:
        return _titlecase_token(word) if is_first else lower
    if _should_preserve_caps_word(word, lower):
        return word
    return _titlecase_token(word)


def _normalize_all_caps_sentence_word(
    word: str, case_map: Dict[str, str], is_first: bool
) -> str:
    lower = word.lower()
    mapped = case_map.get(lower)
    if mapped:
        return _capitalize_mapped(mapped, is_first=is_first)
    if lower in _SMALL_CAPS_STOPWORDS:
        return _titlecase_token(word) if is_first else lower
    if _should_preserve_caps_word(word, lower):
        return word
    if is_first:
        return _titlecase_token(word)
    return lower


def _normalize_heading_line_key(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "").strip())
    return collapsed.casefold()


def _build_heading_line_keys(
    chapter_title: str,
    heading_lines: Optional[Iterable[str]],
) -> set[str]:
    _ = chapter_title
    keys: set[str] = set()
    if not heading_lines:
        return keys
    for line in heading_lines:
        key = _normalize_heading_line_key(str(line))
        if key:
            keys.add(key)
    return keys


def _normalize_all_caps_title_line(
    line: str, matches: List[re.Match], case_map: Dict[str, str]
) -> str:
    pieces: List[str] = []
    last = 0
    for idx, match in enumerate(matches):
        pieces.append(line[last:match.start()])
        replacement = _normalize_all_caps_title_word(
            match.group(0), case_map=case_map, is_first=idx == 0
        )
        pieces.append(replacement)
        last = match.end()
    pieces.append(line[last:])
    return "".join(pieces)


def _normalize_all_caps_sentence_line(
    line: str, matches: List[re.Match], case_map: Dict[str, str]
) -> str:
    pieces: List[str] = []
    last = 0
    sentence_start = True
    for idx, match in enumerate(matches):
        pieces.append(line[last:match.start()])
        replacement = _normalize_all_caps_sentence_word(
            match.group(0), case_map=case_map, is_first=sentence_start
        )
        pieces.append(replacement)
        last = match.end()
        if idx + 1 < len(matches):
            gap = line[last : matches[idx + 1].start()]
            sentence_start = bool(re.search(r"[.!?]", gap))
    pieces.append(line[last:])
    return "".join(pieces)


def _normalize_all_caps_line(
    line: str, case_map: Dict[str, str], heading_line_keys: set[str]
) -> str:
    if not _is_all_caps_block(line):
        return line
    matches = list(_WORD_RE.finditer(line))
    if not matches:
        return line
    key = _normalize_heading_line_key(line)
    if key and key in heading_line_keys:
        return _normalize_all_caps_title_line(line, matches, case_map)
    return _normalize_all_caps_sentence_line(line, matches, case_map)


def normalize_all_caps(
    text: str,
    extra_words: Optional[Iterable[str]] = None,
    chapter_title: str = "",
    heading_lines: Optional[Iterable[str]] = None,
) -> str:
    if not text:
        return text
    case_map = _build_case_map(text, extra_words or [])
    heading_line_keys = _build_heading_line_keys(chapter_title, heading_lines)
    blocks = re.split(r"\n\s*\n", text)
    normalized: List[str] = []
    for block in blocks:
        if "\n" in block:
            lines = block.split("\n")
            normalized_lines = [
                _normalize_all_caps_line(
                    line, case_map=case_map, heading_line_keys=heading_line_keys
                )
                for line in lines
            ]
            normalized.append("\n".join(normalized_lines))
        else:
            normalized.append(
                _normalize_all_caps_line(
                    block, case_map=case_map, heading_line_keys=heading_line_keys
                )
            )
    return "\n\n".join(normalized)


def _case_context_words(metadata: dict, chapter_title: str) -> List[str]:
    extra: List[str] = []
    title = str(metadata.get("title") or "").strip() if isinstance(metadata, dict) else ""
    if title:
        extra.append(title)
    authors = metadata.get("authors") if isinstance(metadata, dict) else []
    if isinstance(authors, list):
        for author in authors:
            name = str(author).strip()
            if name:
                extra.append(name)
    if chapter_title:
        extra.append(chapter_title)
    return extra


def _chapter_heading_lines(entry: dict, chapter_title: str) -> List[str]:
    _ = chapter_title
    items: List[str] = []
    raw = entry.get("headings") if isinstance(entry, dict) else None
    if isinstance(raw, list):
        for value in raw:
            heading = str(value).strip()
            if heading:
                items.append(heading)
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        key = _normalize_heading_line_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _collapse_dialogue_continuation_breaks(text: str) -> str:
    if "\n\n" not in text:
        return text
    blocks = text.split("\n\n")
    if len(blocks) <= 1:
        return text

    out: List[str] = [blocks[0]]
    for block in blocks[1:]:
        prev = out[-1].rstrip()
        if (
            prev
            and prev[-1] in {"、", "，", ",", ":", "："}
            and _DIALOGUE_START_RE.match(block or "")
        ):
            out[-1] = out[-1].rstrip() + "\n" + block.lstrip()
            continue
        out.append(block)
    return "\n\n".join(out)


def normalize_text(text: str) -> str:
    text = (
        text.replace("\u02bc", "'")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u201e", '"')
        .replace("\u201f", '"')
        .replace("\u00ab", '"')
        .replace("\u00bb", '"')
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    if "\n\n\n" in text:
        text = re.sub(r"\n{3,}", f"\n\n{SECTION_BREAK}\n\n", text)
    text = normalize_section_breaks(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r'(^|[\s“("])\[(?P<l>[A-Za-z])\](?=[a-z])', r"\1\g<l>", text)
    text = re.sub(r"-\n(?=\w)", "-", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\.[ \t]*\.[ \t]*\.", "...", text)

    text = _collapse_dialogue_continuation_breaks(text)

    return text.strip()


def apply_section_cutoff(
    text: str, patterns: List[re.Pattern]
) -> Tuple[str, str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return text[: match.start()].rstrip(), pattern.pattern
    return text, ""


def apply_remove_patterns(
    text: str, patterns: List[re.Pattern]
) -> Tuple[str, Dict[str, int]]:
    stats: Dict[str, int] = {}
    for pattern in patterns:
        text, count = pattern.subn("", text)
        stats[pattern.pattern] = stats.get(pattern.pattern, 0) + count
    return text, stats


def _sanitize_text_for_ruby_tracking(
    text: str,
    cutoff_patterns: List[re.Pattern],
    remove_patterns: List[re.Pattern],
    case_words: Iterable[str],
    chapter_title: str,
    heading_lines: Optional[Iterable[str]],
) -> str:
    normalized = normalize_text(text)
    cutoff_text, _ = apply_section_cutoff(normalized, cutoff_patterns)
    cleaned, _ = apply_remove_patterns(cutoff_text, remove_patterns)
    cleaned = normalize_text(cleaned)
    cleaned = normalize_small_caps(cleaned, extra_words=case_words)
    cleaned = normalize_all_caps(
        cleaned,
        extra_words=case_words,
        chapter_title=chapter_title,
        heading_lines=heading_lines,
    )
    return cleaned


def _ruby_marker_token(span_id: int, is_start: bool) -> str:
    marker = _RUBY_MARKER_OPEN if is_start else _RUBY_MARKER_CLOSE
    return f"{marker}{span_id}{marker}"


def _prepare_tracked_ruby_spans(
    text: str, raw_spans: Sequence[dict]
) -> List[dict]:
    prepared: List[dict] = []
    for item in raw_spans:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        base = str(item.get("base") or "")
        reading = str(item.get("reading") or "")
        if not reading:
            continue
        if base and text[start:end] != base:
            continue
        if not base:
            base = text[start:end]
        prepared.append(
            {
                "start": start,
                "end": end,
                "base": base,
                "reading": reading,
            }
        )
    prepared.sort(key=lambda span: (int(span["start"]), int(span["end"])))

    tracked: List[dict] = []
    cursor = 0
    for span in prepared:
        start = int(span["start"])
        end = int(span["end"])
        if start < cursor:
            continue
        tracked.append(
            {
                "id": len(tracked),
                "start": start,
                "end": end,
                "base": str(span["base"]),
                "reading": str(span["reading"]),
            }
        )
        cursor = end
    return tracked


def _inject_ruby_markers(
    text: str, spans: Sequence[dict], use_reading: bool
) -> str:
    if not spans:
        return text
    parts: List[str] = []
    cursor = 0
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        span_id = int(span["id"])
        if start < cursor or end < start or end > len(text):
            continue
        parts.append(text[cursor:start])
        parts.append(_ruby_marker_token(span_id, True))
        if use_reading:
            parts.append(str(span["reading"]))
        else:
            parts.append(str(span["base"]))
        parts.append(_ruby_marker_token(span_id, False))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _extract_ruby_marker_ranges(
    text: str,
) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    plain_parts: List[str] = []
    ranges: Dict[int, Tuple[int, int]] = {}
    open_positions: Dict[int, int] = {}
    last = 0
    plain_pos = 0

    for match in _RUBY_MARKER_TOKEN_RE.finditer(text):
        segment = text[last : match.start()]
        plain_parts.append(segment)
        plain_pos += len(segment)
        start_id = match.group(1)
        end_id = match.group(2)
        if start_id is not None:
            span_id = int(start_id)
            if span_id not in open_positions:
                open_positions[span_id] = plain_pos
        elif end_id is not None:
            span_id = int(end_id)
            start_pos = open_positions.pop(span_id, None)
            if start_pos is not None and plain_pos >= start_pos:
                ranges[span_id] = (start_pos, plain_pos)
        last = match.end()

    plain_parts.append(text[last:])
    plain_text = "".join(plain_parts)
    return plain_text, ranges


def _collect_clean_ruby_spans_from_raw(
    raw_text_original: str,
    raw_spans: Sequence[dict],
    cleaned_text: str,
    cutoff_patterns: List[re.Pattern],
    remove_patterns: List[re.Pattern],
    case_words: Iterable[str],
    chapter_title: str,
    heading_lines: Optional[Iterable[str]],
) -> Optional[List[dict]]:
    if not raw_spans:
        return []
    if _RUBY_MARKER_OPEN in raw_text_original or _RUBY_MARKER_CLOSE in raw_text_original:
        return None

    tracked = _prepare_tracked_ruby_spans(raw_text_original, raw_spans)
    if not tracked:
        return []
    for span in tracked:
        base = str(span["base"])
        reading = str(span["reading"])
        if _RUBY_MARKER_OPEN in base or _RUBY_MARKER_CLOSE in base:
            return None
        if _RUBY_MARKER_OPEN in reading or _RUBY_MARKER_CLOSE in reading:
            return None

    base_marked = _inject_ruby_markers(raw_text_original, tracked, use_reading=False)
    ruby_marked = _inject_ruby_markers(raw_text_original, tracked, use_reading=True)

    base_cleaned_marked = _sanitize_text_for_ruby_tracking(
        base_marked,
        cutoff_patterns,
        remove_patterns,
        case_words,
        chapter_title,
        heading_lines,
    )
    ruby_cleaned_marked = _sanitize_text_for_ruby_tracking(
        ruby_marked,
        cutoff_patterns,
        remove_patterns,
        case_words,
        chapter_title,
        heading_lines,
    )
    plain_base, base_ranges = _extract_ruby_marker_ranges(base_cleaned_marked)
    plain_ruby, ruby_ranges = _extract_ruby_marker_ranges(ruby_cleaned_marked)
    if plain_base != cleaned_text:
        if len(plain_base) != len(cleaned_text):
            return None
        if plain_base.lower() != cleaned_text.lower():
            return None
    if len(plain_base) != len(cleaned_text):
        return None

    spans: List[dict] = []
    for span in tracked:
        span_id = int(span["id"])
        base_range = base_ranges.get(span_id)
        ruby_range = ruby_ranges.get(span_id)
        if base_range is None or ruby_range is None:
            continue
        base_start, base_end = base_range
        ruby_start, ruby_end = ruby_range
        if base_end <= base_start or ruby_end <= ruby_start:
            continue
        base = plain_base[base_start:base_end]
        reading = plain_ruby[ruby_start:ruby_end]
        if tts_util._is_suspicious_ruby_span(base, reading):
            continue
        spans.append(
            {
                "start": base_start,
                "end": base_end,
                "base": base,
                "reading": reading,
            }
        )
    spans.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return spans


def _split_ruby_span_on_kana(
    start: int,
    base: str,
    reading: str,
) -> Optional[List[dict]]:
    if not base or not reading:
        return None
    if not tts_util._has_kanji(base):
        return None
    if not any(tts_util._is_kana_char(ch) for ch in base):
        return None

    segments: List[tuple[bool, str, int, int]] = []
    idx = 0
    length = len(base)
    while idx < length:
        ch = base[idx]
        is_kana = tts_util._is_kana_char(ch)
        start_idx = idx
        idx += 1
        while idx < length and tts_util._is_kana_char(base[idx]) == is_kana:
            idx += 1
        segments.append((is_kana, base[start_idx:idx], start_idx, idx))

    reading_nfkc = unicodedata.normalize("NFKC", reading)
    reading_hira = tts_util._katakana_to_hiragana(reading_nfkc)
    if not reading_hira:
        return []

    def walk(
        segment_idx: int,
        reading_pos: int,
        pending: Optional[tuple[str, int, int]],
    ) -> Optional[List[dict]]:
        if segment_idx >= len(segments):
            if pending is None:
                return [] if reading_pos == len(reading_hira) else None
            segment_text, segment_start, segment_end = pending
            segment_reading = reading_hira[reading_pos:]
            if not segment_reading:
                return None
            if not tts_util._has_kanji(segment_text):
                return [] if reading_pos <= len(reading_hira) else None
            return [
                {
                    "start": start + segment_start,
                    "end": start + segment_end,
                    "base": segment_text,
                    "reading": segment_reading,
                }
            ]

        is_kana, segment_text, segment_start, segment_end = segments[segment_idx]
        if not is_kana:
            if pending is None:
                merged = (segment_text, segment_start, segment_end)
            else:
                prev_text, prev_start, _prev_end = pending
                merged = (prev_text + segment_text, prev_start, segment_end)
            return walk(segment_idx + 1, reading_pos, merged)

        target = tts_util._katakana_to_hiragana(segment_text)
        positions: List[int] = []
        search_pos = reading_hira.find(target, reading_pos)
        while search_pos != -1:
            positions.append(search_pos)
            search_pos = reading_hira.find(target, search_pos + 1)
        if not positions:
            return None

        for found in reversed(positions):
            prefix: List[dict] = []
            if pending is not None:
                pending_text, pending_start, pending_end = pending
                pending_reading = reading_hira[reading_pos:found]
                if not pending_reading:
                    continue
                if tts_util._has_kanji(pending_text):
                    prefix.append(
                        {
                            "start": start + pending_start,
                            "end": start + pending_end,
                            "base": pending_text,
                            "reading": pending_reading,
                        }
                    )
            tail = walk(segment_idx + 1, found + len(target), None)
            if tail is not None:
                return prefix + tail
        return None

    spans = walk(0, 0, None)
    return spans if spans is not None else []


def _diff_ruby_spans(plain: str, ruby: str) -> List[dict]:
    spans: List[dict] = []
    matcher = difflib.SequenceMatcher(None, plain, ruby)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        base = plain[i1:i2]
        reading = ruby[j1:j2]
        if not base or not reading:
            continue
        split_spans = _split_ruby_span_on_kana(i1, base, reading)
        if split_spans is not None:
            if split_spans:
                for span in split_spans:
                    span_base = str(span.get("base") or "")
                    span_reading = str(span.get("reading") or "")
                    if tts_util._is_suspicious_ruby_span(
                        span_base, span_reading
                    ):
                        continue
                    spans.append(span)
            continue
        if tts_util._is_suspicious_ruby_span(base, reading):
            continue
        spans.append({"start": i1, "end": i2, "base": base, "reading": reading})
    return spans


def should_drop_title(
    title: str, patterns: List[re.Pattern]
) -> str:
    for pattern in patterns:
        if pattern.search(title):
            return pattern.pattern
    return ""


def sanitize_book(
    book_dir: Path,
    rules_path: Optional[Path] = None,
    overwrite: bool = False,
) -> int:
    toc_path = book_dir / "toc.json"
    raw_dir = book_dir / "raw" / "chapters"
    clean_dir = book_dir / "clean" / "chapters"
    report_path = book_dir / "clean" / "report.json"
    overrides_path = book_dir / "reading-overrides.json"

    if not toc_path.exists():
        raise FileNotFoundError(f"Missing toc.json at {toc_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw chapters at {raw_dir}")

    if clean_dir.exists():
        existing = [p for p in clean_dir.iterdir() if p.is_file()]
        if existing and not overwrite:
            raise FileExistsError(
                "Clean chapters already exist. Use --overwrite to regenerate."
            )
        if overwrite:
            for path in existing:
                path.unlink()

    if rules_path is None:
        candidate = book_rules_path(book_dir)
        if candidate.exists():
            rules_path = candidate
    rules = load_rules(rules_path)
    drop_patterns = compile_patterns(rules.drop_chapter_title_patterns)
    cutoff_patterns = compile_patterns(rules.section_cutoff_patterns)
    remove_patterns = compile_patterns(rules.remove_patterns)

    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        raise ValueError("Invalid toc.json: chapters must be a list.")

    overrides_data: dict = {}
    ruby_data: dict = {}
    ruby_chapters: dict = {}
    ruby_updated = False
    if overrides_path.exists():
        try:
            overrides_data = json.loads(overrides_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            overrides_data = {}
        if isinstance(overrides_data, dict):
            ruby_data = overrides_data.get("ruby") or {}
            if isinstance(ruby_data, dict):
                ruby_chapters = ruby_data.get("chapters") or {}
                if not isinstance(ruby_chapters, dict):
                    ruby_chapters = {}

    clean_dir.mkdir(parents=True, exist_ok=True)
    report_entries: List[ChapterResult] = []
    pattern_stats: Dict[str, int] = {}
    dropped = 0
    clean_entries: List[dict] = []


    for entry in chapters:
        title = str(entry.get("title") or "").strip()
        raw_rel = entry.get("path")
        if not raw_rel:
            continue
        raw_path = book_dir / Path(raw_rel)
        clean_path = clean_dir / Path(raw_rel).name

        drop_reason = should_drop_title(title, drop_patterns)
        if drop_reason:
            if ruby_chapters:
                chapter_id = tts_util.chapter_id_from_path(
                    int(entry.get("index", len(report_entries) + 1)),
                    title,
                    raw_rel,
                )
                ruby_entry = ruby_chapters.get(chapter_id)
                if isinstance(ruby_entry, dict):
                    ruby_entry.pop("clean_spans", None)
                    ruby_entry.pop("clean_sha256", None)
                    ruby_updated = True
            dropped += 1
            report_entries.append(
                ChapterResult(
                    index=int(entry.get("index", len(report_entries) + 1)),
                    title=title,
                    raw_path=str(raw_path),
                    clean_path=str(clean_path),
                    dropped=True,
                    drop_reason=drop_reason,
                    cutoff_reason="",
                    raw_chars=0,
                    clean_chars=0,
                )
            )
            continue

        if not raw_path.exists():
            if ruby_chapters:
                chapter_id = tts_util.chapter_id_from_path(
                    int(entry.get("index", len(report_entries) + 1)),
                    title,
                    raw_rel,
                )
                ruby_entry = ruby_chapters.get(chapter_id)
                if isinstance(ruby_entry, dict):
                    ruby_entry.pop("clean_spans", None)
                    ruby_entry.pop("clean_sha256", None)
                    ruby_updated = True
            report_entries.append(
                ChapterResult(
                    index=int(entry.get("index", len(report_entries) + 1)),
                    title=title,
                    raw_path=str(raw_path),
                    clean_path=str(clean_path),
                    dropped=True,
                    drop_reason="missing_raw",
                    cutoff_reason="",
                    raw_chars=0,
                    clean_chars=0,
                )
            )
            dropped += 1
            continue

        raw_text_original = raw_path.read_text(encoding="utf-8")
        raw_text = normalize_text(raw_text_original)
        cutoff_text, cutoff_reason = apply_section_cutoff(
            raw_text, cutoff_patterns
        )
        cleaned, stats = apply_remove_patterns(cutoff_text, remove_patterns)
        cleaned = normalize_text(cleaned)
        case_words = _case_context_words(metadata, title)
        heading_lines = _chapter_heading_lines(entry, title)
        cleaned = normalize_small_caps(cleaned, extra_words=case_words)
        cleaned = normalize_all_caps(
            cleaned,
            extra_words=case_words,
            chapter_title=title,
            heading_lines=heading_lines,
        )

        ruby_entry_to_hash = None
        if ruby_chapters:
            chapter_id = tts_util.chapter_id_from_path(
                int(entry.get("index", len(report_entries) + 1)),
                title,
                raw_rel,
            )
            ruby_entry = ruby_chapters.get(chapter_id)
            if isinstance(ruby_entry, dict):
                raw_spans = ruby_entry.get("raw_spans")
                if isinstance(raw_spans, list) and raw_spans:
                    clean_spans = _collect_clean_ruby_spans_from_raw(
                        raw_text_original=raw_text_original,
                        raw_spans=raw_spans,
                        cleaned_text=cleaned,
                        cutoff_patterns=cutoff_patterns,
                        remove_patterns=remove_patterns,
                        case_words=case_words,
                        chapter_title=title,
                        heading_lines=heading_lines,
                    )
                    if clean_spans is None:
                        ruby_text = tts_util.apply_ruby_spans(
                            raw_text_original, raw_spans
                        )
                        ruby_text = normalize_text(ruby_text)
                        ruby_cutoff, _ruby_cutoff_reason = apply_section_cutoff(
                            ruby_text, cutoff_patterns
                        )
                        ruby_cleaned, _ruby_stats = apply_remove_patterns(
                            ruby_cutoff, remove_patterns
                        )
                        ruby_cleaned = normalize_text(ruby_cleaned)
                        ruby_cleaned = normalize_small_caps(
                            ruby_cleaned, extra_words=case_words
                        )
                        ruby_cleaned = normalize_all_caps(
                            ruby_cleaned,
                            extra_words=case_words,
                            chapter_title=title,
                            heading_lines=heading_lines,
                        )
                        clean_spans = _diff_ruby_spans(cleaned, ruby_cleaned)
                    ruby_entry["clean_spans"] = clean_spans
                    ruby_entry_to_hash = ruby_entry
                    ruby_updated = True

        clean_path.write_text(cleaned + "\n", encoding="utf-8")
        if ruby_entry_to_hash is not None:
            clean_for_hash = tts_util._normalize_text(read_clean_text(clean_path))
            ruby_entry_to_hash["clean_sha256"] = tts_util.sha256_str(clean_for_hash)

        for pattern, count in stats.items():
            if count:
                pattern_stats[pattern] = pattern_stats.get(pattern, 0) + count

        report_entries.append(
            ChapterResult(
                index=int(entry.get("index", len(report_entries) + 1)),
                title=title,
                raw_path=str(raw_path),
                clean_path=str(clean_path),
                dropped=False,
                drop_reason="",
                cutoff_reason=cutoff_reason,
                raw_chars=len(raw_text),
                clean_chars=len(cleaned),
            )
        )
        clean_entries.append(
            {
                "index": len(clean_entries) + 1,
                "title": title,
                "path": clean_path.relative_to(book_dir).as_posix(),
                "source_index": entry.get("index", None),
                "headings": heading_lines,
                "kind": "chapter",
            }
        )

    report = {
        "created_unix": int(time.time()),
        "book_dir": str(book_dir),
        "rules_source": str(rules.source_path) if rules.source_path else "",
        "replace_defaults": rules.replace_defaults,
        "rules": {
            "drop_chapter_title_patterns": rules.drop_chapter_title_patterns,
            "section_cutoff_patterns": rules.section_cutoff_patterns,
            "remove_patterns": rules.remove_patterns,
        },
        "stats": {
            "total_chapters": len(report_entries),
            "dropped_chapters": dropped,
            "removed_by_pattern": pattern_stats,
        },
        "chapters": [
            {
                "index": entry.index,
                "title": entry.title,
                "raw_path": entry.raw_path,
                "clean_path": entry.clean_path,
                "dropped": entry.dropped,
                "drop_reason": entry.drop_reason,
                "cutoff_reason": entry.cutoff_reason,
                "raw_chars": entry.raw_chars,
                "clean_chars": entry.clean_chars,
            }
            for entry in report_entries
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    clean_toc = {
        "created_unix": int(time.time()),
        "source_epub": toc.get("source_epub", ""),
        "metadata": metadata,
        "chapters": clean_entries,
    }
    clean_toc_path = book_dir / "clean" / "toc.json"
    clean_toc_path.write_text(
        json.dumps(clean_toc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if ruby_updated and isinstance(overrides_data, dict):
        overrides_data["ruby"] = ruby_data
        overrides_path.write_text(
            json.dumps(overrides_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return len(clean_entries)


def refresh_chunks(
    book_dir: Path,
    max_chars: int = 220,
    pad_ms: int = 350,
    chunk_mode: str = "japanese",
) -> bool:
    tts_dir = book_dir / "tts"
    tts_cleared = tts_dir.exists()
    if tts_dir.exists():
        shutil.rmtree(tts_dir)
    tts_util.chunk_book(
        book_dir=book_dir,
        out_dir=tts_dir,
        max_chars=max_chars,
        pad_ms=pad_ms,
        chunk_mode=chunk_mode,
        rechunk=True,
    )
    return tts_cleared


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _reindex_clean_entries(entries: List[dict]) -> None:
    for idx, entry in enumerate(entries, start=1):
        entry["index"] = idx


def _match_report_entry(entry: dict, rel_path: str, title: str) -> bool:
    clean_path = str(entry.get("clean_path") or "")
    if clean_path.endswith(rel_path):
        return True
    if Path(clean_path).name == Path(rel_path).name:
        return True
    return (entry.get("title") or "") == title


def _update_report(
    book_dir: Path,
    rel_path: str,
    title: str,
    dropped: bool,
    drop_reason: str = "",
    cutoff_reason: str = "",
    raw_chars: Optional[int] = None,
    clean_chars: Optional[int] = None,
) -> None:
    report_path = book_dir / "clean" / "report.json"
    if not report_path.exists():
        return
    report = _load_json(report_path)
    chapters = report.get("chapters", [])
    if not isinstance(chapters, list):
        return
    target = None
    for entry in chapters:
        if isinstance(entry, dict) and _match_report_entry(entry, rel_path, title):
            target = entry
            break
    if not target:
        return

    target["dropped"] = dropped
    target["drop_reason"] = drop_reason if dropped else ""
    if cutoff_reason:
        target["cutoff_reason"] = cutoff_reason
    if raw_chars is not None:
        target["raw_chars"] = raw_chars
    if clean_chars is not None:
        target["clean_chars"] = clean_chars
    target["clean_path"] = str((book_dir / rel_path).resolve())

    stats = report.get("stats", {}) if isinstance(report.get("stats"), dict) else {}
    stats["dropped_chapters"] = sum(
        1 for entry in chapters if isinstance(entry, dict) and entry.get("dropped")
    )
    stats["total_chapters"] = len(chapters)
    report["stats"] = stats
    _write_json(report_path, report)


def _drop_tts_chapter(book_dir: Path, chapter_id: str) -> None:
    tts_dir = book_dir / "tts"
    if not tts_dir.exists():
        return
    for dir_name in ("chunks", "segments"):
        target = tts_dir / dir_name / chapter_id
        if target.exists():
            shutil.rmtree(target)
    manifest_path = tts_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = _load_json(manifest_path)
    chapters = manifest.get("chapters", [])
    if not isinstance(chapters, list):
        return
    filtered = [
        entry
        for entry in chapters
        if isinstance(entry, dict) and (entry.get("id") or "") != chapter_id
    ]
    if len(filtered) == len(chapters):
        return
    manifest["chapters"] = filtered
    tts_util.atomic_write_json(manifest_path, manifest)


def drop_chapter(
    book_dir: Path,
    title: str,
    chapter_index: Optional[int] = None,
) -> bool:
    clean_toc_path = book_dir / "clean" / "toc.json"
    if not clean_toc_path.exists():
        raise FileNotFoundError(f"Missing clean/toc.json at {clean_toc_path}")
    clean_toc = _load_json(clean_toc_path)
    entries = clean_toc.get("chapters", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("clean/toc.json contains no chapters.")

    target = None
    if chapter_index is not None:
        for entry in entries:
            if entry.get("index") == chapter_index:
                target = entry
                break
    if target is None:
        for entry in entries:
            if (entry.get("title") or "") == title:
                target = entry
                break
    if not target:
        return False
    if target.get("kind") == "title":
        raise ValueError("Title chapter cannot be dropped.")

    rel_path = target.get("path") or ""
    chapter_id = tts_util.chapter_id_from_path(
        int(target.get("index") or 0),
        str(target.get("title") or ""),
        rel_path or None,
    )

    if rel_path:
        clean_path = book_dir / rel_path
        if clean_path.exists():
            clean_path.unlink()

    entries = [entry for entry in entries if entry is not target]
    _reindex_clean_entries(entries)
    clean_toc["chapters"] = entries
    _write_json(clean_toc_path, clean_toc)

    if rel_path:
        _update_report(
            book_dir,
            rel_path=rel_path,
            title=str(target.get("title") or ""),
            dropped=True,
            drop_reason="manual_drop",
            clean_chars=0,
        )
    _drop_tts_chapter(book_dir, chapter_id)
    return True


def restore_chapter(
    book_dir: Path,
    title: str,
    chapter_index: Optional[int] = None,
    rules_path: Optional[Path] = None,
) -> bool:
    toc_path = book_dir / "toc.json"
    if not toc_path.exists():
        raise FileNotFoundError(f"Missing toc.json at {toc_path}")
    toc = _load_json(toc_path)
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("toc.json contains no chapters.")

    raw_entry = None
    if chapter_index is not None:
        for entry in chapters:
            if entry.get("index") == chapter_index:
                raw_entry = entry
                break
    if raw_entry is None:
        for entry in chapters:
            if (entry.get("title") or "") == title:
                raw_entry = entry
                break
    if not raw_entry:
        return False

    raw_rel = raw_entry.get("path") or ""
    if not raw_rel:
        raise FileNotFoundError("Missing raw chapter path.")
    raw_path = book_dir / raw_rel
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw chapter at {raw_path}")

    if rules_path is None:
        candidate = book_rules_path(book_dir)
        if candidate.exists():
            rules_path = candidate
    rules = load_rules(rules_path)
    drop_patterns = compile_patterns(rules.drop_chapter_title_patterns)
    cutoff_patterns = compile_patterns(rules.section_cutoff_patterns)
    remove_patterns = compile_patterns(rules.remove_patterns)

    raw_title = str(raw_entry.get("title") or "").strip()
    drop_reason = should_drop_title(raw_title, drop_patterns)
    if drop_reason:
        raise ValueError("Chapter title still matches drop rules.")

    raw_text = raw_path.read_text(encoding="utf-8")
    raw_text = normalize_text(raw_text)
    cutoff_text, cutoff_reason = apply_section_cutoff(raw_text, cutoff_patterns)
    cleaned, _stats = apply_remove_patterns(cutoff_text, remove_patterns)
    cleaned = normalize_text(cleaned)
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    case_words = _case_context_words(metadata, raw_title)
    heading_lines = _chapter_heading_lines(raw_entry, raw_title)
    cleaned = normalize_small_caps(cleaned, extra_words=case_words)
    cleaned = normalize_all_caps(
        cleaned,
        extra_words=case_words,
        chapter_title=raw_title,
        heading_lines=heading_lines,
    )

    clean_dir = book_dir / "clean" / "chapters"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_rel = (clean_dir / Path(raw_rel).name).relative_to(book_dir).as_posix()
    clean_path = book_dir / clean_rel
    clean_path.write_text(cleaned + "\n", encoding="utf-8")
    tts_text = read_clean_text(clean_path)

    clean_toc_path = book_dir / "clean" / "toc.json"
    if not clean_toc_path.exists():
        raise FileNotFoundError(f"Missing clean/toc.json at {clean_toc_path}")
    clean_toc = _load_json(clean_toc_path)
    clean_entries = clean_toc.get("chapters", [])
    if not isinstance(clean_entries, list):
        clean_entries = []

    for entry in clean_entries:
        if (entry.get("path") or "") == clean_rel:
            return False

    source_index = raw_entry.get("index")
    new_entry = {
        "index": 0,
        "title": raw_title or "Chapter",
        "path": clean_rel,
        "source_index": source_index,
        "headings": heading_lines,
        "kind": "chapter",
    }

    insert_at = len(clean_entries)
    for idx, entry in enumerate(clean_entries):
        entry_source = entry.get("source_index")
        if entry_source is None:
            continue
        if source_index is not None and entry_source >= source_index:
            insert_at = idx
            break
    clean_entries.insert(insert_at, new_entry)
    _reindex_clean_entries(clean_entries)
    clean_toc["chapters"] = clean_entries
    _write_json(clean_toc_path, clean_toc)

    _update_report(
        book_dir,
        rel_path=clean_rel,
        title=raw_title,
        dropped=False,
        cutoff_reason=cutoff_reason,
        raw_chars=len(raw_text),
        clean_chars=len(cleaned),
    )

    tts_dir = book_dir / "tts"
    manifest_path = tts_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        manifest_chapters = manifest.get("chapters", [])
        if isinstance(manifest_chapters, list):
            chapter_id = tts_util.chapter_id_from_path(
                int(new_entry.get("index") or 0),
                raw_title,
                clean_rel,
            )
            max_chars = int(manifest.get("max_chars") or 400)
            pad_ms = int(manifest.get("pad_ms") or 350)
            chunk_mode = str(manifest.get("chunk_mode") or "sentence")
            spans = tts_util.make_chunk_spans(
                tts_text, max_chars=max_chars, chunk_mode=chunk_mode
            )
            chunks = [tts_text[start:end] for start, end in spans]
            if not chunks:
                raise ValueError("No chunks generated for restored chapter.")
            span_list = [[start, end] for start, end in spans]
            chunk_dir = tts_dir / "chunks" / chapter_id
            tts_util.write_chunk_files(chunks, chunk_dir, overwrite=True)

            restored_entry = {
                "index": new_entry.get("index"),
                "id": chapter_id,
                "title": raw_title or chapter_id,
                "path": clean_rel,
                "text_sha256": tts_util.sha256_str(tts_text),
                "chunks": chunks,
                "chunk_spans": span_list,
                "durations_ms": [None] * len(chunks),
            }

            order_ids = [
                tts_util.chapter_id_from_path(
                    int(entry.get("index") or 0),
                    str(entry.get("title") or ""),
                    str(entry.get("path") or ""),
                )
                for entry in clean_entries
            ]
            positions = {cid: idx for idx, cid in enumerate(order_ids)}
            existing = [
                entry
                for entry in manifest_chapters
                if isinstance(entry, dict)
                and (entry.get("id") or "") in positions
                and (entry.get("id") or "") != chapter_id
            ]
            insert_at = len(existing)
            for idx, entry in enumerate(existing):
                if positions.get(entry.get("id") or "", 0) > positions.get(chapter_id, 0):
                    insert_at = idx
                    break
            existing.insert(insert_at, restored_entry)
            manifest["chapters"] = existing
            manifest["max_chars"] = max_chars
            manifest["pad_ms"] = pad_ms
            manifest["chunk_mode"] = chunk_mode
            tts_util.atomic_write_json(manifest_path, manifest)
    return True
