"""Review-facing view of a book's ruby evidence.

Groups the (coalesced) ruby spans stored in reading-overrides.json by
(base, reading), annotates each group with flags and sample contexts, and
persists reviewer decisions back into ``ruby.decisions``. The TTS pipeline
consumes those decisions in nik/tts.py; this module only builds the review
payload and writes decisions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import tts as tts_util

_CONTEXT_RADIUS = 16
_MAX_CONTEXTS = 8
# A single kanji repeatedly ruby'd with the same non-dictionary reading is
# usually a person name (一=はじめ); one-offs are usually context readings
# (火=び inside 焚き火), so demand a few occurrences before flagging.
_SINGLE_KANJI_NAME_MIN_COUNT = 3

_SMALL_KANA = {
    "や": "ゃ",
    "ゆ": "ゅ",
    "よ": "ょ",
    "つ": "っ",
    "あ": "ぁ",
    "い": "ぃ",
    "え": "ぇ",
    "お": "ぉ",
    "ヤ": "ャ",
    "ユ": "ュ",
    "ヨ": "ョ",
    "ツ": "ッ",
    "ア": "ァ",
    "イ": "ィ",
    "エ": "ェ",
    "オ": "ォ",
}
_PALATAL_TRIGGERS = set("きしちにひみりぎじぢびぴキシチニヒミリギジヂビピ")
_TE_DE = set("てでテデ")
_FU = set("ふフ")
_KATAKANA_U = set("ウ")
_SHI_JI_CHI = set("しじちシジチ")
_TSU = set("つツ")
_GEMINATION_FOLLOWERS = set(
    "かきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどぱぴぷぺぽ"
    "カキクケコガギグゲゴサシスセソザジズゼゾタチツテトダヂヅデドパピプペポ"
)


def suggest_small_kana_reading(reading: str) -> str:
    """Candidate sutegana repair for ruby whose small kana were typeset big
    (ょゅゃっ printed as よゆやつ). Positions are only converted where a small
    kana is phonotactically plausible; the result is a suggestion for the
    review UI and is never applied automatically. Returns "" when nothing
    plausible was found."""
    chars = list(reading or "")
    changed = False
    for idx in range(1, len(chars)):
        ch = chars[idx]
        small = _SMALL_KANA.get(ch)
        if not small:
            continue
        prev = chars[idx - 1]
        convert = False
        if ch in "やゆよヤユヨ" and prev in _PALATAL_TRIGGERS:
            convert = True
        elif ch in "いイ" and prev in _TE_DE:
            convert = True
        elif ch in "えエ" and prev in (_SHI_JI_CHI | _TE_DE | _FU):
            convert = True
        elif ch in "あいえおアイエオ" and prev in _FU:
            convert = True
        elif ch in "イエオ" and prev in _KATAKANA_U:
            convert = True
        elif (
            ch in _TSU
            and idx + 1 < len(chars)
            and chars[idx + 1] in _GEMINATION_FOLLOWERS
        ):
            convert = True
        if convert:
            chars[idx] = small
            changed = True
    if not changed:
        return ""
    out = "".join(chars)
    return out if out != reading else ""


def _context_payload(
    chapter_id: str, text: str, start: int, end: int, kind: str
) -> dict:
    left = max(0, start - _CONTEXT_RADIUS)
    right = min(len(text), end + _CONTEXT_RADIUS)
    return {
        "chapter_id": chapter_id,
        "start": start,
        "end": end,
        "kind": kind,
        "before": ("…" if left > 0 else "") + text[left:start].replace("\n", " "),
        "hit": text[start:end],
        "after": text[end:right].replace("\n", " ") + ("…" if right < len(text) else ""),
    }


def _overlaps_any(start: int, end: int, reserved: List[tuple[int, int]]) -> bool:
    for r_start, r_end in reserved:
        if start < r_end and end > r_start:
            return True
    return False


def _literal_override_map(entries: List[dict]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for entry in entries:
        base = str(entry.get("base") or "").strip()
        reading = str(entry.get("reading") or "").strip()
        if base and reading:
            out.setdefault(base, set()).add(reading)
    return out


def build_review_groups(book_dir: Path) -> dict:
    ruby_data = tts_util._load_ruby_data(book_dir)
    decisions = tts_util._load_ruby_decisions(ruby_data)
    chapters_map = ruby_data.get("chapters") if isinstance(ruby_data, dict) else None
    chapters_map = chapters_map if isinstance(chapters_map, dict) else {}

    texts: Dict[str, str] = {}
    chapter_order: Dict[str, int] = {}
    try:
        for chapter in tts_util.load_book_chapters(book_dir):
            texts[chapter.id] = chapter.text
            chapter_order[chapter.id] = chapter.index
    except Exception:
        texts = {}

    groups: Dict[str, dict] = {}
    inline_spans_by_chapter: Dict[str, List[tuple[int, int]]] = {}

    def group_for(base: str, reading: str) -> dict:
        key = tts_util._ruby_decision_key(base, reading)
        group = groups.get(key)
        if group is None:
            group = {
                "key": key,
                "base": base,
                "reading": reading,
                "count": 0,
                "chapter_ids": set(),
                "contexts": [],
                "honorific": False,
            }
            groups[key] = group
        return group

    for chapter_id, chapter_entry in chapters_map.items():
        if not isinstance(chapter_entry, dict):
            continue
        text = texts.get(chapter_id) or ""
        matched = (
            tts_util._select_ruby_spans(chapter_id, text, ruby_data) if text else []
        )
        if matched:
            spans = tts_util._coalesce_adjacent_single_kanji_ruby_spans(matched)
            positional = True
        else:
            spans = tts_util._coalesce_adjacent_single_kanji_ruby_spans(
                tts_util._chapter_ruby_spans_for_counts(chapter_entry)
            )
            positional = False
        reserved = inline_spans_by_chapter.setdefault(chapter_id, [])
        for span in spans:
            base = str(span.get("base") or "").strip()
            reading = str(span.get("reading") or "").strip()
            if not base or not reading:
                continue
            group = group_for(base, reading)
            group["count"] += 1
            group["chapter_ids"].add(chapter_id)
            if not positional:
                continue
            try:
                start = int(span.get("start"))
                end = int(span.get("end"))
            except (TypeError, ValueError):
                continue
            reserved.append((start, end))
            if len(group["contexts"]) < _MAX_CONTEXTS:
                group["contexts"].append(
                    _context_payload(chapter_id, text, start, end, "inline")
                )
            tail = text[end : end + 4]
            if tail.startswith(tts_util._RUBY_NAME_HONORIFIC_SUFFIXES):
                group["honorific"] = True

    effective_map = _literal_override_map(tts_util._ruby_global_overrides(ruby_data))
    baseline_data = dict(ruby_data) if isinstance(ruby_data, dict) else {}
    baseline_data.pop("decisions", None)
    baseline_map = _literal_override_map(
        tts_util._ruby_global_overrides(baseline_data)
    )

    payload_groups: List[dict] = []
    for key, group in groups.items():
        base = group["base"]
        reading = group["reading"]
        count = group["count"]
        decision = decisions.get(key)
        corrected = str((decision or {}).get("reading") or "").strip()
        effective_reading = corrected or reading
        decided_scope = str((decision or {}).get("scope") or "").strip()
        default_scope = (
            "global" if reading in baseline_map.get(base, set()) else "inline"
        )
        scope = decided_scope or default_scope
        mode = str((decision or {}).get("mode") or "").strip()
        if not mode and len(base) == 1:
            mode = "isolated"

        aligned = tts_util._ruby_reading_aligned(base, reading)
        suggestion = ""
        if not aligned and not corrected:
            suggestion = suggest_small_kana_reading(reading)
        flags: List[str] = []
        if aligned:
            flags.append("aligned")
        if count == 1:
            flags.append("singleton")
        if (not tts_util._is_kanji_only(base)) or (
            tts_util._reading_is_katakana(reading) and not aligned
        ):
            flags.append("wordplay")
        name_like = group["honorific"] or (
            len(base) == 1
            and count >= _SINGLE_KANJI_NAME_MIN_COUNT
            and not aligned
            and tts_util._is_kanji_only(base)
        )
        if name_like:
            flags.append("name")
        if suggestion:
            flags.append("sutegana")

        # Bare-text occurrences the (would-be) global rule would touch.
        prop_entry: dict = {"base": base, "reading": effective_reading}
        if mode:
            prop_entry["mode"] = mode
        prop_count = 0
        prop_contexts: List[dict] = []
        for chapter_id, text in texts.items():
            if not text or base not in text:
                continue
            candidate_spans = tts_util._reading_override_spans(text, [prop_entry])
            reserved = inline_spans_by_chapter.get(chapter_id, [])
            for span in candidate_spans:
                try:
                    start = int(span.get("start"))
                    end = int(span.get("end"))
                except (TypeError, ValueError):
                    continue
                if _overlaps_any(start, end, reserved):
                    continue
                prop_count += 1
                if len(prop_contexts) < _MAX_CONTEXTS:
                    prop_contexts.append(
                        _context_payload(chapter_id, text, start, end, "propagation")
                    )

        payload_groups.append(
            {
                "key": key,
                "base": base,
                "reading": reading,
                "corrected": corrected,
                "count": count,
                "prop_count": prop_count,
                "chapters": sorted(
                    group["chapter_ids"],
                    key=lambda cid: chapter_order.get(cid, 1 << 30),
                ),
                "scope": scope,
                "default_scope": default_scope,
                "decided": decision is not None,
                "propagates": effective_reading in effective_map.get(base, set()),
                "mode": mode,
                "flags": flags,
                "suggestion": suggestion,
                "contexts": group["contexts"],
                "prop_contexts": prop_contexts,
            }
        )

    payload_groups.sort(key=lambda item: (-int(item["count"]), item["base"]))
    return {"groups": payload_groups}


def update_decisions(book_dir: Path, updates: Dict[str, object]) -> Dict[str, dict]:
    """Merge decision updates into reading-overrides.json (value None deletes
    a decision) and return the stored decisions."""
    path = book_dir / "reading-overrides.json"
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    ruby = data.get("ruby")
    if not isinstance(ruby, dict):
        ruby = {}
        data["ruby"] = ruby
    decisions = ruby.get("decisions")
    if not isinstance(decisions, dict):
        decisions = {}
    ruby["decisions"] = decisions

    now = int(time.time())
    for raw_key, raw_value in (updates or {}).items():
        base, sep, reading = str(raw_key).partition("|")
        if not sep or not base.strip():
            continue
        key = tts_util._ruby_decision_key(base, reading)
        if raw_value is None:
            decisions.pop(key, None)
            continue
        entry = tts_util._normalize_ruby_decision(raw_value)
        if not entry:
            decisions.pop(key, None)
            continue
        entry["updated_unix"] = now
        decisions[key] = entry

    if not decisions:
        ruby.pop("decisions", None)
    data.setdefault("created_unix", now)
    data["updated_unix"] = now

    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)
    return dict(ruby.get("decisions") or {})


__all__ = [
    "build_review_groups",
    "suggest_small_kana_reading",
    "update_decisions",
]
