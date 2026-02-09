from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


SECTION_BREAK = "\uE000"


def normalize_section_breaks(text: str) -> str:
    if not text:
        return text
    if "☆" in text:
        text = re.sub(
            r"(?m)^[ \t\u3000]*☆+[ \t\u3000]*$",
            SECTION_BREAK,
            text,
        )
    if SECTION_BREAK not in text:
        return text
    text = re.sub(
        rf"[ \t]*{SECTION_BREAK}[ \t]*",
        f"\n\n{SECTION_BREAK}\n\n",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(
        rf"(?:\n\n{SECTION_BREAK}){{2,}}",
        f"\n\n{SECTION_BREAK}",
        text,
    )
    text = re.sub(
        rf"(?:{SECTION_BREAK}\n\n){{2,}}",
        f"{SECTION_BREAK}\n\n",
        text,
    )
    text = re.sub(rf"^(?:\s*{SECTION_BREAK}\s*)+", "", text)
    text = re.sub(rf"(?:\s*{SECTION_BREAK}\s*)+$", "", text)
    return text


def strip_section_breaks(text: str) -> str:
    if not text or SECTION_BREAK not in text:
        return text
    return text.replace(SECTION_BREAK, "")


def read_clean_text(path: Path) -> str:
    s = path.read_text(encoding="utf-8", errors="strict")
    s = s.replace("\u02bc", "'").replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip() + "\n"


def _extract_markdown_title(lines: Iterable[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            return heading
        return ""
    return ""


def title_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    if not stem:
        return ""
    stem = re.sub(r"^[0-9]+[-_ ]+", "", stem)
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem or path.stem


def guess_title_from_path(path: Path) -> str:
    heading = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            heading = _extract_markdown_title(handle)
    except (OSError, UnicodeDecodeError):
        heading = ""
    if heading:
        return heading
    fallback = title_from_filename(path)
    return fallback or path.stem or "text"
