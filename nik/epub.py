from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import unquote

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, ITEM_IMAGE, epub

from .text import SECTION_BREAK, normalize_section_breaks


@dataclass(frozen=True)
class TocEntry:
    title: str
    href: str


@dataclass(frozen=True)
class Chapter:
    title: str
    href: str
    source: str
    text: str
    ruby_pairs: List[tuple[str, str]] = field(default_factory=list)
    ruby_spans: List[dict] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)


def read_epub(path: Path) -> epub.EpubBook:
    return epub.read_epub(str(path))


def _first_dc_meta(book: epub.EpubBook, name: str) -> str:
    items = book.get_metadata("DC", name)
    if not items:
        return ""
    value, _attrs = items[0]
    return value or ""


def _all_dc_meta(book: epub.EpubBook, name: str) -> List[str]:
    items = book.get_metadata("DC", name)
    values: List[str] = []
    for value, _attrs in items:
        if value:
            values.append(value)
    return values


def _item_name(item: object) -> str:
    get_name = getattr(item, "get_name", None)
    if callable(get_name):
        return get_name() or ""
    return getattr(item, "file_name", "") or ""


def _item_title(item: object) -> str:
    title = getattr(item, "title", "")
    if title:
        return title
    get_title = getattr(item, "get_title", None)
    if callable(get_title):
        value = get_title()
        if value:
            return value
    return ""


def _item_id(item: object) -> str:
    get_id = getattr(item, "get_id", None)
    if callable(get_id):
        value = get_id()
        if value:
            return value
    return getattr(item, "id", "") or ""


def _find_cover_item(book: epub.EpubBook) -> object | None:
    cover_meta = book.get_metadata("OPF", "cover")
    if cover_meta:
        _value, attrs = cover_meta[0]
        cover_id = attrs.get("content") if attrs else None
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item:
                return item

    opf_meta = book.get_metadata("OPF", "meta")
    for _value, attrs in opf_meta:
        if not attrs:
            continue
        if str(attrs.get("name") or "").lower() != "cover":
            continue
        cover_id = attrs.get("content")
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item:
                return item

    for item in book.get_items():
        props = getattr(item, "properties", []) or []
        if isinstance(props, str):
            props = [props]
        if any("cover-image" in prop for prop in props):
            return item

    for item in book.get_items_of_type(ITEM_IMAGE):
        name = _item_name(item).lower()
        item_id = _item_id(item).lower()
        if "cover" in name or "cover" in item_id:
            return item

    best_item = None
    best_size = 0
    for item in book.get_items_of_type(ITEM_IMAGE):
        try:
            data = item.get_content()
        except Exception:
            continue
        if not data:
            continue
        size = len(data)
        if size > best_size:
            best_size = size
            best_item = item

    if best_item:
        return best_item

    return None


def extract_metadata(book: epub.EpubBook) -> dict:
    title = _first_dc_meta(book, "title")
    authors = _all_dc_meta(book, "creator")
    language = _first_dc_meta(book, "language")
    dates = _all_dc_meta(book, "date")
    year = ""
    for value in dates:
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            year = match.group(0)
            break

    cover_info = None
    cover_item = _find_cover_item(book)
    if cover_item:
        cover_info = {
            "id": _item_id(cover_item),
            "href": _item_name(cover_item),
            "media_type": getattr(cover_item, "media_type", "") or "",
        }

    return {
        "title": title,
        "authors": authors,
        "language": language,
        "dates": dates,
        "year": year,
        "cover": cover_info,
    }


def extract_cover_image(book: epub.EpubBook) -> dict | None:
    cover_item = _find_cover_item(book)
    if not cover_item:
        return None
    data = cover_item.get_content()
    if not data:
        return None
    return {
        "data": data,
        "id": _item_id(cover_item),
        "href": _item_name(cover_item),
        "media_type": getattr(cover_item, "media_type", "") or "",
    }


def normalize_href(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    href = href.split("#", 1)[0]
    # Some EPUBs percent-encode filenames in TOC entries.
    return unquote(href)


def flatten_toc(toc: Iterable) -> List[TocEntry]:
    entries: List[TocEntry] = []

    def walk(nodes: Iterable) -> None:
        for node in nodes:
            if isinstance(node, epub.Link):
                if node.href:
                    entries.append(TocEntry(title=node.title or "", href=node.href))
            elif isinstance(node, epub.Section):
                if node.href:
                    entries.append(TocEntry(title=node.title or "", href=node.href))
                subitems = getattr(node, "subitems", None)
                if subitems:
                    walk(subitems)
            elif isinstance(node, (list, tuple)):
                walk(node)

    walk(toc)
    return entries


def build_toc_entries(book: epub.EpubBook) -> List[TocEntry]:
    toc = book.toc or []
    entries = flatten_toc(toc)
    return [e for e in entries if e.href]


def build_spine_entries(book: epub.EpubBook) -> List[TocEntry]:
    entries: List[TocEntry] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue
        title = _item_title(item) or _item_name(item) or ""
        entries.append(TocEntry(title=title, href=_item_name(item)))
    return entries


def _build_spine_items(book: epub.EpubBook) -> List[tuple[str, object]]:
    items: List[tuple[str, object]] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue
        name = _item_name(item)
        if not name:
            continue
        items.append((normalize_href(name), item))
    return items


def _split_series_key(href: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<prefix>.+?)(?:_split_|-split-)\d+(?P<ext>\.[^./]+)$", href)
    if not match:
        return None
    return (match.group("prefix"), match.group("ext"))


def _join_item_text(
    items: Iterable[object], footnote_index: dict[str, set[str]] | None = None
) -> str:
    parts: List[str] = []
    for item in items:
        text = html_to_text(
            item.get_content(),
            footnote_index=footnote_index,
            source_href=_item_name(item),
        )
        if text:
            parts.append(text)
    if not parts:
        return ""
    return normalize_text("\n\n".join(parts))


def _join_item_text_and_ruby(
    items: Iterable[object],
    footnote_index: dict[str, set[str]] | None = None,
) -> tuple[str, List[tuple[str, str]], List[dict]]:
    parts: List[str] = []
    ruby_pairs: List[tuple[str, str]] = []
    ruby_spans: List[dict] = []
    for item in items:
        content = item.get_content()
        token_text = _html_to_text_with_ruby_tokens(
            content,
            footnote_index=footnote_index,
            source_href=_item_name(item),
        )
        if token_text:
            parts.append(token_text)
    if not parts:
        return "", ruby_pairs, ruby_spans
    joined = normalize_text("\n\n".join(parts))
    text, ruby_spans = _strip_ruby_tokens(joined)
    ruby_pairs = [
        (span.get("base", ""), span.get("reading", "")) for span in ruby_spans
    ]
    return text, ruby_pairs, ruby_spans


_NOTE_MARKER_RE = re.compile(r"^\d{1,4}[a-z]?[.)]?$", re.IGNORECASE)
_RUBY_TOKEN_START = "[[[RUBY_START]]]"
_RUBY_TOKEN_MID = "[[[RUBY_MID]]]"
_RUBY_TOKEN_END = "[[[RUBY_END]]]"


def _normalize_id(value: str) -> str:
    return (value or "").strip().lower()


def _href_fragment(href: str) -> str:
    if not href:
        return ""
    parts = href.split("#", 1)
    if len(parts) < 2:
        return ""
    return unquote(parts[1])


def _looks_like_note_marker(text: str) -> bool:
    if not text:
        return False
    return bool(_NOTE_MARKER_RE.match(text.strip()))


def _collect_footnote_index(
    book: epub.EpubBook,
) -> dict[str, set[str]]:
    note_ids: set[str] = set()
    backref_ids: set[str] = set()

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        content = item.get_content()
        if not content:
            continue
        head = content.lstrip()[:512].lower()
        parser = (
            "lxml-xml"
            if (head.startswith(b"<?xml") or b"xmlns=" in head)
            else "lxml"
        )
        soup = BeautifulSoup(content, parser)
        for tag in soup.find_all(attrs={"epub:type": "footnote"}):
            note_id = _normalize_id(str(tag.get("id") or ""))
            if note_id:
                note_ids.add(note_id)
        for tag in soup.find_all(attrs={"role": "doc-footnote"}):
            note_id = _normalize_id(str(tag.get("id") or ""))
            if note_id:
                note_ids.add(note_id)
        for tag in soup.find_all(["p", "section", "div", "aside", "ol", "ul", "li", "td"]):
            attrs = getattr(tag, "attrs", None)
            if attrs is None:
                continue
            classes = attrs.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            class_text = " ".join(classes).lower()
            id_text = str(attrs.get("id") or "")
            if "footnote" in class_text or "endnote" in class_text:
                note_id = _normalize_id(id_text)
                if note_id:
                    note_ids.add(note_id)
            if id_text.lower().startswith(("fn", "footnote", "endnote")):
                note_id = _normalize_id(id_text)
                if note_id:
                    note_ids.add(note_id)

        for anchor in soup.find_all("a"):
            text = anchor.get_text(strip=True)
            if not _looks_like_note_marker(text):
                continue
            fragment = _normalize_id(_href_fragment(str(anchor.get("href") or "")))
            if fragment:
                backref_ids.add(fragment)
            parent = anchor.parent
            if parent and getattr(parent, "attrs", None):
                parent_id = _normalize_id(str(parent.get("id") or ""))
                if parent_id:
                    parent_text = parent.get_text(strip=True)
                    if parent_text == text:
                        note_ids.add(parent_id)

    return {"note_ids": note_ids, "backref_ids": backref_ids}


def _parse_html_soup(html: bytes | str) -> BeautifulSoup:
    if isinstance(html, bytes):
        head = html.lstrip()[:512].lower()
        parser = (
            "lxml-xml"
            if (head.startswith(b"<?xml") or b"xmlns=" in head)
            else "lxml"
        )
    else:
        head = str(html).lstrip()[:512].lower()
        parser = "lxml-xml" if (head.startswith("<?xml") or "xmlns=" in head) else "lxml"
    return BeautifulSoup(html, parser)


def _ruby_base_reading(ruby: object) -> tuple[str, str]:
    base_parts: List[str] = []
    reading_parts: List[str] = []
    contents = getattr(ruby, "contents", [])
    for child in contents:
        name = getattr(child, "name", None)
        if name in {"rt", "rp"}:
            continue
        if isinstance(child, str):
            base_parts.append(child)
        else:
            base_parts.append(child.get_text(separator="", strip=False))
    for rt in getattr(ruby, "find_all", lambda *_args, **_kwargs: [])("rt"):
        reading_parts.append(rt.get_text(separator="", strip=False))
    base = "".join(base_parts).strip()
    reading = "".join(reading_parts).strip()
    return base, reading


def _soup_to_text(
    soup: BeautifulSoup,
    footnote_index: dict[str, set[str]] | None = None,
    source_href: str = "",
) -> str:
    _ = source_href
    for tag in soup.find_all(["rt", "rp"]):
        tag.decompose()
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    for tag in soup.find_all("sup"):
        tag.decompose()
    for tag in soup.find_all(attrs={"epub:type": "noteref"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"epub:type": "footnote"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "doc-noteref"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "doc-footnote"}):
        tag.decompose()
    if footnote_index:
        note_ids = footnote_index.get("note_ids", set())
        backref_ids = footnote_index.get("backref_ids", set())
        if note_ids or backref_ids:
            for tag in soup.find_all(attrs={"id": True}):
                attrs = getattr(tag, "attrs", None)
                if not attrs:
                    continue
                tag_id = _normalize_id(str(attrs.get("id") or ""))
                if tag_id and tag_id in backref_ids:
                    tag.decompose()
            for anchor in soup.find_all("a"):
                text = anchor.get_text(strip=True)
                if not _looks_like_note_marker(text):
                    continue
                fragment = _normalize_id(
                    _href_fragment(str(anchor.get("href") or ""))
                )
                if fragment and (fragment in note_ids or fragment in backref_ids):
                    anchor.decompose()
    for tag in soup.find_all(["p", "section", "div", "aside", "ol", "ul", "li"]):
        attrs = getattr(tag, "attrs", None)
        if attrs is None:
            continue
        classes = attrs.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        class_text = " ".join(classes).lower()
        id_text = str(attrs.get("id") or "").lower()
        if (
            "footnote" in class_text
            or "endnote" in class_text
            or "copyright" in class_text
            or "credit" in class_text
        ):
            tag.decompose()
            continue
        if id_text.startswith(("fn", "footnote", "endnote")):
            tag.decompose()
    root = soup.body if soup.body else soup

    block_tags = {
        "p",
        "div",
        "li",
        "blockquote",
        "pre",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def _class_key(tag: object) -> str:
        classes = getattr(tag, "get", lambda *_args, **_kwargs: [])("class", [])
        if isinstance(classes, str):
            classes = [classes]
        cleaned = [str(c).strip() for c in classes if str(c).strip()]
        return " ".join(cleaned)

    def _top_level_blocks(name: str) -> List[object]:
        out: List[object] = []
        for elem in root.find_all(name):
            if any(getattr(parent, "name", None) in block_tags for parent in elem.parents):
                continue
            out.append(elem)
        return out

    p_class_counts: dict[str, int] = {}
    for para in _top_level_blocks("p"):
        key = _class_key(para)
        if not key:
            continue
        text = para.get_text(separator="", strip=True)
        if not text:
            continue
        p_class_counts[key] = p_class_counts.get(key, 0) + 1

    normal_classes: set[str] = set()
    if p_class_counts:
        primary_count = max(p_class_counts.values())
        if primary_count >= 8:
            threshold = max(3, int(primary_count * 0.2))
            normal_classes = {
                key for key, count in p_class_counts.items() if count >= threshold
            }
        else:
            normal_classes = set(p_class_counts.keys())

    ornament_re = re.compile(
        r"^(?:[＊*・…‥\u2026\-—=]{3,}|[◆◇■□●○]{3,}|"
        r"(?:\*|\uFF0A)(?:\s*(?:\*|\uFF0A)){2,})$"
    )

    blocks: List[str] = []
    for elem in root.find_all(block_tags):
        if any(getattr(parent, "name", None) in block_tags for parent in elem.parents):
            continue
        if getattr(elem, "name", None) == "hr":
            if blocks:
                blocks.append(SECTION_BREAK)
            continue

        for br in elem.find_all("br"):
            br.replace_with("\n")
        text = elem.get_text(separator="", strip=False)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()
        if not text:
            continue

        if ornament_re.fullmatch(text):
            if blocks:
                blocks.append(SECTION_BREAK)
            continue

        if getattr(elem, "name", None) == "p":
            key = _class_key(elem)
            if key and key not in normal_classes and blocks:
                blocks.append(SECTION_BREAK)

        blocks.append(text)

    if blocks:
        return normalize_text("\n\n".join(blocks))

    text = root.get_text(separator="", strip=False)
    return normalize_text(text)


def _strip_ruby_tokens(text: str) -> tuple[str, List[dict]]:
    spans: List[dict] = []
    out_parts: List[str] = []
    idx = 0
    out_len = 0
    while True:
        start = text.find(_RUBY_TOKEN_START, idx)
        if start == -1:
            tail = text[idx:]
            if tail:
                out_parts.append(tail)
                out_len += len(tail)
            break
        if start > idx:
            segment = text[idx:start]
            out_parts.append(segment)
            out_len += len(segment)
        mid = text.find(_RUBY_TOKEN_MID, start + len(_RUBY_TOKEN_START))
        end = -1
        if mid != -1:
            end = text.find(_RUBY_TOKEN_END, mid + len(_RUBY_TOKEN_MID))
        if mid == -1 or end == -1:
            token_text = text[start : start + len(_RUBY_TOKEN_START)]
            out_parts.append(token_text)
            out_len += len(token_text)
            idx = start + len(_RUBY_TOKEN_START)
            continue
        base = text[start + len(_RUBY_TOKEN_START) : mid]
        reading = text[mid + len(_RUBY_TOKEN_MID) : end]
        out_parts.append(base)
        span_start = out_len
        span_end = span_start + len(base)
        out_len = span_end
        if base and reading:
            spans.append(
                {
                    "start": span_start,
                    "end": span_end,
                    "base": base,
                    "reading": reading,
                }
            )
        idx = end + len(_RUBY_TOKEN_END)
    return "".join(out_parts), spans


def _html_to_text_with_ruby_tokens(
    html: bytes | str,
    footnote_index: dict[str, set[str]] | None = None,
    source_href: str = "",
) -> str:
    soup = _parse_html_soup(html)
    for ruby in soup.find_all("ruby"):
        base, reading = _ruby_base_reading(ruby)
        if base and reading:
            token = f"{_RUBY_TOKEN_START}{base}{_RUBY_TOKEN_MID}{reading}{_RUBY_TOKEN_END}"
            ruby.replace_with(token)
        elif base:
            ruby.replace_with(base)
        else:
            ruby.replace_with("")
    return _soup_to_text(soup, footnote_index=footnote_index, source_href=source_href)


def html_to_text_with_ruby(
    html: bytes | str,
    footnote_index: dict[str, set[str]] | None = None,
    source_href: str = "",
) -> tuple[str, List[dict]]:
    token_text = _html_to_text_with_ruby_tokens(
        html, footnote_index=footnote_index, source_href=source_href
    )
    return _strip_ruby_tokens(token_text)


def html_to_text(
    html: bytes,
    footnote_index: dict[str, set[str]] | None = None,
    source_href: str = "",
) -> str:
    soup = _parse_html_soup(html)
    return _soup_to_text(soup, footnote_index=footnote_index, source_href=source_href)


def extract_ruby_pairs(html: bytes | str) -> List[tuple[str, str]]:
    soup = _parse_html_soup(html)
    pairs: List[tuple[str, str]] = []
    for ruby in soup.find_all("ruby"):
        base, reading = _ruby_base_reading(ruby)
        if not base or not reading:
            continue
        pairs.append((base, reading))
    return pairs


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
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\.[ \t]*\.[ \t]*\.", "...", text)
    return text.strip()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:60] or "chapter"


_TITLE_MAX_LEN = 40


def _title_from_text(text: str, max_len: int = _TITLE_MAX_LEN) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if len(cleaned) > max_len:
            return cleaned[:max_len].rstrip() + "..."
        return cleaned
    return ""


def _extract_heading_title(html: bytes | str) -> str:
    headings = _extract_html_headings(html)
    if headings:
        return headings[0]

    soup = _parse_html_soup(html)
    title_node = soup.find("title")
    if title_node:
        text = title_node.get_text(separator=" ", strip=True)
        if text:
            return text
    return ""


_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_HEADING_CLASS_HINTS = (
    "title",
    "subtitle",
    "chapter",
    "section",
    "headline",
    "heading",
    "midashi",
)


def _looks_like_heading_marker(value: str) -> bool:
    cleaned = str(value or "").strip().lower()
    if not cleaned:
        return False
    if any(token in cleaned for token in _HEADING_CLASS_HINTS):
        return True
    return bool(re.search(r"(^|[-_])(ttl|hd)([-_0-9]|$)", cleaned))


def _extract_html_headings(html: bytes | str) -> List[str]:
    soup = _parse_html_soup(html)
    for tag in soup.find_all(["rt", "rp", "script", "style"]):
        tag.decompose()
    root = soup.body if soup.body else soup
    out: List[str] = []
    seen: set[str] = set()

    def add_node_text(node: object) -> None:
        text = node.get_text(separator=" ", strip=True)
        if not text:
            return
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned or len(cleaned) > 120:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    for tag in _HEADING_TAGS:
        for node in root.find_all(tag):
            add_node_text(node)

    for node in root.find_all(True):
        name = str(getattr(node, "name", "") or "").lower()
        if name in _HEADING_TAGS:
            continue
        classes = getattr(node, "get", lambda *_args, **_kwargs: [])("class", [])
        if isinstance(classes, str):
            classes = [classes]
        id_value = str(getattr(node, "get", lambda *_args, **_kwargs: "")("id", "") or "")
        has_marker = any(_looks_like_heading_marker(cls) for cls in classes)
        if not has_marker and not _looks_like_heading_marker(id_value):
            continue
        add_node_text(node)
    return out


def _looks_like_filename(title: str, href: str) -> bool:
    cleaned = str(title or "").strip()
    if not cleaned:
        return True
    base = Path(href or "")
    stem = base.stem
    name = base.name
    if cleaned == href or cleaned == name or cleaned == stem:
        return True
    lowered = cleaned.lower()
    if lowered.endswith(".xhtml") or lowered.endswith(".html"):
        return True
    if stem and lowered == stem.lower():
        return True
    if re.fullmatch(r"[a-z]\d+[a-z0-9]*", lowered):
        return True
    return False


def _chapters_from_entries(
    book: epub.EpubBook,
    entries: Iterable[TocEntry],
    footnote_index: dict[str, set[str]] | None = None,
    title_overrides: Optional[dict[str, str]] = None,
    fallback_prefix: str = "Chapter",
) -> List[Chapter]:
    seen: set[str] = set()
    chapters: List[Chapter] = []

    for idx, entry in enumerate(entries, start=1):
        base_href = normalize_href(entry.href)
        if not base_href or base_href in seen:
            continue
        seen.add(base_href)

        item = book.get_item_with_href(base_href)
        if not item:
            item = book.get_item_with_id(base_href)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue

        content = item.get_content()
        text, ruby_spans = html_to_text_with_ruby(
            content,
            footnote_index=footnote_index,
            source_href=_item_name(item),
        )
        if not text:
            continue
        ruby_pairs = [
            (span.get("base", ""), span.get("reading", "")) for span in ruby_spans
        ]
        headings = _extract_html_headings(content)

        title = ""
        if title_overrides and base_href in title_overrides:
            title = str(title_overrides[base_href] or "").strip()
        if not title:
            title = entry.title or _item_title(item) or ""
        if _looks_like_filename(title, base_href):
            heading = _extract_heading_title(item.get_content())
            if heading and not _looks_like_filename(heading, base_href):
                title = heading
        if _looks_like_filename(title, base_href):
            title = f"{fallback_prefix} {idx}"
        chapters.append(
            Chapter(
                title=title,
                href=entry.href,
                source=base_href,
                text=text,
                ruby_pairs=ruby_pairs,
                ruby_spans=ruby_spans,
                headings=headings,
            )
        )

    return chapters


def _chapters_from_toc_entries(
    book: epub.EpubBook,
    entries: Iterable[TocEntry],
    footnote_index: dict[str, set[str]] | None = None,
) -> List[Chapter]:
    spine_items = _build_spine_items(book)
    spine_index = {href: idx for idx, (href, _item) in enumerate(spine_items)}
    chapters: List[Chapter] = []
    seen: set[str] = set()
    split_series_counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        base_href = normalize_href(entry.href)
        if not base_href:
            continue
        key = _split_series_key(base_href)
        if key:
            split_series_counts[key] = split_series_counts.get(key, 0) + 1

    for entry in entries:
        base_href = normalize_href(entry.href)
        if not base_href or base_href in seen:
            continue
        seen.add(base_href)
        ruby_spans: List[dict] = []
        headings: List[str] = []

        start_idx = spine_index.get(base_href)
        merged_items: List[object] = []
        if start_idx is not None:
            key = _split_series_key(base_href)
            prev_key = (
                _split_series_key(spine_items[start_idx - 1][0])
                if start_idx > 0
                else None
            )
            if (
                key
                and key != prev_key
                and split_series_counts.get(key, 0) == 1
            ):
                idx = start_idx
                while idx < len(spine_items):
                    href, item = spine_items[idx]
                    if _split_series_key(href) != key:
                        break
                    merged_items.append(item)
                    idx += 1

        if merged_items:
            text, ruby_pairs, ruby_spans = _join_item_text_and_ruby(
                merged_items, footnote_index=footnote_index
            )
            heading_seen: set[str] = set()
            for merged in merged_items:
                merged_content = merged.get_content()
                for heading in _extract_html_headings(merged_content):
                    key = heading.casefold()
                    if key in heading_seen:
                        continue
                    heading_seen.add(key)
                    headings.append(heading)
            item_for_title = merged_items[0]
        else:
            item_for_title = None
            if start_idx is not None:
                item_for_title = spine_items[start_idx][1]
            if not item_for_title:
                item_for_title = book.get_item_with_href(base_href)
            if not item_for_title:
                item_for_title = book.get_item_with_id(base_href)
            if not item_for_title or item_for_title.get_type() != ITEM_DOCUMENT:
                continue
            content = item_for_title.get_content()
            text, ruby_spans = html_to_text_with_ruby(
                content,
                footnote_index=footnote_index,
                source_href=_item_name(item_for_title),
            )
            headings = _extract_html_headings(content)
            ruby_pairs = [
                (span.get("base", ""), span.get("reading", ""))
                for span in ruby_spans
            ]

        if not text:
            continue

        title = entry.title or _item_title(item_for_title) or Path(base_href).stem
        chapters.append(
            Chapter(
                title=title,
                href=entry.href,
                source=base_href,
                text=text,
                ruby_pairs=ruby_pairs,
                ruby_spans=ruby_spans,
                headings=headings,
            )
        )

    return chapters


def extract_chapters(book: epub.EpubBook, prefer_toc: bool = True) -> List[Chapter]:
    footnote_index = _collect_footnote_index(book)
    entries = build_toc_entries(book) if prefer_toc else []
    chapters = (
        _chapters_from_toc_entries(book, entries, footnote_index)
        if entries
        else []
    )
    if entries and not chapters:
        chapters = _chapters_from_entries(book, entries, footnote_index)
    if chapters and prefer_toc:
        spine_chapters = _chapters_from_entries(
            book,
            build_spine_entries(book),
            footnote_index,
            title_overrides={normalize_href(e.href): e.title for e in entries if e.href},
        )
        if spine_chapters:
            toc_chars = sum(len(ch.text) for ch in chapters if ch.text)
            spine_chars = sum(len(ch.text) for ch in spine_chapters if ch.text)
            if spine_chars > 0:
                ratio = toc_chars / spine_chars
                if ratio < 0.2 and len(spine_chapters) > len(chapters):
                    chapters = spine_chapters
    if not chapters:
        chapters = _chapters_from_entries(
            book, build_spine_entries(book), footnote_index
        )
    return chapters
