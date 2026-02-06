import json
from pathlib import Path

from nik import sanitize as sanitize_util
from nik.text import SECTION_BREAK


def _write_book(tmp_path: Path) -> Path:
    book_dir = tmp_path / "book"
    raw_dir = book_dir / "raw" / "chapters"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "0001-chapter.txt"
    raw_path.write_text("本文です。", encoding="utf-8")

    toc = {
        "created_unix": 0,
        "source_epub": "book.epub",
        "metadata": {"title": "Test Book"},
        "chapters": [
            {
                "index": 1,
                "title": "Chapter 1",
                "path": raw_path.relative_to(book_dir).as_posix(),
            }
        ],
    }
    toc_path = book_dir / "toc.json"
    toc_path.write_text(json.dumps(toc, ensure_ascii=False), encoding="utf-8")
    return book_dir


def test_sanitize_book_creates_clean_chapters(tmp_path: Path) -> None:
    book_dir = _write_book(tmp_path)
    written = sanitize_util.sanitize_book(book_dir=book_dir, overwrite=True)
    assert written == 1
    clean_path = book_dir / "clean" / "chapters" / "0001-chapter.txt"
    assert clean_path.exists()
    title_path = book_dir / "clean" / "chapters" / "0000-title.txt"
    assert title_path.exists() is False


def test_refresh_chunks_creates_manifest(tmp_path: Path) -> None:
    book_dir = _write_book(tmp_path)
    sanitize_util.sanitize_book(book_dir=book_dir, overwrite=True)
    sanitize_util.refresh_chunks(book_dir=book_dir)
    assert (book_dir / "tts" / "manifest.json").exists()


def test_normalize_text_preserves_section_break() -> None:
    raw = f"一。\n\n{SECTION_BREAK}\n\n二。"
    cleaned = sanitize_util.normalize_text(raw)
    assert SECTION_BREAK in cleaned
