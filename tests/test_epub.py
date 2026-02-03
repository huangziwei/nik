from pathlib import Path

from nik import epub as epub_util


def test_slugify_japanese_fallback() -> None:
    assert epub_util.slugify("こんにちは") == "chapter"


def test_normalize_text_collapses_whitespace() -> None:
    raw = "Hello\r\n\r\nWorld   \n\n\n"
    assert epub_util.normalize_text(raw) == "Hello\n\nWorld"


def test_extract_chapters_from_sample_epub() -> None:
    epub_path = Path(__file__).parent / "data" / "異人たちとの夏.epub"
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert chapters
