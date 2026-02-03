from pathlib import Path

from nik import epub as epub_util


def test_slugify_japanese_fallback() -> None:
    assert epub_util.slugify("こんにちは") == "chapter"


def test_normalize_text_collapses_whitespace() -> None:
    raw = "Hello\r\n\r\nWorld   \n\n\n"
    assert epub_util.normalize_text(raw) == "Hello\n\nWorld"


def test_title_from_text_uses_first_line_and_truncates() -> None:
    text = "これはとても長い章タイトルですので途中で切ります\n次の行です"
    title = epub_util._title_from_text(text, max_len=10)
    assert title == "これはとても長い章タ..."


def test_extract_chapters_from_sample_epub() -> None:
    epub_path = Path(__file__).parent / "data" / "異人たちとの夏.epub"
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert len(chapters) > 3
    assert not any(ch.title.endswith(".xhtml") for ch in chapters)


def test_extract_ruby_pairs_and_strip_rt() -> None:
    html = "<p><ruby>漢字<rt>かんじ</rt></ruby>です。</p>".encode("utf-8")
    assert epub_util.html_to_text(html) == "漢字です。"
    assert epub_util.extract_ruby_pairs(html) == [("漢字", "かんじ")]
