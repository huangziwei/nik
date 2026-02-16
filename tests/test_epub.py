from pathlib import Path

from ebooklib import epub
import pytest

from nik import epub as epub_util
from nik.text import SECTION_BREAK


def test_slugify_japanese_fallback() -> None:
    assert epub_util.slugify("こんにちは") == "chapter"


def test_normalize_text_collapses_whitespace() -> None:
    raw = "Hello\r\n\r\nWorld   \n\n\n"
    assert epub_util.normalize_text(raw) == "Hello\n\nWorld"


def test_title_from_text_uses_first_line_and_truncates() -> None:
    text = "これはとても長い章タイトルですので途中で切ります\n次の行です"
    title = epub_util._title_from_text(text, max_len=10)
    assert title == "これはとても長い章タ..."


def _write_sample_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("test-book")
    book.set_title("Sample Book")
    book.set_language("ja")

    chapters = []
    for idx, title in enumerate(("序章", "第一章", "第二章", "終章"), start=1):
        chapter = epub.EpubHtml(
            title=title,
            file_name=f"chapter-{idx}.xhtml",
            lang="ja",
        )
        chapter.content = f"<h1>{title}</h1><p>本文{idx}です。</p>"
        book.add_item(chapter)
        chapters.append(chapter)

    book.toc = tuple(chapters)
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_from_sample_epub(tmp_path: Path) -> None:
    epub_path = tmp_path / "sample.epub"
    _write_sample_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert len(chapters) > 3
    assert not any(ch.title.endswith(".xhtml") for ch in chapters)
    assert any("序章" in ch.headings for ch in chapters)


def test_extract_html_headings_supports_heading_tags_and_class_markers() -> None:
    html = """
    <html>
      <body>
        <h2>Part I</h2>
        <p><span class="subtitle">- Intro -</span></p>
      </body>
    </html>
    """.encode("utf-8")
    headings = epub_util._extract_html_headings(html)
    assert "Part I" in headings
    assert "- Intro -" in headings


def test_extract_chapters_real_epub_keeps_heading_markers() -> None:
    epub_path = Path("tests/data/バビロン１ ―女― (講談社タイガ).epub")
    if not epub_path.exists():
        pytest.skip(f"Missing test EPUB: {epub_path}")
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    headings = [heading for chapter in chapters for heading in chapter.headings]
    assert headings
    assert any("Ⅰ" in heading or "Ⅱ" in heading for heading in headings)
    assert any("女" in heading for heading in headings)


def test_extract_ruby_pairs_and_strip_rt() -> None:
    html = "<p><ruby>漢字<rt>かんじ</rt></ruby>です。</p>".encode("utf-8")
    assert epub_util.html_to_text(html) == "漢字です。"
    assert epub_util.extract_ruby_pairs(html) == [("漢字", "かんじ")]


def test_html_to_text_with_ruby_spans() -> None:
    html = "<p>前<ruby>漢字<rt>かんじ</rt></ruby>後</p>".encode("utf-8")
    text, spans = epub_util.html_to_text_with_ruby(html)
    assert text == "前漢字後"
    assert spans == [
        {"start": 1, "end": 3, "base": "漢字", "reading": "かんじ"}
    ]


def test_html_to_text_inserts_section_break_for_rare_class() -> None:
    html = "".join("<p class='main'>本文</p>" for _ in range(10))
    html += "<p class='alt'>区切り</p>"
    html += "<p class='main'>続き</p>"
    text = epub_util.html_to_text(html.encode("utf-8"))
    assert f"\n\n{SECTION_BREAK}\n\n" in text
