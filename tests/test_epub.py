from pathlib import Path

from ebooklib import epub

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


def _write_structural_heading_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("structural-heading-book")
    book.set_title("Structural Heading Sample")
    book.set_language("ja")

    style = epub.EpubItem(
        uid="style",
        file_name="style.css",
        media_type="text/css",
        content=(
            ".book-title { font-weight: bold; }\n"
            ".class_s2 { text-align: left; }\n"
            ".class_s4xe2 { font-weight: bold; }\n"
        ).encode("utf-8"),
    )
    book.add_item(style)

    chapters = []
    for idx, number in enumerate(("１", "２", "３", "４"), start=1):
        chapter = epub.EpubHtml(
            title=f"Chapter {idx}",
            file_name=f"structural-{idx}.xhtml",
            lang="ja",
        )
        chapter.add_item(style)
        chapter.content = f"""
        <html>
          <body>
            <div class="book-title">
              <div class="h-indent-4em">
                <p class="calibre"><span class="gfont2">　一章　つれ込んだら無くなった</span></p>
              </div>
            </div>
            <p class="class_s2"><span class="class_s4xe2">{number}</span></p>
            <p class="class_s2">本文{idx}です。</p>
          </body>
        </html>
        """
        book.add_item(chapter)
        chapters.append(chapter)

    book.toc = tuple(chapters)
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def _write_filename_title_fallback_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("filename-fallback-book")
    book.set_title("Filename Fallback Sample")
    book.set_language("ja")

    chapter = epub.EpubHtml(
        title="chapter-001.xhtml",
        file_name="chapter-001.xhtml",
        lang="ja",
    )
    chapter.content = (
        "<html><body><p>これは先頭の本文です。次の文です。</p></body></html>"
    )
    book.add_item(chapter)
    book.toc = (chapter,)
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def _write_split_title_page_epub(
    epub_path: Path, next_title: str, first_line: str = "「おじさん、ちょっとおじさん」"
) -> None:
    book = epub.EpubBook()
    book.set_identifier("split-title-page-book")
    book.set_title("Split Title Page Sample")
    book.set_language("ja")

    title_page = epub.EpubHtml(
        title="title.xhtml",
        file_name="title.xhtml",
        lang="ja",
    )
    title_page.content = "<html><body><p>森の奥</p></body></html>"

    body_page = epub.EpubHtml(
        title=next_title,
        file_name="body.xhtml",
        lang="ja",
    )
    body_lines = [
        first_line,
        "だれかが呼びかけ、肩をゆさぶっている。",
    ]
    body_lines.extend("本文です。" for _ in range(220))
    body_html = "".join(f"<p>{line}</p>" for line in body_lines)
    body_page.content = f"<html><body>{body_html}</body></html>"

    book.add_item(title_page)
    book.add_item(body_page)
    book.toc = (title_page, body_page)
    book.spine = ["nav", title_page, body_page]
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


def test_extract_chapters_detects_structural_numeric_heading_classes(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "structural-heading.epub"
    _write_structural_heading_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    headings = [heading for chapter in chapters for heading in chapter.headings]
    heading_categories = {}
    for chapter in chapters:
        heading_categories.update(chapter.heading_categories)
    assert headings
    assert any("つれ込んだら無くなった" in heading for heading in headings)
    assert any(heading == "１" for heading in headings)
    assert any(heading == "４" for heading in headings)
    assert heading_categories.get("１") == "section"
    assert heading_categories.get("４") == "section"
    assert any(category == "title" for category in heading_categories.values())


def test_extract_chapters_uses_body_text_title_when_toc_title_is_filename(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "filename-title-fallback.epub"
    _write_filename_title_fallback_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert chapters
    title = chapters[0].title
    assert "これは先頭の本文です。" in title
    assert not title.startswith("Chapter ")


def test_extract_chapters_merges_title_only_page_into_following_body(tmp_path: Path) -> None:
    epub_path = tmp_path / "split-title-page.epub"
    _write_split_title_page_epub(epub_path, next_title="body.xhtml")
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=False)
    assert len(chapters) == 1
    assert chapters[0].title == "森の奥"
    assert chapters[0].text.startswith("「おじさん、ちょっとおじさん」")
    assert "本文です。" in chapters[0].text


def test_extract_chapters_keeps_title_only_page_when_next_starts_with_structural_heading(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "split-title-page-structural.epub"
    _write_split_title_page_epub(epub_path, next_title="body.xhtml", first_line="１")
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=False)
    assert len(chapters) == 2
    assert chapters[0].title == "森の奥"
    assert chapters[1].title == "１"


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
