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
            b".book-title { font-weight: bold; }\n"
            b".class_s2 { text-align: left; }\n"
            b".class_s4xe2 { font-weight: bold; }\n"
        ),
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


def _write_toc_title_page_then_body_epub(epub_path: Path, toc_title: str) -> None:
    book = epub.EpubBook()
    book.set_identifier("toc-title-then-body-book")
    book.set_title("TOC Title Then Body")
    book.set_language("ja")

    title_page = epub.EpubHtml(
        title=toc_title,
        file_name="title-page.xhtml",
        lang="ja",
    )
    title_page.content = "<html><body><img src='missing.jpg' alt=''/></body></html>"

    body_page = epub.EpubHtml(
        title="body.xhtml",
        file_name="body.xhtml",
        lang="ja",
    )
    body_lines = ["１", "本文です。"]
    body_lines.extend("本文です。" for _ in range(80))
    body_html = "".join(f"<p>{line}</p>" for line in body_lines)
    body_page.content = f"<html><body>{body_html}</body></html>"

    book.add_item(title_page)
    book.add_item(body_page)
    book.toc = (title_page,)
    book.spine = ["nav", title_page, body_page]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def _write_inline_contents_toc_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("inline-contents-book")
    book.set_title("Inline Contents Book")
    book.set_language("ja")

    volume = epub.EpubHtml(
        title="第1巻",
        file_name="volume.xhtml",
        lang="ja",
    )
    volume.content = "<html><body><img src='missing.jpg' alt=''/></body></html>"

    contents = epub.EpubHtml(
        title="contents.xhtml",
        file_name="contents.xhtml",
        lang="ja",
    )
    contents.content = """
    <html><body>
      <h1>CONTENTS</h1>
      <p><a href="title-1.xhtml">第一章 はじまり</a></p>
      <p><a href="title-2.xhtml">第二章 つづき</a></p>
      <p><a href="afterword.xhtml">あとがき</a></p>
    </body></html>
    """

    title_1 = epub.EpubHtml(title="t1", file_name="title-1.xhtml", lang="ja")
    title_1.content = "<html><body><img src='missing.jpg' alt=''/></body></html>"

    body_1 = epub.EpubHtml(title="body-1.xhtml", file_name="body-1.xhtml", lang="ja")
    body_1.content = "<html><body><p>１</p><p>本文です。</p></body></html>"

    title_2 = epub.EpubHtml(title="t2", file_name="title-2.xhtml", lang="ja")
    title_2.content = "<html><body><img src='missing.jpg' alt=''/></body></html>"

    body_2 = epub.EpubHtml(title="body-2.xhtml", file_name="body-2.xhtml", lang="ja")
    body_2.content = "<html><body><p>１</p><p>本文です。</p></body></html>"

    afterword = epub.EpubHtml(title="あとがき", file_name="afterword.xhtml", lang="ja")
    afterword.content = "<html><body><p>あとがき</p><p>本文です。</p></body></html>"

    for item in (volume, contents, title_1, body_1, title_2, body_2, afterword):
        book.add_item(item)

    book.toc = (volume,)
    book.spine = ["nav", volume, contents, title_1, body_1, title_2, body_2, afterword]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def _write_toc_split_fragment_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("toc-split-fragment-book")
    book.set_title("TOC Split Fragment Book")
    book.set_language("ja")

    chapter_1_start = epub.EpubHtml(
        title="第一章",
        file_name="chapter-1-start.xhtml",
        lang="ja",
    )
    chapter_1_start.content = "<html><body><p>第一章</p><p>冒頭です。</p></body></html>"

    chapter_1_cont = epub.EpubHtml(
        title="chapter-1-cont.xhtml",
        file_name="chapter-1-cont.xhtml",
        lang="ja",
    )
    chapter_1_cont.content = (
        "<html><body><p>ここから続きです。</p>"
        + "".join("<p>本文です。</p>" for _ in range(140))
        + "</body></html>"
    )

    chapter_2_start = epub.EpubHtml(
        title="第二章",
        file_name="chapter-2-start.xhtml",
        lang="ja",
    )
    chapter_2_start.content = "<html><body><p>第二章</p><p>次章です。</p></body></html>"

    for item in (chapter_1_start, chapter_1_cont, chapter_2_start):
        book.add_item(item)
    book.toc = (chapter_1_start, chapter_2_start)
    book.spine = ["nav", chapter_1_start, chapter_1_cont, chapter_2_start]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def _write_toc_missing_section_starts_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("toc-missing-section-starts-book")
    book.set_title("TOC Missing Section Starts Book")
    book.set_language("ja")

    chapter_1_start = epub.EpubHtml(
        title="第一章",
        file_name="chapter-1-start.xhtml",
        lang="ja",
    )
    chapter_1_start.content = "<html><body><p>第一章</p><p>冒頭です。</p></body></html>"

    chapter_1_section = epub.EpubHtml(
        title="chapter-1-section.xhtml",
        file_name="chapter-1-section.xhtml",
        lang="ja",
    )
    chapter_1_section.content = (
        "<html><body><p>１話 途中見出し</p>"
        + "".join("<p>節本文です。</p>" for _ in range(120))
        + "</body></html>"
    )

    chapter_2_start = epub.EpubHtml(
        title="第二章",
        file_name="chapter-2-start.xhtml",
        lang="ja",
    )
    chapter_2_start.content = "<html><body><p>第二章</p><p>次章です。</p></body></html>"

    for item in (chapter_1_start, chapter_1_section, chapter_2_start):
        book.add_item(item)
    # TOC omits chapter_1_section on purpose.
    book.toc = (chapter_1_start, chapter_2_start)
    book.spine = ["nav", chapter_1_start, chapter_1_section, chapter_2_start]
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
    html = b"""
    <html>
      <body>
        <h2>Part I</h2>
        <p><span class="subtitle">- Intro -</span></p>
      </body>
    </html>
    """
    headings = epub_util._extract_html_headings(html)
    assert "Part I" in headings
    assert "- Intro -" in headings


def test_extract_html_headings_ignores_plain_structural_blocks() -> None:
    """Plain text with no heading markup should not be detected as headings."""
    html = """
    <html>
      <body>
        <p class="calibre2">プロローグ</p>
        <p class="calibre2">これは本文です。しっかり長い段落になっています。</p>
        <p class="calibre2">１</p>
        <p class="calibre2">次の本文です。ここも長い段落です。</p>
        <p class="calibre2">２</p>
        <p class="calibre2">さらに本文が続きます。段落です。</p>
      </body>
    </html>
    """.encode()
    headings = epub_util._extract_html_headings(html)
    assert "プロローグ" not in headings
    assert "１" not in headings
    assert "２" not in headings


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
    assert chapters[0].text.startswith("森の奥\n\n「おじさん、ちょっとおじさん」")
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


def test_extract_chapters_inherits_toc_title_from_empty_title_page(tmp_path: Path) -> None:
    epub_path = tmp_path / "toc-title-then-body.epub"
    _write_toc_title_page_then_body_epub(epub_path, toc_title="第一章 さくら荘へようこそ")
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert len(chapters) == 1
    assert chapters[0].title == "第一章 さくら荘へようこそ"
    assert chapters[0].text.startswith("第一章 さくら荘へようこそ\n\n１")


def test_extract_chapters_does_not_inherit_non_content_toc_title(tmp_path: Path) -> None:
    epub_path = tmp_path / "toc-non-content-then-body.epub"
    _write_toc_title_page_then_body_epub(epub_path, toc_title="目次")
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert len(chapters) == 1
    assert chapters[0].title == "１"
    assert chapters[0].text.startswith("１")


def test_extract_chapters_uses_inline_contents_links_for_chapter_titles(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "inline-contents.epub"
    _write_inline_contents_toc_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    titles = [chapter.title for chapter in chapters]
    assert "第一章 はじまり" in titles
    assert "第二章 つづき" in titles
    target = next(ch for ch in chapters if ch.title == "第一章 はじまり")
    assert target.text.startswith("第一章 はじまり\n\n１")


def test_extract_chapters_merges_toc_split_fragments_from_spine(tmp_path: Path) -> None:
    epub_path = tmp_path / "toc-split-fragment.epub"
    _write_toc_split_fragment_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert len(chapters) == 2
    assert chapters[0].title == "第一章"
    assert "ここから続きです。" in chapters[0].text
    assert chapters[1].title == "第二章"


def test_extract_chapters_keeps_non_toc_structural_section_starts(tmp_path: Path) -> None:
    epub_path = tmp_path / "toc-missing-section-starts.epub"
    _write_toc_missing_section_starts_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    titles = [chapter.title for chapter in chapters]
    assert "１話 途中見出し" in titles
    chapter_1 = next(ch for ch in chapters if ch.title == "第一章")
    section = next(ch for ch in chapters if ch.title == "１話 途中見出し")
    assert "節本文です。" not in chapter_1.text
    assert "節本文です。" in section.text


def test_extract_ruby_pairs_and_strip_rt() -> None:
    html = "<p><ruby>漢字<rt>かんじ</rt></ruby>です。</p>".encode()
    assert epub_util.html_to_text(html) == "漢字です。"
    assert epub_util.extract_ruby_pairs(html) == [("漢字", "かんじ")]


def test_html_to_text_with_ruby_spans() -> None:
    html = "<p>前<ruby>漢字<rt>かんじ</rt></ruby>後</p>".encode()
    text, spans = epub_util.html_to_text_with_ruby(html)
    assert text == "前漢字後"
    assert spans == [
        {"start": 1, "end": 3, "base": "漢字", "reading": "かんじ"}
    ]


def test_html_to_text_with_ruby_drops_emphasis_mark_reading() -> None:
    # Ruby abused for emphasis: <rt> holds 傍点 marks (ヽ) rather than a reading.
    # The base text must survive as plain text with no span recorded.
    html = (
        "<p>非行の<ruby>原因<rt>ヽヽ</rt></ruby>を"
        "<ruby>分析<rt>かい</rt></ruby>する。</p>"
    ).encode()
    text, spans = epub_util.html_to_text_with_ruby(html)
    assert text == "非行の原因を分析する。"
    # Only the genuine reading is kept; the emphasis-mark ruby is dropped.
    assert spans == [{"start": 6, "end": 8, "base": "分析", "reading": "かい"}]


def test_html_to_text_renders_blank_base_ruby_as_reading() -> None:
    html = "<p>よろしく『<ruby>　　<rt>くうはく</rt></ruby>』さん</p>".encode()
    assert epub_util.html_to_text_with_ruby(html) == ("よろしく『くうはく』さん", [])
    assert epub_util.html_to_text(html) == "よろしく『くうはく』さん"


def test_html_to_text_drops_reading_of_ruby_without_base() -> None:
    html = "<p>全部貰<ruby><rt>もら</rt></ruby>い物やから</p>".encode()
    assert epub_util.html_to_text_with_ruby(html) == ("全部貰い物やから", [])
    assert epub_util.html_to_text(html) == "全部貰い物やから"


def test_html_to_text_with_ruby_skips_comments_inside_ruby_base() -> None:
    html = "<p><ruby>Ｃ<!--kisyu=6-->Ｏ<rt>シーオー</rt></ruby>だ</p>".encode()
    text, spans = epub_util.html_to_text_with_ruby(html)
    assert text == "ＣＯだ"
    assert spans == [{"start": 0, "end": 2, "base": "ＣＯ", "reading": "シーオー"}]


def test_html_to_text_with_ruby_keeps_gaiji_alt_inside_ruby_base() -> None:
    html = (
        "<p><ruby><rb>酢</rb><rt>す</rt><rb><img src='g.png' alt='蓮'/></rb>"
        "<rt>ばす</rt></ruby>に卵焼</p>"
    ).encode()
    text, spans = epub_util.html_to_text_with_ruby(html)
    assert text == "酢蓮に卵焼"
    assert spans == [{"start": 0, "end": 2, "base": "酢蓮", "reading": "すばす"}]


def test_html_to_text_inserts_section_break_for_rare_class() -> None:
    html = "".join("<p class='main'>本文</p>" for _ in range(10))
    html += "<p class='alt'>区切り</p>"
    html += "<p class='main'>続き</p>"
    text = epub_util.html_to_text(html.encode("utf-8"))
    assert f"\n\n{SECTION_BREAK}\n\n" in text


def test_html_to_text_renders_image_alt_text() -> None:
    html = (
        "<html><body>"
        "<p id='toc-1'><img src='ch01.jpg' alt='第一章 はじまり'/></p>"
        "<p>本文の最初の段落です。</p>"
        "</body></html>"
    ).encode()
    text = epub_util.html_to_text(html)
    assert text.startswith("第一章 はじまり")
    assert "本文の最初の段落です。" in text


def test_html_to_text_skips_empty_alt_image() -> None:
    html = (
        "<html><body>"
        "<p><img src='deco.jpg' alt=''/></p>"
        "<p>本文です。</p>"
        "</body></html>"
    ).encode()
    text = epub_util.html_to_text(html)
    assert text == "本文です。"


def _write_image_title_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("image-title-book")
    book.set_title("Image Title Book")
    book.set_language("ja")

    chapter = epub.EpubHtml(
        title="プロローグ",
        file_name="prologue.xhtml",
        lang="ja",
    )
    chapter.content = (
        "<html><body>"
        "<p id='toc-1'><img src='img.jpg' alt='プロローグ'/></p>"
        "<p>クリスマスイブの夜の話。</p>"
        + "".join("<p>本文です。</p>" for _ in range(20))
        + "</body></html>"
    )
    book.add_item(chapter)
    book.toc = (chapter,)
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_renders_image_alt_as_chapter_heading_text(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "image-title.epub"
    _write_image_title_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert len(chapters) == 1
    assert chapters[0].title == "プロローグ"
    assert chapters[0].text.startswith("プロローグ")
    assert "クリスマスイブの夜の話。" in chapters[0].text


def _write_toc_chapter_with_intro_followup_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("toc-followup-book")
    book.set_title("TOC Chapter Followup Book")
    book.set_language("ja")

    chapter_title_page = epub.EpubHtml(
        title="第一章 幼い頃の話",
        file_name="ch1-title.xhtml",
        lang="ja",
    )
    chapter_title_page.content = (
        "<html><body><p>第一章 幼い頃の話</p></body></html>"
    )

    chapter_intro = epub.EpubHtml(
        title="ch1-intro.xhtml",
        file_name="ch1-intro.xhtml",
        lang="ja",
    )
    chapter_intro.content = (
        "<html><body>"
        "<p>ある友人からこんな話を聞いたことがある。</p>"
        "<p>幼い頃、いつも天井の隅っこに天使が飛んでいたんだ。</p>"
        "</body></html>"
    )

    first_story = epub.EpubHtml(
        title="第一話 白蛇の夢",
        file_name="story-1.xhtml",
        lang="ja",
    )
    first_story.content = (
        "<html><body>"
        "<p>第一話 白蛇の夢</p>"
        "<p>私が小学生の頃、不思議な話を聞いたことがある。</p>"
        + "".join("<p>本文です。</p>" for _ in range(40))
        + "</body></html>"
    )

    for item in (chapter_title_page, chapter_intro, first_story):
        book.add_item(item)
    book.toc = (chapter_title_page, first_story)
    book.spine = ["nav", chapter_title_page, chapter_intro, first_story]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_folds_non_toc_intro_into_chapter(tmp_path: Path) -> None:
    epub_path = tmp_path / "toc-followup.epub"
    _write_toc_chapter_with_intro_followup_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    titles = [ch.title for ch in chapters]
    assert "第一章 幼い頃の話" in titles
    assert "第一話 白蛇の夢" in titles
    chapter_one = next(ch for ch in chapters if ch.title == "第一章 幼い頃の話")
    assert "ある友人からこんな話を聞いたことがある。" in chapter_one.text
    assert "幼い頃、いつも天井の隅っこに天使が飛んでいたんだ。" in chapter_one.text
    first_story = next(ch for ch in chapters if ch.title == "第一話 白蛇の夢")
    assert "ある友人" not in first_story.text


def _write_pagelist_nav_epub(epub_path: Path) -> None:
    """Numbered chapters wrapped in id-carrying sections, plus a page-list nav
    whose numeric anchors target those section ids."""
    book = epub.EpubBook()
    book.set_identifier("pagelist-book")
    book.set_title("Pagelist Sample")
    book.set_language("ja")

    chapters = []
    for idx in range(1, 4):
        chapter = epub.EpubHtml(
            title=str(idx),
            file_name=f"ch_{idx:03d}.xhtml",
            lang="ja",
        )
        chapter.content = (
            f'<section id="{idx:02d}-{idx}">'
            f"<h2>{idx}</h2><p>短い本文{idx}です。</p>"
            "</section>"
        )
        book.add_item(chapter)
        chapters.append(chapter)

    page_list = epub.EpubHtml(
        title="pages", file_name="pages.xhtml", lang="ja"
    )
    page_list.content = (
        '<nav epub:type="page-list" hidden="hidden"><ol>'
        + "".join(
            f'<li><a href="ch_{idx:03d}.xhtml#{idx:02d}-{idx}">{idx}</a></li>'
            for idx in range(1, 4)
        )
        + "</ol></nav>"
    )
    book.add_item(page_list)

    book.toc = tuple(chapters)
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_ignores_pagelist_nav_anchors(tmp_path: Path) -> None:
    epub_path = tmp_path / "pagelist-nav.epub"
    _write_pagelist_nav_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    titles = [ch.title for ch in chapters]
    assert titles == ["1", "2", "3"]
    for idx, chapter in enumerate(chapters, start=1):
        assert f"短い本文{idx}です。" in chapter.text


def _write_inline_page_links_epub(epub_path: Path) -> None:
    """Numeric page links outside any <nav> element, targeting each chapter's
    root section id; the sections hold the entire chapter text."""
    book = epub.EpubBook()
    book.set_identifier("inline-page-links-book")
    book.set_title("Inline Page Links Sample")
    book.set_language("ja")

    body = "この章の本文はとても長く、注釈ではありません。" * 8
    chapters = []
    for idx in range(1, 4):
        chapter = epub.EpubHtml(
            title=str(idx),
            file_name=f"ch-{idx}.xhtml",
            lang="ja",
        )
        chapter.content = (
            f'<section id="s{idx}"><h2>{idx}</h2><p>{body}</p></section>'
        )
        book.add_item(chapter)
        chapters.append(chapter)

    pages = epub.EpubHtml(title="pages", file_name="pages.xhtml", lang="ja")
    pages.content = "".join(
        f'<p><a href="ch-{idx}.xhtml#s{idx}">{idx}</a></p>' for idx in range(1, 4)
    )
    book.add_item(pages)

    book.toc = tuple(chapters)
    book.spine = ["nav", pages, *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_keeps_sections_targeted_by_inline_page_links(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "inline-page-links.epub"
    _write_inline_page_links_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    titles = [ch.title for ch in chapters]
    assert titles == ["1", "2", "3"]
    for chapter in chapters:
        assert "この章の本文はとても長く" in chapter.text


def _write_footnote_backref_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("footnote-backref-book")
    book.set_title("Footnote Backref Sample")
    book.set_language("ja")

    chapter = epub.EpubHtml(title="一章", file_name="ch-1.xhtml", lang="ja")
    chapter.content = (
        "<h2>一章</h2>"
        '<p>これは本文です<a href="notes.xhtml#note-1">1</a>。本文の続きです。</p>'
    )
    notes = epub.EpubHtml(title="注記", file_name="notes.xhtml", lang="ja")
    notes.content = (
        "<h2>注記</h2>"
        '<p id="note-1">1 これは注釈のテキストです。</p>'
    )
    book.add_item(chapter)
    book.add_item(notes)
    book.toc = (chapter, notes)
    book.spine = ["nav", chapter, notes]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_still_strips_footnote_bodies_and_markers(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "footnote-backref.epub"
    _write_footnote_backref_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    chapter_one = next(ch for ch in chapters if ch.title == "一章")
    assert "これは本文です" in chapter_one.text
    assert "注釈のテキスト" not in chapter_one.text
    assert "1" not in chapter_one.text.replace("一章", "")
    for chapter in chapters:
        assert "注釈のテキスト" not in chapter.text


def _write_untitled_scene_after_contents_epub(epub_path: Path, scene_lines: int) -> None:
    book = epub.EpubBook()
    book.set_identifier("untitled-scene-after-contents-book")
    book.set_title("Untitled Scene After Contents")
    book.set_language("ja")

    contents = epub.EpubHtml(title="目次", file_name="contents.xhtml", lang="ja")
    contents.content = (
        "<html><body><p>目次</p>"
        "<p><a href='chapter-1.xhtml'>～１敗目～　断じて下心はありません</a></p>"
        "<p><a href='chapter-2.xhtml'>～２敗目～　肩の重みに揺らされて</a></p>"
        "</body></html>"
    )

    scene = epub.EpubHtml(title="", file_name="scene.xhtml", lang="ja")
    scene.content = (
        "<html><body><p>　放課後のツワブキ高校図書室。</p>"
        + "".join("<p>　図書室は閑散としている。</p>" for _ in range(scene_lines))
        + "</body></html>"
    )

    chapters = []
    for idx, title in enumerate(
        ("～１敗目～　断じて下心はありません", "～２敗目～　肩の重みに揺らされて"), start=1
    ):
        chapter = epub.EpubHtml(title=title, file_name=f"chapter-{idx}.xhtml", lang="ja")
        chapter.content = (
            f"<html><body><h1>{title}</h1>"
            + "".join(f"<p>　本文{idx}です。</p>" for _ in range(200))
            + "</body></html>"
        )
        chapters.append(chapter)

    for item in (contents, scene, *chapters):
        book.add_item(item)
    book.toc = (contents, *chapters)
    book.spine = [contents, scene, *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def _assert_scene_survives_contents_drop(chapters: list[epub_util.Chapter]) -> None:
    titles = [ch.title for ch in chapters]
    contents = next(ch for ch in chapters if ch.title == "目次")
    assert "放課後のツワブキ高校図書室。" not in contents.text
    scene = next(ch for ch in chapters if "放課後のツワブキ高校図書室。" in ch.text)
    assert scene.title not in {"目次", "～１敗目～　断じて下心はありません"}
    assert titles.index(scene.title) == titles.index("目次") + 1


def test_extract_chapters_keeps_long_untitled_scene_out_of_contents_page(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "long-untitled-scene.epub"
    _write_untitled_scene_after_contents_epub(epub_path, scene_lines=120)
    book = epub_util.read_epub(epub_path)
    _assert_scene_survives_contents_drop(epub_util.extract_chapters(book, prefer_toc=True))


def test_extract_chapters_recovers_short_untitled_scene_after_contents_page(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "short-untitled-scene.epub"
    _write_untitled_scene_after_contents_epub(epub_path, scene_lines=3)
    book = epub_util.read_epub(epub_path)
    _assert_scene_survives_contents_drop(epub_util.extract_chapters(book, prefer_toc=True))


def _write_unlisted_structural_section_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("unlisted-structural-section-book")
    book.set_title("Unlisted Structural Section")
    book.set_language("ja")

    items = []
    for name, heading, body in (
        ("chapter-1.xhtml", "第一章", "最初の本文です。"),
        ("chapter-2.xhtml", "第二章", "見出しの本文です。"),
        ("chapter-2-cont.xhtml", "", "その続きです。"),
        ("chapter-3.xhtml", "第三章", "最後の本文です。"),
    ):
        lines = 3 if name.startswith("chapter-2") else 200
        item = epub.EpubHtml(title=heading or name, file_name=name, lang="ja")
        item.content = (
            "<html><body>"
            + (f"<p>{heading}</p>" if heading else "")
            + "".join(f"<p>{body}</p>" for _ in range(lines))
            + "</body></html>"
        )
        items.append(item)
        book.add_item(item)
    # The TOC omits 第二章 on purpose.
    book.toc = (items[0], items[3])
    book.spine = items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_recovers_section_after_unlisted_structural_heading(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "unlisted-structural-section.epub"
    _write_unlisted_structural_section_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert [ch.title for ch in chapters] == ["第一章", "第二章", "第三章"]
    assert "見出しの本文です。" in chapters[1].text
    assert "その続きです。" in chapters[1].text
    assert "見出しの本文です。" not in chapters[0].text


def _write_calibre_split_series_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("calibre-split-series-book")
    book.set_title("Calibre Split Series")
    book.set_language("ja")

    items = []
    for idx in range(3):
        item = epub.EpubHtml(
            uid=f"split{idx}",
            title=f"part0008_split_00{idx}.xhtml",
            file_name=f"part0008_split_00{idx}.xhtml",
            lang="ja",
        )
        heading = "<h1>第一話</h1>" if idx == 0 else ""
        item.content = (
            f"<html><body>{heading}"
            + "".join(f"<p>断片{idx}の本文です。</p>" for _ in range(40))
            + "</body></html>"
        )
        items.append(item)
    ending = epub.EpubHtml(uid="ending", title="第二話", file_name="part0009.xhtml", lang="ja")
    ending.content = (
        "<html><body><h1>第二話</h1>"
        + "".join("<p>最後の本文です。</p>" for _ in range(40))
        + "</body></html>"
    )
    for item in (*items, ending):
        book.add_item(item)
    book.toc = (
        epub.Link("part0008_split_000.xhtml", "第一話", "ch1"),
        epub.Link("part0009.xhtml", "第二話", "ch2"),
    )
    book.spine = [*items, ending]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_does_not_repeat_merged_split_series(tmp_path: Path) -> None:
    epub_path = tmp_path / "calibre-split-series.epub"
    _write_calibre_split_series_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert [ch.title for ch in chapters] == ["第一話", "第二話"]
    for idx in range(3):
        assert chapters[0].text.count(f"断片{idx}の本文です。") == 40


def _write_image_only_story_before_title_page_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("image-only-story-book")
    book.set_title("Image Only Story")
    book.set_language("ja")

    pages = []
    for name in ("picture-title.xhtml", "picture-1.xhtml", "story-title.xhtml"):
        page = epub.EpubHtml(title=name, file_name=name, lang="ja")
        page.content = "<html><body><img src='missing.jpg' alt=''/></body></html>"
        pages.append(page)
    body = epub.EpubHtml(title="story-body.xhtml", file_name="story-body.xhtml", lang="ja")
    body.content = (
        "<html><body>"
        + "".join("<p>　物語の本文です。</p>" for _ in range(80))
        + "</body></html>"
    )
    for item in (*pages, body):
        book.add_item(item)
    book.toc = (
        epub.Link("picture-title.xhtml", "残酷童話　うつくし姫", "picture"),
        epub.Link("story-title.xhtml", "第零話　あせろらボナペティ", "story"),
    )
    book.spine = [*pages, body]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_does_not_inherit_title_across_another_toc_entry(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "image-only-story.epub"
    _write_image_only_story_before_title_page_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert [ch.title for ch in chapters] == ["第零話　あせろらボナペティ"]


def test_read_epub_recovers_toc_from_malformed_ncx(tmp_path: Path) -> None:
    import zipfile

    book = epub.EpubBook()
    book.set_identifier("malformed-ncx-book")
    book.set_title("Malformed NCX")
    book.set_language("ja")
    chapters = []
    for idx, title in enumerate(("序章", "第一章", "終章"), start=1):
        chapter = epub.EpubHtml(title=title, file_name=f"chapter-{idx}.xhtml", lang="ja")
        chapter.content = f"<h1>{title}</h1><p>本文{idx}です。</p>"
        book.add_item(chapter)
        chapters.append(chapter)
    book.toc = tuple(chapters)
    book.spine = chapters
    book.add_item(epub.EpubNcx())
    source = tmp_path / "source.epub"
    epub.write_epub(str(source), book)

    broken = tmp_path / "broken-ncx.epub"
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(broken, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith(".ncx"):
                data = data.replace(
                    b"<text>Malformed NCX</text>", "<text>業物語 <物語></text>".encode(), 1
                )
            zout.writestr(info, data)
    book = epub_util.read_epub(broken)
    titles = [entry.title for entry in epub_util.build_toc_entries(book)]
    assert titles == ["序章", "第一章", "終章"]
    assert [ch.title for ch in epub_util.extract_chapters(book, prefer_toc=True)] == titles


def _write_numbered_contents_link_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("numbered-contents-link-book")
    book.set_title("Numbered Contents Link")
    book.set_language("ja")

    contents = epub.EpubHtml(title="目次", file_name="contents.xhtml", lang="ja")
    contents.content = (
        "<html><body><p>目次</p>"
        "<p class='toc'><a href='story.xhtml#title1'>１</a>"
        "<a href='story.xhtml#title1'>中古品の子供たちを育てる人々</a></p>"
        "</body></html>"
    )
    story = epub.EpubHtml(title="story.xhtml", file_name="story.xhtml", lang="ja")
    story.content = (
        "<html><body><p id='title1'>The Used-Boy Raisers</p>"
        "<p>１．中古品の子供たちを育てる人々</p>"
        + "".join("<p>　卵のことで気落ちした二人の夫がいた。</p>" for _ in range(40))
        + "</body></html>"
    )
    for item in (contents, story):
        book.add_item(item)
    book.toc = (contents, epub.Link("story.xhtml", "１．中古品の子供たちを育てる人々", "s1"))
    book.spine = [contents, story]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_keeps_target_of_numbered_contents_link(tmp_path: Path) -> None:
    epub_path = tmp_path / "numbered-contents-link.epub"
    _write_numbered_contents_link_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    story = next(ch for ch in chapters if ch.title != "目次")
    assert story.text.startswith("The Used-Boy Raisers")


def _write_numbered_section_heading_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("numbered-section-heading-book")
    book.set_title("Numbered Section Heading")
    book.set_language("ja")

    contents = epub.EpubHtml(title="目次", file_name="contents.xhtml", lang="ja")
    contents.content = (
        "<html><body><p id='mokuji'>目次</p>"
        + "".join(
            f"<p><a href='section-{idx}.xhtml#heading-{idx}'><span>{label}</span></a></p>"
            for idx, label in ((1, "１"), (2, "２"))
        )
        + "</body></html>"
    )
    sections = []
    for idx, label in ((1, "１"), (2, "２")):
        section = epub.EpubHtml(title=label, file_name=f"section-{idx}.xhtml", lang="ja")
        section.content = (
            f"<html><body><p id='heading-{idx}'>　　<span>"
            f"<a href='contents.xhtml#mokuji'>{label}</a></span></p>"
            + "".join(f"<p>　第{idx}節の本文です。</p>" for _ in range(40))
            + "</body></html>"
        )
        sections.append(section)
    for item in (contents, *sections):
        book.add_item(item)
    book.toc = (contents, *sections)
    book.spine = [contents, *sections]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_keeps_numbered_heading_linked_from_contents(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "numbered-section-heading.epub"
    _write_numbered_section_heading_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    sections = [ch for ch in chapters if ch.title in {"１", "２"}]
    assert [ch.text.split("\n", 1)[0] for ch in sections] == ["１", "２"]


def _write_annotation_backlinks_epub(epub_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("annotation-backlinks-book")
    book.set_title("Annotation Backlinks")
    book.set_language("ja")

    stories = []
    for idx, title in enumerate(("畸形の天女", "女妖", "空とぶ円盤"), start=1):
        story = epub.EpubHtml(title=title, file_name=f"story-{idx}.xhtml", lang="ja")
        story.content = (
            f"<html><body><h1>{title}</h1>"
            + "".join("<p>　本文です。</p>" for _ in range(40))
            + f"<p>　活気がみちていた<a href='notes.xhtml#note{idx}'>（{idx}）</a>。</p>"
            + "</body></html>"
        )
        stories.append(story)
    notes = epub.EpubHtml(title="解題", file_name="notes.xhtml", lang="ja")
    notes.content = (
        "<html><body><h1>解題</h1>"
        + "".join(
            f"<p id='note{idx}'><a href='story-{idx}.xhtml#ref{idx}'>（{idx}）</a>"
            "連載された作品である。</p>"
            for idx in range(1, 4)
        )
        + "</body></html>"
    )
    for item in (*stories, notes):
        book.add_item(item)
    book.toc = (stories[0], notes)
    book.spine = [*stories, notes]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)


def test_extract_chapters_ignores_bracketed_annotation_backlinks_as_titles(
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "annotation-backlinks.epub"
    _write_annotation_backlinks_epub(epub_path)
    book = epub_util.read_epub(epub_path)
    titles = [ch.title for ch in epub_util.extract_chapters(book, prefer_toc=True)]
    assert titles == ["畸形の天女", "女妖", "空とぶ円盤", "解題"]
