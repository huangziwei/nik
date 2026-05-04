import json
from pathlib import Path

import pytest

from nik import tts as tts_util
from nik.text import SECTION_BREAK
from nik import voice as voice_util


_UNIDIC_DIR = tts_util._default_unidic_dir()
_UNIDIC_AVAILABLE = (_UNIDIC_DIR / "dicrc").exists()
requires_unidic = pytest.mark.skipif(
    not _UNIDIC_AVAILABLE,
    reason=f"UniDic not present at {_UNIDIC_DIR}; ruby-tagger tests need a real fugashi+UniDic install",
)


@pytest.fixture(autouse=True)
def _reset_first_token_separator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(tts_util.FIRST_TOKEN_SEPARATOR_ENV, raising=False)


def _default_first_token_separator() -> str:
    return tts_util._first_token_separator()


def test_make_chunk_spans_splits_japanese_sentences() -> None:
    text = "今日は良い天気です。明日も晴れるでしょう。"
    spans = tts_util.make_chunk_spans(text, max_chars=10, chunk_mode="japanese")
    assert len(spans) >= 2


def test_chunking_splits_on_commas_and_linebreaks() -> None:
    text = "私は、猫です。\n次の行です。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["私は、猫です。", "次の行です。"]


def test_chunking_splits_on_section_break() -> None:
    text = f"一。\n\n{SECTION_BREAK}\n\n二。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["一。", "二。"]


def test_chunking_splits_on_fullwidth_digits() -> None:
    text = "１\n\n本文です。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["１", "本文です。"]


def test_chunking_splits_on_space_with_digits() -> None:
    text = "さくら荘のペットな彼女　全13巻"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["さくら荘のペットな彼女", "全13巻"]


def test_chunking_packs_sentences_toward_max_chars() -> None:
    text = "一。二。三。四。"
    spans = tts_util.make_chunk_spans(text, max_chars=4, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["一。二。", "三。四。"]


def test_chunking_min_chars_merges_short_neighbors_across_linebreaks() -> None:
    text = "第一章\n\n1\n\n九月十日、火曜日の放課後。\n\nコトリと頭上で音がした。"
    spans = tts_util.make_chunk_spans(
        text, max_chars=0, chunk_mode="japanese", min_chars=12
    )
    chunks = [text[start:end] for start, end in spans]
    assert chunks == [
        "第一章\n\n1\n\n九月十日、火曜日の放課後。",
        "コトリと頭上で音がした。",
    ]


def test_chunking_min_chars_merges_trailing_short_chunk_with_previous() -> None:
    text = "九月十日、火曜日の放課後。\nコトリ"
    spans = tts_util.make_chunk_spans(
        text, max_chars=0, chunk_mode="japanese", min_chars=12
    )
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["九月十日、火曜日の放課後。\nコトリ"]


def test_chunking_keeps_balanced_quote_as_own_chunk() -> None:
    text = "彼は「いいえ。ありがとうございます」と言った。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["彼は", "「いいえ。ありがとうございます」", "と言った。"]


def test_chunking_keeps_inline_dialogue_quote_with_attribution_marker() -> None:
    text = "彼は「はい」と言った。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["彼は", "「はい」", "と言った。"]


def test_chunking_does_not_protect_inline_emphasis_quote() -> None:
    text = "医師の口にした「念のため」にさまざまな解釈をくわえた。"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == [text]


def test_chunking_does_not_protect_inline_emphasis_quote_after_comma() -> None:
    text = "硬いものをよく嚙かんで柔らかくするように、「須藤　花」で記憶を探る。"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == [text]


def test_chunking_keeps_dialogue_quote_after_comma_with_attribution() -> None:
    text = "彼は、「はい」と言った。"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["彼は、", "「はい」", "と言った。"]


def test_chunking_does_not_isolate_quote_adjacent_chunk_across_paragraph_boundary() -> None:
    text = "「夢みたいなことをね。ちょっと」\n\n病院だったんだ。昼過ぎだったんだ。"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["「夢みたいなことをね。ちょっと」", "病院だったんだ。昼過ぎだったんだ。"]


def test_chunking_splits_long_quote_when_over_max_chars() -> None:
    text = "「あいうえお。かきくけこ。さしすせそ。」"
    spans = tts_util.make_chunk_spans(text, max_chars=10, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["「あいうえお。", "かきくけこ。", "さしすせそ。」"]


def test_normalize_numbers_hatachi() -> None:
    assert tts_util._normalize_numbers("20歳を過ぎた") == "はたちを過ぎた"
    assert tts_util._normalize_numbers("二十歳") == "はたち"


def test_chunking_splits_on_commas_when_too_long() -> None:
    text = "これはとても長い文章なので、途中で区切る必要があります。"
    spans = tts_util.make_chunk_spans(text, max_chars=15, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert any("、" in chunk for chunk in chunks)


def test_chunking_keeps_leading_ellipsis_with_sentence() -> None:
    text = "……なんで八奈見がここにいるんだ。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["……なんで八奈見がここにいるんだ。"]


def test_chunking_splits_on_punct_only_line() -> None:
    text = "「……」\n\n「空太？」"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["「……」", "「空太？」"]


def test_prepare_tts_text_collapses_long_ellipsis_runs() -> None:
    assert tts_util.prepare_tts_text("…………") == "⋯"
    assert tts_util.prepare_tts_text("あ……い") == "あ……い"


def test_prepare_tts_pipeline_appends_chunk_tail_separator() -> None:
    pipeline = tts_util._prepare_tts_pipeline(
        "こんにちは",
        add_short_punct=False,
    )
    assert pipeline.prepared == "こんにちは "


def test_prepare_tts_pipeline_does_not_duplicate_chunk_tail_separator() -> None:
    sep = _default_first_token_separator()
    pipeline = tts_util._prepare_tts_pipeline(
        f"こんにちは{sep}",
        add_short_punct=False,
    )
    assert pipeline.prepared == "こんにちは "


def test_prepare_tts_pipeline_skips_chunk_tail_separator_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(tts_util.FIRST_TOKEN_SEPARATOR_ENV, "none")
    pipeline = tts_util._prepare_tts_pipeline(
        "こんにちは",
        add_short_punct=False,
    )
    assert pipeline.prepared == "こんにちは"


def test_prepare_tts_pipeline_strips_leading_chunk_separator() -> None:
    sep = _default_first_token_separator()
    pipeline = tts_util._prepare_tts_pipeline(
        f"『{sep}ばんがいへん{sep}』",
        add_short_punct=True,
    )
    assert pipeline.prepared == "ばんがいへん 。 "


def test_prepare_tts_pipeline_keeps_ellipsis_without_short_tail_full_stop() -> None:
    sep = _default_first_token_separator()
    pipeline = tts_util._prepare_tts_pipeline(
        "…………",
        add_short_punct=True,
    )
    assert pipeline.prepared == "⋯ "


def test_ellipsis_only_run_length_ignores_quotes_and_separator() -> None:
    sep = _default_first_token_separator()
    assert tts_util._ellipsis_only_run_length(f"「{sep}……{sep}」") == 2
    assert tts_util._ellipsis_only_run_length("……三ケ月前") == 0


def test_compute_chunk_pause_multipliers_uses_section_pause_for_explicit_headings() -> None:
    text = f"序章\n\n本文一。\n\n{SECTION_BREAK}\n\n本文二。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["序章", "本文一。", "本文二。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(
        text,
        spans,
        heading_lines=["序章"],
    )
    assert len(multipliers) == len(spans)
    assert multipliers[0] == 3
    assert multipliers[1] >= 3
    assert multipliers[2] == 1


def test_compute_chunk_pause_multipliers_keeps_same_category_pause_for_numbered_headings() -> None:
    text = "Ⅰ．\n\n１\n\n本文。\n\n２\n\n続き。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["Ⅰ．", "１", "本文。", "２", "続き。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(
        text,
        spans,
        heading_lines=["Ⅰ．", "１", "２"],
        heading_categories={"Ⅰ．": "section", "１": "section", "２": "section"},
    )
    assert multipliers[0] == 3
    assert multipliers[1] == 3
    assert multipliers[3] == 3


def test_compute_chunk_pause_multipliers_uses_title_pause_for_title_category() -> None:
    text = "序章\n\n本文一。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["序章", "本文一。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(
        text,
        spans,
        heading_lines=["序章"],
        heading_categories={"序章": "title"},
    )
    assert multipliers[0] >= 5
    assert multipliers[1] == 1


def test_compute_chunk_pause_multipliers_does_not_infer_heading_without_epub_data() -> None:
    text = "序章\n\n本文。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["序章", "本文。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1]


@pytest.mark.parametrize(
    "dialogue",
    [
        "「静かすぎる」",
        "「いい出してくれて、よかったわ」",
    ],
)
def test_compute_chunk_pause_multipliers_does_not_promote_dialogue_lines(
    dialogue: str,
) -> None:
    text = f"前の段落。\n\n{dialogue}\n\n次の段落。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["前の段落。", dialogue, "次の段落。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1, 1]


def test_compute_chunk_pause_multipliers_does_not_promote_split_dialogue_fragments() -> None:
    text = (
        "前の段落。\n\n"
        "「いいえ。もしお嫌じゃなかったら、一度どうぞ、のみにいらして下さい」\n\n"
        "「ありがとうございます。はずかしいわ」\n\n"
        "次の段落。"
    )
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)

    assert "「いいえ。もしお嫌じゃなかったら、一度どうぞ、のみにいらして下さい」" in chunks
    assert "「ありがとうございます。はずかしいわ」" in chunks
    for chunk, pause in zip(chunks, multipliers):
        if chunk in {
            "「いいえ。もしお嫌じゃなかったら、一度どうぞ、のみにいらして下さい」",
            "「ありがとうございます。はずかしいわ」",
        }:
            assert pause == 1


def test_compute_chunk_pause_multipliers_does_not_promote_dialogue_bridge_between_quotes() -> None:
    text = (
        "前文。\n\n"
        "「実をいうとちょっと面白いことがあったんだ。いいか。こいつはここだけの話だぜ」\n"
        "男は首を突き出しながら声をひそめ、\n"
        "「五年くらい前のことかな。博士の研究室にたびたび大きな荷物が送られてくるんだよ。\n\n"
        "後文。"
    )
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)

    for target in (
        "「実をいうとちょっと面白いことがあったんだ。いいか。こいつはここだけの話だぜ」",
        "男は首を突き出しながら声をひそめ、",
        "「五年くらい前のことかな。博士の研究室にたびたび大きな荷物が送られてくるんだよ。",
    ):
        idx = next(i for i, chunk in enumerate(chunks) if chunk == target)
        assert multipliers[idx] == 1


def test_compute_chunk_pause_multipliers_only_promotes_standalone_paragraph_chunks() -> None:
    text = (
        "前の段落。\n\n"
        "医師の口にした「念のため」にさまざまな解釈をくわえ、悲観的な想像をふくらませた。\n\n"
        "次の段落。"
    )
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == [
        "前の段落。",
        "医師の口にした「念のため」にさまざまな解釈をくわえ、悲観的な想像をふくらませた。",
        "次の段落。",
    ]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1, 1]


def test_compute_chunk_pause_multipliers_does_not_promote_sentence_continuation_line() -> None:
    text = (
        "どれだけ自分を、歪めていたんだろう。\n"
        "それはきっと悲しいことだ。なのに、\n"
        "──僕にとっては、能力で歪んでしまった君が、相麻菫なんだよ。"
    )
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == [
        "どれだけ自分を、歪めていたんだろう。",
        "それはきっと悲しいことだ。なのに、",
        "──僕にとっては、能力で歪んでしまった君が、相麻菫なんだよ。",
    ]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1, 1]


def test_compute_chunk_pause_multipliers_does_not_promote_continuation_before_dash_line() -> None:
    text = "それと、\n──応援してるから"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["それと、", "──応援してるから"]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1]


def test_compute_chunk_pause_multipliers_does_not_promote_continuation_before_plain_line() -> None:
    text = "歌詞を見るととても日本語とは思えない。これが実はヘブライ語だと提唱したのは神学博士で、翻訳すれば、\n聖前に主を讃えよ"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == [
        "歌詞を見るととても日本語とは思えない。これが実はヘブライ語だと提唱したのは神学博士で、翻訳すれば、",
        "聖前に主を讃えよ",
    ]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1]


def test_compute_chunk_pause_multipliers_keeps_section_break_after_continuation() -> None:
    text = f"それと、\n\n{SECTION_BREAK}\n\n次。"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["それと、", "次。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers[0] >= 3
    assert multipliers[1] == 1


def test_chunk_book_writes_manifest(tmp_path: Path) -> None:
    book_dir = tmp_path / "book"
    clean_dir = book_dir / "clean" / "chapters"
    clean_dir.mkdir(parents=True)
    chapter_path = clean_dir / "0001-chapter.txt"
    chapter_path.write_text("今日は良い天気です。", encoding="utf-8")

    toc = {
        "created_unix": 0,
        "metadata": {"title": "Sample"},
        "chapters": [
            {
                "index": 1,
                "title": "Chapter 1",
                "path": chapter_path.relative_to(book_dir).as_posix(),
            }
        ],
    }
    toc_path = book_dir / "clean" / "toc.json"
    toc_path.write_text(json.dumps(toc, ensure_ascii=False), encoding="utf-8")

    manifest = tts_util.chunk_book(book_dir)
    assert manifest["chapters"]
    assert (book_dir / "tts" / "manifest.json").exists()


def test_chunk_book_uses_explicit_epub_headings_for_pause_multipliers(
    tmp_path: Path,
) -> None:
    book_dir = tmp_path / "book"
    clean_dir = book_dir / "clean" / "chapters"
    clean_dir.mkdir(parents=True)
    chapter_path = clean_dir / "0001-chapter.txt"
    chapter_path.write_text("序章\n\n本文一。", encoding="utf-8")

    toc = {
        "created_unix": 0,
        "metadata": {"title": "Sample"},
        "chapters": [
            {
                "index": 1,
                "title": "chapter-001.xhtml",
                "headings": ["序章"],
                "heading_categories": {"序章": "section"},
                "path": chapter_path.relative_to(book_dir).as_posix(),
            }
        ],
    }
    toc_path = book_dir / "clean" / "toc.json"
    toc_path.write_text(json.dumps(toc, ensure_ascii=False), encoding="utf-8")

    manifest = tts_util.chunk_book(book_dir)
    chapter = manifest["chapters"][0]
    assert chapter.get("headings") == ["序章"]
    assert chapter.get("heading_categories") == {
        tts_util._normalize_heading_line_key("序章"): "section"
    }
    assert chapter["pause_multipliers"][0] == 3


def test_chunk_book_promotes_heading_matching_chapter_title_to_title_pause(
    tmp_path: Path,
) -> None:
    book_dir = tmp_path / "book"
    clean_dir = book_dir / "clean" / "chapters"
    clean_dir.mkdir(parents=True)
    chapter_path = clean_dir / "0001-chapter.txt"
    chapter_path.write_text("序章\n\n本文一。", encoding="utf-8")

    toc = {
        "created_unix": 0,
        "metadata": {"title": "Sample"},
        "chapters": [
            {
                "index": 1,
                "title": "序章",
                "headings": ["序章"],
                "path": chapter_path.relative_to(book_dir).as_posix(),
            }
        ],
    }
    toc_path = book_dir / "clean" / "toc.json"
    toc_path.write_text(json.dumps(toc, ensure_ascii=False), encoding="utf-8")

    manifest = tts_util.chunk_book(book_dir)
    chapter = manifest["chapters"][0]
    assert chapter.get("heading_categories") == {
        tts_util._normalize_heading_line_key("序章"): "title"
    }
    assert chapter["pause_multipliers"][0] >= 5


def test_chunk_book_does_not_infer_headings_without_epub_heading_metadata(
    tmp_path: Path,
) -> None:
    book_dir = tmp_path / "book"
    clean_dir = book_dir / "clean" / "chapters"
    clean_dir.mkdir(parents=True)
    chapter_path = clean_dir / "0001-chapter.txt"
    chapter_path.write_text("序章\n\n本文一。", encoding="utf-8")

    toc = {
        "created_unix": 0,
        "metadata": {"title": "Sample"},
        "chapters": [
            {
                "index": 1,
                "title": "chapter-001.xhtml",
                "path": chapter_path.relative_to(book_dir).as_posix(),
            }
        ],
    }
    toc_path = book_dir / "clean" / "toc.json"
    toc_path.write_text(json.dumps(toc, ensure_ascii=False), encoding="utf-8")

    manifest = tts_util.chunk_book(book_dir)
    chapter = manifest["chapters"][0]
    assert chapter.get("headings") == []
    assert chapter["pause_multipliers"][0] == 1


def test_prepare_manifest_applies_chapter_boundary_pause_multiplier(
    tmp_path: Path,
) -> None:
    chapters = [
        tts_util.ChapterInput(
            index=1,
            id="c1",
            title="Chapter 1",
            text="第一章本文。",
            path="clean/chapters/0001-c1.txt",
        ),
        tts_util.ChapterInput(
            index=2,
            id="c2",
            title="Chapter 2",
            text="第二章本文。",
            path="clean/chapters/0002-c2.txt",
        ),
    ]
    manifest, _chunks = tts_util._prepare_manifest(
        chapters=chapters,
        out_dir=tmp_path / "tts",
        voice="voice",
        max_chars=120,
        pad_ms=300,
        rechunk=True,
    )
    first = manifest["chapters"][0]["pause_multipliers"]
    second = manifest["chapters"][1]["pause_multipliers"]
    assert first[-1] >= 5
    assert second[-1] == 1


def test_apply_reading_overrides() -> None:
    text = "私は漢字と東京。"
    overrides = [
        {"base": "漢字", "reading": "かんじ"},
        {"base": "東京", "reading": "とうきょう"},
    ]
    assert (
        tts_util.apply_reading_overrides(text, overrides)
        == "私はかんじととうきょう。"
    )


def test_apply_reading_overrides_first_only() -> None:
    text = "山田山田"
    overrides = [{"base": "山田", "reading": "やまだ", "mode": "first"}]
    assert tts_util.apply_reading_overrides(text, overrides) == "やまだ山田"


def test_apply_reading_overrides_isolated_kanji() -> None:
    text = "「間」は間違い。"
    overrides = [{"base": "間", "reading": "ま", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == "「ま」は間違い。"


def test_apply_reading_overrides_kanji_boundary() -> None:
    text = "男の子と男女、人々と人は。"
    overrides = [
        {"base": "男", "reading": "おとこ", "mode": "kanji"},
        {"base": "人", "reading": "ひと", "mode": "kanji"},
    ]
    assert (
        tts_util.apply_reading_overrides(text, overrides)
        == "おとこの子と男女、人々とひとは。"
    )


def test_apply_reading_overrides_isolated_kanji_no_compound() -> None:
    text = "なんだ、これが天国か？"
    overrides = [{"base": "天", "reading": "てん", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == text


def test_apply_reading_overrides_isolated_kanji_no_okurigana() -> None:
    text = "好きにすれば"
    overrides = [{"base": "好", "reading": "こう", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == text


def test_apply_reading_overrides_isolated_kanji_allows_particle_boundary() -> None:
    text = "蟬の声と蟬は。"
    overrides = [{"base": "蟬", "reading": "せみ", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == "せみの声とせみは。"


def test_apply_reading_overrides_isolated_kanji_still_blocks_non_particle_kana() -> None:
    text = "蟬たち"
    overrides = [{"base": "蟬", "reading": "せみ", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == text


def test_apply_reading_overrides_isolated_kanji_allows_copula_boundary() -> None:
    text = "ちょっと期待してしまった帝だが、すぐに失言を悔やむ。"
    overrides = [{"base": "帝", "reading": "みかど", "mode": "isolated"}]
    assert (
        tts_util.apply_reading_overrides(text, overrides)
        == "ちょっと期待してしまったみかどだが、すぐに失言を悔やむ。"
    )


def test_apply_reading_overrides_isolated_kanji_copula_keeps_compound() -> None:
    text = "天帝だ"
    overrides = [{"base": "帝", "reading": "みかど", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == text


def test_split_reading_overrides_single_kanji_mode() -> None:
    data = {
        "chapters": {
            "c1": {
                "replacements": [
                    {"base": "天", "reading": "てん"},
                    {"base": "人", "reading": "ひと", "mode": "isolated"},
                    {"base": "天国", "reading": "てんごく"},
                ]
            }
        }
    }
    _global, chapters = tts_util._split_reading_overrides_data(data)
    entries = chapters["c1"]
    entry_map = {item.get("base"): item for item in entries}
    assert entry_map["天"].get("mode") == "isolated"
    assert entry_map["人"].get("mode") == "isolated"
    assert "mode" not in entry_map["天国"]


def test_split_reading_overrides_global_single_kanji_mode() -> None:
    data = {
        "global": {
            "replacements": [
                {"base": "天", "reading": "てん"},
                {"base": "天国", "reading": "てんごく"},
            ]
        }
    }
    global_entries, _chapters = tts_util._split_reading_overrides_data(data)
    entry_map = {item.get("base"): item for item in global_entries}
    assert entry_map["天"].get("mode") == "isolated"
    assert "mode" not in entry_map["天国"]


def test_parse_reading_overrides_text_single_kanji_defaults_isolated() -> None:
    entries = tts_util._parse_reading_overrides_text("蟬＝セミ\n")
    assert entries == [{"base": "蟬", "reading": "セミ", "mode": "isolated"}]
    assert (
        tts_util.apply_reading_overrides("蟬の声と蟬たち", entries)
        == "セミの声と蟬たち"
    )


def test_apply_reading_overrides_regex() -> None:
    text = "御存知でした。御迷惑でした。"
    overrides = [{"base": "re:御(存知|迷惑)", "reading": r"ご\1"}]
    assert tts_util.apply_reading_overrides(text, overrides) == "ご存知でした。ご迷惑でした。"


def test_apply_reading_overrides_for_tts_formats_readings() -> None:
    text = "私は漢字と東京。"
    overrides = [
        {"base": "漢字", "reading": "かんじ"},
        {"base": "東京", "reading": "とうきょう"},
    ]
    sep = _default_first_token_separator()
    assert (
        tts_util._apply_reading_overrides_for_tts(text, overrides)
        == f"私は{sep}かんじ{sep}と{sep}とうきょう{sep}。"
    )


def test_apply_reading_overrides_for_tts_preserves_mixed_script_readings() -> None:
    text = "母は"
    overrides = [{"base": "母は", "reading": "ハハは"}]
    sep = _default_first_token_separator()
    assert tts_util._apply_reading_overrides_for_tts(text, overrides) == f"{sep}ハハは{sep}"


def test_apply_ruby_spans() -> None:
    text = "前漢字後"
    spans = [{"start": 1, "end": 3, "base": "漢字", "reading": "かんじ"}]
    assert tts_util.apply_ruby_spans(text, spans) == "前かんじ後"


def test_apply_ruby_evidence_to_chunk_formats_readings() -> None:
    chunk_text = "前日本。後半。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [{"start": 1, "end": 3, "base": "日本", "reading": "にほん"}]
    ruby_data = {"global": [{"base": "後半", "reading": "こうはん"}]}
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        ruby_data,
    )
    sep = _default_first_token_separator()
    assert out == f"前{sep}にほん{sep}。{sep}こうはん{sep}。"


def test_apply_ruby_evidence_to_chunk_skips_single_kanji_global() -> None:
    chunk_text = "全13巻。"
    chunk_span = (0, len(chunk_text))
    ruby_data = {"global": [{"base": "巻", "reading": "ま"}]}
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        [],
        ruby_data,
    )
    assert out == chunk_text


def test_apply_ruby_evidence_to_chunk_promotes_conflict_majority_compound() -> None:
    chunk_text = "押し寄せる虚脱感に空太は手で顔を覆った。"
    ruby_data = {
        "conflicts": [
            {
                "base": "空太",
                "majority": "そらた",
                "readings": [
                    {"reading": "そらた", "count": 614},
                    {"reading": "そうた", "count": 2},
                ],
            }
        ]
    }
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        None,
        [],
        ruby_data,
    )
    assert "空太" not in out
    assert "そらた" in out


def test_apply_ruby_evidence_to_chunk_coalesces_adjacent_single_kanji_spans() -> None:
    chunk_text = "緑生さん"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [
        {"start": 0, "end": 1, "base": "緑", "reading": "ろく"},
        {"start": 1, "end": 2, "base": "生", "reading": "お"},
    ]
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        {},
    )
    sep = _default_first_token_separator()
    assert out.startswith(f"{sep}ろくお")
    assert "\\ろく\\お" not in out
    assert "さん" in out


def test_apply_ruby_evidence_to_chunk_drops_suspicious_span() -> None:
    chunk_text = "本町に西田医院という耳鼻科があったのを覚えているか。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [
        {"start": 0, "end": 2, "base": "本町", "reading": "ほんちょう"},
        {
            "start": 7,
            "end": 10,
            "base": "という",
            "reading": "あ" * 30,
        },
    ]
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        {},
    )
    sep = _default_first_token_separator()
    assert out == f"{sep}ほんちょう{sep}に西田医院という耳鼻科があったのを覚えているか。"


def test_is_suspicious_ruby_span_rejects_kanji_in_reading() -> None:
    assert tts_util._is_suspicious_ruby_span("可", "おかしそうに女は")


def test_apply_ruby_evidence_to_chunk_keeps_non_kanji_span() -> None:
    chunk_text = "TTSを試す。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [
        {"start": 0, "end": 3, "base": "TTS", "reading": "ティーティーエス"}
    ]
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        {},
    )
    sep = _default_first_token_separator()
    assert out == f"{sep}ティーティーエス{sep}を試す。"


@requires_unidic
def test_apply_ruby_evidence_to_chunk_keeps_inline_ruby_over_global_override() -> None:
    chunk_text = "暗殺者は暗殺者だ。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [
        {"start": 0, "end": 3, "base": "暗殺者", "reading": "おれ"},
    ]
    ruby_data = {"global": [{"base": "暗殺者", "reading": "あんさつしや"}]}
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        ruby_data,
    )
    sep = _default_first_token_separator()
    assert out.startswith(f"{sep}おれ{sep}は")
    assert f"{sep}あんさつしゃ{sep}だ。" in out
    assert "暗殺者" not in out


def test_apply_ruby_evidence_to_chunk_keeps_explicit_katakana_ruby() -> None:
    chunk_text = "前森野後"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [{"start": 1, "end": 3, "base": "森野", "reading": "モリノ"}]
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        {},
    )
    sep = _default_first_token_separator()
    assert out == f"前{sep}モリノ{sep}後"


def test_apply_ruby_evidence_to_chunk_drops_separator_before_okurigana() -> None:
    chunk_text = "喰おう。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [{"start": 0, "end": 1, "base": "喰", "reading": "く"}]
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        {},
    )
    sep = _default_first_token_separator()
    assert out == f"{sep}くおう。"


def _normalize_ruby_reading_with_stub(
    base: str, reading: str, base_reading_kata: str
) -> str:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [DummyToken(base, base_reading_kata)]

    tts_util._RUBY_BASE_READING_CACHE.clear()
    return tts_util._normalize_ruby_reading(base, reading, DummyTagger())


@pytest.mark.parametrize(
    ("base", "reading", "base_reading_kata", "expected"),
    [
        # GOSICK 全9冊合本版
        ("京介", "きようすけ", "キョウスケ", "きょうすけ"),
        ("豪奢", "ごうしや", "ゴウシャ", "ごうしゃ"),
        ("呪詛", "じゆそ", "ジュソ", "じゅそ"),
        ("几帳面", "きちようめん", "キチョウメン", "きちょうめん"),
        ("戦々恐々", "せんせんきようきよう", "センセンキヨウキョウ", "せんせんきようきょう"),
        # 【合本版】さくら荘のペットな彼女 全13巻（電子特典付き）
        ("表情", "ひようじよう", "ヒョウジョウ", "ひょうじょう"),
        ("一緒", "いつしよ", "イッショ", "いっしょ"),
        ("趣味", "しゆみ", "シュミ", "しゅみ"),
        # 【合本版】まよチキ！ 全12巻
        ("粛清", "しゆくせい", "シュクセイ", "しゅくせい"),
        ("驚愕", "きようがく", "キョウガク", "きょうがく"),
        ("卑怯", "ひきよう", "ヒキョウ", "ひきょう"),
        # 【合本版】俺の妹がこんなに可愛いわけがない 全12冊収録
        ("近況", "きんきよう", "キンキョウ", "きんきょう"),
        ("器量", "きりよう", "キリョウ", "きりょう"),
        # サクラダリセット（角川文庫）【全7冊 合本版】
        ("臆病", "おくびよう", "オクビョウ", "おくびょう"),
        ("信憑性", "しんぴようせい", "シンピョウセイ", "しんぴょうせい"),
        # 異人たちとの夏
        ("南無妙法蓮華経", "なむみようほうれんげきよう", "ナムミョウホウレンゲキョウ", "なむみょうほうれんげきょう"),
        # Ｖシリーズ全10冊合本版
        ("洒落", "しやれ", "シャレ", "しゃれ"),
        ("自重", "じちよう", "ジチョウ", "じちょう"),
    ],
)
def test_normalize_ruby_reading_sutegana_real_pairs(
    base: str, reading: str, base_reading_kata: str, expected: str
) -> None:
    out = _normalize_ruby_reading_with_stub(base, reading, base_reading_kata)
    assert out == expected


@pytest.mark.parametrize(
    ("base", "reading", "base_reading_kata", "expected"),
    [
        # Synthetic coverage to validate the full sutegana mapping surface.
        ("京極堂", "きようごくどう", "キョウゴクドウ", "きょうごくどう"),
        ("野球屋", "やきゆうや", "ヤキュウヤ", "やきゅうや"),
        ("学校通信", "がつこうつうしん", "ガッコウツウシン", "がっこうつうしん"),
        ("未知語", "ふあいる", "ファイル", "ふぁいる"),
        ("未知語", "ていあ", "ティア", "てぃあ"),
        ("未知語", "くわい", "クヮイ", "くゎい"),
        ("未知語", "かかく", "ヵカク", "ゕかく"),
        ("未知語", "カカク", "ヵカク", "ヵカク"),
    ],
)
def test_normalize_ruby_reading_sutegana_dummy_pairs(
    base: str, reading: str, base_reading_kata: str, expected: str
) -> None:
    out = _normalize_ruby_reading_with_stub(base, reading, base_reading_kata)
    assert out == expected


@pytest.mark.parametrize(
    ("base", "reading", "base_reading_kata"),
    [
        ("京介", "きようすけ", "キョウズケ"),
        ("野球屋", "やきゆうや", "ヤキュウバ"),
        ("学園生活", "にちじよう", "ガクエンセイカツ"),
    ],
)
def test_normalize_ruby_reading_sutegana_keeps_unmatched(
    base: str, reading: str, base_reading_kata: str
) -> None:
    out = _normalize_ruby_reading_with_stub(base, reading, base_reading_kata)
    assert out == reading


def test_synthesize_chunk_prefers_explicit_voice_over_manifest_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "book" / "tts"
    out_dir.mkdir(parents=True)
    manifest = {
        "voice": "高橋一生",
        "chapters": [
            {
                "id": "c1",
                "title": "Chapter 1",
                "voice": "高橋一生",
                "chunks": ["こんにちは"],
                "chunk_spans": [[0, 5]],
                "durations_ms": [None],
            }
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_resolve_voice_config(voice: str, base_dir: Path, **_kwargs):
        captured["voice"] = voice
        return voice_util.VoiceConfig(
            name=voice,
            ref_audio="dummy.wav",
            ref_text="dummy",
            language="Japanese",
            x_vector_only_mode=False,
        )

    monkeypatch.setattr(tts_util.synth_irodori, "get_runtime", lambda **_kwargs: object())
    monkeypatch.setattr(
        tts_util.synth_irodori,
        "generate_chunk",
        lambda *_args, **_kwargs: (tts_util.np.zeros(1, dtype=tts_util.np.float32), 48000),
    )
    monkeypatch.setattr(tts_util, "_load_ruby_data", lambda _book_dir: {"chapters": {}})
    monkeypatch.setattr(tts_util, "_load_reading_overrides", lambda _book_dir: ([], {}))
    monkeypatch.setattr(
        tts_util,
        "_prepare_tts_pipeline",
        lambda *_args, **_kwargs: tts_util.TtsPipeline(
            ruby_text="こんにちは",
            after_overrides="こんにちは",
            after_numbers="こんにちは",
            after_kana="こんにちは",
            prepared="こんにちは",
        ),
    )
    monkeypatch.setattr(tts_util, "_write_wav", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tts_util.voice_util, "resolve_voice_config", fake_resolve_voice_config)

    result = tts_util.synthesize_chunk(
        out_dir=out_dir,
        chapter_id="c1",
        chunk_index=0,
        voice="堀江由衣",
        base_dir=tmp_path,
    )
    assert result["status"] == "ok"
    assert captured["voice"] == "堀江由衣"


def test_synthesize_chunk_ellipsis_only_writes_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "book" / "tts"
    out_dir.mkdir(parents=True)
    manifest = {
        "voice": "高橋一生",
        "pad_ms": 350,
        "chapters": [
            {
                "id": "c1",
                "title": "Chapter 1",
                "voice": "高橋一生",
                "chunks": ["…………"],
                "chunk_spans": [[0, 4]],
                "pause_multipliers": [1],
                "durations_ms": [None],
            }
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    calls: dict[str, object] = {"generate_calls": 0, "written": None}

    def fake_generate_chunk(*_args, **_kwargs):
        calls["generate_calls"] = int(calls["generate_calls"]) + 1
        return tts_util.np.zeros(1, dtype=tts_util.np.float32), 48000

    def fake_write_wav(path: Path, audio, sample_rate: int) -> None:
        calls["written"] = (path, int(audio.shape[0]), sample_rate)

    monkeypatch.setattr(tts_util.synth_irodori, "get_runtime", lambda **_kwargs: object())
    monkeypatch.setattr(tts_util.synth_irodori, "generate_chunk", fake_generate_chunk)
    monkeypatch.setattr(tts_util, "_load_ruby_data", lambda _book_dir: {"chapters": {}})
    monkeypatch.setattr(tts_util, "_load_reading_overrides", lambda _book_dir: ([], {}))
    monkeypatch.setattr(
        tts_util,
        "_prepare_tts_pipeline",
        lambda *_args, **_kwargs: tts_util.TtsPipeline(
            ruby_text="…………",
            after_overrides="…………",
            after_numbers="…………",
            after_kana="…………",
            prepared="⋯",
        ),
    )
    monkeypatch.setattr(tts_util, "_write_wav", fake_write_wav)
    monkeypatch.setattr(
        tts_util.voice_util,
        "resolve_voice_config",
        lambda **_kwargs: voice_util.VoiceConfig(
            name="高橋一生",
            ref_audio="dummy.wav",
            ref_text="dummy",
            language="Japanese",
            x_vector_only_mode=False,
        ),
    )

    result = tts_util.synthesize_chunk(
        out_dir=out_dir,
        chapter_id="c1",
        chunk_index=0,
        base_dir=tmp_path,
    )
    assert result["status"] == "ok"
    assert int(calls["generate_calls"]) == 0
    written = calls["written"]
    assert written is not None
    assert written[1] > 0
    assert written[2] == 48000
    assert result["duration_ms"] >= 1000


def test_normalize_numbers_standalone_digits() -> None:
    text = "全7冊 合本版"
    assert tts_util._normalize_numbers(text) == "全ななさつ 合本版"


def test_normalize_numbers_standalone_digit_line_prefers_cardinal() -> None:
    assert tts_util._normalize_numbers("10") == "十"
    assert tts_util._normalize_numbers(" 11 ") == " 十一 "
    assert tts_util._normalize_numbers("12") == "十二"


def test_normalize_numbers_standalone_digits_in_sentence_stay_digit_seq() -> None:
    assert tts_util._normalize_numbers("番号 10") == "番号 いちゼロ"


def test_normalize_numbers_date() -> None:
    text = "2024年7月1日"
    assert tts_util._normalize_numbers(text) == "二千二十四年しちがつついたち"


def test_normalize_numbers_kanji_month() -> None:
    text = "四月"
    assert tts_util._normalize_numbers(text) == "しがつ"


def test_normalize_numbers_kanji_month_day() -> None:
    text = "七月一五日"
    assert tts_util._normalize_numbers(text) == "しちがつじゅうごにち"


def test_normalize_numbers_kanji_month_day_zero() -> None:
    text = "八月一〇日"
    assert tts_util._normalize_numbers(text) == "はちがつとおか"


def test_normalize_numbers_kanji_year_month_zero() -> None:
    text = "二〇一〇年三月"
    assert tts_util._normalize_numbers(text) == "二千十年さんがつ"


def test_normalize_numbers_kanji_time_digits() -> None:
    text = "一二時五九分〇二秒"
    assert tts_util._normalize_numbers(text) == "じゅうにじごじゅうきゅうふんにびょう"


def test_normalize_numbers_kanji_percent_zero() -> None:
    text = "一〇〇パーセント"
    assert tts_util._normalize_numbers(text) == "百パーセント"


def test_normalize_numbers_kanji_year_counter_zero() -> None:
    text = "三〇年"
    assert tts_util._normalize_numbers(text) == "三十年"


def test_normalize_numbers_kanji_digit_day_duration() -> None:
    text = "二〇日ほど経つ"
    assert tts_util._normalize_numbers(text) == "にじゅうにちほど経つ"


def test_normalize_numbers_kanji_digit_sequence_zero() -> None:
    text = "一〇〇の人間"
    assert tts_util._normalize_numbers(text) == "百の人間"


def test_normalize_numbers_kanji_decimal() -> None:
    text = "二．〇"
    assert tts_util._normalize_numbers(text) == "二てんぜろ"


def test_normalize_numbers_decimal() -> None:
    text = "3.14"
    assert tts_util._normalize_numbers(text) == "三てんいちよん"


def test_normalize_numbers_fallback_cardinal() -> None:
    text = "合計13です"
    assert tts_util._normalize_numbers(text) == "合計十三です"


def test_normalize_numbers_counter_wa() -> None:
    text = "１話 土曜日"
    assert tts_util._normalize_numbers(text) == "いちわ 土曜日"


def test_normalize_numbers_preserves_unicode_ellipsis() -> None:
    text = "……三ケ月前"
    assert tts_util._normalize_numbers(text) == "……さんかげつ前"


def test_normalize_numbers_counter_one_readings() -> None:
    assert tts_util._normalize_numbers("一冊") == "いっさつ"
    assert tts_util._normalize_numbers("１回") == "いっかい"
    assert tts_util._normalize_numbers("一分") == "いっぷん"
    assert tts_util._normalize_numbers("一匹") == "いっぴき"
    assert tts_util._normalize_numbers("一杯") == "いっぱい"
    assert tts_util._normalize_numbers("一体") == "いったい"
    assert tts_util._normalize_numbers("一ヶ月") == "いっかげつ"
    assert tts_util._normalize_numbers("一回目") == "いっかいめ"
    assert tts_util._normalize_numbers("一週間") == "いっしゅうかん"
    assert tts_util._normalize_numbers("一通の手紙") == "いっつうの手紙"
    assert tts_util._normalize_numbers("一話だけ") == "いちわだけ"


def test_normalize_numbers_counter_sound_changes() -> None:
    assert tts_util._normalize_numbers("三杯") == "さんばい"
    assert tts_util._normalize_numbers("六杯目") == "ろっぱい目"
    assert tts_util._normalize_numbers("八杯") == "はっぱい"
    assert tts_util._normalize_numbers("十杯") == "じゅっぱい"
    assert tts_util._normalize_numbers("三分") == "さんぷん"
    assert tts_util._normalize_numbers("四分") == "よんぷん"
    assert tts_util._normalize_numbers("六分間") == "ろっぷんかん"
    assert tts_util._normalize_numbers("八分") == "はっぷん"
    assert tts_util._normalize_numbers("十分待つ") == "じゅっぷん待つ"
    assert tts_util._normalize_numbers("十分間隔") == "じゅっぷんかんかく"
    assert tts_util._normalize_numbers("三本") == "さんぼん"
    assert tts_util._normalize_numbers("六本") == "ろっぽん"
    assert tts_util._normalize_numbers("八本") == "はっぽん"
    assert tts_util._normalize_numbers("十本") == "じゅっぽん"
    assert tts_util._normalize_numbers("三匹") == "さんびき"
    assert tts_util._normalize_numbers("六匹") == "ろっぴき"
    assert tts_util._normalize_numbers("八匹") == "はっぴき"
    assert tts_util._normalize_numbers("十匹") == "じゅっぴき"
    assert tts_util._normalize_numbers("八個") == "はっこ"
    assert tts_util._normalize_numbers("十個") == "じゅっこ"
    assert tts_util._normalize_numbers("六個") == "ろっこ"
    assert tts_util._normalize_numbers("三軒") == "さんげん"
    assert tts_util._normalize_numbers("八軒") == "はっけん"
    assert tts_util._normalize_numbers("十軒") == "じゅっけん"
    assert tts_util._normalize_numbers("八カ月") == "はっかげつ"
    assert tts_util._normalize_numbers("六ヵ月") == "ろっかげつ"
    assert tts_util._normalize_numbers("十ヶ月") == "じゅっかげつ"
    assert tts_util._normalize_numbers("八回") == "はっかい"
    assert tts_util._normalize_numbers("十回目") == "じゅっかいめ"


def test_normalize_numbers_juubun_enough() -> None:
    assert tts_util._normalize_numbers("十分に注意する") == "じゅうぶんに注意する"
    assert tts_util._normalize_numbers("十分な準備") == "じゅうぶんな準備"
    assert tts_util._normalize_numbers("十分だ") == "じゅうぶんだ"
    assert tts_util._normalize_numbers("十分です") == "じゅうぶんです"
    assert tts_util._normalize_numbers("十分でしょう") == "じゅうぶんでしょう"
    assert tts_util._normalize_numbers("十分である") == "じゅうぶんである"
    assert tts_util._normalize_numbers("十分。") == "じゅうぶん。"
    assert tts_util._normalize_numbers("十二分に伝わる") == "じゅうにぶんに伝わる"
    assert tts_util._normalize_numbers("百分に活かす") == "ひゃくぶんに活かす"
    assert tts_util._normalize_numbers("十分後") == "じゅっぷん後"
    assert tts_util._normalize_numbers("十分だけ待つ") == "じゅっぷんだけ待つ"


def test_normalize_numbers_counter_one_guards() -> None:
    assert tts_util._normalize_numbers("一通り終えた") == "一通り終えた"
    assert tts_util._normalize_numbers("一回り大きい") == "一回り大きい"
    assert tts_util._normalize_numbers("一個人の判断") == "一個人の判断"
    assert tts_util._normalize_numbers("一軒家に住む") == "一軒家に住む"
    assert tts_util._normalize_numbers("一軒屋で暮らす") == "一軒屋で暮らす"


def test_normalize_numbers_kanji_age_counter() -> None:
    text = "十二歳の頃"
    assert tts_util._normalize_numbers(text) == "じゅうにさいの頃"


def test_normalize_numbers_kanji_digit_list() -> None:
    text = "一、二席聞いて帰る"
    assert tts_util._normalize_numbers(text) == "一、にせき聞いて帰る"


def test_normalize_numbers_kanji_counter_age() -> None:
    text = "二十代半ば"
    assert tts_util._normalize_numbers(text) == "にじゅうだい半ば"


def test_normalize_numbers_kanji_counter_digit_seq() -> None:
    text = "一七五センチはある"
    assert tts_util._normalize_numbers(text) == "ひゃくななじゅうごせんちはある"


def test_normalize_numbers_person_readings() -> None:
    text = "一人で二人と4人"
    assert tts_util._normalize_numbers(text) == "ひとりでふたりとよにん"


def test_normalize_numbers_day_span() -> None:
    text = "二日間の旅"
    assert tts_util._normalize_numbers(text) == "ふつかかんの旅"


def test_normalize_numbers_day_span_single() -> None:
    text = "一日間滞在"
    assert tts_util._normalize_numbers(text) == "いちにちかん滞在"


def test_normalize_numbers_slash_fraction() -> None:
    text = "北の夕鶴2/3の殺人"
    assert tts_util._normalize_numbers(text) == "北の夕鶴さんぶんのにの殺人"


def test_normalize_numbers_bunno_fraction() -> None:
    assert tts_util._normalize_numbers("三分の二") == "さんぶんのに"
    assert tts_util._normalize_numbers("10分の1") == "じゅうぶんのいち"
    assert tts_util._normalize_numbers("十分の一") == "じゅうぶんのいち"


def test_normalize_numbers_slash_date_weekday() -> None:
    text = "2/3(日)開催"
    assert tts_util._normalize_numbers(text) == "にがつみっか(日)開催"


def test_normalize_numbers_slash_date_year() -> None:
    text = "2024/2/3"
    assert tts_util._normalize_numbers(text) == "二千二十四年にがつみっか"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # GOSICK 全９冊合本版 (角川文庫) - 桜庭 一樹.epub
        ("平成26年7月25日 発行", "平成二十六年しちがつにじゅうごにち 発行"),
        ("平成21年9月25日初版発行", "平成二十一年くがつにじゅうごにち初版発行"),
        ("GOSICK Ⅷ 上", "GOSICK はち 上"),
        ("GOSICK VIII 上", "GOSICK はち 上"),
        # 【合本版】俺の妹がこんなに可愛いわけがない...epub
        ("参加者１／２", "参加者にぶんのいち"),
        ("参加者２／３", "参加者さんぶんのに"),
        ("HG 1/144 スサノオ", "HG ひゃくよんじゅうよんぶんのいち スサノオ"),
        ("俺の妹がこんなに可愛いわけがない⑩", "俺の妹がこんなに可愛いわけがない十"),
        ("俺、和泉正宗／十五歳／高一。", "俺、和泉正宗/じゅうごさい/高一。"),
        # 【合本版】まよチキ！ 全12巻...epub
        ("【合本版】まよチキ！ 全12巻", "【合本版】まよチキ! 全じゅうにかん"),
        ("平日10:00～18:00まで", "平日じゅうじぜろふんからじゅうはちじぜろふんまで"),
        ("下心１００％だぜ", "下心百パーセントだぜ"),
        ("手術の成功率は五〇％。", "手術の成功率は五十パーセント。"),
        ("第５話 ダブルデートクライシス", "第ごわ ダブルデートクライシス"),
        ("方法① 校舎裏に呼び出してみる", "方法一 校舎裏に呼び出してみる"),
        ("ver.1.0", "ver.一てんゼロ"),
        # 【合本版】さくら荘のペットな彼女 全13巻（電子特典付き）...epub
        ("【合本版】さくら荘のペットな彼女 全13冊（電子特典付き）", "【合本版】さくら荘のペットな彼女 全じゅうさんさつ(電子特典付き)"),
        # サクラダリセット（角川文庫）【全7冊 合本版】...epub
        ("サクラダリセット（角川文庫）【全7冊 合本版】", "サクラダリセット(角川文庫)【全ななさつ 合本版】"),
        ("１ 八月一三日（金曜日）──スタート地点", "いち はちがつじゅうさんにち(金曜日)──スタート地点"),
        ("受付時間 9：00～17：00（土日 祝日 年末年始を除く）", "受付時間 きゅうじぜろふんからじゅうななじぜろふん(土日 祝日 年末年始を除く)"),
        # Ｖシリーズ全１０冊合本版 - 森博嗣.epub
        ("Ｖシリーズ全１０冊合本版", "Vシリーズ全じゅっさつ合本版"),
        ("0.9×0.8＝0.72", "零てんきゅう×零てんはち=零てんななに"),
    ],
)
def test_normalize_numbers_real_dataset_cases(text: str, expected: str) -> None:
    assert tts_util._normalize_numbers(text) == expected


def test_normalize_numbers_real_dataset_hyphen_address_not_date() -> None:
    # Real data appears in publication/address lines and should not be treated as Y/M/D.
    text = "東京都千代田区富士見2-13-3"
    out = tts_util._normalize_numbers(text)
    assert "千代田区" in out
    assert "年" not in out and "月" not in out and "日" not in out


def test_normalize_numbers_roman_numeral_to_kana() -> None:
    assert tts_util._normalize_numbers("GOSICK Ⅷ 上") == "GOSICK はち 上"


def test_normalize_numbers_ascii_roman_to_kana() -> None:
    assert tts_util._normalize_numbers("GOSICK VIII 上") == "GOSICK はち 上"
    assert tts_util._normalize_numbers("滅義怒羅怨II。") == "滅義怒羅怨に。"


def test_normalize_numbers_ascii_roman_acronyms_not_misconverted() -> None:
    assert tts_util._normalize_numbers("ドラマCD") == "ドラマCD"
    assert tts_util._normalize_numbers("TV CM") == "TV CM"
    assert tts_util._normalize_numbers("情報交換ML") == "情報交換ML"
    assert tts_util._normalize_numbers("東京MX") == "東京MX"


def test_prepare_tts_text_strips_japanese_quotes() -> None:
    assert (
        tts_util.prepare_tts_text("「聖書」『旧約』《新約》“Test” 'OK'〝注〟don't")
        == "聖書、旧約、新約Test OK注don't"
    )


def test_prepare_tts_text_preserves_japanese_internal_single_quote() -> None:
    assert tts_util.prepare_tts_text("りこん'で") == "りこん'で"


def test_prepare_tts_text_normalizes_width_and_format_chars() -> None:
    assert tts_util.prepare_tts_text("ＡＢＣ　１２３") == "ABC 123"
    assert tts_util.prepare_tts_text("a\u200bb") == "ab"


def test_prepare_tts_text_japanese_space_pause() -> None:
    assert (
        tts_util.prepare_tts_text("ウブメのナツ キョウゴクナツヒコ")
        == "ウブメのナツ。キョウゴクナツヒコ"
    )
    assert (
        tts_util.prepare_tts_text("Hello world 日本語 テスト")
        == "Hello world 日本語、テスト"
    )


def test_prepare_tts_text_normalizes_dash_runs() -> None:
    assert tts_util.prepare_tts_text("姑獲烏────") == "姑獲烏"
    assert tts_util.prepare_tts_text("前──後") == "前、後"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Real lines from: 【合本版】さくら荘のペットな彼女 全13巻（電子特典付き）
        ("「こまち、これが新幹線だぞ。早いだろ〜」", "こまち、これが新幹線だぞ。早いだろー"),
        ("「あ〜ん」", "あーん"),
        ("不機嫌な声を出した七海が、目をす〜っと細めて睨んでくる。", "不機嫌な声を出した七海が、目をすーっと細めて睨んでくる。"),
        ("「食事にする？ お風呂？ それとも〜、ま・わ・し？」", "食事にする? お風呂? それともー、ま・わ・し?"),
    ],
)
def test_prepare_tts_text_wave_dash_real_dataset_cases(text: str, expected: str) -> None:
    assert tts_util.prepare_tts_text(text) == expected


def test_prepare_tts_text_quote_boundary_preserves_question() -> None:
    assert tts_util.prepare_tts_text("「え？」と") == "え?と"


def test_prepare_tts_text_adds_short_tail_punct() -> None:
    assert tts_util.prepare_tts_text("まえがき", add_short_punct=True) == "まえがき。"
    assert tts_util.prepare_tts_text("前書き。", add_short_punct=True) == "前書き。"
    assert tts_util.prepare_tts_text("Prologue", add_short_punct=True) == "Prologue"
    long_text = "あ" * (tts_util._SHORT_TAIL_MAX_CHARS + 1)
    assert tts_util.prepare_tts_text(long_text, add_short_punct=True) == long_text


def test_load_reading_overrides(tmp_path: Path) -> None:
    payload = {
        "global": [
            {"base": "妻子", "reading": "さいし"},
            {"pattern": "御(存知)", "reading": r"ご\1"},
        ],
        "chapters": {
            "0001-test": {
                "replacements": [{"base": "漢字", "reading": "かんじ"}],
                "conflicts": [{"base": "生", "readings": ["せい", "なま"]}],
            }
        }
    }
    path = tmp_path / "reading-overrides.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    global_overrides, overrides = tts_util._load_reading_overrides(
        tmp_path, include_template=False
    )
    assert global_overrides == [
        {"base": "妻子", "reading": "さいし"},
        {"pattern": "御(存知)", "reading": r"ご\1"},
    ]
    assert overrides == {"0001-test": [{"base": "漢字", "reading": "かんじ"}]}


def test_ruby_global_overrides_supports_regex_entries() -> None:
    ruby_data = {
        "global": [
            {"base": "暗殺者", "reading": "あんさつしや"},
            {"base": "巻", "reading": "ま"},
            {"base": "上手", "pattern": "上手(?=い)", "reading": "うま"},
        ]
    }
    overrides = tts_util._ruby_global_overrides(ruby_data)
    assert {"base": "暗殺者", "reading": "あんさつしや"} in overrides
    assert {"pattern": "上手(?=い)", "reading": "うま"} in overrides
    assert {"base": "巻", "reading": "ま"} not in overrides


def test_ruby_global_overrides_promotes_high_conflict_majority() -> None:
    ruby_data = {
        "conflicts": [
            {
                "base": "空太",
                "majority": "そらた",
                "readings": [
                    {"reading": "そらた", "count": 614},
                    {"reading": "そうた", "count": 2},
                ],
            },
            {
                "base": "躊躇",
                "majority": "ためら",
                "readings": [
                    {"reading": "ためら", "count": 5},
                    {"reading": "ちゅうちょ", "count": 2},
                ],
            },
            {
                "base": "空",
                "majority": "そら",
                "readings": [
                    {"reading": "そら", "count": 1210},
                    {"reading": "くう", "count": 77},
                ],
            },
        ]
    }
    overrides = tts_util._ruby_global_overrides(ruby_data)
    assert {"base": "空太", "reading": "そらた"} in overrides
    assert {"base": "躊躇", "reading": "ためら"} not in overrides
    assert {"base": "空", "reading": "そら"} not in overrides


def test_ruby_global_overrides_conflict_does_not_override_explicit_global() -> None:
    ruby_data = {
        "global": [{"base": "空太", "reading": "そうた"}],
        "conflicts": [
            {
                "base": "空太",
                "majority": "そらた",
                "readings": [
                    {"reading": "そらた", "count": 614},
                    {"reading": "そうた", "count": 2},
                ],
            }
        ],
    }
    overrides = tts_util._ruby_global_overrides(ruby_data)
    assert overrides == [{"base": "空太", "reading": "そうた"}]


def test_ruby_chapter_compound_overrides_from_adjacent_single_kanji_spans() -> None:
    ruby_data = {
        "chapters": {
            "0005-chapter": {
                "clean_spans": [
                    {"start": 0, "end": 1, "base": "緑", "reading": "ろく"},
                    {"start": 1, "end": 2, "base": "生", "reading": "お"},
                    {"start": 10, "end": 11, "base": "緑", "reading": "ろく"},
                    {"start": 11, "end": 12, "base": "生", "reading": "お"},
                ]
            }
        }
    }
    overrides = tts_util._ruby_chapter_compound_overrides(
        ruby_data,
        chapter_id="0005-chapter",
    )
    assert {"base": "緑生", "reading": "ろくお"} in overrides


def test_ruby_chapter_compound_overrides_skips_ambiguous_readings() -> None:
    ruby_data = {
        "chapters": {
            "0005-chapter": {
                "clean_spans": [
                    {"start": 0, "end": 1, "base": "緑", "reading": "ろく"},
                    {"start": 1, "end": 2, "base": "生", "reading": "お"},
                    {"start": 10, "end": 11, "base": "緑", "reading": "りょく"},
                    {"start": 11, "end": 12, "base": "生", "reading": "せい"},
                ]
            }
        }
    }
    overrides = tts_util._ruby_chapter_compound_overrides(
        ruby_data,
        chapter_id="0005-chapter",
    )
    assert {"base": "緑生", "reading": "ろくお"} not in overrides
    assert {"base": "緑生", "reading": "りょくせい"} not in overrides


def test_augment_chapter_overrides_with_ruby_compounds_keeps_explicit_override() -> None:
    ruby_data = {
        "chapters": {
            "0005-chapter": {
                "clean_spans": [
                    {"start": 0, "end": 1, "base": "緑", "reading": "ろく"},
                    {"start": 1, "end": 2, "base": "生", "reading": "お"},
                    {"start": 10, "end": 11, "base": "緑", "reading": "ろく"},
                    {"start": 11, "end": 12, "base": "生", "reading": "お"},
                ]
            }
        }
    }
    chapter_overrides = [{"base": "緑生", "reading": "みどりお"}]
    augmented = tts_util._augment_chapter_overrides_with_ruby_compounds(
        chapter_overrides,
        ruby_data,
        chapter_id="0005-chapter",
    )
    assert tts_util.apply_reading_overrides("緑生さん", augmented) == "みどりおさん"


@requires_unidic
def test_augment_chapter_overrides_with_ruby_compounds_allows_singleton_name_context() -> None:
    ruby_data = {
        "chapters": {
            "0005-chapter": {
                "clean_spans": [
                    {"start": 0, "end": 1, "base": "緑", "reading": "ろく"},
                    {"start": 1, "end": 2, "base": "生", "reading": "お"},
                ]
            }
        }
    }
    augmented = tts_util._augment_chapter_overrides_with_ruby_compounds(
        [],
        ruby_data,
        chapter_id="0005-chapter",
        chapter_text="緑生さんの名を呼ぶ。",
    )
    assert tts_util.apply_reading_overrides("緑生さん", augmented) == "ろくおさん"


@requires_unidic
def test_augment_chapter_overrides_with_ruby_compounds_allows_singleton_kun_honorific() -> None:
    ruby_data = {
        "chapters": {
            "0005-chapter": {
                "clean_spans": [
                    {"start": 0, "end": 1, "base": "緑", "reading": "ろく"},
                    {"start": 1, "end": 2, "base": "生", "reading": "お"},
                ]
            }
        }
    }
    augmented = tts_util._augment_chapter_overrides_with_ruby_compounds(
        [],
        ruby_data,
        chapter_id="0005-chapter",
        chapter_text="緑生君が来た。",
    )
    assert tts_util.apply_reading_overrides("緑生君", augmented) == "ろくお君"


def test_ruby_chapter_compound_overrides_infers_single_kanji_prefix_name_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ruby_data = {
        "chapters": {
            "0003-chapter": {
                "clean_spans": [
                    {"start": 0, "end": 1, "base": "富", "reading": "と"},
                ]
            }
        }
    }
    chapter_text = "富山明男は来た。"

    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            *,
            pos1: str | None = None,
            pos2: str | None = None,
            pos3: str | None = None,
            pos4: str | None = None,
            type_: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.reading = kana
            self.pos1 = pos1
            self.pos2 = pos2
            self.pos3 = pos3
            self.pos4 = pos4
            self.type = type_

    class DummyToken:
        def __init__(
            self,
            surface: str,
            kana: str | None,
            *,
            pos1: str | None = None,
            pos2: str | None = None,
            pos3: str | None = None,
            pos4: str | None = None,
            type_: str | None = None,
        ) -> None:
            self.surface = surface
            self.feature = DummyFeature(
                kana,
                pos1=pos1,
                pos2=pos2,
                pos3=pos3,
                pos4=pos4,
                type_=type_,
            )

    class DummyTagger:
        def __call__(self, text: str):
            if text == chapter_text:
                return [
                    DummyToken(
                        "富山",
                        "トミヤマ",
                        pos1="名詞",
                        pos2="固有名詞",
                        pos3="人名",
                    ),
                    DummyToken(
                        "明男",
                        "アキオ",
                        pos1="名詞",
                        pos2="固有名詞",
                        pos3="人名",
                    ),
                    DummyToken("は", "ハ", pos1="助詞"),
                ]
            if text == "富山明男":
                return [DummyToken("富山明男", "トミヤマアキオ")]
            if text == "富":
                return [DummyToken("富", "トミ")]
            return []

    tts_util._RUBY_BASE_READING_CACHE.clear()
    monkeypatch.setattr(tts_util, "_get_kana_tagger", lambda: DummyTagger())
    overrides = tts_util._ruby_chapter_compound_overrides(
        ruby_data,
        chapter_id="0003-chapter",
        chapter_text=chapter_text,
    )
    assert {"base": "富山明男", "reading": "とやまあきお"} in overrides


def test_ruby_chapter_compound_overrides_does_not_infer_single_kanji_prefix_without_name_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ruby_data = {
        "chapters": {
            "0003-chapter": {
                "clean_spans": [
                    {"start": 0, "end": 1, "base": "富", "reading": "と"},
                ]
            }
        }
    }
    chapter_text = "富士山は高い。"

    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            *,
            pos1: str | None = None,
            pos2: str | None = None,
            pos3: str | None = None,
            pos4: str | None = None,
            type_: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.reading = kana
            self.pos1 = pos1
            self.pos2 = pos2
            self.pos3 = pos3
            self.pos4 = pos4
            self.type = type_

    class DummyToken:
        def __init__(
            self,
            surface: str,
            kana: str | None,
            *,
            pos1: str | None = None,
            pos2: str | None = None,
            pos3: str | None = None,
            pos4: str | None = None,
            type_: str | None = None,
        ) -> None:
            self.surface = surface
            self.feature = DummyFeature(
                kana,
                pos1=pos1,
                pos2=pos2,
                pos3=pos3,
                pos4=pos4,
                type_=type_,
            )

    class DummyTagger:
        def __call__(self, text: str):
            if text == chapter_text:
                return [
                    DummyToken(
                        "富士山",
                        "フジサン",
                        pos1="名詞",
                        pos2="固有名詞",
                        pos3="地名",
                    ),
                    DummyToken("は", "ハ", pos1="助詞"),
                    DummyToken("高い", "タカイ", pos1="形容詞"),
                ]
            if text == "富士山":
                return [DummyToken("富士山", "フジサン")]
            if text == "富":
                return [DummyToken("富", "トミ")]
            return []

    tts_util._RUBY_BASE_READING_CACHE.clear()
    monkeypatch.setattr(tts_util, "_get_kana_tagger", lambda: DummyTagger())
    overrides = tts_util._ruby_chapter_compound_overrides(
        ruby_data,
        chapter_id="0003-chapter",
        chapter_text=chapter_text,
    )
    assert not any(item.get("base") == "富士山" for item in overrides)


def test_merge_reading_overrides_prefers_chapter() -> None:
    global_overrides = [{"base": "山田太一", "reading": "やまだたいち"}]
    chapter_overrides = [{"base": "山田太一", "reading": "やまだたいいち"}]
    merged = tts_util._merge_reading_overrides(global_overrides, chapter_overrides)
    assert tts_util.apply_reading_overrides("山田太一", merged) == "やまだたいいち"


def test_merge_reading_overrides_prefers_global_over_propagated_chapter_entry() -> None:
    global_overrides = [{"base": "事件", "reading": "じけん"}]
    chapter_overrides = [{"base": "事件", "reading": "ヤマ"}]
    merged = tts_util._merge_reading_overrides(
        global_overrides,
        chapter_overrides,
        chapter_propagated_readings={"事件": {"ヤマ"}},
    )
    assert tts_util.apply_reading_overrides("事件", merged) == "じけん"


def test_merge_reading_overrides_keeps_non_propagated_chapter_override() -> None:
    global_overrides = [{"base": "事件", "reading": "じけん"}]
    chapter_overrides = [{"base": "事件", "reading": "じへん"}]
    merged = tts_util._merge_reading_overrides(
        global_overrides,
        chapter_overrides,
        chapter_propagated_readings={"事件": {"ヤマ"}},
    )
    assert tts_util.apply_reading_overrides("事件", merged) == "じへん"


def test_ruby_propagated_reading_map_includes_chapter_spans() -> None:
    ruby_data = {
        "global": [{"base": "事件", "reading": "ヤマ"}],
        "chapters": {
            "0003-chapter": {
                "clean_spans": [{"base": "蟬", "reading": "せみ"}],
                "raw_spans": [{"base": "須藤", "reading": "すどう"}],
            }
        },
    }

    chapter_map = tts_util._ruby_propagated_reading_map(
        ruby_data,
        chapter_id="0003-chapter",
    )
    assert chapter_map["事件"] == {"ヤマ"}
    assert chapter_map["蟬"] == {"せみ"}
    assert chapter_map["須藤"] == {"すどう"}

    other_map = tts_util._ruby_propagated_reading_map(
        ruby_data,
        chapter_id="0004-chapter",
    )
    assert other_map == {"事件": {"ヤマ"}}


def test_ruby_propagated_reading_map_coalesces_adjacent_single_kanji_spans() -> None:
    ruby_data = {
        "chapters": {
            "0005-chapter": {
                "clean_spans": [
                    {"start": 0, "end": 1, "base": "緑", "reading": "ろく"},
                    {"start": 1, "end": 2, "base": "生", "reading": "お"},
                ],
            }
        }
    }
    chapter_map = tts_util._ruby_propagated_reading_map(
        ruby_data,
        chapter_id="0005-chapter",
    )
    assert chapter_map["緑生"] == {"ろくお"}
    assert chapter_map["緑"] == {"ろく"}
    assert chapter_map["生"] == {"お"}


def test_merge_reading_overrides_prefers_global_over_single_kanji_ruby_chapter_entry() -> None:
    global_overrides = [{"base": "蟬", "reading": "セミ"}]
    chapter_overrides = [{"base": "蟬", "reading": "せみ"}]
    ruby_data = {
        "chapters": {
            "0003-chapter": {
                "clean_spans": [{"base": "蟬", "reading": "せみ"}],
            }
        }
    }
    propagated = tts_util._ruby_propagated_reading_map(
        ruby_data,
        chapter_id="0003-chapter",
    )
    merged = tts_util._merge_reading_overrides(
        global_overrides,
        chapter_overrides,
        chapter_propagated_readings=propagated,
    )
    assert tts_util.apply_reading_overrides("蟬の声", merged) == "セミの声"


def test_load_reading_overrides_includes_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template_path = tmp_path / "template.md"
    template_path.write_text("# header\n妻子＝さいし\n", encoding="utf-8")
    monkeypatch.setattr(
        tts_util, "_template_reading_overrides_path", lambda: template_path
    )
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    (book_dir / "reading-overrides.json").write_text(
        json.dumps({"global": [{"base": "妻子", "reading": "つまこ"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    global_overrides, overrides = tts_util._load_reading_overrides(book_dir)
    assert overrides == {}
    assert tts_util.apply_reading_overrides("妻子", global_overrides) == "つまこ"


