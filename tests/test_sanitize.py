import json
from pathlib import Path

from nik import sanitize as sanitize_util
from nik import tts as tts_util
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


def test_normalize_text_converts_star_section_break() -> None:
    raw = "一。\n☆\n二。"
    cleaned = sanitize_util.normalize_text(raw)
    assert SECTION_BREAK in cleaned
    assert "☆" not in cleaned


def test_normalize_text_converts_fullwidth_asterisk_section_break() -> None:
    raw = "一。\n＊\n二。"
    cleaned = sanitize_util.normalize_text(raw)
    assert SECTION_BREAK in cleaned
    assert "＊" not in cleaned


def test_normalize_text_converts_diamond_section_break() -> None:
    raw = "一。\n◇\n二。"
    cleaned = sanitize_util.normalize_text(raw)
    assert SECTION_BREAK in cleaned
    assert "◇" not in cleaned


def test_normalize_text_does_not_convert_ellipsis_line_as_section_break() -> None:
    raw = "一。\n\n……\n\n二。"
    cleaned = sanitize_util.normalize_text(raw)
    assert SECTION_BREAK not in cleaned
    assert "……" in cleaned


def test_normalize_text_does_not_convert_quoted_ellipsis_line() -> None:
    raw = "一。\n\n「……」\n\n二。"
    cleaned = sanitize_util.normalize_text(raw)
    assert SECTION_BREAK not in cleaned
    assert "「……」" in cleaned


def test_normalize_text_preserves_raw_line_breaks() -> None:
    raw = "First.\nSecond.\n\nSection start.\nThird."
    cleaned = sanitize_util.normalize_text(raw)
    assert SECTION_BREAK not in cleaned
    assert "First.\nSecond." in cleaned
    assert "Section start.\nThird." in cleaned


def test_normalize_text_preserves_dialogue_runs() -> None:
    raw = (
        "「三年前、僕が研究所にはいったばかりの頃だ。"
        "そいつはペーアゼン博士の研究室に頻繁に出入りしていたよ。"
        "あるときからぱったり見かけなくなったが」\n"
        "「学者かい」\n"
        "　そう訊くと男は大きく手を振って、"
    )
    cleaned = sanitize_util.normalize_text(raw)
    assert (
        "ぱったり見かけなくなったが」\n「学者かい」\n　そう訊くと男は大きく手を振って、"
        in cleaned
    )


def test_normalize_text_collapses_dialogue_continuation_break() -> None:
    raw = (
        "いいながら、自嘲するような笑い声を立てた。そしてすぐそれを打ち消すように早口で、\n\n"
        "「シャンペンなの」と明るい声を出し「のみかけのシャンペン。」"
    )
    cleaned = sanitize_util.normalize_text(raw)
    assert (
        "早口で、\n「シャンペンなの」と明るい声を出し「のみかけのシャンペン。」"
        in cleaned
    )
    assert "早口で、\n\n「シャンペンなの」" not in cleaned


def test_normalize_text_preserves_dialogue_linebreak_on_reclean() -> None:
    raw = (
        "いいながら、自嘲するような笑い声を立てた。そしてすぐそれを打ち消すように早口で、\n\n"
        "「シャンペンなの」と明るい声を出し「のみかけのシャンペン。」"
    )
    first = sanitize_util.normalize_text(raw)
    second = sanitize_util.normalize_text(first)
    assert "早口で、\n「シャンペンなの」" in second
    assert "早口で、 「シャンペンなの」" not in second


def test_normalize_text_does_not_promote_breaks_on_reclean() -> None:
    raw = (
        "　夜間は警戒が厳しくて駄目だ。むしろ昼間のほうがいい。\n"
        "　方針さえ決まれば具体的なシナリオは自然に組みあがっていく。\n"
        "　まず、カメラ。"
    )
    first = sanitize_util.normalize_text(raw)
    second = sanitize_util.normalize_text(first)
    assert SECTION_BREAK not in second
    assert "\n\n\n" not in second
    assert "ほうがいい。\n　方針さえ決まれば" in second


def test_diff_ruby_spans_splits_kana_boundary() -> None:
    spans = sanitize_util._diff_ruby_spans("荘で監視", "そうでかんし")
    assert spans == [
        {"start": 0, "end": 1, "base": "荘", "reading": "そう"},
        {"start": 2, "end": 4, "base": "監視", "reading": "かんし"},
    ]


def test_split_ruby_span_on_kana_handles_ambiguous_repeated_boundary() -> None:
    spans = sanitize_util._split_ruby_span_on_kana(
        0,
        "龍之介の姿",
        "りゆうのすけのすがた",
    )
    assert spans == [
        {"start": 0, "end": 3, "base": "龍之介", "reading": "りゆうのすけ"},
        {"start": 4, "end": 5, "base": "姿", "reading": "すがた"},
    ]


def test_collect_clean_ruby_spans_from_raw_tracks_span_identity() -> None:
    raw = (
        "いいながら、自嘲するような笑い声を立てた。そしてすぐそれを打ち消すように早口で、\n\n"
        "「シャンペンなの」と明るい声を出し「のみかけのシャンペン。あけたけど一人じゃのみきれなくて、"
        "のんでいただこうと思って。すみません。明日まで置いたら気がぬけるでしょう」\n\n"
        "可笑しそうに女は笑った。\n\n"
        "「嬉しいけど」\n\n"
        "私も笑顔をつくったが、動かなかった。"
    )
    clean = sanitize_util.normalize_text(raw)
    clean, _ = sanitize_util.apply_section_cutoff(clean, [])
    clean, _ = sanitize_util.apply_remove_patterns(clean, [])
    clean = sanitize_util.normalize_text(clean)
    clean = sanitize_util.normalize_small_caps(clean, extra_words=set())
    clean = sanitize_util.normalize_all_caps(clean, extra_words=set())

    i_okashi = raw.index("可笑")
    i_ureshi = raw.index("嬉")
    raw_spans = [
        {"start": i_okashi, "end": i_okashi + 2, "base": "可笑", "reading": "おか"},
        {"start": i_ureshi, "end": i_ureshi + 1, "base": "嬉", "reading": "うれ"},
    ]
    spans = sanitize_util._collect_clean_ruby_spans_from_raw(
        raw_text_original=raw,
        raw_spans=raw_spans,
        cleaned_text=clean,
        cutoff_patterns=[],
        remove_patterns=[],
        case_words=set(),
    )

    assert spans is not None
    assert any(
        span["base"] == "可笑" and span["reading"] == "おか"
        for span in spans
    )
    assert any(
        span["base"] == "嬉" and span["reading"] == "うれ"
        for span in spans
    )
    assert not any(
        span["base"] == "可" and span["reading"] == "おかしそうに女は"
        for span in spans
    )


def test_diff_ruby_spans_drops_suspicious_long_reading() -> None:
    spans = sanitize_util._diff_ruby_spans(
        "と",
        "がべつの女性と結婚するという。母は、若いころに別の",
    )
    assert spans == []


def test_sanitize_dropped_chapter_clears_ruby_clean_spans(tmp_path: Path) -> None:
    book_dir = _write_book(tmp_path)
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "replace_defaults": False,
                "drop_chapter_title_patterns": ["^Chapter 1$"],
                "section_cutoff_patterns": [],
                "remove_patterns": [],
            }
        ),
        encoding="utf-8",
    )
    overrides_path = book_dir / "reading-overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "ruby": {
                    "chapters": {
                        "0001-chapter": {
                            "raw_sha256": "dummy",
                            "raw_spans": [],
                            "clean_spans": [
                                {
                                    "start": 0,
                                    "end": 1,
                                    "base": "A",
                                    "reading": "B",
                                }
                            ],
                            "clean_sha256": "dummy",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    sanitize_util.sanitize_book(
        book_dir=book_dir, rules_path=rules_path, overwrite=True
    )
    updated = json.loads(overrides_path.read_text(encoding="utf-8"))
    entry = updated["ruby"]["chapters"]["0001-chapter"]
    assert "clean_spans" not in entry
    assert "clean_sha256" not in entry


def test_sanitize_book_preserves_ruby_span_identity_for_dialogue_excerpt(
    tmp_path: Path,
) -> None:
    book_dir = tmp_path / "book"
    raw_dir = book_dir / "raw" / "chapters"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "0001-chapter.txt"
    raw_text = (
        "いいながら、自嘲するような笑い声を立てた。そしてすぐそれを打ち消すように早口で、\n\n"
        "「シャンペンなの」と明るい声を出し「のみかけのシャンペン。あけたけど一人じゃのみきれなくて、"
        "のんでいただこうと思って。すみません。明日まで置いたら気がぬけるでしょう」\n\n"
        "可笑しそうに女は笑った。\n\n"
        "「嬉しいけど」\n\n"
        "私も笑顔をつくったが、動かなかった。"
    )
    raw_path.write_text(raw_text, encoding="utf-8")

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
    (book_dir / "toc.json").write_text(
        json.dumps(toc, ensure_ascii=False),
        encoding="utf-8",
    )

    i_okashi = raw_text.index("可笑")
    i_ureshi = raw_text.index("嬉")
    overrides = {
        "ruby": {
            "chapters": {
                "0001-chapter": {
                    "raw_sha256": "dummy",
                    "raw_spans": [
                        {
                            "start": i_okashi,
                            "end": i_okashi + 2,
                            "base": "可笑",
                            "reading": "おか",
                        },
                        {
                            "start": i_ureshi,
                            "end": i_ureshi + 1,
                            "base": "嬉",
                            "reading": "うれ",
                        },
                    ],
                }
            }
        }
    }
    overrides_path = book_dir / "reading-overrides.json"
    overrides_path.write_text(
        json.dumps(overrides, ensure_ascii=False),
        encoding="utf-8",
    )

    sanitize_util.sanitize_book(book_dir=book_dir, overwrite=True)
    updated = json.loads(overrides_path.read_text(encoding="utf-8"))
    clean_spans = updated["ruby"]["chapters"]["0001-chapter"]["clean_spans"]

    assert any(
        span["base"] == "可笑" and span["reading"] == "おか"
        for span in clean_spans
    )
    assert any(
        span["base"] == "嬉" and span["reading"] == "うれ"
        for span in clean_spans
    )
    assert not any(
        span["base"] == "可" and span["reading"] == "おかしそうに女は"
        for span in clean_spans
    )
    for span in clean_spans:
        assert not tts_util._is_suspicious_ruby_span(
            str(span.get("base") or ""),
            str(span.get("reading") or ""),
        )
