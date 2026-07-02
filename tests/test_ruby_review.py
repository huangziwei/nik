import json
from pathlib import Path

import pytest

from nik import ruby_review
from nik import tts as tts_util
from nik.text import read_clean_text


def test_suggest_small_kana_reading_palatalization() -> None:
    assert ruby_review.suggest_small_kana_reading("じゆ") == "じゅ"
    assert ruby_review.suggest_small_kana_reading("きようすけ") == "きょうすけ"


def test_suggest_small_kana_reading_te_i_and_katakana() -> None:
    assert ruby_review.suggest_small_kana_reading("ていあら") == "てぃあら"
    assert ruby_review.suggest_small_kana_reading("イマニテイ") == "イマニティ"


def test_suggest_small_kana_reading_gemination() -> None:
    assert ruby_review.suggest_small_kana_reading("あさつて") == "あさって"


def test_suggest_small_kana_reading_no_candidates() -> None:
    assert ruby_review.suggest_small_kana_reading("てぃあら") == ""
    assert ruby_review.suggest_small_kana_reading("のぞみ") == ""
    assert ruby_review.suggest_small_kana_reading("ちいさい") == ""
    assert ruby_review.suggest_small_kana_reading("みつ") == ""
    assert ruby_review.suggest_small_kana_reading("") == ""


def _write_review_book(book_dir: Path) -> dict:
    raw_dir = book_dir / "raw" / "chapters"
    raw_dir.mkdir(parents=True)
    chapter_path = raw_dir / "0001-chapter.txt"
    chapter_path.write_text(
        "天愛星は笑った。人生は続く。天愛星は歩いた。天愛星が来た。\n",
        encoding="utf-8",
    )
    text = tts_util._normalize_text(read_clean_text(chapter_path))

    spans = []
    start = 0
    for _ in range(2):
        pos = text.index("天愛星", start)
        spans.append(
            {"start": pos, "end": pos + 3, "base": "天愛星", "reading": "ていあら"}
        )
        start = pos + 3
    jinsei = text.index("人生")
    spans.append(
        {"start": jinsei, "end": jinsei + 2, "base": "人生", "reading": "クソゲー"}
    )
    spans.sort(key=lambda span: span["start"])

    toc = {
        "chapters": [
            {"index": 1, "title": "第一章", "path": "raw/chapters/0001-chapter.txt"}
        ]
    }
    (book_dir / "toc.json").write_text(
        json.dumps(toc, ensure_ascii=False), encoding="utf-8"
    )
    overrides = {
        "chapters": {},
        "ruby": {
            "global": [
                {"base": "天愛星", "reading": "ていあら", "count": 2, "total": 2},
                {"base": "人生", "reading": "クソゲー", "count": 1, "total": 1},
            ],
            "conflicts": [],
            "chapters": {
                "0001-chapter": {
                    "raw_sha256": tts_util.sha256_str(text),
                    "raw_spans": spans,
                }
            },
        },
    }
    (book_dir / "reading-overrides.json").write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"text": text}


def test_build_review_groups_counts_flags_and_impact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tts_util, "_ruby_reading_aligned", lambda base, reading: False
    )
    book_dir = tmp_path / "book"
    _write_review_book(book_dir)

    payload = ruby_review.build_review_groups(book_dir)
    groups = {group["key"]: group for group in payload["groups"]}

    tia = groups["天愛星|ていあら"]
    assert tia["count"] == 2
    assert len(tia["contexts"]) == 2
    assert tia["prop_count"] == 1
    assert tia["default_scope"] == "global"
    assert tia["scope"] == "global"
    assert tia["propagates"] is True
    assert tia["decided"] is False
    assert tia["suggestion"] == "てぃあら"
    assert "sutegana" in tia["flags"]
    assert tia["prop_contexts"][0]["hit"] == "天愛星"

    jinsei = groups["人生|クソゲー"]
    assert jinsei["count"] == 1
    assert jinsei["prop_count"] == 0
    assert jinsei["default_scope"] == "inline"
    assert jinsei["scope"] == "inline"
    assert jinsei["propagates"] is False
    assert "wordplay" in jinsei["flags"]
    assert "singleton" in jinsei["flags"]


def test_build_review_groups_reflects_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tts_util, "_ruby_reading_aligned", lambda base, reading: False
    )
    book_dir = tmp_path / "book"
    _write_review_book(book_dir)

    ruby_review.update_decisions(
        book_dir,
        {
            "天愛星|ていあら": {"reading": "てぃあら", "scope": "global"},
            "人生|クソゲー": {"scope": "off"},
        },
    )
    payload = ruby_review.build_review_groups(book_dir)
    groups = {group["key"]: group for group in payload["groups"]}

    tia = groups["天愛星|ていあら"]
    assert tia["corrected"] == "てぃあら"
    assert tia["decided"] is True
    assert tia["scope"] == "global"
    assert tia["suggestion"] == ""
    assert tia["propagates"] is True

    jinsei = groups["人生|クソゲー"]
    assert jinsei["scope"] == "off"
    assert jinsei["propagates"] is False


def test_update_decisions_roundtrip_and_delete(tmp_path: Path) -> None:
    book_dir = tmp_path / "book"
    book_dir.mkdir()

    stored = ruby_review.update_decisions(
        book_dir, {"一|はじめ": {"scope": "global", "mode": "isolated"}}
    )
    assert "一|はじめ" in stored
    assert stored["一|はじめ"]["scope"] == "global"
    assert stored["一|はじめ"]["mode"] == "isolated"
    assert "updated_unix" in stored["一|はじめ"]

    data = json.loads(
        (book_dir / "reading-overrides.json").read_text(encoding="utf-8")
    )
    assert data["ruby"]["decisions"]["一|はじめ"]["scope"] == "global"

    stored = ruby_review.update_decisions(book_dir, {"一|はじめ": None})
    assert stored == {}
    data = json.loads(
        (book_dir / "reading-overrides.json").read_text(encoding="utf-8")
    )
    assert "decisions" not in data["ruby"]


def test_update_decisions_ignores_malformed_keys(tmp_path: Path) -> None:
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    stored = ruby_review.update_decisions(
        book_dir,
        {
            "no-separator": {"scope": "off"},
            "|のみ": {"scope": "off"},
            "天愛星|ていあら": {"scope": "bogus"},
        },
    )
    assert stored == {}


def test_build_review_groups_flags_repeated_single_kanji_as_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tts_util, "_ruby_reading_aligned", lambda base, reading: False
    )
    book_dir = tmp_path / "book"
    raw_dir = book_dir / "raw" / "chapters"
    raw_dir.mkdir(parents=True)
    chapter_path = raw_dir / "0001-chapter.txt"
    chapter_path.write_text("一が来た。一は笑う。一が去る。\n", encoding="utf-8")
    text = tts_util._normalize_text(read_clean_text(chapter_path))

    spans = []
    start = 0
    for _ in range(3):
        pos = text.index("一", start)
        spans.append({"start": pos, "end": pos + 1, "base": "一", "reading": "はじめ"})
        start = pos + 1

    (book_dir / "toc.json").write_text(
        json.dumps(
            {
                "chapters": [
                    {
                        "index": 1,
                        "title": "第一章",
                        "path": "raw/chapters/0001-chapter.txt",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (book_dir / "reading-overrides.json").write_text(
        json.dumps(
            {
                "ruby": {
                    "chapters": {
                        "0001-chapter": {
                            "raw_sha256": tts_util.sha256_str(text),
                            "raw_spans": spans,
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = ruby_review.build_review_groups(book_dir)
    groups = {group["key"]: group for group in payload["groups"]}
    hajime = groups["一|はじめ"]
    assert "name" in hajime["flags"]
    assert hajime["scope"] == "inline"
    assert hajime["mode"] == "isolated"
    assert hajime["count"] == 3


def test_build_review_groups_does_not_flag_singleton_context_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tts_util, "_ruby_reading_aligned", lambda base, reading: False
    )
    book_dir = tmp_path / "book"
    raw_dir = book_dir / "raw" / "chapters"
    raw_dir.mkdir(parents=True)
    chapter_path = raw_dir / "0001-chapter.txt"
    chapter_path.write_text("焚き火を囲む。\n", encoding="utf-8")
    text = tts_util._normalize_text(read_clean_text(chapter_path))
    pos = text.index("火")
    (book_dir / "toc.json").write_text(
        json.dumps(
            {
                "chapters": [
                    {
                        "index": 1,
                        "title": "第一章",
                        "path": "raw/chapters/0001-chapter.txt",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (book_dir / "reading-overrides.json").write_text(
        json.dumps(
            {
                "ruby": {
                    "chapters": {
                        "0001-chapter": {
                            "raw_sha256": tts_util.sha256_str(text),
                            "raw_spans": [
                                {
                                    "start": pos,
                                    "end": pos + 1,
                                    "base": "火",
                                    "reading": "び",
                                }
                            ],
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = ruby_review.build_review_groups(book_dir)
    groups = {group["key"]: group for group in payload["groups"]}
    hi = groups["火|び"]
    assert "name" not in hi["flags"]
    assert hi["scope"] == "inline"
