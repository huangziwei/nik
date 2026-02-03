import json
from pathlib import Path

from nik import player as player_util


def test_list_voice_ids(tmp_path: Path) -> None:
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "a.json").write_text("{}", encoding="utf-8")
    nested = voices_dir / "b"
    nested.mkdir()
    (nested / "voice.json").write_text("{}", encoding="utf-8")

    ids = player_util._list_voice_ids(tmp_path)
    assert set(ids) == {"a", "b"}


def test_sanitize_voice_map_fallback(tmp_path: Path) -> None:
    payload = {"default": "default", "chapters": {"001": "a"}}
    data = player_util._sanitize_voice_map(payload, tmp_path, fallback_default="fallback")
    assert data["default"] == "fallback"
    assert data["chapters"] == {"001": "a"}


def test_slug_from_title_preserves_japanese() -> None:
    title = "異人たちとの夏"
    slug = player_util._slug_from_title(title, "fallback")
    assert slug == title


def test_slug_from_title_sanitizes_path_chars() -> None:
    slug = player_util._slug_from_title("foo/bar:baz", "fallback")
    assert "/" not in slug
    assert ":" not in slug


def test_ensure_unique_slug(tmp_path: Path) -> None:
    (tmp_path / "book").mkdir()
    unique = player_util._ensure_unique_slug(tmp_path, "book")
    assert unique == "book-2"


def test_normalize_reading_overrides() -> None:
    raw = [
        {"base": "妻子", "reading": "さいし"},
        {"base": "", "reading": "ignored"},
        ["山田太一", "やまだたいいち"],
        {"base": "東京", "kana": "とうきょう"},
        "invalid",
    ]
    cleaned = player_util._normalize_reading_overrides(raw)
    assert cleaned == [
        {"base": "妻子", "reading": "さいし"},
        {"base": "山田太一", "reading": "やまだたいいち"},
        {"base": "東京", "reading": "とうきょう"},
    ]
