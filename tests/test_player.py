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
