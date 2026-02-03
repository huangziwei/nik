from pathlib import Path

import pytest

from nik import asr


def test_normalize_language() -> None:
    assert asr.normalize_language(None) is None
    assert asr.normalize_language("") is None
    assert asr.normalize_language("Japanese") == "ja"
    assert asr.normalize_language("EN") == "en"
    assert asr.normalize_language("xx") == "xx"


def test_transcribe_audio_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        asr.transcribe_audio(missing)
