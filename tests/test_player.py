import json
import shutil
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nik import player as player_util
from nik import voice as voice_util


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='nik'\n", encoding="utf-8")
    root_dir = repo_root / "out"
    root_dir.mkdir()
    return repo_root, root_dir


def _make_book(root_dir: Path, book_id: str = "book") -> Path:
    book_dir = root_dir / book_id
    (book_dir / "clean").mkdir(parents=True)
    (book_dir / "clean" / "toc.json").write_text("{}", encoding="utf-8")
    (book_dir / "tts").mkdir(parents=True, exist_ok=True)
    (book_dir / "tts" / "manifest.json").write_text(
        json.dumps({"voice": "manifest-voice", "chapters": [{"id": "c1", "chunks": ["chunk"]}]}),
        encoding="utf-8",
    )
    return book_dir


def test_book_details_includes_pause_multipliers(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book(root_dir)
    (book_dir / "tts" / "manifest.json").write_text(
        json.dumps(
            {
                "voice": "manifest-voice",
                "chapters": [
                    {
                        "id": "c1",
                        "title": "Chapter 1",
                        "chunk_spans": [[0, 2], [2, 4]],
                        "pause_multipliers": [4, 7],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    details = player_util._book_details(book_dir, repo_root)
    assert details["chapters"] == [
        {
            "id": "c1",
            "title": "Chapter 1",
            "chunk_spans": [[0, 2], [2, 4]],
            "pause_multipliers": [4, 7],
            "chunk_count": 2,
        }
    ]


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


def test_synth_request_defaults_use_hiragana_and_mid_sentence_conversion() -> None:
    payload = player_util.SynthRequest(book_id="book")
    assert payload.kana_style == "hiragana"
    assert payload.partial_mid_kanji is True


def test_delete_m4b_outputs_removes_parts(tmp_path: Path) -> None:
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    library_dir = tmp_path / "_m4b"
    library_dir.mkdir()
    base = library_dir / "book.m4b"
    part1 = library_dir / "book.part01.m4b"
    part2 = library_dir / "book.part02.m4b"
    base.write_bytes(b"base")
    part1.write_bytes(b"part1")
    part2.write_bytes(b"part2")

    removed = player_util._delete_m4b_outputs(book_dir)
    assert removed is True
    assert not base.exists()
    assert not part1.exists()
    assert not part2.exists()


def test_parse_clone_time_accepts_seconds_and_timecode() -> None:
    assert player_util._parse_clone_time("0", "start", allow_zero=True) == "0"
    assert player_util._parse_clone_time("1.250", "start", allow_zero=True) == "1.25"
    assert player_util._parse_clone_time("01:02", "start", allow_zero=True) == "62"
    assert player_util._parse_clone_time("00:01:02.5", "start", allow_zero=True) == "62.5"


@pytest.mark.parametrize(
    "raw,allow_zero,message",
    [
        ("", False, "required"),
        ("-1", True, ">= 0"),
        ("0", False, "> 0"),
        ("00:61", True, "below 60"),
        ("a:b:c", True, "timecode"),
    ],
)
def test_parse_clone_time_rejects_invalid_values(
    raw: str,
    allow_zero: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        player_util._parse_clone_time(raw, "duration", allow_zero=allow_zero)


def test_clone_preview_and_save_reuses_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    source = repo_root / "voice-source.mp3"
    source.write_bytes(b"source")
    source_path = str(source)
    calls: list[tuple[Path, Path, str, str]] = []

    def fake_run_clone_ffmpeg(
        input_path: Path,
        output_path: Path,
        start: str,
        duration: str,
    ) -> None:
        calls.append((input_path, output_path, start, duration))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFFFAKEWAVE")

    real_which = shutil.which

    def fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return real_which(name)

    monkeypatch.setattr(player_util, "_run_clone_ffmpeg", fake_run_clone_ffmpeg)
    monkeypatch.setattr(player_util.shutil, "which", fake_which)
    monkeypatch.setattr(
        player_util.asr_util, "transcribe_audio", lambda *args, **kwargs: "hello world"
    )

    app = player_util.create_app(root_dir)
    client = TestClient(app)

    preview = client.post(
        "/api/voices/clone/preview",
        json={"source": source_path, "start": "1", "duration": "3.5", "gender": "female"},
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["status"] == "ready"
    assert preview_payload["suggested_name"] == "voice-source"
    assert preview_payload["start"] == "1"
    assert preview_payload["duration"] == "3.5"
    assert preview_payload["text"] == "hello world"
    assert len(calls) == 1

    preview_audio = client.get("/api/voices/clone/preview-audio")
    assert preview_audio.status_code == 200
    assert preview_audio.headers["content-type"].startswith("audio/wav")

    save = client.post(
        "/api/voices/clone/save",
        json={
            "source": source_path,
            "start": "1",
            "duration": "3.5",
            "name": "narrator",
            "gender": "female",
        },
    )
    assert save.status_code == 200
    save_payload = save.json()
    assert save_payload["status"] == "saved"
    assert save_payload["used_preview"] is True
    assert save_payload["voice"] == {"label": "narrator", "value": "narrator"}
    assert len(calls) == 1

    voice_audio = repo_root / "voices" / "narrator.wav"
    voice_config = repo_root / "voices" / "narrator.json"
    assert voice_audio.exists()
    assert voice_config.exists()
    payload = json.loads(voice_config.read_text(encoding="utf-8"))
    assert payload["ref_audio"] == "narrator.wav"
    assert payload["ref_text"] == "hello world"
    assert payload["gender"] == "female"


def test_run_clone_ffmpeg_applies_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "in.wav"
    output_path = tmp_path / "out.wav"
    seen_cmd: list[str] = []

    def fake_run(cmd: list[str], capture_output: bool, text: bool) -> SimpleNamespace:
        seen_cmd[:] = cmd
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFFFAKEWAVE")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    normalized: list[Path] = []
    monkeypatch.setattr(player_util.subprocess, "run", fake_run)
    monkeypatch.setattr(
        player_util.audio_norm_util,
        "normalize_clone_wav",
        lambda path: normalized.append(path),
    )

    player_util._run_clone_ffmpeg(input_path, output_path, "0", "2")
    assert seen_cmd
    assert normalized == [output_path]


def test_run_clone_ffmpeg_raises_when_normalization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "in.wav"
    output_path = tmp_path / "out.wav"

    def fake_run(cmd: list[str], capture_output: bool, text: bool) -> SimpleNamespace:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFFFAKEWAVE")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    def fail_normalize(path: Path) -> float:
        raise RuntimeError("bad normalize")

    monkeypatch.setattr(player_util.subprocess, "run", fake_run)
    monkeypatch.setattr(player_util.audio_norm_util, "normalize_clone_wav", fail_normalize)

    with pytest.raises(RuntimeError, match="Failed to normalize clone audio"):
        player_util._run_clone_ffmpeg(input_path, output_path, "0", "2")


def test_voice_metadata_update_and_delete(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    voices_dir = repo_root / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_path = voices_dir / "sample.wav"
    voice_path.write_bytes(b"RIFFFAKEWAVE")
    config_path = voices_dir / "sample.json"
    config = voice_util.VoiceConfig(
        name="sample",
        ref_audio="sample.wav",
        ref_text="hello",
        language="Japanese",
        x_vector_only_mode=False,
    )
    voice_util.write_voice_config(config, config_path)

    app = player_util.create_app(root_dir)
    client = TestClient(app)

    save = client.post(
        "/api/voices/metadata",
        json={"voice": "sample", "gender": "male", "name": "Sample2"},
    )
    assert save.status_code == 200
    save_payload = save.json()
    assert save_payload["voice"] == "Sample2"
    assert save_payload["gender"] == "male"
    assert (voices_dir / "sample.wav").exists() is False
    assert (voices_dir / "Sample2.wav").exists()
    assert (voices_dir / "sample.json").exists() is False
    assert (voices_dir / "Sample2.json").exists()
    updated = json.loads((voices_dir / "Sample2.json").read_text(encoding="utf-8"))
    assert updated["ref_audio"] == "Sample2.wav"
    assert updated["gender"] == "male"

    listed = client.get("/api/voices")
    assert listed.status_code == 200
    local_entry = next(
        item for item in listed.json()["local"] if item["value"] == "Sample2"
    )
    assert local_entry["gender"] == "male"

    deleted = client.post(
        "/api/voices/delete",
        json={"voice": "Sample2"},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "deleted", "voice": "Sample2"}
    assert not (voices_dir / "Sample2.wav").exists()
    assert not (voices_dir / "Sample2.json").exists()


def test_synth_chunk_passes_selected_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, root_dir = _make_repo(tmp_path)
    _make_book(root_dir)
    captured: dict = {}

    def fake_synthesize_chunk(**kwargs):
        captured.update(kwargs)
        return {"chapter_id": "c1", "chunk_index": 0}

    monkeypatch.setattr(player_util.tts_util, "synthesize_chunk", fake_synthesize_chunk)

    app = player_util.create_app(root_dir)
    client = TestClient(app)
    response = client.post(
        "/api/synth/chunk",
        json={
            "book_id": "book",
            "chapter_id": "c1",
            "chunk_index": 0,
            "voice": "new-voice",
            "use_voice_map": False,
        },
    )

    assert response.status_code == 200
    assert captured["voice"] == "new-voice"
    assert captured["voice_map_path"] is None


def test_synth_chunk_passes_voice_map_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, root_dir = _make_repo(tmp_path)
    book_dir = _make_book(root_dir)
    (book_dir / "voice-map.json").write_text(
        json.dumps({"default": "advanced-default", "chapters": {"c1": "chapter-voice"}}),
        encoding="utf-8",
    )
    captured: dict = {}

    def fake_synthesize_chunk(**kwargs):
        captured.update(kwargs)
        return {"chapter_id": "c1", "chunk_index": 0}

    monkeypatch.setattr(player_util.tts_util, "synthesize_chunk", fake_synthesize_chunk)

    app = player_util.create_app(root_dir)
    client = TestClient(app)
    response = client.post(
        "/api/synth/chunk",
        json={
            "book_id": "book",
            "chapter_id": "c1",
            "chunk_index": 0,
            "voice": "selected-default",
            "use_voice_map": True,
        },
    )

    assert response.status_code == 200
    assert captured["voice"] == "selected-default"
    assert captured["voice_map_path"] == book_dir / "voice-map.json"


def test_synth_chunk_allowed_while_tts_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root_dir = _make_repo(tmp_path)
    _make_book(root_dir)
    captured: dict[str, object] = {}

    class DummyProcess:
        returncode = None

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int | None:
            _ = timeout
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def fake_synthesize_chunk(**kwargs):
        captured.update(kwargs)
        return {"chapter_id": "c1", "chunk_index": 0}

    monkeypatch.setattr(player_util.tts_util, "synthesize_chunk", fake_synthesize_chunk)
    monkeypatch.setattr(
        player_util.voice_util,
        "resolve_voice_config",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(player_util.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(player_util.subprocess, "Popen", lambda *_args, **_kwargs: DummyProcess())

    app = player_util.create_app(root_dir)
    client = TestClient(app)

    start = client.post(
        "/api/synth/start",
        json={"book_id": "book", "voice": "running-voice"},
    )
    assert start.status_code == 200

    response = client.post(
        "/api/synth/chunk",
        json={
            "book_id": "book",
            "chapter_id": "c1",
            "chunk_index": 0,
            "voice": "regen-voice",
            "use_voice_map": False,
        },
    )

    assert response.status_code == 200
    assert captured["voice"] == "regen-voice"


def test_sanitize_run_uses_cli_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book(root_dir)
    captured: dict[str, object] = {}

    def fake_run(
        cmd: list[str], cwd: str, stdout, stderr: int
    ) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["stderr"] = stderr
        stdout.write("ok\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(player_util.subprocess, "run", fake_run)

    app = player_util.create_app(root_dir)
    client = TestClient(app)
    response = client.post("/api/sanitize/run", json={"book_id": "book"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "tts_cleared": True}
    assert captured["cmd"] == [
        "uv",
        "run",
        "nik",
        "sanitize",
        "--book",
        str(book_dir),
        "--overwrite",
    ]
    assert captured["cwd"] == str(repo_root)
    assert captured["stderr"] == player_util.subprocess.STDOUT


def test_clear_tts_forces_rechunk_with_manifest_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root_dir = _make_repo(tmp_path)
    book_dir = _make_book(root_dir)
    (book_dir / "tts" / "manifest.json").write_text(
        json.dumps(
            {
                "voice": "manifest-voice",
                "max_chars": 321,
                "pad_ms": 444,
                "chunk_mode": "japanese",
                "chapters": [{"id": "c1", "chunks": ["chunk"]}],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_refresh_chunks(
        *,
        book_dir: Path,
        max_chars: int = 220,
        pad_ms: int = 350,
        chunk_mode: str = "japanese",
    ) -> bool:
        captured["book_dir"] = book_dir
        captured["max_chars"] = max_chars
        captured["pad_ms"] = pad_ms
        captured["chunk_mode"] = chunk_mode
        return True

    monkeypatch.setattr(player_util.sanitize, "refresh_chunks", fake_refresh_chunks)

    app = player_util.create_app(root_dir)
    client = TestClient(app)
    response = client.post("/api/tts/clear", json={"book_id": "book"})

    assert response.status_code == 200
    assert response.json() == {"status": "cleared", "book_id": "book", "tts_cleared": True}
    assert captured["book_dir"] == book_dir
    assert captured["max_chars"] == 321
    assert captured["pad_ms"] == 444
    assert captured["chunk_mode"] == "japanese"


def test_chapter_text_prefers_overrides_over_propagated_ruby(tmp_path: Path) -> None:
    _, root_dir = _make_repo(tmp_path)
    book_dir = _make_book(root_dir)
    chapter_text = "事件が事件だ。"
    chapter_rel = "clean/c1.txt"
    (book_dir / chapter_rel).write_text(chapter_text, encoding="utf-8")
    (book_dir / "tts" / "manifest.json").write_text(
        json.dumps(
            {
                "voice": "manifest-voice",
                "chapters": [{"id": "c1", "path": chapter_rel, "chunks": ["chunk"]}],
            }
        ),
        encoding="utf-8",
    )
    chapter_hash = sha256(chapter_text.encode("utf-8")).hexdigest()
    (book_dir / "reading-overrides.json").write_text(
        json.dumps(
            {
                "global": [{"base": "事件", "reading": "じけん"}],
                "chapters": {"c1": {"replacements": [{"base": "事件", "reading": "ヤマ"}]}},
                "ruby": {
                    "global": [{"base": "事件", "reading": "ヤマ"}],
                    "chapters": {
                        "c1": {
                            "clean_sha256": chapter_hash,
                            "clean_spans": [
                                {"start": 0, "end": 2, "base": "事件", "reading": "ヤマ"}
                            ],
                        }
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    app = player_util.create_app(root_dir)
    client = TestClient(app)
    response = client.get("/api/books/book/chapter-text", params={"chapter_id": "c1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["clean_text"] == chapter_text
    assert payload["ruby_spans"] == [
        {
            "start": 0,
            "end": 2,
            "base": "事件",
            "reading": "ヤマ",
            "kind": "inline",
        }
    ]
    assert payload["ruby_prop_spans"] == [
        {
            "start": 3,
            "end": 5,
            "base": "事件",
            "reading": "じけん",
            "kind": "propagated",
        }
    ]
