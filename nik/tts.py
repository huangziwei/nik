from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import voice as voice_util
from .text import read_clean_text
from .voice import VoiceConfig

torch = None
Qwen3TTSModel = None

_END_PUNCT = set("。！？?!…")
_MID_PUNCT = set("、，,;；:：")
_CLOSE_PUNCT = set("」』）】］〉》」』”’\"'")


@dataclass(frozen=True)
class ChapterInput:
    index: int
    id: str
    title: str
    text: str
    path: Optional[str] = None


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def chapter_id_from_path(index: int, title: str, rel_path: Optional[str]) -> str:
    if rel_path:
        stem = Path(rel_path).stem
        if stem:
            return stem
    return f"{index:04d}-chapter"


def _lazy_import_tts() -> None:
    global torch, Qwen3TTSModel
    if torch is not None and Qwen3TTSModel is not None:
        return
    try:
        import torch as _torch
        from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "qwen-tts dependencies are missing. Install qwen-tts + torch, "
            "or run with uv after adding them to pyproject.toml."
        ) from exc
    torch = _torch
    Qwen3TTSModel = _Qwen3TTSModel


def _require_tts() -> None:
    _lazy_import_tts()


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _trim_span(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _advance_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == "\n":
            span = _trim_span(text, start, i)
            if span:
                spans.append(span)
            i += 1
            i = _advance_ws(text, i)
            start = i
            continue
        if ch in _END_PUNCT:
            j = i + 1
            while j < length and text[j] in _END_PUNCT:
                j += 1
            while j < length and text[j] in _CLOSE_PUNCT:
                j += 1
            span = _trim_span(text, start, j)
            if span:
                spans.append(span)
            j = _advance_ws(text, j)
            start = j
            i = j
            continue
        i += 1
    span = _trim_span(text, start, length)
    if span:
        spans.append(span)
    return spans


def _split_long_span(
    text: str, start: int, end: int, max_chars: int
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    i = start
    while i < end:
        j = min(end, i + max_chars)
        if j < end:
            split_at = -1
            for punct in _MID_PUNCT:
                idx = text.rfind(punct, i, j)
                if idx > split_at:
                    split_at = idx
            if split_at > i:
                j = split_at + 1
        span = _trim_span(text, i, j)
        if span:
            spans.append(span)
        i = _advance_ws(text, j)
    return spans


def make_chunk_spans(
    text: str, max_chars: int, chunk_mode: str = "japanese"
) -> List[Tuple[int, int]]:
    _ = chunk_mode
    spans: List[Tuple[int, int]] = []
    for sent_start, sent_end in split_sentence_spans(text):
        if max_chars > 0 and sent_end - sent_start > max_chars:
            spans.extend(_split_long_span(text, sent_start, sent_end, max_chars))
        else:
            spans.append((sent_start, sent_end))
    return spans


def make_chunks(text: str, max_chars: int, chunk_mode: str = "japanese") -> List[str]:
    spans = make_chunk_spans(text, max_chars=max_chars, chunk_mode=chunk_mode)
    return [text[start:end] for start, end in spans]


def prepare_tts_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_book_chapters(book_dir: Path) -> List[ChapterInput]:
    toc_path = book_dir / "clean" / "toc.json"
    if not toc_path.exists():
        toc_path = book_dir / "toc.json"
    if not toc_path.exists():
        raise FileNotFoundError(f"Missing toc.json at {toc_path}")

    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    entries = toc.get("chapters", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("toc.json contains no chapters.")

    chapters: List[ChapterInput] = []
    for fallback_idx, entry in enumerate(entries, start=1):
        rel = entry.get("path")
        if not rel:
            continue
        path = book_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing chapter file: {path}")

        text = read_clean_text(path)
        if not text.strip():
            continue

        index = int(entry.get("index") or fallback_idx)
        title = str(entry.get("title") or f"Chapter {index}")
        chapter_id = chapter_id_from_path(index, title, rel)

        chapters.append(
            ChapterInput(
                index=index,
                id=chapter_id,
                title=title,
                text=_normalize_text(text),
                path=rel,
            )
        )

    if not chapters:
        raise ValueError("No chapter text found.")

    return chapters


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_status(out_dir: Path, stage: str, detail: Optional[str] = None) -> None:
    payload = {"stage": stage, "updated_unix": int(time.time())}
    if detail:
        payload["detail"] = detail
    atomic_write_json(out_dir / "status.json", payload)


def _load_voice_map(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Voice map not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Voice map must be a JSON object: {path}")
    chapters = data.get("chapters", {})
    if not isinstance(chapters, dict):
        chapters = {}
    return {
        "default": data.get("default"),
        "chapters": chapters,
    }


def _normalize_voice_id(value: Optional[str], default_voice: str) -> str:
    if value is None:
        return default_voice
    cleaned = str(value).strip()
    if not cleaned:
        return default_voice
    if cleaned.lower() == "default":
        return default_voice
    return cleaned


def write_chunk_files(
    chunks: Sequence[str], chunk_dir: Path, overwrite: bool = False
) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in chunk_dir.glob("*.txt"):
            path.unlink()

    for idx, chunk in enumerate(chunks, start=1):
        path = chunk_dir / f"{idx:06d}.txt"
        if overwrite or not path.exists():
            path.write_text(chunk.rstrip() + "\n", encoding="utf-8")

    if overwrite:
        for path in chunk_dir.glob("*.txt"):
            stem = path.stem
            if stem.isdigit() and int(stem) > len(chunks):
                path.unlink()


def _prepare_manifest(
    chapters: Sequence[ChapterInput],
    out_dir: Path,
    voice: str,
    max_chars: int,
    pad_ms: int,
    rechunk: bool,
) -> Tuple[dict, List[List[str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = out_dir / "chunks"
    manifest_path = out_dir / "manifest.json"

    if rechunk and chunk_root.exists():
        shutil.rmtree(chunk_root)

    if manifest_path.exists() and not rechunk:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("chunk_mode") != "japanese":
            raise ValueError(
                "manifest.json chunk_mode differs from requested. "
                "Run with --rechunk to regenerate."
            )
        if int(manifest.get("max_chars") or 0) != int(max_chars):
            raise ValueError(
                "manifest.json max_chars differs from requested. "
                "Run with --rechunk to regenerate."
            )
        manifest_chapters = manifest.get("chapters", [])
        if not isinstance(manifest_chapters, list) or not manifest_chapters:
            raise ValueError("manifest.json contains no chapters.")
        if len(manifest_chapters) != len(chapters):
            raise ValueError(
                "manifest.json chapter count differs from input. "
                "Run with --rechunk to regenerate."
            )
        chapter_chunks: List[List[str]] = []
        for ch_manifest, chapter in zip(manifest_chapters, chapters):
            if ch_manifest.get("text_sha256") != sha256_str(chapter.text):
                raise ValueError(
                    "manifest.json text hash differs from input. "
                    "Run with --rechunk to regenerate."
                )
            chunks = ch_manifest.get("chunks", [])
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("manifest.json missing chunks. Run with --rechunk.")
            chapter_chunks.append([str(c) for c in chunks])
        manifest["voice"] = voice
        manifest["pad_ms"] = int(pad_ms)
        atomic_write_json(manifest_path, manifest)
        return manifest, chapter_chunks

    chapter_chunks: List[List[str]] = []
    manifest_chapters: List[dict] = []
    for ch in chapters:
        spans = make_chunk_spans(ch.text, max_chars=max_chars, chunk_mode="japanese")
        chunks = [ch.text[start:end] for start, end in spans]
        if not chunks:
            raise ValueError(f"No chunks generated for chapter: {ch.id}")
        chapter_chunks.append(chunks)
        manifest_chapters.append(
            {
                "index": ch.index,
                "id": ch.id,
                "title": ch.title,
                "path": ch.path,
                "text_sha256": sha256_str(ch.text),
                "chunks": chunks,
                "chunk_spans": [[start, end] for start, end in spans],
                "durations_ms": [None] * len(chunks),
            }
        )

    manifest = {
        "created_unix": int(time.time()),
        "voice": voice,
        "max_chars": int(max_chars),
        "pad_ms": int(pad_ms),
        "chunk_mode": "japanese",
        "chapters": manifest_chapters,
    }
    atomic_write_json(manifest_path, manifest)

    for ch_entry, chunks in zip(manifest["chapters"], chapter_chunks):
        chunk_dir = chunk_root / ch_entry["id"]
        write_chunk_files(chunks, chunk_dir, overwrite=True)

    return manifest, chapter_chunks


def _resolve_dtype(value: Optional[str]) -> Optional["torch.dtype"]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key or key == "auto":
        return None
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _load_model(
    model_name: str,
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> "Qwen3TTSModel":
    _require_tts()
    kwargs: Dict[str, Any] = {}
    if device_map:
        kwargs["device_map"] = device_map
    torch_dtype = _resolve_dtype(dtype)
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
    _ensure_pad_token_id(model)
    return model


def _ensure_pad_token_id(model: "Qwen3TTSModel") -> None:
    core = getattr(model, "model", None)
    if core is None:
        return
    gen_cfg = getattr(core, "generation_config", None)
    cfg = getattr(core, "config", None)
    eos = None
    if gen_cfg is not None:
        eos = getattr(gen_cfg, "eos_token_id", None)
        if getattr(gen_cfg, "pad_token_id", None) is None and eos is not None:
            gen_cfg.pad_token_id = eos
    if cfg is not None:
        eos = eos if eos is not None else getattr(cfg, "eos_token_id", None)
        if getattr(cfg, "pad_token_id", None) is None and eos is not None:
            cfg.pad_token_id = eos


def _prepare_voice_prompt(
    model: "Qwen3TTSModel", voice: VoiceConfig
) -> Tuple[Any, str]:
    if not voice.audio_path.exists():
        raise FileNotFoundError(f"Voice audio not found: {voice.audio_path}")
    ref_text = voice.ref_text
    if not ref_text and not voice.x_vector_only_mode:
        raise ValueError("Voice config is missing ref_text.")
    kwargs: Dict[str, Any] = {
        "ref_audio": str(voice.audio_path),
        "ref_text": ref_text,
        "language": voice.language,
        "x_vector_only_mode": voice.x_vector_only_mode,
    }
    create_fn = model.create_voice_clone_prompt
    sig = inspect.signature(create_fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    prompt = create_fn(**filtered)
    return prompt, voice.language


def _call_with_supported_kwargs(fn, kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _generate_audio(
    model: "Qwen3TTSModel",
    text: str,
    prompt: Any,
    language: Optional[str],
) -> Tuple[List[np.ndarray], int]:
    if hasattr(model, "generate_voice_clone"):
        kwargs = {
            "text": text,
            "language": language,
            "voice_clone_prompt": prompt,
            "non_streaming_mode": True,
        }
        return _call_with_supported_kwargs(model.generate_voice_clone, kwargs)
    if hasattr(model, "generate"):
        kwargs = {"text": text, "prompt": prompt, "language": language}
        return _call_with_supported_kwargs(model.generate, kwargs)
    raise AttributeError("Qwen3TTSModel has no supported generate method.")


def _is_valid_wav(path: Path) -> bool:
    try:
        with sf.SoundFile(path) as handle:
            return handle.channels == 1 and handle.frames > 0
    except Exception:
        return False


def _wav_duration_ms(path: Path) -> int:
    with sf.SoundFile(path) as handle:
        frames = handle.frames
        rate = handle.samplerate
    if rate <= 0:
        return 0
    return int(round(frames * 1000.0 / rate))


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def build_concat_file(segment_paths: List[Path], concat_path: Path, base_dir: Path) -> None:
    lines = []
    for p in segment_paths:
        rel = p.relative_to(base_dir).as_posix()
        lines.append(f"file '{rel}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chapters_ffmeta(
    chapters: Sequence[Tuple[str, int]], ffmeta_path: Path
) -> None:
    def _ffmeta_escape(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("\n", "\\\n")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#")
        )

    out = [";FFMETADATA1"]
    t = 0
    for title, d in chapters:
        start = t
        end = t + max(int(d), 1)
        out.append("")
        out.append("[CHAPTER]")
        out.append("TIMEBASE=1/1000")
        out.append(f"START={start}")
        out.append(f"END={end}")
        out.append(f"title={_ffmeta_escape(str(title))}")
        t = end
    ffmeta_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def chunk_book(
    book_dir: Path,
    out_dir: Optional[Path] = None,
    voice: str = "voice",
    max_chars: int = 220,
    pad_ms: int = 120,
    chunk_mode: str = "japanese",
    rechunk: bool = True,
) -> dict:
    _ = chunk_mode
    if out_dir is None:
        out_dir = book_dir / "tts"
    chapters = load_book_chapters(book_dir)
    manifest, _chapter_chunks = _prepare_manifest(
        chapters=chapters,
        out_dir=out_dir,
        voice=voice,
        max_chars=max_chars,
        pad_ms=pad_ms,
        rechunk=rechunk,
    )
    return manifest


def synthesize_book(
    book_dir: Path,
    voice: VoiceConfig,
    out_dir: Optional[Path] = None,
    max_chars: int = 220,
    pad_ms: int = 120,
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    only_chapter_ids: Optional[set[str]] = None,
) -> int:
    chapters = load_book_chapters(book_dir)
    if out_dir is None:
        out_dir = book_dir / "tts"
    if base_dir is None:
        base_dir = Path.cwd()

    try:
        voice_map = _load_voice_map(voice_map_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        manifest, chapter_chunks = _prepare_manifest(
            chapters=chapters,
            out_dir=out_dir,
            voice=voice.name,
            max_chars=max_chars,
            pad_ms=pad_ms,
            rechunk=rechunk,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    seg_dir = out_dir / "segments"
    manifest_path = out_dir / "manifest.json"
    concat_path = out_dir / "concat.txt"
    chapters_path = out_dir / "chapters.ffmeta"
    if rechunk and seg_dir.exists():
        shutil.rmtree(seg_dir)

    default_voice = voice.name
    if voice_map:
        default_voice = _normalize_voice_id(voice_map.get("default"), default_voice)

    chapter_voice_map: Dict[str, str] = {}
    voice_overrides: Dict[str, str] = {}
    raw_overrides = voice_map.get("chapters", {}) if voice_map else {}
    for entry in manifest.get("chapters", []):
        chapter_id = entry.get("id") or "chapter"
        raw_value = raw_overrides.get(chapter_id) if isinstance(raw_overrides, dict) else None
        selected = _normalize_voice_id(raw_value, default_voice)
        chapter_voice_map[chapter_id] = selected
        entry["voice"] = selected
        if voice_map and selected != default_voice:
            voice_overrides[chapter_id] = selected
    manifest["voice"] = default_voice
    if voice_map:
        manifest["voice_overrides"] = voice_overrides
    manifest["pad_ms"] = int(pad_ms)
    atomic_write_json(manifest_path, manifest)

    voice_configs: Dict[str, VoiceConfig] = {}
    try:
        for voice_id in sorted(set(chapter_voice_map.values())):
            if voice_id == voice.name:
                voice_configs[voice_id] = voice
            else:
                voice_configs[voice_id] = voice_util.resolve_voice_config(
                    voice=voice_id,
                    base_dir=base_dir,
                )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    write_status(out_dir, "cloning", "Preparing voice")

    model = _load_model(
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    voice_prompts: Dict[str, Any] = {}
    voice_languages: Dict[str, str] = {}
    for voice_id, config in voice_configs.items():
        prompt, language = _prepare_voice_prompt(model, config)
        voice_prompts[voice_id] = prompt
        voice_languages[voice_id] = language

    write_status(out_dir, "synthesizing")

    selected_ids = set(only_chapter_ids) if only_chapter_ids else None
    selected_indices = [
        idx
        for idx, entry in enumerate(manifest["chapters"])
        if not selected_ids or (entry.get("id") or "chapter") in selected_ids
    ]
    if selected_ids and not selected_indices:
        sys.stderr.write("No matching chapters found for synthesis.\n")
        return 2

    total_chunks = sum(len(chapter_chunks[idx]) for idx in selected_indices)
    if total_chunks <= 0:
        sys.stderr.write("No chunks selected for synthesis.\n")
        return 2

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    segment_paths: List[Path] = []

    with progress:
        overall_task = progress.add_task("Total", total=total_chunks)
        chapter_task = progress.add_task("Chapter", total=0)

        for ch_entry, chunks in zip(manifest["chapters"], chapter_chunks):
            chapter_id = ch_entry.get("id") or "chapter"
            chapter_title = ch_entry.get("title") or chapter_id
            chapter_total = len(chunks)
            if selected_ids and chapter_id not in selected_ids:
                continue
            progress.update(
                chapter_task,
                total=chapter_total,
                completed=0,
                description=f"{chapter_id}: {chapter_title}",
            )

            chapter_seg_dir = seg_dir / chapter_id
            voice_id = chapter_voice_map.get(chapter_id, default_voice)
            prompt = voice_prompts[voice_id]
            language = voice_languages[voice_id]

            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                seg_path = chapter_seg_dir / f"{chunk_idx:06d}.wav"
                progress.update(
                    chapter_task,
                    description=f"{chapter_id}: {chapter_title} ({chunk_idx}/{chapter_total})",
                )

                if seg_path.exists() and _is_valid_wav(seg_path):
                    segment_paths.append(seg_path)
                    dms = _wav_duration_ms(seg_path)
                    if ch_entry["durations_ms"][chunk_idx - 1] != dms:
                        ch_entry["durations_ms"][chunk_idx - 1] = dms
                        atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                tts_text = prepare_tts_text(chunk_text)
                if not tts_text:
                    ch_entry["durations_ms"][chunk_idx - 1] = 0
                    atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                wavs, sample_rate = _generate_audio(
                    model=model,
                    text=tts_text,
                    prompt=prompt,
                    language=language,
                )
                if not wavs:
                    sys.stderr.write(
                        f"Skipping empty audio {chapter_id} ({chunk_idx}/{chapter_total}).\n"
                    )
                    ch_entry["durations_ms"][chunk_idx - 1] = 0
                    atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                if isinstance(wavs, np.ndarray):
                    audio = wavs.flatten()
                else:
                    audio = np.concatenate([np.asarray(w).flatten() for w in wavs])

                if pad_ms > 0:
                    pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
                    if pad_samples > 0:
                        pad = np.zeros(pad_samples, dtype=audio.dtype)
                        audio = np.concatenate([audio, pad])

                _write_wav(seg_path, audio, sample_rate)
                segment_paths.append(seg_path)

                dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
                ch_entry["durations_ms"][chunk_idx - 1] = dms
                if manifest.get("sample_rate") != sample_rate:
                    manifest["sample_rate"] = sample_rate
                atomic_write_json(manifest_path, manifest)

                progress.advance(chapter_task, 1)
                progress.advance(overall_task, 1)

    build_concat_file(segment_paths, concat_path, base_dir=out_dir)

    chapter_meta: List[Tuple[str, int]] = []
    for ch_entry in manifest["chapters"]:
        title = ch_entry.get("title") or ch_entry.get("id") or "Chapter"
        durations = ch_entry.get("durations_ms", [])
        total_ms = sum(int(d or 0) for d in durations)
        chapter_meta.append((title, total_ms))

    build_chapters_ffmeta(chapter_meta, chapters_path)
    write_status(out_dir, "done")
    return 0


def synthesize_book_sample(
    book_dir: Path,
    voice: VoiceConfig,
    out_dir: Optional[Path] = None,
    max_chars: int = 220,
    pad_ms: int = 120,
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> int:
    chapters = load_book_chapters(book_dir)
    if not chapters:
        sys.stderr.write("No chapters found for sampling.\n")
        return 2
    if out_dir is None:
        out_dir = book_dir / "tts"

    sample_id = chapters[0].id
    sample_dir = out_dir / "segments" / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        chapters_meta = manifest.get("chapters")
        if isinstance(chapters_meta, list):
            for entry in chapters_meta:
                if entry.get("id") == sample_id:
                    chunks = entry.get("chunks")
                    if isinstance(chunks, list):
                        entry["durations_ms"] = [None] * len(chunks)
                    break
            atomic_write_json(manifest_path, manifest)

    return synthesize_book(
        book_dir=book_dir,
        voice=voice,
        out_dir=out_dir,
        max_chars=max_chars,
        pad_ms=pad_ms,
        rechunk=rechunk,
        voice_map_path=voice_map_path,
        base_dir=base_dir,
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
        only_chapter_ids={sample_id},
    )


def synthesize_chunk(
    out_dir: Path,
    chapter_id: str,
    chunk_index: int,
    voice: Optional[str] = None,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> dict:
    _require_tts()
    if base_dir is None:
        base_dir = Path.cwd()
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chapters = manifest.get("chapters", [])
    if not isinstance(chapters, list):
        raise ValueError("manifest.json chapters missing or invalid")

    entry = None
    for item in chapters:
        if isinstance(item, dict) and item.get("id") == chapter_id:
            entry = item
            break
    if entry is None:
        raise ValueError(f"Unknown chapter_id: {chapter_id}")

    chunks = entry.get("chunks")
    if not isinstance(chunks, list):
        chunks = []
    spans = entry.get("chunk_spans")
    if not isinstance(spans, list):
        spans = []
    chunk_count = len(chunks) or len(spans)
    if chunk_count <= 0:
        chunk_dir = out_dir / "chunks" / chapter_id
        if chunk_dir.exists():
            chunk_count = len([p for p in chunk_dir.glob("*.txt") if p.stem.isdigit()])
    if chunk_count <= 0:
        raise ValueError(f"No chunks available for chapter: {chapter_id}")
    if chunk_index >= chunk_count:
        raise ValueError(f"chunk_index out of range for {chapter_id}")

    chunk_text: Optional[str] = None
    if chunks and chunk_index < len(chunks):
        chunk_text = str(chunks[chunk_index])
    if chunk_text is None:
        chunk_path = out_dir / "chunks" / chapter_id / f"{chunk_index + 1:06d}.txt"
        if chunk_path.exists():
            chunk_text = chunk_path.read_text(encoding="utf-8").rstrip("\n")
    if chunk_text is None:
        raise ValueError(f"Chunk text missing for {chapter_id} #{chunk_index + 1}")

    durations = entry.get("durations_ms")
    if not isinstance(durations, list) or len(durations) != chunk_count:
        durations = [None] * chunk_count
        entry["durations_ms"] = durations

    default_voice = voice or entry.get("voice") or manifest.get("voice") or ""
    if not default_voice:
        raise ValueError("Voice is required for synthesis.")

    voice_id = default_voice
    if voice_map_path:
        voice_map = _load_voice_map(voice_map_path)
        if voice_map:
            default_voice = _normalize_voice_id(voice_map.get("default"), default_voice)
            raw_voice = (
                voice_map.get("chapters", {}).get(chapter_id)
                if isinstance(voice_map.get("chapters"), dict)
                else None
            )
            voice_id = _normalize_voice_id(raw_voice, default_voice)
    else:
        voice_id = _normalize_voice_id(entry.get("voice"), default_voice)

    config = voice_util.resolve_voice_config(voice=voice_id, base_dir=base_dir)
    model = _load_model(
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    prompt, language = _prepare_voice_prompt(model, config)

    tts_text = prepare_tts_text(chunk_text)
    seg_path = out_dir / "segments" / chapter_id / f"{chunk_index + 1:06d}.wav"
    seg_path.parent.mkdir(parents=True, exist_ok=True)

    if not tts_text:
        if seg_path.exists():
            seg_path.unlink()
        durations[chunk_index] = 0
        atomic_write_json(manifest_path, manifest)
        return {
            "status": "skipped",
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "duration_ms": 0,
        }

    wavs, sample_rate = _generate_audio(
        model=model,
        text=tts_text,
        prompt=prompt,
        language=language,
    )
    if not wavs:
        durations[chunk_index] = 0
        atomic_write_json(manifest_path, manifest)
        return {
            "status": "skipped",
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "duration_ms": 0,
        }

    if isinstance(wavs, np.ndarray):
        audio = wavs.flatten()
    else:
        audio = np.concatenate([np.asarray(w).flatten() for w in wavs])

    pad_ms = int(manifest.get("pad_ms") or 0)
    if pad_ms > 0:
        pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
        if pad_samples > 0:
            pad = np.zeros(pad_samples, dtype=audio.dtype)
            audio = np.concatenate([audio, pad])

    _write_wav(seg_path, audio, sample_rate)
    dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
    durations[chunk_index] = dms
    if manifest.get("sample_rate") != sample_rate:
        manifest["sample_rate"] = sample_rate
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "ok",
        "chapter_id": chapter_id,
        "chunk_index": chunk_index,
        "duration_ms": dms,
    }
