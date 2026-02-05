from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from . import asr as asr_util
from . import epub as epub_util
from . import merge as merge_util
from . import sanitize as sanitize_util
from . import tts as tts_util
from . import voice as voice_util
from .text import read_clean_text, guess_title_from_path


def _summarize_ruby_pairs(
    pairs: list[tuple[str, str]],
) -> Optional[dict]:
    if not pairs:
        return None
    readings: dict[str, set[str]] = {}
    for base, reading in pairs:
        base_text = str(base).strip()
        reading_text = str(reading).strip()
        if not base_text or not reading_text:
            continue
        readings.setdefault(base_text, set()).add(reading_text)
    if not readings:
        return None
    replacements = []
    conflicts = []
    for base, values in sorted(
        readings.items(), key=lambda item: (-len(item[0]), item[0])
    ):
        if len(values) == 1:
            replacements.append({"base": base, "reading": sorted(values)[0]})
        else:
            conflicts.append({"base": base, "readings": sorted(values)})
    if not replacements and not conflicts:
        return None
    return {"replacements": replacements, "conflicts": conflicts}


def _ingest_epub(input_path: Path, out_dir: Path, raw_dir: Path) -> int:
    book = epub_util.read_epub(input_path)
    metadata = epub_util.extract_metadata(book)
    cover = epub_util.extract_cover_image(book)
    if cover:
        cover_path = _write_cover_image(cover, out_dir)
        if cover_path:
            cover_info = metadata.get("cover") or {}
            cover_info.setdefault("id", cover.get("id", ""))
            cover_info.setdefault("href", cover.get("href", ""))
            cover_info["path"] = cover_path.relative_to(out_dir).as_posix()
            cover_info["media_type"] = cover.get("media_type") or cover_info.get(
                "media_type", ""
            )
            metadata["cover"] = cover_info
    chapters = epub_util.extract_chapters(book, prefer_toc=True)

    if not chapters:
        sys.stderr.write("No chapters found in EPUB.\n")
        return 2

    toc_items = []
    ruby_overrides: dict[str, dict] = {}
    ruby_chapters: dict[str, dict] = {}
    ruby_counts: dict[str, dict[str, int]] = {}
    ruby_order: dict[str, list[str]] = {}
    for idx, chapter in enumerate(chapters, start=1):
        title = chapter.title or f"Chapter {idx}"
        slug = epub_util.slugify(title)
        filename = f"{idx:04d}-{slug}.txt"
        out_path = raw_dir / filename

        out_path.write_text(chapter.text.rstrip() + "\n", encoding="utf-8")
        chapter_id = Path(filename).stem
        summary = _summarize_ruby_pairs(chapter.ruby_pairs)
        if summary:
            ruby_overrides[chapter_id] = summary
        if chapter.ruby_spans:
            raw_for_hash = tts_util._normalize_text(read_clean_text(out_path))
            ruby_chapters[chapter_id] = {
                "raw_sha256": tts_util.sha256_str(raw_for_hash),
                "raw_spans": chapter.ruby_spans,
            }
        for base, reading in chapter.ruby_pairs:
            base_text = str(base).strip()
            reading_text = str(reading).strip()
            if not base_text or not reading_text:
                continue
            ruby_counts.setdefault(base_text, {})
            ruby_counts[base_text][reading_text] = (
                ruby_counts[base_text].get(reading_text, 0) + 1
            )
            ruby_order.setdefault(base_text, [])
            if reading_text not in ruby_order[base_text]:
                ruby_order[base_text].append(reading_text)
        toc_items.append(
            {
                "index": idx,
                "title": title,
                "href": chapter.href,
                "source": chapter.source,
                "path": out_path.relative_to(out_dir).as_posix(),
            }
        )

    toc_data = {
        "created_unix": int(time.time()),
        "source_epub": str(input_path),
        "metadata": metadata,
        "chapters": toc_items,
    }

    toc_path = out_dir / "toc.json"
    toc_path.write_text(
        json.dumps(toc_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if ruby_overrides:
        overrides_path = out_dir / "reading-overrides.json"
        ruby_global: list[dict] = []
        ruby_conflicts: list[dict] = []
        for base, counts in sorted(ruby_counts.items(), key=lambda item: item[0]):
            total = sum(counts.values())
            order = {reading: idx for idx, reading in enumerate(ruby_order.get(base, []))}
            majority_reading = ""
            majority_count = -1
            for reading, count in counts.items():
                rank = order.get(reading, len(order))
                if (count > majority_count) or (
                    count == majority_count and rank < order.get(majority_reading, 1 << 30)
                ):
                    majority_reading = reading
                    majority_count = count
            if majority_reading:
                ruby_global.append(
                    {
                        "base": base,
                        "reading": majority_reading,
                        "count": majority_count,
                        "total": total,
                    }
                )
            if len(counts) > 1:
                readings = [
                    {"reading": reading, "count": count}
                    for reading, count in sorted(
                        counts.items(),
                        key=lambda item: (-item[1], order.get(item[0], 1 << 30)),
                    )
                ]
                ruby_conflicts.append(
                    {
                        "base": base,
                        "majority": majority_reading,
                        "readings": readings,
                    }
                )
        overrides_payload = {
            "created_unix": int(time.time()),
            "source_epub": str(input_path),
            "chapters": ruby_overrides,
            "ruby": {
                "global": ruby_global,
                "conflicts": ruby_conflicts,
                "chapters": ruby_chapters,
            },
        }
        overrides_path.write_text(
            json.dumps(overrides_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Reading overrides saved to {overrides_path}")

    print(f"Wrote {len(toc_items)} chapters to {raw_dir}")
    print(f"TOC metadata saved to {toc_path}")
    return 0


def _ingest_txt(input_path: Path, out_dir: Path, raw_dir: Path) -> int:
    try:
        text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = input_path.read_text(encoding="utf-8", errors="replace")
    title = guess_title_from_path(input_path)
    chapter_title = title or "Chapter 1"
    slug = epub_util.slugify(chapter_title)
    filename = f"0001-{slug}.txt"
    out_path = raw_dir / filename
    out_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    toc_items = [
        {
            "index": 1,
            "title": chapter_title,
            "href": "",
            "source": "",
            "path": out_path.relative_to(out_dir).as_posix(),
        }
    ]
    toc_data = {
        "created_unix": int(time.time()),
        "source_epub": str(input_path),
        "metadata": {
            "title": title,
            "authors": [],
            "language": "",
            "dates": [],
            "year": "",
            "cover": None,
        },
        "chapters": toc_items,
    }
    toc_path = out_dir / "toc.json"
    toc_path.write_text(
        json.dumps(toc_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(toc_items)} chapters to {raw_dir}")
    print(f"TOC metadata saved to {toc_path}")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Input file not found: {input_path}\n")
        return 2

    suffix = input_path.suffix.lower()
    if suffix not in {".epub", ".txt"}:
        sys.stderr.write("Only .epub or .txt files are supported.\n")
        return 2

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw" / "chapters"

    if raw_dir.exists():
        existing = [p for p in raw_dir.iterdir() if p.is_file()]
        if existing and not args.overwrite:
            sys.stderr.write(
                "Raw chapters already exist. Use --overwrite to regenerate.\n"
            )
            return 2

    raw_dir.mkdir(parents=True, exist_ok=True)
    if suffix == ".txt":
        return _ingest_txt(input_path, out_dir, raw_dir)
    return _ingest_epub(input_path, out_dir, raw_dir)


def _cover_extension(media_type: str, href: str) -> str:
    media_type = (media_type or "").lower()
    if media_type == "image/jpeg":
        return ".jpg"
    if media_type == "image/png":
        return ".png"
    if media_type == "image/webp":
        return ".webp"
    if media_type == "image/gif":
        return ".gif"
    if href:
        suffix = Path(href).suffix.lower()
        if suffix:
            return suffix
    return ".jpg"


def _write_cover_image(cover: dict, out_dir: Path) -> Optional[Path]:
    data = cover.get("data")
    if not data:
        return None
    ext = _cover_extension(cover.get("media_type", ""), cover.get("href", ""))
    path = out_dir / f"cover{ext}"
    path.write_bytes(data)
    return path


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_to(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "nik voice clone"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _build_clone_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    start: str,
    duration: str,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        start,
        "-t",
        duration,
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def _clone(args: argparse.Namespace) -> int:
    source = str(args.source)
    repo_root = voice_util.find_repo_root(Path.cwd())
    voices_dir = repo_root / voice_util.VOICE_DIRNAME
    voices_dir.mkdir(parents=True, exist_ok=True)
    start = args.start or "00:00:00"
    duration_value = float(args.duration or 0)
    if duration_value <= 0:
        sys.stderr.write("--duration must be a positive number of seconds.\n")
        return 2
    duration = str(duration_value)

    if shutil.which("ffmpeg") is None:
        sys.stderr.write("ffmpeg not found on PATH.\n")
        return 2

    voice_text = None
    if args.text_file:
        voice_text = Path(args.text_file).read_text(encoding="utf-8").strip()
    elif args.text:
        voice_text = str(args.text).strip()

    if not voice_text and not args.x_vector_only and not args.auto_text:
        sys.stderr.write(
            "Reference text is required. Provide --text/--text-file, enable Whisper "
            "auto-text, or use --x-vector-only.\n"
        )
        return 2

    output_name = voice_util.coerce_voice_name(args.name, Path(source).stem)
    output_path = voices_dir / f"{output_name}.wav"
    config_path = voices_dir / f"{output_name}.json"

    if _is_http_url(source):
        parsed = urllib.parse.urlparse(source)
        filename = Path(parsed.path).name or "voice.mp3"
        if not Path(filename).suffix:
            filename = f"{filename}.mp3"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / filename
            try:
                _download_to(source, input_path)
            except Exception as exc:
                sys.stderr.write(f"Download failed: {exc}\n")
                return 2
            cmd = _build_clone_ffmpeg_cmd(input_path, output_path, start, duration)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.stderr.write("ffmpeg failed to process the audio.\n")
                return 2
    else:
        input_path = Path(source).expanduser()
        if not input_path.exists():
            sys.stderr.write(f"Input file not found: {input_path}\n")
            return 2
        cmd = _build_clone_ffmpeg_cmd(input_path, output_path, start, duration)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.stderr.write("ffmpeg failed to process the audio.\n")
            return 2

    if not voice_text and args.auto_text:
        whisper_language = args.whisper_language or args.language
        try:
            voice_text = asr_util.transcribe_audio(
                output_path,
                model_name=args.whisper_model,
                language=whisper_language,
                device=args.whisper_device,
                initial_prompt=args.whisper_prompt,
            )
        except Exception as exc:
            sys.stderr.write(f"Whisper transcription failed: {exc}\n")
            return 2
        if not voice_text:
            sys.stderr.write(
                "Whisper returned empty text; provide --text/--text-file instead.\n"
            )
            return 2

    if not voice_text and not args.x_vector_only:
        sys.stderr.write(
            "Reference text is required. Provide --text/--text-file, enable Whisper "
            "auto-text, or use --x-vector-only.\n"
        )
        return 2

    config = voice_util.VoiceConfig(
        name=output_name,
        ref_audio=output_path.name,
        ref_text=voice_text,
        language=args.language or voice_util.DEFAULT_LANGUAGE,
        x_vector_only_mode=bool(args.x_vector_only),
    )
    voice_util.write_voice_config(config, config_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {config_path}")
    return 0


def _sanitize(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    rules_path = Path(args.rules) if args.rules else None
    try:
        written = sanitize_util.sanitize_book(
            book_dir=book_dir,
            rules_path=rules_path,
            overwrite=args.overwrite,
        )
        tts_cleared = sanitize_util.refresh_chunks(book_dir=book_dir)
    except Exception as exc:
        sys.stderr.write(f"Sanitize failed: {exc}\n")
        return 2

    print(f"Wrote {written} cleaned chapters to {book_dir / 'clean' / 'chapters'}")
    print(f"Report saved to {book_dir / 'clean' / 'report.json'}")
    if tts_cleared:
        print("Cleared TTS cache and prepared chunks.")
    else:
        print("Prepared chunks for TTS.")
    return 0


def _merge(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    output_path = Path(args.output)
    try:
        return merge_util.merge_book(
            book_dir=book_dir,
            output_path=output_path,
            bitrate=args.bitrate,
            overwrite=args.overwrite,
            progress_path=Path(args.progress_file) if args.progress_file else None,
            split_hours=args.split_hours,
            split_count=args.split_count,
        )
    except Exception as exc:
        sys.stderr.write(f"Merge failed: {exc}\n")
        return 2


def _play(args: argparse.Namespace) -> int:
    player_util = importlib.import_module("nik.player")
    player_util.run(Path(args.root), host=args.host, port=args.port)
    return 0


def _synth(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    out_dir = Path(args.out) if args.out else None
    voice_text_path = Path(args.voice_text_file) if args.voice_text_file else None
    voice_map_path = Path(args.voice_map) if args.voice_map else None

    try:
        voice_config = voice_util.resolve_voice_config(
            voice=args.voice,
            base_dir=Path.cwd(),
            voice_text=args.voice_text,
            voice_text_path=voice_text_path,
            language=args.language,
            x_vector_only_mode=bool(args.x_vector_only),
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    return tts_util.synthesize_book(
        book_dir=book_dir,
        voice=voice_config,
        out_dir=out_dir,
        max_chars=args.max_chars,
        pad_ms=args.pad_ms,
        rechunk=args.rechunk,
        voice_map_path=voice_map_path,
        base_dir=Path.cwd(),
        model_name=args.model,
        device_map=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
        backend=args.backend,
        kana_normalize=args.kana_normalize,
        kana_style=args.kana_style,
    )


def _sample(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    out_dir = Path(args.out) if args.out else None
    voice_text_path = Path(args.voice_text_file) if args.voice_text_file else None
    voice_map_path = Path(args.voice_map) if args.voice_map else None

    try:
        voice_config = voice_util.resolve_voice_config(
            voice=args.voice,
            base_dir=Path.cwd(),
            voice_text=args.voice_text,
            voice_text_path=voice_text_path,
            language=args.language,
            x_vector_only_mode=bool(args.x_vector_only),
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    return tts_util.synthesize_book_sample(
        book_dir=book_dir,
        voice=voice_config,
        out_dir=out_dir,
        max_chars=args.max_chars,
        pad_ms=args.pad_ms,
        rechunk=args.rechunk,
        voice_map_path=voice_map_path,
        base_dir=Path.cwd(),
        model_name=args.model,
        device_map=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
        backend=args.backend,
        kana_normalize=args.kana_normalize,
        kana_style=args.kana_style,
    )


def _infer_book_and_chapter(path: Path) -> tuple[Optional[Path], Optional[str]]:
    resolved = path.resolve()
    book_dir = None
    for parent in [resolved] + list(resolved.parents):
        if (parent / "tts" / "manifest.json").exists():
            book_dir = parent
            break
    chapter_id = None
    if book_dir:
        chunks_root = (book_dir / "tts" / "chunks").resolve()
        try:
            rel = resolved.relative_to(chunks_root)
            if rel.parts:
                chapter_id = rel.parts[0]
        except Exception:
            chapter_id = None
    return book_dir, chapter_id


def _debug_dump(label: str, value: str) -> None:
    sys.stderr.write(f"{label}: {value}\n")


def _kana(args: argparse.Namespace) -> int:
    raw_input = args.path
    input_path = Path(raw_input)
    using_file = input_path.exists()
    if using_file:
        text = input_path.read_text(encoding="utf-8")
    else:
        text = raw_input
    book_dir = Path(args.book).resolve() if args.book else None
    chapter_id = args.chapter
    chunk_index = None
    if using_file and (not book_dir or not chapter_id):
        inferred_book, inferred_chapter = _infer_book_and_chapter(input_path)
        if book_dir is None:
            book_dir = inferred_book
        if chapter_id is None:
            chapter_id = inferred_chapter
    if using_file:
        stem = input_path.stem
        if stem.isdigit():
            chunk_index = int(stem) - 1

    global_overrides = []
    merged_overrides = []
    chapter_count = 0
    ruby_data = {}
    chapter_ruby_spans: list[dict] = []
    chunk_span = None
    if book_dir and book_dir.exists():
        global_overrides, chapter_overrides = tts_util._load_reading_overrides(book_dir)
        ruby_data = tts_util._load_ruby_data(book_dir)
        if ruby_data and chapter_id and chunk_index is not None:
            manifest_path = book_dir / "tts" / "manifest.json"
            manifest: dict = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    manifest = {}
            chapters_meta = manifest.get("chapters") if isinstance(manifest, dict) else []
            if isinstance(chapters_meta, list):
                for entry in chapters_meta:
                    if isinstance(entry, dict) and (entry.get("id") or "") == chapter_id:
                        rel_path = entry.get("path")
                        if rel_path:
                            chapter_path = (book_dir / rel_path).resolve()
                            if chapter_path.exists():
                                chapter_text = tts_util._normalize_text(
                                    read_clean_text(chapter_path)
                                )
                                chapter_ruby_spans = tts_util._select_ruby_spans(
                                    chapter_id, chapter_text, ruby_data
                                )
                        span_list = entry.get("chunk_spans")
                        if isinstance(span_list, list) and len(span_list) > chunk_index:
                            chunk_span = span_list[chunk_index]
                        break
        if chapter_id:
            chapter_entries = chapter_overrides.get(chapter_id, [])
            chapter_count = len(chapter_entries)
            merged_overrides = tts_util._merge_reading_overrides(
                global_overrides, chapter_entries
            )
        else:
            merged_overrides = global_overrides
    else:
        template_path = tts_util._template_reading_overrides_path()
        if template_path.exists():
            template_text = template_path.read_text(encoding="utf-8")
            global_overrides = tts_util._parse_reading_overrides_text(template_text)
        merged_overrides = global_overrides

    if args.debug:
        _debug_dump("Input", str(input_path) if using_file else "(inline)")
        if not using_file:
            _debug_dump("InputText", text)
        _debug_dump("Book", str(book_dir) if book_dir else "(none)")
        _debug_dump("Chapter", chapter_id or "(none)")
        _debug_dump("Overrides", f"global={len(global_overrides)} chapter={chapter_count}")
        if args.kana_normalize:
            dict_dir = tts_util._resolve_unidic_dir()
            dicrc = dict_dir / "dicrc"
            _debug_dump("UniDic dir", str(dict_dir))
            _debug_dump("UniDic dicrc", "found" if dicrc.exists() else "missing")

    override_bases = {
        str(item.get("base") or "").strip()
        for item in merged_overrides
        if isinstance(item, dict) and str(item.get("base") or "").strip()
    }
    if ruby_data:
        if chunk_span is not None and chapter_ruby_spans:
            text = tts_util._apply_ruby_evidence_to_chunk(
                text,
                chunk_span,
                chapter_ruby_spans,
                ruby_data,
                skip_bases=override_bases,
            )
        else:
            text = tts_util._apply_ruby_evidence(
                text,
                chapter_id,
                ruby_data,
                skip_bases=override_bases,
            )
    tts_source = tts_util.apply_reading_overrides(text, merged_overrides)
    if args.debug:
        _debug_dump("After overrides", tts_source)

    tts_source = tts_util._normalize_numbers(tts_source)
    if args.debug:
        _debug_dump("After numbers", tts_source)

    if args.tokens:
        try:
            tagger = tts_util._get_kana_tagger()
        except RuntimeError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        sys.stderr.write("Tokens (kanji only):\n")
        for token in tagger(tts_source):
            surface = getattr(token, "surface", "")
            if not surface or not tts_util._has_kanji(surface):
                continue
            feature = getattr(token, "feature", None)
            kana = getattr(feature, "kana", None) if feature else None
            pron = getattr(feature, "pron", None) if feature else None
            reading = getattr(feature, "reading", None) if feature else None
            sanitized = tts_util._extract_token_reading(token)
            sys.stderr.write(
                f"  {surface}\treading={sanitized or ''}\t"
                f"kana={kana or ''}\tpron={pron or ''}\treading_raw={reading or ''}\n"
            )

    if args.kana_normalize:
        try:
            tts_source = tts_util._normalize_kana_text(
                tts_source, kana_style=args.kana_style, force_first_kanji=True
            )
        except RuntimeError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        if args.debug:
            _debug_dump("After kana", tts_source)

    tts_text = tts_util.prepare_tts_text(tts_source, add_short_punct=True)
    if args.debug:
        _debug_dump("Prepared", tts_text)

    sys.stdout.write(tts_text + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nik")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser(
        "ingest", help="Extract chapters from an EPUB or TXT (Japanese only)"
    )
    ingest.add_argument("--input", required=True, help="Path to input .epub or .txt")
    ingest.add_argument(
        "--out",
        "--output",
        required=True,
        dest="out",
        help="Output book directory (e.g., out/book)",
    )
    ingest.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing raw chapters"
    )
    ingest.set_defaults(func=_ingest)

    sanitize = subparsers.add_parser("sanitize", help="Clean chapter text")
    sanitize.add_argument("--book", required=True, help="Book output directory")
    sanitize.add_argument(
        "--rules",
        help=(
            "Path to JSON rules file (defaults to sanitize-rules.json in the book "
            "directory if present, otherwise nik/templates/sanitize-rules.json)"
        ),
    )
    sanitize.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing cleaned output"
    )
    sanitize.set_defaults(func=_sanitize)

    clone = subparsers.add_parser(
        "clone", help="Create a voice sample from audio + transcript"
    )
    clone.add_argument("source", help="URL or local path to an audio file")
    clone.add_argument(
        "--name",
        help="Output voice name (default: input filename without extension)",
    )
    clone.add_argument(
        "--start",
        default="00:00:00",
        help="Start timestamp (HH:MM:SS or seconds)",
    )
    clone.add_argument(
        "--duration",
        type=float,
        default=10,
        help="Duration in seconds",
    )
    clone.add_argument("--text", help="Reference transcript for the audio clip")
    clone.add_argument("--text-file", dest="text_file", help="Path to transcript file")
    clone.add_argument(
        "--auto-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-transcribe the clip with Whisper when no text is provided",
    )
    clone.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model name (default: small)",
    )
    clone.add_argument(
        "--whisper-language",
        help="Whisper language code/name (default: derived from --language)",
    )
    clone.add_argument(
        "--whisper-device",
        help="Whisper device (e.g., cpu, cuda, mps)",
    )
    clone.add_argument(
        "--whisper-prompt",
        help="Initial prompt to guide Whisper transcription",
    )
    clone.add_argument(
        "--language",
        default=voice_util.DEFAULT_LANGUAGE,
        help="Language label for qwen-tts (default: Japanese)",
    )
    clone.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Skip text requirement and use x-vector only cloning",
    )
    clone.set_defaults(func=_clone)

    synth = subparsers.add_parser("synth", help="Synthesize audio")
    synth.add_argument("--book", required=True, help="Book output directory")
    synth.add_argument(
        "--out",
        help="Output directory (default: <book>/tts)",
    )
    synth.add_argument(
        "--voice",
        required=True,
        help="Voice config name/path, or a raw audio path with --voice-text",
    )
    synth.add_argument(
        "--voice-map",
        help="Path to voice map JSON for per-chapter voices",
    )
    synth.add_argument(
        "--voice-text",
        help="Reference transcript when providing a raw audio file",
    )
    synth.add_argument(
        "--voice-text-file",
        help="Reference transcript file when providing a raw audio file",
    )
    synth.add_argument(
        "--language",
        default=voice_util.DEFAULT_LANGUAGE,
        help="Language label for qwen-tts (default: Japanese)",
    )
    synth.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Skip text requirement and use x-vector only cloning",
    )
    synth.add_argument("--max-chars", type=int, default=220)
    synth.add_argument("--pad-ms", type=int, default=150)
    synth.add_argument("--rechunk", action="store_true")
    synth.add_argument(
        "--kana-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize kanji to kana with UniDic (default: enabled)",
    )
    synth.add_argument(
        "--kana-style",
        choices=("partial", "hiragana", "katakana", "mixed", "off"),
        default="partial",
        help="Kana style when normalizing (default: partial; use off to keep kanji)",
    )
    synth.add_argument(
        "--model",
        default=None,
        help="Model id/path (defaults depend on backend)",
    )
    synth.add_argument(
        "--backend",
        choices=("auto", "torch", "mlx"),
        default="auto",
        help="TTS backend (auto chooses MLX on Apple Silicon when available)",
    )
    synth.add_argument(
        "--device",
        default=None,
        help="Device map (e.g., cuda:0, cpu, or auto)",
    )
    synth.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype (e.g., bfloat16, float16, float32)",
    )
    synth.add_argument(
        "--attn",
        default=None,
        help="Attention implementation (e.g., flash_attention_2)",
    )
    synth.set_defaults(func=_synth)

    sample = subparsers.add_parser(
        "sample", help="Generate a sample (first chapter)"
    )
    sample.add_argument("--book", required=True, help="Book output directory")
    sample.add_argument(
        "--out",
        help="Output directory (default: <book>/tts)",
    )
    sample.add_argument(
        "--voice",
        required=True,
        help="Voice config name/path, or a raw audio path with --voice-text",
    )
    sample.add_argument(
        "--voice-map",
        help="Path to voice map JSON for per-chapter voices",
    )
    sample.add_argument(
        "--voice-text",
        help="Reference transcript when providing a raw audio file",
    )
    sample.add_argument(
        "--voice-text-file",
        help="Reference transcript file when providing a raw audio file",
    )
    sample.add_argument(
        "--language",
        default=voice_util.DEFAULT_LANGUAGE,
        help="Language label for qwen-tts (default: Japanese)",
    )
    sample.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Skip text requirement and use x-vector only cloning",
    )
    sample.add_argument("--max-chars", type=int, default=220)
    sample.add_argument("--pad-ms", type=int, default=150)
    sample.add_argument("--rechunk", action="store_true")
    sample.add_argument(
        "--kana-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize kanji to kana with UniDic (default: enabled)",
    )
    sample.add_argument(
        "--kana-style",
        choices=("partial", "hiragana", "katakana", "mixed", "off"),
        default="partial",
        help="Kana style when normalizing (default: partial; use off to keep kanji)",
    )
    sample.add_argument(
        "--model",
        default=None,
        help="Model id/path (defaults depend on backend)",
    )
    sample.add_argument(
        "--backend",
        choices=("auto", "torch", "mlx"),
        default="auto",
        help="TTS backend (auto chooses MLX on Apple Silicon when available)",
    )
    sample.add_argument(
        "--device",
        default=None,
        help="Device map (e.g., cuda:0, cpu, or auto)",
    )
    sample.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype (e.g., bfloat16, float16, float32)",
    )
    sample.add_argument(
        "--attn",
        default=None,
        help="Attention implementation (e.g., flash_attention_2)",
    )
    sample.set_defaults(func=_sample)

    kana = subparsers.add_parser(
        "kana",
        help="Normalize kanji to kana for a chunk file and print TTS text",
    )
    kana.add_argument("path", help="Path to a chunk text file or inline text")
    kana.add_argument("--book", help="Book directory (optional)")
    kana.add_argument("--chapter", help="Chapter id (optional)")
    kana.add_argument(
        "--kana-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply UniDic kana normalization (default: enabled)",
    )
    kana.add_argument(
        "--kana-style",
        choices=("partial", "hiragana", "katakana", "mixed", "off"),
        default="partial",
        help="Kana style when normalizing (default: partial; use off to keep kanji)",
    )
    kana.add_argument(
        "--debug",
        action="store_true",
        help="Print intermediate steps to stderr",
    )
    kana.add_argument(
        "--tokens",
        action="store_true",
        help="Print kanji token readings to stderr",
    )
    kana.set_defaults(func=_kana)

    merge = subparsers.add_parser("merge", help="Merge audio into M4B")
    merge.add_argument("--book", required=True, help="Book output directory")
    merge.add_argument("--output", required=True, help="Path to output .m4b")
    merge.add_argument("--bitrate", default="64k", help="Audio bitrate (default: 64k)")
    merge.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if it exists"
    )
    merge.add_argument(
        "--progress-file",
        help="Write merge progress to JSON file",
    )
    merge.add_argument(
        "--split-hours",
        type=float,
        help="Target hours per part; splits at chapter boundaries",
    )
    merge.add_argument(
        "--split-count",
        type=int,
        help="Number of parts to split into; splits at chapter boundaries",
    )
    merge.set_defaults(func=_merge)

    play = subparsers.add_parser("play", help="Play generated audio in a web UI")
    play.add_argument(
        "--root",
        default="out",
        help="Root folder containing book outputs (default: out)",
    )
    play.add_argument("--host", default="0.0.0.0")
    play.add_argument("--port", type=int, default=1912)
    play.set_defaults(func=_play)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return int(args.func(args))
