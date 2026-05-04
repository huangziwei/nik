"""Quick manual synth test: synth a few 放課後 chunks with 津田健次郎.

Run from repo root:
    uv run python tests/manual_synth_test.py
"""

from __future__ import annotations

import time
from pathlib import Path

import soundfile as sf

from nik.synth_irodori import generate_chunk, get_runtime
from nik.voice import load_voice_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = REPO_ROOT / "out" / "放課後" / "tts" / "chunks" / "0002-chapter"
VOICE_JSON = REPO_ROOT / "voices" / "津田健次郎.json"
OUT_DIR = REPO_ROOT / "out" / "_manual_synth_test"

CHUNK_IDS = ["000001", "000003", "000005", "000010", "000020"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    voice = load_voice_config(VOICE_JSON)
    print(f"voice: {voice.name}  ref_audio={voice.ref_audio}")

    print("loading runtime...")
    t0 = time.time()
    runtime = get_runtime()
    print(f"runtime ready in {time.time() - t0:.1f}s")

    for cid in CHUNK_IDS:
        text_path = CHUNKS_DIR / f"{cid}.txt"
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  {cid}: empty, skipping")
            continue
        print(f"  {cid}: {len(text)} chars  '{text[:30]}{'...' if len(text) > 30 else ''}'")
        t0 = time.time()
        audio, sr = generate_chunk(runtime, text, voice, seed=0)
        dur = len(audio) / sr
        out_path = OUT_DIR / f"{cid}.wav"
        sf.write(out_path, audio, sr, subtype="PCM_16")
        print(f"    -> {out_path.name}  {dur:.2f}s @ {sr}Hz  (synth {time.time() - t0:.1f}s)")

    print(f"done. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
