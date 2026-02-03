# nik

Japanese EPUB -> M4B pipeline using Qwen TTS voice cloning.

## Prerequisites

- Python 3.12
- `ffmpeg` on PATH (required for merge, voice clip prep, and Whisper)
- A compatible PyTorch install for your hardware

## Installation

```bash
uv sync
```

Optional (auto-transcribe voice clips with Whisper):
```bash
uv sync --extra whisper
```

## Usage

### 1) Ingest EPUB
```bash
uv run nik ingest \
  --input books/Some-Book.epub \
  --out out/some-book
```

### 2) Sanitize (optional but recommended)
```bash
uv run nik sanitize \
  --book out/some-book \
  --overwrite
```

### 3) Prepare a voice clone
```bash
uv run nik clone \
  voices/source.wav \
  --name myvoice \
  --text "REFERENCE TRANSCRIPT HERE" \
  --duration 10
```

If you omit `--text`, nik will use Whisper to auto-transcribe the clip:
```bash
uv run nik clone \
  voices/source.wav \
  --name myvoice \
  --duration 10
```

Disable auto-transcription with `--no-auto-text`. You can tweak Whisper with
`--whisper-model`, `--whisper-language`, or `--whisper-device`.

This creates `voices/myvoice.json` pointing at `voices/myvoice.wav`.
You can also pass a raw audio path directly with `--voice` and supply
`--voice-text` / `--voice-text-file` at synth time.

### 4) Synthesize audio
```bash
uv run nik synth \
  --book out/some-book \
  --voice myvoice \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

### 5) Merge to M4B
```bash
uv run nik merge \
  --book out/some-book \
  --output out/some-book/some-book.m4b
```

### Play in web UI
```bash
uv run nik play --root out --port 1912
```
