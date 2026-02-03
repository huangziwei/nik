# nik

Japanese EPUB -> M4B pipeline using Qwen TTS voice cloning, superseding [nk](https://github.com/huangziwei/nk).

## Prerequisites

- Apple Silicon Mac (arm64)
- Python 3.12
- `ffmpeg` on PATH (required for merge, voice clip prep, and Whisper)

## Installation

Apple Silicon (MLX backend):
```bash
uv sync --prerelease=allow
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

### Ruby readings (Japanese)
If the EPUB contains `<ruby>` tags, ingest will extract them into
`reading-overrides.json`. These overrides are only applied at TTS time
so the clean text stays unchanged.

You can also add global overrides that apply to every chapter (useful for
names or kanji-only words). Chapter-specific overrides win when the same
base text appears in both. Global defaults live in
`nik/templates/global-reading-overrides.md` and are merged with per-book
overrides.

```json
{
  "global": [
    { "base": "妻子", "reading": "さいし" },
    { "base": "山田太一", "reading": "やまだたいいち" }
  ],
  "chapters": {
    "0003-chapter": {
      "replacements": [
        { "base": "一日", "reading": "いちにち" }
      ]
    }
  }
}
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

Merge auto-splits if the book is longer than 8 hours, keeping parts near-equal
and splitting only at chapter boundaries.

To override the split threshold:
```bash
uv run nik merge \
  --book out/some-book \
  --output out/some-book/some-book.m4b \
  --split-hours 8
```

### Play in web UI
```bash
uv run nik play --root out --port 1912
```
