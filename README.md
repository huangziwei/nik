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

### Play in web UI
```bash
uv run nik play --port 2999
```
