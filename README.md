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

### UniDic setup (kana normalization)
- The first TTS run will download the UniDic dictionary to `~/.cache/nik/unidic-cwj-202512_full`.
- Override the dictionary path with `NIK_UNIDIC_DIR=/path/to/unidic`.
- Override the download URL with `NIK_UNIDIC_URL=...` if you maintain a custom mirror.

## Usage

### Play in web UI
```bash
uv run nik play --port 2999
```
