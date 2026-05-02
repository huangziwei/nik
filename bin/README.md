# pmx: Podman wrapper for nik

For Intel Mac (or anywhere `mlx-audio` isn't available), `pmx` runs `nik` and
its TTS backend (Irodori-TTS) inside a Linux container.

The script is intentionally a verbatim copy of [`neb/bin/pmx`](https://github.com/huangziwei/neb/blob/main/bin/pmx)
so that any improvements stay portable between projects. The README is what
differs.

## Setup (Podman)

```bash
mkdir -p .cache/huggingface
chmod +x bin/pmx

brew install podman
podman --version # tested with 5.8.0

# Any podman machine sized for CPU-only TTS work will do. 8 GiB is enough for
# Irodori-TTS-500M inference at bf16; bump to 16 GiB if you OOM during longer
# synth runs. If a machine already exists for another project, just reuse it.
podman machine init --cpus 6 --memory 8192 --disk-size 80 --now tts

# Install project dependencies into .venv (required for `nik` CLI).
# Port 2999 is the default for `nik play`.
PMX_OPTS="-p 2999:2999" ./bin/pmx uv sync --prerelease=allow
```

## Run the player

```bash
./bin/pmx uv run nik play --host 0.0.0.0 --port 2999
```

Open `http://localhost:2999`.

> `--host 0.0.0.0` is required so Podman's port forwarding can reach the
> process inside the container; binding to `127.0.0.1` would only listen on the
> container's loopback.

## nik workflow (prefix with ./bin/pmx)

```bash
./bin/pmx uv run nik ingest   --input books/Some-Book.epub --out out/some-book
./bin/pmx uv run nik sanitize --book out/some-book --overwrite
./bin/pmx uv run nik synth    --book out/some-book
./bin/pmx uv run nik merge    --book out/some-book --output out/some-book/some-book.m4b
```

### ffmpeg inside Podman

`nik merge` requires `ffmpeg` on PATH. Install it in the same run (or bake a
custom image):

```bash
./bin/pmx bash -lc 'apt-get update && apt-get install -y ffmpeg && uv run nik merge --book out/some-book --output out/some-book/some-book.m4b'
```

### Hugging Face cache

`HF_HOME` is set to `/work/.cache/huggingface` so model weights persist in the
project's `.cache/` dir on the host (already gitignored via the bind mount).
First run of Irodori-TTS will download:

- `Aratako/Irodori-TTS-500M-v2`
- `Aratako/Semantic-DACVAE-Japanese-32dim`

To pre-fetch them outside a synth run:

```bash
./bin/pmx uv run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Aratako/Irodori-TTS-500M-v2'); \
  snapshot_download('Aratako/Semantic-DACVAE-Japanese-32dim')"
```

## pmx tips

- `PMX_OPTS` adds container run options (e.g., port mappings).
- `PMX_RESET=1` recreates the named container (useful after changing `PMX_OPTS`).
- `PMX_RM=1` runs a one-off container and removes it on exit.
- `PMX_PRUNE=1` removes all containers matching the current `PMX_NAMESPACE`
  prefix before running. Without `PMX_NAMESPACE` set this is a no-op (the
  prefix is empty), so it will not touch unrelated containers.
- Default container name is the basename of `$PWD` (so `nik` here). Override
  with `PMX_NAME=foo` or scope with `PMX_NAMESPACE=team` to get `team-nik`.
