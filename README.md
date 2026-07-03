# nik

`nik` is essentially a Japanese counterpart to [neb](https://github.com/huangziwei/neb): it converts Japanese EPUBs to M4B using [Irodori-TTS](https://github.com/Aratako/Irodori-TTS), superseding [nk](https://github.com/huangziwei/nk).

Only works and tested on Apple Silicon.

```bash
uv sync
uv run nik play --port 2999
```

Open http://localhost:2999.

## Backends

Synthesis runs through the [mlx-audio](https://github.com/Blaizzy/mlx-audio) port of Irodori-TTS by default (~2× faster than the upstream PyTorch path on Apple Silicon, audible parity confirmed). Weights are pulled on first run from `mlx-community/Irodori-TTS-500M-v2-4bit`.

To fall back to the upstream PyTorch path:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git .cache/Irodori-TTS
(cd .cache/Irodori-TTS && git checkout 2708d3cadf726d4389d25eb4bb7a0344517a9a40)
NIK_BACKEND=torch uv run nik play --port 2999
```
