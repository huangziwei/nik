# nik

`nik` is essentially a Japanese counterpart to [ptts](https://github.com/huangziwei/ptts):
it converts Japanese EPUBs to M4B using [Irodori-TTS](https://github.com/Aratako/Irodori-TTS),
superseding [nk](https://github.com/huangziwei/nk).

Irodori-TTS is a Japanese-only model, so it never falls into Mandarin pronunciation —
the kanji-to-kana priming hacks the previous Qwen3-TTS backend needed are gone.

## Setup

Irodori-TTS is vendored as a gitignored clone (its upstream `pyproject.toml` is not
pip-installable due to a setuptools flat-layout discovery bug); its model code is
loaded via `sys.path` at runtime, while its transitive deps live in nik's own
`pyproject.toml`. Same setup on every host:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git .cache/Irodori-TTS
(cd .cache/Irodori-TTS && git checkout 2708d3cadf726d4389d25eb4bb7a0344517a9a40)
```

### Apple Silicon (M-series)

Native install. The default torch wheel for macOS arm64 already includes MPS, no
CUDA bloat. Half-precision on MPS is a real speedup.

```bash
uv sync --prerelease=allow
NIK_MODEL_PRECISION=bf16 uv run nik play --port 1912
```

### Intel Mac (or anywhere without GPU)

Run inside Podman via the `bin/pmx` wrapper — see [bin/README.md](bin/README.md).
nik's `pyproject.toml` pins torch to the CPU-only index for Linux to drop ~5 GB of
unused CUDA libs.

```bash
./bin/pmx uv sync --prerelease=allow
./bin/pmx uv run nik play --host 0.0.0.0 --port 1912
```

Open http://localhost:1912.

## Speed knobs

CPU inference is roughly 30-65× slower than realtime. Levers (env vars, no code edits):

| Env var | Default | Notes |
|---|---|---|
| `NIK_NUM_STEPS` | `40` | Diffusion steps. Halving to `20` ≈ 2× speedup, modest quality cost. Try this first. |
| `NIK_MODEL_PRECISION` | `fp32` | `bf16` / `fp16` halve weight memory. On Apple Silicon MPS this is also faster; on Intel CPU compute is emulated, so memory only. |
| `NIK_MODEL_DEVICE` | auto | `cpu` / `mps` / `cuda`. Auto-detects by default. |

Example: faster preset on M4 Pro:

```bash
NIK_NUM_STEPS=20 NIK_MODEL_PRECISION=bf16 uv run nik play --port 1912
```
