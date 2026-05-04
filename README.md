# nik

`nik` is essentially a Japanese counterpart to [ptts](https://github.com/huangziwei/ptts):
it converts Japanese EPUBs to M4B using [Irodori-TTS](https://github.com/Aratako/Irodori-TTS),
superseding [nk](https://github.com/huangziwei/nk).

Irodori-TTS is a Japanese-only model, so it never falls into Mandarin pronunciation —
the kanji-to-kana priming hacks the previous Qwen3-TTS backend needed are gone.

## Setup

Apple Silicon (M-series) only. The default torch wheel for macOS arm64 already
includes MPS; half-precision on MPS is a real speedup.

Irodori-TTS is vendored as a gitignored clone (its upstream `pyproject.toml` is not
pip-installable due to a setuptools flat-layout discovery bug); its model code is
loaded via `sys.path` at runtime, while its transitive deps live in nik's own
`pyproject.toml`.

```bash
git clone https://github.com/Aratako/Irodori-TTS.git .cache/Irodori-TTS
(cd .cache/Irodori-TTS && git checkout 2708d3cadf726d4389d25eb4bb7a0344517a9a40)

uv sync --prerelease=allow
uv run nik play --port 2999
```

Open http://localhost:2999.

## Speed knobs

Levers (env vars, no code edits):

| Env var | Default | Notes |
|---|---|---|
| `NIK_NUM_STEPS` | `10` | Diffusion steps. 10 was empirically indistinguishable from 40; below 5 audibly degrades. Higher (`20`, `40`) costs wallclock with no audible gain. |
| `NIK_MODEL_PRECISION` | `bf16` | `bf16` / `fp16` halve weight memory and run faster on MPS. `fp32` is the slow fallback. |
| `NIK_MODEL_DEVICE` | auto | `cpu` / `mps`. Auto-detects MPS on Apple Silicon. |
