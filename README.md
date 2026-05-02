# nik

`nik` is essentially a Japanese counterpart to [ptts](https://github.com/huangziwei/ptts):
it converts Japanese EPUBs to M4B using [Irodori-TTS](https://github.com/Aratako/Irodori-TTS),
superseding [nk](https://github.com/huangziwei/nk).

Irodori-TTS is a Japanese-only model, so it never falls into Mandarin pronunciation —
the kanji-to-kana priming hacks the previous Qwen3-TTS backend needed are gone.

## Usage

On Apple Silicon you can install Irodori-TTS directly. On Intel Mac (or anywhere else
without GPU) run inside Podman via the `bin/pmx` wrapper — see [bin/README.md](bin/README.md).

```bash
./bin/pmx uv sync --prerelease=allow
./bin/pmx uv run nik play --host 0.0.0.0 --port 1912
```

Open http://localhost:1912.

Irodori-TTS is vendored as a gitignored clone at `.cache/Irodori-TTS` (its upstream
`pyproject.toml` is not pip-installable due to a setuptools flat-layout discovery bug);
its model code is loaded via `sys.path` at runtime, while its transitive deps are
declared in nik's own `pyproject.toml`.
