# nik

`nik` is essentially a Japanese counterpart to [neb](https://github.com/huangziwei/neb): it converts Japanese EPUBs to M4B using [Irodori-TTS](https://github.com/Aratako/Irodori-TTS), superseding [nk](https://github.com/huangziwei/nk).

Only works and tested on Apple Silicon.

```bash
git clone https://github.com/Aratako/Irodori-TTS.git .cache/Irodori-TTS
(cd .cache/Irodori-TTS && git checkout 2708d3cadf726d4389d25eb4bb7a0344517a9a40)

uv sync
uv run nik play --port 2999
```

Open http://localhost:2999.
