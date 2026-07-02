# nik

`nik` is essentially a Japanese counterpart to [neb](https://github.com/huangziwei/neb): it converts Japanese EPUBs to M4B using [Irodori-TTS](https://github.com/Aratako/Irodori-TTS), superseding [nk](https://github.com/huangziwei/nk).

Only works and tested on Apple Silicon.

```bash
uv sync
uv run nik play --port 2999
```

Open http://localhost:2999.

## Ruby readings

In-book ruby (furigana) is treated as evidence, not ground truth. The player's edit view
has a ルビ読みレビュー panel that groups every (word, reading) pair found in the book —
with counts, sample contexts, and how many extra places a reading would propagate to —
and lets you correct a reading once (e.g. typeset-big small kana like ていあら→てぃあら)
so it heals every occurrence, pin wordplay ruby to its printed spots only, ignore garbage
ruby, or propagate a reading globally (single kanji propagate in isolated mode so
compounds like 一つ/第一 stay untouched). Clicking a ruby word in the reader opens the
same controls. Decisions live under `ruby.decisions` in each book's
`reading-overrides.json`; by default one-off and wordplay-suspect readings no longer
propagate beyond their printed spots.

## Backends

Synthesis runs through the [mlx-audio](https://github.com/Blaizzy/mlx-audio) port of Irodori-TTS by default (~2× faster than the upstream PyTorch path on Apple Silicon, audible parity confirmed). Weights are pulled on first run from `mlx-community/Irodori-TTS-500M-v2-4bit`.

To fall back to the upstream PyTorch path:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git .cache/Irodori-TTS
(cd .cache/Irodori-TTS && git checkout 2708d3cadf726d4389d25eb4bb7a0344517a9a40)
NIK_BACKEND=torch uv run nik play --port 2999
```
