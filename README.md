# nik

`nik` is essentially a Japanese counterpart to [ptts](https://github.com/huangziwei/ptts): it converts Japanese EPUBs to M4B using Qwen3-TTS instead of Pocket-TTS, superseding [nk](https://github.com/huangziwei/nk).

The most difficult part of using Qwen3-TTS is forcing it to speak monolingually. It is challenging because Japanese and Chinese share a lot of vocabulary, and the model seems to default to Mandarin even when we set `language="japanese"` during TTS synthesis. I tried many things:

1. Convert all kanji to kana. This turns out to be a bad idea because the model loses context and cannot read the words with correct pitch.
2. Convert some kanji if they are [common Chinese words](https://github.com/huangziwei/mcc) or rare Japanese words. This helps, but it still converts too many kanji words and introduces unnatural sounds.
3. The trick that works so far is to convert only the first kanji word at chunk beginnings. It seems to prime the model to speak only Japanese :)

## Usage

```bash
uv sync --prerelease=allow
uv run nik play --port 2999
```
