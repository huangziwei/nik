"""Adapter for Irodori-TTS via the MLX port (mlx-audio's ``irodori_tts``).

Surface shared with `synth_irodori` (the torch fallback):
  - `resolve_hf_repo(explicit=None)` — the checkpoint a run loads.
  - `default_num_steps()` — NIK_NUM_STEPS when set, else `DEFAULT_NUM_STEPS`.
  - `get_runtime(*, hf_repo=None)` — the loaded model (cached per repo); its
    `sample_rate` feeds `_resolve_output_sample_rate`.
  - `generate_chunk(runtime, text, voice, ...)` → `(float32 mono, sample_rate)`.

Default checkpoint: Irodori-TTS 500M v3 at 8 bits
(`mlx-community/Irodori-TTS-500M-v3-8bit`), sampled with 20 Euler steps (its
config suggests 40). Override with `NIK_MLX_HF_REPO` / `NIK_NUM_STEPS`; any
mlx-community Irodori v2/v3/v4 repo loads. Measured on nik's own books, v4-Large
needs about 8x v3's time per chunk, and v4.1-Small-MF, the one v4 model faster
than v3, misreads about twice as much text.

Other sampling settings follow the checkpoint's `config.json` (CFG text 3,
speaker 5, independent guidance); nik overrides only the step count,
NIK_MLX_CFG_MODE and the seed. The model predicts each chunk's duration, so no
length is passed.

Work that does not change between chunks is done once: a voice's reference
clip is encoded once per runtime (keyed by path, size and mtime), and the
codec's weight-normalised convolutions are folded into plain weights at load.
`generate_chunk` otherwise follows mlx-audio's `Model.generate` step for step,
so the audio is identical to it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .voice import VoiceConfig

ENV_NUM_STEPS = "NIK_NUM_STEPS"
ENV_HF_REPO = "NIK_MLX_HF_REPO"
ENV_CFG_MODE = "NIK_MLX_CFG_MODE"
ENV_STYLE_EMOJI = "NIK_STYLE_EMOJI"

DEFAULT_HF_REPO = "mlx-community/Irodori-TTS-500M-v3-8bit"
DEFAULT_NUM_STEPS = 20
_CFG_MODES = {"independent", "joint", "alternating"}
# Same values as mlx-audio's Model.generate: 50-frame decode chunks, and the
# trailing-silence window it trims with.
_DECODE_CHUNK_FRAMES = 50
_SILENCE_WINDOW = 20
_SILENCE_STD = 0.05
_SILENCE_MEAN = 0.1
_REF_CACHE_MAX = 16

_runtime_cache: dict[str, object] = {}
_ref_cache: dict[tuple, tuple] = {}


def resolve_hf_repo(explicit: str | None = None) -> str:
    return explicit or os.environ.get(ENV_HF_REPO) or DEFAULT_HF_REPO


def default_num_steps() -> int:
    raw = os.environ.get(ENV_NUM_STEPS)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_NUM_STEPS


def _env_cfg_mode() -> str | None:
    raw = (os.environ.get(ENV_CFG_MODE) or "").strip()
    return raw if raw in _CFG_MODES else None


def _style_prefix() -> str:
    """Optional emotion annotation prepended to every chunk's text.

    Irodori is emoji-driven: a control emoji (e.g. 😌 calm) in the input steers
    delivery. Set globally per synth run via NIK_STYLE_EMOJI (the player wires it
    from the emotion pills). Applied here at the model boundary so it never
    touches the cached/hashed chunk text or the manifest.
    """
    raw = os.environ.get(ENV_STYLE_EMOJI)
    return raw.strip() if raw else ""


def _fold_weight_norm(codec) -> None:
    """Store each weight-normalised conv's effective weight once.

    mlx-audio's DACVAE convs recompute `g * v / ||v||` on every call; the
    weight evaluated once is the same array, so outputs are bit-identical.
    """
    import mlx.core as mx
    from mlx_audio.codec.models.dacvae.codec import WNConv1d, WNConvTranspose1d

    for _, module in codec.named_modules():
        if not isinstance(module, (WNConv1d, WNConvTranspose1d)):
            continue
        if not module.use_weight_norm:
            continue
        weight = module._get_weight()
        mx.eval(weight)
        module.weight = weight
        module.use_weight_norm = False
        del module["weight_g"]
        del module["weight_v"]


def get_runtime(*, hf_repo: str | None = None):
    """Load (or fetch from cache) an mlx-audio Model for the given repo."""
    repo = resolve_hf_repo(hf_repo)
    cached = _runtime_cache.get(repo)
    if cached is not None:
        return cached

    # Imported lazily so the PyTorch backend doesn't pay the MLX init cost
    # (and so a missing Metal device doesn't break unrelated imports).
    from mlx_audio.tts import load as mlx_tts_load

    model = mlx_tts_load(repo)
    if getattr(model, "dacvae", None) is not None:
        _fold_weight_norm(model.dacvae)
    _runtime_cache[repo] = model
    return model


def _reference_latent(model, ref_audio: str) -> tuple:
    """The voice clip's codec latent and mask, encoded once per model and file."""
    import mlx.core as mx

    path = Path(ref_audio)
    try:
        stat = path.stat()
        key: tuple | None = (
            id(model),
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
        )
    except OSError:
        key = None
    if key is not None and key in _ref_cache:
        return _ref_cache[key]

    waveform = model._load_ref_waveform(str(ref_audio))
    latent, mask = model._encode_ref_audios([waveform])
    mx.eval(latent, mask)
    if key is not None:
        if len(_ref_cache) >= _REF_CACHE_MAX:
            _ref_cache.pop(next(iter(_ref_cache)))
        _ref_cache[key] = (latent, mask)
    return latent, mask


def _silence_frame(latent) -> int:
    """First frame whose window is near-silent; `latent.shape[0]` when none.

    Same rule as mlx-audio's `_find_silence_point` (std < 0.05 and |mean| < 0.1
    over a 20-frame window, zero-padded at the end), with every window
    reduced in one pass instead of two GPU syncs per frame.
    """
    import mlx.core as mx

    frames, dim = latent.shape
    padded = mx.concatenate(
        [latent, mx.zeros((_SILENCE_WINDOW, dim), dtype=latent.dtype)], axis=0
    )
    windows = mx.stack([padded[i : i + _SILENCE_WINDOW] for i in range(frames)])
    std = np.asarray(windows.std(axis=(1, 2)), dtype=np.float64)
    mean = np.asarray(windows.mean(axis=(1, 2)), dtype=np.float64)
    hits = np.flatnonzero((std < _SILENCE_STD) & (np.abs(mean) < _SILENCE_MEAN))
    return int(hits[0]) if hits.size else int(frames)


def _synthesize(
    model,
    text: str,
    ref_latent,
    ref_mask,
    *,
    rng_seed: int,
    sampling_kwargs: dict[str, object],
) -> np.ndarray:
    """mlx-audio's `Model.generate` for one clip, given an encoded reference."""
    import mlx.core as mx

    if getattr(model, "dacvae", None) is None:
        raise ValueError("Irodori-TTS requires its DACVAE codec to be loaded.")
    config = model.config
    latent, latent_steps = model.generate_latents(
        text=text,
        ref_latent=ref_latent,
        ref_mask=ref_mask,
        caption=None,
        rng_seed=rng_seed,
        seconds=None,
        duration_scale=1.0,
        min_seconds=float(config.sampler.min_seconds),
        max_seconds=float(config.sampler.max_seconds),
        **sampling_kwargs,
    )
    decoded = model.dacvae.decode(
        mx.transpose(latent, (0, 2, 1)), chunk_size=_DECODE_CHUNK_FRAMES
    )
    audio = decoded[:, :, 0]
    mx.eval(audio)

    # Clamp to the predicted duration, then trim trailing silence; a silence
    # point of 0 must not collapse the clip to nothing.
    hop = config.audio_downsample_factor
    trim = latent_steps * hop
    silence = _silence_frame(latent[0]) * hop
    if silence > 0:
        trim = min(trim, silence)
    return np.asarray(audio[0, :trim], dtype=np.float32)


def generate_chunk(
    runtime,
    text: str,
    voice: VoiceConfig,
    *,
    num_steps: int | None = None,
    cfg_scale_text: float | None = None,
    cfg_scale_speaker: float | None = None,
    seed: int | None = None,
    style_emoji: str | None = None,
) -> tuple[np.ndarray, int]:
    """Synthesize one chunk; returns (audio_float32_1d, sample_rate).

    `num_steps` defaults to `default_num_steps()`; the CFG scales left as None
    keep the checkpoint's own. `seed` None is mlx-audio's default seed 0.
    `style_emoji` (an explicit emotion annotation) overrides the
    NIK_STYLE_EMOJI env fallback when not None; pass "" to force neutral. The
    env path drives the batch/sample subprocesses; the explicit arg drives
    in-process chunk re-render.
    """
    prefix = _style_prefix() if style_emoji is None else style_emoji.strip()
    gen_text = f"{prefix}{text}" if prefix else text

    steps = default_num_steps() if num_steps is None else int(num_steps)
    sampling_kwargs: dict[str, object] = {"num_steps": steps}
    if cfg_scale_text is not None:
        sampling_kwargs["cfg_scale_text"] = float(cfg_scale_text)
    if cfg_scale_speaker is not None:
        sampling_kwargs["cfg_scale_speaker"] = float(cfg_scale_speaker)
    cfg_mode = _env_cfg_mode()
    if cfg_mode:
        sampling_kwargs["cfg_guidance_mode"] = cfg_mode

    ref_latent, ref_mask = _reference_latent(runtime, voice.ref_audio)
    audio = _synthesize(
        runtime,
        gen_text,
        ref_latent,
        ref_mask,
        rng_seed=0 if seed is None else int(seed),
        sampling_kwargs=sampling_kwargs,
    )
    return audio, int(runtime.sample_rate)
