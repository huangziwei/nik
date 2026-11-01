"""Adapter for Irodori-TTS via the MLX port (mlx-audio's ``irodori_tts``).

Surface shared with `synth_irodori` (the torch fallback):
  - `resolve_hf_repo(explicit=None)` — the checkpoint a run loads.
  - `env_num_steps()` — NIK_NUM_STEPS when set, else None.
  - `get_runtime(*, hf_repo=None)` — the loaded model (cached per repo); its
    `sample_rate` feeds `_resolve_output_sample_rate`.
  - `generate_chunk(runtime, text, voice, ...)` → `(float32 mono, sample_rate)`.

Default checkpoint: **Irodori-TTS v4-Large** at its native bf16
(`mlx-community/Irodori-TTS-v4-Large-bf16`, 3.29B, T5Gemma 2 text encoder).
Use bf16, not fp16: T5Gemma 2 activations overflow fp16 and degrade the output.
Override with `NIK_MLX_HF_REPO` (any mlx-community Irodori v2/v3/v4 repo).

Sampling follows the checkpoint's own `config.json` (40 Euler steps; CFG text 3,
speaker 5, caption 3; independent guidance). nik passes only what it overrides:
NIK_NUM_STEPS, NIK_MLX_CFG_MODE, a per-book pinned step count, or a seed. The
model predicts each chunk's duration itself, so no length is passed either.
"""

from __future__ import annotations

import os

import numpy as np

from .voice import VoiceConfig

ENV_NUM_STEPS = "NIK_NUM_STEPS"
ENV_HF_REPO = "NIK_MLX_HF_REPO"
ENV_CFG_MODE = "NIK_MLX_CFG_MODE"
ENV_STYLE_EMOJI = "NIK_STYLE_EMOJI"

DEFAULT_HF_REPO = "mlx-community/Irodori-TTS-v4-Large-bf16"
_CFG_MODES = {"independent", "joint", "alternating"}

_runtime_cache: dict[str, object] = {}


def resolve_hf_repo(explicit: str | None = None) -> str:
    return explicit or os.environ.get(ENV_HF_REPO) or DEFAULT_HF_REPO


def env_num_steps() -> int | None:
    raw = os.environ.get(ENV_NUM_STEPS)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return None


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
    _runtime_cache[repo] = model
    return model


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

    Sampling arguments left as None keep the checkpoint's defaults
    (`num_steps` falls back to NIK_NUM_STEPS first). `style_emoji` (an explicit
    emotion annotation) overrides the NIK_STYLE_EMOJI env fallback when not
    None; pass "" to force neutral. The env path drives the batch/sample
    subprocesses; the explicit arg drives in-process chunk re-render.
    """
    if num_steps is None:
        num_steps = env_num_steps()
    prefix = _style_prefix() if style_emoji is None else style_emoji.strip()
    gen_text = f"{prefix}{text}" if prefix else text

    sampling_kwargs: dict[str, object] = {}
    if num_steps is not None:
        sampling_kwargs["num_steps"] = int(num_steps)
    if cfg_scale_text is not None:
        sampling_kwargs["cfg_scale_text"] = float(cfg_scale_text)
    if cfg_scale_speaker is not None:
        sampling_kwargs["cfg_scale_speaker"] = float(cfg_scale_speaker)
    cfg_mode = _env_cfg_mode()
    if cfg_mode:
        sampling_kwargs["cfg_guidance_mode"] = cfg_mode
    if seed is not None:
        sampling_kwargs["rng_seed"] = int(seed)

    # Model.generate yields a single GenerationResult for non-streaming mode.
    result = next(
        runtime.generate(
            text=gen_text,
            ref_audio=voice.ref_audio,
            **sampling_kwargs,
        )
    )

    audio = np.asarray(result.audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    return audio, int(result.sample_rate)
