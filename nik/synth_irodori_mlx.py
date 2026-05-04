"""Adapter for Irodori-TTS via the MLX port (mlx-audio).

Mirrors the surface of `synth_irodori`:
  - `get_runtime(*, hf_repo=...)` returns an opaque model object with a
    `sample_rate` attribute (consumed by `_resolve_output_sample_rate`).
  - `generate_chunk(runtime, text, voice, ...)` returns
    `(np.ndarray float32 1d, sample_rate)`.

Selected via `NIK_BACKEND=mlx`. Quantization is picked via the HF repo:
default `mlx-community/Irodori-TTS-500M-v2-fp16`; override with
`NIK_MLX_HF_REPO=mlx-community/Irodori-TTS-500M-v2-{8bit,4bit}`.

Memory budget on 24 GB unified memory: the upstream MLX port reports
`sequence_length=750` + `cfg_guidance_mode=independent` ≈ 24 GB. We default
to `sequence_length=400` (plenty for nik's chunk lengths — 400 × 1920 / 48000
= 16 s of audio max) which is ~9 GB on `independent`. Tune with
`NIK_MLX_SEQUENCE_LENGTH` and `NIK_MLX_CFG_MODE`.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .voice import VoiceConfig

ENV_NUM_STEPS = "NIK_NUM_STEPS"
ENV_HF_REPO = "NIK_MLX_HF_REPO"
ENV_SEQUENCE_LENGTH = "NIK_MLX_SEQUENCE_LENGTH"
ENV_CFG_MODE = "NIK_MLX_CFG_MODE"

DEFAULT_HF_REPO = "mlx-community/Irodori-TTS-500M-v2-fp16"
DEFAULT_SEQUENCE_LENGTH = 400
DEFAULT_CFG_MODE = "independent"

_runtime_cache: dict[str, "object"] = {}


def _resolve_hf_repo(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return os.environ.get(ENV_HF_REPO) or DEFAULT_HF_REPO


def _default_num_steps() -> int:
    raw = os.environ.get(ENV_NUM_STEPS)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 20


def _default_sequence_length() -> int:
    raw = os.environ.get(ENV_SEQUENCE_LENGTH)
    if raw:
        try:
            return max(50, int(raw))
        except ValueError:
            pass
    return DEFAULT_SEQUENCE_LENGTH


def _default_cfg_mode() -> str:
    raw = os.environ.get(ENV_CFG_MODE)
    if raw and raw in {"independent", "joint", "alternating"}:
        return raw
    return DEFAULT_CFG_MODE


def get_runtime(*, hf_repo: Optional[str] = None):
    """Load (or fetch from cache) an mlx-audio Model for the given repo."""
    repo = _resolve_hf_repo(hf_repo)
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
    num_steps: Optional[int] = None,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Synthesize one chunk; returns (audio_float32_1d, sample_rate).

    Same defaults as `synth_irodori.generate_chunk` so the two backends are
    drop-in interchangeable at the call site.
    """
    if num_steps is None:
        num_steps = _default_num_steps()

    sampling_kwargs = dict(
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_guidance_mode=_default_cfg_mode(),
        sequence_length=_default_sequence_length(),
    )
    if seed is not None:
        sampling_kwargs["rng_seed"] = int(seed)

    # Model.generate yields a single GenerationResult for non-streaming mode.
    result = next(
        runtime.generate(
            text=text,
            ref_audio=voice.ref_audio,
            **sampling_kwargs,
        )
    )

    audio = np.asarray(result.audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    return audio, int(result.sample_rate)
