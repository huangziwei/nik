"""Adapter for Irodori-TTS via the upstream PyTorch code (fallback backend).

The default backend is MLX (`synth_irodori_mlx`); `NIK_BACKEND=torch` opts in
here, with the same surface: `resolve_hf_repo`, `env_num_steps`, `get_runtime`,
`generate_chunk`.

Irodori-TTS upstream is not pip-installable (its setuptools flat-layout discovers
both `configs/` and `irodori_tts/` as top-level packages and fails to build).
This backend imports a clone at `.cache/Irodori-TTS` via `sys.path`
(`git clone https://github.com/Aratako/Irodori-TTS .cache/Irodori-TTS`; v4-Large
needs a checkout from 2026-09-12 or later). The clone is only required when this
backend is selected — see `_ensure_vendor_on_path`.

Default checkpoint: `Aratako/Irodori-TTS-v4-Large` (override with
`NIK_TORCH_HF_REPO`). Apple Silicon only: device auto-picks `mps` when available
(CPU otherwise), and precision is fixed at `fp32` — Irodori rejects `bf16`/`fp16`
outside CUDA and we don't ship any other platforms.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from . import CACHE_DIR
from .voice import VoiceConfig

ENV_NUM_STEPS = "NIK_NUM_STEPS"
ENV_HF_REPO = "NIK_TORCH_HF_REPO"
ENV_STYLE_EMOJI = "NIK_STYLE_EMOJI"

_VENDOR_PATH = CACHE_DIR / "Irodori-TTS"

DEFAULT_HF_REPO = "Aratako/Irodori-TTS-v4-Large"


def _ensure_vendor_on_path() -> None:
    """Lazy: only attach the vendored Irodori-TTS clone when this backend runs.

    MLX is the default backend, so most installs don't need the vendor clone.
    Importing this module must not fail in that case.
    """
    if str(_VENDOR_PATH) not in sys.path:
        sys.path.insert(0, str(_VENDOR_PATH))
    if not _VENDOR_PATH.exists():
        raise RuntimeError(
            f"NIK_BACKEND=torch requires the upstream Irodori-TTS clone at "
            f"{_VENDOR_PATH} (git clone https://github.com/Aratako/Irodori-TTS "
            f"{_VENDOR_PATH}), or use the default MLX backend."
        )


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


def _checkpoint_path(repo_id: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename="model.safetensors")


def get_runtime(*, hf_repo: str | None = None):
    """Load (or fetch from cache) an InferenceRuntime for the given checkpoint."""
    _ensure_vendor_on_path()
    from irodori_tts.inference_runtime import (
        RuntimeKey,
        default_runtime_device,
        get_cached_runtime,
    )

    device = default_runtime_device()
    key = RuntimeKey(
        checkpoint=_checkpoint_path(resolve_hf_repo(hf_repo)),
        model_device=device,
        model_precision="fp32",
        codec_device=device,
        codec_precision="fp32",
    )
    runtime, _was_cached = get_cached_runtime(key)
    return runtime


def _style_prefix() -> str:
    """Optional emotion annotation prepended to every chunk's text.

    Mirrors the MLX backend: an emoji from NIK_STYLE_EMOJI (e.g. 😌 calm) steers
    Irodori's delivery. Applied at the model boundary, never cached.
    """
    raw = os.environ.get(ENV_STYLE_EMOJI)
    return raw.strip() if raw else ""


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

    Arguments left as None keep upstream's defaults: `num_steps` falls back to
    NIK_NUM_STEPS, then to the checkpoint's recommendation. `style_emoji`
    overrides the NIK_STYLE_EMOJI env fallback when not None (pass "" to force
    neutral); mirrors the MLX backend.
    """
    _ensure_vendor_on_path()
    from irodori_tts.inference_runtime import SamplingRequest

    if num_steps is None:
        num_steps = env_num_steps()
    prefix = _style_prefix() if style_emoji is None else style_emoji.strip()
    gen_text = f"{prefix}{text}" if prefix else text
    overrides: dict[str, float] = {}
    if cfg_scale_text is not None:
        overrides["cfg_scale_text"] = float(cfg_scale_text)
    if cfg_scale_speaker is not None:
        overrides["cfg_scale_speaker"] = float(cfg_scale_speaker)
    request = SamplingRequest(
        text=gen_text,
        ref_wav=voice.ref_audio,
        num_steps=num_steps,
        seed=seed,
        **overrides,
    )
    result = runtime.synthesize(request, log_fn=None)
    audio = result.audio.detach().cpu().numpy().astype(np.float32, copy=False)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    return audio, int(result.sample_rate)
