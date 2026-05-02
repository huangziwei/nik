"""Adapter for Irodori-TTS as the synthesis backend.

Irodori-TTS upstream is not pip-installable (its setuptools flat-layout discovers
both `configs/` and `irodori_tts/` as top-level packages and fails to build).
We vendor a clone at `.cache/Irodori-TTS` (gitignored, pinned by the user via
`git checkout <sha>`) and import via `sys.path`. Its transitive deps are listed
directly in `pyproject.toml`.

See .claude/plans/refactor-qwen-to-irodori.md for the full migration plan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .voice import VoiceConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_PATH = _REPO_ROOT / ".cache" / "Irodori-TTS"

if str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))

# These imports require the sys.path tweak above.
from huggingface_hub import hf_hub_download  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
)

DEFAULT_HF_REPO = "Aratako/Irodori-TTS-500M-v2"


def _checkpoint_path(repo_id: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename="model.safetensors")


def get_runtime(
    *,
    hf_repo: str = DEFAULT_HF_REPO,
    model_device: str = "cpu",
    model_precision: str = "fp32",
    codec_device: Optional[str] = None,
    codec_precision: str = "fp32",
) -> InferenceRuntime:
    """Load (or fetch from cache) an InferenceRuntime for the given checkpoint."""
    key = RuntimeKey(
        checkpoint=_checkpoint_path(hf_repo),
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device or model_device,
        codec_precision=codec_precision,
    )
    runtime, _was_cached = get_cached_runtime(key)
    return runtime


def generate_chunk(
    runtime: InferenceRuntime,
    text: str,
    voice: VoiceConfig,
    *,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Synthesize one chunk; returns (audio_float32_1d, sample_rate)."""
    request = SamplingRequest(
        text=text,
        ref_wav=voice.ref_audio,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        seed=seed,
    )
    result = runtime.synthesize(request, log_fn=None)
    audio = result.audio.detach().cpu().numpy().astype(np.float32, copy=False)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    return audio, int(result.sample_rate)
