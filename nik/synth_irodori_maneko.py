"""Adapter for Irodori-TTS via the native maneko wheel (Rust/Candle pyo3).

Mirrors the surface of `synth_irodori` / `synth_irodori_mlx`:
  - `get_runtime(*, hf_repo=None)` returns an opaque runtime exposing a
    `sample_rate` attribute (consumed by `_resolve_output_sample_rate`).
  - `generate_chunk(runtime, text, voice, ...)` returns
    `(np.ndarray float32 1d, sample_rate)`.

Selected via `NIK_BACKEND=maneko`. maneko is self-contained: weights (q8 DiT +
f16 DACVAE + tokenizer) auto-resolve from `HF_HOME`, or pull from the public
`zwaiwng/maneko` on first run — rev-pinned inside the wheel, no token.

Device: `NIK_MANEKO_DEVICE=metal|cpu` (default `metal`; falls back to `cpu` if
the installed wheel was built without Metal). nik is Apple-Silicon-only, so
Metal is the norm.

Steps: default 20 — matches what nik shipped on the MLX backend, so output
tracks today's. maneko's own validated default is 8 (Whisper-exact, ~2.5× fewer
DiT forwards); set `NIK_NUM_STEPS=8` to A/B the speed win. CFG (text=3.0,
speaker=5.0) is *not* a wheel knob — maneko's internal defaults are exactly
those values (parity-validated against the same Irodori), so nik's current sound
is the default path.

Book ref-reuse: the narrator latent is encoded once per ref-audio path and
reused across that voice's chunks (maneko's `encode_ref`/`generate_with_ref`),
turning the per-call DACVAE-encode into a once-per-narrator cost.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .voice import VoiceConfig

ENV_NUM_STEPS = "NIK_NUM_STEPS"
ENV_DEVICE = "NIK_MANEKO_DEVICE"

DEFAULT_NUM_STEPS = 20
DEFAULT_DEVICE = "metal"


def _default_num_steps() -> int:
    raw = os.environ.get(ENV_NUM_STEPS)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_NUM_STEPS


def _default_device() -> str:
    raw = (os.environ.get(ENV_DEVICE) or "").strip().lower()
    if raw in {"metal", "cpu"}:
        return raw
    return DEFAULT_DEVICE


class _Runtime:
    """One `maneko.Irodori` plus a per-ref-audio encoded-latent cache.

    nik calls `get_runtime()` once and `generate_chunk(runtime, text, voice)`
    per chunk; caching the `RefVoice` keyed by `voice.ref_audio` makes the
    DACVAE-encode a once-per-narrator cost instead of per-chunk.
    """

    def __init__(self, irodori, device: str) -> None:
        self._irodori = irodori
        self.device = device
        self._ref_cache: dict[Optional[str], object] = {}

    @property
    def sample_rate(self) -> int:
        return int(self._irodori.sample_rate)

    @property
    def irodori(self):
        return self._irodori

    def ref_for(self, ref_audio: Optional[str]):
        key = ref_audio or None
        ref = self._ref_cache.get(key)
        if ref is None:
            ref = self._irodori.encode_ref(key)
            self._ref_cache[key] = ref
        return ref


def _build_irodori(device: str):
    # Imported lazily so importing this module never fails when maneko isn't
    # installed (e.g. on the MLX/torch backends).
    import maneko

    return maneko.Irodori(device=device)


def get_runtime(*, hf_repo: Optional[str] = None):
    """Construct a maneko Irodori runtime (cached weights; pulls on first run).

    `hf_repo` is accepted for signature-compatibility with the other backends
    but ignored — maneko's weights are rev-pinned inside the wheel.
    """
    _ = hf_repo
    device = _default_device()
    try:
        irodori = _build_irodori(device)
    except Exception as exc:  # noqa: BLE001
        # Only auto-fall-back metal->cpu when the user didn't pin a device.
        if device == "metal" and not os.environ.get(ENV_DEVICE):
            irodori = _build_irodori("cpu")
            device = "cpu"
        else:
            raise RuntimeError(
                f"maneko Irodori(device={device!r}) failed to initialize: {exc}"
            ) from exc
    return _Runtime(irodori, device)


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

    `cfg_scale_text`/`cfg_scale_speaker`/`seed` are accepted for drop-in
    compatibility with the MLX/torch adapters but not plumbed: the maneko wheel
    surfaces only `{text, voice, seconds, steps}`, and its internal CFG defaults
    are exactly 3.0/5.0 — so the nik-default call already matches. `seconds` is
    left `None` so v3 auto-predicts the duration from text + speaker.
    """
    _ = (cfg_scale_text, cfg_scale_speaker, seed)
    if num_steps is None:
        num_steps = _default_num_steps()

    ref = runtime.ref_for(voice.ref_audio)
    samples = runtime.irodori.generate_with_ref(
        text, ref, seconds=None, steps=int(num_steps)
    )

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    return audio, int(runtime.sample_rate)
