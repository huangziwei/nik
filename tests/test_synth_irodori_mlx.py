from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_audio.codec.models.dacvae.codec import DACVAE, DACVAEConfig
from mlx_audio.tts.models.irodori_tts.irodori_tts import _find_silence_point

from nik import synth_irodori_mlx as adapter
from nik.voice import VoiceConfig

HOP = 4


class FakeRuntime:
    """Just enough of mlx-audio's Irodori Model for `generate_chunk`."""

    sample_rate = 48_000

    def __init__(self, *, loud_frames: int = 10, frames: int = 30, steps: int = 25):
        self.config = SimpleNamespace(
            sampler=SimpleNamespace(min_seconds=0.5, max_seconds=30.0),
            audio_downsample_factor=HOP,
        )
        self.dacvae = SimpleNamespace(decode=self._decode)
        self.loud_frames = loud_frames
        self.frames = frames
        self.steps = steps
        self.encoded: list[str] = []
        self.calls: list[dict] = []

    def _load_ref_waveform(self, path: str) -> mx.array:
        self.encoded.append(path)
        return mx.zeros((1, 8))

    def _encode_ref_audios(self, audios: list[mx.array]) -> tuple[mx.array, mx.array]:
        return mx.ones((1, 2, 3)), mx.ones((1, 2), dtype=mx.bool_)

    def generate_latents(self, **kwargs):
        self.calls.append(kwargs)
        rng = np.random.default_rng(0)
        latent = np.zeros((1, self.frames, 3), dtype=np.float32)
        latent[0, : self.loud_frames] = rng.normal(0, 1, (self.loud_frames, 3))
        return mx.array(latent), self.steps

    def _decode(self, latent_t: mx.array, chunk_size: int) -> mx.array:
        assert chunk_size == 50
        samples = latent_t.shape[2] * HOP
        return mx.arange(samples, dtype=mx.float32).reshape(1, samples, 1)


@pytest.fixture(autouse=True)
def _fresh_ref_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "_ref_cache", {})
    monkeypatch.delenv("NIK_NUM_STEPS", raising=False)
    monkeypatch.delenv("NIK_MLX_CFG_MODE", raising=False)
    monkeypatch.delenv("NIK_STYLE_EMOJI", raising=False)


def _voice(path: Path) -> VoiceConfig:
    return VoiceConfig(name=path.stem, ref_audio=str(path), ref_text=None)


def test_defaults_are_v3_at_20_steps() -> None:
    assert adapter.resolve_hf_repo() == "mlx-community/Irodori-TTS-500M-v3-8bit"
    assert adapter.default_num_steps() == 20


def test_reference_clip_is_encoded_once_per_voice_file(tmp_path: Path) -> None:
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"0" * 10)
    other = tmp_path / "b.wav"
    other.write_bytes(b"0" * 10)
    runtime = FakeRuntime()

    for _ in range(3):
        adapter.generate_chunk(runtime, "本文。", _voice(clip))
    assert runtime.encoded == [str(clip)]

    adapter.generate_chunk(runtime, "本文。", _voice(other))
    assert runtime.encoded == [str(clip), str(other)]

    # A re-cloned voice (new file content) is encoded again.
    clip.write_bytes(b"1" * 20)
    adapter.generate_chunk(runtime, "本文。", _voice(clip))
    assert runtime.encoded == [str(clip), str(other), str(clip)]

    # Each loaded model keeps its own encodings.
    second = FakeRuntime()
    adapter.generate_chunk(second, "本文。", _voice(clip))
    assert second.encoded == [str(clip)]


def test_generate_chunk_sampling_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"0")
    runtime = FakeRuntime()

    adapter.generate_chunk(runtime, "本文。", _voice(clip))
    first = runtime.calls[-1]
    assert first["num_steps"] == 20
    assert first["rng_seed"] == 0
    assert first["text"] == "本文。"
    assert "cfg_scale_text" not in first and "cfg_guidance_mode" not in first

    monkeypatch.setenv("NIK_NUM_STEPS", "12")
    monkeypatch.setenv("NIK_MLX_CFG_MODE", "joint")
    adapter.generate_chunk(
        runtime, "本文。", _voice(clip), seed=7, cfg_scale_text=2, style_emoji="😌"
    )
    second = runtime.calls[-1]
    assert second["num_steps"] == 12
    assert second["rng_seed"] == 7
    assert second["cfg_scale_text"] == 2.0
    assert second["cfg_guidance_mode"] == "joint"
    assert second["text"] == "😌本文。"

    adapter.generate_chunk(runtime, "本文。", _voice(clip), num_steps=40)
    assert runtime.calls[-1]["num_steps"] == 40


@pytest.mark.parametrize(
    ("loud_frames", "steps", "expected_frames"),
    [
        (10, 25, 10),  # trailing silence trimmed
        (10, 6, 6),  # predicted duration is shorter still
        (0, 25, 25),  # silence from frame 0 must not empty the clip
        (30, 25, 25),  # no silence: predicted duration
    ],
)
def test_generate_chunk_trims_like_mlx_audio(
    tmp_path: Path, loud_frames: int, steps: int, expected_frames: int
) -> None:
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"0")
    runtime = FakeRuntime(loud_frames=loud_frames, steps=steps)
    audio, sample_rate = adapter.generate_chunk(runtime, "本文。", _voice(clip))
    assert sample_rate == 48_000
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(
        audio, np.arange(expected_frames * HOP, dtype=np.float32)
    )


def test_silence_frame_matches_mlx_audio() -> None:
    rng = np.random.default_rng(1101)
    for _ in range(60):
        frames = int(rng.integers(1, 120))
        latent = rng.normal(0, rng.choice([0.01, 0.03, 0.06, 1.0]), (frames, 8))
        cut = int(rng.integers(0, frames))
        latent[cut:] *= rng.choice([0.0, 0.01, 0.02])
        x = mx.array(latent.astype(np.float32))
        assert adapter._silence_frame(x) == _find_silence_point(x)


def test_fold_weight_norm_keeps_codec_output_identical() -> None:
    mx.random.seed(0)
    codec = DACVAE(
        DACVAEConfig(
            encoder_dim=4,
            encoder_rates=[2, 3],
            latent_dim=12,
            decoder_dim=24,
            decoder_rates=[3, 2],
            codebook_dim=4,
        )
    )
    mx.eval(codec.parameters())
    wav = mx.random.normal((1, 96, 1)) * 0.1
    latent = codec.encode(wav)
    audio = codec.decode(latent, chunk_size=6)
    mx.eval(latent, audio)

    adapter._fold_weight_norm(codec)

    assert not any(
        getattr(module, "use_weight_norm", False) for _, module in codec.named_modules()
    )
    folded_latent = codec.encode(wav)
    np.testing.assert_array_equal(np.array(folded_latent), np.array(latent))
    np.testing.assert_array_equal(
        np.array(codec.decode(folded_latent, chunk_size=6)), np.array(audio)
    )
