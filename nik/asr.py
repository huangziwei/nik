from __future__ import annotations

from pathlib import Path
from typing import Optional


_LANGUAGE_MAP = {
    "japanese": "ja",
    "ja": "ja",
    "english": "en",
    "en": "en",
    "chinese": "zh",
    "zh": "zh",
    "korean": "ko",
    "ko": "ko",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "spanish": "es",
    "es": "es",
}


def normalize_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    return _LANGUAGE_MAP.get(lowered, lowered)


def transcribe_audio(
    audio_path: Path,
    model_name: str = "small",
    language: Optional[str] = None,
    device: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    try:
        import whisper
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Whisper is not installed. Install it with `uv sync --extra whisper`."
        ) from exc

    kwargs = {}
    normalized_lang = normalize_language(language)
    if normalized_lang:
        kwargs["language"] = normalized_lang
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    fp16 = False
    try:  # pragma: no cover - torch/device logic
        import torch

        if device:
            if "cuda" in device or "mps" in device:
                fp16 = True
        elif torch.cuda.is_available():
            fp16 = True
    except Exception:
        fp16 = False
    kwargs["fp16"] = fp16

    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(str(audio_path), **kwargs)
    text = str(result.get("text") or "").strip()
    return text
