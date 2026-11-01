import os
from pathlib import Path

__all__ = ["CACHE_DIR", "__version__"]

__version__ = "0.1.0"

# Model weights, dictionaries and derived caches live in the repo's .cache only:
# one copy, in one known place.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

# huggingface_hub reads HF_HOME when it is first imported, so this must run before
# any nik module (or mlx-audio/transformers) pulls it in. An explicit HF_HOME wins.
if (CACHE_DIR.parent / "pyproject.toml").is_file():
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
