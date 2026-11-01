import os
import subprocess
import sys
from pathlib import Path

import nik

_REPO_ROOT = Path(nik.__file__).resolve().parent.parent
_PROBE = (
    "import nik\nfrom huggingface_hub import constants\nprint(constants.HF_HUB_CACHE)\n"
)


def _hub_cache_in_fresh_interpreter(env: dict[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_importing_nik_keeps_model_weights_in_repo_cache() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"}
    }
    expected = _REPO_ROOT / ".cache" / "huggingface" / "hub"
    assert _hub_cache_in_fresh_interpreter(env) == str(expected)


def test_explicit_hf_home_is_respected(tmp_path: Path) -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"}
    }
    env["HF_HOME"] = str(tmp_path / "hf")
    assert _hub_cache_in_fresh_interpreter(env) == str(tmp_path / "hf" / "hub")
