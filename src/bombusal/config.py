"""Central configuration loader for the Bombus AL pipeline.

Reads config.yaml once and resolves the project-relative directories to absolute
paths. Every pipeline script imports `CONFIG` (and the path helpers) from here
instead of hardcoding an absolute path.

    from bombusal.config import CONFIG, data_root, results_dir
"""
from pathlib import Path
import yaml

# config.yaml sits at the repository root (two levels up from this file).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config.yaml"

with _CONFIG_PATH.open("r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

_paths = CONFIG["paths"]
_project_root = Path(_paths.get("project_root") or _REPO_ROOT)


def data_root() -> Path:
    """Root of the confocal dataset (<species>/<sample>/*.tif)."""
    return Path(_paths["data_root"])


def project_root() -> Path:
    return _project_root


def metadata_dir() -> Path:
    return _project_root / _paths.get("metadata_dir", "metadata")


def results_dir() -> Path:
    d = _project_root / _paths.get("results_dir", "results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    return _project_root / _paths.get("models_dir", "models")
