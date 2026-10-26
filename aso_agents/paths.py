"""Portable locations for installed resources, user data and run outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile


def config_file() -> Path:
    override = os.environ.get("MODAL_ASO_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/MODAL-ASO/config.json"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "modal-aso/config.json"


def settings() -> dict:
    path = config_file()
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read MODAL-ASO settings at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"MODAL-ASO settings must be an object: {path}")
    return value


def save_settings(updates: dict) -> None:
    from .asset_store import file_lock, atomic_json
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(".lock")):
        atomic_json(path, {**settings(), **updates})


def data_root() -> Path:
    configured = os.environ.get("MODAL_ASO_DATA_DIR") or settings().get("data_dir")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/MODAL-ASO"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "modal-aso"


def cache_root() -> Path:
    configured = os.environ.get("MODAL_ASO_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/MODAL-ASO"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "modal-aso"


def reference_root() -> Path:
    configured = os.environ.get("ASO_REFERENCE_DATA_ROOT") or settings().get("reference_dir")
    return Path(configured).expanduser().resolve() if configured else data_root() / "references"


def oligoai_root() -> Path:
    configured = os.environ.get("ASO_OLIGOAI_ROOT") or settings().get("oligoai_root")
    if configured is not None:
        if not isinstance(configured, (str, os.PathLike)):
            raise ValueError("The configured OligoAI directory must be a path")
        if not str(configured).strip():
            raise ValueError("The configured OligoAI directory must not be empty")
        return Path(configured).expanduser().resolve()
    return data_root() / "models/oligoai"


def splice_root() -> Path:
    override = os.environ.get("ASO_SPLICETRANSFORMER_ROOT")
    return Path(override).expanduser().resolve() if override else data_root() / "models/SpliceTransformer"


def runs_root() -> Path:
    configured = os.environ.get("ASO_RUNS_DIR")
    return Path(configured).expanduser().resolve() if configured else Path.cwd() / "runs"


def resource_root() -> Path:
    return Path(__file__).resolve().parent / "resources"


def shared_gpu_lock() -> Path:
    configured = os.environ.get("ASO_SHARED_GPU_LOCK")
    return Path(configured).expanduser() if configured else Path(tempfile.gettempdir()) / f"modal-aso-gpu-{os.getuid()}.lock"
