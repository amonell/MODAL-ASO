"""Offline installation discovery and explicit runtime preflight for OligoAI."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

from .paths import oligoai_root, save_settings


class OligoAIUnavailable(RuntimeError):
    """The optional predictor is inaccessible, not a failed scientific result."""


def inspect_installation(root=None, *, verify=False):
    """Inspect assets without importing Torch, using the GPU, or downloading."""
    from .oligoai import MODEL_FILE, MODEL_SHA256, SOURCE_REVISION, file_sha

    root = Path(root or oligoai_root()).expanduser().resolve()
    paths = {
        "Python interpreter": root / "venv/bin/python",
        "model checkpoint": root / MODEL_FILE,
        "model source": root / "source",
    }
    missing = [f"{name}: {path}" for name, path in paths.items()
               if not (path.is_dir() if name == "model source" else path.is_file())]
    if paths["Python interpreter"].is_file() and not os.access(paths["Python interpreter"], os.X_OK):
        missing.append(f"Python interpreter is not executable: {paths['Python interpreter']}")
    result = {"root": str(root), "status": "incomplete" if root.exists() else "missing",
              "assets_verified": False, "runtime_checked": False}
    if missing:
        result["detail"] = (
            f"OligoAI is not accessible at {root}. Missing or inaccessible: "
            + "; ".join(missing)
            + ". If it is already installed elsewhere, register it with "
            "modal-aso assets import-model oligoai PATH. Otherwise run "
            "modal-aso setup --features oligoai. In a container or sandbox, "
            "make the model directory and its interpreter accessible there, "
            "then run modal-aso doctor --require-model oligoai in that environment."
        )
        return result
    if verify:
        from .model_assets import _source
        if file_sha(paths["model checkpoint"]) != MODEL_SHA256:
            result.update(status="invalid", detail=f"OligoAI checkpoint checksum mismatch at {paths['model checkpoint']}. Repair the configured installation with modal-aso setup --features oligoai.")
            return result
        if _source("OligoAI source", paths["model source"], SOURCE_REVISION)["status"] != "verified":
            result.update(status="invalid", detail=f"OligoAI model source is modified or does not match the pinned release: {paths['model source']}.")
            return result
        result["assets_verified"] = True
    result.update(status="installed", detail=f"OligoAI assets are present at {root}; runtime/GPU access has not been tested.")
    return result


def register_existing(root):
    """Remember verified assets; never install packages or claim GPU readiness."""
    result = inspect_installation(root, verify=True)
    if result["status"] != "installed":
        raise RuntimeError(result["detail"])
    save_settings({"oligoai_root": result["root"]})
    return {**result, "registered": True,
            "next_step": "Run modal-aso doctor --require-model oligoai in the environment that will execute the request."}


def _install_for_run(root: Path, log_path: Path) -> None:
    """Use the existing pinned installer, keeping its output off the live timer."""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONHOME", None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(
            [sys.executable, "-m", "aso_agents.oligoai_runtime", "--root", str(root)],
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            check=True,
        )


def ensure_installation_for_prediction(root, directory):
    """Offer setup when uncached, eligible predictions need missing assets."""
    result = inspect_installation(root)
    if result["status"] == "installed":
        return result
    from .runtime_warnings import emit_progress, emit_warning, request_model_installation

    root = Path(root).expanduser().resolve()
    if not request_model_installation("OligoAI", str(root), result["detail"]):
        raise OligoAIUnavailable(result["detail"])
    log_path = Path(directory) / "installation.log"
    emit_progress("Installing OligoAI for gapmer efficacy predictions.")
    try:
        _install_for_run(root, log_path)
        result = inspect_installation(root)
        if result["status"] != "installed":
            raise RuntimeError(result["detail"])
    except Exception as exc:
        detail = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and log_path.is_file():
            lines = log_path.read_text(errors="replace").strip().splitlines()
            if lines:
                detail = lines[-1]
        message = (f"OligoAI installation failed: {detail}. "
                   f"Continuing without gapmer efficacy predictions. Installation log: {log_path}. "
                   "Retry setup with modal-aso setup --features oligoai.")
        emit_warning(message, category="OligoAI")
        raise OligoAIUnavailable(message) from exc
    emit_progress("OligoAI is ready. Predicting gapmer efficacy.")
    return result


def require_oligoai():
    """Fail before a request starts unless the actual isolated runtime works."""
    result = inspect_installation(verify=True)
    if result["status"] != "installed":
        raise RuntimeError(result["detail"])
    from .oligoai_runtime import _probe
    root = Path(result["root"])
    try:
        runtime = _probe(root / "venv/bin/python", root / "source")
    except Exception as exc:
        # Assets being present does not prove that their interpreter, shared
        # libraries or CUDA device are available inside the current namespace.
        captured = getattr(exc, "stderr", None)
        if captured:
            lines = (captured.decode(errors="replace") if isinstance(captured, bytes) else str(captured)).strip().splitlines()
            detail = lines[-1] if lines else "The isolated runtime did not provide an error detail."
        elif isinstance(exc, subprocess.CalledProcessError):
            detail = f"The isolated runtime exited with status {exc.returncode}."
        else:
            detail = str(exc)
        raise RuntimeError(
            f"OligoAI is installed at {root}, but its runtime is not usable in this environment. "
            "Check access to its Python environment and a supported NVIDIA GPU. "
            "Run modal-aso setup --features oligoai to repair the configured installation. "
            f"Runtime check: {detail}"
        ) from exc
    return {**result, "status": "ready", "runtime_checked": True,
            "runtime": runtime, "detail": f"OligoAI runtime and CUDA access verified at {root}."}
