"""Inspect installed model checkpoints, sources and runtime records."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .asset_store import verify, sha256
from .paths import oligoai_root, splice_root, data_root, resource_root


def _checkpoint(name, path, expected, *, full=False):
    return {"name": name, "path": str(path), "status": "optional" if not path.exists() else "verified" if verify(path, expected_sha256=expected, full=full) else "unverified",
            "bytes": path.stat().st_size if path.is_file() else 0}


def _source(name, root, expected):
    result = {"name": name, "path": str(root), "status": "optional" if not root.exists() else "unverified", "revision": expected}
    if (root / ".git").exists():
        try:
            actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL).strip()
            if actual == expected and not dirty:
                result["status"] = "verified"
        except (OSError, subprocess.CalledProcessError):
            pass
    return result


def inventory(species=("human", "mouse"), *, full=False):
    from .oligoai import MODEL_FILE, MODEL_SHA256, SOURCE_REVISION
    from .openspliceai_assets import SHA256 as MOUSE_SHA256
    from .splice_eval_assets import SPLICETRANSFORMER_WEIGHTS_SHA256, SPLICETRANSFORMER_REVISION
    values = []
    if "human" in species:
        root = splice_root()
        values += [_checkpoint("SpliceTransformer checkpoint", root / "model/weights/SpTransformer_pytorch.ckpt", SPLICETRANSFORMER_WEIGHTS_SHA256, full=full),
                   _source("SpliceTransformer source", root, SPLICETRANSFORMER_REVISION)]
    if "mouse" in species:
        import os
        root = Path(os.environ.get("OPENSPLICEAI_MODEL_DIR", data_root() / "models/openSpliceAI"))
        values.append(_checkpoint("OpenSpliceAI mouse checkpoint", root / "OSAI_mouse_10000nt_model1.pt", MOUSE_SHA256, full=full))
    root = oligoai_root()
    values += [_checkpoint("OligoAI checkpoint", root / MODEL_FILE, MODEL_SHA256, full=full),
               _source("OligoAI source", root / "source", SOURCE_REVISION)]
    record = root / "installation.json"
    result = {"name": "OligoAI environment", "path": str(root / "venv"), "status": "optional" if not record.exists() else "unverified"}
    try:
        saved = json.loads(record.read_text())
        expected = sha256(resource_root() / "requirements/oligoai.txt")
        if (saved["identity"]["requirements_sha256"] == expected and
                (root / "venv/bin/python").is_file() and saved["identity"]["model_sha256"] == MODEL_SHA256):
            result["status"] = "verified"
    except (OSError, ValueError, KeyError):
        pass
    if result["status"] == "unverified":
        result["detail"] = (f"OligoAI environment at {root / 'venv'} is present or recorded, "
                            "but its pinned installation record is incomplete or unverified. "
                            "Run modal-aso setup --features oligoai to validate or repair it.")
    values.append(result)
    return values
