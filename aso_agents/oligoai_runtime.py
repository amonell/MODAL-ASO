"""Install an independent, pinned Linux/CUDA environment for OligoAI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .asset_store import asset_lock, atomic_json, download, file_lock, sha256
from .model_sources import install_source
from .oligoai import SOURCE_REVISION, MODEL_REVISION, MODEL_SHA256, MODEL_FILE
from .paths import cache_root, oligoai_root, resource_root, save_settings

FLASH_NAME = "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
FLASH_WHEEL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/" + FLASH_NAME.replace("+", "%2B")
FLASH_SHA256 = "58853b28a5a926cae14402bfd8d4d93a45ebf8f9e79533f37ab09d0d77a99c05"


def _probe(python, source):
    script = """
import json,sys
sys.path.insert(0, sys.argv[1])
import torch, flash_attn
from flash_attn.layers.rotary import RotaryEmbedding
from train_aso import ASOInhibitionPredictionWrapper
assert sys.version_info[:2] == (3, 11), 'Python 3.11 is required'
assert torch.__version__.split('+')[0] == '2.6.0', 'PyTorch version differs'
assert not torch._C._GLIBCXX_USE_CXX11_ABI, 'FlashAttention ABI differs'
assert torch.cuda.is_available(), 'CUDA is unavailable; check the NVIDIA driver and GPU access'
assert torch.cuda.get_device_capability(0)[0] >= 8, 'OligoAI requires Ampere or newer NVIDIA hardware'
with torch.inference_mode():
    qkv = torch.randn(1, 16, 3, 2, 64, device='cuda', dtype=torch.float16)
    RotaryEmbedding(64, device='cuda')(qkv)
    flash_attn.flash_attn_qkvpacked_func(qkv)
torch.cuda.synchronize()
print(json.dumps({'python':sys.version.split()[0], 'torch':torch.__version__,
                  'flash_attention':flash_attn.__version__, 'gpu':torch.cuda.get_device_name(0)}))
"""
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return json.loads(subprocess.check_output([str(python), "-c", script, str(source)], env=env, text=True, stderr=subprocess.PIPE))


def install(root=None, native_python=None):
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("OligoAI requires Linux x86-64 and an NVIDIA Ampere or newer GPU")
    root = Path(root or oligoai_root()).expanduser().resolve()
    native_python = Path(native_python or sys.executable)
    requirements = resource_root() / "requirements" / "oligoai.txt"
    identity = {"source_revision": SOURCE_REVISION, "model_revision": MODEL_REVISION,
                "model_sha256": MODEL_SHA256, "requirements_sha256": sha256(requirements),
                "flash_sha256": FLASH_SHA256, "environment_schema": 1}
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(asset_lock(root / "installation.json")):
        source = root / "source"
        install_source("https://github.com/barneyhill/OligoAI.git", SOURCE_REVISION, source)
        checkpoint = root / MODEL_FILE
        download(f"https://huggingface.co/barneyhill/OligoAI/resolve/{MODEL_REVISION}/{MODEL_FILE}",
                 checkpoint, expected_sha256=MODEL_SHA256)
        python = root / "venv" / "bin" / "python"
        try:
            previous = json.loads((root / "installation.json").read_text())
        except (OSError, ValueError):
            previous = {}
        if previous.get("identity") == identity and python.is_file():
            _probe(python, source)
            save_settings({"oligoai_root": str(root)})
            return root
        if not python.is_file():
            subprocess.run([str(native_python), "-m", "venv", str(root / "venv")], check=True)
        legacy_link = root / "venv/lib/python3.11/site-packages/modal_runtime.pth"
        if legacy_link.exists():
            legacy_link.replace(legacy_link.with_suffix(".pth.disabled"))
        # Every dependency is installed here; host site-packages are never added.
        env = dict(os.environ, PIP_CACHE_DIR=str(cache_root() / "pip"), PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        subprocess.run([str(python), "-m", "pip", "install", "--require-hashes", "-r", str(requirements)], env=env, check=True)
        wheel = cache_root() / "wheels" / FLASH_NAME
        download(FLASH_WHEEL, wheel, expected_sha256=FLASH_SHA256, kind="zip")
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], env=env, check=True)
        probe = _probe(python, source)
        packages = json.loads(subprocess.check_output([str(python), "-m", "pip", "list", "--format=json"], env=env, text=True))
        atomic_json(root / "installation.json", {"identity": identity, "runtime": probe, "packages": packages})
    save_settings({"oligoai_root": str(root)})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--native-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    print(f"Ready: {install(args.root, args.native_python)}")


if __name__ == "__main__":
    main()
