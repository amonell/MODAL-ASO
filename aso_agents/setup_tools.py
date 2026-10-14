"""Install command-line tools in a persistent, user-owned environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from .asset_store import download, file_lock, asset_lock, atomic_json, sha256, attest, verify
from .paths import data_root, cache_root, resource_root

MAMBA_RELEASE = "2.9.0-0"
MAMBA_SHA256 = {
    "linux-64": "366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3",
    "osx-arm64": "ec2a072f028e1a7cf20f3e2e74d5a8127cf5a5f27636375b5359811565f4e5be",
}
TOOL_SPECS = ["bowtie=1.3.1", "bedtools=2.31.1", "samtools=1.22.1",
              "subread=2.1.1", "viennarna=2.7.0", "ucsc-bedtobigbed=482"]
BINARIES = ["bowtie", "bowtie-build", "bowtie-inspect", "bedtools", "samtools",
            "featureCounts", "RNAfold", "RNAplfold", "bedToBigBed"]


def platform_key() -> str:
    value = (platform.system(), platform.machine().lower())
    keys = {("Linux", "x86_64"): "linux-64", ("Darwin", "arm64"): "osx-arm64"}
    if value not in keys:
        raise RuntimeError("Native setup supports Linux x86-64 and Apple silicon macOS. On Windows, run inside WSL2.")
    return keys[value]


def install_tools(*, progress=print) -> dict:
    key = platform_key()
    prefix = data_root() / "runtime"
    receipt = prefix / "installation.json"
    lock = resource_root() / "requirements" / ("tools-" + key + ".txt")
    lock_digest = sha256(lock)
    with file_lock(asset_lock(prefix)):
        try:
            previous = json.loads(receipt.read_text())
        except (OSError, ValueError):
            previous = {}
        if (previous.get("platform") == key and previous.get("lock_sha256") == lock_digest
                and all(verify(prefix / "bin" / name) for name in BINARIES)):
            progress("Ready: command-line tools")
            return {"status": "ready", "prefix": str(prefix)}
        bootstrap = data_root() / "bootstrap" / ("micromamba-" + key)
        url = f"https://github.com/mamba-org/micromamba-releases/releases/download/{MAMBA_RELEASE}/micromamba-{key}"
        download(url, bootstrap, expected_sha256=MAMBA_SHA256[key], progress=progress)
        bootstrap.chmod(0o755)
        env = dict(os.environ, MAMBA_ROOT_PREFIX=str(cache_root() / "mamba"))
        operation = ["install", "--force-reinstall"] if (prefix / "conda-meta").is_dir() else ["create"]
        command = [str(bootstrap), *operation, "--yes", "--prefix", str(prefix),
                   "--override-channels", "--file", str(lock)]
        progress("Installing command-line tools...")
        subprocess.run(command, env=env, check=True)
        missing = [name for name in BINARIES if not (prefix / "bin" / name).is_file()]
        if missing:
            raise RuntimeError("Missing runtime tools after installation: " + ", ".join(missing))
        for name in BINARIES:
            attest(prefix / "bin" / name, source="runtime-lock:sha256:" + lock_digest)
        packages = json.loads(subprocess.check_output([str(bootstrap), "list", "--prefix", str(prefix), "--json"], env=env, text=True))
        atomic_json(receipt, {"platform": key, "specifications": TOOL_SPECS, "lock_sha256": lock_digest, "packages": packages})
        return {"status": "installed", "prefix": str(prefix)}
