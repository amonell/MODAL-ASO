"""Persistent, revision-pinned model source checkouts."""

from pathlib import Path
import subprocess

from .asset_store import asset_lock, file_lock


def install_source(url: str, revision: str, destination: Path) -> dict:
    destination = Path(destination)
    with file_lock(asset_lock(destination)):
        if (destination / ".git").is_dir():
            actual = subprocess.check_output(
                ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
            modified = subprocess.check_output(
                ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
            if actual == revision and not modified:
                return {"status": "ready", "revision": revision, "path": str(destination)}
            raise ValueError(f"Model source differs from the pinned release: {destination}. Choose a new model directory.")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Model directory contains unverified files: {destination}")
        temporary = destination.with_name(destination.name + ".installing")
        if not (temporary / ".git").is_dir():
            temporary.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(temporary)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(temporary), "fetch", "--depth", "1", url, revision], check=True)
        subprocess.run(["git", "-C", str(temporary), "checkout", "--detach", revision], check=True)
        if destination.exists():
            destination.rmdir()
        temporary.replace(destination)
        return {"status": "installed", "revision": revision, "path": str(destination)}
