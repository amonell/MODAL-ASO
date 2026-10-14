"""Install the pinned SpliceTransformer source and checkpoint in shared storage."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import json
import sys

from .asset_store import asset_lock, attest, download, file_lock, verify
from .model_sources import install_source
from .paths import splice_root, reference_root
from .reference_assets import CATALOG

SPLICETRANSFORMER_REVISION = "b67a51dabf27e2980331cec197e4396513c0b34c"
SPLICETRANSFORMER_WEIGHTS_FILE_ID = "1d8n4vHDSbXqpPc_JFEswLomSUDBgHvno"
SPLICETRANSFORMER_WEIGHTS_SHA256 = "bf9937c234139f850a770f84a26c65213a80d696d82426949a432ed1ccb9aa4e"
SPLICETRANSFORMER_WEIGHTS_URL = f"https://drive.usercontent.google.com/download?id={SPLICETRANSFORMER_WEIGHTS_FILE_ID}&export=download&confirm=t"
ProgressFn = Callable[[str], None]


def splicetransformer_root(repo_root: Path | None = None) -> Path:
    return Path(repo_root) if repo_root is not None else splice_root()


def required_splicetransformer_files(repo_root: Path | None = None) -> list[Path]:
    root = splicetransformer_root(repo_root)
    return [root / "tasks_annotate_mutations.py",
            root / "model" / "weights" / "SpTransformer_pytorch.ckpt",
            root / "data" / "data_package" / "hg38.fa"]


def missing_splicetransformer_files(repo_root: Path | None = None) -> list[Path]:
    return [p for p in required_splicetransformer_files(repo_root) if not p.is_file()]


def _ensure_checkpoint(root: Path, progress: ProgressFn) -> dict:
    destination = root / "model" / "weights" / "SpTransformer_pytorch.ckpt"
    try:
        return download(SPLICETRANSFORMER_WEIGHTS_URL, destination,
                        expected_sha256=SPLICETRANSFORMER_WEIGHTS_SHA256, progress=progress, attempts=1)
    except RuntimeError:
        # Google Drive occasionally requires its browser confirmation flow.
        import gdown
        with file_lock(asset_lock(destination)):
            if verify(destination, expected_sha256=SPLICETRANSFORMER_WEIGHTS_SHA256):
                return {"status": "ready", "path": str(destination)}
            temporary = destination.with_name(destination.name + ".gdown-partial")
            try:
                gdown.download(id=SPLICETRANSFORMER_WEIGHTS_FILE_ID, output=str(temporary), quiet=False)
                if not verify(temporary, expected_sha256=SPLICETRANSFORMER_WEIGHTS_SHA256, full=True):
                    raise ValueError("SpliceTransformer checkpoint checksum mismatch")
                temporary.replace(destination)
                attest(destination, source=SPLICETRANSFORMER_WEIGHTS_URL,
                       expected_sha256=SPLICETRANSFORMER_WEIGHTS_SHA256)
            finally:
                temporary.unlink(missing_ok=True)
        return {"status": "downloaded", "path": str(destination)}


def ensure_splicetransformer_assets(repo_root: Path | None = None,
                                     progress: ProgressFn | None = None) -> dict:
    root, emit = splicetransformer_root(repo_root), progress or print
    steps = []
    try:
        steps.append(install_source("https://github.com/ShenLab-Genomics/SpliceTransformer.git",
                                    SPLICETRANSFORMER_REVISION, root))
        steps.append(_ensure_checkpoint(root, emit))
        genome = reference_root() / CATALOG["human"]["fasta"]
        if not genome.is_file():
            raise FileNotFoundError("Install the human reference first: modal-aso setup --species human --features references")
        destination = root / "data" / "data_package" / "hg38.fa"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(asset_lock(destination)):
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != genome.resolve():
                    raise ValueError(f"SpliceTransformer uses a different genome: {destination}")
            else:
                destination.symlink_to(genome.resolve())
            source_index = Path(str(genome) + ".fai")
            target_index = Path(str(destination) + ".fai")
            if source_index.exists() and not target_index.exists():
                target_index.symlink_to(source_index)
        steps.append({"status": "ready", "path": str(destination)})
        return {"status": "ok", "steps": steps, "missing_files": [], "error": ""}
    except Exception as exc:
        return {"status": "error", "steps": steps,
                "missing_files": [str(p) for p in missing_splicetransformer_files(root)], "error": str(exc)}


def main(argv=None) -> int:
    result = ensure_splicetransformer_assets()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
