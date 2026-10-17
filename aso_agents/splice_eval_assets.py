"""Install local SpliceTransformer assets that preflight cannot assume are present.

Human splice scoring needs the SpliceTransformer submodule, its PyTorch
checkpoint, and an hg38 FASTA at the upstream data_package path. The
checkpoint is hosted on Google Drive; the FASTA can be reused from the
already-downloaded human tiling genome when available.
"""

from __future__ import annotations

import gzip
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


ProgressFn = Callable[[str], None]

SPLICETRANSFORMER_SUBMODULE = "spliceAI_integration/SpliceTransformer"
SPLICETRANSFORMER_WEIGHTS_FILE_ID = "1d8n4vHDSbXqpPc_JFEswLomSUDBgHvno"
SPLICETRANSFORMER_WEIGHTS_URL = (
    "https://drive.google.com/file/d/1d8n4vHDSbXqpPc_JFEswLomSUDBgHvno/view?usp=drive_link"
)
SPLICETRANSFORMER_HG38_FASTA_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
MIN_CHECKPOINT_BYTES = 10 * 1024 * 1024


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def splicetransformer_root(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or _repo_root()) / "spliceAI_integration" / "SpliceTransformer"


def required_splicetransformer_files(repo_root: Optional[Path] = None) -> List[Path]:
    root = splicetransformer_root(repo_root)
    return [
        root / "tasks_annotate_mutations.py",
        root / "model" / "weights" / "SpTransformer_pytorch.ckpt",
        root / "data" / "data_package" / "hg38.fa",
    ]


def missing_splicetransformer_files(repo_root: Optional[Path] = None) -> List[Path]:
    return [path for path in required_splicetransformer_files(repo_root) if not path.exists()]


def _emit(progress: Optional[ProgressFn], message: str) -> None:
    if progress:
        progress(message)
    else:
        print(message, flush=True)


def _existing_human_genome_fasta() -> Optional[Path]:
    try:
        from aso_agents.species_config import get_species_config

        fasta = Path(get_species_config("human").genome_fasta)
    except Exception:
        return None
    return fasta if fasta.exists() and fasta.stat().st_size > 0 else None


def _link_or_copy(source: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(source.resolve())
        return "linked"
    except OSError:
        shutil.copy2(source, dest)
        return "copied"


def _init_splicetransformer_submodule(repo_root: Path, progress: Optional[ProgressFn]) -> Dict[str, Any]:
    tasks = splicetransformer_root(repo_root) / "tasks_annotate_mutations.py"
    if tasks.exists():
        return {"status": "ok", "action": "already_present", "path": str(tasks)}
    _emit(progress, "Initializing SpliceTransformer submodule...")
    proc = subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive", SPLICETRANSFORMER_SUBMODULE],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0 or not tasks.exists():
        detail = (proc.stderr or proc.stdout or "").strip()
        return {
            "status": "error",
            "action": "submodule_init",
            "path": str(tasks),
            "error": detail or f"git submodule update failed with code {proc.returncode}",
        }
    return {"status": "ok", "action": "initialized", "path": str(tasks)}


def _ensure_hg38_fasta(repo_root: Path, progress: Optional[ProgressFn]) -> Dict[str, Any]:
    dest = splicetransformer_root(repo_root) / "data" / "data_package" / "hg38.fa"
    if dest.exists() and dest.stat().st_size > 0:
        return {"status": "ok", "action": "already_present", "path": str(dest)}
    existing = _existing_human_genome_fasta()
    if existing is not None:
        _emit(progress, f"Reusing the local human genome FASTA for SpliceTransformer: {existing}")
        method = _link_or_copy(existing, dest)
        return {"status": "ok", "action": method, "path": str(dest), "source": str(existing)}
    _emit(progress, f"Downloading hg38 FASTA from {SPLICETRANSFORMER_HG38_FASTA_URL} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    gz_tmp = dest.with_suffix(dest.suffix + ".gz.partial")
    fa_tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(SPLICETRANSFORMER_HG38_FASTA_URL, gz_tmp)
        with gzip.open(gz_tmp, "rb") as src, fa_tmp.open("wb") as out:
            shutil.copyfileobj(src, out)
        fa_tmp.replace(dest)
    except Exception as exc:
        for leftover in (gz_tmp, fa_tmp):
            if leftover.exists():
                leftover.unlink()
        return {"status": "error", "action": "download_fasta", "path": str(dest), "error": str(exc)}
    if gz_tmp.exists():
        gz_tmp.unlink()
    return {"status": "ok", "action": "downloaded", "path": str(dest)}


def _looks_like_html(path: Path) -> bool:
    try:
        head = path.read_bytes()[:256].lstrip().lower()
    except Exception:
        return False
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


def _download_google_drive_file(file_id: str, dest: Path, progress: Optional[ProgressFn]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        import gdown  # type: ignore

        _emit(progress, "Downloading SpliceTransformer weights with gdown...")
        gdown.download(id=file_id, output=str(tmp), quiet=False)
    except Exception:
        _emit(progress, "Downloading SpliceTransformer weights from Google Drive...")
        _download_google_drive_urllib(file_id, tmp)
    if not tmp.exists() or tmp.stat().st_size < MIN_CHECKPOINT_BYTES or _looks_like_html(tmp):
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            "Google Drive did not return the SpliceTransformer checkpoint. "
            f"Download it from {SPLICETRANSFORMER_WEIGHTS_URL} and save it as {dest}."
        )
    tmp.replace(dest)


def _stream_response_to_path(response: Any, dest: Path) -> None:
    with dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_google_drive_urllib(file_id: str, dest: Path) -> None:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with opener.open(url, timeout=120) as response:
        content_type = str(response.headers.get("Content-Type") or "")
        if "text/html" in content_type:
            html = response.read().decode("utf-8", errors="replace")
            confirm = ""
            match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
            if match:
                confirm = match.group(1)
            if not confirm:
                match = re.search(r'name="confirm"\s+value="([^"]+)"', html)
                if match:
                    confirm = match.group(1)
            confirm_url = (
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm or 't'}"
            )
            with opener.open(confirm_url, timeout=120) as confirmed:
                _stream_response_to_path(confirmed, dest)
            return
        _stream_response_to_path(response, dest)


def _ensure_checkpoint(repo_root: Path, progress: Optional[ProgressFn]) -> Dict[str, Any]:
    dest = splicetransformer_root(repo_root) / "model" / "weights" / "SpTransformer_pytorch.ckpt"
    if dest.exists() and dest.stat().st_size >= MIN_CHECKPOINT_BYTES and not _looks_like_html(dest):
        return {"status": "ok", "action": "already_present", "path": str(dest)}
    try:
        _download_google_drive_file(SPLICETRANSFORMER_WEIGHTS_FILE_ID, dest, progress)
    except Exception as exc:
        return {
            "status": "error",
            "action": "download_checkpoint",
            "path": str(dest),
            "error": str(exc),
            "url": SPLICETRANSFORMER_WEIGHTS_URL,
        }
    return {"status": "ok", "action": "downloaded", "path": str(dest)}


def ensure_splicetransformer_assets(
    repo_root: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Install missing SpliceTransformer files. Safe to re-run; existing files are kept."""
    root = repo_root or _repo_root()
    steps = [
        _init_splicetransformer_submodule(root, progress),
        _ensure_hg38_fasta(root, progress),
        _ensure_checkpoint(root, progress),
    ]
    missing = [str(path) for path in missing_splicetransformer_files(root)]
    errors = [step for step in steps if step.get("status") != "ok"]
    status = "ok" if not missing and not errors else "error"
    return {
        "status": status,
        "steps": steps,
        "missing_files": missing,
        "error": errors[0].get("error") if errors else "",
    }


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    result = ensure_splicetransformer_assets()
    print(result, flush=True)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
