"""Verified, resumable downloads and atomic asset installation."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
from http.client import HTTPException
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile
import zlib

from .paths import cache_root

PARALLEL_THRESHOLD = 128 * 1024 * 1024
DOWNLOAD_CONNECTIONS = 4


class _RangeUnavailable(OSError):
    pass


def _parallel_transfer(url: str, partial: Path, *, offset: int, total: int,
                       etag: str, headers: dict) -> int:
    """Fetch disjoint ranges, then append a contiguous prefix to the partial file."""
    staging = partial.with_name(partial.name + ".parts")
    staging.mkdir(exist_ok=True)
    identity = {"url": url, "etag": etag, "total": total, "offset": offset}
    manifest = staging / "transfer.json"
    try:
        previous = json.loads(manifest.read_text())
    except (OSError, ValueError):
        previous = {}
    if previous != identity:
        for path in staging.iterdir():
            if path.is_file():
                path.unlink()
        atomic_json(manifest, identity)
    size = (total - offset + DOWNLOAD_CONNECTIONS - 1) // DOWNLOAD_CONNECTIONS
    spans = [(start, min(start + size, total) - 1) for start in range(offset, total, size)]

    def fetch(span):
        start, end = span
        target = staging / f"{start}-{end}"
        present = target.stat().st_size if target.exists() else 0
        if present > end - start + 1:
            target.unlink()
            present = 0
        if present == end - start + 1:
            return target, 0
        chunk_headers = {**headers, "Range": f"bytes={start + present}-{end}", "If-Range": etag}
        received = 0
        with urlopen(Request(url, headers=chunk_headers), timeout=120) as response:
            expected_range = f"bytes {start + present}-{end}/{total}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise _RangeUnavailable("Server does not support the requested byte ranges")
            if response.headers.get("ETag") != etag:
                raise _RangeUnavailable("Asset changed during a range transfer")
            with target.open("ab" if present else "wb") as stream:
                while chunk := response.read(8 * 1024 * 1024):
                    if received + len(chunk) > end - start + 1 - present:
                        raise _RangeUnavailable("Server sent data outside the requested range")
                    stream.write(chunk)
                    received += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if target.stat().st_size != end - start + 1:
            raise OSError("Range transfer ended early")
        return target, received

    with ThreadPoolExecutor(max_workers=DOWNLOAD_CONNECTIONS) as pool:
        completed = list(pool.map(fetch, spans))
    if not partial.exists():
        partial.touch()
    if partial.stat().st_size != offset:
        raise OSError("Partial file changed during range transfer")
    with partial.open("ab") as output:
        for path, _ in completed:
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
            path.unlink()
    manifest.unlink()
    staging.rmdir()
    return sum(received for _, received in completed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def asset_lock(path: Path) -> Path:
    identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    return cache_root() / "locks" / (identity + ".lock")


def receipt_path(path: Path) -> Path:
    return path.parent / ".modal-aso" / (path.name + ".json")


def fingerprint(path: Path) -> dict:
    value = path.stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "device": value.st_dev, "inode": value.st_ino}


def read_receipt(path: Path) -> dict:
    try:
        value = json.loads(receipt_path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def validate_file(path: Path, kind: str | None = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Asset is empty or missing: {path}")
    with path.open("rb") as stream:
        head = stream.read(4096)
    if head.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError(f"Download returned an HTML page: {path}")
    if kind == "gzip" or (kind is None and path.name.endswith(".gz")):
        with gzip.open(path, "rb") as stream:
            while stream.read(8 * 1024 * 1024):
                pass
    elif kind == "zip" or (kind is None and path.name.endswith(".zip")):
        with zipfile.ZipFile(path) as archive:
            failed = archive.testzip()
            if failed:
                raise ValueError(f"Corrupt archive member: {failed}")
    elif kind == "fasta" and not head.startswith(b">"):
        raise ValueError(f"Expected a FASTA header: {path}")
    elif kind == "bigwig" and head[:4] not in (b"\x26\xfc\x8f\x88", b"\x88\x8f\xfc\x26"):
        raise ValueError(f"Expected a bigWig file: {path}")
    elif kind == "hdf5":
        signature = b"\x89HDF\r\n\x1a\n"
        found, offset = head[:8] == signature, 512
        # HDF5 permits an optional user block before the superblock.
        with path.open("rb") as stream:
            while not found and offset < path.stat().st_size:
                stream.seek(offset)
                found = stream.read(8) == signature
                offset *= 2
        if not found:
            raise ValueError(f"Expected an HDF5 file: {path}")


def verify(path: Path, *, expected_sha256: str | None = None, full: bool = False) -> bool:
    if not path.is_file():
        return False
    receipt = read_receipt(path)
    expected = expected_sha256 or receipt.get("sha256")
    if not expected:
        return False
    if not full and receipt.get("fingerprint") == fingerprint(path) and receipt.get("sha256") == expected:
        return True
    return sha256(path) == expected


def attest(path: Path, *, source: str, expected_sha256: str | None = None,
           kind: str | None = None, metadata: dict | None = None) -> dict:
    validate_file(path, kind)
    actual = sha256(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"Checksum mismatch: {path}")
    receipt = {"schema_version": 1, "source": source, "sha256": actual,
               "fingerprint": fingerprint(path), "metadata": metadata or {}}
    atomic_json(receipt_path(path), receipt)
    return receipt


def download(url: str, destination: Path, *, expected_sha256: str | None = None,
             kind: str | None = None, progress=print, attempts: int = 3,
             max_bytes: int | None = None) -> dict:
    """Reuse a verified asset; otherwise resume an incomplete transfer safely."""
    destination = Path(destination)
    with file_lock(asset_lock(destination)):
        old = read_receipt(destination)
        same_source = old.get("source") == url or bool(expected_sha256)
        if same_source and verify(destination, expected_sha256=expected_sha256):
            progress(f"Ready: {destination.name}")
            return {"status": "ready", "path": str(destination), "downloaded_bytes": 0}
        # A checksum-pinned existing model can be adopted without network access.
        if expected_sha256 and destination.is_file() and verify(destination, expected_sha256=expected_sha256, full=True):
            attest(destination, source=url, expected_sha256=expected_sha256, kind=kind)
            return {"status": "imported", "path": str(destination), "downloaded_bytes": 0}
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        state_path = partial.with_name(partial.name + ".json")
        transferred = 0
        parallel_allowed = True
        for attempt in range(attempts):
            try:
                try:
                    state = json.loads(state_path.read_text())
                except (OSError, ValueError):
                    state = {}
                offset = partial.stat().st_size if partial.exists() and state.get("url") == url else 0
                if offset and not state.get("etag") and not expected_sha256:
                    offset = 0
                headers = {"User-Agent": "MODAL-ASO/0.1", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                    if state.get("etag"):
                        headers["If-Range"] = state["etag"]
                progress(f"Downloading: {destination.name}" + (f" (resuming at {offset:,} bytes)" if offset else ""))
                try:
                    response = urlopen(Request(url, headers=headers), timeout=120)
                except HTTPError as exc:
                    if exc.code == 416 and offset:
                        partial.unlink(missing_ok=True)
                        state_path.unlink(missing_ok=True)
                    raise
                with response:
                    resumed = response.status == 206 and offset > 0
                    if resumed and not str(response.headers.get("Content-Range", "")).startswith(f"bytes {offset}-"):
                        partial.unlink(missing_ok=True)
                        state_path.unlink(missing_ok=True)
                        raise ValueError("Server returned a different download range")
                    if resumed and state.get("etag") and response.headers.get("ETag") != state["etag"]:
                        partial.unlink(missing_ok=True)
                        state_path.unlink(missing_ok=True)
                        raise ValueError("Asset changed during download; restarting")
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        remaining = int(content_length)
                        if max_bytes is not None and remaining + (offset if resumed else 0) > max_bytes:
                            raise ValueError(f"Asset size exceeds max_bytes {max_bytes}: {destination.name}")
                        if shutil.disk_usage(destination.parent).free < remaining + 16 * 1024 * 1024:
                            raise OSError(f"Insufficient free space for {destination.name} ({remaining:,} bytes remaining)")
                    atomic_json(state_path, {"url": url, "etag": response.headers.get("ETag")})
                    received = 0
                    etag = response.headers.get("ETag")
                    use_parallel = (parallel_allowed and content_length and int(content_length) >= PARALLEL_THRESHOLD
                                    and response.headers.get("Accept-Ranges") == "bytes" and etag and not etag.startswith("W/"))
                    if use_parallel:
                        base = offset if resumed else 0
                        remaining = int(content_length)
                        if shutil.disk_usage(destination.parent).free < remaining + remaining // DOWNLOAD_CONNECTIONS + 16 * 1024 * 1024:
                            raise OSError("Insufficient space for parallel transfer staging")
                        response.close()
                        if not base:
                            partial.write_bytes(b"")
                        progress(f"Using {DOWNLOAD_CONNECTIONS} download connections: {destination.name}")
                        try:
                            transferred += _parallel_transfer(url, partial, offset=base, total=base + remaining, etag=etag, headers=headers)
                        except _RangeUnavailable:
                            parallel_allowed = False
                            raise
                        received = remaining
                    else:
                        with partial.open("ab" if resumed else "wb") as stream:
                            while chunk := response.read(8 * 1024 * 1024):
                                stream.write(chunk)
                                received += len(chunk)
                                transferred += len(chunk)
                                if max_bytes is not None and received + (offset if resumed else 0) > max_bytes:
                                    raise ValueError(f"Asset size exceeds max_bytes {max_bytes}: {destination.name}")
                            stream.flush()
                            os.fsync(stream.fileno())
                    if content_length and received != int(content_length):
                        raise OSError("Download ended before Content-Length was received")
                validate_file(partial, kind or ("gzip" if destination.name.endswith(".gz") else "zip" if destination.name.endswith(".zip") else None))
                actual = sha256(partial)
                if expected_sha256 and actual != expected_sha256:
                    partial.unlink(missing_ok=True)
                    state_path.unlink(missing_ok=True)
                    raise ValueError(f"Checksum mismatch downloading {destination.name}")
                partial.replace(destination)
                atomic_json(receipt_path(destination), {"schema_version": 1, "source": url,
                            "sha256": actual, "fingerprint": fingerprint(destination), "metadata": {}})
                state_path.unlink(missing_ok=True)
                staging = partial.with_name(partial.name + ".parts")
                if staging.is_dir():
                    shutil.rmtree(staging)
                return {"status": "downloaded", "path": str(destination), "downloaded_bytes": transferred}
            except (OSError, ValueError, EOFError, HTTPException, zipfile.BadZipFile, zlib.error) as exc:
                if attempt + 1 == attempts:
                    raise RuntimeError(f"Could not install {destination.name}: {exc}") from exc
                time.sleep(min(attempt + 1, 3))
        raise AssertionError("Download attempts exhausted")


def decompress(source: Path, destination: Path, *, kind: str | None = None,
               expected_sha256: str | None = None) -> dict:
    """Decompress once, binding the result to its compressed source checksum."""
    source_identity = read_receipt(source).get("sha256") or sha256(source)
    binding = "gzip:sha256:" + source_identity
    with file_lock(asset_lock(destination)):
        if read_receipt(destination).get("source") == binding and verify(destination, expected_sha256=expected_sha256):
            return {"status": "ready", "path": str(destination)}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".extracting")
        try:
            with gzip.open(source, "rb") as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            validate_file(temporary, kind)
            if expected_sha256 and sha256(temporary) != expected_sha256:
                raise ValueError(f"Extracted asset checksum mismatch: {destination.name}")
            temporary.replace(destination)
            attest(destination, source=binding, expected_sha256=expected_sha256, kind=kind)
        finally:
            temporary.unlink(missing_ok=True)
        return {"status": "extracted", "path": str(destination)}
