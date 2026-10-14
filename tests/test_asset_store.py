"""Installation lifecycle checks against a real local HTTP server."""
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

from aso_agents import asset_store as store


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_ASO_CACHE_DIR", str(tmp_path / "cache"))
    payload = b"verified-asset\n" * 10000
    state = {"requests": [], "interrupt": False, "html": False, "ignore_ranges": False}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(dict(self.headers))
            content = b"<html>download failed</html>" if state["html"] else payload
            requested = self.headers.get("Range") if not state["ignore_ranges"] else None
            first, last = (requested or "bytes=0-").split("=")[1].split("-")
            offset, end = int(first), int(last) if last else len(content)-1
            self.send_response(206 if requested else 200)
            self.send_header("Content-Length", str(end-offset+1))
            self.send_header("ETag", '"fixture-one"')
            self.send_header("Accept-Ranges", "bytes")
            if requested:
                self.send_header("Content-Range", f"bytes {offset}-{end}/{len(content)}")
            self.end_headers()
            try:
                if state["interrupt"] or (requested and state.pop("interrupt_ranges", False)):
                    self.wfile.write(content[offset:offset+1000])
                    state["interrupt"] = False
                    self.close_connection = True
                else:
                    self.wfile.write(content[offset:end+1])
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/asset", payload, state
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_verified_warm_asset_needs_no_network(server, tmp_path, monkeypatch):
    url, payload, state = server
    path = tmp_path / "asset"
    expected = hashlib.sha256(payload).hexdigest()
    assert store.download(url, path, expected_sha256=expected)["status"] == "downloaded"
    monkeypatch.setattr(store, "urlopen", lambda *a, **k: pytest.fail("warm setup contacted the network"))
    assert store.download(url, path, expected_sha256=expected)["downloaded_bytes"] == 0
    assert path.read_bytes() == payload
    assert len(state["requests"]) == 1


def test_interrupted_download_resumes_and_is_never_exposed(server, tmp_path):
    url, payload, state = server
    state["interrupt"] = True
    path = tmp_path / "asset"
    with pytest.raises(RuntimeError):
        store.download(url, path, attempts=1)
    assert not path.exists()
    assert path.with_name("asset.partial").is_file()
    store.download(url, path, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert path.read_bytes() == payload
    assert state["requests"][-1].get("Range") == "bytes=1000-"


def test_corrupt_download_does_not_replace_valid_asset(server, tmp_path):
    url, payload, state = server
    path = tmp_path / "asset"
    path.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        store.download(url, path, expected_sha256="0" * 64, attempts=1)
    assert path.read_bytes() == b"original"
    assert not store.verify(path, expected_sha256="0" * 64)


def test_html_error_is_rejected(server, tmp_path):
    url, _, state = server
    state["html"] = True
    with pytest.raises(RuntimeError, match="HTML"):
        store.download(url, tmp_path / "model", attempts=1)
    assert not (tmp_path / "model").exists()


def test_simultaneous_setups_download_once(server, tmp_path):
    url, payload, state = server
    path = tmp_path / "shared" / "asset"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.download(url, path), range(4)))
    assert sum(r["status"] == "downloaded" for r in results) == 1
    assert len(state["requests"]) == 1
    assert path.read_bytes() == payload


def test_changed_file_invalidates_receipt_and_is_repaired(server, tmp_path):
    url, payload, _ = server
    path = tmp_path / "asset"
    store.download(url, path)
    path.write_bytes(b"broken")
    assert not store.verify(path)
    assert store.download(url, path)["status"] == "downloaded"
    assert store.verify(path, full=True)


def test_decompression_reuses_output_and_detects_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_ASO_CACHE_DIR", str(tmp_path / "cache"))
    source, target = tmp_path / "sequence.gz", tmp_path / "sequence.fa"
    source.write_bytes(gzip.compress(b">chr1\nACGT\n"))
    store.attest(source, source="fixture", kind="gzip")
    assert store.decompress(source, target, kind="fasta")["status"] == "extracted"
    before = target.stat().st_mtime_ns
    assert store.decompress(source, target, kind="fasta")["status"] == "ready"
    assert target.stat().st_mtime_ns == before
    target.write_text("corrupted")
    assert store.decompress(source, target, kind="fasta")["status"] == "extracted"
    assert target.read_text() == ">chr1\nACGT\n"


def test_wrong_extracted_checksum_preserves_previous_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_ASO_CACHE_DIR", str(tmp_path / "cache"))
    source, target = tmp_path / "sequence.gz", tmp_path / "sequence.fa"
    source.write_bytes(gzip.compress(b">chr1\nACGT\n"))
    target.write_bytes(b"previous complete file")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.decompress(source, target, kind="fasta", expected_sha256="0" * 64)
    assert target.read_bytes() == b"previous complete file"


def test_download_size_limit_keeps_incomplete_files_separate(server, tmp_path):
    url, _, _ = server
    path = tmp_path / "large-asset"
    with pytest.raises(RuntimeError, match="max_bytes"):
        store.download(url, path, max_bytes=100, attempts=1)
    assert not path.exists()


@pytest.mark.parametrize("offset", [0, 512, 4096])
def test_hdf5_user_blocks_are_supported(tmp_path, offset):
    path = tmp_path / "dataset.h5ad"
    path.write_bytes(b" " * offset + b"\x89HDF\r\n\x1a\n" + b"placeholder")
    store.validate_file(path, "hdf5")


def test_parallel_ranges_preserve_exact_file_and_clean_staging(server, tmp_path, monkeypatch):
    url, payload, state = server
    monkeypatch.setattr(store, "PARALLEL_THRESHOLD", 1024)
    path = tmp_path / "large-asset"
    result = store.download(url, path, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert result["downloaded_bytes"] == len(payload)
    assert path.read_bytes() == payload
    assert len([r for r in state["requests"] if "Range" in r]) == 4
    assert not path.with_name(path.name + ".partial.parts").exists()


def test_parallel_ranges_can_finish_an_existing_sequential_partial(server, tmp_path, monkeypatch):
    url, payload, state = server
    path = tmp_path / "large-asset"
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload[:4096])
    store.atomic_json(partial.with_name(partial.name + ".json"), {"url": url, "etag": '"fixture-one"'})
    monkeypatch.setattr(store, "PARALLEL_THRESHOLD", 1024)
    result = store.download(url, path, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert path.read_bytes() == payload
    assert result["downloaded_bytes"] == len(payload)-4096
    assert state["requests"][0]["Range"] == "bytes=4096-"


def test_server_ignoring_ranges_falls_back_to_a_single_transfer(server, tmp_path, monkeypatch):
    url, payload, state = server
    state["ignore_ranges"] = True
    monkeypatch.setattr(store, "PARALLEL_THRESHOLD", 1024)
    path = tmp_path / "large-asset"
    store.download(url, path, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert path.read_bytes() == payload
    assert not path.with_name(path.name + ".partial.parts").exists()


def test_interrupted_parallel_transfer_reuses_completed_ranges(server, tmp_path, monkeypatch):
    url, payload, state = server
    state["interrupt_ranges"] = True
    monkeypatch.setattr(store, "PARALLEL_THRESHOLD", 1024)
    path = tmp_path / "large-asset"
    with pytest.raises(RuntimeError):
        store.download(url, path, expected_sha256=hashlib.sha256(payload).hexdigest(), attempts=1)
    assert not path.exists()
    staging = path.with_name(path.name + ".partial.parts")
    reused = sum(p.stat().st_size for p in staging.iterdir() if p.name != "transfer.json")
    assert reused > 0
    result = store.download(url, path, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert result["downloaded_bytes"] == len(payload)-reused
    assert path.read_bytes() == payload


def test_verified_mirror_resumes_the_same_partial_file(server, tmp_path):
    mirror, payload, state = server
    primary = "http://127.0.0.1:1/unavailable"
    path = tmp_path / "mirrored-asset"
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload[:4096])
    store.atomic_json(partial.with_name(partial.name + ".json"), {"url": primary, "etag": '"fixture-one"'})
    result = store.download(primary, path, expected_sha256=hashlib.sha256(payload).hexdigest(), mirror_urls=(mirror,))
    assert path.read_bytes() == payload
    assert result["downloaded_bytes"] == len(payload)-4096
    assert state["requests"][0]["Range"] == "bytes=4096-"
    assert store.read_receipt(path)["source"] == primary
    assert store.read_receipt(path)["metadata"]["download_url"] == mirror


def test_unpinned_mirror_is_rejected_before_network(server, tmp_path):
    url, _, state = server
    with pytest.raises(ValueError, match="pinned SHA256"):
        store.download(url, tmp_path/'asset', mirror_urls=(url,))
    assert state["requests"] == []
