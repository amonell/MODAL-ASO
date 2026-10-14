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
    state = {"requests": [], "interrupt": False, "html": False}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(dict(self.headers))
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            content = b"<html>download failed</html>" if state["html"] else payload
            self.send_response(206 if offset else 200)
            self.send_header("Content-Length", str(len(content) - offset))
            self.send_header("ETag", '"fixture-one"')
            if offset:
                self.send_header("Content-Range", f"bytes {offset}-{len(content)-1}/{len(content)}")
            self.end_headers()
            if state["interrupt"]:
                self.wfile.write(content[offset:offset+1000])
                state["interrupt"] = False
                self.close_connection = True
            else:
                self.wfile.write(content[offset:])
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
