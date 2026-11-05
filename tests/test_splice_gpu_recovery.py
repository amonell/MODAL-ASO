import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aso_agents import gpu_resources as gpu
from aso_agents import splice_recovery as recovery


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ASO_SHARED_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("ASO_RESOURCE_LOCK_ROOT", str(tmp_path / "resources"))
    monkeypatch.delenv("MODAL_ASO_SPLICE_WORKER", raising=False)
    monkeypatch.setattr(gpu, "_cuda_context_failed", False)
    tiles = tmp_path / "tiles.json"
    tiles.write_text('{"rows": [{"name": "a"}, {"name": "b"}]}')
    return tiles


def test_retry_releases_gpu_ownership_preserves_inputs_and_avoids_poisoned_context(environment, monkeypatch):
    calls, retries = [], []
    def native(tiles_json_path, transcript_id, pad=10):
        calls.append(tiles_json_path)
        raise RuntimeError("splice_transformer_calc_failed: CUDA error: an illegal memory access was encountered")
    def worker(arguments, error):
        # This would fail if the parent still owned the advisory GPU lock.
        with gpu.shared_gpu_lock().open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        retries.append((dict(arguments), error))
        return {"results": [{"name": "a"}, {"name": "b"}]}
    monkeypatch.setattr(recovery, "run_in_fresh_process", worker)
    wrapped = gpu.splice_resource_limited(native)
    assert len(wrapped(str(environment), "TX", pad=12)["results"]) == 2
    assert len(wrapped(str(environment), "TX2")["results"]) == 2
    assert len(calls) == 1  # All later model work uses fresh processes.
    assert retries[0][0] == {"tiles_json_path": str(environment), "transcript_id": "TX", "pad": 12}
    assert retries[1][0]["transcript_id"] == "TX2"


@pytest.mark.parametrize("message", ["invalid reference", "CUDA out of memory", "invalid sequence coordinates"])
def test_other_errors_are_preserved_without_retry(environment, monkeypatch, message):
    def native(tiles_json_path):
        raise RuntimeError(message)
    def unexpected(*args):
        pytest.fail("Unexpected retry")
    monkeypatch.setattr(recovery, "run_in_fresh_process", unexpected)
    with pytest.raises(RuntimeError, match=message):
        gpu.splice_resource_limited(native)(str(environment))
    assert not gpu._cuda_context_failed


def test_worker_cannot_recursively_retry(environment, monkeypatch):
    monkeypatch.setenv("MODAL_ASO_SPLICE_WORKER", "1")
    def native(tiles_json_path):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        gpu.splice_resource_limited(native)(str(environment))
    assert not list(environment.parent.glob("splice-recovery-*"))


def test_worker_launch_error_is_recorded(environment, monkeypatch):
    def run(*args, **kwargs):
        raise OSError("interpreter unavailable")
    monkeypatch.setattr(recovery.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="could not start"):
        recovery.run_in_fresh_process({"tiles_json_path": str(environment)}, "original CUDA failure")
    receipt = json.loads(next(environment.parent.glob("splice-recovery-*/receipt.json")).read_text())
    assert receipt["status"] == "failed" and receipt["error"] == "interpreter unavailable"


@pytest.mark.parametrize("returncode,payload,success", [
    (0, {"results": [{"name": "a"}, {"name": "b"}]}, True),
    (1, {"error": "CUDA error: an illegal memory access was encountered"}, False),
    (0, {"error": "model unavailable"}, False),
    (0, {"results": [{"name": "a"}]}, False),
    (0, None, False),
])
def test_fresh_process_receipt_and_failed_retry(environment, monkeypatch, returncode, payload, success):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        request = json.loads(Path(command[-2]).read_text())
        assert request["transcript_id"] == "TX" and request["tiles_json_path"] == str(environment)
        assert kwargs["env"]["MODAL_ASO_SPLICE_WORKER"] == "1"
        if payload is not None:
            Path(command[-1]).write_text(json.dumps(payload))
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(recovery.subprocess, "run", run)
    arguments = {"tiles_json_path": str(environment), "transcript_id": "TX"}
    if success:
        result = recovery.run_in_fresh_process(arguments, "original CUDA failure")
        assert result["results"] == payload["results"]
        assert result["runtime_recovery"]["status"] == "recovered"
    else:
        with pytest.raises(RuntimeError, match="GPU recovery failed"):
            recovery.run_in_fresh_process(arguments, "original CUDA failure")
    assert len(calls) == 1
    receipt = json.loads(next(environment.parent.glob("splice-recovery-*/receipt.json")).read_text())
    assert receipt["status"] == ("recovered" if success else "failed")
    assert receipt["original_error"] == "original CUDA failure"
    assert receipt["attempts"] == 1
