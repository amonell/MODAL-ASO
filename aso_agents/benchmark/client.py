"""Small neutral client mounted into both sandboxes; no oracle or ledger access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
import fcntl
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def rpc(path, body=None):
    base = os.environ["ASO_BENCHMARK_SERVICE_URL"].rstrip("/")
    req = Request(base + path, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + os.environ["BENCHMARK_SERVICE_TOKEN"], "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urlopen(req, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code < 500 or attempt == 3:
                raise
        except (URLError, TimeoutError, ConnectionError, HTTPException):
            if attempt == 3:
                raise
        time.sleep(min(2 ** attempt, 5))


def _synchronous_scope():
    frames = sys._current_frames()
    for thread in threading.enumerate():
        if thread is threading.current_thread():
            continue
        frame = frames.get(thread.ident)
        if frame is None:
            return False
        module, name = frame.f_globals.get("__name__"), frame.f_code.co_name
        # Idle executor threads remain alive between tasks. Their top frame is
        # _worker at queue.get; active work has a different top frame.
        if module == "concurrent.futures.thread" and name == "_worker":
            continue
        if module == "threading" and name in {"wait", "join", "_wait_for_tstate_lock"}:
            continue
        if module == "queue" and name == "get":
            continue
        return False  # Unknown/runnable/async IO work is not certified exclusive.
    return True


def _write_marker(marker, owner, value=None):
    with marker.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if value is None:
            try:
                if json.loads(marker.read_text()).get("owner") == owner:
                    marker.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
        else:
            temporary = marker.with_suffix("." + owner + ".pending")
            temporary.write_text(json.dumps(value))
            temporary.replace(marker)


def submit_and_wait(kind, payload):
    marker = Path(os.environ["BENCHMARK_CLOCK_FILE"]) if os.environ.get("BENCHMARK_CLOCK_FILE") else None
    owner = uuid4().hex
    started = time.time()

    def attest(job):
        if marker:
            _write_marker(marker, owner, {"started": started, "kind": kind, "job_id": job["job_id"],
                "exclusive": _synchronous_scope(), "pid": os.getpid(), "tid": threading.get_native_id(), "owner": owner})

    if marker:
        # The supervisor independently checks process CPU activity. A call from
        # a background/parallel tool is conservatively not an exclusive wait.
        _write_marker(marker, owner, {"started": started, "kind": kind, "exclusive": False, "pid": os.getpid(), "owner": owner})
    try:
        return _submit_and_wait(kind, payload, attest=attest)
    finally:
        if marker:
            _write_marker(marker, owner)


def _submit_and_wait(kind, payload, *, attest=lambda _: None):
    # Persist before POST. A lost response/process restart reuses the logical key.
    folder = Path(os.environ.get("BENCHMARK_CLIENT_STATE", "/output/.service_requests"))
    folder.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps([kind, payload], sort_keys=True).encode()).hexdigest()
    pending = folder / (identity + ".pending.json")
    with (folder / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if pending.exists():
            request = json.loads(pending.read_text())
        else:
            request = {"kind": kind, "payload": payload, "key": uuid4().hex}
            temporary = folder / (identity + ".tmp")
            with temporary.open("w") as out:
                json.dump(request, out)
                out.flush()
                os.fsync(out.fileno())
            temporary.replace(pending)
    receipt = rpc("/jobs", request)
    attest(receipt)
    while True:
        job = rpc("/jobs/" + receipt["job_id"])
        attest(job)
        if job["state"] == "finished":
            with (folder / ".lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if pending.exists() and json.loads(pending.read_text())["key"] == request["key"]:
                    pending.replace(folder / (request["key"] + ".completed.json"))
            return job["result"]
        if job["state"] == "failed":
            with (folder / ".lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if pending.exists() and json.loads(pending.read_text())["key"] == request["key"]:
                    pending.replace(folder / (request["key"] + ".failed.json"))
            raise RuntimeError(str(job["result"].get("failure_kind", "infrastructure_error")) + ": " + str(job["result"]))
        # Per-execution supervisor determines timeout from attested queue intervals.
        time.sleep(0.25)


def ask_user(question: str) -> str:
    """Ask the user a necessary clarification; returns only their grounded answer."""
    response = submit_and_wait("simulated_user", {"questions": [question]})
    return response["answers"][0]["answer"]


def splice_predict(chrom: str, strand: str, snp_pos: int, ref: str, alt: str,
                   center_posL: int, center_posR: int, pre_mutate_pos: int | None = None,
                   pre_mutate_alt: str | None = None) -> dict:
    """SpliceTransformer reference/alternative probabilities, not experimental ASO efficacy.

    Coordinates are 1-based genomic. The ROI is inclusive. Sequences ref/alt are
    genomic forward orientation even for a '-' RNA. The returned matrices follow
    RNA 5'->3' orientation; columns 0/1/2 are neither/acceptor/donor. A deletion or
    replacement is a computational perturbation surrogate for ASO occupancy.
    Human GRCh38 is the fixed supported reference. Optional pre-mutation context
    is applied identically in reference and alternative predictions.
    """
    return submit_and_wait("gpu", {"operation": "snp_misaligned", "reference": "GRCh38",
        "snp": asdict(SNPIntervalMutated(chrom, strand, snp_pos, ref, alt, center_posL, center_posR,
                                        pre_mutate_pos=pre_mutate_pos, pre_mutate_alt=pre_mutate_alt)), "context_len": 4000})


@dataclass
class SNPIntervalMutated:
    chrom: str
    strand: str
    snp_pos: int
    ref: str
    alt: str
    center_posL: int
    center_posR: int
    center_pos: int | None = None
    pre_mutate_pos: int | None = None
    pre_mutate_alt: str | None = None


class RemoteSpliceDriver:
    def __init__(self, ref_fasta, load_db=False):
        if ref_fasta != "hg38":
            raise ValueError("shared SpliceTransformer service requires pinned GRCh38 reference")
        self.ref_fasta = None  # Native caller installs its read-only FASTA handle.

    def calc_snp_misaligned(self, snp, context_len):
        import numpy as np
        result = submit_and_wait("gpu", {"operation": "snp_misaligned", "reference": "GRCh38",
                                         "snp": asdict(snp), "context_len": context_len})
        return np.asarray(result["reference_scores"]), np.asarray(result["alternative_scores"])


def splice_resource_limited(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if os.environ.get("ASO_BENCHMARK_SERVICE_URL"):
            return func(*args, **kwargs)
        from aso_agents.resource_limits import acquire_resource
        with acquire_resource("gpu"):
            return func(*args, **kwargs)
    return wrapped
