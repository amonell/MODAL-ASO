"""Serialize local model inference through one shared GPU ownership lock."""

from functools import wraps
import fcntl
import inspect
import os

from .paths import shared_gpu_lock

_cuda_context_failed = False


def _cuda_execution_error(error):
    message = str(error).lower()
    return "cuda" in message and any(term in message for term in (
        "illegal memory access", "device-side assert", "misaligned address",
        "unspecified launch failure", "illegal instruction"))


def splice_resource_limited(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        global _cuda_context_failed
        from .resource_limits import acquire_resource
        from .splice_recovery import run_in_fresh_process
        isolated = os.environ.get("MODAL_ASO_SPLICE_WORKER") == "1"
        if _cuda_context_failed and not isolated:
            bound = inspect.signature(func).bind(*args, **kwargs)
            return run_in_fresh_process(bound.arguments, "Earlier CUDA execution failure in this controller process.")
        try:
            with acquire_resource("gpu"):
                path = shared_gpu_lock()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a+") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    return func(*args, **kwargs)
        except RuntimeError as exc:
            if isolated or not _cuda_execution_error(exc):
                raise
            _cuda_context_failed = True
            # CUDA execution faults poison the current context. Release both
            # ownership locks before a new interpreter acquires them. The worker
            # gets one attempt and cannot recursively retry or skip input rows.
            bound = inspect.signature(func).bind(*args, **kwargs)
            return run_in_fresh_process(bound.arguments, str(exc))
    return wrapped
