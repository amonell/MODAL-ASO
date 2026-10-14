"""Serialize local model inference through one shared GPU ownership lock."""

from functools import wraps
import fcntl

from .paths import shared_gpu_lock


def splice_resource_limited(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        from .resource_limits import acquire_resource
        with acquire_resource("gpu"):
            path = shared_gpu_lock()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                return func(*args, **kwargs)
    return wrapped
