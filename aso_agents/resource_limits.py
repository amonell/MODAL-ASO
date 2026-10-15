"""Cross-process capacity limits for resource-intensive pipeline operations."""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, IO, Iterator, TypeVar, cast


_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def _resource_name(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in normalized
    ):
        raise ValueError(f"invalid resource name: {value!r}")
    return normalized


def scoped_resource_name(resource: str, scope: str) -> str:
    """Return a safe resource name for a dynamically selected scope."""

    normalized_resource = _resource_name(resource)
    normalized_scope = "".join(
        character if character.isalnum() else "_"
        for character in str(scope or "").strip().lower()
    ).strip("_")
    if not normalized_scope:
        raise ValueError(
            "resource scope must contain at least one alphanumeric character"
        )
    return _resource_name(f"{normalized_resource}_{normalized_scope}")


def _resource_limit(resource: str) -> int:
    variable = f"ASO_RESOURCE_LIMIT_{resource.upper()}"
    raw = str(os.environ.get(variable) or "1").strip()
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError(f"{variable} must be a positive integer") from exc
    if limit < 1:
        raise ValueError(f"{variable} must be a positive integer")
    return limit


def _lock_root() -> Path:
    configured = str(os.environ.get("ASO_RESOURCE_LOCK_ROOT") or "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(tempfile.gettempdir()) / f"modal-aso-resource-locks-{os.getuid()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _release_lock(handle: IO[str] | None) -> None:
    if handle is None:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _remove_waiter(path: Path, handle: IO[str] | None) -> None:
    if handle is None:
        return
    try:
        path.unlink(missing_ok=True)
    finally:
        _release_lock(handle)


def _try_exclusive_lock(handle: IO[str]) -> bool:
    """Probe a lock without assuming one filesystem's contention errno.

    Network filesystems can report EACCES for a conflicting nonblocking
    lock where a local filesystem reports EAGAIN. Only these lock-operation
    errors mean contention; file-open permissions and other failures propagate.
    """

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise
    return True


def _create_waiter(root: Path, resource: str) -> tuple[Path, IO[str]]:
    """Create a live, process-owned FIFO ticket for one resource request."""

    waiter_root = root / f"{resource}.waiters"
    waiter_root.mkdir(exist_ok=True)
    ticket_name = (
        f"{time.monotonic_ns():020d}-{os.getpid():010d}-{uuid.uuid4().hex}.waiter"
    )
    pending_path = waiter_root / f".{ticket_name}.pending"
    ticket_path = waiter_root / ticket_name
    handle = pending_path.open("x+")
    try:
        # A shared lock marks the ticket as live. If the process exits while
        # waiting, the kernel releases it and another caller can prune it.
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        pending_path.replace(ticket_path)
    except Exception:
        pending_path.unlink(missing_ok=True)
        _release_lock(handle)
        raise
    return ticket_path, handle


def _ordered_live_waiters(waiter_root: Path) -> list[Path]:
    """Return live tickets in arrival order and discard abandoned tickets."""

    live: list[Path] = []
    for path in sorted(waiter_root.glob("*.waiter")):
        try:
            probe = path.open("r+")
        except FileNotFoundError:
            continue
        with probe:
            if not _try_exclusive_lock(probe):
                live.append(path)
                continue
            try:
                path.unlink(missing_ok=True)
            finally:
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
    return live


def _record_acquisition(acquisition: dict[str, Any]) -> None:
    """Forward admission timing when an evaluation trace is active."""

    try:
        from aso_agents.telemetry import record_resource_acquisition
    except (AttributeError, ImportError):
        return
    record_resource_acquisition(acquisition)


def resource_cooldown_seconds(resource: str) -> float:
    """Read the shared provider cooldown without reserving a request slot."""

    path = _lock_root() / f"{_resource_name(resource)}.cooldown"
    if not path.exists():
        return 0.0
    with path.open("r") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        content = handle.read()
        # Another process can create the file just before obtaining its first
        # exclusive lock. An as-yet-unpublished cooldown is empty.
        until = float(json.loads(content)["until"]) if content else 0.0
    return max(0.0, until - time.time())


def defer_resource(resource: str, seconds: float) -> None:
    """Extend a cooldown atomically; concurrent workers cannot shorten it."""

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("resource cooldown must be finite and nonnegative")
    path = _lock_root() / f"{_resource_name(resource)}.cooldown"
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        previous = handle.read()
        until = float(json.loads(previous)["until"]) if previous else 0.0
        handle.seek(0)
        handle.truncate()
        json.dump({"until": max(until, time.time() + seconds)}, handle)
        handle.flush()


@contextmanager
def acquire_resource(
    resource: str,
    *,
    poll_seconds: float = 0.1,
) -> Iterator[dict[str, Any]]:
    """Acquire one slot for a named host resource using advisory file locks."""

    normalized = _resource_name(resource)
    limit = _resource_limit(normalized)
    root = _lock_root()
    started = time.monotonic()
    handle: IO[str] | None = None
    slot = None
    ticket_path, ticket_handle = _create_waiter(root, normalized)
    try:
        while handle is None:
            cooldown = resource_cooldown_seconds(normalized)
            if cooldown > 0:
                time.sleep(min(1.0, cooldown))
                continue
            waiters = _ordered_live_waiters(ticket_path.parent)
            try:
                position = waiters.index(ticket_path)
            except ValueError:
                raise RuntimeError(
                    f"live resource waiter disappeared: {ticket_path}"
                ) from None
            if position < limit:
                for candidate in range(limit):
                    candidate_handle = (
                        root / f"{normalized}.{candidate}.lock"
                    ).open("a+")
                    try:
                        acquired = _try_exclusive_lock(candidate_handle)
                    except BaseException:
                        candidate_handle.close()
                        raise
                    if not acquired:
                        candidate_handle.close()
                        continue
                    handle = candidate_handle
                    slot = candidate
                    break
            # A failed in-flight request may publish a cooldown while this
            # waiter is acquiring a slot. Return the slot before waiting.
            if handle is not None and resource_cooldown_seconds(normalized) > 0:
                _release_lock(handle)
                handle = None
                slot = None
            if handle is None:
                time.sleep(max(0.01, float(poll_seconds)))
        _remove_waiter(ticket_path, ticket_handle)
        ticket_handle = None
        acquisition = {
            "resource": normalized,
            "slot": slot,
            "limit": limit,
            "wait_seconds": round(time.monotonic() - started, 3),
        }
        _record_acquisition(acquisition)
        yield acquisition
    finally:
        _remove_waiter(ticket_path, ticket_handle)
        _release_lock(handle)


@contextmanager
def acquire_resources(
    resources: tuple[str, ...],
) -> Iterator[list[dict[str, Any]]]:
    """Acquire multiple named resources in stable order to avoid lock inversion."""

    normalized = tuple(sorted({_resource_name(item) for item in resources}))
    acquisitions: list[dict[str, Any]] = []
    with ExitStack() as stack:
        for resource in normalized:
            acquisitions.append(stack.enter_context(acquire_resource(resource)))
        yield acquisitions


def resource_limited(*resources: str) -> Callable[[_Callable], _Callable]:
    """Declare and enforce host-resource requirements for a callable."""

    declared = tuple(sorted({_resource_name(item) for item in resources}))
    if not declared:
        raise ValueError("at least one resource must be declared")

    def decorator(func: _Callable) -> _Callable:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with acquire_resources(declared):
                return func(*args, **kwargs)

        setattr(wrapped, "_aso_resource_requirements", declared)
        return cast(_Callable, wrapped)

    return decorator


__all__ = [
    "acquire_resource",
    "acquire_resources",
    "defer_resource",
    "resource_cooldown_seconds",
    "resource_limited",
    "scoped_resource_name",
]
