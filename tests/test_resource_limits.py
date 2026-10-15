from __future__ import annotations

import errno
import fcntl
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from aso_agents.resource_limits import (
    _try_exclusive_lock,
    acquire_resource,
    resource_limited,
    scoped_resource_name,
)


@pytest.fixture(params=[errno.EAGAIN, errno.EACCES])
def contention_errno(monkeypatch, request):
    """Exercise local and network lock contention with actual held locks."""
    original = fcntl.flock

    def flock(fd, operation):
        try:
            return original(fd, operation)
        except OSError as exc:
            if operation & fcntl.LOCK_NB and exc.errno in {errno.EAGAIN, errno.EACCES}:
                raise OSError(request.param, "resource is locked") from exc
            raise

    monkeypatch.setattr(fcntl, "flock", flock)


def test_resource_limit_serializes_callers(monkeypatch, tmp_path, contention_errno) -> None:
    monkeypatch.setenv("ASO_RESOURCE_LOCK_ROOT", str(tmp_path))
    monkeypatch.setenv("ASO_RESOURCE_LIMIT_GPU", "1")
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()

    @resource_limited("gpu")
    def work() -> None:
        with state_lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.05)
        with state_lock:
            state["active"] -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: work(), range(4)))

    assert state["maximum"] == 1
    assert work._aso_resource_requirements == ("gpu",)


def test_resource_limit_allows_configured_capacity(monkeypatch, tmp_path, contention_errno) -> None:
    monkeypatch.setenv("ASO_RESOURCE_LOCK_ROOT", str(tmp_path))
    monkeypatch.setenv("ASO_RESOURCE_LIMIT_GPU", "2")
    barrier = threading.Barrier(2)

    @resource_limited("gpu")
    def work() -> None:
        barrier.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: work(), range(2)))


def test_resource_limit_admits_waiters_in_arrival_order(monkeypatch, tmp_path, contention_errno) -> None:
    monkeypatch.setenv("ASO_RESOURCE_LOCK_ROOT", str(tmp_path))
    monkeypatch.setenv("ASO_RESOURCE_LIMIT_LLM_PROVIDER", "1")
    entered: list[str] = []

    def wait_for_ticket_count(expected: int) -> None:
        deadline = time.monotonic() + 2
        waiter_root = tmp_path / "llm_provider.waiters"
        while time.monotonic() < deadline:
            if len(list(waiter_root.glob("*.waiter"))) == expected:
                return
            time.sleep(0.01)
        raise AssertionError(f"did not observe {expected} queued resource waiters")

    def enter(label: str) -> None:
        with acquire_resource("llm_provider", poll_seconds=0.01):
            entered.append(label)
            time.sleep(0.02)

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        with acquire_resource("llm_provider"):
            first = executor.submit(enter, "first")
            wait_for_ticket_count(1)
            second = executor.submit(enter, "second")
            wait_for_ticket_count(2)
        first.result(timeout=2)
        second.result(timeout=2)
    finally:
        executor.shutdown(wait=True)
    assert entered == ["first", "second"]
    assert not list((tmp_path / "llm_provider.waiters").glob("*.waiter"))


@pytest.mark.parametrize("error_number", [errno.EPERM, errno.EBADF, errno.ENOLCK])
def test_noncontention_lock_errors_propagate(monkeypatch, tmp_path, error_number) -> None:
    def broken_lock(fd, operation):
        raise OSError(error_number, "lock unavailable")

    with (tmp_path / "probe.lock").open("a+") as handle:
        monkeypatch.setattr(fcntl, "flock", broken_lock)
        with pytest.raises(OSError) as error:
            _try_exclusive_lock(handle)
    assert error.value.errno == error_number


def test_resource_limit_rejects_invalid_capacity(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ASO_RESOURCE_LOCK_ROOT", str(tmp_path))
    monkeypatch.setenv("ASO_RESOURCE_LIMIT_GPU", "0")

    with pytest.raises(ValueError, match="positive integer"):
        with acquire_resource("gpu"):
            pass


def test_scoped_resource_name_is_safe_and_data_driven() -> None:
    assert scoped_resource_name("llm", "Provider / FP8") == (
        "llm_provider___fp8"
    )
