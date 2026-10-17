"""Utilities for explicit synchronous-to-asynchronous call boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


def run_async_from_sync(operation: Callable[[], Awaitable[Any]]) -> Any:
    """Run an async operation once, without swallowing its ``RuntimeError``.

    The event-loop probe is intentionally isolated from the operation.  Catching
    ``RuntimeError`` around ``asyncio.run`` would also catch provider failures
    raised by the coroutine and could execute an expensive model call twice.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation())
    raise RuntimeError("synchronous boundary called from an active event loop")
