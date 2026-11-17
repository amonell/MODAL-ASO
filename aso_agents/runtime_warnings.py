"""Deliver actionable runtime warnings even while the CLI captures tool logs."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator


_handler: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "modal_aso_warning_handler", default=None,
)


@contextmanager
def warning_handler(handler: Callable[[dict[str, Any]], None]) -> Iterator[None]:
    token = _handler.set(handler)
    try:
        yield
    finally:
        _handler.reset(token)


def emit_warning(message: str) -> None:
    handler = _handler.get()
    if handler is None:
        print(f"Reference setup: {message}", flush=True)
    else:
        handler({"type": "warning", "message": message})


def emit_progress(message: str) -> None:
    """Use the CLI's existing run-scoped event channel from nested tools."""
    handler = _handler.get()
    if handler is None:
        print(message, flush=True)
    else:
        handler({"type": "progress", "message": message})
