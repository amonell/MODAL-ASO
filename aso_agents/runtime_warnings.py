"""Deliver actionable runtime warnings even while the CLI captures tool logs."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator


_handler: ContextVar[Callable[[dict[str, Any]], Any] | None] = ContextVar(
    "modal_aso_warning_handler", default=None,
)


@contextmanager
def warning_handler(handler: Callable[[dict[str, Any]], Any]) -> Iterator[None]:
    token = _handler.set(handler)
    try:
        yield
    finally:
        _handler.reset(token)


def emit_warning(message: str, *, category: str = "Reference setup") -> None:
    handler = _handler.get()
    if handler is None:
        print(f"{category}: {message}", flush=True)
    else:
        handler({"type": "warning", "message": message, "category": category})


def request_model_installation(model: str, root: str, detail: str) -> bool:
    """Only an explicit CLI response may authorize an optional download."""
    handler = _handler.get()
    if handler is None:
        emit_warning(
            f"{detail} Continuing without {model} efficacy predictions.",
            category="Model setup",
        )
        return False
    return handler({"type": "model_installation", "model": model,
                    "root": root, "message": detail}) is True


def emit_progress(message: str) -> None:
    """Use the CLI's existing run-scoped event channel from nested tools."""
    handler = _handler.get()
    if handler is None:
        print(message, flush=True)
    else:
        handler({"type": "progress", "message": message})
