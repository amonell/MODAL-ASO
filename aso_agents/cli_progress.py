"""Elapsed-time updates for terminal users, including redirected output."""

from contextlib import contextmanager
import threading
import time
from typing import Any, Callable, Iterator


def elapsed_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class TerminalProgress:
    """Print the active operation's elapsed time without estimating completion."""

    def __init__(
        self, stream: Any, *, animate: bool = False, interval: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stream = stream
        self.animate = bool(animate and getattr(stream, "isatty", lambda: False)())
        self.interval = interval
        self.clock = clock
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stop.set()
        self._width = 0
        self._frame = 0
        self._message = ""
        self._started = self._last_update = self.clock()

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:
            pass

    def _clear(self) -> None:
        if self._width:
            self._write("\r" + " " * self._width + "\r")
            self._width = 0

    def write(self, text: str) -> None:
        with self._lock:
            self._clear()
            self._write(text)
            if text.strip():
                self._last_update = self.clock()

    def _tick(self) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            now = self.clock()
            elapsed = elapsed_text(now - self._started)
            if self.animate:
                frame = ("|", "/", "-", "\\")[self._frame % 4]
                label = self._message if len(self._message) <= 72 else self._message[:69] + "..."
                line = f"  {frame} {label} ({elapsed})"
                self._write("\r" + line.ljust(self._width))
                self._width = len(line)
                self._frame += 1
            elif now - self._last_update >= self.interval:
                self._clear()
                self._write(f"Still running: {self._message} ({elapsed} elapsed).\n")
                self._last_update = now

    def start(self, message: str) -> None:
        self.stop()
        self._message = message.strip().rstrip(".")
        self._started = self.clock()
        stop = threading.Event()
        self._stop = stop
        # Rapid activity changes must not keep postponing the worker's first tick.
        self._tick()

        def monitor() -> None:
            while not stop.wait(min(0.5, self.interval)):
                self._tick()

        self._thread = threading.Thread(target=monitor, name="modal-aso-progress", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._clear()
            self._thread = None

    @contextmanager
    def running(self, message: str) -> Iterator[None]:
        self.start(message)
        try:
            yield
        finally:
            self.stop()
