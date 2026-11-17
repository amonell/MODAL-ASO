"""Finite, configurable execution budgets for generated analysis programs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess


DEFAULT_TIMEOUT_SECONDS = 300.0
TIMEOUT_ENV = 'ASO_ANALYSIS_TIMEOUT_SECONDS'


def analysis_timeout_seconds(value: float | None = None) -> float:
    """Resolve a positive budget; missing configuration uses five minutes."""
    configured = os.environ.get(TIMEOUT_ENV) if value is None else value
    if configured is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        if isinstance(configured, bool):
            raise ValueError
        seconds = float(configured)
        if not math.isfinite(seconds) or seconds <= 0 or seconds > 86400:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{TIMEOUT_ENV} must be a positive number of seconds, at most 86400.') from exc
    return seconds


def run_program(command: list[str], *, directory: Path, timeout: float,
                stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    """Reap the program and its process group before inspecting any outputs.

    File-backed streams retain diagnostics without a pipe waiting on a child
    that inherited its parent's stdout. Linux, macOS and WSL support sessions.
    """
    timeout = analysis_timeout_seconds(timeout)
    timed_out = False
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        with subprocess.Popen(command, cwd=directory, stdout=stdout, stderr=stderr,
                              start_new_session=os.name == 'posix') as process:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # This also removes background children after a parent exits.
                if os.name == 'posix':
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                process.wait()
    out = stdout_path.read_text(errors='replace')
    err = stderr_path.read_text(errors='replace')
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout, output=out, stderr=err)
    return subprocess.CompletedProcess(command, process.returncode, out, err)
