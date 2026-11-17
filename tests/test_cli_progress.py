import io
import sys
import threading

import pytest

from aso_agents.cli_progress import TerminalProgress


class Terminal(io.StringIO):
    def __init__(self, tty=False):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


def test_periodic_updates_use_elapsed_time_and_restart_for_each_step():
    now = [100.0]
    screen = Terminal()
    progress = TerminalProgress(screen, clock=lambda: now[0])
    with progress.running("Predicting splice effects."):
        now[0] += 29
        progress._tick()
        assert screen.getvalue() == ""
        now[0] += 1
        progress._tick()
        progress._tick()
        now[0] += 30
        progress._tick()
        assert screen.getvalue().splitlines() == [
            "Still running: Predicting splice effects (30s elapsed).",
            "Still running: Predicting splice effects (1m 00s elapsed).",
        ]
        progress.start("Writing the report")
        now[0] += 30
        progress._tick()
        assert screen.getvalue().endswith("Writing the report (30s elapsed).\n")
    before = screen.getvalue()
    now[0] += 60
    progress._tick()
    assert screen.getvalue() == before
    assert progress._thread is None
    assert "\r" not in before


def test_background_updates_appear_while_the_main_thread_waits():
    updated = threading.Event()

    class ObservedTerminal(Terminal):
        def write(self, text):
            result = super().write(text)
            if "Still running:" in text:
                updated.set()
            return result

    progress = TerminalProgress(ObservedTerminal(), interval=0.02)
    with progress.running("Waiting for the language model"):
        assert updated.wait(2), "The waiting thread received no progress update"
    assert progress._thread is None


def test_warning_does_not_stop_elapsed_updates_or_overwrite_terminal_lines():
    now = [0.0]
    screen = Terminal(tty=True)
    progress = TerminalProgress(screen, animate=True, clock=lambda: now[0])
    with progress.running("Mapping transcripts"):
        for second in (1, 2, 30, 60):
            now[0] = second
            progress._tick()
        assert "Mapping transcripts (1s)" in screen.getvalue()
        assert "Mapping transcripts (2s)" in screen.getvalue()
        assert "Mapping transcripts (30s)" in screen.getvalue()
        assert "Mapping transcripts (1m 00s)" in screen.getvalue()
        assert "\n" not in screen.getvalue()
        progress.write("Reference setup: Local annotation is unavailable.\n")
        now[0] = 90
        progress._tick()
    output = screen.getvalue()
    assert "Local annotation is unavailable.\n\r" in output
    assert "Mapping transcripts (1m 30s)" in output
    assert output.count("\n") == 1
    assert "Still running:" not in output
    assert progress._width == 0


@pytest.mark.parametrize("tty", [False, True])
@pytest.mark.parametrize("ending", ["run_complete", "fatal", "exception", "interrupt", "needs_user_input"])
def test_cli_progress_covers_waits_and_stops_on_every_exit(monkeypatch, tmp_path, tty, ending):
    import run_goal_intake as runtime
    from aso_agents import agentic_runner
    from aso_agents.runtime_warnings import emit_warning

    now = [0.0]
    screen = Terminal(tty=tty)
    instances = []
    elapsed_suffix = "" if tty else " elapsed"

    def make_progress(stream, **kwargs):
        progress = TerminalProgress(stream, clock=lambda: now[0], **kwargs)
        instances.append(progress)
        return progress

    def wait_for_tick():
        now[0] += 30
        instances[0]._tick()

    path = tmp_path / "state.json"
    path.write_text("{}")

    def run(**kwargs):
        callback = kwargs["event_callback"]
        callback({"type": "model_call"})
        wait_for_tick()
        assert f"Waiting for the language model (30s{elapsed_suffix})" in screen.getvalue()
        callback({"type": "model_result"})
        callback({"type": "tool_call", "tool": "tool_run_splice_eval"})
        wait_for_tick()
        assert f"Predicting splice effects (30s{elapsed_suffix})" in screen.getvalue()
        emit_warning("Local reference warning")
        wait_for_tick()
        assert f"Predicting splice effects (1m 00s{elapsed_suffix})" in screen.getvalue()
        if tty:
            assert "Still running:" not in screen.getvalue()
        if ending == "exception":
            raise RuntimeError("Failed calculation")
        if ending == "interrupt":
            raise KeyboardInterrupt()
        callback({"type": "tool_result", "tool": "tool_run_splice_eval"})
        return {"status": ending, "run_dir": str(tmp_path), "state_path": str(path),
                "pending_questions": ["Which tissue?"], "error": "Calculation failed"}

    def input_unavailable(prompt=""):
        assert instances[0]._thread is None
        raise EOFError()

    monkeypatch.setattr(runtime, "TerminalProgress", make_progress)
    monkeypatch.setattr(agentic_runner, "run_agentic_pipeline", run)
    monkeypatch.setattr("builtins.input", input_unavailable)
    monkeypatch.setattr(sys, "stdout", screen)
    expected = {"fatal": SystemExit, "exception": RuntimeError, "interrupt": KeyboardInterrupt}
    if ending in expected:
        with pytest.raises(expected[ending]):
            runtime.main(argv=["Evaluate these ASOs", "--no-stream-ui"])
    else:
        runtime.main(argv=["Evaluate these ASOs", "--no-stream-ui"])
    assert len(instances) == 1
    assert instances[0]._thread is None
    assert "Reference setup: Local reference warning" in screen.getvalue()
    before = screen.getvalue()
    wait_for_tick()
    assert screen.getvalue() == before


def test_slow_internal_check_has_a_readable_activity():
    import run_goal_intake as runtime

    now = [0.0]
    screen = Terminal()
    narrator = runtime._CliNarrator(stream=screen, enable_stream=False, enable_spinner=False)
    narrator._progress = TerminalProgress(screen, clock=lambda: now[0])
    with narrator._progress.running("Preparing the next step"):
        narrator.on_runner_event({"type": "tool_call", "tool": "evaluate_step_output"})
        assert screen.getvalue() == ""
        now[0] = 30
        narrator._progress._tick()
    assert "Checking the run and preparing the next step" in screen.getvalue()
    assert "evaluate_step_output" not in screen.getvalue()


def test_nested_shortlist_progress_reaches_terminal_and_elapsed_updates():
    import run_goal_intake as runtime
    from aso_agents.runtime_warnings import emit_progress, warning_handler

    now = [0.0]
    screen = Terminal()
    narrator = runtime._CliNarrator(stream=screen, enable_stream=False, enable_spinner=False)
    narrator._progress = TerminalProgress(screen, clock=lambda: now[0])
    try:
        with warning_handler(narrator.on_runner_event):
            emit_progress('Shortlist review: 128 of 256 ASOs reviewed (1 of 2 batches).')
            now[0] = 30
            narrator._progress._tick()
        text = screen.getvalue()
        assert text.startswith('Shortlist review: 128 of 256 ASOs reviewed (1 of 2 batches).\n')
        assert 'Still running: Shortlist review: 128 of 256 ASOs reviewed' in text
        assert 'Reference setup' not in text
    finally:
        narrator.stop_spinner()


def test_quiet_activity_changes_and_controller_passes_cannot_postpone_updates():
    now = [0.0]
    screen = Terminal()
    progress = TerminalProgress(screen, clock=lambda: now[0])
    for _ in range(3):
        with progress.running("Preparing the next step"):
            now[0] += 5
            progress.start("Waiting for the language model")
            now[0] += 5
            progress._tick()
    assert screen.getvalue() == "Still running: Waiting for the language model (5s elapsed).\n"


def test_rapid_activity_changes_emit_overdue_updates_without_waiting_for_worker():
    now = [0.0]
    screen = Terminal()
    progress = TerminalProgress(screen, clock=lambda: now[0])
    with progress.running("Checking the run"):
        for _ in range(6):
            now[0] += 5
            progress.start("Checking the next result")
    assert screen.getvalue() == "Still running: Checking the next result (0s elapsed).\n"


@pytest.mark.parametrize("tty", [False, True])
@pytest.mark.parametrize("raw_cli", [False, True])
def test_launcher_loading_also_gets_elapsed_updates(monkeypatch, tty, raw_cli):
    import builtins
    from types import SimpleNamespace
    from aso_agents import cli, cli_progress

    now = [0.0]
    screen = Terminal(tty=tty)
    instances = []
    inline = tty and not raw_cli
    original_import = builtins.__import__

    def make_progress(stream, **kwargs):
        progress = TerminalProgress(stream, clock=lambda: now[0], **kwargs)
        instances.append(progress)
        return progress

    def delayed_import(name, *args, **kwargs):
        if name == "run_goal_intake":
            now[0] += 30
            instances[0]._tick()
            elapsed_suffix = "" if inline else " elapsed"
            assert f"Loading MODAL-ASO (30s{elapsed_suffix})" in screen.getvalue()
            if inline:
                assert "Still running:" not in screen.getvalue()
            else:
                assert "\r" not in screen.getvalue()
            return SimpleNamespace(main=lambda **kwargs: kwargs["result_callback"]({"status": "run_complete"}))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", delayed_import)
    monkeypatch.setattr(cli_progress, "TerminalProgress", make_progress)
    monkeypatch.setattr(sys, "stdout", screen)
    argv = ["Evaluate these ASOs"] + (["--raw-cli"] if raw_cli else [])
    assert cli._run(argv) == 0
    assert len(instances) == 1
    assert instances[0]._thread is None
