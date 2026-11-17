"""Missing efficacy models are an explicit choice before a CLI run finishes."""
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

import run_goal_intake as runtime
from aso_agents import agentic_runner, oligoai, oligoai_readiness as ready
from aso_agents.response_artifacts import digest, write_json
from aso_agents.runtime_warnings import warning_handler
from test_cli_progress import Terminal
from test_oligoai import reference, row


@pytest.fixture
def missing_model(tmp_path, monkeypatch):
    root = tmp_path / "model with spaces"
    checkpoint = b"test checkpoint"
    monkeypatch.setenv("ASO_OLIGOAI_ROOT", str(root))
    monkeypatch.setattr(oligoai, "MODEL_SHA256", hashlib.sha256(checkpoint).hexdigest())
    calls = []

    def install(location, log_path):
        assert location == root
        calls.append("install")
        (root / "source").mkdir(parents=True)
        (root / "venv/bin").mkdir(parents=True)
        (root / "venv/bin/python").symlink_to(sys.executable)
        (root / oligoai.MODEL_FILE).write_bytes(checkpoint)
        log_path.write_text("Installed test runtime\n")

    def infer(command, **kwargs):
        assert "aso_agents.oligoai_runner" in command
        calls.append("predict")
        inputs = json.loads(Path(command[command.index("--input") + 1]).read_text())["inputs"]
        for item in inputs:
            oligoai.validate_model_input(item)
        write_json(command[command.index("--output") + 1], {
            "model_sha256": oligoai.MODEL_SHA256, "source_revision": oligoai.SOURCE_REVISION,
            "input_digest": digest(inputs),
            "runner_sha256": oligoai.file_sha(Path(oligoai.__file__).with_name("oligoai_runner.py")),
            "predictions": [{"input_id": item["input_id"], "predicted_inhibition_percent": 73.125}
                            for item in inputs],
        })

    monkeypatch.setattr(ready, "_install_for_run", install)
    monkeypatch.setattr(oligoai.subprocess, "run", infer)
    return root, calls


def test_install_choice_runs_predictions_before_cli_completion(missing_model, tmp_path, monkeypatch):
    root, calls = missing_model
    screen, captured_narrators = Terminal(tty=True), []
    narrator_class = runtime._CliNarrator
    original = row()
    candidate = deepcopy(original)
    outcome = {}

    def narrator(**kwargs):
        result = narrator_class(**kwargs)
        captured_narrators.append(result)
        return result

    def answer():
        assert "Install OligoAI now" in screen.getvalue()
        assert captured_narrators[0]._progress._thread is None
        calls.append("answer")
        return "yes"

    def run(**kwargs):
        kwargs["event_callback"]({"type": "tool_call", "tool": "tool_evaluate_existing_aso_set"})
        result = oligoai.evaluate_candidates([candidate], {}, tmp_path, references=[reference()])
        assert result["predicted_count"] == 1
        assert json.loads(Path(result["artifact_json"]).read_text())["predicted_count"] == 1
        assert "73.12" in oligoai.markdown_section(result)
        assert Path(result["artifact_json"]).with_name("installation.log").is_file()
        assert "Installing OligoAI" in screen.getvalue()
        assert "Installing OligoAI" not in sys.stdout.getvalue()
        # Repeated reporting uses the existing calculation without another prompt.
        again = oligoai.evaluate_candidates([candidate], {}, tmp_path, references=[reference()])
        assert again["predicted_count"] == 1
        calls.append("complete")
        return {"status": "run_complete", "run_dir": str(tmp_path)}

    monkeypatch.setattr(runtime, "_CliNarrator", narrator)
    monkeypatch.setattr(agentic_runner, "run_agentic_pipeline", run)
    monkeypatch.setattr(sys, "stdout", screen)
    monkeypatch.setattr(sys, "stdin", Terminal(tty=True))
    monkeypatch.setattr("builtins.input", answer)
    runtime.main(argv=["Evaluate these ASOs", "--no-stream-ui"], result_callback=outcome.update)
    assert calls == ["answer", "install", "predict", "complete"]
    assert outcome["status"] == "run_complete"
    assert screen.getvalue().count("Install OligoAI now") == 1
    assert captured_narrators[0]._progress._thread is None
    assert {k: v for k, v in candidate.items() if k != "oligoai_evaluation"} == original


@pytest.mark.parametrize("answer", ["no", "", EOFError(), OSError("No input")])
def test_decline_or_missing_answer_warns_once_without_installing(missing_model, tmp_path, monkeypatch, answer):
    root, calls = missing_model
    screen, captured = Terminal(tty=True), io.StringIO()
    narrator = runtime._CliNarrator(stream=screen, enable_stream=False)
    monkeypatch.setattr(sys, "stdin", Terminal(tty=True))

    def respond():
        assert narrator._progress._thread is None
        calls.append("answer")
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("builtins.input", respond)
    with contextlib.redirect_stdout(captured), warning_handler(narrator.on_runner_event), narrator._progress.running("Reporting"):
        for _ in range(2):
            result = oligoai.evaluate_candidates([row()], {}, tmp_path, references=[reference()])
            assert result["predicted_count"] == 0
            assert result["candidates"][0]["status"] == "unavailable"
            assert "unavailable" in oligoai.markdown_section(result)
    assert calls == ["answer"]
    assert not root.exists()
    assert screen.getvalue().count("Continuing without gapmer efficacy predictions") == 1
    assert screen.getvalue().count("Install OligoAI now") == 1
    assert "modal-aso setup --features oligoai" in screen.getvalue()
    assert "modal-aso assets import-model" in screen.getvalue()
    assert not captured.getvalue()
    assert narrator._progress._thread is None


@pytest.mark.parametrize("tty,stdin_tty,raw", [(False, False, False), (True, False, False),
                                             (False, True, False), (True, True, True)])
def test_unattended_and_raw_cli_warn_without_prompting(missing_model, tmp_path, monkeypatch, tty, stdin_tty, raw):
    _, calls = missing_model
    screen = Terminal(tty=tty)
    monkeypatch.setattr(sys, "stdout", screen)
    monkeypatch.setattr(sys, "stdin", Terminal(tty=stdin_tty))
    monkeypatch.setattr("builtins.input", lambda: pytest.fail("No interactive model installation"))

    def run(**kwargs):
        result = oligoai.evaluate_candidates([row()], {}, tmp_path, references=[reference()])
        assert result["runtime_availability"]["status"] == "unavailable"
        return {"status": "run_complete", "run_dir": str(tmp_path)}

    monkeypatch.setattr(agentic_runner, "run_agentic_pipeline", run)
    argv = ["Evaluate these ASOs", "--no-stream-ui"] + (["--raw-cli"] if raw else [])
    runtime.main(argv=argv)
    assert not calls
    assert "Continuing without gapmer efficacy predictions" in screen.getvalue()
    assert "Install OligoAI now" not in screen.getvalue()


def test_failed_install_is_visible_and_not_retried_in_the_same_run(missing_model, tmp_path, monkeypatch):
    _, calls = missing_model
    screen = Terminal(tty=True)
    narrator = runtime._CliNarrator(stream=screen, enable_stream=False)
    monkeypatch.setattr(sys, "stdin", Terminal(tty=True))
    monkeypatch.setattr("builtins.input", lambda: "y")

    def fail(root, log_path):
        calls.append("install")
        log_path.write_text("installer output\nCUDA is unavailable\n")
        raise subprocess.CalledProcessError(1, ["installer"])

    monkeypatch.setattr(ready, "_install_for_run", fail)
    with warning_handler(narrator.on_runner_event), narrator._progress.running("Reporting"):
        result = oligoai.evaluate_candidates([row()], {}, tmp_path, references=[reference()])
        assert "CUDA is unavailable" in result["runtime_availability"]["detail"]
        assert "installation.log" in result["candidates"][0]["reason"]
        oligoai.evaluate_candidates([row()], {}, tmp_path, references=[reference()])
    assert calls == ["install"]
    assert "OligoAI installation failed" in screen.getvalue()
    assert "CUDA is unavailable" in screen.getvalue()
    assert screen.getvalue().count("Install OligoAI now") == 1
    assert narrator._progress._thread is None


@pytest.mark.parametrize("case", ["steric", "disabled", "unsupported", "no_reference"])
def test_no_install_offer_when_predictions_cannot_use_the_model(missing_model, tmp_path, case):
    _, calls = missing_model
    events = []
    candidate = row("ACGTCAGACTTAGCACTGCA", "steric-hindrance" if case == "steric" else "2ome:5-10-5" if case == "unsupported" else "5-10-5")
    goal = {"oligoai_evaluation": {"enabled": False}} if case == "disabled" else {}
    with warning_handler(events.append):
        oligoai.evaluate_candidates([candidate], goal, tmp_path,
                                   references=[] if case == "no_reference" else [reference()])
    assert not events and not calls


def test_run_installer_uses_pinned_setup_and_captures_subprocess_output(tmp_path, monkeypatch):
    root, log_path = tmp_path / "model path", tmp_path / "prediction/installation.log"
    observed = []

    def launch(command, **kwargs):
        observed.append(command)
        assert command == [sys.executable, "-m", "aso_agents.oligoai_runtime", "--root", str(root)]
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.STDOUT and kwargs["check"]
        assert kwargs["env"]["PYTHONPATH"] == str(Path(ready.__file__).resolve().parents[1])
        assert "PYTHONHOME" not in kwargs["env"]
        kwargs["stdout"].write("Installer output\n")

    monkeypatch.setattr(ready.subprocess, "run", launch)
    monkeypatch.setenv("PYTHONHOME", "/unrelated/runtime")
    ready._install_for_run(root, log_path)
    assert len(observed) == 1
    assert log_path.read_text() == "Installer output\n"
