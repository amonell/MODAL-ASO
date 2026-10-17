from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _credentialless_cli_env() -> dict[str, str]:
    env = dict(os.environ)
    env["ASO_LLM_PROVIDER"] = "openai"
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENROUTER_API_KEY", None)
    return env


def test_modal_aso_launcher_imports_without_editable_pth(tmp_path: Path) -> None:
    env = _credentialless_cli_env()
    env["PYTHONPATH"] = ""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "modal-aso"), "--help"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage" in proc.stdout.lower() or "modal-aso" in proc.stdout.lower()


def test_public_help_hides_internal_automation_flags() -> None:
    env = _credentialless_cli_env()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_goal_intake.py"), "--help"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    help_text = proc.stdout
    assert "--raw-cli" not in help_text
    assert "--no-stream-ui" not in help_text
    assert "--agentic-answers-json" not in help_text
    assert "--agentic-answer-summary" not in help_text
    assert "--router-log-tool-calls" not in help_text
    assert "--router-debug-dir" not in help_text
    assert "--router-print-summary" not in help_text
    assert "raw " + "JSON" not in help_text
    assert "summary " + "JSON" not in help_text
    assert "modal-aso" in help_text or "Run the MODAL-ASO pipeline controller" in help_text


def test_cli_intro_uses_neutral_stepwise_copy() -> None:
    import run_goal_intake

    stream = StringIO()
    narrator = run_goal_intake._CliNarrator(
        stream=stream,
        enable_stream=False,
        enable_spinner=False,
    )

    narrator.intro()

    output = stream.getvalue()
    assert "I'm taking your request and working through it step by step." in output
    assert "pipeline step by step" not in output


def test_cli_pauses_cleanly_when_clarification_stdin_is_unavailable(monkeypatch, tmp_path: Path, capsys) -> None:
    import builtins

    import aso_agents.agentic_runner as agentic_runner
    import run_goal_intake

    state_path = tmp_path / "run_state.json"
    state_path.write_text("{}", encoding="utf-8")

    def fake_run_agentic_pipeline(**_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "needs_user_input",
            "terminal_status": "needs_user_input",
            "state_path": str(state_path),
            "run_dir": str(tmp_path),
            "pending_questions": ["Which artifact should I inspect?"],
            "planning": {
                "clarification_request": {
                    "reason": "artifact_refs_missing",
                    "questions": ["Which artifact should I inspect?"],
                }
            },
        }

    monkeypatch.setattr(agentic_runner, "run_agentic_pipeline", fake_run_agentic_pipeline)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": (_ for _ in ()).throw(EOFError))
    monkeypatch.setattr(sys, "argv", ["modal-aso", "Explain a missing run"])

    run_goal_intake.main()

    output = capsys.readouterr().out
    assert "No clarification answer was available on stdin" in output
    assert "AGENTIC_RUN_RESULT" in output
    assert '"status": "needs_user_input"' in output


def test_genbank_exporter_is_packaged_and_non_routing_skip_is_clear(tmp_path: Path, capsys) -> None:
    import run_goal_intake

    assert run_goal_intake._export_asos_to_genbank is not None

    (tmp_path / "structured_goal.json").write_text("{}", encoding="utf-8")
    result = run_goal_intake._auto_export_genbank(str(tmp_path))

    output = capsys.readouterr().out
    assert result is None
    assert "neither routed nor generated ASO candidates were found" in output
    assert "utility not available" not in output


def test_write_steric_bed_creates_canonical_parent(tmp_path: Path) -> None:
    import run_goal_intake

    result = run_goal_intake._write_steric_bed(
        [["chr5", 100, 120, "acceptor_window", 0, "+"]],
        str(tmp_path),
    )

    assert result is not None
    result_path = Path(result)
    assert result_path.is_file()
    assert result_path.read_text(encoding="utf-8") == "chr5\t100\t120\tacceptor_window\t0\t+\n"


def test_aso_designer_trackhub_generation_is_optional() -> None:
    snakefile = (ROOT / "aso-designer-main" / "aso-designer-main" / "Snakefile").read_text(
        encoding="utf-8"
    )

    assert "command -v maketrackhubs" in snakefile
    assert "TRACKHUB SKIPPED: maketrackhubs executable not found." in snakefile
    assert "trackhub unavailable; maketrackhubs did not provide a final hub URL" in snakefile


def test_initialize_run_state_ignores_external_run_dir_override(monkeypatch, tmp_path: Path) -> None:
    import json

    from aso_agents import pipeline_tools

    monkeypatch.setattr(pipeline_tools, "_repo_root", lambda: tmp_path)
    external_artifact_path = tmp_path.parent / "user_artifact_path"

    result = pipeline_tools.initialize_run_state(
        request=f"Inspect {external_artifact_path} and summarize the run.",
        run_dir_override=str(external_artifact_path),
    )

    assert result["status"] == "ok"
    state = json.loads(Path(result["state_path"]).read_text(encoding="utf-8"))
    run_dir = Path(state["run_dir"])
    assert run_dir != external_artifact_path
    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.name.startswith("goal_")
    assert Path(result["state_path"]).parent.parent == run_dir / "artifacts" / "controller"
    assert Path(result["state_path"]).parent.name.startswith("agentic_")
    assert state["ignored_run_dir_override"] == str(external_artifact_path)
    assert not external_artifact_path.exists()


def test_initialize_run_state_accepts_managed_run_dir_override(monkeypatch, tmp_path: Path) -> None:
    import json

    from aso_agents import pipeline_tools

    monkeypatch.setattr(pipeline_tools, "_repo_root", lambda: tmp_path)
    managed_run_dir = tmp_path / "runs" / "goal_managed"

    result = pipeline_tools.initialize_run_state(
        request="Design human MAPT gapmer ASOs.",
        run_dir_override=str(managed_run_dir),
    )

    assert result["status"] == "ok"
    state = json.loads(Path(result["state_path"]).read_text(encoding="utf-8"))
    assert state["run_dir"] == str(managed_run_dir)
    assert managed_run_dir.is_dir()
    assert Path(result["state_path"]).parent.parent == managed_run_dir / "artifacts" / "controller"
    assert Path(result["state_path"]).parent.name.startswith("agentic_")


def test_initialize_run_state_does_not_activate_artifact_analysis_input_run(monkeypatch, tmp_path: Path) -> None:
    import json

    from aso_agents import pipeline_tools

    monkeypatch.setattr(pipeline_tools, "_repo_root", lambda: tmp_path)
    source_run_dir = tmp_path / "runs" / "goal_source_ttn"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "generated_aso_set.json").write_text('{"rows": []}', encoding="utf-8")

    result = pipeline_tools.initialize_run_state(
        request=(
            f"Evaluate the MODAL-ASO output in {source_run_dir}. Do not design new ASOs "
            "and do not rerun tiling. Return the top 10 ASOs."
        ),
        run_dir_override=str(source_run_dir),
    )

    assert result["status"] == "ok"
    state = json.loads(Path(result["state_path"]).read_text(encoding="utf-8"))
    run_dir = Path(state["run_dir"])
    assert run_dir != source_run_dir
    assert run_dir.parent == tmp_path / "runs"
    assert Path(result["state_path"]).parent.parent == run_dir / "artifacts" / "controller"
    assert Path(result["state_path"]).parent.name.startswith("agentic_")
    assert not any(source_run_dir.glob("agentic_*"))
    assert state["ignored_run_dir_override"] == str(source_run_dir)
    assert state["ignored_run_dir_override_reason"] == "artifact_analysis_input_run"
    assert state["request_envelope"]["workflow_name"] == "analyze_artifact"
    assert state["request_envelope"]["tasks"][0]["task_type"] == "analyze_artifact"


def test_relocate_agentic_state_moves_sibling_folder_into_run_dir(monkeypatch, tmp_path: Path) -> None:
    from aso_agents import pipeline_tools

    monkeypatch.setattr(pipeline_tools, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline_tools, "_derive_next_actions", lambda _state: {})
    run_dir = tmp_path / "runs" / "goal_123_MAPT"
    run_dir.mkdir(parents=True)
    agentic_dir = tmp_path / "runs" / "agentic_123"
    agentic_dir.mkdir()
    state_path = agentic_dir / "run_state.json"
    state = {"run_dir": str(run_dir), "events": [], "snapshot_seq": 0}
    state_path.write_text("{}", encoding="utf-8")

    moved = pipeline_tools._relocate_agentic_state_into_run_dir(state, str(state_path), str(run_dir))

    assert Path(moved) == run_dir / "artifacts" / "controller" / "agentic_123" / "run_state.json"
    assert Path(moved).is_file()
    assert not agentic_dir.exists()
    assert pipeline_tools._find_agentic_state_for_run_dir(str(run_dir)) == moved


def test_rename_run_dir_rewrites_nested_agentic_state_path(tmp_path: Path, monkeypatch: Any) -> None:
    from aso_agents import pipeline_tools

    monkeypatch.setattr(pipeline_tools, "_derive_next_actions", lambda _state: {})
    old_run = tmp_path / "goal_123_GOAL"
    agentic_dir = old_run / "agentic_456"
    agentic_dir.mkdir(parents=True)
    state_path = agentic_dir / "run_state.json"
    state_path.write_text("{}", encoding="utf-8")
    state = {"run_dir": str(old_run), "events": []}

    new_run, _payload, _goal, _final, renamed, new_state_path = pipeline_tools._maybe_rename_run_dir_with_gene(
        state=state,
        payload={},
        structured_goal={"gene_symbol": "MAPT"},
        final_payload={},
        reason="test",
        state_path=str(state_path),
    )

    assert renamed is True
    assert Path(new_run).name == "goal_123_MAPT"
    assert new_state_path == str(Path(new_run) / "agentic_456" / "run_state.json")
    assert Path(new_state_path).is_file()
    assert not old_run.exists()
    resolved = pipeline_tools._state_path_to_obj(str(state_path))
    assert resolved == Path(new_state_path)
    loaded = pipeline_tools._load_state(str(state_path))
    assert loaded["run_dir"] == new_run


def test_canonical_state_path_follows_renamed_goal_dir(tmp_path: Path) -> None:
    from aso_agents import agentic_runner
    from aso_agents import pipeline_tools

    old_run = tmp_path / "goal_999_GOAL"
    new_run = tmp_path / "goal_999_MAPT"
    agentic_dir = new_run / "agentic_999"
    agentic_dir.mkdir(parents=True)
    state_path = agentic_dir / "run_state.json"
    state_path.write_text('{"run_dir": "%s"}' % new_run, encoding="utf-8")

    stale = old_run / "agentic_999" / "run_state.json"
    assert not stale.exists()
    assert pipeline_tools._state_path_to_obj(str(stale)) == state_path
    assert agentic_runner._canonical_state_path(str(stale)) == str(state_path)
    assert agentic_runner._authoritative_state_path(
        existing_state_path=str(stale),
        controller_state_path=None,
        responses=[],
    ) == str(state_path)
