from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aso_agents import agentic_runner
from aso_agents import pipeline_tools


def test_unplanned_branch_tool_cannot_advance_steric_tiling_state(tmp_path: Path) -> None:
    state_path = tmp_path / "run_state.json"
    state = {
        "version": "2",
        "status": "tiling_config_ready",
        "phase": "tiling_config_ready",
        "run_dir": str(tmp_path),
        "last_tool": "evaluate_step_output",
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    before = state_path.read_bytes()

    planning = pipeline_tools._derive_next_actions(state)
    assert planning["next_actions"][0]["tool"] == "tool_run_steric_tiling"
    assert "tool_run_gapmer_tiling" in planning["allowed_next_tools"]
    assert "restart_from_stage" not in planning["allowed_next_tools"]
    assert "restart_from_stage" not in [action["tool"] for action in planning["next_actions"]]

    ungrounded_restart = pipeline_tools.restart_from_stage(
        state_path=str(state_path), stage="steric_tiling",
        issue_summary="Try the same setup again.",
    )
    assert ungrounded_restart["error"] == "tool_not_allowed_in_current_state"
    assert state_path.read_bytes() == before

    rejected = pipeline_tools.tool_run_gapmer_tiling(state_path=str(state_path))
    assert rejected["status"] == "retryable_error"
    assert rejected["error"] == "tool_not_selected_by_current_plan"
    assert rejected["next_actions"][0]["tool"] == "tool_run_steric_tiling"
    assert state_path.read_bytes() == before


def test_existing_state_runs_planned_action_without_model_controller(tmp_path: Path, monkeypatch: Any) -> None:
    state_path = tmp_path / "run_state.json"
    state = {
        "version": "2",
        "status": "goal_schema_ready",
        "phase": "goal_schema_ready",
        "run_dir": str(tmp_path),
        "last_tool": "evaluate_step_output",
        "planning": {},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def fake_derive(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "terminal_status": "in_progress",
            "next_actions": [{"tool": "tool_resolve_isoforms_and_sites"}],
            "allowed_next_tools": ["tool_resolve_isoforms_and_sites"],
        }

    def fake_resolve(*, state_path: str, **_kwargs: Any) -> dict[str, Any]:
        payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
        payload["status"] = "resolution_ready"
        payload["phase"] = "resolution_ready"
        payload["last_tool"] = "tool_resolve_isoforms_and_sites"
        Path(state_path).write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "ok", "state_path": state_path, "run_dir": str(tmp_path)}

    def fail_controller(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("model controller should not be called for planned deterministic action")

    monkeypatch.setattr(agentic_runner, "_derive_next_actions", fake_derive)
    monkeypatch.setattr(pipeline_tools, "tool_resolve_isoforms_and_sites", fake_resolve)
    monkeypatch.setattr(agentic_runner.Runner, "_run_sync", fail_controller)

    result = agentic_runner.run_agentic_pipeline(
        user_request="resolve test",
        existing_state_path=str(state_path),
        max_steps=1,
    )

    assert result["deterministic_controller"] is True
    assert result["deterministic_tool"] == "tool_resolve_isoforms_and_sites"
    assert result["status"] == "in_progress"

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["active_runner"] is None
    assert payload["runner_history"][-2]["event"] == "started"
    assert payload["runner_history"][-1]["event"] == "finished"


def test_interrupted_planned_action_marks_recoverable_state(tmp_path: Path, monkeypatch: Any) -> None:
    state_path = tmp_path / "run_state.json"
    state = {
        "version": "2",
        "status": "steric_expression_ready",
        "phase": "steric_expression_ready",
        "run_dir": str(tmp_path),
        "last_tool": "evaluate_step_output",
        "stage_attempts": {},
        "planning": {},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def fake_derive(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "terminal_status": "in_progress",
            "next_actions": [{"tool": "tool_run_splice_eval"}],
            "allowed_next_tools": ["tool_run_splice_eval"],
        }

    def fake_splice_eval(*, state_path: str, **_kwargs: Any) -> dict[str, Any]:
        raise KeyboardInterrupt()

    monkeypatch.setattr(agentic_runner, "_derive_next_actions", fake_derive)
    monkeypatch.setattr(pipeline_tools, "tool_run_splice_eval", fake_splice_eval)

    with pytest.raises(KeyboardInterrupt):
        agentic_runner._run_planned_next_action_once(state_path)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["phase"] == "controller_tool_failed"
    assert payload["active_runner"] is None
    assert payload["last_error"]["kind"] == "deterministic_tool_interrupted"
    assert payload["last_error"]["tool"] == "tool_run_splice_eval"

    planning = pipeline_tools._derive_next_actions(payload)
    assert planning["terminal_status"] == "in_progress"
    assert planning["next_actions"][0]["tool"] == "tool_run_splice_eval"


def test_deterministic_tool_exception_restarts_its_stage_once(tmp_path: Path) -> None:
    state = {
        "version": "2",
        "status": "failed",
        "phase": "controller_tool_failed",
        "run_dir": str(tmp_path),
        "last_tool": "evaluate_step_output",
        "last_error": {
            "kind": "deterministic_tool_exception",
            "tool": "tool_score_gapmer_uniqueness",
            "message": "unexpected implementation error",
        },
        "runner_history": [
            {
                "event": "failed",
                "status": "deterministic_tool_exception",
                "tool": "tool_score_gapmer_uniqueness",
            }
        ],
        "retry_policy": {"max_stage_retries": 1},
        "stage_attempts": {},
    }

    planning = pipeline_tools._derive_next_actions(state)

    assert planning["terminal_status"] == "in_progress"
    assert planning["next_actions"][0]["tool"] == "restart_from_stage"
    assert planning["next_actions"][0]["arguments"]["stage"] == "gapmer_scoring"

    state["runner_history"].append(dict(state["runner_history"][0]))
    exhausted = pipeline_tools._derive_next_actions(state)

    assert exhausted["terminal_status"] == "fatal"
    assert exhausted["stop_reason"] == "deterministic_tool_exception_retry_exhausted"


def test_unknown_atomic_error_retries_from_producer_metadata(
    tmp_path: Path,
) -> None:
    state = {
        "version": "2",
        "status": "failed",
        "phase": "tool_parse_goal_failed",
        "run_dir": str(tmp_path),
        "last_tool": "evaluate_step_output",
        "last_error": {
            "kind": "new_intake_validation_failure",
            "failed_tool": "tool_parse_goal_request",
            "stage": "intake",
            "message": "A schema-bound intake checker rejected the result.",
            "attempt": 1,
            "retryable": True,
        },
        "retry_policy": {"max_stage_retries": 1},
        "stage_attempts": {"intake": 1},
    }

    planning = pipeline_tools._derive_next_actions(state)

    assert planning["terminal_status"] == "in_progress"
    assert planning["next_actions"][0]["tool"] == "tool_parse_goal_request"
    assert planning["next_actions"][0]["arguments"]["force_rerun"] is True

    state["stage_attempts"]["intake"] = 2
    exhausted = pipeline_tools._derive_next_actions(state)

    assert exhausted["terminal_status"] == "fatal"
    assert exhausted["stop_reason"] == "intake_retry_exhausted"


def test_planned_action_follows_goal_dir_rename(tmp_path: Path, monkeypatch: Any) -> None:
    old_run = tmp_path / "goal_111_GOAL"
    agentic_dir = old_run / "agentic_111"
    agentic_dir.mkdir(parents=True)
    state_path = agentic_dir / "run_state.json"
    state = {
        "version": "2",
        "status": "initialized",
        "phase": "initialized",
        "run_dir": str(old_run),
        "last_tool": "evaluate_step_output",
        "events": [],
        "snapshot_seq": 0,
        "planning": {},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def fake_derive(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "terminal_status": "in_progress",
            "next_actions": [{"tool": "tool_parse_goal_request"}],
            "allowed_next_tools": ["tool_parse_goal_request"],
        }

    def fake_parse(*, state_path: str, **_kwargs: Any) -> dict[str, Any]:
        old_state = Path(state_path)
        source_run = old_state.parent.parent
        new_run = source_run.parent / "goal_111_MAPT"
        source_run.rename(new_run)
        new_state = new_run / old_state.parent.name / old_state.name
        payload = json.loads(new_state.read_text(encoding="utf-8"))
        payload["status"] = "intake_ready"
        payload["phase"] = "intake_ready"
        payload["run_dir"] = str(new_run)
        payload["last_tool"] = "tool_parse_goal_request"
        new_state.write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "ok", "state_path": str(new_state), "run_dir": str(new_run)}

    monkeypatch.setattr(agentic_runner, "_derive_next_actions", fake_derive)
    monkeypatch.setattr(pipeline_tools, "tool_parse_goal_request", fake_parse)

    result = agentic_runner._run_planned_next_action_once(state_path)

    assert result is not None
    assert result["status"] != "fatal"
    assert "state_read_failed" not in str(result.get("error") or "")
    assert Path(result["state_path"]).is_file()
    assert Path(result["state_path"]).parent.parent.name == "goal_111_MAPT"
    assert not old_run.exists()
    assert (tmp_path / "goal_111_MAPT" / "agentic_111" / "run_state.json").is_file()
    assert not (tmp_path / "goal_111_GOAL").exists()

    inferred = agentic_runner._infer_status_from_state(state_path)
    assert inferred["status"] != "fatal"
    assert Path(inferred["state_path"]).parent.parent.name == "goal_111_MAPT"


def test_progress_signature_tracks_string_last_tool_and_active_runner(tmp_path: Path) -> None:
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "snapshot_seq": 7,
                "updated_at": "2026-08-02T00:00:00Z",
                "status": "steric_expression_ready",
                "phase": "steric_expression_ready",
                "last_tool": "tool_annotate_offtarget_expression",
                "tool_history": [
                    {"tool": "tool_annotate_offtarget_expression", "status": "ok"},
                ],
                "active_runner": {
                    "tool": "tool_run_splice_eval",
                    "pid": 12345,
                    "heartbeat_at": "2026-08-02T00:01:00Z",
                },
            }
        ),
        encoding="utf-8",
    )

    signature = pipeline_tools._agentic_progress_signature(str(state_path))

    assert signature is not None
    assert "tool_annotate_offtarget_expression" in signature
    assert "ok" in signature
    assert "tool_run_splice_eval" in signature
    assert 12345 in signature


def test_active_execution_tool_can_request_clarification(tmp_path: Path) -> None:
    state_path = tmp_path / "run_state.json"
    state = {
        "version": "2",
        "status": "preflight_ready",
        "phase": "preflight_ready",
        "run_dir": str(tmp_path),
        "last_tool": "evaluate_step_output",
        "active_runner": {
            "tool": "tool_materialize_region_constraints",
            "pid": 12345,
            "heartbeat_at": "2026-08-02T00:01:00Z",
        },
        "planning": {},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = pipeline_tools.request_clarification(
        state_path=str(state_path),
        questions=["Which exact site should be targeted?"],
        reason="site_coordinates_missing",
        clarification_kind="goal_intake",
        restart_stage="intake",
    )

    assert result["status"] == "ok"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "needs_user_input"
    assert payload["pending_questions"] == ["Which exact site should be targeted?"]
