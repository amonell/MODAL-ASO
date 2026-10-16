"""Report admission follows requested programs without bypassing missing work."""
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.design_delivery import unexecuted_design_branches


def gapmer_context():
    goal = {
        "intent": "splice_modulation",
        "design_programs": [
            {"program_id": "rna_depletion", "modality": "gapmer", "intent": "knockdown"}
        ],
    }
    final = {"gapmer_tiling_results": {"row_count": 2}}
    return goal, final


@pytest.mark.parametrize("intent", ["binding", "splice_modulation", "knockdown"])
@pytest.mark.parametrize("legacy_flags", [{}, {"run_splice_eval": True, "run_splice_router": True}])
def test_explicit_gapmer_program_does_not_invent_unexecuted_steric_work(intent, legacy_flags):
    goal, final = gapmer_context()
    goal["intent"] = intent
    final.update(legacy_flags)
    original = deepcopy((goal, final))
    branches = pipeline._build_branch_outcomes_for_report(goal, final)
    assert set(branches) == {"gapmer"}
    assert branches["gapmer"]["status"] == "outputs_present"
    assert unexecuted_design_branches(goal, branches) == []
    assert (goal, final) == original


def test_explicit_mixed_request_still_requires_unexecuted_steric_program():
    goal, final = gapmer_context()
    goal["design_programs"].append(
        {"program_id": "regulatory_block", "modality": "steric", "intent": "binding"}
    )
    branches = pipeline._build_branch_outcomes_for_report(goal, final)
    assert branches["steric"]["status"] == "unknown"
    assert unexecuted_design_branches(goal, branches) == ["steric"]


def test_recorded_steric_job_is_not_hidden_by_gapmer_only_program_summary():
    goal, final = gapmer_context()
    final["steric_design_jobs"] = [{
        "job_id": "unresolved_regulatory_block", "region_rows": [],
        "status": "skipped", "empty_reason_code": "planning_missing",
        "reason": "No region plan was produced.",
    }]
    branches = pipeline._build_branch_outcomes_for_report(goal, final)
    assert branches["steric"]["status"] == "unknown"
    assert unexecuted_design_branches(goal, branches) == ["steric"]


@pytest.mark.parametrize("intent", ["binding", "splice_modulation"])
def test_legacy_request_without_programs_keeps_required_steric_branch(intent):
    goal = {"intent": intent}
    branches = pipeline._build_branch_outcomes_for_report(goal, {})
    assert unexecuted_design_branches(goal, branches) == ["steric"]


def test_missing_gapmer_execution_remains_blocked_despite_correct_branch_scope():
    goal, _ = gapmer_context()
    branches = pipeline._build_branch_outcomes_for_report(goal, {})
    assert unexecuted_design_branches(goal, branches) == ["gapmer"]


@pytest.mark.parametrize("tool_name", ["tool_build_report_payload", "tool_write_markdown_report"])
@pytest.mark.parametrize("mixed_request", [False, True])
def test_report_admission_agrees_with_explicit_program_scope(monkeypatch, tmp_path, tool_name, mixed_request):
    goal, final = gapmer_context()
    if mixed_request:
        goal["design_programs"].append({"program_id": "regulatory_block", "modality": "steric"})
    monkeypatch.setattr(pipeline, "_load_state", lambda _: {"run_dir": str(tmp_path)})
    monkeypatch.setattr(pipeline, "_reject_disallowed_tool_call", lambda **_: None)
    monkeypatch.setattr(pipeline, "_mark_last_tool", lambda *_: None)
    monkeypatch.setattr(pipeline, "_relocate_agentic_state_into_run_dir", lambda state, path, run: path)
    monkeypatch.setattr(pipeline, "_load_goal_context_from_run_dir", lambda _: {
        "structured_goal": goal, "final_payload": final,
    })
    result = pipeline._prepare_execution_stage(
        state_path=str(tmp_path / "state.json"), tool_name=tool_name,
        stage_name="report", force_rerun=False, issue_summary="",
        invalidation_stage="report", invalidation_default_summary="",
    )
    if mixed_request:
        assert result["status"] == "blocked"
        assert result["error"] == "design_execution_pending"
        assert result["requested_branches"] == ["steric"]
    else:
        assert result["status"] == "ok"
