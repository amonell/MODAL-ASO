"""Effect routing cannot require predictions explicitly excluded by a branch."""
from copy import deepcopy

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline


def context(evaluate=False, *, persist_jobs=False):
    program = {
        "program_id": "sequence_library", "modality": "steric",
        "intent": "splice_modulation", "desired_effect": "skipping",
        "run_splice_router": True,
    }
    if evaluate is not None:
        program["run_splice_eval"] = evaluate
    goal = {"intent": "splice_modulation", "design_programs": [program]}
    final = {}
    if persist_jobs:
        final["steric_design_jobs"] = [{**program, "job_id": "library",
            "region_rows": [["chr8", 100, 150, "window", ".", "+"]]}]
    return {"structured_goal": goal, "final_payload": final, "payload": {}}


@pytest.mark.parametrize("persist_jobs", [False, True])
@pytest.mark.parametrize("evaluate", [False, True, None])
def test_execution_scope_preserves_intent_and_legacy_prediction_default(monkeypatch, evaluate, persist_jobs):
    current = context(evaluate, persist_jobs=persist_jobs)
    before = deepcopy(current)
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: current)
    assert pipeline._resolve_run_splice_router("unused") is (evaluate is not False)
    assert runtime._resolve_run_splice_router(current["structured_goal"], "") is True
    assert pipeline._resolve_intent_steric("unused") is True
    assert current == before


@pytest.mark.parametrize("enabled_index", [0, 1])
def test_mixed_jobs_route_only_members_with_predictions(monkeypatch, enabled_index):
    current = context(False, persist_jobs=True)
    jobs = current["final_payload"]["steric_design_jobs"]
    jobs.append({**jobs[0], "job_id": "other", "program_id": "other_library"})
    jobs[enabled_index]["run_splice_eval"] = True
    current["structured_goal"]["run_splice_eval"] = False
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: current)
    assert pipeline._resolve_run_splice_router("unused") is True
    assert pipeline._steric_jobs_requiring_splice_router(
        current["final_payload"], current["structured_goal"]
    ) == [jobs[enabled_index]]


def test_legacy_explicit_exclusion_prevents_effect_routing(monkeypatch):
    current = {"structured_goal": {"intent": "splice_modulation",
        "run_splice_eval": False, "run_splice_router": True}, "final_payload": {}, "payload": {}}
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: current)
    assert not pipeline._design_request_requires_splice_router(None, fallback_goal=current["structured_goal"])
    assert not pipeline._resolve_run_splice_router("unused")
    assert pipeline._resolve_intent_steric("unused")


def test_atomic_route_tool_skips_excluded_predictions_without_scored_files(monkeypatch, tmp_path):
    current = context(False, persist_jobs=True)
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: current)
    monkeypatch.setattr(pipeline, "_prepare_execution_stage", lambda **_: {
        "status": "ok", "state": {}, "run_dir": str(tmp_path), "force_stage": False})
    monkeypatch.setattr(pipeline, "_routing_required_after_splice_eval", lambda _: False)
    monkeypatch.setattr(pipeline, "_finish_stage_idempotent", lambda **kwargs: kwargs)
    monkeypatch.setattr(pipeline, "_artifact_markers_for_run",
        lambda _: pytest.fail("Excluded effect routing must not look for prediction artifacts"))
    result = pipeline.tool_build_routing_plan(state_path=str(tmp_path / "state.json"))
    assert result["status"] == "routing_ready"
    assert result["extra"]["skipped"] is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("evaluate", [False, True])
def test_completion_requires_only_requested_prediction_evidence(monkeypatch, evaluate):
    current = context(evaluate)
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: current)
    monkeypatch.setattr(pipeline, "_artifact_markers_for_run", lambda _: {})
    monkeypatch.setattr(pipeline, "_load_generated_set_payload", lambda _: {"rows": [{"sequence": "ACGT"}]})
    monkeypatch.setattr(pipeline, "_route_candidate_count", lambda *args: 0)
    result = pipeline._classify_design_outcome(child_run_dir="unused", task_context={},
        child_status="run_complete", generated_artifacts={})
    if evaluate:
        assert {"splice_eval", "splice_routing"} <= set(result["missing_required_evidence"])
    else:
        assert result["missing_required_evidence"] == []
        assert result["outcome_status"] == "completed_with_candidates"
