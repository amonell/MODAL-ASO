"""An intake question can precede a graph; it cannot substitute for one."""
import asyncio
import json
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline


def pending():
    return {"status": "need_clarification", "questions": ["Which sample should be used?"],
            "reason": "A user-controlled input is absent.", "request_envelope": None}


def ready():
    return {"status": "ready", "questions": [], "reason": None, "request_envelope": {
        "schema_version": "v3", "request_mode": "single_task", "request_summary": "Evaluate the supplied ASO",
        "request_classifications": ["evaluate_existing_aso_set"], "tasks": [{
            "task_id": "evaluate", "task_type": "evaluate_existing_aso_set", "label": "Evaluate",
            "native": True, "can_run_in_parallel": False, "context": {
                "gene_symbol": "NEUTRAL", "species": "human", "target_context": "cultured cells",
                "provided_asos": ["ACGTACGTACGTACGTACGT"]}}],
        "task_dependencies": [], "final_outputs": ["ASO evaluation"], "response_mode": "analysis",
        "requires_execution": True, "assumptions": []}}


def review(keep=False):
    return {"blocking_question_indexes": [0] if keep else [], "additional_blocking_questions": [],
            "reason": "The selected sample is missing." if keep else "The request already supplies the input."}


def test_null_draft_is_legal_only_for_a_nonready_response():
    assert pipeline._request_workbench_contract_errors(pending()) == []
    malformed = pending()
    malformed.update(status="ready", questions=[], reason=None)
    assert "ready output requires a complete envelope" in " ".join(pipeline._request_workbench_contract_errors(malformed))
    malformed = pending()
    malformed["questions"] = []
    assert pipeline._request_workbench_contract_errors(malformed)
    malformed = pending()
    malformed["request_envelope"] = {"tasks": []}
    assert pipeline._request_workbench_contract_errors(malformed)
    malformed = ready()
    malformed["request_envelope"]["tasks"][0]["context"]["invented_scientific_field"] = 42
    assert pipeline._request_workbench_contract_errors(malformed)


def test_needed_question_reaches_the_user_without_constructing_a_graph(monkeypatch):
    async def plan(_prompt):
        return pending()
    async def keep_question(**kwargs):
        assert json.loads(kwargs["prompt"])["proposed_questions"][0]["question"] == pending()["questions"][0]
        return review(True)
    async def no_graph(_prompt):
        raise AssertionError("There is no graph to verify or execute.")
    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    monkeypatch.setattr(pipeline, "_run_schema_validated_agent_once", keep_question)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_agent_once", no_graph)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once("Use my selected sample."))
    assert result["status"] == "need_clarification"
    assert result["request_envelope"] is None
    assert result["questions"] == pending()["questions"]
    assert not pipeline._request_workbench_contract_errors(result)


def test_rejected_question_cannot_promote_a_null_draft_to_ready(monkeypatch):
    async def reject(**_kwargs):
        return review()
    monkeypatch.setattr(pipeline, "_run_schema_validated_agent_once", reject)
    proposal = pending()
    result = asyncio.run(pipeline._review_request_clarifications_if_needed("Use sample X.", proposal))
    assert result["error"] == "request_graph_required_after_clarification_review"
    assert result["prior_response"] == proposal
    assert result["clarification_review"] == review()
    assert proposal == pending()


def test_rejected_question_returns_to_graph_planning_with_original_evidence(monkeypatch):
    plans, graphs = [], []
    async def plan(prompt):
        plans.append(prompt)
        if len(plans) == 1:
            return pending()
        feedback = json.loads(prompt.split("\n\n", 1)[1])
        assert feedback["original_evidence"] == "Use sample X."
        assert feedback["verified_graph_issues"][0]["prior_response"] == pending()
        assert feedback["verified_graph_issues"][0]["clarification_review"] == review()
        return ready()
    async def reject(**_kwargs):
        return review()
    async def graph(prompt):
        graphs.append(prompt)
        return {"status": "valid", "issues": []}
    async def finalize(_prompt, planned):
        return planned
    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    monkeypatch.setattr(pipeline, "_run_schema_validated_agent_once", reject)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_agent_once", graph)
    monkeypatch.setattr(pipeline, "_finalize_verified_request_plan", finalize)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once("Use sample X."))
    assert result == ready()
    assert len(plans) == 2 and len(graphs) == 2


def test_clarification_and_graph_repairs_share_the_existing_revision_budget(monkeypatch):
    monkeypatch.setenv("ASO_REQUEST_GRAPH_MAX_REVISIONS", "1")
    plans = []
    async def plan(prompt):
        plans.append(prompt)
        return pending() if len(plans) == 1 else ready()
    async def reject(**_kwargs):
        return review()
    async def bad_graph(_prompt):
        return {"status": "revise", "issues": [{"issue": "missing requested operation"}]}
    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    monkeypatch.setattr(pipeline, "_run_schema_validated_agent_once", reject)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_agent_once", bad_graph)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_adjudicator_once", bad_graph)
    monkeypatch.setattr(pipeline, "_request_graph_review_evidence_errors", lambda *_args: [])
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once("Use sample X."))
    assert result["error"] == "request_graph_verification_failed"
    assert len(plans) == 2


def test_repeated_rejected_question_stops_without_adding_repair_attempts(monkeypatch):
    plans = []
    async def plan(prompt):
        plans.append(prompt)
        return pending()
    async def reject(**_kwargs):
        return review()
    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    monkeypatch.setattr(pipeline, "_run_schema_validated_agent_once", reject)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once("Use sample X."))
    assert result["error"] == "request_graph_verification_no_progress"
    assert len(plans) == 2


@pytest.mark.parametrize("cached", [False, True])
def test_pending_null_draft_survives_persistence_and_cache_without_invented_tasks(tmp_path, monkeypatch, cached):
    request = "Design ASOs using my selected sample."
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps({"version": "2", "status": "initialized", "phase": "initialized",
        "request": request, "run_dir": str(run_dir), "flags": {}, "last_error": None,
        "pending_questions": [], "clarification": {"isoform_scope_confirmed": False},
        "clarification_qa_history": [], "force_rerun_stages": [], "events": [], "artifacts": {}}))
    if cached:
        pipeline._save_goal_payload(str(run_dir), {"request": request, "qa_history": [],
            "request_envelope": None}, {}, pending())
    async def plan(_prompt, **_kwargs):
        assert not cached, "Cached questions should not rerun planning."
        return pending()
    def no_audit(**_kwargs):
        raise AssertionError("A question without a graph must not reach execution auditing.")
    monkeypatch.setattr(pipeline, "_run_verified_request_workbench_agent_once", plan)
    monkeypatch.setattr(pipeline, "_repair_request_envelope_if_needed", no_audit)
    result = pipeline.tool_parse_goal_request(str(state_path))
    state = pipeline._load_state(result["state_path"])
    assert result["status"] == "ok", result
    assert result["intake_partial"] is True
    assert state["status"] == "intake_partial"
    assert state["pending_questions"] == pending()["questions"]
    assert state["task_graph"] == {"tasks": [], "dependencies": []}
    assert not state["active_task_ids"] and not state["request_envelope"]
    saved = pipeline._load_goal_context_from_run_dir(state["run_dir"])
    assert not saved["request_envelope"] and not saved["structured_goal"]
    assert saved["final_payload"]["questions"] == pending()["questions"]
