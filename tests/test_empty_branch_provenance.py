import pytest

from aso_agents.pipeline_tools import _build_branch_outcomes_for_report
from aso_agents.design_delivery import design_delivery


@pytest.mark.parametrize("modality,intent", [("gapmer", "knockdown"), ("steric", "binding")])
@pytest.mark.parametrize("reason_code", ["region_build_failed", "planning_missing"])
def test_missing_plan_is_not_reported_as_exhausted_search(modality, intent, reason_code):
    goal = {"intent": intent, "candidate_minimum": 0}
    payload = {modality + "_design_jobs": [{"job_id": "p", "program_id": "p", "region_rows": [],
                "empty_reason_code": reason_code, "status": "skipped", "reason": "No plan was derived."}]}
    branches = _build_branch_outcomes_for_report(goal, payload)
    assert branches[modality]["status"] == "unknown"
    assert "No plan was derived." in branches[modality]["reason"]
    result = design_delivery(goal, {}, branches)
    assert "requested_design_not_executed:" + modality in result["errors"]
    assert "empty_result_search_completion_unestablished" in result["errors"]


@pytest.mark.parametrize("modality,intent", [("gapmer", "knockdown"), ("steric", "binding")])
def test_verified_constraint_exhaustion_remains_distinct(modality, intent):
    goal = {"intent": intent, "candidate_minimum": 0}
    payload = {modality + "_design_jobs": [{"job_id": "p", "program_id": "p", "region_rows": [],
                "empty_reason_code": "constraints_exhausted", "status": "skipped",
                "reason": "The resolved region intersection is empty."}]}
    branches = _build_branch_outcomes_for_report(goal, payload)
    assert branches[modality]["status"] == "no_candidates_after_constraints"
    assert design_delivery(goal, {}, branches)["status"] == "fulfilled"
