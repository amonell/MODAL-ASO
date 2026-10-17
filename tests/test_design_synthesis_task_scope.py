"""A design compiler must receive the verified ownership of composite work."""
from copy import deepcopy
import json

from aso_agents import pipeline_tools as pipeline


def compile_input(envelope):
    prompt = pipeline._design_goal_synthesis_prompt(
        request_evidence="Original request and clarification evidence.",
        request_envelope=envelope,
        task_index=0,
    )
    return json.loads(prompt.split("\n\n", 1)[1])


def composite_request():
    return {
        "tasks": [
            {"task_id": "design", "task_type": "design_new_asos", "label": "New candidates",
             "merged_from_task_ids": ["earlier_design"],
             "context": {"species": "human", "structured_goal": {"stale": True}}},
            {"task_id": "evaluate", "task_type": "evaluate_existing_aso_set",
             "label": "Evaluate supplied molecules", "context": {"provided_asos": [{"id": "fixed"}]}},
            {"task_id": "report", "task_type": "build_composite_report",
             "label": "Assemble all results", "context": {"format": "xlsx"}},
            {"task_id": "other_design", "task_type": "design_new_asos", "label": "Independent design",
             "context": {"species": "mouse", "structured_goal": {"do_not_copy": True}}},
        ],
        "task_dependencies": [{"upstream_task_id": "design", "downstream_task_id": "report"}],
        "final_outputs": [
            {"task_id": "earlier_design", "kind": "candidate_table", "path": "candidates.csv"},
            {"task_id": "evaluate", "kind": "evaluation_table", "path": "supplied.csv"},
            {"task_id": "report", "kind": "workbook", "path": "combined.xlsx"},
        ],
    }


def test_compiler_preserves_non_design_task_ownership_without_copying_goals():
    envelope = composite_request()
    original = deepcopy(envelope)
    payload = compile_input(envelope)
    scopes = {task["task_id"]: task for task in payload["other_task_scopes"]}
    assert set(scopes) == {"evaluate", "report", "other_design"}
    assert scopes["evaluate"]["task_type"] == "evaluate_existing_aso_set"
    assert scopes["report"]["task_type"] == "build_composite_report"
    assert scopes["evaluate"]["context"]["provided_asos"] == [{"id": "fixed"}]
    assert scopes["other_design"]["context"] == {"species": "mouse"}
    assert "structured_goal" not in payload["selected_task"]["context"]
    assert payload["related_dependencies"] == envelope["task_dependencies"]
    assert payload["request_evidence"] == "Original request and clarification evidence."
    assert envelope == original


def test_compiler_routes_outputs_by_verified_task_and_merged_alias():
    envelope = composite_request()
    payload = compile_input(envelope)
    assert payload["requested_final_outputs"] == [envelope["final_outputs"][0]]
    scopes = {task["task_id"]: task for task in payload["other_task_scopes"]}
    assert scopes["evaluate"]["requested_final_outputs"] == [envelope["final_outputs"][1]]
    assert scopes["report"]["requested_final_outputs"] == [envelope["final_outputs"][2]]
    assert scopes["other_design"]["requested_final_outputs"] == []


def test_compiler_does_not_invent_outputs_when_only_reporting_owns_delivery():
    envelope = composite_request()
    envelope["final_outputs"] = [envelope["final_outputs"][2]]
    assert compile_input(envelope)["requested_final_outputs"] == []
    # Unscoped user requirements remain available; scoping cannot discard them.
    envelope["final_outputs"].extend(["Explain limitations", {"kind": "methods"}])
    assert compile_input(envelope)["requested_final_outputs"] == [
        "Explain limitations", {"kind": "methods"},
    ]
