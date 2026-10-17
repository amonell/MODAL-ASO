"""Contradictory selection algorithms must be repairable before design runs."""
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from aso_agents import pipeline_tools as pipeline
from aso_agents.role_selection import select_ordered_roles


def goal():
    return {
        "gene_symbol": "NEUTRAL", "species": "human", "target_context": "cultured cells",
        "design_programs": [{
            "program_id": "pool", "label": "requested pool", "modality": "steric",
            "intent": "binding", "objective": "a requested sequence library", "length": 17,
        }],
        "candidate_minimum": 2, "candidate_limit": 2,
        "candidate_selection_policy": "nonoverlapping_target_sites",
        "candidate_role_selection": {
            "reuse_policy": "distinct_sequences", "roles": [
                {"role_id": role, "program_id": "pool", "count": 1,
                 "criteria": [{"metric": "longest_homopolymer", "direction": "minimize"}]}
                for role in ["first", "second"]
            ],
        },
        "candidate_panel_objectives": [
            {"metric": "longest_homopolymer", "aggregate": "maximum", "direction": "minimize"}
        ],
    }


def envelope(value):
    return {"tasks": [{"task_id": "design", "task_type": "design_new_asos",
                        "context": {"structured_goal": value}},
                       {"task_id": "report", "task_type": "summarize_results",
                        "depends_on": ["design"],
                        "context": {"ranking_request": "Include individual ranks for reference."}}]}


def conflict_records(request):
    validation = pipeline._request_validation_summary(request)
    return [record for record in pipeline._goal_contract_issue_records(request, validation)
            if record.get("semantic_conflict", {}).get("code") ==
            "joint_and_sequential_ranking_scope_conflict"]


def test_runtime_scope_failure_is_visible_in_intake_with_both_repair_paths():
    current = goal()
    before = deepcopy(current)
    request = envelope(current)
    records = conflict_records(request)
    assert {r["json_pointer"] for r in records} == {
        "/tasks/0/context/structured_goal/candidate_role_selection",
        "/tasks/0/context/structured_goal/candidate_panel_objectives",
    }
    assert all(r["required_operation"] == "replace" for r in records)
    assert all(r["expected_schema"].get("type") in {"object", "array"} for r in records)
    selected, receipt = select_ordered_roles([], current, lambda row: [])
    assert selected == []
    assert receipt["reason"] == "joint_and_sequential_ranking_scope_conflict"
    assert current == before


@pytest.mark.parametrize("keep", ["candidate_panel_objectives", "candidate_role_selection"])
def test_evidence_bound_repair_supports_either_algorithm_without_changing_other_requirements(keep):
    request = envelope(goal())
    before = deepcopy(request)
    records = conflict_records(request)
    discard = next(field for field in ("candidate_role_selection", "candidate_panel_objectives")
                   if field != keep)
    quote = ("Choose the complete pair jointly; individual ranks are for reference only."
             if keep == "candidate_panel_objectives" else
             "Select first, then second from the remaining sequences; do not optimize the pair jointly.")
    sources = [{"evidence_id": "request", "kind": "user_request", "text": quote}]
    patch = {"op": "remove", "path": "/tasks/0/context/structured_goal/" + discard,
             "value": None, "evidence_refs": ["request"], "evidence_quote": quote,
             "reason": "The explicit request establishes the final-selection algorithm."}
    prompt = pipeline._goal_audit_prompt(
        request_envelope=request, validation_issues=[], issue_records=records,
        evidence_sources=sources, semantic_audit=False,
    )
    protocol = pipeline._goal_audit_protocol_payload(prompt)
    assert protocol["canonical_contract_issues"] == records
    schema = pipeline._goal_contract_audit_schema_for_prompt(prompt)
    assert not list(Draft202012Validator(schema["properties"]["patches"]).iter_errors([patch]))
    result = pipeline._apply_goal_audit_patches(
        request, [patch], issue_records=records, evidence_sources=sources,
    )
    assert not result["rejected"]
    expected = deepcopy(before)
    del expected["tasks"][0]["context"]["structured_goal"][discard]
    assert result["request_envelope"] == expected
    assert not conflict_records(expected)
    assert request == before


@pytest.mark.parametrize("missing", ["candidate_role_selection", "candidate_panel_objectives", "both"])
def test_single_or_default_selection_is_not_a_conflict(missing):
    current = goal()
    for key in ("candidate_role_selection", "candidate_panel_objectives"):
        if missing in {key, "both"}:
            current.pop(key)
    assert not conflict_records(envelope(current))


def test_independent_task_scopes_can_use_different_algorithms():
    request = envelope(goal())
    second = deepcopy(request["tasks"][0])
    second["task_id"] = "other_design"
    request["tasks"].append(second)
    request["tasks"][0]["context"]["structured_goal"].pop("candidate_role_selection")
    second["context"]["structured_goal"].pop("candidate_panel_objectives")
    assert not conflict_records(request)


@pytest.mark.parametrize("discard", ["candidate_role_selection", "candidate_panel_objectives"])
def test_unverified_evidence_cannot_delete_either_algorithm(discard):
    request = envelope(goal())
    before = deepcopy(request)
    result = pipeline._apply_goal_audit_patches(request, [{
        "op": "remove", "path": "/tasks/0/context/structured_goal/" + discard,
        "value": None, "evidence_refs": ["request"], "evidence_quote": "Invented choice.",
        "reason": "Resolve a conflict.",
    }], issue_records=conflict_records(request),
        evidence_sources=[{"evidence_id": "request", "kind": "user_request",
                           "text": "Produce a library and discuss both algorithms."}])
    assert result["rejected"]
    assert result["request_envelope"] == before
    assert request == before


def test_replacing_conflicting_contract_with_itself_does_not_clear_intake_failure():
    request = envelope(goal())
    quote = "The selection scope needs to be resolved."
    result = pipeline._apply_goal_audit_patches(request, [{
        "op": "replace", "path": "/tasks/0/context/structured_goal/candidate_role_selection",
        "value": deepcopy(request["tasks"][0]["context"]["structured_goal"]["candidate_role_selection"]),
        "evidence_refs": ["request"], "evidence_quote": quote, "reason": "Unchanged declaration.",
    }], issue_records=conflict_records(request),
        evidence_sources=[{"evidence_id": "request", "kind": "user_request", "text": quote}])
    assert len(conflict_records(result["request_envelope"])) == 2
