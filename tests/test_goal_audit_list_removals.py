from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as pipeline


ROOT = "/tasks/0/context/structured_goal"
EVIDENCE = [{"evidence_id": "request", "kind": "user_request",
             "text": "Keep the original constraint and remove its redundant copies."}]


def envelope(goal):
    return {"tasks": [{"task_id": "design", "task_type": "design_new_asos",
                        "context": {"structured_goal": goal}}]}


def removal(path):
    return {"op": "remove", "path": path, "value": None,
            "evidence_refs": ["request"], "evidence_quote": EVIDENCE[0]["text"],
            "reason": "Remove the redundant entry while retaining the original."}


@pytest.mark.parametrize("field,items", [
    ("region_constraints", [{"constraint_id": "original"}, {"constraint_id": "duplicate"}]),
    ("preservation_targets", [{"label": "original"}, {"label": "duplicate"}]),
    ("off_target_species", ["human", "human"]),
])
def test_schema_offers_existing_optional_list_members(field, items):
    current = envelope({field: items})
    before = deepcopy(current)
    paths = pipeline._canonical_optional_remove_paths(current)
    assert f"{ROOT}/{field}/0" in paths
    assert f"{ROOT}/{field}/1" in paths
    assert f"{ROOT}/{field}/2" not in paths
    assert current == before


def test_nested_member_removal_does_not_offer_required_program_fields():
    current = envelope({"design_programs": [{
        "program_id": "only-program", "region_constraints": [
            {"constraint_id": "original"}, {"constraint_id": "duplicate"}],
    }]})
    paths = pipeline._canonical_optional_remove_paths(current)
    assert f"{ROOT}/design_programs/0/region_constraints/1" in paths
    assert f"{ROOT}/design_programs/0" not in paths
    assert f"{ROOT}/design_programs" not in paths
    assert f"{ROOT}/design_programs/0/program_id" not in paths


@pytest.mark.parametrize("count,allowed", [(1, False), (2, True)])
def test_required_collection_keeps_its_minimum_member_count(count, allowed):
    current = envelope({"design_programs": [{"program_id": str(i)} for i in range(count)]})
    paths = pipeline._canonical_optional_remove_paths(current)
    assert (f"{ROOT}/design_programs/0" in paths) is allowed


def test_required_fields_and_unknown_collections_still_cannot_be_removed():
    current = envelope({"candidate_minimum": 2, "unknown_list": ["unrecognized"]})
    paths = pipeline._canonical_optional_remove_paths(current, required_goal_fields=["candidate_minimum"])
    for path in [ROOT, f"{ROOT}/candidate_minimum", f"{ROOT}/unknown_list/0", "/tasks/0"]:
        assert path not in paths


def test_audit_accepts_one_evidence_bound_list_removal_without_schema_retry(monkeypatch):
    current = envelope({"region_constraints": [
        {"constraint_id": "original", "operator": "include"},
        {"constraint_id": "duplicate", "operator": "include"},
    ]})
    before = deepcopy(current)
    path = f"{ROOT}/region_constraints/1"
    prompt = pipeline._goal_audit_prompt(
        request_envelope=current, validation_issues=[], issue_records=[],
        evidence_sources=EVIDENCE, semantic_audit=True,
        unresolved_feedback=[{"path": path, "required_action": "remove_model_authored_field",
                              "reason": "The second constraint redundantly repeats the first."}],
    )
    spans = pipeline._evidence_span_map(pipeline._declared_evidence_sources_from_prompt(prompt))
    patch = removal(path)
    patch.pop("evidence_quote")
    patch["evidence_span_id"] = next(iter(spans))
    response = {"status": "repairable", "patches": [patch], "goal_replacements": [],
                "field_dispositions": [], "unresolved": []}
    calls = []

    async def run(_agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(response))

    monkeypatch.setattr(pipeline.Runner, "run", run)
    result = asyncio.run(pipeline._run_goal_contract_audit_agent_once(prompt))
    assert len(calls) == 1
    assert result["status"] == "repairable"
    assert result["patches"][0]["evidence_quote"] == EVIDENCE[0]["text"]
    applied = pipeline._apply_goal_audit_patches(
        current, result["patches"], issue_records=[], evidence_sources=EVIDENCE,
    )
    assert not applied["rejected"]
    assert applied["request_envelope"]["tasks"][0]["context"]["structured_goal"]["region_constraints"] == before["tasks"][0]["context"]["structured_goal"]["region_constraints"][:1]
    assert current == before


def test_multiple_member_removals_preserve_original_index_meaning():
    current = envelope({"off_target_species": ["human", "mouse", "human", "rat"]})
    before = deepcopy(current)
    result = pipeline._apply_goal_audit_patches(
        current, [removal(f"{ROOT}/off_target_species/0"), removal(f"{ROOT}/off_target_species/2")],
        issue_records=[], evidence_sources=EVIDENCE,
    )
    assert not result["rejected"]
    assert result["request_envelope"]["tasks"][0]["context"]["structured_goal"]["off_target_species"] == ["mouse", "rat"]
    assert current == before


def test_executor_still_rejects_removing_all_required_members():
    current = envelope({"design_programs": [{"program_id": "first"}, {"program_id": "second"}]})
    result = pipeline._apply_goal_audit_patches(
        current, [removal(f"{ROOT}/design_programs/0"), removal(f"{ROOT}/design_programs/1")],
        issue_records=[], evidence_sources=EVIDENCE,
    )
    assert len(result["applied"]) == len(result["rejected"]) == 1
    assert result["request_envelope"]["tasks"][0]["context"]["structured_goal"]["design_programs"] == [{"program_id": "first"}]


def test_executor_still_requires_verified_evidence_for_member_removal():
    current = envelope({"off_target_species": ["human", "mouse"]})
    patch = removal(f"{ROOT}/off_target_species/1")
    patch["evidence_quote"] = "Invented instruction absent from the request."
    result = pipeline._apply_goal_audit_patches(current, [patch], issue_records=[], evidence_sources=EVIDENCE)
    assert not result["applied"]
    assert result["request_envelope"] == current
    assert result["rejected"][0]["reason"] == "evidence_not_verified"
