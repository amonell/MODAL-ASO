"""Prompt repair options agree with the actual canonical patch schema."""
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from aso_agents import pipeline_tools as pipeline


@pytest.mark.parametrize("field,present,required,expected", [
    ("candidate_role_selection", True, False, ["replace", "remove"]),
    ("candidate_panel_objectives", True, False, ["replace", "remove"]),
    ("candidate_panel_objectives", True, True, ["replace"]),
    ("candidate_panel_objectives", False, False, ["add"]),
    ("design_programs", True, True, ["replace"]),
    ("design_programs/0/objective", True, False, ["replace"]),
])
def test_offered_issue_operations_match_requiredness_existence_and_schema(field, present, required, expected):
    goal = {"design_programs": [{"program_id": "pool", "label": "pool", "modality": "steric",
             "intent": "binding", "objective": "sequence library"}]}
    if present and field not in {"design_programs", "design_programs/0/objective"}:
        goal[field] = ({} if field == "candidate_role_selection" else
                       [{"metric": "longest_homopolymer", "aggregate": "maximum", "direction": "minimize"}])
    envelope = {"tasks": [{"task_id": "design", "task_type": "design_new_asos",
                          "context": {"structured_goal": goal}}]}
    pointer = "/tasks/0/context/structured_goal/" + field
    record = {"task_id": "design", "json_pointer": pointer,
              "validation_issue": "context.structured_goal." + field,
              "required_operation": "replace" if present else "add",
              "expected_schema": {"type": "string"} if field.endswith("objective") else {}}
    before = deepcopy((envelope, record))
    prompt = pipeline._goal_audit_prompt(
        request_envelope=envelope, validation_issues=[record["validation_issue"]],
        issue_records=[record], evidence_sources=[], semantic_audit=False,
        required_goal_fields=[field] if required else [],
    )
    protocol = pipeline._goal_audit_protocol_payload(prompt)
    assert protocol["canonical_contract_issues"][0]["allowed_operations"] == expected
    schema = pipeline._goal_contract_audit_schema_for_prompt(prompt)["properties"]["patches"]
    patch = {"op": "remove", "path": pointer, "value": None, "evidence_refs": ["request"],
             "evidence_quote": "An evidence quote.", "reason": "Resolve unsupported optional scope."}
    assert (not list(Draft202012Validator(schema).iter_errors([patch]))) == ("remove" in expected)
    assert (envelope, record) == before
