from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from aso_agents import pipeline_tools as pipeline


def goal():
    return {"gene_symbol": "NEUTRAL", "species": "human", "target_context": "cultured cells",
            "design_programs": [{"program_id": "p", "label": "p", "modality": "gapmer",
                                 "intent": "knockdown", "objective": "reduce the intended RNA",
                                 "desired_delta": "decrease", "transcript_id": "REF_A.1",
                                 "transcript_ids": ["REF_A.1", "REF_B.2"]}],
            "preservation_targets": [{"action": "preserve", "preservation_target_type": "transcript",
                                      "transcript_id": "REF_B.2"}]}


def issues(value):
    return pipeline._canonical_design_goal_contract_issues(value)


@pytest.mark.parametrize("field,value", [("transcript_ids", ["REF_A.1", "REF_B.2"]),
                                          ("transcript_id", "REF_B.2"),
                                          ("isoform_label", "REF_B.2")])
def test_knockdown_target_cannot_also_be_a_declared_preserved_transcript(field, value):
    current = goal()
    current["design_programs"][0].pop("transcript_ids")
    current["design_programs"][0][field] = value
    before = deepcopy(current)
    assert f"context.structured_goal.design_programs[0].{field}" in issues(current)
    assert current == before


@pytest.mark.parametrize("variant", ["distinct", "different_version", "other_gene", "steric", "unresolved_alias"])
def test_valid_or_unresolved_identity_roles_are_not_silently_reinterpreted(variant):
    current = goal()
    if variant == "distinct":
        current["design_programs"][0]["transcript_ids"] = ["REF_A.1", "REF_C.1"]
    elif variant == "different_version":
        current["preservation_targets"][0]["transcript_id"] = "REF_B.3"
    elif variant == "other_gene":
        current["preservation_targets"][0]["gene_symbol"] = "OTHER"
    elif variant == "steric":
        current["design_programs"][0].update(modality="steric", intent="binding")
    else:
        current["preservation_targets"][0]["transcript_id"] = "REF_B"
    assert "context.structured_goal.design_programs[0].transcript_ids" not in issues(current)


def test_case_and_whitespace_do_not_hide_an_exact_declared_role_conflict():
    current = goal()
    current["preservation_targets"][0]["transcript_id"] = " ref_b.2 "
    assert "context.structured_goal.design_programs[0].transcript_ids" in issues(current)


def test_conflict_is_bound_to_its_program_and_repair_evidence():
    current = goal()
    second = deepcopy(current["design_programs"][0])
    second.update(program_id="other", transcript_ids=["REF_C.1", "REF_D.1"], transcript_id="REF_C.1")
    current["design_programs"].append(second)
    envelope = {"tasks": [{"task_id": "design", "task_type": "design_new_asos",
                            "context": {"structured_goal": current}}]}
    validation = pipeline._request_validation_summary(envelope)
    assert "context.structured_goal.design_programs[0].transcript_ids" in validation["tasks"]["design"]["repairable_issues"]
    records = pipeline._goal_contract_issue_records(envelope, validation)
    record, = [r for r in records if r["json_pointer"].endswith("/transcript_ids")]
    assert record["json_pointer"] == "/tasks/0/context/structured_goal/design_programs/0/transcript_ids"
    conflict = record["semantic_conflict"]
    assert conflict["preservation_paths"] == ["preservation_targets[0].transcript_id"]
    assert "preserv" in conflict["reason"].lower()
    validator = Draft202012Validator(record["expected_schema"])
    assert list(validator.iter_errors(["REF_A.1", "REF_B.2"]))
    assert not list(validator.iter_errors(["REF_A.1"]))
    assert current["design_programs"][0]["transcript_ids"] == ["REF_A.1", "REF_B.2"]


def test_evidenced_repair_retains_preservation_and_revalidates_the_goal():
    current = goal()
    envelope = {"tasks": [{"task_id": "design", "task_type": "design_new_asos",
                            "context": {"structured_goal": current}}]}
    evidence = "Target REF_A.1 and preserve REF_B.2."
    result = pipeline._apply_goal_audit_patches(envelope, [{
        "op": "remove", "path": "/tasks/0/context/structured_goal/design_programs/0/transcript_ids/1",
        "value": None, "evidence_refs": ["request"], "evidence_quote": evidence,
        "reason": "The second accession is a protected reference, not a knockdown target.",
    }], issue_records=[], evidence_sources=[{"evidence_id": "request", "kind": "user_request", "text": evidence}])
    assert not result["rejected"]
    repaired = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert repaired["preservation_targets"] == current["preservation_targets"]
    assert repaired["design_programs"][0]["transcript_ids"] == ["REF_A.1"]
    assert "context.structured_goal.design_programs[0].transcript_ids" not in issues(repaired)
    assert current["design_programs"][0]["transcript_ids"] == ["REF_A.1", "REF_B.2"]
