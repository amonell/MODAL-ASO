"""Chemistry contradictions must reach intake repair before design tools run."""
from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline


def goal():
    return {
        "gene_symbol": "NEUTRAL", "species": "human", "target_context": "cultured cells",
        "design_programs": [{
            "program_id": "panel", "label": "panel", "modality": "gapmer",
            "intent": "knockdown", "objective": "reduce the intended RNA",
            "desired_delta": "decrease", "length": 18,
        }],
    }


def envelope(value):
    return {"tasks": [{"task_id": "design", "task_type": "design_new_asos",
                        "context": {"structured_goal": value}}]}


@pytest.mark.parametrize("scope", ["goal", "program"])
@pytest.mark.parametrize("chemistry", ["4-10-4", {"recommendation": {"CHEMISTRY": "4-10-4"}}])
def test_runtime_conflict_is_repairable_at_intake_without_reinterpreting_input(scope, chemistry):
    current = goal()
    selected = current if scope == "goal" else current["design_programs"][0]
    selected.update(chemistry=chemistry, chemistry_policy="leave_unspecified")
    before = deepcopy(current)
    request = envelope(current)
    validation = pipeline._request_validation_summary(request)
    path = "chemistry_policy" if scope == "goal" else "design_programs[0].chemistry_policy"
    assert "context.structured_goal." + path in validation["tasks"]["design"]["repairable_issues"]
    records = pipeline._goal_contract_issue_records(request, validation)
    record, = [row for row in records if row["canonical_path"] == "context.structured_goal." + path]
    pointer = "/tasks/0/context/structured_goal/" + ("" if scope == "goal" else "design_programs/0/")
    assert record["json_pointer"] == pointer + "chemistry_policy"
    assert record["semantic_conflict"]["explicit_chemistry"] == chemistry
    assert record["semantic_conflict"]["chemistry_path"] == path.replace("chemistry_policy", "chemistry")
    assert current == before
    with pytest.raises(ValueError, match="conflicts"):
        pipeline._ensure_program_chemistry_profiles(structured_goal=deepcopy(current), final_payload={})


@pytest.mark.parametrize("defer", [False, True])
def test_existing_evidence_bound_repair_can_resolve_either_user_intent(defer):
    current = goal()
    current.update(chemistry="4-10-4", chemistry_policy="leave_unspecified")
    request = envelope(current)
    records = pipeline._goal_contract_issue_records(request, pipeline._request_validation_summary(request))
    evidence = "Leave chemistry unspecified." if defer else "Use 4-10-4 chemistry."
    patch = {
        "op": "remove" if defer else "replace",
        "path": "/tasks/0/context/structured_goal/" + ("chemistry" if defer else "chemistry_policy"),
        "value": None if defer else "select", "evidence_refs": ["request"],
        "evidence_quote": evidence, "reason": "Resolve the conflicting declaration from the explicit request.",
    }
    protocol = {"canonical_contract_issues": records, "current_request_envelope": request}
    schema = pipeline._goal_contract_audit_schema_for_prompt(json.dumps(protocol))
    assert not list(Draft202012Validator(schema["properties"]["patches"]).iter_errors([patch]))
    result = pipeline._apply_goal_audit_patches(
        request, [patch], issue_records=records,
        evidence_sources=[{"evidence_id": "request", "kind": "user_request", "text": evidence}],
    )
    assert not result["rejected"]
    repaired = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert not pipeline._chemistry_policy_conflicts(repaired)
    assert "context.structured_goal.chemistry_policy" not in pipeline._canonical_design_goal_contract_issues(repaired)
    profiles = pipeline._ensure_program_chemistry_profiles(structured_goal=repaired, final_payload={})
    assert profiles["by_program"]["panel"]["recommendation"]["CHEMISTRY"] == ("unspecified" if defer else "4-10-4")
    assert current["chemistry"] == "4-10-4" and current["chemistry_policy"] == "leave_unspecified"


@pytest.mark.parametrize("variant", ["missing", "selected", "deferred", "unspecified", "payload"])
def test_consistent_or_omitted_declarations_are_not_chemistry_conflicts(variant):
    current = goal()
    current.update({
        "missing": {}, "selected": {"chemistry": "4-10-4", "chemistry_policy": "select"},
        "deferred": {"chemistry_policy": "leave_unspecified"},
        "unspecified": {"chemistry": "unspecified", "chemistry_policy": "leave_unspecified"},
        "payload": {"chemistry": {"recommendation": {"CHEMISTRY": "unspecified"}},
                    "chemistry_policy": "leave_unspecified"},
    }[variant])
    assert not pipeline._chemistry_policy_conflicts(current)
    assert not [p for p in pipeline._canonical_design_goal_contract_issues(current) if p.endswith("chemistry_policy")]


@pytest.mark.parametrize("defer_program", [False, True])
def test_program_override_of_inherited_policy_or_chemistry_remains_valid(defer_program):
    current = goal()
    current.update({"chemistry": "4-10-4"} if defer_program else {"chemistry_policy": "leave_unspecified"})
    current["design_programs"][0].update(
        {"chemistry_policy": "leave_unspecified"} if defer_program else {"chemistry": "4-10-4"}
    )
    assert not pipeline._chemistry_policy_conflicts(current)
    resolved = runtime._design_program_goal_copy(current, current["design_programs"][0])
    assert not pipeline._chemistry_policy_conflicts(resolved)
    for _ in range(2):
        profiles = pipeline._ensure_program_chemistry_profiles(structured_goal=current, final_payload={})
        assert profiles["by_program"]["panel"]["recommendation"]["CHEMISTRY"] == ("unspecified" if defer_program else "4-10-4")
        assert not pipeline._chemistry_policy_conflicts(current)
