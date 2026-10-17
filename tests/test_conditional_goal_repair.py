"""Conditional execution requirements must not turn generated roles into user facts."""
from copy import deepcopy
import json

import pytest

from aso_agents import pipeline_tools as tools


ROOT = "/tasks/0/context/structured_goal/evaluation_targets/0"


def goal(target):
    return {
        "gene_symbol": "GENE1", "species": "human", "target_context": "cultured cells",
        "design_programs": [{"program_id": "binding", "label": "Binding pilot",
            "modality": "steric", "intent": "binding", "objective": "Cover the supplied site"}],
        "evaluation_targets": [target],
    }


def envelope(target):
    return {"schema_version": "v3", "tasks": [{"task_id": "design",
        "task_type": "design_new_asos", "label": "Binding pilot",
        "context": {"gene_symbol": "GENE1", "species": "human",
            "target_context": "cultured cells", "structured_goal": goal(target)}}],
        "task_dependencies": []}


@pytest.mark.parametrize("role,direction,expected", [
    ("identity_anchor", None, []), ("monitoring_safeguard", None, []),
    ("identity_anchor", "increase", ["desired_delta"]),
    ("monitoring_safeguard", "decrease", ["desired_delta"]),
    ("primary_outcome", None, ["desired_delta"]),
    ("route_target", None, ["desired_delta"]),
    ("primary_outcome", "increase", []), ("primary_outcome", "decrease", []),
    ("route_target", "increase", []), ("route_target", "decrease", []),
    ("primary_outcome", "neutral", ["desired_delta"]),
    ("identity_anchor", "neutral", ["desired_delta"]),
    ("not_a_role", None, ["objective_type"]),
])
def test_execution_gate_retains_directional_requirements(role, direction, expected):
    target = {"target_type": "regulatory_region", "objective_type": role}
    if direction is not None:
        target["desired_delta"] = direction
    original = deepcopy(target)
    assert tools._canonical_design_goal_contract_issues(goal(target)) == [
        "context.structured_goal.evaluation_targets[0]." + field for field in expected]
    assert target == original


def test_sparse_leaf_context_exposes_actual_condition_and_sibling_repair():
    value = envelope({"objective_type": "route_target", "target_type": "regulatory_region"})
    original = deepcopy(value)
    context = tools._canonical_goal_path_context(value, ROOT + "/desired_delta")
    assert context["ancestry"][-1]["schema_required"] is False
    condition, = context["execution_conditions"]
    assert condition["current_execution_issues"] == [ROOT + "/desired_delta"]
    assert condition["controlling_field"] == ROOT + "/objective_type"
    assert condition["dependent_field"] == ROOT + "/desired_delta"
    assert condition["alternatives"]["identity_anchor"]["desired_delta"] == "forbidden"
    assert condition["alternatives"]["route_target"]["desired_delta"] == "required"
    condition["current_object"]["objective_type"] = "identity_anchor"
    condition["alternatives"]["route_target"]["desired_delta"] = "forbidden"
    assert value == original
    fresh = tools._canonical_goal_path_context(value, ROOT + "/objective_type")
    assert fresh["execution_conditions"][0]["alternatives"]["route_target"]["desired_delta"] == "required"


@pytest.mark.parametrize("pointer", [
    "/tasks/0/context/structured_goal/design_programs/0/desired_delta",
    "/tasks/0/context/structured_goal/evaluation_targets/7/desired_delta",
    "/tasks/0/context/structured_goal/target_context",
])
def test_conditions_do_not_leak_across_objects_or_create_absent_targets(pointer):
    value = envelope({"objective_type": "identity_anchor"})
    original = deepcopy(value)
    assert tools._canonical_goal_path_context(value, pointer)["execution_conditions"] == []
    assert value == original


def test_mixed_directional_and_identity_targets_remain_independent():
    value = goal({"target_type": "region", "objective_type": "identity_anchor"})
    value["evaluation_targets"].append({"target_type": "gene",
        "objective_type": "primary_outcome", "desired_delta": "decrease"})
    assert tools._canonical_design_goal_contract_issues(value) == []
    del value["evaluation_targets"][1]["desired_delta"]
    assert tools._canonical_design_goal_contract_issues(value) == [
        "context.structured_goal.evaluation_targets[1].desired_delta"]


def test_existing_repair_loop_can_fix_controlling_role_without_direction_or_question(monkeypatch):
    request = "Cover the supplied regulatory region with a steric binding pilot in cultured cells; no expression direction is requested."
    value = envelope({"target_type": "regulatory_region", "objective_type": "primary_outcome"})
    original = deepcopy(value)
    calls = []
    unresolved = {"path": ROOT + "/desired_delta", "question": "What expression direction?",
                  "reason": "The generated primary outcome requires a direction.", "resolution": "user_input"}

    async def audit(prompt):
        protocol = tools._goal_audit_protocol_payload(prompt)
        calls.append(protocol)
        result = dict(status="valid", patches=[], goal_replacements=[], field_dispositions=[], unresolved=[])
        if len(calls) == 1:
            result.update(status="needs_clarification", unresolved=[unresolved])
        elif len(calls) == 2:
            assert protocol["unresolved_adjudication_feedback"][0]["path"] == ROOT + "/objective_type"
            result.update(status="repairable", patches=[{
                "op": "replace", "path": ROOT + "/objective_type", "value": "identity_anchor",
                "evidence_refs": ["request"], "evidence_quote": "no expression direction is requested",
                "reason": "The supplied region establishes target identity, not an abundance outcome.",
            }])
        return result

    def adjudicate(payload):
        context = payload["proposed_unresolved"][0]["canonical_path_context"]
        condition, = context["execution_conditions"]
        assert condition["current_object"]["objective_type"] == "primary_outcome"
        assert condition["current_execution_issues"] == [ROOT + "/desired_delta"]
        return {"verdicts": [{"unresolved_index": 0, "disposition": "repair_from_evidence",
            "canonical_path": condition["controlling_field"], "evidence_value_json": json.dumps("identity_anchor"),
            "reason": "The supplied region establishes target identity, not an abundance outcome."}]}

    monkeypatch.setattr(tools, "_run_goal_contract_audit_agent_once", audit)
    monkeypatch.setattr(tools, "_run_goal_unresolved_adjudication_sync", adjudicate)
    result = tools._repair_request_envelope_if_needed(request_text=request, run_dir="",
        structured_goal={}, request_envelope=value, qa_history=[], semantic_audit=True)
    assert result["status"] == "ready", result
    assert len(calls) == 3
    target = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]["evaluation_targets"][0]
    assert target["objective_type"] == "identity_anchor"
    assert "desired_delta" not in target
    assert value == original
