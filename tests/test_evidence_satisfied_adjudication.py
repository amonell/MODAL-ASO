"""A rejected clarification must not force a mutation of an already correct value."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as tools


def _verdict(value, *, disposition="repair_from_evidence", path="/tasks/0/context/target_context"):
    return {"unresolved_index": 0, "disposition": disposition,
            "canonical_path": path, "evidence_value_json": value,
            "reason": "The declared evidence determines the required value."}


def _payload(context):
    return {"current_request_envelope": {"tasks": [{"context": context}]},
            "proposed_unresolved": [{"path": "/tasks/0/context/target_context",
                "question": "Should the declared context be retained?",
                "reason": "The audit proposed an unnecessary choice.",
                "resolution": "user_input"}],
            "evidence_sources": [{"evidence_id": "request", "text": "Use cultured cells."}]}


@pytest.mark.parametrize("current,expected", [
    ("cultured cells", "cultured cells"),
    ({"anchor": "left", "range": [-2, 5]}, {"range": [-2, 5], "anchor": "left"}),
    (None, None),
    (["a", "b"], ["a", "b"]),
])
def test_evidence_value_already_present_is_nonblocking_without_another_call(monkeypatch, current, expected):
    payload = _payload({"target_context": current})
    before = deepcopy(payload)
    raw = {"verdicts": [_verdict(json.dumps(expected))]}
    calls = []

    async def run(_agent, *, input, **kwargs):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(raw))

    monkeypatch.setattr(tools.Runner, "run", run)
    result = asyncio.run(tools._run_goal_unresolved_adjudication_agent_once(payload))
    assert len(calls) == 1
    assert result["verdicts"][0]["disposition"] == "nonblocking"
    assert result["verdicts"][0]["canonical_path"] is None
    assert result["verdicts"][0]["evidence_value_json"] is None
    assert payload == before
    assert raw["verdicts"][0]["disposition"] == "repair_from_evidence"


@pytest.mark.parametrize("context,expected", [
    ({"target_context": "wrong context"}, "cultured cells"),
    ({}, "cultured cells"),
    ({}, None),
    ({"target_context": {"enabled": True}}, {"enabled": 1}),
    ({"target_context": ["a", "b"]}, ["b", "a"]),
])
def test_different_or_absent_values_still_require_repair(monkeypatch, context, expected):
    raw = {"verdicts": [_verdict(json.dumps(expected))]}

    async def run(_agent, *, input, **kwargs):
        return SimpleNamespace(final_output=json.dumps(raw))

    monkeypatch.setattr(tools.Runner, "run", run)
    result = asyncio.run(tools._run_goal_unresolved_adjudication_agent_once(_payload(context)))
    assert result == raw


@pytest.mark.parametrize("encoded", [None, "", "unquoted", '{"x":1,"x":2}', "NaN", "Infinity", "1e999"])
def test_repair_requires_an_unambiguous_json_evidence_value(encoded):
    errors = tools._goal_unresolved_adjudication_contract_errors(
        {"verdicts": [_verdict(encoded)]}, item_count=1,
        schema=tools._goal_unresolved_adjudication_schema(1))
    assert any("evidence_value_json" in error for error in errors)


@pytest.mark.parametrize("disposition", ["nonblocking", "user_input", "source_retrieval", "remove_model_authored_field"])
def test_other_dispositions_cannot_carry_an_unapplied_replacement(disposition):
    path = "/tasks/0/context/target_context" if disposition == "remove_model_authored_field" else None
    schema = tools._goal_unresolved_adjudication_schema(1)
    assert tools._goal_unresolved_adjudication_contract_errors(
        {"verdicts": [_verdict('"cultured cells"', disposition=disposition, path=path)]},
        item_count=1, schema=schema)
    assert not tools._goal_unresolved_adjudication_contract_errors(
        {"verdicts": [_verdict(None, disposition=disposition, path=path)]},
        item_count=1, schema=schema)


@pytest.mark.parametrize("disposition", ["nonblocking", "user_input", "source_retrieval"])
def test_schema_exposes_the_nonrepair_null_pointer_rule(disposition):
    schema = tools._goal_unresolved_adjudication_schema(1)
    malformed = {"verdicts": [_verdict(None, disposition=disposition)]}
    assert tools._json_schema_validation_errors(malformed, schema)
    malformed["verdicts"][0]["canonical_path"] = None
    assert not tools._json_schema_validation_errors(malformed, schema)


def _envelope():
    return {"schema_version": "v3", "tasks": [{"task_id": "evaluate",
        "task_type": "evaluate_existing_aso_set", "label": "Evaluate provided molecule",
        "context": {"gene_symbol": "GENE1", "species": "human",
            "target_context": "cultured cells", "provided_asos": ["ACGTACGTACGTACGTACGT"]}}],
        "task_dependencies": []}


@pytest.mark.parametrize("missing_required_input", [False, True])
def test_complete_audit_preserves_satisfied_value_without_bypassing_host_validation(monkeypatch, missing_required_input):
    envelope = _envelope()
    if missing_required_input:
        del envelope["tasks"][0]["context"]["provided_asos"]
    original = deepcopy(envelope)
    audit_calls = []

    async def audit(prompt):
        audit_calls.append(prompt)
        return {"status": "needs_clarification", "patches": [], "goal_replacements": [],
                "field_dispositions": [], "unresolved": _payload({})["proposed_unresolved"]}

    async def run(_agent, *, input, **kwargs):
        return SimpleNamespace(final_output=json.dumps({"verdicts": [_verdict('"cultured cells"')]}))

    monkeypatch.setattr(tools, "_run_goal_contract_audit_agent_once", audit)
    monkeypatch.setattr(tools.Runner, "run", run)
    result = tools._repair_request_envelope_if_needed(
        request_text="Evaluate the provided molecule in cultured cells.", run_dir="",
        structured_goal={}, request_envelope=envelope, qa_history=[], semantic_audit=True)
    assert envelope == original
    if missing_required_input:
        assert result["status"] != "ready"
    else:
        assert result["status"] == "ready"
        assert result["request_envelope"]["tasks"][0]["context"]["target_context"] == "cultured cells"
        assert len(audit_calls) == 1
