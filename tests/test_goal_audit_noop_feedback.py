from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools


def audit_fixture(*, open_issue=True):
    pointer = "/tasks/0/context/structured_goal/candidate_limit"
    envelope = {"tasks": [{
        "task_id": "design", "task_type": "design_new_asos",
        "context": {"structured_goal": {"candidate_limit": 3}},
    }]}
    prompt = pipeline_tools._goal_audit_prompt(
        request_envelope=envelope,
        validation_issues=["candidate_limit"] if open_issue else [],
        issue_records=[{
            "json_pointer": pointer, "required_operation": "replace",
            "expected_schema": {"type": "integer", "minimum": 1},
        }] if open_issue else [],
        evidence_sources=[{
            "evidence_id": "request", "kind": "user_request",
            "text": "Return seven candidates." if open_issue else "Return three candidates.",
        }],
        semantic_audit=not open_issue,
    )
    spans = pipeline_tools._evidence_span_map(
        pipeline_tools._declared_evidence_sources_from_prompt(prompt)
    )
    response = {
        "status": "repairable", "patches": [{
            "op": "replace", "path": pointer, "value": 3,
            "evidence_refs": ["request"], "evidence_span_id": next(iter(spans)),
            "reason": "Preserve the requested candidate count.",
        }],
        "goal_replacements": [], "field_dispositions": [], "unresolved": [],
    }
    return prompt, response, envelope


@pytest.mark.parametrize("operation", ["replace", "remove"])
def test_ineffective_audit_repair_receives_path_specific_feedback(monkeypatch, operation):
    prompt, first, envelope = audit_fixture()
    rejected = deepcopy(first)
    if operation == "remove":
        rejected["patches"][0].update(
            op="remove", path="/tasks/0/context/structured_goal/absent_optional_field",
            value=None,
        )
    calls = []

    async def run(_agent, *, input, max_steps):
        calls.append(input)
        if len(calls) == 1:
            return SimpleNamespace(final_output=json.dumps(rejected))
        errors = input.split("Validation errors:\n", 1)[1].split("\n\n", 1)[0]
        assert rejected["patches"][0]["path"] in errors
        assert "does not change the envelope" in errors
        assert "Changing only status cannot repair it" in errors
        assert "Do not invent a different scientific value" in errors
        corrected = deepcopy(first)
        corrected["patches"][0]["value"] = 7
        return SimpleNamespace(final_output=json.dumps(corrected))

    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    result = asyncio.run(pipeline_tools._run_goal_contract_audit_agent_once(prompt))

    assert len(calls) == 2
    assert result["status"] == "repairable"
    assert result["patches"][0]["value"] == 7
    assert result["patches"][0]["evidence_quote"] == "Return seven candidates."
    assert envelope["tasks"][0]["context"]["structured_goal"]["candidate_limit"] == 3
    assert first["patches"][0]["value"] == 3


def test_repeated_noop_never_discharges_open_issue_and_stays_bounded(monkeypatch):
    prompt, response, _ = audit_fixture()
    calls = []

    async def run(_agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(response))

    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    result = asyncio.run(pipeline_tools._run_goal_contract_audit_agent_once(prompt))

    assert len(calls) == 4
    assert result["error"] == "invalid_structured_output"
    assert "does not change the envelope" in result["reason"]
    assert "already present" in result["reason"]


def test_harmless_noop_with_no_open_issue_remains_valid_without_retry(monkeypatch):
    prompt, response, _ = audit_fixture(open_issue=False)
    calls = []

    async def run(_agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(response))

    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    result = asyncio.run(pipeline_tools._run_goal_contract_audit_agent_once(prompt))

    assert len(calls) == 1
    assert result["status"] == "valid"
    assert result["patches"] == []
