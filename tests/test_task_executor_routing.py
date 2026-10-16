"""A redundant routing flag cannot bypass an operation or waste plan repair."""

import asyncio
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.goal_contract import CANONICAL_TASK_TYPES


def proposal(task_type="analyze_artifact", native=False):
    return {
        "status": "ready", "questions": [], "reason": None,
        "request_envelope": {
            "tasks": [{"task_id": "operation", "task_type": task_type,
                       "native": native, "label": "Requested operation",
                       "context": {"artifact_refs": [{"path": "/input/table.csv"}]}}],
            "task_dependencies": [], "final_outputs": [],
        },
    }


@pytest.mark.parametrize("task_type", sorted(CANONICAL_TASK_TYPES))
def test_supported_operation_keeps_science_and_uses_its_registered_executor(task_type):
    expected_native = task_type != "sandbox_analysis"
    raw = proposal(task_type, not expected_native)
    before = deepcopy(raw)
    bound = pipeline._bind_request_task_executors(raw)
    expected = deepcopy(before)
    expected["request_envelope"]["tasks"][0]["native"] = expected_native
    assert bound == expected
    assert raw == before
    assert pipeline._bind_request_task_executors(bound) == bound
    bound["request_envelope"]["tasks"][0]["context"].clear()
    assert raw == before  # Never rewrite the provider evidence in place.


@pytest.mark.parametrize("task_type", ["unrecognized", "", None, ["analyze_artifact"]])
def test_unknown_operation_is_not_silently_converted(task_type):
    raw = proposal(task_type)
    assert pipeline._bind_request_task_executors(raw) == raw


@pytest.mark.parametrize("value", [None, "false", 0])
def test_malformed_execution_metadata_is_left_for_schema_validation(value):
    raw = proposal(native=value)
    assert pipeline._bind_request_task_executors(raw) == raw


def test_legacy_missing_metadata_retains_existing_default_handling():
    raw = proposal()
    del raw["request_envelope"]["tasks"][0]["native"]
    assert pipeline._bind_request_task_executors(raw) == raw


@pytest.mark.parametrize("raw", [
    {"error": "provider_failed"},
    {"status": "out_of_scope", "request_envelope": None},
    {"status": "need_clarification", "questions": ["Which input?"], "request_envelope": None},
])
def test_non_graph_responses_stay_unchanged(raw):
    assert pipeline._bind_request_task_executors(raw) == raw


def install_reviews(monkeypatch):
    reviews = []

    async def no_clarification(_prompt, planned):
        return planned

    async def review(prompt):
        reviews.append(prompt)
        return {"status": "valid", "issues": []}

    async def finalize(_prompt, planned):
        return planned

    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", no_clarification)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_agent_once", review)
    monkeypatch.setattr(pipeline, "_finalize_verified_request_plan", finalize)
    return reviews


@pytest.mark.parametrize("task_type", ["analyze_artifact", "build_composite_report", "sandbox_analysis"])
def test_verified_pipeline_does_not_spend_revision_on_execution_metadata(monkeypatch, task_type):
    raw = proposal(task_type, native=task_type == "sandbox_analysis")
    calls = []

    async def planner(prompt):
        calls.append(prompt)
        assert len(calls) == 1, "Execution metadata must not trigger another planner call"
        return raw

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", planner)
    reviews = install_reviews(monkeypatch)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once("Use the supplied table."))
    assert result == pipeline._bind_request_task_executors(raw)
    assert len(reviews) == 2  # Topology AND operation coverage still reviewed.
    assert raw["request_envelope"]["tasks"][0]["native"] == (task_type == "sandbox_analysis")


def test_binding_does_not_allow_reference_lookup_in_self_contained_executor(monkeypatch):
    raw = proposal("sandbox_analysis", native=True)
    raw["request_envelope"]["tasks"][0]["context"]["web_retrieval"] = {"required": True}
    calls = []

    async def planner(prompt):
        calls.append(prompt)
        return raw

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", planner)
    reviews = install_reviews(monkeypatch)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once("Retrieve a biological reference."))
    assert result["error"] == "request_graph_verification_no_progress"
    assert len(calls) == 2
    assert "wrong_task_type" in calls[-1]
    assert reviews == []


def test_each_revised_graph_binds_metadata_without_overriding_scientific_repair(monkeypatch):
    wrong = proposal("sandbox_analysis", native=True)
    wrong["request_envelope"]["tasks"][0]["context"]["web_retrieval"] = {"required": True}
    corrected = deepcopy(wrong)
    corrected["request_envelope"]["tasks"][0].update(task_type="answer_question", native=False)
    proposals = [wrong, corrected]

    async def planner(_prompt):
        return proposals.pop(0)

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", planner)
    reviews = install_reviews(monkeypatch)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once("Retrieve the requested reference."))
    assert result == pipeline._bind_request_task_executors(corrected)
    assert not proposals
    assert len(reviews) == 2
