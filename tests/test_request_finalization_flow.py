"""Finalization reviews new questions and the synthesized execution contract."""

import asyncio
from copy import deepcopy

from aso_agents import pipeline_tools as pipeline


def ready_plan():
    return {
        "status": "ready",
        "questions": [],
        "reason": None,
        "request_envelope": {"tasks": []},
    }


def test_ready_plan_is_synthesized_before_its_final_necessity_review(monkeypatch):
    planned = ready_plan()
    events = []

    async def complete(evidence, current):
        events.append("completeness")
        assert evidence == "The supplied request"
        return current

    async def synthesize(evidence, current):
        events.append("synthesis")
        result = deepcopy(current)
        result["request_envelope"]["tasks"] = [{"task_id": "generated_goal"}]
        return result

    async def necessity(evidence, current):
        events.append("necessity")
        # A ready input has already been reviewed by the graph compiler. The
        # next review must inspect the newly synthesized contract.
        assert current["request_envelope"]["tasks"] == [
            {"task_id": "generated_goal"}
        ]
        return current

    monkeypatch.setattr(pipeline, "_review_request_execution_completeness", complete)
    monkeypatch.setattr(pipeline, "_synthesize_missing_design_goals", synthesize)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", necessity)

    result = asyncio.run(
        pipeline._finalize_verified_request_plan("The supplied request", planned)
    )

    assert events == ["completeness", "synthesis", "necessity"]
    assert result["request_envelope"]["tasks"] == [{"task_id": "generated_goal"}]
    assert planned == ready_plan()


def test_new_completeness_question_still_blocks_synthesis(monkeypatch):
    question = "Where is the required private sample table?"
    events = []

    async def complete(evidence, current):
        result = deepcopy(current)
        result.update(status="need_clarification", questions=[question])
        return result

    async def necessity(evidence, current):
        events.append("necessity")
        assert current["questions"] == [question]
        return current

    async def forbidden_synthesis(*args):
        raise AssertionError("A required private input is still missing")

    monkeypatch.setattr(pipeline, "_review_request_execution_completeness", complete)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", necessity)
    monkeypatch.setattr(pipeline, "_synthesize_missing_design_goals", forbidden_synthesis)

    result = asyncio.run(pipeline._finalize_verified_request_plan("request", ready_plan()))

    assert result["status"] == "need_clarification"
    assert result["questions"] == [question]
    assert events == ["necessity"]


def test_synthesis_question_is_reviewed_before_leaving_finalization(monkeypatch):
    question = "Which of the two private sample groups should be evaluated?"
    seen_questions = []

    async def complete(evidence, current):
        return current

    async def synthesize(evidence, current):
        result = deepcopy(current)
        result.update(status="need_clarification", questions=[question])
        return result

    async def necessity(evidence, current):
        seen_questions.append(current["questions"])
        return current

    monkeypatch.setattr(pipeline, "_review_request_execution_completeness", complete)
    monkeypatch.setattr(pipeline, "_synthesize_missing_design_goals", synthesize)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", necessity)

    result = asyncio.run(pipeline._finalize_verified_request_plan("request", ready_plan()))

    assert result["questions"] == [question]
    assert seen_questions[-1] == [question]


def test_completeness_error_does_not_enter_synthesis_or_review(monkeypatch):
    failure = {"error": "completeness_contract_invalid"}

    async def complete(*args):
        return failure

    async def forbidden(*args):
        raise AssertionError("Completeness failed")

    monkeypatch.setattr(pipeline, "_review_request_execution_completeness", complete)
    monkeypatch.setattr(pipeline, "_synthesize_missing_design_goals", forbidden)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", forbidden)

    assert asyncio.run(
        pipeline._finalize_verified_request_plan("request", ready_plan())
    ) is failure
