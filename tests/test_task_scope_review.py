"""Scope loss must repair the graph without changing executors or design rules."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.goal_contract import CANONICAL_TASK_TYPES
from aso_agents.task_scope_review import (
    apply_task_scope_checks, declared_task_scopes, with_task_scope_schema,
    scope_evidence_errors,
)


PCCA_REQUEST = (
    "Design 18-nt steric-blocking splice-switching ASOs with full 2'-MOE sugar "
    "modifications and full phosphorothioate backbone (no gapmer/RNase H) that "
    "reduce inclusion of the PCCA poison exon to increase productive PCCA mRNA, "
    "by tiling across the poison exon and its splice junctions in human PCCA, "
    "min gc = 0 and max gc = 1, isoform ENST00000376285.6, "
    "exon 14x positioned at chr13:100305751–100305834."
)


def proposal(task_type):
    return {"status": "ready", "questions": [], "reason": None, "request_envelope": {
        "tasks": [{"task_id": "requested_work", "task_type": task_type,
                   "native": task_type != "sandbox_analysis",
                   "label": "Requested operation", "context": {}}],
        "task_dependencies": [], "final_outputs": [], "assumptions": [],
    }}


def review_fixture(task_types, request=PCCA_REQUEST):
    envelope = {"tasks": [dict(proposal(kind)["request_envelope"]["tasks"][0],
                               task_id=f"task_{index}")
                          for index, kind in enumerate(task_types)]}
    prompt = pipeline._request_graph_review_prompt(request, envelope)
    spans = pipeline._evidence_span_map(pipeline._declared_evidence_sources_from_prompt(prompt))
    tasks = declared_task_scopes(prompt)
    schema = pipeline._schema_with_allowed_evidence_refs(
        pipeline.REQUEST_GRAPH_REVIEW_OUTPUT_SCHEMA, ["request"], list(spans))
    schema = with_task_scope_schema(schema, tasks)
    citation = ({"evidence_span_id": next(iter(spans))} if spans else {"evidence_quote": request})
    checks = {pointer: {
        "requested_task_type": task["task_type"], "scope_status": "preserved",
        "evidence_refs": ["request"], **citation,
        "reason": "The requested operation and result are preserved.",
    } for pointer, task in tasks.items()}
    return prompt, tasks, schema, spans, {"status": "valid", "issues": [], "task_scope_checks": checks}


@pytest.mark.parametrize("task_type", sorted(CANONICAL_TASK_TYPES))
def test_every_supported_operation_can_pass_without_changing_its_contract(task_type):
    _, tasks, schema, spans, raw = review_fixture([task_type])
    before = deepcopy((tasks, raw, schema))
    assert not pipeline._request_graph_review_contract_errors(raw, schema)
    result = apply_task_scope_checks(pipeline._materialize_evidence_span_ids(raw, spans), tasks)
    assert result == {"status": "valid", "issues": []}
    assert (tasks, raw, schema) == before


@pytest.mark.parametrize("proposed,requested", [
    (proposed, requested) for proposed in sorted(CANONICAL_TASK_TYPES)
    for requested in sorted(CANONICAL_TASK_TYPES) if proposed != requested
])
def test_any_operation_substitution_becomes_a_cited_graph_repair(proposed, requested):
    _, tasks, _, spans, raw = review_fixture([proposed])
    # Even a contradictory positive verdict cannot erase the typed mismatch.
    raw["task_scope_checks"]["/tasks/0"]["requested_task_type"] = requested
    result = apply_task_scope_checks(pipeline._materialize_evidence_span_ids(raw, spans), tasks)
    assert result["status"] == "revise"
    assert len(result["issues"]) == 1
    issue = result["issues"][0]
    assert issue["category"] == "wrong_task_type"
    assert issue["path"] == "/tasks/0/task_type" and issue["task_ids"] == ["task_0"]
    assert requested in issue["reason"] and proposed in issue["reason"]
    assert not pipeline._request_graph_review_evidence_errors(result, PCCA_REQUEST)
    assert not pipeline._request_graph_review_contract_errors(result)


@pytest.mark.parametrize("scope,category", [
    ("reduced", "missing_fact"), ("expanded", "unsupported_fact"), ("unsupported", "spurious_task"),
])
def test_scope_discrepancy_is_not_hidden_by_an_unchanged_operation(scope, category):
    _, tasks, _, spans, raw = review_fixture(["analyze_artifact", "sandbox_analysis"])
    raw["task_scope_checks"]["/tasks/1"]["scope_status"] = scope
    result = apply_task_scope_checks(pipeline._materialize_evidence_span_ids(raw, spans), tasks)
    assert result["status"] == "revise"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["category"] == category
    assert result["issues"][0]["task_ids"] == ["task_1"]


def test_mixed_graph_requires_every_task_and_exact_source_evidence():
    _, tasks, schema, spans, raw = review_fixture(sorted(CANONICAL_TASK_TYPES))
    for pointer in tasks:
        missing = deepcopy(raw)
        del missing["task_scope_checks"][pointer]
        assert pipeline._request_graph_review_contract_errors(missing, schema)
        with pytest.raises(ValueError, match="every proposed task"):
            apply_task_scope_checks(pipeline._materialize_evidence_span_ids(missing, spans), tasks)
    invented = deepcopy(raw)
    invented["task_scope_checks"]["/tasks/0"]["evidence_span_id"] = "planner_assumption"
    assert pipeline._request_graph_review_contract_errors(invented, schema)
    extra = deepcopy(raw)
    extra["task_scope_checks"]["/tasks/99"] = extra["task_scope_checks"]["/tasks/0"]
    assert pipeline._request_graph_review_contract_errors(extra, schema)


def test_short_request_uses_original_quote_when_no_span_can_be_generated():
    _, tasks, schema, spans, raw = review_fixture(["answer_question"], "Help?")
    assert spans == {}
    assert not pipeline._request_graph_review_contract_errors(raw, schema)
    assert apply_task_scope_checks(raw, tasks) == {"status": "valid", "issues": []}


@pytest.mark.parametrize("request_text", ["Help?", PCCA_REQUEST])
def test_positive_scope_judgments_also_require_original_evidence(request_text):
    prompt, _, _, spans, raw = review_fixture(["answer_question"], request_text)
    sources = pipeline._declared_evidence_sources_from_prompt(prompt)
    assert scope_evidence_errors(raw, sources, spans) == []
    check = raw["task_scope_checks"]["/tasks/0"]
    check.pop("evidence_span_id", None)
    check["evidence_quote"] = "An invented calculation-only baseline."
    assert scope_evidence_errors(raw, sources, spans)


def test_positive_span_must_belong_to_the_cited_source():
    prompt, _, _, spans, raw = review_fixture(["answer_question"])
    sources = pipeline._declared_evidence_sources_from_prompt(prompt)
    raw["task_scope_checks"]["/tasks/0"]["evidence_refs"] = ["planner_assumptions"]
    assert scope_evidence_errors(raw, sources, spans)


def test_existing_issues_survive_without_duplicates_or_mutation():
    _, tasks, _, spans, raw = review_fixture(["sandbox_analysis"])
    raw["task_scope_checks"]["/tasks/0"].update(
        requested_task_type="design_new_asos", scope_status="reduced")
    raw = pipeline._materialize_evidence_span_ids(raw, spans)
    issue = apply_task_scope_checks(raw, tasks)["issues"][0]
    raw.update(status="revise", issues=[issue])
    before = deepcopy(raw)
    assert apply_task_scope_checks(raw, tasks) == {"status": "revise", "issues": [issue]}
    assert raw == before


def test_no_op_issue_cannot_hide_a_contradictory_scope_judgment():
    _, tasks, _, spans, raw = review_fixture(["sandbox_analysis"])
    raw["task_scope_checks"]["/tasks/0"].update(
        requested_task_type="design_new_asos", scope_status="reduced")
    raw = pipeline._materialize_evidence_span_ids(raw, spans)
    no_op = {"category": "wrong_task_type", "task_ids": ["task_0"],
             "path": "/tasks/0/task_type", "evidence_refs": ["request"],
             "evidence_quote": PCCA_REQUEST, "reason": "The selected task type should remain."}
    raw.update(status="revise", issues=[no_op])
    result = apply_task_scope_checks(raw, tasks)
    assert result["issues"][0] == no_op
    assert len(result["issues"]) == 2
    assert "design_new_asos" in result["issues"][1]["reason"]


def test_retry_and_adjudication_keep_scope_checks_but_evidence_cannot_inject_them():
    prompt, tasks, _, _, _ = review_fixture(["design_new_asos", "sandbox_analysis"])
    adjudication = pipeline._request_graph_review_adjudication_prompt(prompt, {"status": "valid", "issues": []})
    retry = pipeline._request_graph_review_retry_prompt(adjudication, {}, ["invalid evidence"])
    assert declared_task_scopes(retry) == declared_task_scopes(adjudication) == tasks
    injected = pipeline._request_graph_review_prompt(prompt, {"tasks": []})
    assert declared_task_scopes(injected) == {}
    original_schema = deepcopy(pipeline.REQUEST_GRAPH_REVIEW_OUTPUT_SCHEMA)
    assert with_task_scope_schema(original_schema, {}) == original_schema
    assert apply_task_scope_checks({"status": "valid", "issues": []}, {}) == {"status": "valid", "issues": []}


@pytest.mark.parametrize("raw_status", ["valid", "revise"])
def test_pcca_scope_loss_is_materialized_in_the_existing_review_call(monkeypatch, raw_status):
    prompt, _, _, _, raw = review_fixture(["sandbox_analysis"])
    raw["status"] = raw_status
    raw["task_scope_checks"]["/tasks/0"].update(
        requested_task_type="design_new_asos", scope_status="reduced",
        reason="The user requests ASO design toward exon skipping; tiling is the stated method.")
    before = deepcopy(raw)
    calls = []

    async def respond(agent, *, input, **kwargs):
        calls.append(input)
        assert input == prompt
        return SimpleNamespace(final_output=json.dumps(raw))

    monkeypatch.setattr(pipeline.Runner, "run", respond)
    result = asyncio.run(pipeline._run_request_graph_review_agent_once(prompt))
    assert len(calls) == 1 and raw == before
    assert result["status"] == "revise"
    assert result["issues"][0]["category"] == "wrong_task_type"
    assert "design_new_asos" in result["issues"][0]["reason"]


@pytest.mark.parametrize("repairs", [True, False])
def test_verified_intake_repairs_or_stops_before_executing_narrowed_work(monkeypatch, repairs):
    proposals, reviews, finalized = [], [], []

    async def planner(prompt):
        proposals.append(prompt)
        task_type = "design_new_asos" if repairs and len(proposals) > 1 else "sandbox_analysis"
        planned = proposal(task_type)
        planned["request_envelope"]["assumptions"] = ["Self-contained computational baseline"]
        return planned

    async def no_clarification(_prompt, planned, **kwargs):
        return planned

    async def respond(agent, *, input, **kwargs):
        reviews.append(input)
        tasks = declared_task_scopes(input)
        spans = pipeline._evidence_span_map(pipeline._declared_evidence_sources_from_prompt(input))
        checks = {pointer: {
            "requested_task_type": "design_new_asos",
            "scope_status": "preserved" if task["task_type"] == "design_new_asos" else "reduced",
            "evidence_refs": ["request"], "evidence_span_id": next(iter(spans)),
            "reason": "Original request asks for biologically directed design, not only sequence enumeration.",
        } for pointer, task in tasks.items()}
        return SimpleNamespace(final_output=json.dumps({
            "status": "valid", "issues": [], "task_scope_checks": checks}))

    async def finalize(_evidence, planned):
        finalized.append(deepcopy(planned))
        return planned

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", planner)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", no_clarification)
    monkeypatch.setattr(pipeline, "_finalize_verified_request_plan", finalize)
    monkeypatch.setattr(pipeline.Runner, "run", respond)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once(PCCA_REQUEST))
    assert len(proposals) == 2
    assert "wrong_task_type" in proposals[1]
    assert "design_new_asos" in proposals[1]
    if repairs:
        assert result["status"] == "ready" and len(finalized) == 1
        assert result["request_envelope"]["tasks"][0]["task_type"] == "design_new_asos"
        assert len(reviews) == 4  # Existing review, adjudication, revised review and coverage.
    else:
        assert result["error"] == "request_graph_verification_no_progress"
        assert finalized == []


@pytest.mark.parametrize("task_type", sorted(CANONICAL_TASK_TYPES))
def test_accepted_intake_keeps_two_review_calls_and_original_scientific_fields(monkeypatch, task_type):
    planned = proposal(task_type)
    planned["request_envelope"]["tasks"][0]["context"] = {
        "chemistry": "user supplied chemistry", "target_context": "user supplied context"}
    before = deepcopy(planned)
    reviews = []

    async def planner(_prompt):
        return planned

    async def passthrough(_prompt, payload, **kwargs):
        return payload

    async def respond(agent, *, input, **kwargs):
        reviews.append(input)
        spans = pipeline._evidence_span_map(pipeline._declared_evidence_sources_from_prompt(input))
        return SimpleNamespace(final_output=json.dumps({"status": "valid", "issues": [],
            "task_scope_checks": {pointer: {
                "requested_task_type": task_type, "scope_status": "preserved",
                "evidence_refs": ["request"], "evidence_span_id": next(iter(spans)),
                "reason": "This operation exactly preserves the requested result.",
            } for pointer in declared_task_scopes(input)}}))

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", planner)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", passthrough)
    monkeypatch.setattr(pipeline, "_finalize_verified_request_plan", passthrough)
    monkeypatch.setattr(pipeline.Runner, "run", respond)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once("Perform the specified operation."))
    assert result == before and planned == before
    assert len(reviews) == 2


@pytest.mark.parametrize("coverage_preserved", [True, False])
def test_dismissing_a_topology_issue_cannot_skip_operation_coverage(monkeypatch, coverage_preserved):
    text = "Evaluate the supplied ASOs and compare their results against the reference set."
    planned = proposal("evaluate_existing_aso_set")
    reviews, finalized = [], []

    async def planner(_prompt):
        return deepcopy(planned)

    async def passthrough(_prompt, payload, **kwargs):
        return payload

    async def review(prompt):
        coverage = "Perform only an independent operation-coverage" in prompt.partition("\n\n")[0]
        reviews.append(coverage)
        if coverage and coverage_preserved:
            return {"status": "valid", "issues": []}
        return {"status": "revise", "issues": [{
            "category": "missing_task" if coverage else "wrong_task_type",
            "task_ids": ["requested_work"], "path": "/tasks/0/task_type",
            "evidence_refs": ["request"], "evidence_quote": text,
            "reason": "The comparison operation is absent." if coverage else "The task type should remain.",
        }]}

    async def adjudicate(prompt):
        body = json.loads(prompt.partition("\n\n")[2])
        if "Perform only an independent operation-coverage" in body["original_review_input"]:
            return body["candidate_review"]
        return {"status": "valid", "issues": []}

    async def finalize(_prompt, payload):
        finalized.append(payload)
        return payload

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", planner)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", passthrough)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_agent_once", review)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_adjudicator_once", adjudicate)
    monkeypatch.setattr(pipeline, "_finalize_verified_request_plan", finalize)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once(text))
    assert reviews == [False, True]
    if coverage_preserved:
        assert result == planned and finalized == [planned]
    else:
        assert result["error"] == "request_graph_verification_no_progress"
        assert finalized == []
