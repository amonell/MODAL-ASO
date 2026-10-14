import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as native
from aso_agents.comparison_evidence import response_comparison_context
from aso_agents.response_context import decode_response_context
from aso_agents.workbench_response_agent import evaluation_evidence_audit_agent, evaluation_evidence_repair_agent


def comparison():
    sets = [{"set_id": name, "source_type": "generated", "top_rows_in_source_order": [
        {"isoform_hit_count": count, "recorded_scientific_fields": {
            "isoform_coverage_fraction": fraction, "catalog_size": size}}]}
        for name, count, size, fraction in (("A", 12, 20, .6), ("B", 8, 10, .8))]
    return {"task_type": "compare_aso_sets", "request_text": "Compare the supplied candidate evidence.",
            "comparison_summary": response_comparison_context({"candidate_evidence_by_set": sets})}


def draft(correct):
    return {"answer_summary": "Potency cannot be ordered from these catalog measurements.",
            "markdown": ("Set B has higher fractional coverage but fewer raw isoform hits than set A."
                         if correct else "Set B has higher counts and coverage than set A."),
            "limitations": ["The per-set catalogs differ. Biological potency is unmeasured."]}


def contradiction():
    return {"response_path": "/markdown", "quote": draft(False)["markdown"],
            "kind": "contradicted_fact", "evidence_paths": [
                "/comparison_summary/candidate_evidence_by_set/0/recorded_attribute_summaries/isoform_hit_count",
                "/comparison_summary/candidate_evidence_by_set/1/recorded_attribute_summaries/isoform_hit_count"],
            "reason": "Eight recorded hits is lower than twelve, despite the higher fractional coverage."}


def decode(prompt):
    body = json.loads(prompt)
    return decode_response_context(body) if body.get("context_encoding") else body


def test_native_comparison_repairs_metric_conflation_preserving_data_and_uncertainty(monkeypatch):
    source = comparison()
    original = deepcopy(source)
    proposed, audited = [], []
    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            body = decode(kwargs["input"])
            assert body["source"] == source
            audited.append(body["draft_response"])
            answer = {"findings": [contradiction()] if len(audited) == 1 else []}
        else:
            proposed.append(kwargs["input"])
            answer = ({"corrections": {"/markdown": draft(True)["markdown"]}}
                      if agent.name == evaluation_evidence_repair_agent.name else draft(False))
        return SimpleNamespace(final_output=json.dumps(answer))
    monkeypatch.setattr(native.Runner, "run", run)
    result = asyncio.run(native._run_workbench_response_batch_once(source))
    assert len(proposed) == len(audited) == 2
    assert result["markdown"] == draft(True)["markdown"]
    assert result["answer_summary"] == draft(False)["answer_summary"]
    assert result["limitations"] == draft(False)["limitations"]
    assert result["native_consistency_review"]["attempts"] == 2
    assert contradiction()["quote"] in proposed[1] and "evidence_sha256" in proposed[1]
    assert source == original


def test_correct_opposite_metric_orderings_and_indeterminate_potency_need_no_rewrite(monkeypatch):
    source = comparison()
    calls = []
    async def run(agent, **kwargs):
        calls.append(agent.name)
        answer = {"findings": []} if agent.name == evaluation_evidence_audit_agent.name else draft(True)
        return SimpleNamespace(final_output=json.dumps(answer))
    monkeypatch.setattr(native.Runner, "run", run)
    result = asyncio.run(native._run_workbench_response_batch_once(source))
    assert len(calls) == 2
    assert {k: v for k, v in result.items() if k != "native_consistency_review"} == draft(True)
    assert result["native_consistency_review"]["attempts"] == 1


def test_catalog_caveats_do_not_bypass_repeated_concrete_count_contradiction(monkeypatch):
    source = comparison()
    calls = []
    async def run(agent, **kwargs):
        calls.append(agent.name)
        if agent.name == evaluation_evidence_audit_agent.name:
            answer = {"findings": [contradiction()]}
        elif agent.name == evaluation_evidence_repair_agent.name:
            # Valid scoped patches can still repeat the same factual error.
            number = calls.count(evaluation_evidence_repair_agent.name)
            answer = {"corrections": {"/markdown": draft(False)["markdown"] + f" Revision {number}."}}
        else:
            answer = draft(False)
        return SimpleNamespace(final_output=json.dumps(answer))
    monkeypatch.setattr(native.Runner, "run", run)
    with pytest.raises(RuntimeError, match="workbench_response_contract_failed"):
        asyncio.run(native._run_workbench_response_batch_once(source))
    assert calls.count(evaluation_evidence_repair_agent.name) == 3
    # The conservative replacement receives one additional independent audit;
    # this deliberately invalid reviewer response still cannot be accepted.
    assert calls.count(evaluation_evidence_audit_agent.name) == 5
