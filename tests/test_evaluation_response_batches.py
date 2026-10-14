import asyncio
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import pytest
from aso_agents.workbench_response_agent import evaluation_evidence_audit_agent, panel_comparison_audit_agent
from aso_agents.response_assembly import assembly_audit_agent

from aso_agents.evaluation_evidence import attach_measurement_references
from aso_agents.response_context import (
    decode_response_context, evaluation_response_batches, merge_evaluation_responses,
)


def panel(count=3):
    rows = []
    for i in range(count):
        rows.append({"input_row_id": f"row_{i}", "name": "same_display_name", "sequence": "ACGT",
                     "chemistry": "steric" if i % 2 else "gapmer",
                     "mapping_hits": [], "near_match_candidates": [
                         {"transcript_id": f"TX_{i}_{j}", "mapping_scope": "spliced_cdna",
                          "rna_context": "mature_spliced_cdna", "match_orientation": "reverse_complement",
                          "mismatch_count": j, "genomic_segments": [
                              {"chrom": "chrNeutral", "start": j + 1, "end": j + 20}],
                          "coordinate_system": "1based_inclusive"} for j in range(240)],
                     "off_target_exact_count": 0, "measured_activity": None,
                     "knockdown_proxy_rank": count - i,
                     "unexpected_measurement_column": {"value": 0, "measured": False}})
    return {"task_type": "evaluate_existing_aso_set", "request_text": "Evaluate every supplied observation.",
            "evaluation_summary": attach_measurement_references({"provided_aso_count": count,
                                                                  "provided_asos": rows})}


def answer(row):
    return {"answer_summary": "Activity remains unresolved.", "markdown": "",
            "evaluation_assessments": [{"input_row_id": row["input_row_id"], "name": row["name"],
                "mechanistic_plausibility": row["reference_tokens"]["alignment_1"],
                "selectivity_limitations": "The measured screening scope is limited.",
                "uncertainty": "No activity measurement.", "ranking_rationale": "Proxy rank is not potency."}],
            "limitations": ["Common caveat", "Specific to " + row["input_row_id"]]}


def test_every_row_and_its_original_citations_are_preserved_without_panel_sampling():
    payload = panel()
    original = deepcopy(payload)
    batches = evaluation_response_batches(payload, target_context_bytes=1)
    assert payload == original and len(batches) == 3
    for i, batch in enumerate(batches):
        summary = batch["evaluation_summary"]
        assert summary["provided_asos"] == [original["evaluation_summary"]["provided_asos"][i]]
        assert summary["provided_aso_count"] == 1 and summary["full_panel_observation_count"] == 3
        assert {item["input_row_id"] for item in summary["measurement_evidence"].values()} == {f"row_{i}"}
        overview = summary["panel_comparison_overview"]
        assert [r["knockdown_proxy_rank"] for r in overview] == [3, 2, 1]
        assert [r["off_target_exact_count"] for r in overview] == [0, 0, 0]
        assert all(r["unexpected_measurement_column"] == {"value": 0, "measured": False} for r in overview)
        assert all("near_match_candidates" not in r and r["alignment_inventory_counts"]["retained_near_match_count"] == 240 for r in overview)


def test_assembly_preserves_first_judgments_uncertainties_and_original_input_order():
    payload = panel()
    responses = [answer(row) for row in payload["evaluation_summary"]["provided_asos"]]
    original = deepcopy(responses)
    merged = merge_evaluation_responses(list(reversed(responses)), payload)
    assert responses == original
    assert merged["evaluation_assessments"] == [r["evaluation_assessments"][0] for r in original]
    assert merged["original_component_responses"] == list(reversed(original))
    assert merged["limitations"] == ["Common caveat", "Specific to row_2", "Specific to row_1", "Specific to row_0"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign"])
def test_assembly_rejects_missing_duplicate_or_foreign_observations(mutation):
    payload = panel()
    responses = [answer(row) for row in payload["evaluation_summary"]["provided_asos"]]
    if mutation == "missing":
        responses.pop()
    elif mutation == "duplicate":
        responses.append(deepcopy(responses[0]))
    else:
        responses[0]["evaluation_assessments"][0]["input_row_id"] = "foreign"
    with pytest.raises(ValueError):
        merge_evaluation_responses(responses, payload)


def test_ordinary_small_evaluation_and_other_task_types_keep_one_request():
    for payload in ({"task_type": "evaluate_existing_aso_set", "evaluation_summary": {"provided_asos": []}},
                    {"task_type": "compare_aso_sets", "evaluation_summary": panel()["evaluation_summary"]}):
        assert evaluation_response_batches(payload) == [payload]


def test_native_synchronous_model_io_runs_concurrently_and_merges_every_original_answer(monkeypatch):
    from aso_agents import pipeline_tools
    payload = panel()
    barrier = threading.Barrier(3)
    received = []
    audited = []

    async def run(agent, **kwargs):
        if agent.name in {evaluation_evidence_audit_agent.name, panel_comparison_audit_agent.name,
                          assembly_audit_agent.name}:
            audited.append(agent.name)
            return SimpleNamespace(final_output=json.dumps({'findings': []}))
        wire = json.loads(kwargs["input"])
        decoded = decode_response_context(wire) if wire.get("context_encoding") else wire
        row = decoded["evaluation_summary"]["provided_asos"][0]
        assert len(decoded["evaluation_summary"]["provided_asos"]) == 1
        received.append(row["input_row_id"])
        barrier.wait(timeout=10)
        return SimpleNamespace(final_output=json.dumps(answer(row)))

    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    response = asyncio.run(pipeline_tools._run_workbench_response_agent_once(payload))
    assert sorted(received) == ["row_0", "row_1", "row_2"]
    assert audited.count(evaluation_evidence_audit_agent.name) == 3
    assert audited.count(panel_comparison_audit_agent.name) == 3
    assert audited.count(assembly_audit_agent.name) == 0
    assert response['native_assembly_review']['method'] == 'deterministic_scoped_component_assembly'
    assert response['native_assembly_review']['scientific_component_count'] == 3
    assert response['native_assembly_review']['status'] == 'component_scopes_preserved'
    for delivered, row in zip(response["evaluation_assessments"],
                              payload["evaluation_summary"]["provided_asos"]):
        original = answer(row)["evaluation_assessments"][0]
        for field in original:
            if field != "ranking_rationale":
                assert delivered[field] == original[field]
        assert "not measured efficacy" in delivered["ranking_rationale"]
