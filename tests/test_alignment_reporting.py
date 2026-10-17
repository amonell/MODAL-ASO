import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from aso_agents.workbench_response_agent import evaluation_evidence_audit_agent

from aso_agents.alignment_reporting import alignment_geometry
from aso_agents.evaluation_evidence import attach_measurement_references, bind_assessment_measurements
from aso_agents.response_contracts import response_contract_errors


def summary(strand="-"):
    segments = [{"chrom": "chr8", "start": 310, "end": 319},
                {"chrom": "chr8", "start": 110, "end": 119}]
    if strand == "+":
        segments.reverse()
    hit = {"transcript_id": "neutral_transcript", "strand": strand,
           "mapping_scope": "spliced_cdna", "rna_context": "mature_spliced_cdna",
           "match_orientation": "reverse_complement", "coordinate_system": "1based_inclusive",
           "genomic_segments": segments, "genomic_span": {"chrom": "chr8", "start": 110, "end": 319},
           "matched_target_sequence": "ACGTACGTACGTACGTACGT", "mismatch_count": 1}
    return {"provided_asos": [{"name": "neutral_probe", "input_row_id": "input_1",
                              "near_match_candidates": [hit], "mapping_hits": []}]}


def response(text):
    return {"answer_summary": "Occupancy and activity remain unresolved.", "markdown": "",
            "evaluation_assessments": [{"name": "neutral_probe", "input_row_id": "input_1",
                "mechanistic_plausibility": text,
                "selectivity_limitations": "Other transcripts may bind.",
                "uncertainty": "No measured activity.", "ranking_rationale": "No potency ranking.",
                "evidence_refs": []}]}


@pytest.mark.parametrize("strand", ["+", "-"])
def test_discontinuous_geometry_retains_transcript_order_and_cannot_become_span(strand):
    data = summary(strand)
    original = deepcopy(data["provided_asos"][0]["near_match_candidates"])
    attach_measurement_references(data)
    row = data["provided_asos"][0]
    hypothesis = row["evidence_inventory"]["retained_hypotheses"][0]
    answer = response("Recorded alignment: " + hypothesis["geometry_reference_token"])
    bound = bind_assessment_measurements(answer, data)
    citation = next(v for v in bound["measurement_citations"].values() if v["field"] == "near_match_candidates[0]")
    assert citation["value"]["recorded_genomic_segment_count"] == 2
    assert citation["value"]["genomic_segments"] == original[0]["genomic_segments"]
    assert citation["value"]["strand"] == strand
    assert citation["value"]["rna_context"] == "mature_spliced_cdna"
    assert citation["value"]["matched_target_sequence"] == original[0]["matched_target_sequence"]
    assert data["provided_asos"][0]["near_match_candidates"] == original
    assert "evaluation_evidence.html#R001_alignment_1" in bound["markdown"]
    assert bound["answer_summary"] == answer["answer_summary"]


@pytest.mark.parametrize("segments", [None, [], [{"chrom": "chr8", "start": 1}],
                                      [{"chrom": "chr8", "start": 9, "end": 2}],
                                      [{"chrom": "chr8", "start": True, "end": 2}]])
def test_missing_or_invalid_segments_do_not_gain_a_block_count_from_the_span(segments):
    assert alignment_geometry({"genomic_segments": segments,
                               "genomic_span": {"chrom": "chr8", "start": 1, "end": 900}})[
        "recorded_genomic_segment_count"] is None


def locus_hit():
    return {"transcript_id": "neutral_locus", "strand": "+",
            "mapping_scope": "pre_mrna_locus", "rna_context": "unspliced_transcript_locus_pre_mrna",
            "match_orientation": "reverse_complement", "coordinate_system": "1based_inclusive",
            "genomic_span": {"chrom": "chr7", "start": 701, "end": 710},
            "matched_window_sequence": "ACGTACGTAC", "mismatch_count": 1,
            "mismatch_positions_in_supplied_sequence_1based": [3]}


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("orientation", ["input_sequence", "reverse_complement"])
@pytest.mark.parametrize("explicit_segments", [True, False])
def test_validated_unspliced_continuity_is_separate_from_raw_recorded_count(strand, orientation, explicit_segments):
    hit = locus_hit()
    hit.update(strand=strand, match_orientation=orientation)
    if explicit_segments:
        hit["genomic_segments"] = [deepcopy(hit["genomic_span"])]
    original = deepcopy(hit)
    geometry = alignment_geometry(hit)
    validated = geometry["validated_contiguous_locus_geometry"]
    assert validated["genomic_segment_count"] == 1
    assert validated["genomic_segments"] == [{"chrom": "7", "start": 701, "end": 710}]
    assert validated["matched_sequence_length"] == 10
    assert geometry["recorded_genomic_segment_count"] == (1 if explicit_segments else None)
    assert geometry["genomic_segments"] == hit.get("genomic_segments")
    assert hit == original


@pytest.mark.parametrize("change", [
    {"mapping_scope": "spliced_cdna", "rna_context": "mature_spliced_cdna"},
    {"rna_context": "mature_spliced_cdna"},
    {"rna_context": None},
    {"genomic_span": {"chrom": "chr7", "start": 701, "end": 720}},
    {"genomic_span": {"chrom": "chr7", "start": 0, "end": 9}},
    {"genomic_segments": []},
    {"genomic_segments": [{"chrom": "chr7", "start": 702, "end": 711}]},
    {"matched_window_sequence": None},
    {"matched_window_sequence": "ACGTA-GTAC"},
    {"mismatch_positions_in_supplied_sequence_1based": []},
    {"strand": None},
    {"match_orientation": "unknown"},
])
def test_incomplete_conflicting_or_spliced_records_do_not_gain_validated_locus_geometry(change):
    hit = locus_hit(); hit.update(change)
    assert "validated_contiguous_locus_geometry" not in alignment_geometry(hit)


def test_validated_locus_geometry_reaches_its_own_citation_without_mutating_source():
    hit = locus_hit(); original = deepcopy(hit)
    data = {"provided_asos": [{"name": "neutral_probe", "input_row_id": "input_1",
                              "near_match_candidates": [hit], "mapping_hits": []}]}
    attach_measurement_references(data)
    token = data["provided_asos"][0]["reference_tokens"]["alignment_1"]
    bound = bind_assessment_measurements(response("Retained geometry: " + token), data)
    citation = next(iter(bound["measurement_citations"].values()))["value"]
    assert citation["recorded_genomic_segment_count"] is None
    assert citation["validated_contiguous_locus_geometry"]["genomic_segment_count"] == 1
    assert hit == original


def test_separate_hypotheses_of_the_same_molecule_keep_distinct_geometry():
    data = summary()
    near = deepcopy(data["provided_asos"][0]["near_match_candidates"][0])
    near.update(transcript_id="alternative_transcript", rna_context="unspliced_transcript_locus_pre_mrna",
                mapping_scope="pre_mrna_locus", genomic_segments=[{"chrom": "chr8", "start": 40, "end": 59}])
    data["provided_asos"][0]["near_match_candidates"].append(near)
    attach_measurement_references(data)
    hypotheses = data["provided_asos"][0]["evidence_inventory"]["retained_hypotheses"]
    assert [h["geometry"]["recorded_genomic_segment_count"] for h in hypotheses] == [2, 1]
    assert hypotheses[0]["geometry_reference_token"] != hypotheses[1]["geometry_reference_token"]
    bound = bind_assessment_measurements(response(hypotheses[1]["geometry_reference_token"]), data)
    assert next(iter(bound["measurement_citations"].values()))["value"]["transcript_id"] == "alternative_transcript"


def test_geometry_citation_cannot_be_borrowed_from_another_input_row():
    data = summary()
    second = deepcopy(data["provided_asos"][0]); second.update(name="other_probe", input_row_id="input_2")
    data["provided_asos"].append(second); attach_measurement_references(data)
    token = second["reference_tokens"]["alignment_1"]
    with pytest.raises(ValueError, match="different molecule/input row"):
        bind_assessment_measurements(response(token), data)


@pytest.mark.parametrize("text", ["It occupies a single genomic block.", "It has two blocks.",
                                  "Its segment count is 2.", "The hit has 3 discontiguous genomic segments."])
def test_geometry_prose_requires_semantic_review_not_a_phrase_ban(text):
    errors = response_contract_errors(response(text), {"task_type": "evaluate_existing_aso_set",
        "evaluation_summary": attach_measurement_references(summary())})
    assert not errors


@pytest.mark.parametrize("text", ["A gapmer has a DNA gap and modified wings.",
                                  "Mature RNA may contain exon junctions; activity is unknown.",
                                  "Recorded geometry: {{R001_alignment_1}}."])
def test_general_mechanism_uncertainty_and_bound_geometry_remain_deliverable(text):
    assert not response_contract_errors(response(text), {"task_type": "evaluate_existing_aso_set",
        "evaluation_summary": attach_measurement_references(summary())})


def test_native_composer_repairs_contradicted_geometry_preserving_other_fields(monkeypatch):
    from aso_agents import pipeline_tools
    from aso_agents.response_context import decode_response_context
    data = attach_measurement_references(summary())
    original = response("The retained alignment has a single genomic block; occupancy is unresolved.")
    repaired = response("The retained alignment is {{R001_alignment_1}}; occupancy is unresolved.")
    calls, audits = [], []

    async def run(agent, **kwargs):
        wire = json.loads(kwargs['input'])
        context = decode_response_context(wire) if wire.get('context_encoding') else wire
        if agent.name == evaluation_evidence_audit_agent.name:
            audits.append(context['draft_response'])
            answer = {'findings': [] if len(audits) > 1 else [{
                'response_path': '/evaluation_assessments/0/mechanistic_plausibility',
                'quote': 'a single genomic block', 'kind': 'contradicted_fact',
                'evidence_paths': ['/evaluation_summary/measurement_evidence/R001_alignment_1/value/genomic_segments'],
                'reason': 'This retained spliced alignment has two explicitly recorded genomic segments.'}]}
        else:
            calls.append(context)
            answer = original if len(calls) == 1 else {'corrections': {
                '/evaluation_assessments/0/mechanistic_plausibility': repaired['evaluation_assessments'][0]['mechanistic_plausibility']}}
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    actual = asyncio.run(pipeline_tools._run_workbench_response_agent_once(
        {'task_type': 'evaluate_existing_aso_set', 'evaluation_summary': data}))
    assert actual.pop('native_consistency_review')['attempts'] == 2
    assert actual == repaired and len(calls) == 2
    assert calls[1]['preceding_response'] == original
    bound = bind_assessment_measurements(actual, data)
    assert bound['measurement_citations']['R001_alignment_1']['value']['recorded_genomic_segment_count'] == 2
    for field in ['selectivity_limitations', 'uncertainty', 'ranking_rationale']:
        assert bound['evaluation_assessments'][0][field] == original['evaluation_assessments'][0][field]
