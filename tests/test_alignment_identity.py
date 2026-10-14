from copy import deepcopy

import pytest

from aso_agents.alignment_identity import recorded_window_identity
from aso_agents.evaluation_evidence import row_evidence_inventory, attach_measurement_references


def paired_row():
    common = {"transcript_id": "TX_NEUTRAL.2", "reference_build": "assembly-A",
              "mapping_scope": "pre_mrna_locus", "rna_context": "unspliced_transcript_locus_pre_mrna",
              "match_orientation": "input_sequence", "strand": "+",
              "genomic_span": {"chrom": "7", "start": 701, "end": 710}}
    exact = {**deepcopy(common), "matched_target_sequence": "ACGTTGCAAC",
             "genomic_segments": [{"chrom": "7", "start": 701, "end": 710}]}
    near = {**deepcopy(common), "coordinate_system": "1based_inclusive",
            "matched_window_sequence": "ACGTTGCAAC", "mismatch_count": 0,
            "mismatch_positions_in_supplied_sequence_1based": []}
    return {"name": "neutral", "input_row_id": "row-A", "mapping_hits": [exact], "near_match_candidates": [near]}


def test_exact_and_zero_mismatch_search_preserve_two_records_for_one_window():
    row = paired_row(); original = deepcopy(row)
    summary = attach_measurement_references({"provided_asos": [row]})
    inventory = row["evidence_inventory"]
    assert inventory["exact_mapping_count"] == inventory["retained_near_match_count"] == 1
    groups = inventory["recorded_window_identity"]
    assert groups["identified_window_group_count"] == 1 and groups["complete_for_retained_records"]
    assert groups["identified_window_groups"][0]["source_fields"] == ["mapping_hits[0]", "near_match_candidates[0]"]
    assert groups["identified_window_groups"][0]["source_record_count"] == 2
    refs = summary["measurement_evidence"]
    assert refs["R001_alignment_1"]["row_local_window_group_id"] == refs["R001_alignment_2"]["row_local_window_group_id"]
    assert refs["R001_alignment_1"]["value"]["recorded_genomic_segment_count"] == 1
    assert refs["R001_alignment_2"]["value"]["recorded_genomic_segment_count"] is None
    assert row["mapping_hits"] == original["mapping_hits"] and row["near_match_candidates"] == original["near_match_candidates"]


@pytest.mark.parametrize('change', [
    {"transcript_id": "TX_NEUTRAL.3"}, {"reference_build": "assembly-B"},
    {"reference_annotation": "different-annotation"}, {"reference_identity": {"transcript_id": "TX_NEUTRAL.9", "sequence_sha256": "different"}}, {"strand": "-"},
    {"match_orientation": "reverse_complement"}, {"matched_window_sequence": "TCGTTGCAAC"},
    {"mismatch_count": 1, "mismatch_positions_in_supplied_sequence_1based": [1]},
    {"genomic_span": {"chrom": "7", "start": 702, "end": 711}},
])
def test_distinct_windows_or_counterevidence_are_never_collapsed(change):
    row = paired_row(); row["near_match_candidates"][0].update(change)
    inventory = row_evidence_inventory(row)["recorded_window_identity"]
    assert inventory["identified_window_group_count"] == 2


@pytest.mark.parametrize('change', [
    {"genomic_span": None}, {"matched_window_sequence": None}, {"strand": None},
    {"coordinate_system": "0based_half_open"}, {"mismatch_count": None},
    {"mismatch_count": True}, {"mismatch_count": 1, "mismatch_positions_in_supplied_sequence_1based": [True]},
    {"genomic_segments": [{"chrom": "7", "start": 702, "end": 711}]},
    {"matched_target_sequence": "TCGTTGCAAC"},
])
def test_incomplete_or_conflicting_identity_remains_unresolved(change):
    row = paired_row(); row["near_match_candidates"][0].update(change)
    inventory = row_evidence_inventory(row)["recorded_window_identity"]
    assert inventory["identified_window_group_count"] == 1
    assert inventory["unresolved_source_fields"] == ["near_match_candidates[0]"]
    assert not inventory["complete_for_retained_records"]


def test_spliced_bounding_span_cannot_replace_missing_blocks_or_hide_another_junction():
    row = paired_row(); hit = row["near_match_candidates"][0]
    hit.update(mapping_scope="spliced_cdna", rna_context="mature_spliced_cdna", cdna_start=41, cdna_end=50)
    assert recorded_window_identity(hit, "near_match") is None
    hit['genomic_segments'] = [{"chrom": "7", "start": 601, "end": 605}, {"chrom": "7", "start": 706, "end": 710}]
    first = recorded_window_identity(hit, "near_match")
    assert first is not None
    hit['genomic_segments'] = [{"chrom": "7", "start": 501, "end": 505}, {"chrom": "7", "start": 706, "end": 710}]
    assert recorded_window_identity(hit, "near_match") != first
    hit['genomic_segments'].reverse()
    assert recorded_window_identity(hit, "near_match") != first


def test_recognized_contig_and_nucleotide_notation_does_not_create_a_new_window():
    row = paired_row(); hit = row['near_match_candidates'][0]
    hit['genomic_span']['chrom'] = 'chr7'
    hit['matched_window_sequence'] = 'acguugcaac'
    assert row_evidence_inventory(row)['recorded_window_identity']['identified_window_group_count'] == 1


def test_other_input_rows_never_supply_aliases_or_missing_window_evidence():
    first = paired_row(); second = {"name": "another", "input_row_id": "row-B", "mapping_hits": [], "near_match_candidates": []}
    attach_measurement_references({"provided_asos": [first, second]})
    assert second['evidence_inventory']['recorded_window_identity']['identified_window_groups'] == []
    assert second['evidence_inventory']['retained_hypotheses'] == []
