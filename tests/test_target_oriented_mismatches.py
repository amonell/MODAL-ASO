from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.alignment_identity import recorded_window_identity
from aso_agents.alignment_reporting import alignment_geometry
from aso_agents.comparison_evidence import comparison_binding_evidence
from aso_agents.mapping_evidence import cdna_near_matches, mismatch_annotations


@pytest.mark.parametrize('orientation, supplied_positions', [
    ('reverse_complement', [1, 6]), ('input_sequence', [2, 7]),
])
def test_bases_and_positions_follow_target_direction(orientation, supplied_positions):
    result = mismatch_annotations('ACGTCGA', 'ATGTCGT', orientation)
    assert result['mismatch_positions_in_supplied_sequence_1based'] == supplied_positions
    assert result['mismatch_positions_in_target_sequence_1based'] == [2, 7]
    assert [(x['target_position_1based'], x['reference_base'], x['aligned_query_base'])
            for x in result['mismatch_details_in_target_orientation']] == [(2, 'C', 'T'), (7, 'A', 'T')]
    assert 'reference-transcript' in result['mismatch_annotation_coordinate_system']


def test_exact_alignment_has_explicit_empty_difference_inventory():
    result = mismatch_annotations('acgta', 'ACGTA', 'reverse_complement')
    assert result['mismatch_positions_in_target_sequence_1based'] == []
    assert result['mismatch_positions_in_supplied_sequence_1based'] == []
    assert result['mismatch_details_in_target_orientation'] == []


@pytest.mark.parametrize('reference,query,orientation', [
    ('ACGT', 'ACG', 'input_sequence'), ('ACGT', 'ACGT', 'unknown'),
])
def test_unknown_alignment_semantics_are_rejected(reference, query, orientation):
    with pytest.raises(ValueError):
        mismatch_annotations(reference, query, orientation)


@pytest.mark.parametrize('strand', ['+', '-'])
def test_mature_alignment_and_report_keep_target_and_oligo_positions(strand):
    entry = {'sequence': 'TCGTAACC', 'reverse_complement_sequence': 'GGTTACGA'}
    transcript = {'transcript_id': 'TX.2', 'strand': strand}
    segments = [('7', 101, 104), ('7', 201, 204)]
    if strand == '-':
        segments.reverse()
    matches = cdna_near_matches(entry, {'TX': transcript}, {'TX': 'GGTTACGT'},
                                lambda *args: segments, 'TX.2')
    hit = next(h for h in matches if h['match_orientation'] == 'reverse_complement')
    assert hit['mismatch_count'] == 1
    assert hit['mismatch_positions_in_supplied_sequence_1based'] == [1]
    assert hit['mismatch_positions_in_target_sequence_1based'] == [8]
    assert hit['mismatch_details_in_target_orientation'] == [{
        'target_position_1based': 8, 'reference_base': 'T', 'aligned_query_base': 'A',
        'supplied_sequence_position_1based': 1,
    }]
    geometry = alignment_geometry(hit)
    representative = comparison_binding_evidence({'near_match_candidates': [hit]})['groups'][0]['representative']
    for result in [geometry, representative]:
        assert result['mismatch_details_in_target_orientation'] == hit['mismatch_details_in_target_orientation']
        assert result['mismatch_positions_in_target_sequence_1based'] == [8]
    identity = recorded_window_identity(hit, 'near_match')
    assert identity['mismatch_positions_1based'] == [1]  # legacy field stays explicitly oligo oriented
    assert identity['mismatch_positions_orientation'] == 'supplied_sequence_5prime_to_3prime'
    assert identity['mismatch_positions_in_target_sequence_1based'] == [8]
    conflict = deepcopy(hit)
    conflict['mismatch_positions_in_target_sequence_1based'] = [1]
    assert recorded_window_identity(conflict, 'near_match') is None


@pytest.mark.parametrize('strand', ['+', '-'])
def test_locus_alignment_uses_the_same_orientation_contract(monkeypatch, strand):
    monkeypatch.setattr(pipeline, '_fetch_transcript_locus_sequence', lambda **kwargs: {
        'transcript_oriented_sequence': 'GGTTACGT', 'transcript_id': 'TX.2',
        'strand': strand, 'chrom': '7', 'locus_start': 101, 'locus_end': 108,
    })
    matches = pipeline._best_locus_near_matches(
        entry={'sequence': 'TCGTAACC', 'reverse_complement_sequence': 'GGTTACGA'},
        isoform_transcripts={'TX': {'strand': strand}}, runtime=None, genome=True,
        preferred_transcript_id='TX.2')
    hit = next(h for h in matches if h['match_orientation'] == 'reverse_complement')
    assert hit['mismatch_positions_in_supplied_sequence_1based'] == [1]
    assert hit['mismatch_positions_in_target_sequence_1based'] == [8]
    assert hit['mismatch_details_in_target_orientation'][0]['reference_base'] == 'T'
    assert hit['genomic_span'] == {'chrom': '7', 'start': 101, 'end': 108, 'length': 8}
