from copy import deepcopy
import pytest
from aso_agents.transcript_metadata import resolved_transcript_metadata, reference_annotation_flags


@pytest.mark.parametrize('strand', [1, -1])
def test_resolved_structure_replaces_conflicting_selection_coordinates(strand):
    source = {'id': 'TX', 'version': 4, 'versioned_id': 'TX.4', 'reference_source': 'local_gtf',
              'strand': strand, 'seq_region_name': '8',
              'exons_tx': [{'start': 101, 'end': 110}, {'start': 201, 'end': 220}]}
    earlier = {'id': 'TX', 'version': 5, 'versioned_id': 'TX.5', 'reference_source': 'rest',
               'start': 90, 'end': 220, 'length': 131, 'cdna_length': 41, 'exon_count': 2,
               'five_prime_terminal_exon_start': 90, 'five_prime_terminal_exon_length': 21,
               'protein_length_aa': 13, 'is_canonical': True}
    before = deepcopy((source, earlier))
    result = resolved_transcript_metadata(source, earlier)
    assert (source, earlier) == before
    assert result['versioned_id'] == 'TX.4' and result['reference_source'] == 'local_gtf'
    assert (result['start'], result['end'], result['length'], result['cdna_length']) == (101, 220, 120, 30)
    assert result['five_prime_terminal_exon_start'] == (101 if strand == 1 else 201)
    assert result['five_prime_terminal_exon_length'] == (10 if strand == 1 else 20)
    assert result['selection_snapshot']['record'] == earlier
    assert result['is_canonical'] is None and 'protein_length_aa' not in result


def test_missing_resolved_structure_does_not_relabel_old_measurements():
    result = resolved_transcript_metadata({'id': 'TX', 'versioned_id': 'TX.4'},
                                         {'start': 90, 'cdna_length': 41, 'versioned_id': 'TX.5'})
    assert result['versioned_id'] == 'TX.4'
    assert 'start' not in result and 'cdna_length' not in result


@pytest.mark.parametrize('current,previous', [(True, False), (False, True), (None, True), (None, False)])
def test_annotation_flags_are_bound_to_current_reference_without_changing_prior_provenance(current, previous):
    structure = {'id': 'TX', 'versioned_id': 'TX.4',
                 'annotation_features': {'is_mane_select': current, 'is_canonical': current}}
    earlier = {'id': 'TX', 'versioned_id': 'TX.5', 'is_mane_select': previous,
               'MANE_Select': previous, 'is_canonical': previous}
    before = deepcopy((structure, earlier))
    result = resolved_transcript_metadata(structure, earlier)
    assert result['is_mane_select'] is current and result['is_canonical'] is current
    assert 'MANE_Select' not in result
    assert result['is_mane_plus_clinical'] is None
    assert result['selection_snapshot']['record'] == earlier
    assert (structure, earlier) == before


@pytest.mark.parametrize('payload,expected', [
    ({}, None), ({'is_mane_select': False}, False), ({'is_mane_select': True}, True),
    ({'is_mane_select': 'false'}, False), ({'MANE_Select': '1'}, True),
    ({'is_mane_select': False, 'MANE_Select': True}, None),
    ({'is_mane_select': 'unavailable'}, None), ({'is_mane_select': []}, None),
])
def test_annotation_absence_false_malformed_and_conflicting_values_are_distinct(payload, expected):
    assert reference_annotation_flags(payload)['is_mane_select'] is expected


def test_native_structure_retains_flags_from_the_same_record_as_its_exons():
    from aso_agents.transcript_chooser_agent import _transcript_structure_from_payload
    payload = {'id': 'TX', 'version': 4, 'reference_source': 'local_gtf',
               'reference_annotation': 'annotation-A.gtf', 'seq_region_name': '7',
               'start': 101, 'end': 110, 'strand': 1, 'Exon': [{'start': 101, 'end': 110}],
               'is_canonical': 1, 'is_mane_select': True, 'is_mane_plus_clinical': False}
    before = deepcopy(payload)
    structure = _transcript_structure_from_payload(payload, 'TX.4')
    assert structure['annotation_features'] == {'is_canonical': True, 'is_mane_select': True,
                                               'is_mane_plus_clinical': False}
    result = resolved_transcript_metadata(structure, {'is_mane_select': False})
    assert result['reference_annotation'] == 'annotation-A.gtf' and result['is_mane_select'] is True
    assert result['cdna_length'] == 10 and payload == before


def test_candidate_listing_and_model_sanitization_preserve_unknown_flags():
    from aso_agents.transcript_chooser_agent import _extract_transcript_rows, _sanitize_transcript_records
    payload = {'id': 'TX', 'start': 101, 'end': 110, 'strand': 1,
               'Exon': [{'start': 101, 'end': 110}], 'is_canonical': 0}
    rows = _extract_transcript_rows({'Transcript': [payload]})
    for row in (rows[0], _sanitize_transcript_records(rows)[0]):
        assert row['is_canonical'] is False
        assert row['is_mane_select'] is None and row['is_mane_plus_clinical'] is None
