"""Bare attached identifiers must not become phantom molecules."""
from copy import deepcopy
import json

import pytest

from aso_agents import pipeline_tools as pipeline


def write_rows(tmp_path, rows):
    path = tmp_path/'observations.json'
    path.write_text(json.dumps(rows))
    return path


def test_unique_bare_file_ids_resolve_with_source_provenance_and_stable_physical_ids(tmp_path):
    path = write_rows(tmp_path, [
        {'name': 'sample-01', 'sequence': 'AACCGGTA', 'chemistry': 'steric'},
        {'name': 'sample-02', 'sequence': 'AACCGGTA', 'chemistry': 'gapmer'},
        {'name': '17', 'sequence': 'GTACGTAC', 'chemistry': 'steric'},
    ])
    request = {'provided_aso_files': [str(path)], 'provided_asos': ['sample-01', ' sample-02 ', '17']}
    original = deepcopy(request)
    before = pipeline._normalize_provided_asos({'provided_aso_files': [str(path)]})
    rows = pipeline._normalize_provided_asos(request)
    assert len(rows) == 3 and all(row['valid_bases'] for row in rows)
    assert [row['input_row_id'] for row in rows] == [row['input_row_id'] for row in before]
    assert [row['chemistry'] for row in rows] == ['steric', 'gapmer', 'steric']
    for position, row in enumerate(rows):
        alias = row['input_reconciliation'][0]
        assert alias['kind'] == 'attached_row_id_reference' and alias['inline_position'] == position
        assert alias['matching_source_rows'] == [{'source_file': str(path), 'source_row': position + 1}]
    assert request == original


@pytest.mark.parametrize('reference', ['ACGT', 'A-CG U', 'RYSWKM', '/5Phos/ACGT'])
def test_sequence_like_string_is_preserved_even_when_equal_to_a_file_id(tmp_path, reference):
    path = write_rows(tmp_path, [{'name': reference, 'sequence': 'TTTTGGGG'}])
    rows = pipeline._normalize_provided_asos({'provided_aso_files': [str(path)], 'provided_asos': [reference]})
    assert len(rows) == 2
    assert rows[1]['raw_sequence'] == reference
    assert not rows[0].get('input_reconciliation')


def test_ambiguous_unknown_and_explicit_malformed_sequences_remain_visible(tmp_path):
    path = write_rows(tmp_path, [{'name': 'sample-01', 'sequence': 'AACCGGTA'}] * 2)
    rows = pipeline._normalize_provided_asos({'provided_aso_files': [str(path)],
        'provided_asos': ['sample-01', 'missing-02', {'name': 'explicit', 'sequence': 'sample-01', 'input_origin': 'inline_user'}]})
    assert len(rows) == 5 and len({row['input_row_id'] for row in rows}) == 5
    assert [row['valid_bases'] for row in rows] == [True, True, False, False, False]
    assert not any(row.get('input_reconciliation') for row in rows)


def test_bare_reference_does_not_repair_a_malformed_physical_file_observation(tmp_path):
    path = write_rows(tmp_path, [{'name': 'sample-01', 'sequence': 'INVALID123'}])
    rows = pipeline._normalize_provided_asos({'provided_aso_files': [str(path)], 'provided_asos': ['sample-01']})
    assert len(rows) == 1 and not rows[0]['valid_bases']
    assert rows[0]['raw_sequence'] == 'INVALID123' and rows[0]['source_row'] == 1


def test_repeated_references_do_not_change_observation_count_or_collapse_real_file_rows(tmp_path):
    path = write_rows(tmp_path, [{'name': 'sample-01', 'sequence': 'AACCGGTA'},
                                {'name': 'sample-02', 'sequence': 'AACCGGTA'}])
    rows = pipeline._normalize_provided_asos({'provided_aso_files': [str(path)],
                                             'provided_asos': ['sample-01', 'sample-01', 'sample-02']})
    assert len(rows) == 2 and len({row['input_row_id'] for row in rows}) == 2
    assert [len(row['input_reconciliation']) for row in rows] == [2, 1]
