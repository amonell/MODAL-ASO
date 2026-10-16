"""A complete planner preview is provenance, not additional physical molecules."""
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline


def source(tmp_path, sequences=None):
    path = tmp_path / 'observations.csv'
    path.write_text('name,sequence,chemistry,batch\n'
                    'repeat,AACCGGTA,gapmer,one\n'
                    'repeat,AACCGGTA,gapmer,two\n'
                    'variant,AACCGGTA,steric,three\n'
                    'other,TGCACGAT,steric,four\n')
    return {
        'request_text': 'Evaluate every physical observation in the attached table.',
        'provided_aso_files': [str(path)],
        'provided_asos': sequences if sequences is not None else
                         ['TGCACGAT', 'AACCGGTA', 'AACCGGTA', 'AACCGGTA'],
    }


def test_complete_sequence_preview_preserves_file_rows_and_all_origins(tmp_path):
    request = source(tmp_path)
    original = deepcopy(request)
    file_only = pipeline._normalize_provided_asos({**request, 'provided_asos': []})
    rows = pipeline._normalize_provided_asos(request)
    assert len(rows) == 4
    assert request == original
    assert [r['input_row_id'] for r in rows] == [r['input_row_id'] for r in file_only]
    for row, physical in zip(rows, file_only):
        assert {k:v for k,v in row.items() if k != 'input_reconciliation'} == physical
    aliases = [a for row in rows for a in row.get('input_reconciliation', [])]
    assert len(aliases) == 2
    assert sorted(i['inline_position'] for a in aliases for i in a['parsed_items']) == list(range(4))
    assert sorted(i['source_row'] for a in aliases for i in a['matching_source_rows']) == [2, 3, 4, 5]
    assert all(a['kind'] == 'parsed_file_sequence_inventory' for a in aliases)
    assert len(aliases[0]['matching_source_rows']) == len(aliases[0]['parsed_items'])
    assert [r['chemistry'] for r in rows] == ['gapmer', 'gapmer', 'steric', 'steric']


@pytest.mark.parametrize('preview', [
    ['AACCGGTA'],  # A subset does not attest a complete file expansion.
    ['TGCACGAT', 'AACCGGTA', 'AACCGGTA', 'TGCACGAT'],  # Multiplicity matters.
    ['TGCACGAT', 'AACCGGTA', 'AACCGGTA', 'AACCGGTT'],
    ['TGCACGAT', 'AACCGGTA', 'AACCGGTA', 'AACCGGTA', 'AACCGGTA'],
    ['TGCACGAT', 'AACCGGTA', 'AACCGGTA', {'sequence':'AACCGGTA', 'input_origin':'inline_user'}],
])
def test_partial_changed_or_explicit_observations_are_not_collapsed(tmp_path, preview):
    rows = pipeline._normalize_provided_asos(source(tmp_path, preview))
    assert len(rows) == 4 + len(preview)
    assert len({r['input_row_id'] for r in rows}) == len(rows)


@pytest.mark.parametrize('request_text', [
    '',
    'Also evaluate AACCGGTA as an inline molecule.',
    'Also evaluate the inline DNA string A A C-CG GTA.',
])
def test_original_inline_sequence_or_missing_request_preserves_inventory(tmp_path, request_text):
    request = source(tmp_path)
    request['request_text'] = request_text
    assert len(pipeline._normalize_provided_asos(request)) == 8


def test_same_complete_inventory_across_distinct_files_retains_both(tmp_path):
    a, b = tmp_path / 'first.csv', tmp_path / 'second.csv'
    for path in (a, b):
        path.write_text('name,sequence,chemistry\nsame,AACCGGTA,PS\n')
    request = {'provided_aso_files': [str(a), str(b)],
               'provided_asos': ['AACCGGTA', 'AACCGGTA'],
               'request_text': 'Evaluate every row in the two attached tables.'}
    rows = pipeline._normalize_provided_asos(request)
    assert len(rows) == 2
    assert [r['source_file'] for r in rows] == [str(a), str(b)]
    assert len({r['input_row_id'] for r in rows}) == 2
    assert len(rows[0]['input_reconciliation'][0]['matching_source_rows']) == 2


def test_no_file_preview_does_not_change_inline_inputs(tmp_path):
    request = source(tmp_path)
    request['provided_aso_files'] = []
    rows = pipeline._normalize_provided_asos(request)
    assert len(rows) == 4
    assert [r['sequence'] for r in rows] == request['provided_asos']


def test_task_scoped_native_input_uses_original_request_and_file_provenance(tmp_path):
    request = source(tmp_path)
    task = {'task_id': 'evaluate_panel', 'task_type': 'evaluate_existing_aso_set',
            'context': {k:v for k,v in request.items() if k != 'request_text'}}
    envelope = {'request_text': request['request_text'], 'default_context': {}, 'tasks': [task]}
    scoped = pipeline._task_scoped_request_envelope(envelope, task)
    assert len(pipeline._normalize_provided_asos(scoped)) == 4


def test_explicit_whole_inline_panel_keeps_its_chemistry_and_origins(tmp_path):
    request = source(tmp_path)
    request['provided_asos'] = [
        {'name':f'inline_{index}', 'sequence':sequence, 'chemistry':'PMO', 'input_origin':'inline_user'}
        for index,sequence in enumerate(request['provided_asos'])]
    rows = pipeline._normalize_provided_asos(request)
    assert len(rows) == 8
    assert [row['chemistry'] for row in rows[-4:]] == ['PMO'] * 4
    assert all(row['input_origin'] == 'inline_user' for row in rows[-4:])


@pytest.mark.parametrize('sequence', ['AACCGGTN', 'AACCGGUA', '[mA]ACCGGTA'])
def test_non_dna_or_modified_notation_is_not_simplified_into_a_preview(tmp_path, sequence):
    path = tmp_path / 'notation.csv'
    path.write_text(f'name,sequence\noriginal,{sequence}\n')
    request = {'provided_aso_files':[str(path)], 'provided_asos':[sequence],
               'request_text':'Evaluate the supplied observations.'}
    file_rows = pipeline._normalize_provided_asos({**request, 'provided_asos':[]})
    # Unsupported file syntax can already be excluded by the existing reader;
    # this repair must not additionally absorb the independent inline input.
    rows = pipeline._normalize_provided_asos(request)
    assert len(rows) == len(file_rows) + 1
    assert not any(item['kind'] == 'parsed_file_sequence_inventory'
                   for row in rows for item in row.get('input_reconciliation', []))
