from copy import deepcopy
import csv
import json

import pytest

from aso_agents import pipeline_tools as p
from aso_agents.evaluation_evidence import attach_measurement_references, bind_assessment_measurements
from aso_agents.response_contracts import response_contract_errors


def panel(tmp_path):
    path = tmp_path / 'neutral.csv'
    path.write_text('name,sequence,chemistry,context\nshared,AACCGGTA,gapmer,one\nshared,AACCGGTA,gapmer,one\nshared,AACCGGTA,steric,two\n')
    return path


def test_file_aliases_reconcile_without_merging_physical_observations(tmp_path):
    path = panel(tmp_path)
    alias = tmp_path / 'alias.csv'; alias.symlink_to(path)
    source = {'request_text': f'Evaluate every observation from {path}.',
              'provided_aso_files': [str(path), str(alias)],
              'provided_asos': [{'name':'shared', 'sequence':'AACCGGTA', 'chemistry':'gapmer'}]}
    rows = p._normalize_provided_asos(source)
    assert len(rows) == 3
    assert [r['source_row'] for r in rows] == [2, 3, 4]
    assert len({r['input_row_id'] for r in rows}) == 3
    assert rows[0]['input_reconciliation'][0]['parsed_row']['sequence'] == 'AACCGGTA'
    assert rows[2]['chemistry'] == 'steric'
    assert [r['input_row_id'] for r in rows] == [r['input_row_id'] for r in p._normalize_provided_asos(source)]


def test_planner_notes_on_file_expansion_do_not_double_count_rows(tmp_path):
    path = tmp_path / 'sets.csv'
    path.write_text('set_id,candidate_id,molecule_5to3,intended_chemistry\n'
                    "A,F-A1,GCATAAGCCTATGCCTAGAT,5-10-5 2'-MOE/DNA full-PS gapmer\n"
                    "B,F-B2,GCATAAGCCTATGCCTAGAT,5-10-5 2'-MOE/DNA full-PS gapmer\n")
    source = {'request_text': f'Compare the physical records in {path}.',
              'provided_aso_files': [str(path)],
              'provided_asos': [
                  {'name': 'F-A1', 'sequence': 'GCATAAGCCTATGCCTAGAT',
                   'chemistry': "5-10-5 2'-MOE/DNA full-PS gapmer", 'platform': 'RNase H gapmer',
                   'notes': 'Set A candidate'},
                  {'name': 'F-B2', 'sequence': 'GCATAAGCCTATGCCTAGAT',
                   'chemistry': "5-10-5 2'-MOE/DNA full-PS gapmer", 'platform': 'RNase H gapmer',
                   'notes': 'Set B candidate'},
              ]}
    rows = p._normalize_provided_asos(source)
    assert len(rows) == 2
    assert [row['aso_id'] for row in rows] == ['F-A1', 'F-B2']
    assert [row['input_metadata']['set_id'] for row in rows] == ['A', 'B']
    assert all(len(row['input_reconciliation']) == 1 for row in rows)


@pytest.mark.parametrize('request_text,extra', [
    ('Also evaluate inline AACCGGTA as an additional observation.', {}),
    ('Evaluate the supplied observations.', {'input_origin':'inline_user'}),
    ('Evaluate the supplied observations.', {'context':'different'}),
    ('', {}),
])
def test_real_inline_or_different_context_observations_are_preserved(tmp_path, request_text, extra):
    rows = p._normalize_provided_asos({'provided_aso_files':[str(panel(tmp_path))],
        'request_text':request_text, 'provided_asos':[{'name':'shared','sequence':'AACCGGTA','chemistry':'gapmer',**extra}]})
    assert len(rows) == 4
    assert len({r['input_row_id'] for r in rows}) == 4


def test_duplicate_names_receive_their_own_evidence_and_exported_assessment(tmp_path):
    rows = p._normalize_provided_asos({'provided_aso_files':[str(panel(tmp_path))]})
    for index, row in enumerate(rows):
        row['splice_eval_summary'] = {'delta_deletion': index / 10}
    summary = p._compact_existing_aso_evaluation_for_response({'provided_asos': rows})
    response = {'evaluation_assessments':[
        {'name':r['name'],'input_row_id':r['input_row_id'], 'mechanistic_plausibility':r['measurement_tokens']['delta_deletion'],
         'ranking_rationale':f'Original assessment {index}.'} for index,r in enumerate(summary['provided_asos'])]}
    assert not response_contract_errors(response, {'task_type':'evaluate_existing_aso_set','evaluation_summary':summary})
    rendered = bind_assessment_measurements(response, summary)
    rendered['evaluation_summary'] = {'provided_asos':rows}
    exported = p._provided_aso_export_rows(rendered)
    assert [r['model_assessment_status'] for r in exported] == ['provided'] * 3
    assert [r['model_assessment']['mechanistic_plausibility'] for r in exported] == ['0.0','0.1','0.2']
    assert all(r['input_row_id'] in rendered['markdown'] for r in rows)
    bad = deepcopy(response)
    bad['evaluation_assessments'][1]['mechanistic_plausibility'] = '{{M001_delta_deletion}}'
    with pytest.raises(ValueError, match='different molecule/input row'):
        bind_assessment_measurements(bad, summary)
    bad = deepcopy(response); del bad['evaluation_assessments'][1]['input_row_id']
    assert response_contract_errors(bad, {'task_type':'evaluate_existing_aso_set','evaluation_summary':summary})
    bad = deepcopy(response); bad['evaluation_assessments'].append(deepcopy(bad['evaluation_assessments'][0]))
    assert any('received 2' in e for e in response_contract_errors(bad, {'task_type':'evaluate_existing_aso_set','evaluation_summary':summary}))


def test_supplied_json_cannot_replace_physical_row_identity(tmp_path):
    path = tmp_path / 'observations.json'
    path.write_text(json.dumps([{'name':'sample','sequence':'AACCGGTA','source_file':'claimed.csv','source_row':99}]*2))
    rows = p._normalize_provided_asos({'provided_aso_files':[str(path)]})
    assert [r['source_row'] for r in rows] == [1, 2]
    assert all(r['source_file'] == str(path) for r in rows)
    assert all(r['input_metadata']['declared_source_row'] == 99 for r in rows)
    assert len({r['input_row_id'] for r in rows}) == 2
