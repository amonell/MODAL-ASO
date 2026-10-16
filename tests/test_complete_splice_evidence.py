from copy import deepcopy
import json

import pytest

from aso_agents.evaluation_evidence import attach_measurement_references, bind_assessment_measurements
from aso_agents.evaluation_report import evaluation_markdown, write_evaluation_evidence
from aso_agents.splice_prediction_records import splice_prediction_records


def observation(row_id='input-1'):
    return {'name': 'repeated-name', 'input_row_id': row_id, 'splice_eval_summary': {
        'delta_inclusion_deletion': None,
        'splice_site_deltas': [
            {'label': 'acceptor@501', 'site_kind': 'acceptor', 'exon_index': 2,
             'reference_transcript_id': 'TX.2', 'reference_binding_status': 'unresolved',
             'acceptor': {'chrom': 'chr7', 'position': 501, 'strand': '+'},
             'evaluation_status': 'evaluated', 'delta_acceptor_deletion': 0.0,
             'delta_acceptor_replacement': -0.75, 'delta_inclusion_deletion': None},
            {'label': 'donor@705', 'site_kind': 'donor', 'exon_index': 4,
             'evaluation_status': 'skipped_outside_context', 'delta_donor_deletion': None},
        ]}}


def test_site_navigation_keeps_zero_missing_status_and_geometry_without_mutating_source():
    row = observation(); before = deepcopy(row)
    records = splice_prediction_records(row)
    assert len(records) == 3
    assert records[0]['numeric_measurements'] == {}
    assert records[1]['numeric_measurements'] == {'delta_acceptor_deletion': 0.0, 'delta_acceptor_replacement': -0.75}
    assert records[1]['recorded_identity']['acceptor'] == {'chrom': 'chr7', 'position': 501, 'strand': '+'}
    assert records[1]['unrecorded_metric_fields'] == ['delta_inclusion_deletion']
    assert records[2]['recorded_identity']['evaluation_status'] == 'skipped_outside_context'
    assert row == before


def test_repeated_labels_are_separate_records_with_separate_tokens():
    row = observation()
    row['splice_eval_summary']['splice_site_deltas'].append(deepcopy(row['splice_eval_summary']['splice_site_deltas'][0]))
    summary = attach_measurement_references({'provided_asos': [row]})
    records = summary['provided_asos'][0]['recorded_splice_predictions']
    tokens = [record['measurement_tokens'].get('delta_acceptor_replacement') for record in records]
    assert tokens[1] and tokens[3] and tokens[1] != tokens[3]
    assert records[1]['recorded_identity'] == records[3]['recorded_identity']


def test_site_token_binding_retains_exact_identity_and_rejects_cross_observation_borrowing():
    summary = attach_measurement_references({'provided_asos': [observation('a'), observation('b')]})
    token = summary['provided_asos'][0]['recorded_splice_predictions'][1]['measurement_tokens']['delta_acceptor_replacement']
    assessment = {'name': 'repeated-name', 'input_row_id': 'a', 'mechanistic_plausibility': 'Recorded delta_acceptor_replacement=' + token}
    result = bind_assessment_measurements({'evaluation_assessments': [assessment]}, summary)
    assert result['evaluation_assessments'][0]['mechanistic_plausibility'].endswith('=-0.75')
    evidence = summary['measurement_evidence'][token[2:-2]]
    assert evidence['field'] == 'splice_eval_summary.splice_site_deltas[0].delta_acceptor_replacement'
    assert evidence['recorded_identity']['reference_binding_status'] == 'unresolved'
    assessment['input_row_id'] = 'b'
    with pytest.raises(ValueError, match='different molecule'):
        bind_assessment_measurements({'evaluation_assessments': [assessment]}, summary)


def test_report_delivers_sites_even_when_model_omits_them_and_keeps_its_judgment():
    summary = {'provided_asos': [observation()]}
    response = {'evaluation_assessments': [{'name': 'repeated-name', 'input_row_id': 'input-1',
                                          'uncertainty': 'Original uncertain model judgment.'}]}
    before = deepcopy(response)
    report = evaluation_markdown(response, summary)
    for required in ('Recorded splice predictions', 'acceptor@501', 'donor@705', 'TX.2',
                     'unresolved', 'skipped_outside_context', '0.0', '-0.75', 'Inclusion proxy',
                     'Original uncertain model judgment.'):
        assert required in report
    assert response == before


def test_summary_primary_and_selected_measurements_never_inherit_one_anothers_label():
    row = {'name': 'probe', 'splice_eval_summary': {'primary_label': 'donor@901',
           'selected_label': 'acceptor@801', 'primary_delta_deletion': -0.4, 'selected_delta_deletion': 0.2}}
    report = evaluation_markdown({}, {'provided_asos': [row]})
    assert 'primary_label: donor@901; selected_label: acceptor@801' in report
    assert 'primary_delta_deletion' in report and 'selected_delta_deletion' in report


@pytest.mark.parametrize('value', ['-0.1, 0.2', '0.5', True, float('nan'), float('inf')])
def test_legacy_or_invalid_values_are_not_guessed_to_be_numeric_predictions(value):
    row = {'splice_eval_summary': {'delta_deletion': value}}
    assert splice_prediction_records(row)[0]['numeric_measurements'] == {}


def test_evidence_file_retains_every_original_site_and_numeric_citation(tmp_path):
    summary = attach_measurement_references({'provided_asos': [observation()]})
    token = summary['provided_asos'][0]['recorded_splice_predictions'][1]['measurement_tokens']['delta_acceptor_replacement']
    result = bind_assessment_measurements({'answer_summary': 'Recorded ' + token,
                                          'evaluation_assessments': []}, summary)
    result['response_evaluation_summary'] = summary
    paths = write_evaluation_evidence(tmp_path, result)
    evidence = json.loads((tmp_path/'evaluation_evidence.json').read_text())
    assert evidence['provided_asos'][0]['splice_eval_summary'] == observation()['splice_eval_summary']
    assert evidence['measurement_citations'][token[2:-2]]['recorded_identity']['label'] == 'acceptor@501'
    assert paths['evaluation_evidence_html']
