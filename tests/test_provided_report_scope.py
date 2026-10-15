"""One evaluation cannot acquire another evaluation's splice scores or status."""
from copy import deepcopy
import json

from aso_agents import pipeline_tools as tools


def _foreign_result(run, rows):
    path = run / 'artifacts/design/v2_tasks/other_context/scored_tiles.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'rows': rows}))


def test_report_rows_ignore_other_task_scores_and_missing_status(tmp_path):
    row = {'name': 'same-molecule', 'input_row_id': 'input-2', 'sequence': 'ATGC',
           'splice_eval_summary': {'delta_deletion': -0.1, 'delta_names': 'acceptor@450'},
           'stage_results': {'splice_eval': {'status': 'pass'}}}
    payload = {'provided_asos': [row]}
    before = deepcopy(payload)
    _foreign_result(tmp_path, [{'name': 'same-molecule', 'Sequence': 'ATGC',
        'delta_deletion': 0.8, 'delta_names': 'donor@920',
        'splice_eval_status': 'skipped_no_nearby_splice_site'}])
    result = tools._summary_splice_rows_from_payload(payload, str(tmp_path))
    assert result == [row]
    assert 'splice_eval_status' not in result[0]
    assert tools._summary_splice_delta(result[0], 'delta_deletion') == -0.1
    assert tools._summary_splice_site_label(result[0]) == 'acceptor@450'
    assert payload == before


def test_no_bound_rows_does_not_fall_back_to_parent_run_results(tmp_path):
    _foreign_result(tmp_path, [{'name': 'elsewhere', 'delta_deletion': -0.9}])
    assert tools._summary_splice_rows_from_payload({}, str(tmp_path)) == []
    assert tools._summary_splice_rows_from_payload({'provided_asos': []}, str(tmp_path)) == []


def test_repeated_molecule_names_preserve_individual_evaluation_rows(tmp_path):
    rows = [
        {'name': 'repeated', 'input_row_id': 'one', 'sequence': 'ATGC',
         'splice_eval_summary': {'delta_deletion': 0.0, 'delta_names': 'donor@230'}},
        {'name': 'repeated', 'input_row_id': 'two', 'sequence': 'ATGC',
         'splice_eval_summary': {'splice_eval_status': 'skipped_no_nearby_splice_site'}},
    ]
    result = tools._summary_splice_rows_from_payload({'provided_asos': rows}, str(tmp_path))
    assert result == rows
    assert tools._summary_splice_delta(result[0], 'delta_deletion') == 0.0
    assert tools._summary_splice_delta(result[1], 'delta_deletion') is None


def test_complete_summary_uses_only_this_evaluation(tmp_path):
    _foreign_result(tmp_path, [{'name': 'probe', 'Sequence': 'ATGC',
        'delta_deletion': -0.95, 'delta_names': 'acceptor@999',
        'splice_eval_status': 'skipped_no_nearby_splice_site'}])
    payload = {'context_summary': {'gene_symbol': 'TARGET', 'transcript_id': 'REF.1'},
        'splice_related_request': True,
        'evaluation_targets': [{'exon_label': 4, 'desired_effect': 'skip'}],
        'provided_asos': [{'name': 'probe', 'sequence': 'ATGC', 'mapping_status': 'mapped',
            'splice_eval_summary': {'delta_deletion': 0.0, 'delta_names': 'acceptor@450'}}]}
    report = tools._provided_aso_summary_markdown(payload=payload, response_basename='assessment', run_dir=str(tmp_path))
    assert 'acceptor@450' in report
    assert 'acceptor@999' not in report
    assert 'skipped_no_nearby_splice_site' not in report
    assert 'scored 1/1 row(s) and skipped 0/1 row(s)' in report


def test_summary_preserves_skipped_requested_site_despite_local_score(tmp_path):
    _foreign_result(tmp_path, [{'name': 'probe', 'Sequence': 'ATGC',
        'delta_deletion': -0.95, 'delta_names': 'acceptor@999'}])
    payload = {'context_summary': {'gene_symbol': 'TARGET', 'transcript_id': 'REF.1'},
        'splice_related_request': True,
        'evaluation_targets': [{'exon_label': 4, 'desired_effect': 'skip'}],
        'provided_asos': [{'name': 'probe', 'sequence': 'ATGC', 'mapping_status': 'mapped',
            'splice_eval_summary': {'splice_eval_status': 'skipped_no_nearby_splice_site',
                                    'splice_eval_skip_reason': 'no_splice_sites_within_2000bp'}}]}
    report = tools._provided_aso_summary_markdown(payload=payload, response_basename='assessment', run_dir=str(tmp_path))
    assert 'acceptor@999' not in report
    assert 'Unscored: splice evaluation did not return a usable delta' in report
    assert 'scored 0/1 row(s) and skipped 1/1 row(s)' in report
