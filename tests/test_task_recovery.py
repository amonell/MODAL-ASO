import json
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.task_recovery import reset_task_and_dependents


@pytest.mark.parametrize('task_type,tool', [
    ('evaluate_existing_aso_set', 'tool_evaluate_existing_aso_set'),
    ('compare_aso_sets', 'tool_compare_aso_sets'),
    ('analyze_artifact', 'tool_analyze_artifact'),
    ('trace_filter_decisions', 'tool_trace_filter_decisions'),
    ('answer_question', 'tool_answer_question'),
    # This tool is intentionally absent from the registered public tool set;
    # recovery must retain the guarded inspection action, not enter design.
    ('sandbox_analysis', 'inspect_run_outputs'),
    ('build_composite_report', 'tool_build_composite_report'),
])
def test_pipeline_restart_keeps_task_workflow_and_preserves_inputs(tmp_path, task_type, tool):
    table = tmp_path / 'supplied.csv'
    table.write_text('id,sequence\nrow1,ACGTACGTACGTACGTACGT\n')
    task = {'task_id': 'active', 'task_type': task_type, 'label': 'Process the supplied evidence',
            'native': task_type != 'sandbox_analysis',
            'context': {'gene_symbol': 'GENE_A', 'species': 'human',
                        'question': 'Assess the supplied evidence.', 'artifact_refs': [str(table)],
                        'provided_asos': [{'name': 'row1', 'sequence': 'ACGTACGTACGTACGTACGT'}]}}
    envelope = pipeline._build_request_envelope(request_text='Process the supplied evidence.',
        request_id='recovery_control', structured_goal={}, raw_envelope={
            'request_mode': 'single_task', 'workflow_name': task_type,
            'response_mode': 'analysis', 'tasks': [task]})
    state = {'version': '2', 'request': 'Process the supplied evidence.',
             'status': 'failed', 'phase': 'task_response_failed', 'run_dir': str(tmp_path),
             'request_envelope': envelope, 'current_task_id': 'active',
             'last_error': {'kind': 'task_response_failed'}, 'pending_questions': [],
             'events': [], 'stage_attempts': {}, 'task_statuses': {}}
    pipeline._ensure_v2_request_metadata(state)
    state['task_statuses']['active']['status'] = 'failed'
    state_path = tmp_path / 'state.json'
    state_path.write_text(json.dumps(state))
    prerequisite = tmp_path / 'generated_aso_set.json'
    prerequisite.write_text('{"source":"upstream input"}')
    result = pipeline.restart_from_stage(str(state_path), 'pipeline', 'Repair the response.')
    assert result['status'] == 'ok'
    saved = json.loads(state_path.read_text())
    assert pipeline._derive_next_actions(saved)['next_actions'][0]['tool'] == tool
    assert prerequisite.read_text() == '{"source":"upstream input"}'
    assert table.read_text().startswith('id,sequence')
    assert not saved.get('last_error')


def test_pipeline_restart_still_resumes_design_at_reference_preflight(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({'version': '2', 'status': 'failed', 'phase': 'pipeline_failed',
        'run_dir': str(tmp_path), 'events': [], 'stage_attempts': {},
        'last_error': {'kind': 'pipeline_failed'}}))
    assert pipeline.restart_from_stage(str(path), 'pipeline', 'Retry design stage.')['status'] == 'ok'
    saved = json.loads(path.read_text())
    assert saved['status'] == 'resolution_ready'
    assert pipeline._derive_next_actions(saved)['next_actions'][0]['tool'] == 'tool_reference_preflight'


def test_restart_invalidates_dependents_and_preserves_completed_prerequisites():
    tasks = [{'task_id': key, 'task_type': 'analyze_artifact'} for key in ['upstream', 'active', 'sibling', 'dependent', 'report']]
    tasks[3]['depends_on'] = ['active']
    state = {'task_graph': {'tasks': tasks, 'dependencies': [
        {'upstream_task_id': 'upstream', 'downstream_task_id': 'active'},
        {'upstream_task_id': 'dependent', 'downstream_task_id': 'report'}]},
        'task_statuses': {t['task_id']: {'status': 'completed'} for t in tasks},
        'task_artifacts': {t['task_id']: {'json': t['task_id'] + '.json'} for t in tasks}}
    previous = deepcopy(state)
    assert reset_task_and_dependents(state, 'active') == ['active', 'dependent', 'report']
    for tid in ['upstream', 'sibling']:
        assert state['task_statuses'][tid] == previous['task_statuses'][tid]
        assert state['task_artifacts'][tid] == previous['task_artifacts'][tid]
    for tid in ['active', 'dependent', 'report']:
        assert state['task_statuses'][tid]['status'] == 'pending'
        assert tid not in state['task_artifacts']
        assert state['task_restart_history'][0]['previous_results'][tid]['artifacts'] == previous['task_artifacts'][tid]


def test_unknown_restart_task_cannot_invalidate_other_results():
    state = {'task_graph': {'tasks': []}, 'task_artifacts': {'retained': {'json': 'evidence.json'}}}
    before = deepcopy(state)
    with pytest.raises(ValueError, match='unknown task'):
        reset_task_and_dependents(state, 'missing')
    assert state == before
