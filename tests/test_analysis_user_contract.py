from copy import deepcopy
import json
from pathlib import Path

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.analysis_review import assess_review


USER_REQUEST = ('  Write the total in analysis.json with fields total and n_used.\n'
                'Separately, render a plot in chart.svg.  ')
TASK_LABEL = 'Summarize the numbers'
QUOTE = 'fields total and n_used'


def prepared_task(tmp_path, monkeypatch, *, request=USER_REQUEST,
                  mode='composite', task_type='sandbox_analysis'):
    task = {'task_id': 'numbers', 'task_type': task_type, 'label': TASK_LABEL,
            'context': {}, 'depends_on': []}
    envelope = {'schema_version': 'v3', 'request_mode': mode, 'tasks': [task,
        {'task_id': 'chart', 'task_type': 'sandbox_analysis', 'label': 'Plot the observations'}],
        'final_outputs': [
            {'task_id': 'numbers', 'kind': 'file', 'path': str(tmp_path / 'analysis.json')},
            {'task_id': 'chart', 'kind': 'file', 'path': str(tmp_path / 'chart.svg')}],
        'request_text': request}
    state = {'request': request, 'request_envelope': envelope}
    prepared = {'state_path': str(tmp_path / 'state.json'), 'state': state,
                'run_dir': str(tmp_path), 'task_id': 'numbers', 'task': task,
                'task_request_envelope': pipeline._task_scoped_request_envelope(envelope, task),
                'task_context': {}}
    monkeypatch.setattr(pipeline, '_task_output_dir', lambda *a, **k: tmp_path / 'task')
    monkeypatch.setattr(pipeline, '_build_sandbox_declared_inputs', lambda *a: {'observations': [2, 3]})
    monkeypatch.setattr(pipeline, '_should_finalize_current_task', lambda *a: False)
    monkeypatch.setattr(pipeline, '_complete_v2_task', lambda **kw: kw)
    monkeypatch.setattr(pipeline, '_tool_failure', lambda **kw: kw)
    return prepared


@pytest.mark.parametrize('mode,task_type', [
    ('composite', 'sandbox_analysis'),
    ('single_task', 'sandbox_analysis'),
    ('composite', 'trace_filter_decisions'),
])
def test_coder_and_runtime_keep_original_request_without_expanding_outputs(tmp_path, monkeypatch, mode, task_type):
    from aso_agents import analysis_execution
    prepared = prepared_task(tmp_path, monkeypatch, mode=mode, task_type=task_type)
    before = deepcopy(prepared)
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return {'ok': False, 'error': 'captured without generation', 'attempts': []}
    monkeypatch.setattr(analysis_execution, 'execute_analysis_program', capture)
    pipeline._execute_sandbox_analysis(prepared, tool_name='analysis')
    call, = calls
    payload = call['payload']
    manifest = json.loads((tmp_path / 'task/sandbox/declared_inputs.json').read_text())
    assert payload['request_text'] == USER_REQUEST
    assert payload['request_envelope']['request_text'] == USER_REQUEST
    assert TASK_LABEL in payload['task_request_text']
    assert manifest['request_text'] == USER_REQUEST
    assert manifest['task_request_text'] == payload['task_request_text']
    assert manifest['request_envelope'] == payload['request_envelope']
    assert call['required_outputs'] == [tmp_path / 'analysis.json']
    assert [t['task_id'] for t in payload['request_envelope']['tasks']] == ['numbers']
    assert prepared == before


@pytest.mark.parametrize('parent_text', [None, '', '  '])
def test_legacy_missing_parent_request_retains_scoped_instruction(tmp_path, monkeypatch, parent_text):
    from aso_agents import analysis_execution
    prepared = prepared_task(tmp_path, monkeypatch, request=parent_text)
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return {'ok': False, 'error': 'captured without generation', 'attempts': []}
    monkeypatch.setattr(analysis_execution, 'execute_analysis_program', capture)
    pipeline._execute_sandbox_analysis(prepared, tool_name='analysis')
    payload = calls[0]['payload']
    assert payload['request_text'] == TASK_LABEL
    assert payload['task_request_text'] == TASK_LABEL
    manifest = json.loads((tmp_path / 'task/sandbox/declared_inputs.json').read_text())
    assert manifest['request_text'] == TASK_LABEL


def test_original_requirement_reaches_review_and_repairs_actual_export(tmp_path, monkeypatch):
    prepared = prepared_task(tmp_path, monkeypatch)
    generated, reviewed = [], []
    def generate(payload):
        generated.append(deepcopy(payload))
        result = {'total': 5}
        if len(generated) > 1:
            result['n_used'] = 2
        return {'python_code': 'import json\nfrom pathlib import Path\n'
            + f'data = {result!r}\n'
            + f'Path({str(tmp_path / "analysis.json")!r}).write_text(json.dumps(data))\n'
            + "Path('sandbox_result.json').write_text(json.dumps(data))\n"
            + "Path('sandbox_summary.md').write_text('Computed the total from the two observations.')\n"}
    def review(payload):
        reviewed.append(deepcopy(payload))
        output = json.loads((tmp_path / 'analysis.json').read_text())
        # A checker cannot establish the omitted clause when it receives only a label.
        missing = QUOTE in payload['request_text'] and 'n_used' not in output
        return {'program_sha256': payload['program_sha256'], 'checks': [{
            'requirement_quote': QUOTE, 'obligation': 'Export the number of observations used.',
            'status': 'violated' if missing else 'satisfied',
            'implementation_evidence': 'The actual exported keys are ' + repr(list(output)),
            'problem': 'The required observation count is missing.' if missing else None,
            'suggested_fix': 'Include the observed input count in the JSON export.' if missing else None}]}
    monkeypatch.setattr(pipeline, '_run_sandbox_analysis_sync', generate)
    monkeypatch.setattr(pipeline, '_run_analysis_contract_review_sync', review)
    result = pipeline._execute_sandbox_analysis(prepared, tool_name='analysis')
    assert result['status'] == 'sandbox_analysis_ready'
    assert len(generated) == len(reviewed) == 2
    assert json.loads((tmp_path / 'analysis.json').read_text()) == {'total': 5, 'n_used': 2}
    assert not (tmp_path / 'chart.svg').exists()
    assert all(p['request_text'] == USER_REQUEST and p['task_request_text'] == TASK_LABEL for p in reviewed)
    first = json.loads((tmp_path / 'task/sandbox/execution_attempts/01/contract_review.json').read_text())
    assert first['status'] == 'needs_repair'
    assert first['findings'][0]['requirement_quote'] == QUOTE
    assert QUOTE in generated[1]['execution_repair']['error']
    # A planner label is scope metadata; it cannot invent a user requirement.
    raw = deepcopy(first['raw_review'])
    raw['checks'][0]['requirement_quote'] = TASK_LABEL
    code = (tmp_path / 'task/sandbox/execution_attempts/01/generated_code.py').read_text()
    assert not assess_review(raw, code, USER_REQUEST)['findings']
