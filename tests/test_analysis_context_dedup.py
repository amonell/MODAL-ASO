import asyncio
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.analysis_repair import code_digest, plan_digest
from aso_agents.analysis_review import program_digest, review_schema


def context_payload():
    context = {
        'request_text': 'Compare all sensor readings; retain zero and unknown μ values.',
        'task_request_text': 'Compute the supplied reading counts and export observations.json.',
        'request_summary': 'Compare readings without dropping observations.',
    }
    envelope = {
        'request_text': context['request_text'], 'request_summary': context['request_summary'],
        'tasks': [{'id': 'compare', 'label': context['task_request_text']}],
        'final_outputs': [{'path': 'observations.json'}],
        'future_fields': {'zero': 0, 'missing': None, 'confirmed': False},
    }
    return {**context, 'request_envelope': envelope,
            'declared_inputs': {**context, 'request_envelope': deepcopy(envelope),
                                'readings': [0, None, 7], 'future_input': {'empty': []}}}


def assert_compact_context(prompt, original):
    for key in ['request_text', 'task_request_text', 'request_summary']:
        assert prompt[key] == original[key]
        assert key not in prompt['declared_inputs']
    assert 'request_envelope' not in prompt['declared_inputs']
    expected_envelope = deepcopy(original['request_envelope'])
    del expected_envelope['request_text'], expected_envelope['request_summary']
    assert prompt['request_envelope'] == expected_envelope
    assert prompt['declared_inputs']['readings'] == [0, None, 7]
    assert prompt['declared_inputs']['future_input'] == {'empty': []}
    # Repairs add a schema and numbered code; compare the same context view.
    fields = ['request_text', 'task_request_text', 'request_summary', 'request_envelope', 'declared_inputs']
    context_size = lambda value: len(json.dumps({key: value[key] for key in fields}, ensure_ascii=False))
    assert context_size(prompt) < context_size(original)


@pytest.mark.parametrize('design_workbook', [False, True])
def test_generation_preserves_data_with_one_request_context(monkeypatch, design_workbook):
    from aso_agents import pipeline_tools as pipeline
    payload = context_payload()
    if design_workbook:
        payload['source_bound_design_workbooks'] = ['measurements.xlsx']
    before = deepcopy(payload)
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(agent.name)
        assert max_steps == 4
        assert_compact_context(json.loads(input), before)
        return SimpleNamespace(final_output='{"python_code":"pass"}')

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    assert asyncio.run(pipeline._run_sandbox_analysis_agent_once(payload))['python_code'] == 'pass'
    assert len(calls) == 1 and payload == before


def test_executed_review_keeps_program_results_and_binding(monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    payload = {**context_payload(), 'program': 'total = 7',
               'result': {'total': 7, 'missing': 1}, 'summary': 'All readings retained.'}
    before = deepcopy(payload)
    reply = {'program_sha256': program_digest(payload['program']), 'checks': [{
        'requirement_quote': payload['request_text'], 'obligation': 'Retain observations.',
        'status': 'satisfied', 'implementation_evidence': 'The result retains all readings.',
        'problem': None, 'suggested_fix': None}]}

    async def generate(**kwargs):
        prompt = json.loads(kwargs['prompt'])
        assert_compact_context(prompt, before)
        for key in ['program', 'result', 'summary']:
            assert prompt[key] == before[key]
        assert kwargs['schema'] == review_schema(payload['program'])
        return reply

    monkeypatch.setattr(pipeline, '_run_schema_validated_agent_once', generate)
    assert asyncio.run(pipeline._run_analysis_contract_review_agent_once(payload)) == reply
    assert payload == before


def test_repair_keeps_original_code_and_plan_hashes(monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    code = 'value = None\nprint(value + 1)\n'
    plan = {'python_code': code, 'analysis_plan': 'Preserve missing readings.'}
    payload = {**context_payload(), 'execution_repair': {'previous_code': code,
               'previous_plan': plan, 'error': 'TypeError', 'previous_attempts': [{'attempt': 1}]}}
    before = deepcopy(payload)
    repair = {'base_sha256': code_digest(code), 'base_plan_sha256': plan_digest(plan),
              'patches': [{'start_line': 2, 'end_line': 2, 'replacement': 'print(value is None)'}],
              'plan_updates': [], 'explanation': 'Retain the unknown observation.'}

    async def generate(**kwargs):
        prompt = json.loads(kwargs['prompt'])
        assert_compact_context(prompt, before)
        assert [line['text'] for line in prompt['source_lines']] == code.splitlines()
        assert prompt['execution_repair']['error'] == 'TypeError'
        assert prompt['execution_repair']['previous_attempts'] == [{'attempt': 1}]
        assert kwargs['contract_validator'](repair) == []
        return repair

    monkeypatch.setattr(pipeline, '_run_schema_validated_agent_once', generate)
    result = asyncio.run(pipeline._run_sandbox_analysis_agent_once(payload))
    assert result['python_code'].splitlines() == ['value = None', 'print(value is None)']
    assert payload == before


@pytest.mark.parametrize('key', ['request_text', 'task_request_text', 'request_summary'])
@pytest.mark.parametrize('missing_root', [False, True])
def test_distinct_and_nested_only_context_remains_visible(monkeypatch, key, missing_root):
    from aso_agents import pipeline_tools as pipeline
    payload = context_payload()
    if missing_root:
        del payload[key]
    else:
        payload['declared_inputs'][key] = 'A distinct scoped source statement.'
    before = deepcopy(payload)

    async def run(agent, *, input, max_steps):
        prompt = json.loads(input)
        assert prompt['declared_inputs'][key] == before['declared_inputs'][key]
        assert (key in prompt) == (key in before)
        return SimpleNamespace(final_output='{"python_code":"pass"}')

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    asyncio.run(pipeline._run_sandbox_analysis_agent_once(payload))
    assert payload == before


def test_equal_python_values_with_distinct_json_types_are_preserved(monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    payload = context_payload()
    payload['request_envelope']['future_fields']['confirmed'] = True
    payload['declared_inputs']['request_envelope']['future_fields']['confirmed'] = 1
    before = deepcopy(payload)

    async def run(agent, *, input, max_steps):
        prompt = json.loads(input)
        assert prompt['request_envelope']['future_fields']['confirmed'] is True
        assert type(prompt['declared_inputs']['request_envelope']['future_fields']['confirmed']) is int
        return SimpleNamespace(final_output='{"python_code":"pass"}')

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    asyncio.run(pipeline._run_sandbox_analysis_agent_once(payload))
    assert payload == before


def test_executed_program_reads_complete_unchanged_manifest(monkeypatch, tmp_path):
    from aso_agents import pipeline_tools as pipeline
    payload = context_payload()
    before = deepcopy(payload)
    directory = tmp_path / 'execution'
    directory.mkdir()
    manifest = directory / 'declared_inputs.json'
    manifest.write_text(json.dumps(payload['declared_inputs'], ensure_ascii=False))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    code = ("import json\nfrom pathlib import Path\n"
            "d = json.loads(Path('declared_inputs.json').read_text())\n"
            "r = {'request': d['request_text'], 'task': d['task_request_text'], "
            "'count': len(d['readings']), 'unknown': d['readings'].count(None), "
            "'zero': d['readings'].count(0), 'context': d['request_envelope']['future_fields']}\n"
            "Path('observations.json').write_text(json.dumps(r))\n"
            "Path('sandbox_result.json').write_text(json.dumps(r))\n"
            "Path('sandbox_summary.md').write_text('All readings and complete request context retained.')\n")

    generation_requests = []

    async def run(agent, *, input, max_steps):
        assert_compact_context(json.loads(input), generation_requests[-1])
        return SimpleNamespace(final_output=json.dumps({'python_code': code}))

    def generate(request):
        generation_requests.append(deepcopy(request))
        return asyncio.run(pipeline._run_sandbox_analysis_agent_once(request))

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    result = execute_analysis_program(directory=directory, payload=payload,
        generate=generate,
        required_outputs=[directory / 'observations.json'], max_attempts=1)
    assert result['ok'] and len(result['attempts']) == 1
    assert result['result']['request'] == before['request_text']
    assert result['result']['task'] == before['task_request_text']
    assert (result['result']['count'], result['result']['unknown'], result['result']['zero']) == (3, 1, 1)
    assert result['result']['context'] == before['request_envelope']['future_fields']
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == digest
    assert payload == before
