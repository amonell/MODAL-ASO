import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.analysis_repair import code_digest, plan_digest
from aso_agents.analysis_review import assess_review, program_digest, review_schema


def payload():
    text = 'Compare all sensor readings, retain unknown values, and export observations.json.'
    envelope = {'tasks': [{'id': 'compare', 'units': 'μ', 'all_rows': True}],
                'final_outputs': [{'path': 'observations.json'}],
                'future_fields': {'missing': None, 'zero': 0, 'empty': []}}
    return {'request_text': text, 'request_envelope': envelope,
            'declared_inputs': {'request_text': text, 'request_envelope': envelope,
                                'task_request_text': text, 'readings': [2, None, 5],
                                'artifact_refs': [{'path': 'observations.tsv', 'sha256': 'a' * 64}]},
            'execution_contract': {'required_output_files': ['observations.json'],
                                   'readonly_output_files': ['observations.tsv']}}


def assert_complete_context(prompt, original):
    assert prompt['request_text'] == original['request_text']
    assert prompt['request_envelope'] == original['request_envelope']
    assert 'request_text' not in prompt['declared_inputs']
    assert 'request_envelope' not in prompt['declared_inputs']
    restored = {**prompt, 'declared_inputs': {
        **prompt['declared_inputs'], 'request_text': prompt['request_text'],
        'request_envelope': prompt['request_envelope']}}
    assert restored == original


@pytest.mark.parametrize('bound_workbook', [False, True])
def test_initial_generation_sees_each_request_once_without_changing_execution_data(monkeypatch, bound_workbook):
    from aso_agents import pipeline_tools as pipeline

    original = payload()
    if bound_workbook:
        original['source_bound_design_workbooks'] = ['measurements.xlsx']
    before = deepcopy(original)
    calls = []
    plan = {'python_code': 'pass', 'analysis_plan': 'Read every row.'}

    async def run(agent, *, input, max_steps):
        calls.append(agent)
        assert max_steps == 4
        assert_complete_context(json.loads(input), before)
        return SimpleNamespace(final_output=json.dumps(plan))

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    assert asyncio.run(pipeline._run_sandbox_analysis_agent_once(original)) == plan
    assert len(calls) == 1 and original == before


@pytest.mark.parametrize('ending', ['\n', '\r\n'])
def test_repair_keeps_original_program_binding_and_all_failure_evidence(monkeypatch, ending):
    from aso_agents import pipeline_tools as pipeline

    code = ending.join(['value = None', 'print(value + 1)', ''])
    plan = {'python_code': code, 'analysis_plan': 'Retain unknown readings.',
            'expected_outputs': ['observations.json']}
    original = payload()
    original['execution_repair'] = {
        'previous_code': code, 'previous_plan': plan, 'error': 'TypeError',
        'runtime_diagnostics': {'locals': {'value': None}},
        'previous_attempts': [{'attempt': 1, 'error': 'TypeError'}]}
    before = deepcopy(original)
    repair = {'base_sha256': code_digest(code), 'base_plan_sha256': plan_digest(plan),
              'patches': [{'start_line': 2, 'end_line': 2,
                           'replacement': "print('unknown' if value is None else value + 1)"}],
              'plan_updates': [], 'explanation': 'Retain missing observations.'}
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        prompt = json.loads(kwargs['prompt'])
        assert 'request_text' not in prompt['declared_inputs']
        assert 'request_envelope' not in prompt['declared_inputs']
        for key in ['request_text', 'request_envelope', 'declared_inputs', 'execution_contract']:
            if key != 'declared_inputs':
                assert prompt[key] == before[key]
        assert prompt['declared_inputs']['readings'] == [2, None, 5]
        for key in ['error', 'runtime_diagnostics', 'previous_attempts']:
            assert prompt['execution_repair'][key] == before['execution_repair'][key]
        assert [line['text'] for line in prompt['source_lines']] == code.splitlines()
        assert kwargs['contract_validator'](repair) == []
        assert kwargs['contract_validator']({**repair, 'base_sha256': code_digest(code + ' ')})
        return repair

    monkeypatch.setattr(pipeline, '_run_schema_validated_agent_once', generate)
    result = asyncio.run(pipeline._run_sandbox_analysis_agent_once(original))
    assert len(calls) == 1 and original == before
    assert result['python_code'].splitlines()[1] == repair['patches'][0]['replacement']
    assert result['expected_outputs'] == plan['expected_outputs']


def test_contract_review_preserves_program_results_and_review_schema(monkeypatch):
    from aso_agents import pipeline_tools as pipeline

    original = {**payload(), 'program': 'total = sum(values)',
                'result': {'total': 7, 'missing_count': 1},
                'summary': 'All observed readings included.',
                'output_paths': ['observations.json']}
    before = deepcopy(original)
    reply = {'program_sha256': program_digest(original['program']), 'checks': [{
        'requirement_quote': original['request_text'], 'obligation': 'Retain every reading.',
        'status': 'satisfied', 'implementation_evidence': 'All three readings are retained.',
        'problem': None, 'suggested_fix': None}]}
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        assert_complete_context(json.loads(kwargs['prompt']), before)
        assert kwargs['schema'] == review_schema(original['program'])
        assert kwargs['schema']['properties']['program_sha256']['enum'] == [reply['program_sha256']]
        return reply

    monkeypatch.setattr(pipeline, '_run_schema_validated_agent_once', generate)
    result = asyncio.run(pipeline._run_analysis_contract_review_agent_once(original))
    assert result == reply
    assert assess_review(result, original['program'], original['request_text'])['status'] == 'no_concrete_violation_identified'
    assert len(calls) == 1 and original == before


@pytest.mark.parametrize('field', ['request_text', 'request_envelope'])
@pytest.mark.parametrize('missing_root', [False, True])
def test_distinct_or_nested_only_request_context_remains_visible(monkeypatch, field, missing_root):
    from aso_agents import pipeline_tools as pipeline

    original = payload()
    if missing_root:
        del original[field]
    else:
        original[field] = 'Distinct top-level source' if field == 'request_text' else {'task': 'different'}
    before = deepcopy(original)

    async def run(agent, *, input, max_steps):
        prompt = json.loads(input)
        assert prompt['declared_inputs'][field] == before['declared_inputs'][field]
        assert (field in prompt) == (field in before)
        if field in before:
            assert prompt[field] == before[field]
        return SimpleNamespace(final_output='{"python_code":"pass"}')

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    assert asyncio.run(pipeline._run_sandbox_analysis_agent_once(original))['python_code'] == 'pass'
    assert original == before


@pytest.mark.parametrize('inputs', [None, [], 'declared_inputs.json'])
def test_other_declared_input_shapes_are_unchanged(monkeypatch, inputs):
    from aso_agents import pipeline_tools as pipeline

    original = {**payload(), 'declared_inputs': inputs}
    before = deepcopy(original)

    async def run(agent, *, input, max_steps):
        assert json.loads(input) == before
        return SimpleNamespace(final_output='{"python_code":"pass"}')

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    asyncio.run(pipeline._run_sandbox_analysis_agent_once(original))
    assert original == before


def test_execution_reads_complete_unchanged_runtime_manifest(monkeypatch, tmp_path):
    from aso_agents import pipeline_tools as pipeline

    original = payload()
    manifest = tmp_path / 'declared_inputs.json'
    manifest.write_text(json.dumps(original['declared_inputs'], ensure_ascii=False))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    original['declared_inputs']['manifest_path'] = str(manifest)
    before = deepcopy(original)
    code = (
        'import json\nfrom pathlib import Path\n'
        f'data = json.loads(Path({str(manifest)!r}).read_text())\n'
        "values = data['readings']\n"
        "result = {'total': sum(v for v in values if v is not None), "
        "'missing_count': values.count(None), 'task': data['request_envelope']['tasks'][0]['id']}\n"
        "Path('observations.json').write_text(json.dumps(result))\n"
        "Path('sandbox_result.json').write_text(json.dumps(result))\n"
        "Path('sandbox_summary.md').write_text('Every observed reading included; missing values retained.')\n")
    current = []

    async def run(agent, *, input, max_steps):
        assert_complete_context(json.loads(input), current[-1])
        return SimpleNamespace(final_output=json.dumps({'python_code': code}))

    def generate(request):
        current.append(deepcopy(request))
        return asyncio.run(pipeline._run_sandbox_analysis_agent_once(request))

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    directory = tmp_path / 'execution'
    result = execute_analysis_program(directory=directory, payload=original, generate=generate,
                                      required_outputs=[directory / 'observations.json'], max_attempts=1)
    assert result['ok'] and len(current) == 1
    assert result['result']['total'] == 7 and result['result']['missing_count'] == 1
    assert result['result']['task'] == 'compare'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == digest
    assert original == before
