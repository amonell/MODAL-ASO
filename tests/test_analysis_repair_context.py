import asyncio
from copy import deepcopy
import json

import pytest

from aso_agents.analysis_repair import code_digest, plan_digest


@pytest.mark.parametrize('ending', ['\n', '\r\n'])
@pytest.mark.parametrize('with_plan', [False, True])
def test_one_program_view_preserves_repair_evidence_and_original_bindings(monkeypatch, ending, with_plan):
    from aso_agents import pipeline_tools as pipeline

    code = ending.join(["label = 'measured μ'", 'value = None', 'print(value + 1)', ''])
    plan = ({'python_code': code, 'analysis_plan': 'Report the observation.',
             'assumptions': ['Missing remains unknown.'], 'limitations': [],
             'expected_outputs': [], 'custom_metadata': {'unit': 'μ'}} if with_plan else {})
    payload = {'declared_inputs': {'values': [0, None]}, 'request_text': 'Preserve missing values.',
               'execution_repair': {'previous_code': code, 'previous_plan': plan,
                                    'error': 'TypeError: unsupported operand',
                                    'runtime_diagnostics': {'frames': [{'locals': {'value': None}}]},
                                    'previous_attempts': [{'attempt': 1, 'error': 'TypeError'}]}}
    before = deepcopy(payload)
    repair = {'base_sha256': code_digest(code), 'base_plan_sha256': plan_digest(plan),
              'patches': [{'start_line': 3, 'end_line': 3,
                           'replacement': "print('unknown' if value is None else value + 1)"}],
              'plan_updates': [], 'explanation': 'Preserve missing observations.'}

    async def generate(**kwargs):
        prompt = json.loads(kwargs['prompt'])
        assert [line['line'] for line in prompt['source_lines']] == [1, 2, 3]
        assert [line['text'] for line in prompt['source_lines']] == code.splitlines()
        visible = prompt['execution_repair']
        assert 'previous_code' not in visible and 'python_code' not in visible['previous_plan']
        assert visible['previous_plan'] == {k: v for k, v in plan.items() if k != 'python_code'}
        for key in ['error', 'runtime_diagnostics', 'previous_attempts']:
            assert visible[key] == payload['execution_repair'][key]
        assert prompt['declared_inputs'] == payload['declared_inputs']
        assert prompt['request_text'] == payload['request_text']
        assert prompt['output_schema']['properties']['base_sha256']['const'] == code_digest(code)
        assert prompt['output_schema']['properties']['base_plan_sha256']['const'] == plan_digest(plan)
        assert kwargs['contract_validator'](repair) == []
        stale = {**repair, 'base_sha256': code_digest(code + ' ')}
        assert kwargs['contract_validator'](stale)
        return repair

    monkeypatch.setattr(pipeline, '_run_schema_validated_agent_once', generate)
    result = asyncio.run(pipeline._run_sandbox_analysis_agent_once(payload))
    assert payload == before
    assert result['python_code'].splitlines()[:2] == code.splitlines()[:2]
    assert result['python_code'].splitlines()[2] == repair['patches'][0]['replacement']
    assert result['native_code_repair']['base_plan_sha256'] == plan_digest(plan)
    for key, value in plan.items():
        if key != 'python_code':
            assert result[key] == value
