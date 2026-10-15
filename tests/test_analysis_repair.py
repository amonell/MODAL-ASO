import asyncio
from copy import deepcopy
import hashlib
import json

import pytest

from aso_agents.analysis_repair import apply_program_repair, code_digest, plan_digest, repair_schema


def test_line_repair_preserves_scientific_plan_and_untouched_program():
    code = "criterion = 0.5\nvalue = None\nprint(f'{value:.3f}')\n"
    plan = {"analysis_plan": "Use the requested threshold.", "python_code": code,
            "assumptions": ["Missing is unknown."]}
    repaired = apply_program_repair(plan, code, {"base_sha256": code_digest(code), "patches": [
        {"start_line": 3, "end_line": 3, "replacement": "print('unknown' if value is None else f'{value:.3f}')"}]})
    assert repaired['analysis_plan'] == plan['analysis_plan']
    assert repaired['assumptions'] == plan['assumptions']
    assert repaired['python_code'].splitlines()[:2] == code.splitlines()[:2]
    assert plan['python_code'] == code


@pytest.mark.parametrize('fault', ['hash', 'overlap', 'span', 'unchanged', 'syntax'])
def test_unsafe_or_ineffective_patch_is_rejected(fault):
    code = 'value = 1\nprint(value)'
    patch = {'base_sha256': code_digest(code), 'patches': [
        {'start_line': 1, 'end_line': 1, 'replacement': 'value = 2'}]}
    if fault == 'hash': patch['base_sha256'] = 'wrong'
    if fault == 'overlap': patch['patches'].append(dict(patch['patches'][0]))
    if fault == 'span': patch['patches'][0]['end_line'] = 9
    if fault == 'unchanged': patch['patches'][0]['replacement'] = 'value = 1'
    if fault == 'syntax': patch['patches'][0]['replacement'] = 'value = ('
    with pytest.raises((ValueError, SyntaxError)):
        apply_program_repair({}, code, patch)


def revised_description_fixture():
    code = "output = 'old.csv'\nprint(output)"
    plan = {'python_code': code, 'analysis_plan': 'Write old.csv.',
            'assumptions': ['Values are used as supplied.'], 'limitations': [],
            'expected_outputs': ['old.csv'], 'scientific_criterion': {'minimum': 3}}
    repair = {'base_sha256': code_digest(code), 'base_plan_sha256': plan_digest(plan),
              'patches': [{'start_line': 1, 'end_line': 1, 'replacement': "output = 'requested.csv'"}],
              'plan_updates': [{'field': 'analysis_plan', 'value': 'Write requested.csv.'},
                               {'field': 'expected_outputs', 'value': ['requested.csv']}],
              'explanation': 'Match the required output path and its descriptions.'}
    return plan, code, repair


def test_bound_descriptions_change_with_code_without_rewriting_scientific_criteria():
    plan, code, repair = revised_description_fixture()
    before = deepcopy(plan)
    result = apply_program_repair(plan, code, repair)
    assert result['analysis_plan'] == 'Write requested.csv.'
    assert result['expected_outputs'] == ['requested.csv']
    assert result['assumptions'] == plan['assumptions']
    assert result['scientific_criterion'] == plan['scientific_criterion']
    assert result['python_code'].splitlines()[1] == code.splitlines()[1]
    assert plan == before
    result['expected_outputs'].append('unrelated')
    assert repair['plan_updates'][1]['value'] == ['requested.csv']


@pytest.mark.parametrize('fault', ['plan_hash', 'missing_hash', 'duplicate', 'forbidden',
                                  'wrong_scalar', 'wrong_list', 'extra_key', 'missing_updates'])
def test_unbound_or_out_of_scope_plan_updates_are_rejected(fault):
    plan, code, repair = revised_description_fixture()
    if fault == 'plan_hash':
        plan['assumptions'].append('Changed after the repair was proposed.')
    elif fault == 'missing_hash':
        repair.pop('base_plan_sha256')
    elif fault == 'duplicate':
        repair['plan_updates'].append(deepcopy(repair['plan_updates'][0]))
    elif fault == 'forbidden':
        repair['plan_updates'][0]['field'] = 'scientific_criterion'
    elif fault == 'wrong_scalar':
        repair['plan_updates'][0]['value'] = ['Not a scalar description.']
    elif fault == 'wrong_list':
        repair['plan_updates'][1]['value'] = 'Not an output list.'
    elif fault == 'extra_key':
        repair['plan_updates'][0]['unbound'] = True
    elif fault == 'missing_updates':
        repair.pop('plan_updates')
    before = deepcopy(plan)
    with pytest.raises(ValueError):
        apply_program_repair(plan, code, repair)
    assert plan == before


def test_empty_metadata_update_is_explicit_and_keeps_valid_descriptions():
    plan, code, repair = revised_description_fixture()
    plan['analysis_plan'] = 'Write the requested output.'
    plan['expected_outputs'] = ['requested.csv']
    repair['base_plan_sha256'] = plan_digest(plan)
    repair['plan_updates'] = []
    result = apply_program_repair(plan, code, repair)
    assert result['analysis_plan'] == plan['analysis_plan']
    assert result['assumptions'] == plan['assumptions']
    assert result['native_code_repair']['base_plan_sha256'] == plan_digest(plan)


def test_native_repair_contract_requires_both_bindings_and_validates_updates(monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    plan, code, repair = revised_description_fixture()
    called = []
    async def generate(**kwargs):
        schema = kwargs['schema']
        assert schema == repair_schema(code, plan)
        assert {'base_plan_sha256', 'plan_updates'} <= set(schema['required'])
        prompt = json.loads(kwargs['prompt'])
        assert prompt['execution_repair']['previous_plan'] == plan
        assert kwargs['contract_validator'](repair) == []
        stale = deepcopy(repair)
        stale['base_plan_sha256'] = 'wrong'
        assert kwargs['contract_validator'](stale)
        called.append(True)
        return repair
    monkeypatch.setattr(pipeline, '_run_schema_validated_agent_once', generate)
    result = asyncio.run(pipeline._run_sandbox_analysis_agent_once({
        'execution_repair': {'previous_code': code, 'previous_plan': plan}}))
    assert called and result['analysis_plan'] == 'Write requested.csv.'


def test_executed_revision_receipt_binds_updated_plan_and_preserves_prior_attempt(tmp_path):
    from aso_agents.analysis_execution import execute_analysis_program
    code = ("from pathlib import Path\nimport json\n"
            "Path('old.csv').write_text('value\\n1\\n')\n"
            "Path('sandbox_result.json').write_text(json.dumps({'value': 1}))\n"
            "Path('sandbox_summary.md').write_text('Saved one observation.')\n")
    plan = {'python_code': code, 'analysis_plan': 'Write old.csv.',
            'assumptions': ['Values are used as supplied.'], 'expected_outputs': ['old.csv']}
    def generate(request):
        previous = request.get('execution_repair')
        if not previous:
            return deepcopy(plan)
        return apply_program_repair(previous['previous_plan'], previous['previous_code'], {
            'base_sha256': code_digest(previous['previous_code']),
            'base_plan_sha256': plan_digest(previous['previous_plan']),
            'patches': [{'start_line': 3, 'end_line': 3,
                         'replacement': "Path('requested.csv').write_text('value\\n1\\n')"}],
            'plan_updates': [{'field': 'analysis_plan', 'value': 'Write requested.csv.'},
                             {'field': 'expected_outputs', 'value': ['requested.csv']}],
            'explanation': 'Use the required filename.'})
    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[tmp_path/'requested.csv'], max_attempts=2)
    assert result['ok'] and len(result['attempts']) == 2
    assert result['plan']['analysis_plan'] == 'Write requested.csv.'
    assert result['result'] == {'value': 1}
    assert (tmp_path/'requested.csv').read_text() == 'value\n1\n'
    assert json.loads((tmp_path/'execution_attempts/01/plan.json').read_text()) == plan
    for receipt in result['attempts']:
        from pathlib import Path
        assert hashlib.sha256(Path(receipt['plan_path']).read_bytes()).hexdigest() == receipt['plan_file_sha256']
        assert hashlib.sha256(Path(receipt['program_path']).read_bytes()).hexdigest() == receipt['code_sha256']
