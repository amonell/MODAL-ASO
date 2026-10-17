import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as tools
from aso_agents.region_audit import region_audit_obligations, region_audit_coverage_errors
from test_focused_region_audit import fixture, GLOBAL, LOCAL, ROOT


def test_global_coverage_reaches_every_branch_without_using_names_as_intent():
    _, prompt, _, _ = fixture()
    protocol = tools._goal_audit_protocol_payload(prompt)
    original = deepcopy(protocol)
    result = region_audit_obligations(protocol)
    assert [(x['constraint_pointer'], x['program_pointer']) for x in result] == [
        (GLOBAL + '/0', ROOT + '/design_programs/0'),
        (GLOBAL + '/0', ROOT + '/design_programs/1'),
        (GLOBAL + '/1', ROOT + '/design_programs/0'),
        (GLOBAL + '/1', ROOT + '/design_programs/1'),
    ]
    assert protocol['region_scope_checks_required'] == result
    assert protocol == original
    # Program identifiers are carried as labels; changing them cannot change
    # the pointers/count or invent a supported/unsupported finding.
    programs = protocol['current_request_envelope']['tasks'][0]['context']['structured_goal']['design_programs']
    programs[0]['program_id'], programs[1]['program_id'] = 'other', 'first'
    renamed = region_audit_obligations(protocol)
    assert [{k:v for k,v in o.items() if k != 'program_id'} for o in result] == [
        {k:v for k,v in o.items() if k != 'program_id'} for o in renamed]
    assert all('verdict' not in o for o in renamed)


def test_local_coverage_does_not_include_siblings_and_duplicate_scopes_do_not_duplicate_checks():
    _, prompt, _, _ = fixture()
    protocol = tools._goal_audit_protocol_payload(prompt)
    protocol['semantic_scope_pointers'] = [LOCAL, LOCAL]
    assert region_audit_obligations(protocol) == [{
        'check_id': 'check_0', 'constraint_pointer': LOCAL + '/0',
        'program_pointer': ROOT + '/design_programs/0', 'program_id': 'first'}]


def test_no_program_goal_still_has_its_root_constraint_checked():
    _, prompt, _, _ = fixture()
    protocol = tools._goal_audit_protocol_payload(prompt)
    goal = protocol['current_request_envelope']['tasks'][0]['context']['structured_goal']
    goal.pop('design_programs')
    result = region_audit_obligations(protocol)
    assert [o['program_pointer'] for o in result] == [ROOT, ROOT]
    assert all(o['program_id'] is None for o in result)


@pytest.mark.parametrize('defect', ['missing_checks', 'missing_pair', 'unsupported_without_patch',
                                  'unsupported_noop', 'unrelated_correction', 'unclear_without_question'])
def test_real_runner_cannot_accept_unsupported_or_incomplete_scope_verdict(monkeypatch, defect):
    _, prompt, answer, _ = fixture()
    if defect == 'missing_checks':
        answer.pop('region_scope_checks')
    elif defect == 'missing_pair':
        answer['region_scope_checks'].pop('check_3')
    elif defect == 'unsupported_without_patch':
        answer.update(status='valid', patches=[])
    elif defect == 'unsupported_noop':
        answer['patches'][0].update(op='replace', path=GLOBAL + '/1/operator', value='intersect')
    elif defect == 'unrelated_correction':
        answer['patches'][0].update(op='replace', path=GLOBAL + '/0/operator', value='union')
    else:
        answer.update(status='valid', patches=[])
        answer['region_scope_checks']['check_3']['verdict'] = 'unclear'
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, 'run', run)
    result = asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))
    assert result['error'] == 'invalid_structured_output'
    assert 'region_scope_checks' in result['reason']
    assert len(calls) == 4  # Existing bounded retry policy, no new audit loop.


def test_check_quote_must_belong_to_the_named_source():
    _, prompt, answer, _ = fixture()
    protocol = tools._goal_audit_protocol_payload(prompt)
    spans = tools._evidence_span_map(protocol['evidence_sources'])
    answer['region_scope_checks']['check_0']['evidence_refs'] = ['different_source']
    errors = region_audit_coverage_errors(answer, region_audit_obligations(protocol), spans)
    assert len(errors) == 1
    assert 'span must belong' in errors[0]


def test_old_compact_contract_remains_unchanged_for_broad_audit(monkeypatch):
    current, _, _, evidence = fixture()
    prompt = tools._goal_audit_prompt(request_envelope=current,
        validation_issues=[], issue_records=[], evidence_sources=evidence, semantic_audit=True)
    assert 'region_scope_checks_required' not in tools._goal_audit_protocol_payload(prompt)
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps({'status':'valid','patches':[],
            'goal_replacements':[],'field_dispositions':[],'unresolved':[]}))

    monkeypatch.setattr(tools.Runner, 'run', run)
    assert asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))['status'] == 'valid'
    assert len(calls) == 1
