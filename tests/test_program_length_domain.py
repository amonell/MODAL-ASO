import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as tools
from aso_agents.program_domains import (
    program_length_audit_obligations, program_length_coverage_errors,
    PROGRAM_LENGTH_AUDIT_INSTRUCTIONS,
)


ROOT = '/tasks/0/context/structured_goal/design_programs'


def fixture():
    evidence = [{'evidence_id':'request','kind':'user_request',
        'text':'Design steric blockers. Compare 14nt and 19nt on the first branch. Use 23nt on the second branch.'}]
    programs = [{'program_id':'first-short','length':14},
                {'program_id':'first-long','length':19}, {'program_id':'second'}]
    for index, program in enumerate(programs):
        program.update(modality='steric', intent='binding',
            label='First branch' if index<2 else 'Second branch',
            objective='Compare first-branch lengths' if index<2 else 'Use requested second-branch length')
    current = {'tasks':[{'task_id':'design','task_type':'design_new_asos',
        'context':{'structured_goal':{'design_programs':programs}}}]}
    prompt = tools._goal_audit_prompt(request_envelope=current, validation_issues=[],
        issue_records=[], evidence_sources=evidence, semantic_audit=True)
    protocol = tools._goal_audit_protocol_payload(prompt)
    span = next(iter(tools._evidence_span_map(protocol['evidence_sources'])))
    findings = {}
    for i in range(3):
        findings[f'program_{i}'] = {'verdict':'complete' if i<2 else 'incomplete',
            'requested_lengths':[14,19] if i<2 else [23],
            'represented_program_ids':['first-short','first-long'] if i<2 else ['second'],
            'evidence_refs':['request'],'evidence_span_id':span,
            'reason':'First branch requests two lengths; second requests one.'}
    answer = {'status':'repairable','patches':[{'op':'add','path':ROOT+'/2/length','value':23,
        'evidence_refs':['request'],'evidence_span_id':span,'reason':'Represent the explicit second-branch length.'}],
        'goal_replacements':[],'field_dispositions':[],'unresolved':[], 'program_length_checks':findings}
    return current, prompt, answer, evidence


def test_domain_check_keeps_program_scope_and_records_actual_lengths_without_inference():
    current, prompt, _, _ = fixture()
    before = deepcopy(current)
    protocol = tools._goal_audit_protocol_payload(prompt)
    obligations = program_length_audit_obligations(protocol)
    assert [o['program_id'] for o in obligations] == ['first-short','first-long','second']
    assert obligations[0]['available_program_lengths'] == {'first-short':14,'first-long':19,'second':None}
    assert all('requested_lengths' not in o for o in obligations)
    assert protocol['program_length_checks_required'] == obligations
    assert current == before


@pytest.mark.parametrize('overrides', [
    {'semantic_scope_pointers':[ROOT+'/0/region_constraints']},
    {'canonical_contract_issues':[{'json_pointer':ROOT+'/0/length'}]},
    {'noncanonical_field_findings':[{'source_path':ROOT+'/0/unknown'}]},
    {'audit_mode':'contract_repair'},
])
def test_other_phases_do_not_gain_a_length_domain_audit(overrides):
    _, prompt, _, _ = fixture()
    protocol = tools._goal_audit_protocol_payload(prompt)
    protocol.update(overrides)
    assert program_length_audit_obligations(protocol) == []


def test_real_audit_accepts_evidence_bound_missing_length_repair_in_one_call(monkeypatch):
    current, prompt, answer, evidence = fixture()
    original = deepcopy(current)
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(input)
        assert PROGRAM_LENGTH_AUDIT_INSTRUCTIONS in agent.instructions
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, 'run', run)
    result = asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))
    assert len(calls) == 1
    assert result['status'] == 'repairable' and 'program_length_checks' not in result
    applied = tools._apply_goal_audit_patches(current, result['patches'],
        issue_records=[], evidence_sources=evidence)
    assert not applied['rejected']
    programs = applied['request_envelope']['tasks'][0]['context']['structured_goal']['design_programs']
    assert [p['length'] for p in programs] == [14,19,23]
    assert current == original


def test_existing_scalar_cannot_cover_two_requested_variants(monkeypatch):
    current, _, _, evidence = fixture()
    programs = current['tasks'][0]['context']['structured_goal']['design_programs']
    programs[:] = programs[:1]
    prompt = tools._goal_audit_prompt(request_envelope=current, validation_issues=[],
        issue_records=[], evidence_sources=evidence[:1], semantic_audit=True)
    protocol = tools._goal_audit_protocol_payload(prompt)
    span = next(iter(tools._evidence_span_map(protocol['evidence_sources'])))
    variant = {**programs[0], 'program_id':'first-long', 'length':19}
    answer = {'status':'repairable','patches':[{'op':'add','path':ROOT+'/-',
        'value':variant, 'evidence_refs':['request'],
        'evidence_span_id':span, 'reason':'Preserve both requested first-branch variants.'}],
        'goal_replacements':[],'field_dispositions':[],'unresolved':[],
        'program_length_checks':{'program_0':{'verdict':'incomplete','requested_lengths':[14,19],
            'represented_program_ids':['first-short'], 'evidence_refs':['request'],
            'evidence_span_id':span,'reason':'Only one of two requested variants is represented.'}}}

    async def run(agent, *, input, max_steps):
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, 'run', run)
    result = asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))
    assert result['status'] == 'repairable'
    applied = tools._apply_goal_audit_patches(current, result['patches'],
        issue_records=[], evidence_sources=evidence)
    assert not applied['rejected']
    assert [p['length'] for p in applied['request_envelope']['tasks'][0]['context']['structured_goal']['design_programs']] == [14,19]


def test_duplicate_length_values_do_not_make_a_complete_domain():
    _, prompt, answer, _ = fixture()
    protocol = tools._goal_audit_protocol_payload(prompt)
    obligations = program_length_audit_obligations(protocol)
    for obligation in obligations:
        obligation['available_program_lengths']['first-long'] = 14
    errors = program_length_coverage_errors(answer, obligations,
        tools._evidence_span_map(protocol['evidence_sources']))
    assert len(errors) == 2
    assert all('exactly one typed program.length' in error for error in errors)


@pytest.mark.parametrize('defect', ['omitted_checks','missing_program','false_complete',
                                  'missing_variant','wrong_domain','unsupported_default',
                                  'incomplete_without_patch','unrelated_patch','invented_span'])
def test_domain_verdict_cannot_bypass_actual_variant_coverage(monkeypatch, defect):
    _, prompt, answer, _ = fixture()
    checks = answer['program_length_checks']
    if defect == 'omitted_checks':
        answer.pop('program_length_checks')
    elif defect == 'missing_program':
        checks.pop('program_2')
    elif defect == 'false_complete':
        checks['program_2']['verdict'] = 'complete'
        answer['patches'] = []
    elif defect == 'missing_variant':
        checks['program_0']['represented_program_ids'] = ['first-short']
    elif defect == 'wrong_domain':
        checks['program_0']['requested_lengths'] = [14,21]
    elif defect == 'unsupported_default':
        checks['program_0']['verdict'] = 'not_requested'
    elif defect == 'incomplete_without_patch':
        answer['patches'] = []
    elif defect == 'unrelated_patch':
        answer['patches'][0].update(op='replace',path=ROOT+'/0/length',value=17)
    else:
        checks['program_2']['evidence_span_id'] = 'invented.span'
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, 'run', run)
    result = asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))
    assert result['error'] == 'invalid_structured_output'
    assert 'program_length_checks' in result['reason']
    assert len(calls) == 4
