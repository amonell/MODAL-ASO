import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as tools


ROOT = '/tasks/0/context/structured_goal/design_programs'


def fixture():
    evidence = [
        {'evidence_id': 'request', 'kind': 'user_request',
         'text': 'Compare lengths 13 and 17 in the first interval. Preserve its constraints.'},
        {'evidence_id': 'context', 'kind': 'user_request', 'text': 'Use the supplied reference.'},
    ]
    first = {'program_id': 'first', 'label': 'First interval', 'modality': 'steric',
             'intent': 'binding', 'objective': 'Compare lengths', 'length': 13,
             'target_context': 'declared reference', 'run_splice_eval': False,
             'region_constraints': [{'source_type': 'explicit_rows',
                 'intervals_bed0_half_open': [{'chrom': 'chr7', 'start': 901, 'end': 946, 'strand': '+'}],
                 'coordinate_space': 'genomic', 'aso_overlap_requirement': 'full_containment'}]}
    second = {**deepcopy(first), 'program_id': 'other', 'length': 17,
              'label': 'Different interval', 'target_context': 'different branch'}
    envelope = {'tasks': [{'task_id': 'design', 'task_type': 'design_new_asos',
                          'context': {'structured_goal': {'design_programs': [first, second]}}}],
                'final_outputs': [{'task_id': 'design', 'kind': 'file'}]}
    prompt = tools._goal_audit_prompt(request_envelope=envelope, validation_issues=[],
        issue_records=[], evidence_sources=evidence, semantic_audit=True)
    protocol = tools._goal_audit_protocol_payload(prompt)
    spans = tools._evidence_span_map(protocol['evidence_sources'])
    span = next(key for key, value in spans.items() if value['evidence_id'] == 'request')
    finding = {'verdict': 'complete', 'requested_lengths': [13, 17],
               'represented_program_ids': ['first', 'new-long'], 'evidence_refs': ['request'],
               'evidence_span_id': span, 'reason': 'Both explicit lengths exist after the repair.'}
    variant = {**deepcopy(first), 'program_id': 'new-long', 'label': 'Long variant', 'length': 17}
    patch = {'op': 'add', 'path': ROOT + '/-', 'value': variant,
             'evidence_refs': ['request'], 'evidence_span_id': span,
             'reason': 'Add the missing length while preserving the first branch.'}
    answer = {'status': 'repairable', 'patches': [patch], 'goal_replacements': [],
              'field_dispositions': [], 'unresolved': [], 'program_length_checks': {
                  'program_0': finding,
                  'program_1': {**deepcopy(finding), 'verdict': 'not_requested',
                                'requested_lengths': [], 'represented_program_ids': []}}}
    return envelope, prompt, answer, evidence


def run_audit(monkeypatch, prompt, answer):
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, 'run', run)
    return asyncio.run(tools._run_goal_contract_audit_agent_once(prompt)), calls


def test_complete_after_append_uses_real_executor_and_preserves_input_and_graph(monkeypatch):
    envelope, prompt, answer, evidence = fixture()
    original = deepcopy(envelope)
    raw = deepcopy(answer)
    result, calls = run_audit(monkeypatch, prompt, answer)
    assert len(calls) == 1
    assert result['status'] == 'repairable'
    applied = tools._apply_goal_audit_patches(envelope, result['patches'],
        issue_records=[], evidence_sources=evidence)
    assert not applied['rejected']
    goal = applied['request_envelope']['tasks'][0]['context']['structured_goal']
    assert [(p['program_id'], p['length']) for p in goal['design_programs']] == [
        ('first', 13), ('other', 17), ('new-long', 17)]
    assert applied['request_envelope']['final_outputs'] == original['final_outputs']
    assert envelope == original and answer == raw


def test_complete_after_scalar_replacement_checks_new_length(monkeypatch):
    _, prompt, answer, _ = fixture()
    answer['patches'][0].update(op='replace', path=ROOT + '/0/length', value=17)
    answer['program_length_checks']['program_0'].update(
        requested_lengths=[17], represented_program_ids=['first'])
    result, calls = run_audit(monkeypatch, prompt, answer)
    assert result['status'] == 'repairable' and len(calls) == 1


def test_incomplete_remains_a_before_state_finding_with_effective_repair(monkeypatch):
    _, prompt, answer, _ = fixture()
    answer['program_length_checks']['program_0'].update(
        verdict='incomplete', represented_program_ids=['first'])
    result, calls = run_audit(monkeypatch, prompt, answer)
    assert result['status'] == 'repairable' and len(calls) == 1


@pytest.mark.parametrize('defect', [
    'invented_id', 'unapplied_id', 'duplicate_id', 'duplicate_length', 'wrong_length',
    'other_branch', 'omitted_constraint', 'changed_constraint', 'changed_execution_flag',
    'unverified_patch_evidence', 'invalid_append_operation', 'incomplete_future_id',
])
def test_projection_cannot_turn_invalid_or_unrelated_variants_into_coverage(monkeypatch, defect):
    envelope, prompt, answer, _ = fixture()
    original = deepcopy(envelope)
    finding = answer['program_length_checks']['program_0']
    patch = answer['patches'][0]
    variant = patch['value']
    if defect == 'invented_id':
        finding['represented_program_ids'][1] = 'not-created'
    elif defect == 'unapplied_id':
        answer['patches'] = []
    elif defect == 'duplicate_id':
        variant['program_id'] = 'other'
        finding['represented_program_ids'][1] = 'other'
    elif defect == 'duplicate_length':
        variant['length'] = 13
    elif defect == 'wrong_length':
        variant['length'] = 21
    elif defect == 'other_branch':
        finding['represented_program_ids'][1] = 'other'
    elif defect == 'omitted_constraint':
        variant.pop('region_constraints')
    elif defect == 'changed_constraint':
        variant['region_constraints'][0]['intervals_bed0_half_open'][0]['end'] += 11
    elif defect == 'changed_execution_flag':
        variant['run_splice_eval'] = True
    elif defect == 'unverified_patch_evidence':
        patch['evidence_refs'] = ['context']
    elif defect == 'invalid_append_operation':
        patch['op'] = 'replace'
    else:
        finding['verdict'] = 'incomplete'
    result, calls = run_audit(monkeypatch, prompt, answer)
    assert result['error'] == 'invalid_structured_output'
    assert len(calls) == 4  # Existing bounded retry policy is unchanged.
    assert envelope == original
