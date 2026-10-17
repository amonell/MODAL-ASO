import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as tools
from aso_agents.request_workbench_agent import (
    REGION_CONSTRAINT_AUDIT_INSTRUCTIONS,
    goal_contract_audit_agent,
)


ROOT = '/tasks/0/context/structured_goal'
GLOBAL = ROOT + '/region_constraints'
LOCAL = ROOT + '/design_programs/0/region_constraints'


@pytest.mark.parametrize('scopes', [[GLOBAL], [LOCAL], [GLOBAL, LOCAL]])
@pytest.mark.parametrize('mode', ['semantic_completeness_and_contract_repair',
                                  'post_repair_semantic_verification'])
def test_existing_region_pass_changes_instructions_only(scopes, mode):
    protocol = {'audit_mode': mode, 'semantic_scope_pointers': scopes}
    before = deepcopy(protocol)
    agent = tools._goal_audit_agent_for_protocol(protocol)
    assert agent.instructions == REGION_CONSTRAINT_AUDIT_INSTRUCTIONS
    assert len(agent.instructions) < len(goal_contract_audit_agent.instructions) / 3
    for field, value in vars(goal_contract_audit_agent).items():
        if field != 'instructions':
            assert getattr(agent, field) is value
    assert protocol == before


@pytest.mark.parametrize('override', [
    {'semantic_scope_pointers': []},
    {'semantic_scope_pointers': None},
    {'semantic_scope_pointers': [GLOBAL, ROOT + '/coordinate_conventions']},
    {'semantic_scope_pointers': [GLOBAL + '/0']},
    {'semantic_scope_pointers': ['/unrelated/region_constraints']},
    {'semantic_scope_pointers': [None]},
    {'audit_mode': 'contract_repair'},
    {'canonical_contract_issues': [{'json_pointer': GLOBAL}]},
    {'noncanonical_field_findings': [{'source_path': GLOBAL}]},
])
def test_other_audit_phases_keep_original_agent(override):
    protocol = {'audit_mode': 'semantic_completeness_and_contract_repair',
                'semantic_scope_pointers': [GLOBAL], **override}
    assert tools._goal_audit_agent_for_protocol(protocol) is goal_contract_audit_agent


def fixture():
    evidence = [{'evidence_id': 'request', 'kind': 'user_request',
                 'text': 'Apply the shared window to both branches. Only the first branch must overlap the inner feature.'}]
    shared = {'source_type': 'explicit_rows', 'coordinate_space': 'cdna',
              'operator': 'intersect', 'aso_overlap_requirement': 'any_overlap',
              'intervals_bed0_half_open': [{'chrom': 'TRANSCRIPT.3', 'start': 10, 'end': 70}]}
    local = {**shared, 'intervals_bed0_half_open': [{'chrom': 'TRANSCRIPT.3', 'start': 30, 'end': 35}]}
    current = {'tasks': [{'task_id': 'd', 'task_type': 'design_new_asos',
        'context': {'structured_goal': {'region_constraints': [shared, deepcopy(local)],
            'design_programs': [{'program_id': 'first', 'region_constraints': [local]},
                                {'program_id': 'second'}]}}}]}
    prompt = tools._goal_audit_prompt(request_envelope=current,
        validation_issues=[], issue_records=[], evidence_sources=evidence,
        semantic_audit=True, semantic_scope_pointers=[GLOBAL])
    spans = tools._evidence_span_map(tools._declared_evidence_sources_from_prompt(prompt))
    answer = {'status': 'repairable', 'patches': [{
        'op': 'remove', 'path': GLOBAL + '/1', 'value': None,
        'evidence_refs': ['request'], 'evidence_span_id': next(iter(spans)),
        'reason': 'The inner-feature requirement belongs only to the first branch.'}],
        'goal_replacements': [], 'field_dispositions': [], 'unresolved': []}
    answer['region_scope_checks'] = {
        f'check_{index}': {
            'verdict': 'unsupported' if index == 3 else 'supported',
            'effective_restriction': 'Shared window or inner-feature intersection on this branch.',
            'evidence_refs': ['request'], 'evidence_span_id': next(iter(spans)),
            'reason': 'The shared window applies to both branches; the inner feature only to the first.',
        } for index in range(4)
    }
    return current, prompt, answer, evidence


def test_real_focused_runner_keeps_one_call_evidence_binding_and_application(monkeypatch):
    current, prompt, answer, evidence = fixture()
    before = deepcopy(current)
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(input)
        assert agent.instructions.startswith(REGION_CONSTRAINT_AUDIT_INSTRUCTIONS)
        assert agent.model is goal_contract_audit_agent.model
        assert input == prompt
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, 'run', run)
    result = asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))
    assert len(calls) == 1
    assert result['status'] == 'repairable'
    assert 'region_scope_checks' not in result  # Existing downstream API remains stable.
    assert result['patches'][0]['evidence_quote'] in evidence[0]['text']
    applied = tools._apply_goal_audit_patches(current, result['patches'],
        issue_records=[], evidence_sources=evidence)
    assert not applied['rejected']
    goal = applied['request_envelope']['tasks'][0]['context']['structured_goal']
    original = before['tasks'][0]['context']['structured_goal']
    assert goal['region_constraints'] == original['region_constraints'][:1]
    assert goal['design_programs'] == original['design_programs']
    assert current == before


@pytest.mark.parametrize('defect', ['out_of_scope', 'invented_evidence'])
def test_focused_runner_still_rejects_bad_paths_and_evidence(monkeypatch, defect):
    _, prompt, answer, _ = fixture()
    if defect == 'out_of_scope':
        answer['patches'][0]['path'] = LOCAL + '/0'
    else:
        answer['patches'][0]['evidence_span_id'] = 'invented.span'
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, 'run', run)
    result = asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))
    assert result['error'] == 'invalid_structured_output'
    assert len(calls) == 4


def test_focused_runner_drops_noop_without_fabricating_a_repair(monkeypatch):
    _, prompt, answer, _ = fixture()
    answer['patches'][0].update(op='replace', path=GLOBAL + '/0/operator', value='intersect')
    answer['region_scope_checks']['check_3']['verdict'] = 'supported'
    calls = []

    async def run(agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, 'run', run)
    result = asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))
    assert result['status'] == 'valid'
    assert result['patches'] == []
    assert len(calls) == 1
