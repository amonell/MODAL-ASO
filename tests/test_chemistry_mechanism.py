from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools
from aso_agents.chemistry_mechanism import chemistry_mechanism_errors, mechanism_audit_schema, recommendation_digest


def proposal(text):
    return {"recommendation": {"CHEMISTRY": "steric-hindrance", "platform": "steric_blocker",
                               "length": 21, "rationale": text}}


@pytest.mark.parametrize('raw', [{}, {'issues': None}, {'issues': [], 'passed': True},
    {'issues': [{'claim': 'not in proposal', 'reason': 'unsupported'}]},
    {'issues': [{'claim': 'RNA binding', 'reason': ''}]}, 'invalid JSON'])
def test_invalid_audit_or_invented_quote_cannot_attest_mechanism(raw):
    assert chemistry_mechanism_errors(raw, proposal('RNA binding'))


def test_qualified_mechanism_and_quoted_contradiction_remain_distinct():
    good = proposal('RNA binding; its site-dependent outcome remains uncertain.')
    assert not chemistry_mechanism_errors({'issues': []}, good)
    bad = proposal('The profile alone guarantees inclusion.')
    findings = {'issues': [{'claim': 'guarantees inclusion', 'reason': 'Chemistry does not determine site action.'}]}
    assert 'Chemistry does not determine site action.' in chemistry_mechanism_errors(findings, bad)[0]


def test_mechanism_schema_binds_findings_to_exact_selected_rationale():
    selected = proposal('The profile alone guarantees inclusion.')
    schema = mechanism_audit_schema(selected)
    claim = schema['properties']['issues']['items']['properties']['claim']
    assert claim['enum'] == [selected['recommendation']['rationale']]
    assert schema['additionalProperties'] is False


def test_checker_receives_source_bound_output_contract(monkeypatch):
    selected = proposal('RNase-H-silent RNA binding; splice effect remains uncertain.')
    calls = []
    async def run(agent, *, input, **kwargs):
        body = json.loads(input)
        calls.append((getattr(agent, 'name', None), body))
        if getattr(agent, 'name', None) == 'Chemistry Mechanism Consistency Check':
            assert body['output_schema']['properties']['issues']['items']['properties']['claim']['enum'] == [
                selected['recommendation']['rationale']]
            return SimpleNamespace(final_output='{"issues": []}')
        return SimpleNamespace(final_output=json.dumps(selected))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = pipeline_tools._run_chemistry_agent_for_goal(object(), {'intent': 'binding'})
    assert result['mechanism_review']['recommendation_sha256'] == recommendation_digest(result)
    assert len(calls) == 2


def test_native_bounded_repair_keeps_profile_and_corrects_supported_issue(monkeypatch):
    initial = proposal('The profile alone guarantees inclusion.')
    final = proposal('RNase-H-silent steric binding; the chosen site determines any splice effect.')
    goal = {'intent': 'splice_modulation', 'aso_length_nt': 21, 'target_context': 'general baseline'}
    prompts, audits = [], []
    async def run(agent, *, input, **kwargs):
        body = json.loads(input)
        if getattr(agent, 'name', None) == 'Chemistry Finding Adjudicator':
            return SimpleNamespace(final_output=json.dumps({'decisions': [
                {'finding_index': index, 'decision': 'uphold', 'reason': item['reason']}
                for index, item in enumerate(body['issues'])]}))
        if getattr(agent, 'name', None) == 'Chemistry Mechanism Consistency Check':
            audits.append(body)
            issues = [{'claim': 'guarantees inclusion', 'reason': 'Chemistry does not determine site action.'}] if len(audits) == 1 else []
            return SimpleNamespace(final_output=json.dumps({'issues': issues}))
        prompts.append(body)
        return SimpleNamespace(final_output=json.dumps(initial if len(prompts) == 1 else final))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = pipeline_tools._run_chemistry_agent_for_goal(object(), goal)
    assert result['recommendation'] == final['recommendation']
    assert len(prompts) == len(audits) == 2
    assert all(a['structured_goal'] == goal for a in audits)
    assert 'does not determine site action' in prompts[1]['output_contract_feedback']['validation_errors'][0]
    assert result['mechanism_review']['recommendation_sha256'] == recommendation_digest(result)
    assert result['mechanism_review']['independent_validation'] is False
    assert pipeline_tools._canonicalize_chemistry_payload(result)['mechanism_review'] == result['mechanism_review']
    changed = deepcopy(result)
    changed['recommendation']['rationale'] = 'A different unreviewed claim.'
    assert 'mechanism_review' not in pipeline_tools._canonicalize_chemistry_payload(changed)


def test_unavailable_checker_does_not_regenerate_valid_chemistry(monkeypatch):
    calls = []
    async def run(agent, **kwargs):
        calls.append(agent)
        if getattr(agent, 'name', None) == 'Chemistry Mechanism Consistency Check':
            raise TimeoutError('neutral network fixture')
        return SimpleNamespace(final_output=json.dumps(proposal('RNase-H-silent RNA binding.')))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    with pytest.raises(ValueError, match='chemistry_mechanism_check_unavailable'):
        pipeline_tools._run_chemistry_agent_for_goal(object(), {'intent': 'binding'})
    assert len(calls) == 2


def test_three_native_repair_attempts_remain_bounded(monkeypatch):
    calls = []
    async def run(agent, **kwargs):
        calls.append(agent)
        if getattr(agent, 'name', None) == 'Chemistry Mechanism Consistency Check':
            return SimpleNamespace(final_output='{}')
        return SimpleNamespace(final_output=json.dumps(proposal('RNase-H-silent RNA binding.')))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    with pytest.raises(ValueError, match='chemistry_contract_unresolved'):
        pipeline_tools._run_chemistry_agent_for_goal(object(), {'intent': 'binding'})
    # Three checker attempts, but no proposal regeneration for checker errors.
    assert len(calls) == 4
    assert sum(getattr(agent, 'name', None) == 'Chemistry Mechanism Consistency Check' for agent in calls) == 3


@pytest.mark.parametrize('invalid', ['invalid JSON', json.dumps({'issues': [{'claim': 'invented quote', 'reason': 'unsupported'}]})])
def test_checker_contract_repair_preserves_selected_chemistry_and_rationale(monkeypatch, invalid):
    selected = proposal('RNase-H-silent RNA binding with a site-dependent effect.')
    selections, audits = [], []
    async def run(agent, *, input, **kwargs):
        data = json.loads(input)
        if getattr(agent, 'name', None) == 'Chemistry Mechanism Consistency Check':
            audits.append(data)
            if len(audits) == 1:
                return SimpleNamespace(final_output=invalid)
            assert data['output_contract_feedback']['previous_response'] == invalid
            assert data['output_contract_feedback']['validation_errors']
            assert data['chemistry'] == audits[0]['chemistry']
            return SimpleNamespace(final_output='{"issues": []}')
        selections.append(data)
        return SimpleNamespace(final_output=json.dumps(selected))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = pipeline_tools._run_chemistry_agent_for_goal(object(), {'intent': 'binding'})
    assert len(selections) == 1 and len(audits) == 2
    assert result['recommendation'] == selected['recommendation']
    assert result['mechanism_review']['recommendation_sha256'] == recommendation_digest(result)


def test_checker_repair_and_upheld_mechanism_issue_share_one_budget(monkeypatch):
    calls = []
    async def run(agent, *, input, **kwargs):
        name = getattr(agent, 'name', 'selector')
        calls.append(name)
        data = json.loads(input)
        if name == 'Chemistry Mechanism Consistency Check':
            if calls.count(name) == 1:
                return SimpleNamespace(final_output='{}')
            return SimpleNamespace(final_output=json.dumps({'issues': [
                {'claim': 'guarantees inclusion', 'reason': 'Chemistry alone cannot determine the splice outcome.'}]}))
        if name == 'Chemistry Finding Adjudicator':
            return SimpleNamespace(final_output=json.dumps({'decisions': [
                {'finding_index': 0, 'decision': 'uphold', 'reason': 'The assertion is unsupported.'}]}))
        return SimpleNamespace(final_output=json.dumps(proposal('The profile alone guarantees inclusion.')))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    with pytest.raises(ValueError, match='chemistry_contract_unresolved'):
        pipeline_tools._run_chemistry_agent_for_goal(object(), {'intent': 'binding'})
    assert calls.count('selector') == 2
    assert calls.count('Chemistry Mechanism Consistency Check') == 3
    assert calls.count('Chemistry Finding Adjudicator') == 2
