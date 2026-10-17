from copy import deepcopy
from types import SimpleNamespace
import asyncio
import json

import pytest

from aso_agents.request_obligation_completion import complete_panel_obligation


SOURCE = 'Use TX_ALPHA.4. Compare complete panels by the ordered ascending mature-start tuple.'
QUOTE = 'Compare complete panels by the ordered ascending mature-start tuple.'


def audit():
    return {'candidate_count': {'count': 3, 'kind': 'exact', 'quote': 'three candidates'},
            'candidate_panel_ranking': {'applies_to_all_candidates': True, 'scope': 'joint_panel',
                'quote': QUOTE, 'criteria': [{'metric': 'mature_start_1based',
                    'aggregate': 'sorted_ascending', 'direction': 'minimize'}]}}


def completed():
    panel = deepcopy(audit()['candidate_panel_ranking'])
    panel['criteria'][0]['reference_transcript_id'] = 'TX_ALPHA.4'
    return panel


def run_completion(original, response, source=SOURCE):
    calls = []
    async def run(agent, *, input, max_steps):
        calls.append(json.loads(input))
        assert not agent.tools and max_steps == 4
        return SimpleNamespace(final_output=json.dumps(response))
    result, receipt = asyncio.run(complete_panel_obligation(
        original, source, {'original_request': source}, run, json.loads))
    return result, receipt, calls


def test_missing_reference_is_completed_once_with_unchanged_scientific_values():
    original = audit()
    before = deepcopy(original)
    result, receipt, calls = run_completion(original, completed())
    assert original == before and result is not original
    assert result['candidate_count'] == before['candidate_count']
    assert result['candidate_panel_ranking'] == completed()
    assert receipt['status'] == 'completed' and receipt['attempts'] == len(calls) == 1
    assert receipt['added_paths'] == [['criteria', 0, 'reference_transcript_id']]
    assert calls[0]['original_evidence'] == {'original_request': SOURCE}
    assert calls[0]['panel_obligation'] == before['candidate_panel_ranking']


@pytest.mark.parametrize('change', [
    lambda p: p['criteria'][0].update(reference_transcript_id='TX_ALPHA'),
    lambda p: p['criteria'][0].update(reference_transcript_id='TX_BETA.9'),
    lambda p: p['criteria'][0].update(direction='maximize'),
    lambda p: p['criteria'][0].update(absolute_distance_from=0),
    lambda p: p['criteria'].append(deepcopy(p['criteria'][0])),
    lambda p: p.update(quote='An invented quotation'),
    lambda p: p.update(extra='unrequested'),
    lambda p: p['criteria'][0].pop('reference_transcript_id'),
])
def test_unverified_or_changed_constraints_cannot_be_admitted(change):
    original, response = audit(), completed()
    before = deepcopy(original)
    change(response)
    result, receipt, calls = run_completion(original, response)
    assert result == original == before and result is original
    assert receipt['status'] == 'unresolved' and receipt['attempts'] == len(calls) == 1


@pytest.mark.parametrize('original', [{}, {'candidate_panel_ranking': None},
    {'candidate_panel_ranking': completed()}])
def test_absent_or_valid_contract_makes_no_extra_call(original):
    result, receipt, calls = run_completion(original, None)
    assert result is original and calls == [] and receipt['status'] == 'not_needed'


def test_invalid_existing_value_is_outside_missing_field_completion():
    original = audit()
    original['candidate_panel_ranking']['criteria'][0]['aggregate'] = 'invented'
    result, receipt, calls = run_completion(original, completed())
    assert result is original and calls == [] and receipt['status'] == 'not_completable'


def test_unverified_quote_cannot_trigger_completion():
    original = audit()
    original['candidate_panel_ranking']['quote'] = 'A quote absent from the request'
    result, receipt, calls = run_completion(original, completed())
    assert result is original and calls == [] and receipt['status'] == 'not_completable'


def test_transport_failure_preserves_original_for_existing_validation():
    original = audit()
    async def run(*args, **kwargs):
        raise ConnectionError('not available')
    result, receipt = asyncio.run(complete_panel_obligation(original, SOURCE, {}, run, json.loads))
    assert result is original and receipt['status'] == 'unresolved' and receipt['attempts'] == 1


def test_native_audit_applies_completed_contract_without_redundant_clarification(monkeypatch):
    import run_goal_intake as intake
    original = audit()
    original.pop('candidate_count')
    seen = []
    async def run(agent, *, input, max_steps):
        seen.append(agent.name)
        value = original if len(seen) == 1 else completed()
        return SimpleNamespace(final_output=json.dumps(value))
    monkeypatch.setattr(intake.Runner, 'run', run)
    goal = {'intent': 'binding', 'design_programs': [
        {'program_id': 'one', 'modality': 'steric', 'intent': 'binding', 'length': 23}]}
    result = asyncio.run(intake._audit_request_obligations(
        {'status': 'ready', 'structured_goal': goal}, SOURCE, []))
    assert result['status'] == 'ready' and not result.get('questions')
    assert result['structured_goal']['candidate_panel_objectives'] == completed()['criteria']
    assert result['request_obligation_audit']['contract_completion']['status'] == 'completed'
    assert seen == ['ASO Request Obligation Auditor', 'ASO Obligation Contract Completion']
