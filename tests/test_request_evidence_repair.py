import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents.request_evidence_repair import invalid_quote_paths, repair_audit_quotes


SOURCE = 'Design exactly two 18-nt ASOs and four 24-nt ASOs.'
AUDIT = {'program_candidate_counts': [
    {'length_nt': 18, 'count': 2, 'kind': 'exact', 'quote': 'exactly two 18-nt ASOs'},
    {'length_nt': 24, 'count': 4, 'kind': 'exact', 'quote': 'exactly four 24-nt ASOs'}]}
POINTER = '/program_candidate_counts/1/quote'


def invoke(reply, audit=None, source=SOURCE):
    calls = []
    async def run(agent, **kwargs):
        calls.append((agent.name, kwargs))
        return SimpleNamespace(final_output=json.dumps(reply))
    result = asyncio.run(repair_audit_quotes(deepcopy(AUDIT if audit is None else audit), source,
                         {'original_request': source}, run, json.loads))
    return *result, calls


def test_quote_repair_changes_only_the_unattested_string():
    repaired, receipt, calls = invoke({'repairs': [{'path': POINTER, 'quote': 'four 24-nt ASOs'}],
                                       'unresolved_paths': []})
    expected = deepcopy(AUDIT)
    expected['program_candidate_counts'][1]['quote'] = 'four 24-nt ASOs'
    assert repaired == expected
    assert receipt['status'] == 'repaired' and len(calls) == 1
    assert calls[0][1]['max_steps'] == 2
    evidence = json.loads(calls[0][1]['input'])
    assert evidence['audit'] == AUDIT and evidence['user_evidence']['original_request'] == SOURCE
    assert AUDIT['program_candidate_counts'][1]['quote'].startswith('exactly')


@pytest.mark.parametrize('reply', [
    {'repairs': [], 'unresolved_paths': [POINTER]},
    {'repairs': [{'path': POINTER, 'quote': 'five 24-nt ASOs'}], 'unresolved_paths': []},
    {'repairs': [{'path': '/program_candidate_counts/1/count', 'quote': 'four 24-nt ASOs'}]},
    {'repairs': [{'path': POINTER, 'quote': 'four 24-nt ASOs', 'count': 9}]},
    {'repairs': [{'path': POINTER, 'quote': 'four 24-nt ASOs'}, {'path': POINTER, 'quote': 'four 24-nt ASOs'}]},
    [],
    {'repairs': [{'path': POINTER, 'quote': None}]},
])
def test_failed_or_out_of_scope_repair_preserves_first_and_never_retries(reply):
    repaired, receipt, calls = invoke(reply)
    assert repaired == AUDIT
    assert receipt['status'] == 'unresolved' and len(calls) == 1


def test_valid_audit_makes_no_additional_model_call():
    valid = deepcopy(AUDIT)
    valid['program_candidate_counts'][1]['quote'] = 'four 24-nt ASOs'
    result, receipt, calls = invoke({}, valid)
    assert result == valid and not calls
    assert receipt == {'status': 'not_required', 'model_calls': 0}


def test_reference_quotes_are_checked_and_pointer_escaping_is_reversible():
    value = {'a/b~c': {'references_quote': 'missing'}}
    assert invalid_quote_paths(value, SOURCE)[0]['path'] == '/a~1b~0c/references_quote'
    reply = {'repairs': [{'path': '/a~1b~0c/references_quote', 'quote': 'four 24-nt ASOs'}]}
    result, receipt, _ = invoke(reply, value)
    assert result['a/b~c']['references_quote'] == 'four 24-nt ASOs'
    assert receipt['status'] == 'repaired'


def test_transport_failure_keeps_unsupported_first_and_error_provenance():
    async def fail(*args, **kwargs):
        raise TimeoutError('test transport')
    result, receipt = asyncio.run(repair_audit_quotes(AUDIT, SOURCE, {}, fail, json.loads))
    assert result == AUDIT
    assert receipt['error_type'] == 'TimeoutError'
    assert receipt['status'] == 'unresolved'


def test_native_audit_repairs_only_quotes_then_applies_original_typed_requirements(monkeypatch):
    import run_goal_intake as intake
    goal = {'design_programs': [{'program_id': 'short', 'modality': 'steric', 'intent': 'binding', 'length': 18},
                               {'program_id': 'long', 'modality': 'steric', 'intent': 'binding', 'length': 24}]}
    seen = []
    async def run(agent, **kwargs):
        seen.append(agent.name)
        value = AUDIT if len(seen) == 1 else {'repairs': [{'path': POINTER, 'quote': 'four 24-nt ASOs'}]}
        return SimpleNamespace(final_output=json.dumps(value))
    monkeypatch.setattr(intake.Runner, 'run', run)
    result = asyncio.run(intake._audit_request_obligations({'status': 'ready', 'structured_goal': goal}, SOURCE, []))
    assert result['status'] == 'ready'
    assert [p['candidate_limit'] for p in result['structured_goal']['design_programs']] == [2, 4]
    assert result['request_obligation_audit']['quote_repair']['status'] == 'repaired'
    assert len(seen) == 2


def test_retracts_only_unrequested_object_and_keeps_supported_siblings():
    audit = {'candidate_count': {'count': 3, 'scope': 'global', 'kind': 'exact',
                                'quote': 'Select three candidates'},
             'target_accessibility': {'required': True, 'quote': None}}
    result, receipt, calls = invoke(
        {'repairs': [], 'retractions': [{'path': '/target_accessibility', 'reason': 'not_requested'}]},
        audit, 'Select three candidates from the supplied interval.')
    assert result == {**audit, 'target_accessibility': None}
    assert receipt['status'] == 'repaired' and len(calls) == 1
    assert json.loads(calls[0][1]['input'])['eligible_retractions'] == ['/target_accessibility']
    assert audit['target_accessibility'] == {'required': True, 'quote': None}


def test_quote_correction_and_independent_retraction_share_one_existing_call():
    audit = {**deepcopy(AUDIT), 'target_feature': {'feature': 'utr3', 'quote': None}}
    result, receipt, calls = invoke({'repairs': [{'path': POINTER, 'quote': 'four 24-nt ASOs'}],
        'retractions': [{'path': '/target_feature', 'reason': 'not_requested'}]}, audit)
    assert result['target_feature'] is None
    assert result['program_candidate_counts'][1]['quote'] == 'four 24-nt ASOs'
    assert receipt['status'] == 'repaired' and len(calls) == 1


@pytest.mark.parametrize('audit,reply', [
    ({'target_accessibility': {'required': True, 'quote': None}},
     {'repairs': [], 'retractions': [{'path': '/candidate_count', 'reason': 'not_requested'}]}),
    ({'target_accessibility': {'required': True, 'quote': None}},
     {'repairs': [], 'retractions': [{'path': '/target_accessibility', 'reason': 'underspecified'}]}),
    ({'target_accessibility': {'required': True, 'quote': None}},
     {'repairs': [{'path': '/target_accessibility/quote', 'quote': 'Design exactly two 18-nt ASOs'}],
      'retractions': [{'path': '/target_accessibility', 'reason': 'not_requested'}]}),
    ({'target_accessibility': {'required': True, 'quote': None}},
     {'repairs': [], 'retractions': [{'path': '/target_accessibility', 'reason': 'not_requested'}] * 2}),
    ({'target_accessibility': {'required': True, 'quote': None}},
     {'repairs': [], 'retractions': [{'path': '/target_accessibility', 'reason': 'not_requested', 'value': False}]}),
    ({'shared_binding_cohort': {'quote': 'Design exactly two 18-nt ASOs', 'references_quote': None}},
     {'repairs': [], 'retractions': [{'path': '/shared_binding_cohort', 'reason': 'not_requested'}]}),
    ({'program_candidate_counts': deepcopy(AUDIT['program_candidate_counts'])},
     {'repairs': [], 'retractions': [{'path': '/program_candidate_counts/1', 'reason': 'not_requested'}]}),
    ({'unknown_obligation': {'quote': None}},
     {'repairs': [], 'retractions': [{'path': '/unknown_obligation', 'reason': 'not_requested'}]}),
    ({'target_accessibility': {'required': True, 'quote': None}, 'target_feature': {'quote': None}},
     {'repairs': [], 'retractions': [{'path': '/target_accessibility', 'reason': 'not_requested'}]}),
])
def test_retraction_cannot_discard_verified_evidence_or_escape_offered_scope(audit, reply):
    result, receipt, calls = invoke(reply, audit)
    assert result == audit
    assert receipt['status'] == 'unresolved' and len(calls) == 1


def test_real_but_underspecified_requirement_remains_unresolved():
    source = 'Select three candidates near the translation start.'
    audit = {'target_window': {'feature': 'translation_start', 'upstream_nt': 30,
                              'downstream_nt': 20, 'quote': 'within a 50 nt window'}}
    result, receipt, calls = invoke({'repairs': [], 'retractions': [],
                                    'unresolved_paths': ['/target_window/quote']}, audit, source)
    assert result == audit and receipt['status'] == 'unresolved'
    assert len(calls) == 1


def test_native_audit_applies_supported_count_after_retracting_unrequested_object(monkeypatch):
    import run_goal_intake as intake
    source = 'Select exactly three new candidates from the complete tiling library.'
    audit = {'candidate_count': {'count': 3, 'scope': 'global', 'kind': 'exact',
                                'quote': 'Select exactly three new candidates'},
             'target_accessibility': {'required': True, 'quote': None}}
    goal = {'candidate_minimum': 19, 'candidate_limit': 19,
            'design_programs': [{'program_id': 'neutral', 'modality': 'gapmer', 'intent': 'knockdown'}]}
    replies = [audit, {'repairs': [], 'retractions': [
        {'path': '/target_accessibility', 'reason': 'not_requested'}]}]
    calls = []
    async def run(agent, **kwargs):
        calls.append(agent.name)
        return SimpleNamespace(final_output=json.dumps(replies[len(calls) - 1]))
    monkeypatch.setattr(intake.Runner, 'run', run)
    result = asyncio.run(intake._audit_request_obligations(
        {'status': 'ready', 'structured_goal': deepcopy(goal)}, source, []))
    assert result['status'] == 'ready'
    assert result['structured_goal']['candidate_minimum'] == 3
    assert result['structured_goal']['candidate_limit'] == 3
    assert result['request_obligation_audit']['quote_repair']['status'] == 'repaired'
    assert len(calls) == 2
