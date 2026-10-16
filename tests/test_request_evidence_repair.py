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
