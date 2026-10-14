from copy import deepcopy
import asyncio
import json
from types import SimpleNamespace
import pytest

from aso_agents.evaluation_response_audit import evidence_pointer_candidates
from aso_agents.evaluation_response_audit import pointer_value


def test_nested_scope_hint_is_existing_and_does_not_mutate_source():
    source = {'evaluation': {'scoring': {'scope': {'search': 'full', 'count': 'subset'}}}}
    before = deepcopy(source)
    hint = evidence_pointer_candidates(source, '/evaluation/scope')
    assert hint == {'candidates': ['/evaluation/scoring/scope'], 'total_candidates': 1,
                    'searched_parent': '/evaluation'}
    assert pointer_value(source, hint['candidates'][0]) == {'search': 'full', 'count': 'subset'}
    assert source == before


def test_a_missing_row_field_is_not_borrowed_from_another_row():
    source = {'rows': [{'scoring': {'scope': 'this row'}}, {'scope': 'different row'}]}
    hint = evidence_pointer_candidates(source, '/rows/0/scope')
    assert hint['candidates'] == ['/rows/0/scoring/scope']
    assert hint['searched_parent'] == '/rows/0'
    assert evidence_pointer_candidates({'rows': [{}, {'scope': 'other'}]}, '/rows/0/scope')['candidates'] == []


def test_escaped_keys_and_list_indices_are_preserved():
    source = {'a/b': [{'inner': {'~scope': 17}}]}
    hint = evidence_pointer_candidates(source, '/a~1b/0/~0scope')
    assert hint['candidates'] == ['/a~1b/0/inner/~0scope']
    assert pointer_value(source, hint['candidates'][0]) == 17


def test_ambiguous_candidates_are_reported_as_hints_in_stable_bounded_order():
    source = {'root': {f'part_{i:02d}': {'scope': i} for i in range(12)}}
    hint = evidence_pointer_candidates(source, '/root/scope')
    assert hint['total_candidates'] == 12 and len(hint['candidates']) == 8
    assert hint['candidates'] == sorted(hint['candidates'])
    assert all(pointer_value(source, path) is not None for path in hint['candidates'])


@pytest.mark.parametrize('pointer', [None, '', 'not/a/pointer', '/rows/900', '/rows/-1', '/rows/00'])
def test_malformed_or_missing_array_indices_are_not_guessed(pointer):
    assert evidence_pointer_candidates({'rows': [{'scope': 'known'}]}, pointer)['candidates'] == []


def test_valid_pointer_needs_no_repair_and_unrelated_missing_key_gets_no_hint():
    source = {'scope': {'value': 3}}
    assert evidence_pointer_candidates(source, '/scope/value')['candidates'] == []
    assert evidence_pointer_candidates(source, '/scope/not_present')['candidates'] == []


def test_invalid_citation_remains_rejected_even_when_a_valid_hint_is_available():
    from aso_agents.evaluation_response_audit import response_audit_contract_errors, response_audit_repair_findings
    source = {'evaluation': {'scoring': {'scope': {'search': 'full', 'count': 'subset'}}}}
    response = {'answer_summary': 'The excluded regions were not searched.'}
    audit = {'findings': [{'response_path': '/answer_summary', 'quote': response['answer_summary'],
                          'kind': 'contradicted_fact', 'evidence_paths': ['/evaluation/scope'],
                          'reason': 'Count exclusion does not imply absence from the search.'}]}
    before = deepcopy((source, response, audit))
    errors = response_audit_contract_errors(audit, response, source)
    assert len(errors) == 1 and '/evaluation/scoring/scope' in errors[0]
    with pytest.raises(ValueError, match='unknown source evidence pointer'):
        response_audit_repair_findings(audit, response, source)
    assert (source, response, audit) == before


def test_native_schema_retry_exposes_candidates_without_rewriting_the_finding(monkeypatch):
    from aso_agents import pipeline_tools
    from aso_agents.evaluation_response_audit import response_audit_schema, response_audit_contract_errors
    from aso_agents.workbench_response_agent import evaluation_evidence_audit_agent
    source = {'evaluation': {'scoring': {'scope': {'search': 'full', 'count': 'subset'}}}}
    response = {'answer_summary': 'The excluded regions were not searched.'}
    finding = {'response_path': '/answer_summary', 'quote': response['answer_summary'],
               'kind': 'contradicted_fact', 'evidence_paths': ['/evaluation/scope'],
               'reason': 'Count exclusion does not imply absence from the search.'}
    corrected = {**finding, 'evidence_paths': ['/evaluation/scoring/scope']}
    prompts = []

    async def run(agent, **kwargs):
        prompts.append(kwargs['input'])
        if len(prompts) > 1:
            assert '/evaluation/scoring/scope' in kwargs['input']
            assert 'hints only' in kwargs['input']
        return SimpleNamespace(final_output=json.dumps({'findings': [finding if len(prompts) == 1 else corrected]}))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    schema = response_audit_schema(response)
    result = asyncio.run(pipeline_tools._run_schema_validated_agent_once(
        agent=evaluation_evidence_audit_agent,
        prompt=json.dumps({'source': source, 'draft_response': response}),
        schema=schema, error_prefix='neutral_evidence_pointer_control',
        contract_validator=lambda body: response_audit_contract_errors(body, response, source),
    ))
    assert len(prompts) == 2
    assert result == {'findings': [corrected]}
    assert finding['evidence_paths'] == ['/evaluation/scope']
