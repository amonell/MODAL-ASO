import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as native
from aso_agents.input_file_roles import (
    declared_candidate_files, with_file_role_schema, apply_file_role_checks,
)


def fixture():
    request = ('The records in /input/a.fa are submitted candidate target-site strings. '
               'Use /input/b.tsv only as the reference transcript, not as candidate observations.')
    envelope = {'tasks': [{'task_id': 'evaluate', 'context': {
        'provided_sequence_role': 'target_site_sequence',
        'provided_aso_files': [{'path': '/input/a.fa'}, {'path': '/input/b.tsv'}]}}]}
    prompt = native._request_graph_review_prompt(request, envelope)
    evidence = native._declared_evidence_sources_from_prompt(prompt)
    spans = native._evidence_span_map(evidence)
    base = native._schema_with_allowed_evidence_refs(
        native.REQUEST_GRAPH_REVIEW_OUTPUT_SCHEMA, ['request'], list(spans))
    files = declared_candidate_files(prompt)
    schema = with_file_role_schema(base, files)
    checks = {pointer: {'role': 'candidate_input' if index == 0 else 'reference_or_annotation_only',
                        'evidence_refs': ['request'], 'evidence_span_id': next(iter(spans)),
                        'reason': 'Role explicitly assigned in the original request.'}
              for index, pointer in enumerate(files)}
    return request, prompt, files, schema, checks, spans


def test_every_file_requires_its_own_bound_role_and_evidence():
    _, _, files, schema, checks, _ = fixture()
    answer = {'status': 'valid', 'issues': [], 'file_role_checks': checks}
    assert not native._request_graph_review_contract_errors(answer, schema)
    assert native._request_graph_review_contract_errors({'status': 'valid', 'issues': []}, schema)
    partial = deepcopy(answer)
    partial['file_role_checks'].pop(next(iter(files)))
    assert native._request_graph_review_contract_errors(partial, schema)
    wrong = deepcopy(answer)
    wrong['file_role_checks'][next(iter(files))]['evidence_span_id'] = 'invented'
    assert native._request_graph_review_contract_errors(wrong, schema)


@pytest.mark.parametrize('raw_status', ['valid', 'revise'])
def test_native_review_routes_reference_judgment_into_repair_without_an_extra_call(monkeypatch, raw_status):
    request, prompt, files, _, checks, _ = fixture()
    raw = {'status': raw_status, 'issues': [], 'file_role_checks': checks}
    original = deepcopy(raw)
    calls = []

    async def respond(agent, *, input, **kwargs):
        calls.append(input)
        assert input == prompt
        return SimpleNamespace(final_output=json.dumps(raw))

    monkeypatch.setattr(native.Runner, 'run', respond)
    result = asyncio.run(native._run_request_graph_review_agent_once(prompt))
    assert len(calls) == 1 and raw == original
    assert result['status'] == 'revise' and len(result['issues']) == 1
    issue = result['issues'][0]
    assert issue['category'] == 'misplaced_input'
    assert issue['path'] == list(files)[1]
    assert '/input/b.tsv' in issue['reason'] and 'artifact_refs' in issue['reason']
    assert issue['evidence_quote'] in request
    assert not native._request_graph_review_contract_errors(result)
    assert not native._request_graph_review_evidence_errors(result, request)


def test_declared_target_site_and_dual_role_candidates_are_not_removed():
    _, _, files, _, checks, spans = fixture()
    for check in checks.values():
        check['role'] = 'candidate_input'
        check['reason'] = 'The user explicitly assigns a candidate role, including target-site or dual use.'
    answer = {'status': 'valid', 'issues': [], 'file_role_checks': checks}
    materialized = native._materialize_evidence_span_ids(answer, spans)
    assert apply_file_role_checks(materialized, files) == {'status': 'valid', 'issues': []}
    assert all(entry['provided_sequence_role'] == 'target_site_sequence' for entry in files.values())


def test_an_existing_cited_file_issue_is_preserved_without_a_duplicate():
    _, _, files, _, checks, spans = fixture()
    answer = native._materialize_evidence_span_ids(
        {'status': 'valid', 'issues': [], 'file_role_checks': checks}, spans)
    first = apply_file_role_checks(answer, files)['issues'][0]
    answer['issues'] = [deepcopy(first)]
    answer['status'] = 'revise'
    result = apply_file_role_checks(answer, files)
    assert result['issues'] == [first]


def test_unestablished_role_becomes_an_evidence_resolution_issue_not_silent_deletion():
    _, _, files, _, checks, spans = fixture()
    for check in checks.values():
        check['role'] = 'not_established'
    result = apply_file_role_checks(native._materialize_evidence_span_ids(
        {'status': 'valid', 'issues': [], 'file_role_checks': checks}, spans), files)
    assert result['status'] == 'revise'
    assert all(item['category'] == 'missing_fact' and 'preserve the file' in item['reason']
               for item in result['issues'])


def test_original_review_wrappers_preserve_inventory_but_evidence_text_cannot_define_it():
    _, prompt, files, _, _, _ = fixture()
    adjudication = native._request_graph_review_adjudication_prompt(prompt, {'status': 'valid', 'issues': []})
    retry = native._request_graph_review_retry_prompt(adjudication, {}, ['invalid citation'])
    assert declared_candidate_files(adjudication) == declared_candidate_files(retry) == files
    no_files = native._request_graph_review_prompt(prompt, {'tasks': []})
    assert declared_candidate_files(no_files) == {}
    base = deepcopy(native.REQUEST_GRAPH_REVIEW_OUTPUT_SCHEMA)
    assert with_file_role_schema(base, {}) == base
    answer = {'status': 'valid', 'issues': []}
    assert apply_file_role_checks(answer, {}) == answer
