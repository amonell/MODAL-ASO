import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as native
from aso_agents.evaluation_response_audit import pointer_value, response_audit_context
from aso_agents.response_context import decode_response_context
from aso_agents.workbench_response_agent import evaluation_evidence_audit_agent, evaluation_evidence_repair_agent


def source():
    return {'task_type': 'build_composite_report', 'request_text': 'Summarize all completed tasks.',
            'completed_tasks': [
                {'task_id': 'requested', 'task_type': 'evaluate_existing_aso_set', 'payload': {
                    'context_summary': {'species': 'human', 'transcript_id': 'TX_ALPHA.3'},
                    'evaluation_targets': [{'exon_label': 4}],
                    'provided_asos': [
                        {'input_row_id': 'probe_a', 'name': 'Alpha', 'splice_eval_summary': {
                            'delta_inclusion_deletion': 0.0, 'delta_inclusion_replacement': .013,
                            'delta_names': 'donor@501'}},
                        {'input_row_id': 'probe_b', 'name': 'Beta', 'splice_eval_summary': {
                            'splice_eval_status': 'skipped_no_nearby_splice_site',
                            'splice_eval_skip_reason': 'no_splice_sites_within_context'}}]}},
                {'task_id': 'local_context', 'task_type': 'evaluate_existing_aso_set', 'payload': {
                    'context_summary': {'species': 'human', 'transcript_id': 'TX_ALPHA.3'},
                    'evaluation_targets': [{'exon_label': 9}],
                    'provided_asos': [
                        {'input_row_id': 'probe_b', 'name': 'Beta', 'splice_eval_summary': {
                            'delta_inclusion_replacement': -.2, 'delta_names': 'acceptor@902'}}]}},
                {'task_id': 'properties', 'task_type': 'sandbox_analysis', 'payload': {
                    'results': [{'input_row_id': 'probe_a', 'length': 20},
                                {'input_row_id': 'probe_b', 'length': 20}]}}]}


def draft(correct=False):
    return {'answer_summary': 'Two supplied molecules were examined.',
            'markdown': ('Alpha has recorded deltas at the requested target. Beta was skipped there '
                         'and separately scored at the local target.' if correct else
                         'Splice evaluation was skipped for Alpha and completed for Beta at the requested target.'),
            'limitations': ['Biological activity remains unmeasured.']}


def finding():
    return {'response_path': '/markdown', 'quote': draft()['markdown'], 'kind': 'contradicted_fact',
            'evidence_paths': [
                '/completed_tasks/0/payload/provided_asos/0/splice_eval_summary',
                '/completed_tasks/0/payload/provided_asos/1/splice_eval_summary',
                '/completed_tasks/1/payload/provided_asos/0/splice_eval_summary',
                '/completed_tasks/0/payload/evaluation_targets',
                '/completed_tasks/1/payload/evaluation_targets'],
            'reason': 'Alpha has recorded requested-target calculations; Beta is skipped there and scored in a separate target context.'}


def decode(prompt):
    body = json.loads(prompt)
    return decode_response_context(body) if body.get('context_encoding') else body


def test_composite_scope_index_keeps_each_task_row_and_zero_value_separate():
    payload = source()
    before = deepcopy(payload)
    context = response_audit_context(payload, draft())
    indexed = {item['source_pointer']: item['recorded_value'] for item in context['evidence_scope_index']}
    for path in finding()['evidence_paths']:
        assert indexed[path] == pointer_value(payload, path)
    assert indexed['/completed_tasks/0/payload/provided_asos/0/splice_eval_summary']['delta_inclusion_deletion'] == 0.0
    assert context['source'] == payload == before
    assert context['draft_response'] == draft()


def test_composite_repair_checks_every_task_without_changing_scientific_results(monkeypatch):
    payload = source()
    original = deepcopy(payload)
    calls, audited = [], []
    async def run(agent, **kwargs):
        body = decode(kwargs['input'])
        calls.append(agent.name)
        if agent.name == evaluation_evidence_audit_agent.name:
            assert body['source'] == payload
            audited.append(body['draft_response'])
            answer = {'findings': [finding()] if len(audited) == 1 else []}
        elif agent.name == evaluation_evidence_repair_agent.name:
            assert body['allowed_narrative_repair_paths'] == ['/markdown']
            assert body['source'] == payload
            answer = {'corrections': {'/markdown': draft(True)['markdown']}}
        else:
            answer = draft()
        return SimpleNamespace(final_output=json.dumps(answer))
    monkeypatch.setattr(native.Runner, 'run', run)
    result = asyncio.run(native._run_workbench_response_batch_once(payload))
    assert result['markdown'] == draft(True)['markdown']
    assert result['answer_summary'] == draft()['answer_summary']
    assert result['limitations'] == draft()['limitations']
    assert result['native_consistency_review']['attempts'] == 2
    assert calls.count(evaluation_evidence_repair_agent.name) == 1
    assert payload == original


def test_supported_composite_narrative_is_audited_and_preserved(monkeypatch):
    calls = []
    async def run(agent, **kwargs):
        calls.append(agent.name)
        answer = {'findings': []} if agent.name == evaluation_evidence_audit_agent.name else draft(True)
        return SimpleNamespace(final_output=json.dumps(answer))
    monkeypatch.setattr(native.Runner, 'run', run)
    result = asyncio.run(native._run_workbench_response_batch_once(source()))
    assert result.pop('native_consistency_review')['attempts'] == 1
    assert result == draft(True)
    assert calls == [native.workbench_response_agent.name, evaluation_evidence_audit_agent.name]


def test_composite_cannot_bypass_unavailable_evidence_review(monkeypatch):
    async def run(agent, **kwargs):
        return SimpleNamespace(final_output=json.dumps(draft(True)))
    async def unavailable(**kwargs):
        return {'error': 'fixture_review_unavailable'}
    monkeypatch.setattr(native.Runner, 'run', run)
    monkeypatch.setattr(native, '_run_schema_validated_agent_once', unavailable)
    with pytest.raises(RuntimeError, match='evaluation_evidence_consistency_review_failed'):
        asyncio.run(native._run_workbench_response_batch_once(source()))
