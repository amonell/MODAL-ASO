"""Prediction transport must not change the quantity being reported."""
from copy import deepcopy
import json

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.evaluation_evidence import attach_measurement_references


@pytest.mark.parametrize('primary,selected,inclusion', [
    (0.0, -0.7, None), (-0.8, 0.0, 0.0), (None, 0.4, -0.2),
])
def test_selected_site_is_never_substituted_for_inclusion(primary, selected, inclusion):
    row = {'primary_label': 'acceptor@410', 'primary_delta_deletion': primary,
           'selected_delta_deletion': selected, 'selected_site_kind': 'acceptor',
           'delta_inclusion_deletion': inclusion}
    summary = pipeline._splice_summary_from_scored_row(row)
    assert summary['delta_inclusion_deletion'] == inclusion
    assert summary['delta_inclusion_replacement'] is None
    if primary is not None:
        assert summary['primary_delta_deletion'] == primary
    if selected is not None:
        assert summary['selected_delta_deletion'] == selected
    assert summary['selected_site_kind'] == 'acceptor'


@pytest.mark.parametrize('serialized', [False, True])
def test_per_site_measurements_survive_compaction_and_citation(serialized):
    sites = [{'site_kind': 'acceptor', 'reference_transcript_id': 'TX_NEUTRAL.2',
              'acceptor': {'chrom': 'chrN', 'position': 410, 'strand': '-'},
              'delta_acceptor_deletion': -0.73, 'delta_acceptor_replacement': 0.0,
              'delta_donor_deletion': None, 'delta_inclusion_deletion': None,
              'evaluation_status': 'evaluated'}]
    scored = {'splice_site_deltas': json.dumps(sites) if serialized else sites,
              'delta_inclusion_deletion': None, 'delta_deletion': '-0.7300',
              'delta_names': 'acceptor@410'}
    before = deepcopy(scored)
    summary = pipeline._splice_summary_from_scored_row(scored)
    compact = pipeline._compact_existing_aso_evaluation_for_response(
        {'provided_asos': [{'name': 'neutral', 'input_row_id': 'row-1',
                            'splice_eval_summary': summary}]})
    attach_measurement_references(compact)
    retained = compact['provided_asos'][0]['splice_eval_summary']
    assert retained['splice_site_deltas'] == scored['splice_site_deltas']
    assert retained['delta_inclusion_deletion'] is None
    reference = compact['measurement_evidence']['M001_splice_site_deltas']
    assert reference['value'] == scored['splice_site_deltas']
    assert reference['field'] == 'splice_eval_summary.splice_site_deltas'
    if not serialized:
        retained['splice_site_deltas'][0]['acceptor']['position'] = 999
    assert scored == before


def test_summary_names_the_metric_it_actually_selected():
    row = {'name': 'neutral', 'splice_eval_summary': {
        'delta_inclusion_deletion': -0.1, 'delta_replacement': -0.6,
        'delta_names': 'acceptor@410'}}
    text = '\n'.join(pipeline._provided_aso_knockdown_answer_paragraphs(
        [], gene='NEUTRAL', species='human', target_context='', splice_rows=[row]))
    assert '`delta_replacement` = -0.6000' in text
    assert '`delta_inclusion_deletion` = -0.6000' not in text
    assert pipeline._summary_row_skip_delta(row) == -0.6


def test_absent_metrics_stay_unknown_and_zero_stays_recorded():
    assert pipeline._summary_row_skip_measurement({}) == (None, None)
    row = {'splice_eval_summary': {'delta_inclusion_deletion': None, 'delta_deletion': 0.0}}
    assert pipeline._summary_row_skip_measurement(row) == ('delta_deletion', 0.0)
    selected = {'splice_eval_summary': {'primary_label': 'acceptor@410', 'primary_delta_deletion': 0.0}}
    assert pipeline._summary_row_has_splice_score(selected)
    assert pipeline._summary_row_skip_measurement(selected) == ('primary_delta_deletion', 0.0)
    assert pipeline._summary_splice_site_label(selected) == 'acceptor@410'


def test_audit_navigation_keeps_named_metrics_and_repeated_rows_in_their_tasks():
    from aso_agents.evaluation_response_audit import response_audit_context, pointer_value
    row = {'name': 'neutral', 'input_row_id': 'same-row', 'splice_eval_summary': {
        'delta_inclusion_replacement': None, 'splice_site_deltas': [
            {'site_kind': 'acceptor', 'delta_acceptor_replacement': -0.73,
             'delta_inclusion_replacement': None, 'delta_donor_replacement': 0.0}]}}
    other = deepcopy(row)
    other['splice_eval_summary'] = {'delta_inclusion_replacement': 0.0}
    source = {'completed_tasks': [{'payload': {'provided_asos': [row]}},
                                  {'payload': {'provided_asos': [other]}}]}
    before = deepcopy(source)
    context = response_audit_context(source, {'markdown': 'Recorded predictions only.'})
    index = context['splice_metric_index']
    assert len(index) == 2
    assert index[0]['row_source_pointer'] != index[1]['row_source_pointer']
    for entry in index:
        assert entry['row_identity']['input_row_id'] == 'same-row'
        for record in entry['measurements']:
            actual = pointer_value(source, record['source_pointer'])
            assert all(actual[key] == value for key, value in record['recorded_fields'].items())
    assert index[0]['measurements'][0]['recorded_fields']['delta_inclusion_replacement'] is None
    assert index[1]['measurements'][0]['recorded_fields']['delta_inclusion_replacement'] == 0.0
    assert context['source'] == source == before


def test_focused_check_only_runs_for_recorded_individual_site_metrics():
    from aso_agents.evaluation_response_audit import splice_metric_audit_packet
    draft = {'markdown': 'Reported prediction.'}
    assert splice_metric_audit_packet({'provided_asos': []}, draft) is None
    source = {'provided_asos': [{'splice_eval_summary': {'delta_inclusion_replacement': -0.3}}]}
    assert splice_metric_audit_packet(source, draft) is None
    source['provided_asos'][0]['splice_eval_summary']['delta_replacement'] = 0.0
    assert splice_metric_audit_packet(source, draft)['recorded_splice_evidence']


def test_unrelated_absence_claim_is_not_admitted_from_splice_only_packet():
    from aso_agents.evaluation_response_audit import filter_splice_metric_findings
    metric = {'quote': 'delta_acceptor_replacement is -0.73'}
    unrelated = {'quote': 'The exact off-target gene count is zero.'}
    admitted, rejected = filter_splice_metric_findings({'findings': [unrelated, metric]})
    assert admitted == [metric]
    assert [item['finding'] for item in rejected] == [unrelated]


def test_focused_review_repairs_only_the_bound_narrative(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from aso_agents.response_context import decode_response_context
    from aso_agents.evaluation_response_audit import splice_metric_finding_errors
    from aso_agents.workbench_response_agent import (
        evaluation_evidence_audit_agent, evaluation_evidence_repair_agent, splice_metric_audit_agent)
    source = {'task_type': 'build_composite_report', 'completed_tasks': [
        {'task_id': 'neutral', 'payload': {'provided_asos': [
            {'name': 'neutral', 'input_row_id': 'row-1', 'splice_eval_summary': {
                'delta_inclusion_replacement': None, 'delta_replacement': -0.73,
                'delta_names': 'acceptor@410'}}]}}]}
    before = deepcopy(source)
    first = {'answer_summary': 'Recorded predictions.',
             'markdown': 'Exon inclusion replacement delta is -0.73.'}
    corrected = 'The acceptor-site replacement delta is -0.73; inclusion was not recorded.'
    path = '/completed_tasks/0/payload/provided_asos/0/splice_eval_summary'
    finding = {'response_path': '/markdown', 'quote': first['markdown'],
               'kind': 'contradicted_fact', 'evidence_paths': [path],
               'reason': 'An individual acceptor value was relabeled as an unrecorded inclusion proxy.'}
    assert splice_metric_finding_errors({'findings': [finding]}, first, source) == []
    wrong = {**finding, 'evidence_paths': ['/completed_tasks/0/task_id']}
    assert splice_metric_finding_errors({'findings': [wrong]}, first, source)
    calls = []

    async def run(agent, **kwargs):
        body = json.loads(kwargs['input'])
        if body.get('context_encoding'):
            body = decode_response_context(body)
        calls.append(agent.name)
        if agent.name == evaluation_evidence_audit_agent.name:
            assert body['source'] == before
            answer = {'findings': []}
        elif agent.name == splice_metric_audit_agent.name:
            assert body['recorded_splice_evidence'][0]['row_identity']['name'] == 'neutral'
            answer = {'findings': [finding] if body['draft_narrative_fields']['/markdown'] == first['markdown'] else []}
        elif agent.name == evaluation_evidence_repair_agent.name:
            assert body['allowed_narrative_repair_paths'] == ['/markdown']
            assert body['source'] == before
            answer = {'corrections': {'/markdown': corrected}}
        else:
            raise AssertionError(agent.name)
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    result = asyncio.run(pipeline._run_workbench_response_batch_once(source, initial_response=first))
    assert result['markdown'] == corrected
    assert result['answer_summary'] == first['answer_summary']
    assert result['native_consistency_review']['splice_metric_identity_audit']
    assert calls.count(splice_metric_audit_agent.name) == 2
    assert source == before
