import asyncio
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as p
from aso_agents.response_assembly import (
    apply_scope_edits, assembly_audit_agent, assembly_repair_agent, assembly_source,
    delivered_narrative_fields, narrative_pages, review_assembled_response, scope_finding_errors,
    scope_review_context,
)
from aso_agents.response_context import merge_evaluation_responses


def fixture():
    rows = [{'input_row_id': 'input_a', 'name': 'duplicate', 'sequence': 'AACG'},
            {'input_row_id': 'input_b', 'name': 'duplicate', 'sequence': 'TTAG'}]
    payload = {'task_type': 'evaluate_existing_aso_set', 'request_text': 'Evaluate both molecules.',
               'evaluation_summary': {'provided_asos': rows, 'provided_aso_count': 2}}
    parts = [{'answer_summary': 'Only one observation was supplied.', 'markdown': '',
              'evaluation_assessments': [{**row, 'mechanistic_plausibility': 'Activity is unmeasured.',
                  'ranking_rationale': 'No potency ranking is established.'}]} for row in rows]
    response = merge_evaluation_responses(parts, payload)
    finding = {'response_path': '/answer_summary', 'quote': 'Only one observation was supplied.',
        'kind': 'contradicted_fact', 'evidence_paths': ['/request_scope/observation_count'],
        'scope_kind': 'request_membership',
        'reason': 'Two original observations were supplied and delivered.'}
    return payload, response, finding


def test_source_register_retains_distinct_ids_and_actual_delivery_without_scientific_sampling():
    payload, response, _ = fixture()
    before = deepcopy((payload, response))
    source = assembly_source(payload, response)
    assert source['request_scope']['observations'] == [
        {key: row[key] for key in ('input_row_id', 'name')}
        for row in payload['evaluation_summary']['provided_asos']]
    assert 'request_text' not in source
    assert source['delivery_scope']['assessed_input_row_ids'] == ['input_a', 'input_b']
    assert source['delivery_scope'] == {
        'assessment_count': 2, 'assessed_input_row_ids': ['input_a', 'input_b']}
    assert [part['evaluation_assessments'][0]['input_row_id']
            for part in response['original_component_responses']] == ['input_a', 'input_b']
    assert (payload, response) == before
    assert all(not path.startswith('/original_component_responses') for path in delivered_narrative_fields(response))


def test_evidence_availability_distinguishes_retained_empty_missing_and_declared_inventory():
    payload, response, _ = fixture()
    a, b = payload['evaluation_summary']['provided_asos']
    a.update(mapping_hits=[], near_match_candidates=[{'sequence': 'TTAA'}],
        evidence_inventory={'exact_mapping_count': 0, 'retained_near_match_count': 1})
    b['evidence_inventory'] = {'exact_mapping_count': 2}
    before = deepcopy((payload, response))
    evidence = assembly_source(payload, response)['evidence_scope']['observations']
    assert evidence[0]['mapping_hits']['records_present'] and evidence[0]['mapping_hits']['record_count'] == 0
    assert evidence[0]['near_match_candidates']['records_present'] and evidence[0]['near_match_candidates']['record_count'] == 1
    assert evidence[0]['near_match_candidates']['raw_geometry_availability']['raw_segment_count_unknown_record_count'] == 1
    assert evidence[1]['mapping_hits'] == {'records_present': False, 'record_count': None}
    assert evidence[1]['recorded_inventory'] == {'exact_mapping_count': 2}
    assert evidence[0]['available_alignment_record_count'] == 1
    assert evidence[1]['available_alignment_record_count'] == 0
    assert (payload, response) == before
    assert 'sequence' not in json.dumps(evidence)


def test_report_content_contract_is_separate_from_runtime_file_existence():
    payload, response, _ = fixture()
    payload['request_envelope'] = {'final_outputs': [{'kind': 'panel.md'}]}
    source = assembly_source(payload, response)
    rendering = source['rendering_scope']
    assert rendering['narrative_perspective'] == 'delivered_report'
    assert 'stage' not in rendering
    assert rendering['requested_outputs'] == [{'kind': 'panel.md'}]
    assert 'every available retained evidence record' in rendering['contract']
    assert 'Artifact existence and export success are verified after rendering' in rendering['verification_boundary']


@pytest.mark.parametrize('exact_count,register_count,junction_count', [(2, 0, 1), (0, 1, 0), (0, 0, 0)])
def test_assembly_preserves_independent_feature_evidence_registers(exact_count, register_count, junction_count):
    from aso_agents.evaluation_evidence import row_evidence_inventory
    payload, response, _ = fixture()
    row = payload['evaluation_summary']['provided_asos'][0]
    row['mapping_hits'] = [{'mismatch_count': 0} for _ in range(exact_count)]
    row['near_match_candidates'] = [{
        'mismatch_count': 2,
        'requested_site_overlaps': [{'site_id': 'neutral_site'}] if register_count else [],
        'reference_junctions_spanned': [{'boundary': 101}] if junction_count else [],
    }]
    row['evidence_inventory'] = row_evidence_inventory(row)
    before = deepcopy(payload)
    source = assembly_source(payload, response)
    exported = source['evidence_scope']['observations'][0]['recorded_inventory']
    features = exported['feature_evidence_availability']
    assert exported['exact_mapping_count'] == exact_count
    assert features == row['evidence_inventory']['feature_evidence_availability']
    assert features['requested_site_register']['source_record_count'] == register_count
    assert features['independent_reference_junction_geometry']['source_record_count'] == junction_count
    assert features['requested_feature_membership']['status'] == 'not_recorded'
    assert 'retained_hypotheses' not in exported
    features['requested_site_register']['status'] = 'changed by consumer'
    assert payload == before


def test_missing_feature_inventory_stays_unknown_in_assembly_context():
    payload, response, _ = fixture()
    row = payload['evaluation_summary']['provided_asos'][0]
    row.update(mapping_hits=[{}], evidence_inventory={'exact_mapping_count': 1})
    source = assembly_source(payload, response)
    assert 'feature_evidence_availability' not in source['evidence_scope']['observations'][0]['recorded_inventory']


@pytest.mark.parametrize('other_gene_count,status', [
    (0, 'complete_within_declared_scope'), (3, 'complete_within_declared_scope'),
    (None, 'incomplete_genomic_query_overflow'),
])
def test_assembly_keeps_on_target_records_separate_from_screening_counts(other_gene_count, status):
    payload, response, _ = fixture()
    row = payload['evaluation_summary']['provided_asos'][0]
    row.update(mapping_hits=[{}, {}], off_target_gene_count=other_gene_count,
               off_target_observed_gene_count=0, off_target_count_status=status,
               off_target_count_scope={'count_unit': 'distinct other genes',
                                       'intended_target_exclusion': 'whole intended gene'})
    stage = {'status': 'not_run', 'reason': 'no_exact_off_target_hits'}
    payload['evaluation_summary']['provided_aso_scoring'] = {'off_target_expression': stage}
    before = deepcopy(payload)
    evidence = assembly_source(payload, response)['evidence_scope']
    observed = evidence['observations'][0]
    assert observed['mapping_hits']['record_count'] == 2
    assert observed['recorded_off_target_screen']['off_target_gene_count'] == other_gene_count
    assert observed['recorded_off_target_screen']['off_target_count_status'] == status
    assert evidence['observations'][1]['recorded_off_target_screen'] == {}
    assert evidence['recorded_screening_stages']['off_target_expression'] == stage
    observed['recorded_off_target_screen']['off_target_count_scope']['count_unit'] = 'modified'
    evidence['recorded_screening_stages']['off_target_expression']['status'] = 'modified'
    assert payload == before


def test_window_group_completeness_is_retained_without_reopening_alignment_geometry():
    payload, response, _ = fixture()
    row = payload['evaluation_summary']['provided_asos'][0]
    row.update(mapping_hits=[{}], near_match_candidates=[{}, {}], evidence_inventory={
        'recorded_window_identity': {
            'identified_window_group_count': 2, 'complete_for_retained_records': True,
            'unresolved_source_fields': [], 'scope': 'Reference-bound aligned window identity.',
            'identified_window_groups': [{'identity': {'sequence': 'ACGT'}}]}})
    source = assembly_source(payload, response)
    observation = source['evidence_scope']['observations'][0]
    assert observation['available_alignment_record_count'] == 3
    identity = observation['recorded_inventory']['recorded_window_identity']
    assert identity['identified_window_group_count'] == 2
    assert identity['complete_for_retained_records'] is True
    assert identity['unresolved_source_fields'] == []
    assert 'identified_window_groups' not in identity
    assert 'ACGT' not in json.dumps(source)
    assert 'recorded_window_identity' not in source['evidence_scope']['observations'][1]['recorded_inventory']


def test_proxy_order_does_not_become_input_register_order_or_efficacy():
    payload, response, _ = fixture()
    rows = payload['evaluation_summary']['provided_asos']
    for row, rank in zip(rows, [2, 1]):
        row.update(knockdown_proxy_rank=rank, ranking_scope='generic proxy only; not efficacy')
    source = assembly_source(payload, response)
    assert [r['input_row_id'] for r in source['request_scope']['observations']] == ['input_a', 'input_b']
    rankings = [r['recorded_proxy_ranking'] for r in source['evidence_scope']['observations']]
    assert [r['knockdown_proxy_rank'] for r in rankings] == [2, 1]
    assert all(r['ranking_scope'] == 'generic proxy only; not efficacy' for r in rankings)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'foreign'])
def test_incomplete_or_corrupt_assembly_cannot_reach_scope_auditor(change):
    payload, response, _ = fixture()
    if change == 'missing':
        response['evaluation_assessments'].pop()
    elif change == 'duplicate':
        response['evaluation_assessments'].append(response['evaluation_assessments'][0])
    else:
        response['evaluation_assessments'][0]['input_row_id'] = 'foreign'
    with pytest.raises(ValueError):
        assembly_source(payload, response)


def test_only_source_bound_span_can_change_preserving_science_and_original_components():
    payload, response, finding = fixture()
    response['answer_summary'] = 'Activity is unmeasured. ' + finding['quote'] + ' {{evidence:abc}} remains uncertain.'
    source = assembly_source(payload, response)
    before = deepcopy(response)
    fixed = apply_scope_edits(response, {'replacements': {'0': 'Both observations were assessed.'}}, [finding], source)
    assert fixed['answer_summary'] == 'Activity is unmeasured. Both observations were assessed. {{evidence:abc}} remains uncertain.'
    assert response == before
    assert {k: v for k, v in fixed.items() if k != 'answer_summary'} == {k: v for k, v in before.items() if k != 'answer_summary'}


@pytest.mark.parametrize('change', ['wrong_quote', 'duplicate_quote', 'overlap', 'archive', 'wrong_evidence'])
def test_invalid_scope_findings_cannot_authorize_edits(change):
    payload, response, finding = fixture()
    findings = [finding]
    if change == 'wrong_quote':
        finding['quote'] = 'never supplied'
    elif change == 'duplicate_quote':
        response['answer_summary'] *= 2
    elif change == 'overlap':
        findings.append(deepcopy(finding))
    elif change == 'archive':
        finding['response_path'] = '/original_component_responses/0/answer_summary'
    else:
        finding['evidence_paths'] = ['/unknown']
    assert scope_finding_errors({'findings': findings}, response, assembly_source(payload, response))


def test_paging_covers_every_field_in_order_without_truncation():
    fields = {f'/items/{i}': 'x' * i for i in range(1, 100)}
    pages = narrative_pages(fields, target_bytes=300)
    assert len(pages) > 1
    assert {key: value for page in pages for key, value in page.items()} == fields


def test_long_report_scope_review_keeps_small_complete_narrative_pages():
    fields = {f'/limitations/{i}': 'An explicit scope and uncertainty. ' * 20 for i in range(120)}
    fields['/answer_summary'] = 'A long indivisible field. ' * 600
    pages = narrative_pages(fields)
    assert len(pages) > 6
    assert {key: value for page in pages for key, value in page.items()} == fields
    assert all(len(page) == 1 or len(json.dumps(page, ensure_ascii=False).encode()) < 12_000
               for page in pages)


def test_scope_context_preserves_literal_pointer_identity_and_unknown_evidence():
    payload, response, _ = fixture()
    source = assembly_source(payload, response)
    value = {'source': source, 'draft_narrative_fields': delivered_narrative_fields(response)}
    decoded = json.loads(scope_review_context(value))
    assert decoded == value
    assert decoded['source']['evidence_scope']['observations'][1]['mapping_hits'] == {
        'records_present': False, 'record_count': None}
    assert '$table' not in scope_review_context(value)


def test_assembly_cannot_reinterpret_sequence_identity_or_other_scientific_fields():
    payload, response, _ = fixture()
    response['evaluation_assessments'][0]['mechanistic_plausibility'] = 'The supplied sequence role is unknown.'
    source = assembly_source(payload, response)
    finding = {'response_path': '/evaluation_assessments/0/mechanistic_plausibility',
        'quote': 'The supplied sequence role is unknown.', 'kind': 'contradicted_fact',
        'scope_kind': 'request_membership', 'evidence_paths': ['/request_scope/observations/0'],
        'reason': 'This observation is present in the request.'}
    assert finding['response_path'] not in delivered_narrative_fields(response)
    with pytest.raises(ValueError, match='delivered narrative field'):
        apply_scope_edits(response, {'replacements': {'0': 'The molecule has a confirmed role.'}}, [finding], source)
    assert response['evaluation_assessments'][0]['mechanistic_plausibility'] == finding['quote']


def test_scope_findings_require_evidence_from_their_declared_dimension():
    payload, response, finding = fixture()
    finding['scope_kind'] = 'retained_evidence_availability'
    assert any('scope register' in error for error in scope_finding_errors(
        {'findings': [finding]}, response, assembly_source(payload, response)))


def test_independent_pages_overlap_execution_without_changing_the_draft(monkeypatch):
    payload, response, _ = fixture()
    response['answer_summary'] = 'Both observations are assessed.'
    response['limitations'] = ['Activity remains unmeasured. ' * 40 for _ in range(30)]
    before = deepcopy(response)
    active, maximum, calls = 0, 0, []
    barrier = threading.Barrier(len(narrative_pages(delivered_narrative_fields(response))), timeout=3)
    guard = threading.Lock()
    async def run(agent, **kwargs):
        nonlocal active, maximum
        assert agent.name == assembly_audit_agent.name
        with guard:
            active += 1
            maximum = max(maximum, active)
            calls.append(json.loads(kwargs['input'])['draft_narrative_fields'])
        barrier.wait()
        with guard:
            active -= 1
        return SimpleNamespace(final_output=json.dumps({'findings': []}))
    monkeypatch.setattr(p.Runner, 'run', run)
    result = asyncio.run(review_assembled_response(payload, response))
    assert maximum > 1
    assert {key: value for page in calls for key, value in page.items()} == delivered_narrative_fields(response)
    assert response == before
    assert {key: value for key, value in result.items() if key != 'native_assembly_review'} == before


def test_native_audit_repairs_scope_then_rechecks_without_recomposing_or_remapping(monkeypatch, tmp_path):
    payload, response, finding = fixture()
    source_before, response_before = deepcopy(payload), deepcopy(response)
    calls = []
    async def run(agent, **kwargs):
        calls.append(agent.name)
        data = json.loads(kwargs['input'])
        if agent.name == assembly_audit_agent.name:
            assert data['source']['request_scope']['observation_count'] == 2
            output = {'findings': [finding] if len(calls) == 1 else []}
        else:
            assert agent.name == assembly_repair_agent.name
            output = {'replacements': {'0': 'Both supplied observations were assessed.'}}
        return SimpleNamespace(final_output=json.dumps(output))
    monkeypatch.setattr(p.Runner, 'run', run)
    result = asyncio.run(review_assembled_response(payload, response, artifact_dir=tmp_path))
    assert calls == [assembly_audit_agent.name, assembly_repair_agent.name, assembly_audit_agent.name]
    assert result['answer_summary'] == 'Both supplied observations were assessed.'
    assert result['evaluation_assessments'] == response_before['evaluation_assessments']
    assert result['original_component_responses'] == response_before['original_component_responses']
    assert payload == source_before and response == response_before
    count = len(calls)
    cached = asyncio.run(review_assembled_response(payload, response, artifact_dir=tmp_path))
    assert cached == result and len(calls) == count


def test_consistent_assembly_needs_no_repair_and_no_scientific_changes(monkeypatch):
    payload, response, _ = fixture()
    response['answer_summary'] = 'Both observations are assessed; activity remains uncertain.'
    before = deepcopy(response)
    async def run(agent, **kwargs):
        assert agent.name == assembly_audit_agent.name
        return SimpleNamespace(final_output=json.dumps({'findings': []}))
    monkeypatch.setattr(p.Runner, 'run', run)
    result = asyncio.run(review_assembled_response(payload, response))
    assert {k: v for k, v in result.items() if k != 'native_assembly_review'} == before
    assert response == before


def test_raw_segment_availability_is_separate_from_complete_window_identity():
    from aso_agents.evaluation_evidence import row_evidence_inventory
    payload, response, _ = fixture()
    row = payload['evaluation_summary']['provided_asos'][0]
    locus = {'transcript_id': 'TX_NEUTRAL.2', 'strand': '+',
        'rna_context': 'unspliced_transcript_locus_pre_mrna', 'mapping_scope': 'pre_mrna_locus',
        'match_orientation': 'reverse_complement', 'matched_window_sequence': 'ACGTACGT',
        'mismatch_count': 0, 'mismatch_positions_in_supplied_sequence_1based': [],
        'genomic_span': {'chrom': 'chr4', 'start': 101, 'end': 108}}
    known = {**deepcopy(locus), 'genomic_segments': [{'chrom': 'chr4', 'start': 101, 'end': 108}]}
    row['mapping_hits'] = []
    row['near_match_candidates'] = [locus, known]
    row['evidence_inventory'] = row_evidence_inventory(row)
    assert row['evidence_inventory']['recorded_window_identity']['complete_for_retained_records']
    before = deepcopy(payload)
    source = assembly_source(payload, response)
    raw = source['evidence_scope']['observations'][0]['near_match_candidates']['raw_geometry_availability']
    assert raw['raw_segment_count_known_record_count'] == 1
    assert raw['raw_segment_count_unknown_record_count'] == 1
    assert raw['validated_contiguous_locus_record_count'] == 2
    assert raw['validated_contiguous_locus_with_unknown_raw_segment_count'] == 1
    assert payload == before


def test_ambiguity_alone_does_not_authorize_changes_or_another_model_call(monkeypatch, tmp_path):
    payload, response, finding = fixture()
    response['answer_summary'] = 'Every supplied observation has retained alignment evidence.'
    finding.update(kind='ambiguous_evidence_scope', quote=response['answer_summary'],
        reason='A terminology preference does not contradict the recorded coverage.')
    before = deepcopy(response); calls = []
    async def run(agent, **kwargs):
        calls.append(agent.name)
        return SimpleNamespace(final_output=json.dumps({'findings': [finding]}))
    monkeypatch.setattr(p.Runner, 'run', run)
    result = asyncio.run(review_assembled_response(payload, response, artifact_dir=tmp_path))
    assert calls == [assembly_audit_agent.name]
    assert {k:v for k,v in result.items() if k != 'native_assembly_review'} == before
    assert result['native_assembly_review']['noncontradiction_observations_not_applied'] == 1
    assert response == before
    with pytest.raises(ValueError, match='factual contradiction'):
        apply_scope_edits(response, {'replacements': {'0': 'Changed'}}, [finding], assembly_source(payload, response))


def test_persistently_wrong_scope_exits_with_failure_instead_of_looping(monkeypatch, tmp_path):
    payload, response, finding = fixture()
    counts = {'repair': 0}
    async def run(agent, **kwargs):
        data = json.loads(kwargs['input'])
        if agent.name == assembly_audit_agent.name:
            current = deepcopy(finding)
            current['quote'] = data['draft_narrative_fields']['/answer_summary']
            result = {'findings': [current]}
        else:
            counts['repair'] += 1
            result = {'replacements': {'0': f'Incorrect scope version {counts["repair"]}.'}}
        return SimpleNamespace(final_output=json.dumps(result))
    monkeypatch.setattr(p.Runner, 'run', run)
    with pytest.raises(RuntimeError, match='scope_repair_exhausted'):
        asyncio.run(review_assembled_response(payload, response, artifact_dir=tmp_path))
    assert counts['repair'] == 3
    assert list(tmp_path.rglob('failure.json'))
