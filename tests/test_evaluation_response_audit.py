import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools
from aso_agents.evaluation_evidence import attach_measurement_references
from aso_agents.evaluation_response_audit import (
    narrative_fields, pointer_value, response_audit_contract_errors,
    response_audit_repair_findings, response_audit_receipt,
    merge_bound_narrative_repair, apply_bound_narrative_patch, narrative_repair_schema,
)
from aso_agents.response_context import decode_response_context
from aso_agents.workbench_response_agent import (evaluation_evidence_audit_agent,
    evaluation_evidence_repair_agent, chemical_position_audit_agent)


def source_and_response():
    row = {'input_row_id': 'observation_x', 'name': 'probe', 'sequence': 'AGCT',
           'mapping_hits': [], 'near_match_candidates': [
               {'transcript_id': 'KEEP.2', 'rna_context': 'mature_spliced_cdna',
                'match_orientation': 'reverse_complement', 'mismatch_count': 2}]}
    source = {'task_type': 'evaluate_existing_aso_set',
              'evaluation_summary': attach_measurement_references({'provided_asos': [row]})}
    response = {'answer_summary': 'Activity remains unknown.', 'markdown': '',
                'evaluation_assessments': [{'input_row_id': 'observation_x', 'name': 'probe',
                    'mechanistic_plausibility': 'Conditional binding is possible.',
                    'selectivity_limitations': 'All retained matches are same-strand similarities.',
                    'uncertainty': 'Occupancy and activity are unmeasured.',
                    'ranking_rationale': 'No measured potency ranking.'}]}
    return source, response


def finding():
    return {'response_path': '/evaluation_assessments/0/selectivity_limitations',
            'quote': 'All retained matches are same-strand similarities.',
            'kind': 'contradicted_fact',
            'evidence_paths': ['/evaluation_summary/provided_asos/0/near_match_candidates/0/match_orientation'],
            'reason': 'The retained KEEP alignment is reverse_complement; weak binding does not change orientation.'}


def decode(text):
    value = json.loads(text)
    return decode_response_context(value) if value.get('context_encoding') else value



def wire_answer(agent, answer):
    if agent.name == evaluation_evidence_repair_agent.name and 'evaluation_assessments' in answer:
        return {'corrections': {'/evaluation_assessments/0/selectivity_limitations':
            answer['evaluation_assessments'][0]['selectivity_limitations']}}
    return answer


def test_inventory_groups_partition_all_retained_hypotheses_without_strength_filtering():
    source, _ = source_and_response()
    row = source['evaluation_summary']['provided_asos'][0]
    row['near_match_candidates'] += [
        {'transcript_id': 'KEEP.2', 'rna_context': 'mature_spliced_cdna',
         'match_orientation': 'reverse_complement', 'mismatch_count': 12},
        {'transcript_id': 'KEEP.2', 'rna_context': 'mature_spliced_cdna',
         'match_orientation': 'same_strand', 'mismatch_count': 0},
        {'transcript_id': 'OTHER.4', 'rna_context': 'unspliced_transcript_locus_pre_mrna',
         'match_orientation': 'reverse_complement', 'mismatch_count': 1}]
    attach_measurement_references(source['evaluation_summary'])
    groups = row['evidence_inventory']['retained_hypothesis_groups']
    assert sorted(group['retained_count'] for group in groups) == [1, 1, 2]
    assert sorted(field for group in groups for field in group['source_fields']) == [
        'near_match_candidates[0]', 'near_match_candidates[1]', 'near_match_candidates[2]', 'near_match_candidates[3]']


@pytest.mark.parametrize('pointer', ['/missing', '/evaluation_summary/provided_asos/-1',
                                      '/evaluation_summary/provided_asos/00',
                                      '/evaluation_summary/provided_asos/900'])
def test_review_cannot_cite_nonexistent_or_noncanonical_evidence(pointer):
    source, response = source_and_response()
    item = finding(); item['evidence_paths'] = [pointer]
    assert response_audit_contract_errors({'findings': [item]}, response, source)
    with pytest.raises(ValueError):
        response_audit_repair_findings({'findings': [item]}, response, source)


def test_review_is_bound_to_actual_claim_and_exact_evidence_and_cannot_mutate_either():
    source, response = source_and_response()
    before = deepcopy((source, response))
    item = finding()
    bound = response_audit_repair_findings({'findings': [item]}, response, source)
    assert len(next(iter(bound[0]['evidence_sha256'].values()))) == 64
    item['quote'] = 'No such claim exists in this draft.'
    assert response_audit_contract_errors({'findings': [item]}, response, source)
    assert (source, response) == before
    assert pointer_value({'a/b': {'~key': [7]}}, '/a~1b/~0key/0') == 7


def test_empty_finding_does_not_upgrade_uncertainty_or_claim_independent_validation():
    source, response = source_and_response()
    receipt = response_audit_receipt(response, source, attempts=1, model='same_native_model')
    assert set(receipt['reviewed_fields']) == set(narrative_fields(response))
    changed = deepcopy(response); changed['answer_summary'] = 'Changed.'
    assert receipt['draft_sha256'] != response_audit_receipt(changed, source, attempts=1, model='same_native_model')['draft_sha256']
    assert response_audit_repair_findings({'findings': []}, response, source) == []
    assert response['answer_summary'] == 'Activity remains unknown.'


def test_native_composer_repairs_a_bound_factual_error_preserving_other_rows_and_uncertainty(monkeypatch):
    source, original = source_and_response()
    corrected = deepcopy(original)
    corrected['evaluation_assessments'][0]['selectivity_limitations'] = 'A retained antisense hypothesis exists on KEEP; occupancy remains unmeasured.'
    calls = []
    compose_count = 0
    audit_count = 0

    async def run(agent, **kwargs):
        nonlocal compose_count, audit_count
        calls.append((agent.name, kwargs['input']))
        if agent.name == evaluation_evidence_audit_agent.name:
            assert decode(kwargs['input'])['source'] == source
            assert agent.model_settings.extra_body['text']['format']['type'] == 'json_schema'
            audit_count += 1
            answer = {'findings': [finding()] if audit_count == 1 else []}
        else:
            compose_count += 1
            answer = original if compose_count == 1 else corrected
        return SimpleNamespace(final_output=json.dumps(wire_answer(agent, answer)))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))
    assert compose_count == audit_count == 2
    assert result['native_consistency_review']['attempts'] == 2
    assert result['evaluation_assessments'] == corrected['evaluation_assessments']
    assert result['answer_summary'] == original['answer_summary']
    repair_prompt = [prompt for name, prompt in calls if name != evaluation_evidence_audit_agent.name][1]
    assert finding()['quote'] in repair_prompt and 'evidence_sha256' in repair_prompt
    assert source == source_and_response()[0]


def test_native_composer_repairs_wrong_supplied_gapmer_region_without_changing_numeric_evidence(monkeypatch):
    source, draft = source_and_response()
    row = source['evaluation_summary']['provided_asos'][0]
    row['sequence'] = 'A' * 20
    row['chemistry'] = 'full-PS 5-10-5 2-MOE/DNA gapmer'
    row['near_match_candidates'] = [{
        'transcript_id': 'KEEP.2', 'match_orientation': 'reverse_complement',
        'mismatch_count': 1, 'mismatch_positions_in_supplied_sequence_1based': [9]}]
    attach_measurement_references(source['evaluation_summary'])
    wrong = 'The recorded mismatch at supplied-ASO position 9 lies in a MOE wing.'
    draft['evaluation_assessments'][0]['mechanistic_plausibility'] = wrong
    focused_calls = []
    repair_calls = []

    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            answer = {'findings': []}
        elif agent.name == chemical_position_audit_agent.name:
            packet = json.loads(kwargs['input'])
            focused_calls.append(packet)
            answer = {'findings': []}
        elif agent.name == evaluation_evidence_repair_agent.name:
            repair_calls.append(kwargs['input'])
            raise AssertionError('Typed chemical-position correction must not need a model repair')
        else:
            answer = draft
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))
    assert '9 (DNA gap)' in result['evaluation_assessments'][0]['mechanistic_plausibility']
    assert 'MOE wing' not in result['evaluation_assessments'][0]['mechanistic_plausibility']
    assert repair_calls == []
    assert source['evaluation_summary']['provided_asos'][0]['near_match_candidates'][0][
        'mismatch_positions_in_supplied_sequence_1based'] == [9]
    assert result['native_consistency_review']['chemical_position_audit_packets'] == 1
    assert len(focused_calls) == 1
    assert focused_calls[0]['position_evidence'][-1]['recorded_value'][
        'recorded_near_match_mismatch_positions'] == [{'position_1based': 9, 'region': 'dna_gap'}]


def test_false_focused_model_finding_cannot_rewrite_a_correct_region_claim(monkeypatch):
    source, draft = source_and_response()
    row = source['evaluation_summary']['provided_asos'][0]
    row['sequence'] = 'A' * 20
    row['chemistry'] = '5-10-5'
    row['near_match_candidates'] = [{
        'mismatch_count': 1, 'mismatch_positions_in_supplied_sequence_1based': [9]}]
    attach_measurement_references(source['evaluation_summary'])
    correct = 'The recorded mismatch at supplied-ASO position 9 lies in the DNA gap.'
    draft['evaluation_assessments'][0]['mechanistic_plausibility'] = correct
    calls = []

    async def run(agent, **kwargs):
        calls.append(agent.name)
        if agent.name == evaluation_evidence_audit_agent.name:
            answer = {'findings': []}
        elif agent.name == chemical_position_audit_agent.name:
            answer = {'findings': [{
                'response_path': '/evaluation_assessments/0/mechanistic_plausibility',
                'quote': correct, 'kind': 'contradicted_fact',
                'evidence_paths': ['/evaluation_summary/provided_asos/0/chemical_position_evidence'],
                'reason': 'The claim is correct, but the model emitted a finding.'}]}
        else:
            answer = draft
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))
    assert result['evaluation_assessments'][0]['mechanistic_plausibility'] == correct
    assert result['native_consistency_review']['chemical_position_findings_rejected_as_unverified'] == 1
    assert evaluation_evidence_repair_agent.name not in calls


def test_unresolved_native_consistency_errors_do_not_deliver_an_asserted_valid_response(monkeypatch):
    source, original = source_and_response()
    async def run(agent, **kwargs):
        answer = {'findings': [finding()]} if agent.name == evaluation_evidence_audit_agent.name else original
        return SimpleNamespace(final_output=json.dumps(wire_answer(agent, answer)))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    with pytest.raises(RuntimeError, match='workbench_response_contract_failed|evaluation_evidence_consistency_review_failed'):
        asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))


def test_malformed_native_review_is_not_interpreted_as_no_findings(monkeypatch):
    source, response = source_and_response()
    async def run(agent, **kwargs):
        answer = {} if agent.name == evaluation_evidence_audit_agent.name else response
        return SimpleNamespace(final_output=json.dumps(wire_answer(agent, answer)))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    with pytest.raises(RuntimeError, match='evaluation_evidence_consistency_review_failed'):
        asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))


def test_citation_contract_repair_precedes_semantic_review_and_preserves_assessments(monkeypatch):
    source, original = source_and_response()
    original['assumptions'] = ['Citation syntax example {{...}}.']
    repaired = deepcopy(original)
    repaired['assumptions'] = ['Citations refer to the supplied evidence.']
    compose_prompts, audited = [], []

    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            audited.append(decode(kwargs['input'])['draft_response'])
            answer = {'findings': []}
        else:
            compose_prompts.append(kwargs['input'])
            answer = original if len(compose_prompts) == 1 else repaired
        return SimpleNamespace(final_output=json.dumps(wire_answer(agent, answer)))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))
    assert len(compose_prompts) == 2 and audited == [repaired]
    assert 'Unknown evaluation measurement reference' in compose_prompts[1]
    assert result['evaluation_assessments'] == original['evaluation_assessments']
    assert result['native_consistency_review']['attempts'] == 1


def test_factual_repair_cannot_rewrite_unflagged_claims_or_observation_identity():
    source, original = source_and_response()
    candidate = deepcopy(original)
    candidate['answer_summary'] = 'Invented experimental success.'
    candidate['evaluation_assessments'][0]['uncertainty'] = 'All uncertainty removed.'
    candidate['evaluation_assessments'][0]['selectivity_limitations'] = 'The retained record is an antisense hypothesis.'
    before = deepcopy((source, original, candidate))
    merged = merge_bound_narrative_repair(original, candidate, [finding()], source)
    assert merged['answer_summary'] == original['answer_summary']
    assert merged['evaluation_assessments'][0]['uncertainty'] == original['evaluation_assessments'][0]['uncertainty']
    assert merged['evaluation_assessments'][0]['selectivity_limitations'] == candidate['evaluation_assessments'][0]['selectivity_limitations']
    assert (source, original, candidate) == before
    candidate['evaluation_assessments'][0]['input_row_id'] = 'different_observation'
    with pytest.raises(ValueError, match='identity'):
        merge_bound_narrative_repair(original, candidate, [finding()], source)


def test_factual_repair_requires_quoted_evidence_and_a_string_at_the_flagged_path():
    source, original = source_and_response()
    candidate = deepcopy(original)
    candidate['evaluation_assessments'][0]['selectivity_limitations'] = ['Changed field type']
    with pytest.raises(ValueError, match='types'):
        merge_bound_narrative_repair(original, candidate, [finding()], source)
    unknown = finding(); unknown['quote'] = 'Not present in the reviewed draft'
    with pytest.raises(ValueError, match='quote'):
        merge_bound_narrative_repair(original, candidate, [unknown], source)
    assert merge_bound_narrative_repair(original, candidate, [], source) == original


def test_malformed_correction_keeps_the_previous_repair_scope(monkeypatch):
    source, original = source_and_response()
    corrected = deepcopy(original)
    corrected['evaluation_assessments'][0]['selectivity_limitations'] = 'A retained antisense hypothesis exists; activity is unknown.'
    corrected['answer_summary'] = 'Unrelated unwanted rewrite.'
    changed_identity = {'corrections': {'/evaluation_assessments/0/input_row_id': 'wrong_row'}}
    drafts = iter([original, changed_identity, corrected])
    audits = []

    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            audits.append(decode(kwargs['input'])['draft_response'])
            answer = {'findings': [finding()] if len(audits) == 1 else []}
        else:
            answer = next(drafts)
        return SimpleNamespace(final_output=json.dumps(wire_answer(agent, answer)))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))
    assert len(audits) == 2
    assert result['answer_summary'] == original['answer_summary']
    assert result['evaluation_assessments'][0]['input_row_id'] == 'observation_x'
    assert result['evaluation_assessments'][0]['selectivity_limitations'] == corrected['evaluation_assessments'][0]['selectivity_limitations']


def test_native_scope_ambiguity_repair_preserves_evidence_and_precise_uncertainty(monkeypatch):
    source, original = source_and_response()
    row = source['evaluation_summary']['provided_asos'][0]
    row['near_match_candidates'][0]['reference_junctions_spanned'] = [{
        'transcript_id': 'KEEP.2', 'chrom': '8', 'strand': '+', 'donor': 130, 'acceptor': 301}]
    row['requested_feature_membership'] = [{'identity_resolution': {'status': 'unresolved'}, 'observations': []}]
    attach_measurement_references(source['evaluation_summary'])
    quote = 'No direct geometric evidence for the requested feature is present.'
    original['evaluation_assessments'][0]['selectivity_limitations'] = quote
    corrected = deepcopy(original)
    corrected['evaluation_assessments'][0]['selectivity_limitations'] = 'Reference-junction geometry is recorded. Correspondence to the requested feature name and biological selectivity remain unresolved.'
    item = {'response_path': '/evaluation_assessments/0/selectivity_limitations', 'quote': quote,
            'kind': 'ambiguous_evidence_scope',
            'evidence_paths': ['/evaluation_summary/provided_asos/0/evidence_inventory/feature_evidence_availability'],
            'reason': 'The statement could deny recorded reference geometry or describe unverified requested-feature naming. The source records geometry but lacks a requested-site register, so these readings have different factual meanings.'}
    before = deepcopy(source)
    counts = {'audit': 0, 'composer': 0}

    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            counts['audit'] += 1
            context = decode(kwargs['input'])
            assert context['source'] == source
            assert any(entry['source_pointer'] == item['evidence_paths'][0] for entry in context['evidence_scope_index'])
            answer = {'findings': [item] if counts['audit'] == 1 else []}
        else:
            counts['composer'] += 1
            answer = original if counts['composer'] == 1 else corrected
        return SimpleNamespace(final_output=json.dumps(wire_answer(agent, answer)))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))
    assert counts == {'audit': 2, 'composer': 2}
    assert result['native_consistency_review']['attempts'] == 2
    assert result['evaluation_assessments'] == corrected['evaluation_assessments']
    assert result['answer_summary'] == original['answer_summary']
    assert source == before


def test_rejected_scoped_correction_and_active_finding_remain_visible(monkeypatch):
    source, original = source_and_response()
    invalid = deepcopy(original)
    invalid['evaluation_assessments'][0]['selectivity_limitations'] = 'A retained antisense hypothesis exists: {{UNKNOWN}}; activity is unknown.'
    invalid['answer_summary'] = 'Unrelated rewrite must not be accepted.'
    corrected = deepcopy(original)
    corrected['evaluation_assessments'][0]['selectivity_limitations'] = 'A retained antisense hypothesis exists; activity is unknown.'
    drafts = iter([original, invalid, corrected])
    prompts, audits = [], []

    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            audits.append(decode(kwargs['input'])['draft_response'])
            answer = {'findings': [finding()] if len(audits) == 1 else []}
        else:
            prompts.append(kwargs['input'])
            if len(prompts) == 3:
                data = decode(kwargs['input'])
                rejected = data['rejected_repair_candidate']
                assert rejected['corrections']['/evaluation_assessments/0/selectivity_limitations'] == invalid['evaluation_assessments'][0]['selectivity_limitations']
                assert data['preceding_response'] == original
                assert data['evidence_consistency_findings'][0]['quote'] == finding()['quote']
                assert data['allowed_narrative_repair_paths'] == [finding()['response_path']]
                assert 'Unknown evaluation measurement reference' in data['contract_errors'][0]
            answer = next(drafts)
        return SimpleNamespace(final_output=json.dumps(wire_answer(agent, answer)))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))
    assert len(prompts) == 3 and len(audits) == 2
    assert result['answer_summary'] == original['answer_summary']
    assert result['evaluation_assessments'] == corrected['evaluation_assessments']
    assert source == source_and_response()[0]


def test_numeric_search_parameter_still_receives_semantic_review(monkeypatch):
    source, original = source_and_response()
    source['evaluation_summary']['search_configuration'] = {'max_substitutions': 0, 'status': 'complete'}
    original['evaluation_assessments'][0]['selectivity_limitations'] = 'The search ran at 2 mismatches.'
    item = {**finding(), 'quote': 'The search ran at 2 mismatches.',
        'evidence_paths': ['/evaluation_summary/search_configuration'],
        'reason': 'The completed search permits zero substitutions, not two.'}
    corrected = deepcopy(original)
    corrected['evaluation_assessments'][0]['selectivity_limitations'] = 'The search ran at 0 mismatches; activity is unknown.'
    audits, proposals = [], []

    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            audits.append(decode(kwargs['input'])['draft_response'])
            answer = {'findings': [item] if len(audits) == 1 else []}
        else:
            proposals.append(kwargs['input'])
            answer = original if len(proposals) == 1 else corrected
        return SimpleNamespace(final_output=json.dumps(wire_answer(agent, answer)))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_batch_once(source))
    assert len(proposals) == len(audits) == 2
    assert audits[0] == original and audits[1] == corrected
    assert result['evaluation_assessments'] == corrected['evaluation_assessments']
