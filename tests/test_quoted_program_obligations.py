from copy import deepcopy
import asyncio
import json
from types import SimpleNamespace

import pytest

from aso_agents.quoted_program_obligations import apply_program_obligations


REQUEST = ('Use TX_A.2 and TX_B.7. Every ASO must bind both mature transcripts exactly. '
           'Each target must cover the entire annotated start codon in both transcripts. '
           'Design two 18-nt ASOs and four 24-nt ASOs.')


def goal():
    return {'intent': 'binding', 'transcript_lock': {'value': 'TX_A', 'version': 2},
            'candidate_minimum': 6, 'candidate_limit': 6,
            'design_programs': [
                {'program_id': 'short', 'modality': 'steric', 'intent': 'binding', 'length': 18},
                {'program_id': 'long', 'modality': 'steric', 'intent': 'binding', 'length': 24}]}


def audit():
    return {
        'shared_binding_cohort': {'applies_to_all_candidates': True, 'reference_molecule': 'mature_rna',
            'transcript_ids': ['TX_A.2', 'TX_B.7'], 'references_quote': 'TX_A.2 and TX_B.7',
            'quote': 'Every ASO must bind both mature transcripts exactly'},
        'feature_requirements': [{'feature_name': 'start_codon', 'aso_overlap_requirement': 'contains_feature',
            'minimum_overlap_bp': 1, 'applies_to_all_candidates': True, 'reference_scope': 'shared_binding_cohort',
            'quote': 'Each target must cover the entire annotated start codon in both transcripts'}],
        'program_candidate_counts': [
            {'length_nt': 18, 'count': 2, 'kind': 'exact', 'quote': 'two 18-nt ASOs'},
            {'length_nt': 24, 'count': 4, 'kind': 'exact', 'quote': 'four 24-nt ASOs'}]}


def test_typed_cohort_features_and_separate_counts_preserve_anchor_and_total():
    g = goal()
    original = deepcopy(g)
    receipt = apply_program_obligations(g, audit(), REQUEST)
    assert receipt['status'] == 'applied'
    assert g['transcript_lock'] == original['transcript_lock']
    assert (g['candidate_minimum'], g['candidate_limit']) == (6, 6)
    for program, count in zip(g['design_programs'], (2, 4)):
        assert program['transcript_ids'] == ['TX_A.2', 'TX_B.7']
        assert program['coordinate_space'] == 'cdna'
        assert (program['candidate_minimum'], program['candidate_limit']) == (count, count)
        rules = program['region_constraints']
        assert [r['reference_transcript_id'] for r in rules] == ['TX_A.2', 'TX_B.7']
        assert all(r['operator'] == 'intersect' and r['aso_overlap_requirement'] == 'contains_feature' for r in rules)
        assert all(r['source_type'] == 'transcript_feature' and 'rows' not in r for r in rules)


@pytest.mark.parametrize('change', [
    lambda a: a['shared_binding_cohort'].update(quote='An invented shared binding instruction'),
    lambda a: a['shared_binding_cohort'].update(transcript_ids=['TX_A.2', 'TX_B.70']),
    lambda a: a['shared_binding_cohort'].update(transcript_ids=['TX_A', 'TX_B.7']),
    lambda a: a['shared_binding_cohort'].update(applies_to_all_candidates=False),
    lambda a: a['shared_binding_cohort'].update(reference_molecule='precursor_rna'),
    lambda a: a['feature_requirements'][0].update(reference_scope='unverified'),
    lambda a: a['feature_requirements'][0].update(aso_overlap_requirement='cover_all'),
    lambda a: a['program_candidate_counts'][0].update(length_nt=17),
    lambda a: a['program_candidate_counts'][0].update(count=True),
    lambda a: a['program_candidate_counts'][0].update(quote='unquoted count'),
])
def test_unsupported_positive_evidence_is_atomic_and_unresolved(change):
    g, a = goal(), audit()
    before = deepcopy(g)
    change(a)
    receipt = apply_program_obligations(g, a, REQUEST)
    assert receipt['status'] == 'unresolved'
    assert not receipt['applied']
    assert g == before


def test_length_count_needs_unique_program_and_is_never_multiplied():
    g = goal()
    g['design_programs'].append({**g['design_programs'][0], 'program_id': 'another_site'})
    before = deepcopy(g)
    result = apply_program_obligations(g, audit(), REQUEST)
    assert result['status'] == 'unresolved'
    assert any(x['reason'] == 'selector_not_unique' for x in result['unresolved'])
    assert g == before


def test_modality_selector_can_disambiguate_equal_lengths():
    g, a = goal(), audit()
    g['design_programs'].append({'program_id': 'other', 'modality': 'gapmer', 'length': 18})
    a = {'program_candidate_counts': [{**a['program_candidate_counts'][0], 'modality': 'steric'}]}
    assert apply_program_obligations(g, a, REQUEST)['status'] == 'applied'
    assert g['design_programs'][0]['candidate_limit'] == 2
    assert 'candidate_limit' not in g['design_programs'][-1]


def test_existing_interval_and_boolean_order_survive_idempotent_feature_reconciliation():
    g = goal()
    existing = {'source_type': 'explicit_rows', 'coordinate_space': 'cdna',
                'rows': [['TX_A.2', 30, 90]], 'operator': 'union', 'aso_overlap_requirement': 'full_containment'}
    g['design_programs'][0]['region_constraints'] = [existing]
    apply_program_obligations(g, audit(), REQUEST)
    first = deepcopy(g)
    apply_program_obligations(g, audit(), REQUEST)
    assert g == first
    assert g['design_programs'][0]['region_constraints'][0] == existing


@pytest.mark.parametrize('kind,expected', [('minimum', (2, None)), ('maximum', (1, 2))])
def test_scoped_count_ranges_keep_correct_open_bound(kind, expected):
    g = goal()
    g['design_programs'][0].update(candidate_minimum=1, candidate_limit=None)
    a = {'program_candidate_counts': [{**audit()['program_candidate_counts'][0], 'kind': kind}]}
    assert apply_program_obligations(g, a, REQUEST)['status'] == 'applied'
    p = g['design_programs'][0]
    assert (p.get('candidate_minimum'), p.get('candidate_limit')) == expected


def test_conflicting_counts_do_not_partially_apply():
    g, a = goal(), audit()
    before = deepcopy(g)
    a['program_candidate_counts'].append({**a['program_candidate_counts'][0], 'count': 3})
    assert apply_program_obligations(g, a, REQUEST)['status'] == 'unresolved'
    assert g == before


def test_primary_feature_does_not_invent_a_reference_or_shared_cohort():
    g = goal()
    item = {**audit()['feature_requirements'][0], 'reference_scope': 'primary_reference'}
    assert apply_program_obligations(g, {'feature_requirements': [item]}, REQUEST)['status'] == 'applied'
    for p in g['design_programs']:
        assert 'transcript_ids' not in p
        assert 'reference_transcript_id' not in p['region_constraints'][0]


def test_explicit_feature_reference_requires_exact_quoted_identity():
    g = goal()
    item = {**audit()['feature_requirements'][0], 'reference_scope': 'explicit_references',
            'transcript_ids': ['TX_B.7'], 'references_quote': 'TX_A.2 and TX_B.7'}
    assert apply_program_obligations(g, {'feature_requirements': [item]}, REQUEST)['status'] == 'applied'
    assert g['design_programs'][0]['region_constraints'][0]['reference_transcript_id'] == 'TX_B.7'


def test_splice_program_is_not_silently_reinterpreted_as_mature_binding():
    g = goal()
    g['design_programs'][1]['intent'] = 'splice_modulation'
    before = deepcopy(g)
    assert apply_program_obligations(g, audit(), REQUEST)['status'] == 'unresolved'
    assert g == before


def test_old_audit_without_new_obligations_leaves_goal_unchanged():
    g = goal()
    before = deepcopy(g)
    assert apply_program_obligations(g, {}, REQUEST)['status'] == 'not_requested'
    assert g == before


def test_native_audit_synchronizes_programs_into_single_task_envelope(monkeypatch):
    import run_goal_intake as intake
    g = goal()
    draft = {'status': 'ready', 'structured_goal': g, 'request_envelope': {
        'tasks': [{'task_type': 'design_new_asos', 'context': {'structured_goal': deepcopy(g)}}]}}
    async def respond(*args, **kwargs):
        return SimpleNamespace(final_output=json.dumps(audit()))
    monkeypatch.setattr(intake.Runner, 'run', respond)
    result = asyncio.run(intake._audit_request_obligations(draft, REQUEST, []))
    assert result['status'] == 'ready'
    nested = result['request_envelope']['tasks'][0]['context']['structured_goal']
    assert nested['design_programs'] == result['structured_goal']['design_programs']
    assert [p['candidate_limit'] for p in nested['design_programs']] == [2, 4]
    assert result['request_obligation_audit']['program_obligations']['status'] == 'applied'


def test_native_audit_does_not_call_an_ambiguous_obligation_accepted(monkeypatch):
    import run_goal_intake as intake
    g = goal()
    g['design_programs'].append({**g['design_programs'][0], 'program_id': 'other_site'})
    async def respond(*args, **kwargs):
        return SimpleNamespace(final_output=json.dumps(audit()))
    monkeypatch.setattr(intake.Runner, 'run', respond)
    result = asyncio.run(intake._audit_request_obligations({'status': 'ready', 'structured_goal': g}, REQUEST, []))
    assert result['status'] == 'need_clarification'
    assert result['request_obligation_audit']['status'] == 'unresolved'
    assert result['questions']
