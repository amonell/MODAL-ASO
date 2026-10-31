"""Isoform nomenclature must not discard unsearched selective junctions."""
from copy import deepcopy

import pytest
import run_goal_intake as native


def goal_with_isoform_outcome():
    entry = {'transcript_id': 'ENST00000000001', 'chrom': 'chr1', 'strand': '+',
             'exons_tx': [{'index': 1, 'start': 100, 'end': 149},
                          {'index': 2, 'start': 300, 'end': 349}],
             'metadata': {'reference_accession_resolution': {
                 'requested_accession': 'NM_123456.7', 'exact_accession_verified': True}}}
    program = {'program_id': 'selective', 'modality': 'gapmer', 'intent': 'knockdown',
               'transcript_id': 'NM_123456.7', 'exon_label': 2, 'focus_exons_only': True}
    return {'isoform_transcripts': {'target': entry}, 'design_programs': [program],
            'evaluation_targets': [{'target_type': 'isoform', 'objective_type': 'primary_outcome',
                                    'transcript_id': 'ENST00000000001', 'desired_delta': 'decrease'}]}


def regions(goal):
    return native._build_gapmer_program_region_rows(
        goal, goal['isoform_transcripts'], 'target', goal['design_programs'][0])


def test_isoform_outcome_exon_annotation_keeps_complete_mature_rna_search():
    goal = goal_with_isoform_outcome()
    before = deepcopy(goal)
    rows, meta = regions(goal)
    assert rows[0][1:3] == [0, 100]
    assert meta['sources'] == ['full_transcript']
    assert meta['exon_scope']['role'] == 'isoform_outcome_context'
    assert goal == before


@pytest.mark.parametrize('target_type', ['isoform', 'transcript', 'isoform_cohort', 'transcript_cohort'])
def test_program_outcome_identity_is_not_an_exon_binding_constraint(target_type):
    goal = goal_with_isoform_outcome()
    goal.pop('evaluation_targets')
    goal['design_programs'][0]['target_type'] = target_type
    assert regions(goal)[0][0][1:3] == [0, 100]


@pytest.mark.parametrize('policy,expected', [('default_exploratory', [0, 100]),
                                             ('explicit_user_region', [50, 100])])
def test_explicit_program_binding_policy_has_priority(policy, expected):
    goal = goal_with_isoform_outcome()
    goal['design_programs'][0]['targeting_policy'] = policy
    assert regions(goal)[0][0][1:3] == expected


def test_unrelated_isoform_outcome_does_not_change_legacy_exon_binding_scope():
    goal = goal_with_isoform_outcome()
    goal['evaluation_targets'][0]['transcript_id'] = 'NM_123456.8'
    assert regions(goal)[0][0][1:3] == [50, 100]


def test_outcome_context_does_not_discard_explicit_feature_scope(monkeypatch):
    goal = goal_with_isoform_outcome()
    goal['design_programs'][0]['target_region'] = 'utr3'
    monkeypatch.setattr(native, '_build_program_target_region_rows', lambda *a, **k:
                        ([['ENST00000000001', 80, 100, 'utr3', 0, '+']], {'sources': ['program_target_region']}))
    rows, meta = regions(goal)
    assert rows[0][1:3] == [80, 100]
    assert meta['sources'] == ['program_target_region']


def test_isoform_junction_description_does_not_select_neighboring_ordinals():
    goal = goal_with_isoform_outcome()
    goal['design_programs'][0]['label'] = 'Junction-bearing isoform knockdown'
    assert regions(goal)[0][0][1:3] == [0, 100]
