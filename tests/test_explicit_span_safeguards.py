"""A supplied physical exon must keep its own identity and reciprocal bounds."""
from copy import deepcopy
import pytest
import run_goal_intake as runtime
from aso_agents.splice_recommendations import (
    resolve_routing_exon_objectives, preserve_splice_site_evidence, objective_prediction_evidence,
)


def explicit_goal(strand='+', effect='skipping'):
    intervals = [(100, 140), (400, 440), (700, 740)]
    if strand == '-':
        intervals.reverse()
    base = {'transcript_id': 'TX_BASE', 'chrom': 'chr2', 'strand': strand,
        'metadata': {'id': 'TX_BASE', 'versioned_id': 'TX_BASE.3', 'version': 3},
        'exons_tx': [{'index': i, 'start': a, 'end': b, 'seq_region_name': 'chr2'}
                     for i, (a, b) in enumerate(intervals, 1)]}
    direction = 'decrease' if effect == 'skipping' else 'increase'
    program = {'program_id': 'custom_event', 'modality': 'steric', 'transcript_id': 'TX_BASE.3',
        'exon_label': 'cassette-x', 'desired_effect': effect}
    target = {'chrom': 'chr2', 'start': 250, 'end': 280, 'strand': strand,
        'isoform_label': 'base', 'exon_label': 'cassette-x', 'requested_exon_label': 'cassette-x',
        'desired_delta': direction}
    goal = {'isoform_transcripts': {'base': base}, 'explicit_splice_targets': [target],
        'design_programs': [program], 'routing_job': dict(program),
        'splice_isoform_objectives': [{'isoform_label': 'base', 'transcript_id': 'TX_BASE',
                                      'desired_abundance': 'increase'}]}
    runtime._ensure_explicit_splice_target_context(goal, {})
    explicit = goal['explicit_splice_targets'][0]
    goal['splice_site_plan'] = [{'site': kind, 'position': position, 'chrom': 'chr2', 'strand': strand,
        'delta_direction': direction, 'present_in': [explicit['isoform_label']], 'absent_in': ['base'],
        'exon_indices': {explicit['isoform_label']: explicit['exon_label']}}
        for kind, position in [('acceptor', 250 if strand == '+' else 280),
                               ('donor', 280 if strand == '+' else 250)]]
    return goal


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('effect', ['skipping', 'inclusion'])
def test_explicit_span_binds_paired_predictions_without_annotation_identity_theft(strand, effect):
    goal = explicit_goal(strand, effect)
    before = deepcopy(goal)
    objective, = resolve_routing_exon_objectives(goal)
    assert goal == before
    assert objective['status'] == 'resolved'
    identity = objective['identity']
    assert identity['status'] == 'verified_explicit_span'
    assert identity['biological_label_verified'] is False
    assert identity['base_reference_transcript_id'] == 'TX_BASE.3'
    assert objective['reference_transcript_id'] == 'TX_BASE::explicit_target_model'
    sites = runtime._build_splice_sites_for_eval(goal, None, 10)
    for site in sites:
        site.update(evaluation_status='evaluated', delta_inclusion_deletion=-.2 if effect == 'skipping' else .2,
                    delta_inclusion_replacement=-.1 if effect == 'skipping' else .1)
    preserved = preserve_splice_site_evidence({'splice_site_deltas': sites}, goal['isoform_transcripts'])
    evidence = objective_prediction_evidence({'splice_site_deltas': preserved}, objective)
    assert evidence['status'] == 'recorded_pair_predictions'
    assert all(m['prediction']['state'] == 'direction_supported' for m in evidence['measurements'])


@pytest.mark.parametrize('effect,bound', [('skipping', 'max_increase'), ('inclusion', 'max_decrease')])
@pytest.mark.parametrize('measured', [True, False])
def test_explicit_reciprocal_guard_survives_absence_from_annotated_isoform(effect, bound, measured, monkeypatch):
    goal = explicit_goal(effect=effect)
    monkeypatch.setattr(runtime, 'resolve_ambiguous_safeguards', lambda *a, **k: {})
    mapping = runtime._build_safeguard_candidates_by_target(goal['splice_site_plan'], [], goal,
        measured_site_labels={'acceptor@250', 'donor@280'} if measured else {'acceptor@250'})
    guards = runtime._default_safeguards_from_candidates(mapping['acceptor@250'])
    reciprocal = next(g for g in guards if g['label'] == 'donor@280')
    assert reciprocal[bound] == .05 and not reciprocal.get('required')


@pytest.mark.parametrize('fault', ['version', 'chromosome', 'strand', 'span', 'label', 'direction',
                                  'base_version', 'model_exon', 'explicit_ordinal', 'ambiguous_span'])
def test_conflicting_or_ambiguous_explicit_span_does_not_bind(fault):
    goal = explicit_goal()
    job, raw = goal['routing_job'], goal['explicit_splice_targets'][0]
    if fault == 'version': job['transcript_id'] = 'TX_BASE.4'
    elif fault == 'chromosome': raw['chrom'] = 'chr9'
    elif fault == 'strand': raw['strand'] = '-'
    elif fault == 'span': raw['end'] += 1
    elif fault == 'label': job['exon_label'] = 'other-label'
    elif fault == 'direction': job['desired_effect'] = 'inclusion'
    elif fault == 'base_version':
        goal['isoform_transcripts']['base::explicit_target_model']['metadata']['base_reference_transcript_id'] = 'TX_BASE.4'
    elif fault == 'model_exon':
        goal['isoform_transcripts']['base::explicit_target_model']['exons_tx'][1]['end'] += 1
    elif fault == 'explicit_ordinal': job['exon_index'] = 1
    else:
        duplicate = deepcopy(raw); duplicate['start'] += 1
        goal['explicit_splice_targets'].append(duplicate)
    objective, = resolve_routing_exon_objectives(goal)
    assert objective['status'] == 'unresolved'


def test_exact_annotated_physical_exon_keeps_annotated_reference():
    goal = explicit_goal()
    raw = goal['explicit_splice_targets'][0]
    raw.update(start=400, end=440, isoform_label='base', base_isoform_label='base', exon_label=2,
               mapped_start=400, mapped_end=440, synthetic_exon_created=False)
    goal['isoform_transcripts'].pop('base::explicit_target_model')
    objective, = resolve_routing_exon_objectives(goal)
    assert objective['status'] == 'resolved'
    assert objective['reference_transcript_id'] == 'TX_BASE.3'


def test_explicit_guard_policy_is_not_replaced_by_derived_reciprocal_bound(monkeypatch):
    goal = explicit_goal()
    monkeypatch.setattr(runtime, 'resolve_ambiguous_safeguards', lambda *a, **k: {})
    mapping = runtime._build_safeguard_candidates_by_target(goal['splice_site_plan'],
        [{'label': 'donor@280', 'max_increase': .12, 'required': True}], goal)
    guard = next(g for g in runtime._default_safeguards_from_candidates(mapping['acceptor@250'])
                 if g['label'] == 'donor@280')
    assert guard['max_increase'] == .12 and guard['required'] is True
