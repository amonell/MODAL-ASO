"""Intron boundary ordinals must not create independent exon isoform objectives."""
from copy import deepcopy
import pytest
import run_goal_intake as native
from aso_agents.intron_events import program_intron_sites


def intron_goal(kind):
    program = {'program_id': 'intron', 'modality': 'steric', 'intent': 'splice_modulation',
               'transcript_id': 'ENST00000000001.2', 'exon_labels': [1, 2],
               'splice_event': {'event_type': kind, 'intron_number': 1,
                                'reference_transcript_id': 'ENST00000000001.2'}}
    return {'gene_symbol': 'NEUTRAL', 'intent': 'splice_modulation',
            'exon_label': 1, 'exon_labels': [1, 2], 'focus_exons_only': True,
            'design_programs': [program]}


@pytest.mark.parametrize('kind,direction', [('intron_retention', 'decrease'), ('intron_excision', 'increase')])
def test_intron_flank_numbers_do_not_synthesize_exon_isoforms(monkeypatch, kind, direction):
    monkeypatch.setattr(native, 'resolve_transcript_exon_nomenclature_sync',
                        lambda **kwargs: {'status': 'unresolved', 'source': 'neutral_test_annotation'})
    goal = intron_goal(kind)
    catalog = {'ref': {'transcript_id': 'ENST00000000001.2', 'chrom': 'chr7', 'strand': '+',
                      'exons_tx': [{'index': 1, 'start': 101, 'end': 150},
                                   {'index': 2, 'start': 301, 'end': 350},
                                   {'index': 3, 'start': 501, 'end': 550}]}}
    assert native._collect_splice_target_exon_effects(goal) == []
    native._supplement_splice_context_from_exon_targets(goal)
    native._seed_default_isoform_objectives(goal)
    native._seed_default_splice_context_direction(goal, 'ref')
    assert not goal.get('splice_isoform_objectives')
    assert not goal.get('site_direction_hints')
    plan, _ = native._compute_splice_site_plan([], catalog, goal)
    assert not goal.get('splice_isoform_objectives')
    sites = program_intron_sites(goal['design_programs'][0], catalog, plan)
    assert {(site['site'], site['position'], site['delta_direction']) for site in sites} == {
        ('donor', 150, direction), ('acceptor', 301, direction)}


def test_independent_exon_program_survives_beside_intron_flanks():
    goal = intron_goal('intron_retention')
    goal['design_programs'].append({'program_id': 'exon', 'modality': 'steric',
                                   'intent': 'splice_modulation', 'exon_label': 3,
                                   'desired_effect': 'skipping'})
    assert native._collect_splice_target_exon_effects(goal) == [(3, 'skipping')]
    native._supplement_splice_context_from_exon_targets(goal)
    assert len(goal['splice_isoform_objectives']) == 2
    assert all('exon 3' in row['isoform_label'] for row in goal['splice_isoform_objectives'])


def test_explicit_isoform_objective_is_preserved_beside_an_intron_event():
    goal = intron_goal('intron_retention')
    objective = {'isoform_label': 'requested transcript', 'transcript_id': 'ENST00000000002.1',
                 'objective_scope': 'specific_transcript', 'desired_abundance': 'increase'}
    goal['splice_isoform_objectives'] = [deepcopy(objective)]
    native._supplement_splice_context_from_exon_targets(goal)
    assert goal['splice_isoform_objectives'] == [objective]
