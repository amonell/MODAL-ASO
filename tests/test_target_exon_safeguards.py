"""The targeted exon can move in the requested direction at both boundaries."""
from copy import deepcopy
import importlib
import json

import pytest
import run_goal_intake as runtime


def exon_goal(strand='+', effect='inclusion', *, missing_catalog_reference=False):
    intervals = [(100, 140), (400, 440), (700, 740)]
    if strand == '-':
        intervals.reverse()
    transcript = {'transcript_id': 'TARGET.1', 'chrom': 'chr2', 'strand': strand,
        'exons_tx': [{'index': i, 'start': a, 'end': b}
                     for i, (a, b) in enumerate(intervals, 1)]}
    other = {**deepcopy(transcript), 'transcript_id': 'OTHER.1'}
    direction = 'increase' if effect == 'inclusion' else 'decrease'
    plan = [{'chrom': 'chr2', 'strand': strand, 'site': kind, 'position': position,
             'exon_indices': {'target': 2}, 'present_in': ['target'], 'delta_direction': direction}
            for kind, position in [('acceptor', 400 if strand == '+' else 440),
                                   ('donor', 440 if strand == '+' else 400)]]
    return {'isoform_transcripts': {'target': transcript},
        'all_transcripts_full': {'other': other} if missing_catalog_reference else {'target': transcript, 'other': other},
        'routing_job': {'intent': 'splice_modulation', 'desired_effect': effect,
                        'desired_delta': direction, 'exon_index': 2, 'transcript_id': 'TARGET.1'},
        'splice_isoform_objectives': [{'isoform_label': 'target', 'desired_abundance': direction}],
        'splice_site_plan': plan}


def guards_for(goal, *, candidates=(), single_anchor=False, measured=True):
    plan = goal['splice_site_plan'][:1] if single_anchor else goal['splice_site_plan']
    labels = {runtime._canonical_site_label(site) for site in plan} if measured else set()
    mapping = runtime._build_safeguard_candidates_by_target(plan, list(candidates), goal, labels)
    return runtime._default_safeguards_from_candidates(mapping[runtime._canonical_site_label(plan[0])])


@pytest.fixture(autouse=True)
def no_model_decisions(monkeypatch):
    monkeypatch.setattr(runtime, 'resolve_ambiguous_safeguards', lambda *a, **k: {})


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('effect,bound', [('inclusion', 'max_decrease'), ('skipping', 'max_increase')])
@pytest.mark.parametrize('missing_catalog_reference', [False, True])
@pytest.mark.parametrize('single_anchor', [False, True])
@pytest.mark.parametrize('measured', [False, True])
def test_target_partner_allows_intended_movement(strand, effect, bound, missing_catalog_reference, single_anchor, measured):
    goal = exon_goal(strand, effect, missing_catalog_reference=missing_catalog_reference)
    partner = runtime._canonical_site_label(goal['splice_site_plan'][1])
    guards = guards_for(goal, single_anchor=single_anchor, measured=measured)
    guard = next(g for g in guards if g['label'] == partner)
    assert guard.get(bound) == .05
    assert 'max_abs_delta' not in guard
    assert 'targeted exon' in guard['note']
    assert not guard.get('required')


def test_reference_paired_boundary_changes_ranking_without_losing_opposing_changes(tmp_path):
    goal = exon_goal()
    guards = guards_for(goal)
    assert any(g['label'] == 'donor@140' and g.get('max_abs_delta') == .05 for g in guards)
    source = tmp_path/'scores.json'
    rows = []
    for name, primary, partner, neighbor in [('both_support_goal', .8, .7, 0),
            ('weaker', .65, .01, 0), ('opposing_partner', .9, -.2, 0),
            ('neighbor_change', .95, .8, .2)]:
        rows.append({'Name': name, 'Sequence': 'ACGT', 'delta_names': 'acceptor@400,donor@440,donor@140,acceptor@700',
                     'delta_deletion': f'{primary},{partner},{neighbor},0',
                     'delta_replacement': f'{primary},{partner},{neighbor},0'})
    source.write_text(json.dumps({'rows': rows}))
    router = importlib.import_module('aso_agents.splice_strategy_router_agent')
    output = tmp_path/'route.json'
    result = router.auto_threshold_constrained_site_metrics(json_path=str(source),
        primary_label='acceptor@400', primary_direction='increase', safeguards_json=json.dumps(guards),
        min_count=1, max_results=0, output_path=str(output))
    assert not result.get('error')
    selected = json.loads(output.read_text())['rows']
    assert selected[0]['Name'] == 'both_support_goal'
    by_name = {r['Name']: r for r in selected}
    assert by_name['both_support_goal']['safeguards_passed'] is True
    assert by_name['opposing_partner']['safeguards_passed'] is False
    assert by_name['neighbor_change']['safeguards_passed'] is False


@pytest.mark.parametrize('policy', [
    {'max_abs_delta': .02}, {'max_increase': .12, 'required': True}, {'required': True},
])
def test_explicit_guard_policy_keeps_precedence(policy):
    goal = exon_goal()
    guard = next(g for g in guards_for(goal, candidates=[{'label': 'donor@440', **policy}]) if g['label'] == 'donor@440')
    assert all(guard[k] == value for k, value in policy.items())


def test_explicit_preserved_transcript_keeps_two_sided_protection():
    goal = exon_goal()
    goal['preservation_targets'] = [{'transcript_id': 'TARGET.1', 'action': 'preserve'}]
    guard = next(g for g in guards_for(goal) if g['label'] == 'donor@440')
    assert guard.get('max_abs_delta') == .05


@pytest.mark.parametrize('fault', ['chromosome', 'strand', 'coordinate', 'ordinal', 'direction', 'intron_event', 'missing_goal', 'ambiguous_pair'])
def test_unproven_or_different_event_keeps_original_policy(fault):
    goal = exon_goal()
    if fault == 'chromosome': goal['isoform_transcripts']['target']['chrom'] = 'chr9'
    elif fault == 'strand': goal['isoform_transcripts']['target']['strand'] = '-'
    elif fault == 'coordinate': goal['isoform_transcripts']['target']['exons_tx'][1]['end'] += 1
    elif fault == 'ordinal': goal['splice_site_plan'][0]['exon_indices']['target'] = 1
    elif fault == 'direction': goal['routing_job']['desired_delta'] = 'decrease'
    elif fault == 'intron_event': goal['routing_job']['splice_event'] = {'event_type': 'intron_retention'}
    elif fault == 'missing_goal': goal['routing_job'].pop('desired_effect')
    else:
        alternate = deepcopy(goal['isoform_transcripts']['target'])
        alternate['transcript_id'] = 'ALTERNATE.1'
        alternate['exons_tx'][1]['end'] += 3
        goal['isoform_transcripts']['alternate'] = alternate
        goal['splice_site_plan'][0]['exon_indices']['alternate'] = 2
    guards = guards_for(goal)
    assert all('targeted exon' not in guard.get('note', '') for guard in guards)


def test_current_job_direction_replaces_cached_automatic_policy():
    goal = exon_goal(effect='skipping', missing_catalog_reference=True)
    cached = [{'label': 'donor@440', 'suggested_guard': 'no_decrease'}]
    guard = next(g for g in guards_for(goal, candidates=cached) if g['label'] == 'donor@440')
    assert guard.get('max_increase') == .05 and 'max_decrease' not in guard
    assert cached == [{'label': 'donor@440', 'suggested_guard': 'no_decrease'}]


def test_cached_paired_policy_cannot_relax_a_different_job():
    goal = exon_goal()
    mapping = runtime._build_safeguard_candidates_by_target(goal['splice_site_plan'], [], goal)
    cached = mapping['acceptor@400']
    assert any(c.get('paired_exon_objective') for c in cached)
    goal['routing_job']['desired_effect'] = None
    guard = next(g for g in guards_for(goal, candidates=cached) if g['label'] == 'donor@440')
    assert guard.get('max_abs_delta') == .05
    assert 'targeted exon' not in guard.get('note', '')


def test_long_exon_partner_is_measured_beyond_neighbor_radius():
    from aso_agents.splice_safeguards import include_neighborhood_measurements, neighborhood_guard_sites
    goal = exon_goal()
    for entry in [*goal['isoform_transcripts'].values(), *goal['all_transcripts_full'].values()]:
        entry['exons_tx'][1]['end'] = 5400
        entry['exons_tx'][2].update(start=7000, end=7040)
    goal['splice_site_plan'][1]['position'] = 5400
    anchor = goal['splice_site_plan'][0]
    contexts = neighborhood_guard_sites(goal, [anchor], radius_bp=100)
    assert {s['position'] for s in contexts} == {5400}
    guard = next(g for g in guards_for(goal, single_anchor=True) if g['label'] == 'donor@5400')
    assert guard.get('max_decrease') == .05
    evaluated = include_neighborhood_measurements(goal, [anchor], window_bp=10)
    assert any(s.get('site_kind') == 'donor' and s['donor']['position'] == 5400 for s in evaluated)
    # Evaluation plans carry their coordinates inside the per-site windows.
    evaluation_anchor = {'site_kind': 'acceptor', 'delta_direction': 'increase',
        'acceptor': {'chrom': 'chr2', 'strand': '+', 'position': 400},
        'donor': {'chrom': 'chr2', 'strand': '+', 'position': 5400}}
    evaluated = include_neighborhood_measurements(goal, [evaluation_anchor], window_bp=10)
    assert any(s.get('site_kind') == 'donor' and s['donor']['position'] == 5400 for s in evaluated)


def test_terminal_boundary_is_not_added_as_a_paired_splice_site():
    from aso_agents.splice_safeguards import neighborhood_guard_sites, target_exon_pairs
    goal = exon_goal()
    anchor = {'site': 'donor', 'position': 140, 'chrom': 'chr2', 'strand': '+',
              'delta_direction': 'increase', 'exon_indices': {'target': 1}}
    assert target_exon_pairs(goal, [anchor]) == {}
    assert all(s['position'] != 100 for s in neighborhood_guard_sites(goal, [anchor]))


def test_alternative_acceptor_keeps_its_own_partner_and_original_site_check():
    goal = exon_goal()
    alternate = deepcopy(goal['isoform_transcripts']['target'])
    alternate['transcript_id'] = 'ALTERNATE.1'
    alternate['exons_tx'][1]['start'] = 403
    goal['isoform_transcripts']['alternate'] = alternate
    goal['all_transcripts_full']['alternate'] = alternate
    goal['routing_job']['isoform_label'] = 'alternate'
    goal['splice_isoform_objectives'] = [{'isoform_label': 'alternate', 'desired_abundance': 'increase'}]
    acceptor, donor = goal['splice_site_plan']
    acceptor.update(position=403, exon_indices={'alternate': 2}, present_in=['alternate'])
    donor['exon_indices']['alternate'] = 2
    mapping = runtime._build_safeguard_candidates_by_target(goal['splice_site_plan'], [], goal)
    donor_guards = runtime._default_safeguards_from_candidates(mapping['donor@440'])
    target = next(g for g in donor_guards if g['label'] == 'acceptor@403')
    authentic = next(g for g in donor_guards if g['label'] == 'acceptor@400')
    assert target.get('max_decrease') == .05
    assert authentic.get('max_abs_delta') == .05
    assert 'targeted exon' not in authentic.get('note', '')
