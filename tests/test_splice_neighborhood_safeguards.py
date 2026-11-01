"""Focused design must still measure and disclose effects on nearby junctions."""
from copy import deepcopy
import importlib
import json

import pytest

import run_goal_intake as runtime
from aso_agents.splice_recommendations import (
    objective_prediction_evidence, resolve_routing_exon_objectives,
    report_recommendation_fields,
)
from test_structural_event_binding import conversion_fixture


def resolved_goal(monkeypatch, strand='+', effect='inclusion'):
    goal, program = conversion_fixture(strand)
    goal['desired_effect'] = program['desired_effect'] = effect
    for objective in goal['splice_isoform_objectives']:
        included = objective['isoform_label'] == 'included'
        objective['desired_abundance'] = 'increase' if included == (effect == 'inclusion') else 'decrease'
    monkeypatch.setattr(runtime, 'resolve_transcript_exon_nomenclature_sync',
                        lambda **kwargs: {'status': 'unresolved'})
    goal['splice_site_plan'], _ = runtime._compute_splice_site_plan(
        goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    goal['routing_job'] = dict(program)
    return goal


@pytest.mark.parametrize('strand', ['+', '-'])
def test_focused_evaluation_includes_shared_neighbor_without_new_design_route(monkeypatch, strand):
    goal = resolved_goal(monkeypatch, strand)
    before = deepcopy(goal)
    sites = runtime._build_splice_sites_for_eval(goal, None, 10)
    context = [s for s in sites if s.get('monitoring_purpose') == 'splice_neighborhood']
    shared = 'acceptor@' + str(3000 if strand == '+' else 1030)
    assert shared in {s['canonical_label'] for s in context}
    assert all(s['delta_direction'] == 'context' for s in context)
    assert all(set(s['present_in']) == {'included', 'excluded'} for s in context)
    assert goal == before
    # Transcript ends are not splice junctions.
    assert ('acceptor@100' if strand == '+' else 'acceptor@4030') not in {s['canonical_label'] for s in sites}
    assert ('donor@4030' if strand == '+' else 'donor@100') not in {s['canonical_label'] for s in sites}


def test_neighbor_selection_is_physical_and_bounded(monkeypatch):
    from aso_agents.splice_safeguards import neighborhood_guard_sites
    goal = resolved_goal(monkeypatch)
    for name, entry in list(goal['isoform_transcripts'].items()):
        other = deepcopy(entry)
        other['chrom'] = 'chr9'
        goal['isoform_transcripts']['other_' + name] = other
    context = neighborhood_guard_sites(goal, goal['splice_site_plan'], radius_bp=500)
    assert context == []
    context = neighborhood_guard_sites(goal, goal['splice_site_plan'], radius_bp=1500)
    assert context and all(s['chrom'] == 'chr8' for s in context)
    assert {s['position'] for s in context} <= {1000, 1030, 3000, 3030}


@pytest.mark.parametrize('measured', [True, False])
def test_shared_neighbor_guard_survives_focus_and_missing_measurements(monkeypatch, measured):
    goal = resolved_goal(monkeypatch)
    def drop_everything(_request, _objectives, cases, **kwargs):
        return {(c['primary_label'], c['candidate_label']): {'action': 'drop'} for c in cases}
    monkeypatch.setattr(runtime, 'resolve_ambiguous_safeguards', drop_everything)
    labels = {'acceptor@2000', 'donor@2030'} | ({'acceptor@3000'} if measured else set())
    chosen = runtime._build_safeguard_candidates_by_target(goal['splice_site_plan'], [], goal,
                                                         measured_site_labels=labels)
    guards = runtime._default_safeguards_from_candidates(chosen['acceptor@2000'])
    shared = next(g for g in guards if g['label'] == 'acceptor@3000')
    assert shared['max_abs_delta'] == .05 and not shared.get('required')
    assert 'Shared with beneficial' in shared['note']
    assert next(g for g in guards if g['label'] == 'donor@2030')['max_decrease'] == .05


def test_explicit_neighbor_bound_is_preserved(monkeypatch):
    goal = resolved_goal(monkeypatch)
    monkeypatch.setattr(runtime, 'resolve_ambiguous_safeguards', lambda *a, **k: {})
    explicit = {'label': 'acceptor@3000', 'max_increase': .2, 'required': True}
    chosen = runtime._build_safeguard_candidates_by_target(goal['splice_site_plan'], [explicit], goal)
    guards = runtime._default_safeguards_from_candidates(chosen['acceptor@2000'])
    guard = next(g for g in guards if g['label'] == 'acceptor@3000')
    assert guard['max_increase'] == .2 and guard['required'] is True
    assert 'max_abs_delta' not in guard


def test_cached_neighbor_from_another_program_cannot_lose_monitoring_policy(monkeypatch):
    goal = resolved_goal(monkeypatch)
    local = deepcopy(goal['splice_site_plan'])
    goal['splice_site_plan'].append({'site': 'acceptor', 'position': 3000, 'chrom': 'chr8', 'strand': '+',
        'present_in': ['included', 'excluded'], 'exon_indices': {'included': 4, 'excluded': 3}})
    monkeypatch.setattr(runtime, 'resolve_ambiguous_safeguards', lambda request, objectives, cases, **k: {
        (c['primary_label'], c['candidate_label']): {'action': 'drop'} for c in cases})
    mapping = runtime._build_safeguard_candidates_by_target(local, [{'label': 'acceptor@3000'}], goal)
    for candidates in mapping.values():
        guard = next(g for g in runtime._default_safeguards_from_candidates(candidates) if g['label'] == 'acceptor@3000')
        assert guard['max_abs_delta'] == .05


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('effect', ['inclusion', 'skipping'])
def test_verified_event_reaches_paired_route_evidence(monkeypatch, strand, effect):
    goal = resolved_goal(monkeypatch, strand, effect)
    objective, = resolve_routing_exon_objectives(goal)
    assert objective['status'] == 'resolved' and objective['exon_index'] == 3
    assert objective['identity']['status'] == 'verified_structural_event'
    assert objective['identity']['biological_label_verified'] is False
    pair = {'reference_transcript_id': 'TXIN.4', 'evaluation_status': 'evaluated',
            'window_bp': 10, 'delta_inclusion_deletion': .03, 'delta_inclusion_replacement': .02}
    for kind in ['acceptor', 'donor']:
        pair[kind] = {'chrom': 'chr8', 'strand': strand, 'position': objective[kind + '_position']}
    evidence = objective_prediction_evidence({'splice_site_deltas': [pair]}, objective)
    assert evidence['status'] == 'recorded_pair_predictions'
    state = evidence['measurements'][0]['prediction']['state']
    assert state == ('direction_supported' if effect == 'inclusion' else 'opposing')


@pytest.mark.parametrize('change', ['receipt', 'structure', 'version', 'direction', 'explicit_ordinal'])
def test_event_binding_cannot_override_conflicting_or_stale_evidence(monkeypatch, change):
    goal = resolved_goal(monkeypatch)
    if change == 'receipt':
        goal['exon_conversion_bindings'][0]['reference_structure_sha256'] = 'stale'
    elif change == 'structure':
        goal['isoform_transcripts']['included']['exons_tx'][2]['end'] += 1
    elif change == 'version':
        goal['routing_job']['transcript_id'] = 'TXIN.5'
    elif change == 'direction':
        goal['splice_isoform_objectives'][1]['desired_abundance'] = 'increase'
    else:
        goal['routing_job']['exon_index'] = 2
    objective, = resolve_routing_exon_objectives(goal)
    assert objective['status'] == 'unresolved'


def test_safeguard_priority_changes_order_without_erasing_advisory_failures(tmp_path):
    router = importlib.import_module('aso_agents.splice_strategy_router_agent')
    source, output = tmp_path / 'input.json', tmp_path / 'route.json'
    rows = []
    for name, primary, guard in [('large_breach', .8, .2), ('supported_safe', .04, .01), ('unknown_guard', .7, None)]:
        rows.append({'Name': name, 'Sequence': 'ACGT' * 4, 'GC': .5,
                     'delta_names': 'acceptor@2000,acceptor@3000',
                         'shared_context_label': 'acceptor@3000', 'shared_context_abs_delta': abs(guard) if guard is not None else None,
                         'shared_context_delta': guard, 'shared_context_site_kind': 'acceptor', 'shared_context_covers_all': True,
                     'delta_deletion': f'{primary},{guard if guard is not None else ""}',
                     'delta_replacement': f'{primary},{guard if guard is not None else ""}'})
    source.write_text(json.dumps({'rows': rows}))
    result = router.auto_threshold_constrained_site_metrics(json_path=str(source),
        primary_label='acceptor@2000', primary_direction='increase', min_count=3,
        safeguards_json=json.dumps([{'label': 'acceptor@3000', 'max_abs_delta': .05}]),
        min_gc=.1, max_gc=.9, output_path=str(output))
    assert not result.get('error')
    chosen = json.loads(output.read_text())['rows']
    assert [r['Name'] for r in chosen] == ['supported_safe', 'unknown_guard', 'large_breach']
    assert [r['safeguards_passed'] for r in chosen] == [True, None, False]
    assert chosen[-1]['safeguard_priority']['status'] == 'failed'
    assert report_recommendation_fields(chosen[-1])['safeguard_priority'] == chosen[-1]['safeguard_priority']


@pytest.mark.parametrize('raw,status,expected', [(.05004, 'evaluated', False),
    (.04996, 'evaluated', True), (None, 'evaluated', None),
    (.01, 'outside_evaluated_window', None)])
def test_guard_uses_original_precision_and_missingness(tmp_path, raw, status, expected):
    router = importlib.import_module('aso_agents.splice_strategy_router_agent')
    source, output = tmp_path / 'input.json', tmp_path / 'route.json'
    point = {'chrom': 'chr8', 'strand': '+', 'position': 3000}
    row = {'Sequence': 'ACGT' * 4, 'GC': .5, 'delta_names': 'acceptor@2000,acceptor@3000',
        'delta_deletion': '.4,.05', 'delta_replacement': '.4,.05',
        'splice_site_deltas': [{'canonical_label': 'acceptor@3000', 'site_kind': 'acceptor',
            'acceptor': point, 'window_bp': 10, 'evaluation_status': status,
            'delta_acceptor_deletion': raw, 'delta_acceptor_replacement': raw}]}
    source.write_text(json.dumps({'rows': [row]}))
    router.auto_threshold_constrained_site_metrics(json_path=str(source), primary_label='acceptor@2000',
        primary_direction='increase', safeguards_json=json.dumps([{'label': 'acceptor@3000'}]),
        min_count=1, min_gc=.1, max_gc=.9, output_path=str(output))
    data = json.loads(output.read_text())
    assert data['safeguards'][0]['max_abs_delta'] == .05
    assert data['rows'][0]['safeguards_passed'] is expected


def test_conflicting_raw_measurements_do_not_become_a_rounded_pass():
    router = importlib.import_module('aso_agents.splice_strategy_router_agent')
    site = {'canonical_label': 'donor@2030', 'site_kind': 'donor',
        'donor': {'chrom': 'chr8', 'strand': '+', 'position': 2030},
        'window_bp': 10, 'evaluation_status': 'evaluated',
        'delta_donor_deletion': .01, 'delta_donor_replacement': .01}
    row = {'delta_names': 'donor@2030', 'delta_deletion': '.01', 'delta_replacement': '.01',
           'splice_site_deltas': [site, {**site, 'delta_donor_deletion': .2}]}
    metric, = router._parse_site_metrics(row)
    assert metric['delta_deletion'] is metric['delta_replacement'] is None


def test_exported_neighbor_has_physical_sharing_context(tmp_path, monkeypatch):
    from aso_agents.pipeline_tools import _annotate_route_json_with_site_context
    goal = resolved_goal(monkeypatch)
    path = tmp_path / 'route.json'
    path.write_text(json.dumps({'safeguards': [{'label': 'acceptor@3000', 'max_abs_delta': .05}],
        'rows': [{'safeguard_report': [{'label': 'acceptor@3000', 'max_abs_delta': .05}]}]}))
    _annotate_route_json_with_site_context(json_path=str(path), structured_goal=goal)
    data = json.loads(path.read_text())
    for guard in [data['safeguards'][0], data['rows'][0]['safeguard_report'][0]]:
        assert guard['reference_scope']['status'] == 'shared'
        assert guard['shared_with_beneficial'] is True
        assert set(guard['selected_present_in_transcripts']) == {'TXIN', 'TXOUT'}


def test_final_shortlist_keeps_guard_priority_and_disclosure(tmp_path):
    from aso_agents.candidate_assessment import assess_candidates, rank_candidates, write_assessment_artifacts
    from aso_agents.splice_recommendations import safeguard_priority
    from test_candidate_assessment import row
    rows = []
    for sequence, passes, count in [('ACGTTGCA', False, 0), ('AACCTTGG', None, 0), ('GGAACTCT', True, 3)]:
        reports = [{'label': 'donor@2030', 'passes': passes, 'values': [.2] if passes is False else [],
                    'max_abs_delta': .05}]
        rows.append({**row(sequence, count=count), 'safeguard_report': reports,
            'safeguard_priority': safeguard_priority(reports)})
    assessed = assess_candidates(rows, {})
    ranked = rank_candidates(assessed)
    assert [r['synthesized_antisense_sequence'] for r in ranked] == ['GGAACTCT', 'AACCTTGG', 'ACGTTGCA']
    assert all(r['candidate_assessment']['selection_eligible'] for r in ranked)
    assert [r['candidate_assessment']['recommendation_status'] for r in ranked][1:] == ['exploratory', 'exploratory']
    outputs = write_assessment_artifacts(ranked, ranked, {'shortlist_size_preference': 'all_eligible'}, tmp_path)
    data = json.loads(open(outputs['recommended_shortlist_json']).read())
    assert [r['safeguard_status'] for r in data['records']] == ['passed', 'unmeasured', 'failed']
    assert all(r['safeguard_report'] for r in data['records'])
    assert 'Splice safeguards' in open(outputs['recommended_shortlist_md']).read()


def test_explicit_guard_cap_is_not_shrunk_by_legacy_context_filter(tmp_path):
    router = importlib.import_module('aso_agents.splice_strategy_router_agent')
    source, output = tmp_path / 'input.json', tmp_path / 'route.json'
    source.write_text(json.dumps({'rows': [{'Sequence': 'ACGT'*4, 'GC': .5,
        'delta_names': 'acceptor@2000,acceptor@3000', 'delta_deletion': '.4,.25', 'delta_replacement': '.4,.25',
        'shared_context_label': 'acceptor@3000', 'shared_context_abs_delta': .25,
        'shared_context_delta': .25, 'shared_context_site_kind': 'acceptor', 'shared_context_covers_all': True}]}))
    router.auto_threshold_constrained_site_metrics(json_path=str(source), primary_label='acceptor@2000',
        primary_direction='increase', safeguards_json=json.dumps([{'label': 'acceptor@3000', 'max_abs_delta': .3, 'required': True}]),
        min_count=1, min_gc=.1, max_gc=.9, output_path=str(output))
    data = json.loads(output.read_text())
    assert len(data['rows']) == 1 and data['rows'][0]['safeguards_passed'] is True
