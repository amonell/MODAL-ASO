"""Native routing/export controls with neutral, explicitly synthetic predictions."""
from copy import deepcopy
import csv
import importlib
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as pipeline
router = importlib.import_module('aso_agents.splice_strategy_router_agent')
from aso_agents.splice_recommendations import resolve_routing_exon_objectives
import run_goal_intake as runtime

from test_splice_recommendation_evidence import annotated, goal, site


def sequence(index):
    return 'ACGTACGTACG' + ''.join('ACGT'[(index >> shift) & 3] for shift in (6, 4, 2, 0))


@pytest.mark.parametrize('effect,sign', [('inclusion', 1), ('skipping', -1)])
def test_scored_route_shortlist_download_and_comparison_keep_direction_and_scope(tmp_path, effect, sign):
    rows = []
    # Place a large adverse prediction first, followed by thirteen supportive
    # candidates. Every row has a favorable selected-site prediction, so the
    # paired objective is essential. A final row violates a required guard.
    for index in range(15):
        pair = (-sign*.8, -sign*.7) if index == 0 else (sign*.03, sign*.02)
        primary = sign*.8 if index == 0 else sign*.03
        rows.append({'Sequence': sequence(index), 'Chrom': 'chr8', 'Start': 180+index,
            'End': 195+index, 'Strand': '+', 'GC': .5, 'Score': 100-index,
            'delta_names': 'acceptor@201,donor@999',
            'delta_deletion': f'{primary},{-.9 if index == 14 else 0}',
            'delta_replacement': f'{sign*.02},0',
            'splice_site_deltas': [site(pair)]})
    source = tmp_path/'scores.json'
    source.write_text(json.dumps({'rows': rows}))
    folder = tmp_path/'strategy_routes'
    folder.mkdir()
    output = folder/'neutral.route.json'
    objectives = resolve_routing_exon_objectives(goal(effect))
    result = router.auto_threshold_constrained_site_metrics(
        json_path=str(source), primary_label='acceptor@201',
        primary_direction='increase' if sign > 0 else 'decrease',
        safeguards_json=json.dumps([{'label': 'donor@999', 'required': True, 'max_abs_delta': .1}]),
        exon_objectives_json=json.dumps(objectives), start_threshold=.5,
        min_threshold=.01, min_count=14, min_gc=.1, max_gc=.9,
        max_results=50, output_path=str(output))
    assert result['count'] == 14, result
    selected = json.loads(output.read_text())['rows']
    assert [row['Sequence'] for row in selected[:13]] == [sequence(i) for i in range(1, 14)]
    assert selected[-1]['Sequence'] == sequence(0)
    assert selected[-1]['recommendation_status'] == 'opposing_predictions'
    assert 'exploratory' in selected[-1]['recommendation_note']
    assert all(row['Sequence'] != sequence(14) for row in selected)
    assert selected[0]['exon_objective_evidence'][0]['reference_transcript_id'] == 'TX_NEUTRAL.2'

    html_rows = runtime._collect_route_data(str(tmp_path))['acceptor@201']['asos']
    assert len(html_rows) == 12
    assert [row['sequence'] for row in html_rows] == [sequence(i) for i in range(1, 13)]
    assert all(row['recommendation_status'] == 'direction_supported' for row in html_rows)

    # Exercise real CSV loading and generated-candidate normalization. Deliberate
    # reverse order ensures final selection cannot depend on the route's order.
    csv_path = tmp_path/'download.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in selected))))
        writer.writeheader()
        for row in reversed(selected):
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    loaded = pipeline._load_candidate_rows_from_csv(csv_path)
    normalized = [pipeline._normalized_generated_candidate_row(row) for row in loaded]
    assert normalized[0]['recommendation_status'] == 'opposing_predictions'
    final, provenance = pipeline._apply_generated_candidate_limit(
        normalized, candidate_limit=12, per_site=True, group_scope='route')
    assert len(final) == 12 and provenance['eligible_count'] == 14
    assert all(row['recommendation_status'] == 'direction_supported' for row in final)
    assert all(row['exon_objective_evidence'][0]['exon_index'] == 2 for row in final)
    compact = pipeline._compact_candidate_row_for_ranking(normalized[0])
    assert compact['recommendation_status'] == 'opposing_predictions'
    assert compact['exon_objective_evidence'][0]['measurements'][0]['values'] == [-sign*.8, -sign*.7]


def test_nonsplice_count_selection_retains_existing_rank_behavior():
    rows = [{'sequence': sequence(i), 'selection_rank': rank, 'job_label': 'neutral'}
            for i, rank in enumerate([3, 1, 2])]
    selected, _ = pipeline._apply_generated_candidate_limit(rows, candidate_limit=2, per_site=False)
    assert [row['selection_rank'] for row in selected] == [1, 2]
    selected, _ = pipeline._apply_generated_candidate_limit(rows, candidate_limit=2, per_site=True)
    assert selected == rows[1:]


def test_scoring_persistence_keeps_reference_bound_raw_pairs_without_job_outcome(tmp_path, monkeypatch):
    pair = site(reference_transcript_id=None, canonical_label='acceptor@201', site_kind='acceptor',
                delta_acceptor_deletion=.1, delta_acceptor_replacement=.2)
    raw = {'results': [{'splice_site_deltas': [pair], 'delta_inclusion_deletion': .03,
                        'delta_inclusion_replacement': .02, 'splice_eval_status': 'evaluated'}]}
    unchanged = deepcopy(raw)
    monkeypatch.setattr(pipeline, 'evaluate_splice_shifts_on_tiles', lambda **kwargs: deepcopy(raw))
    monkeypatch.setattr(pipeline, '_annotate_repeat_risk_json', lambda **kwargs: {})
    tiles = tmp_path/'input.json'
    tiles.write_text(json.dumps({'rows': [{'Sequence': sequence(1), 'Chrom': 'chr8',
        'Start': 180, 'End': 195, 'Strand': '+'}]}))
    result = pipeline._run_splice_eval_on_tiles_json(initial_tiles_path=str(tiles),
        structured_goal=goal(), transcript_id='TX_NEUTRAL.2', patient_reference_fasta=None,
        splice_sites_json=None, runtime=SimpleNamespace(filter_tiles_json=runtime.filter_tiles_json))
    assert 'error' not in result, result
    saved = json.loads((tmp_path/'scored_tiles.json').read_text())['rows'][0]
    assert saved['splice_site_deltas'][0]['reference_transcript_id'] == 'TX_NEUTRAL.2'
    assert saved['splice_site_deltas'][0]['evaluation_status'] == 'evaluated'
    assert saved['splice_site_deltas'][0]['delta_inclusion_deletion'] == .03
    assert not any(key.startswith('recommendation_') for key in saved)
    assert 'desired_effect' not in saved['splice_site_deltas'][0]
    assert saved['delta_deletion'] == '0.1000'
    assert raw == unchanged


def test_atomic_plan_executor_transmits_the_resolved_job_objective(tmp_path, monkeypatch):
    payload = goal()
    entry = {'label': 'acceptor@201', 'site': 'acceptor', 'position': 201,
             'decision_direction': 'increase'}
    payload['splice_site_plan'] = [entry]
    payload['routing_min_count'] = 1
    source = tmp_path/'scores.json'
    source.write_text(json.dumps({'rows': [{'Sequence': sequence(1), 'delta_names': 'acceptor@201',
        'delta_deletion': '.03', 'delta_replacement': '.02', 'splice_site_deltas': [site()]}]}))
    monkeypatch.setattr(pipeline, '_routing_target_entries', lambda *args, **kwargs: [entry])
    monkeypatch.setattr(pipeline, '_rank_routing_entries_with_llm', lambda **kwargs: {'ranked_entries': [entry]})
    plan = pipeline._build_atomic_routing_plan(structured_goal=payload, scored_path=str(source),
        run_dir=str(tmp_path), flags={})
    assert 'error' not in plan, plan
    step = plan['planner_plan']['steps'][0]
    assert json.loads(step['args']['exon_objectives_json']) == resolve_routing_exon_objectives(payload)
    result = router.execute_router_plan(plan['planner_plan'], str(source), plan['output_dir'])
    assert not result.get('error') and not result.get('errors'), result
    rows = json.loads(open(step['args']['output_path']).read())['rows']
    assert len(rows) == 1 and rows[0]['recommendation_status'] == 'direction_supported'


def test_rendered_table_preserves_backend_order_and_explains_both_predictions():
    from test_html_count_units import render_table
    good = annotated((.03, .02), (.03, .02))
    bad = annotated((-.8, -.7), (.8, -.9))
    good.update(sequence=sequence(1), mean_splice_delta=.025)
    bad.update(sequence=sequence(0), mean_splice_delta=-.05)
    html = render_table([good, bad])
    assert html.index(sequence(1)) < html.index(sequence(0))
    assert 'TX_NEUTRAL.2' in html and 'exon ordinal 2' in html
    assert '0.03 / 0.02' in html and '-0.8 / -0.7' in html
    assert 'exploratory' in html and 'unmeasured' in html
