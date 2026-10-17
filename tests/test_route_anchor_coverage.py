from copy import deepcopy
from types import SimpleNamespace

from aso_agents import pipeline_tools as pipeline


def site(position, direction='decrease', **extra):
    return {'site': 'acceptor', 'position': position, 'delta_direction': direction, **extra}


def runtime(monkeypatch):
    value = SimpleNamespace(_canonical_site_label=lambda e: f"{e['site']}@{e['position']}")
    monkeypatch.setattr(pipeline, '_goal_runtime_module', lambda: value)
    return value


def test_primary_candidate_projection_does_not_drop_selected_secondary_route(monkeypatch):
    runtime(monkeypatch)
    goal = {'steric_site_candidates': [site(100)],
            'splice_site_plan': [site(100), site(8000, uniqueness='shared',
                shared_with_beneficial=True, conflict_between_objectives=True)]}
    before = deepcopy(goal)
    anchors = pipeline._routing_anchor_source_entries(goal)
    assert [e['position'] for e in anchors] == [100, 8000]
    assert anchors[1]['conflict_between_objectives'] is True
    assert goal == before


def test_local_scope_does_not_import_global_sibling_routes(monkeypatch):
    runtime(monkeypatch)
    goal = {'routing_local_splice_site_plan': [site(100, 'increase')],
            'routing_local_steric_site_candidates': [site(100)],
            'steric_site_candidates': [site(100), site(8000)],
            'splice_site_plan': [site(100), site(8000)]}
    assert pipeline._routing_anchor_source_entries(goal) == [site(100, 'increase')]


def test_secondary_route_requires_measured_support(monkeypatch):
    rt = runtime(monkeypatch)
    goal = {'steric_site_candidates': [site(100)],
            'splice_site_plan': [site(100), site(8000)]}
    observed = [site(100, decision_direction='decrease')]
    assert [e['position'] for e in pipeline._objective_relevant_routing_entries(goal, observed, rt)] == [100]


def test_selected_secondary_route_is_not_subject_to_nearby_fallback_window(monkeypatch):
    rt = runtime(monkeypatch)
    goal = {'steric_site_candidates': [site(100)],
            'splice_site_plan': [site(100), site(8000)]}
    observed = [site(100, decision_direction='decrease'),
                site(8000, decision_direction='decrease')]
    selected = pipeline._objective_relevant_routing_entries(goal, observed, rt)
    assert {e['position'] for e in selected} == {100, 8000}


def test_route_metadata_records_resolved_provider_model(monkeypatch):
    runtime(monkeypatch)
    monkeypatch.setattr(pipeline, 'get_provider_config', lambda: SimpleNamespace(
        resolve_model=lambda _: 'provider/actual-model'))
    monkeypatch.setattr(pipeline, 'rank_route_candidates_with_llm', lambda _: {
        'ranked_routes':[{'label':'acceptor@100','direction':'decrease',
                          'rank':1,'rationale':'Observed decrease matches the objective.'}]})
    result = pipeline._rank_routing_entries_with_llm(
        structured_goal={}, routing_entries=[site(100)], debug_dir='')
    assert result['ranking_model'] == 'provider/actual-model'
    assert result['ranked_entries'][0]['ranking_model'] == 'provider/actual-model'
