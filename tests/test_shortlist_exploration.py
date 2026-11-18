"""Model-guided narrowing, challenge recovery, native contracts and resumption."""
from copy import deepcopy
import json
import pytest

from aso_agents import shortlist_exploration as exploration
from aso_agents import shortlist_review as review
from aso_agents.shortlist_evidence import evidence_catalog, molecule_id
from test_shortlist_review import model, run
from test_shortlist_scaling import large_pool, nomination_model


def querying_model(phase, payload, index, records, save, configuration):
    tools = {tool._tool_name: tool for tool in exploration._tools(index, records, save)}
    if phase == 'explore':
        field = next(key for key, item in index.fields.items() if item['path'].endswith('/selection_rank'))
        result = tools['filter_candidates']({'input_sets': ['all'], 'combine': 'union',
            'predicates': [{'field_id': field, 'scope_ids': [], 'operator': 'lte', 'value': 32,
                'aggregation': 'any', 'missing': 'include'}], 'join': 'all', 'context_policy': 'any',
            'rationale': 'Retain candidates for the first comparison.', 'order_by': [], 'offset': 0, 'limit': 64})
        assert 'error' not in result
        ids = result['candidate_ids']
        outsiders = []
    else:
        outsiders = payload['outsider_ids']
        for offset in range(0, len(outsiders), 16):
            result = tools['compare_candidates']({'candidate_ids': outsiders, 'field_ids': [], 'offset': offset, 'limit': 16})
            assert 'error' not in result
        ids = outsiders[:4]
        field = None
    response = {'attribute_ids': [field] if field else [],
        'summary': 'The panel comparison retains complementary binding sites and recorded screening evidence.'}
    if phase == 'explore':
        response.update(candidate_ids=ids, basis_set_ids=[result['set_id']])
    else:
        response.update(additional_candidate_ids=ids, strategy_supported=True,
            blocking_issues=[], caveats=[], reviewed_outsider_ids=outsiders)
    return response, []


def test_query_workflow_retains_every_candidate_and_reviews_complete_finalists(tmp_path, monkeypatch):
    rows = large_pool(200)
    before = deepcopy(rows)
    monkeypatch.setattr(exploration, '_run_model', querying_model)
    monkeypatch.setattr(review, '_model_call', nomination_model)
    selected, selection, receipt = run(rows, tmp_path, {'candidate_limit': 4}, review_strategy='auto')
    assert receipt['status'] == 'completed'
    assert len(selected) == 4
    assert receipt['review_strategy'] == 'query_guided'
    assert set(receipt['ranked_ids']) == {molecule_id(row) for row in rows}
    coverage = receipt['review_coverage']
    assert coverage['indexed_molecule_count'] == 200
    assert 4 <= coverage['fully_reviewed_molecule_count'] < 200
    assert coverage['compared_molecule_count'] < 200
    assert set(receipt['selected_ids']) <= set(receipt['fully_reviewed_ids'])
    assert rows == before
    def offline(*args, **kwargs):
        raise AssertionError('An accepted matching review should resume offline.')
    monkeypatch.setattr(review, '_model_call', offline)
    monkeypatch.setattr(exploration, '_run_model', offline)
    again, _, cached = run(rows, tmp_path, {'candidate_limit': 4}, review_strategy='auto')
    assert cached['status'] == 'completed'
    assert cached['selected_ids'] == receipt['selected_ids']


def test_native_selection_pulls_in_and_reviews_required_outside_candidates(tmp_path, monkeypatch):
    from test_shortlist_review import selector
    rows = large_pool(200)
    monkeypatch.setattr(exploration, '_run_model', querying_model)
    monkeypatch.setattr(review, '_model_call', nomination_model)
    native = selector(tmp_path)
    outside = rows[150]
    def required(pool, goal):
        selected, selection = native(pool, goal)
        found = next(row for row in pool if molecule_id(row) == molecule_id(outside))
        return [found, *[row for row in selected if molecule_id(row) != molecule_id(outside)][:3]], selection
    selected, _, receipt = review.recommend_panel(rows, {'candidate_limit': 4}, tmp_path,
        select_panel=required, allow_model_calls=True)
    assert receipt['status'] == 'completed'
    assert molecule_id(outside) in receipt['fully_reviewed_ids']
    assert molecule_id(outside) in receipt['selected_ids']


def test_empty_or_unproductive_exploration_uses_full_review(tmp_path, monkeypatch):
    def exhausted(*args):
        raise exploration.ExplorationExhausted('No defensible narrowing.')
    monkeypatch.setattr(exploration, '_run_model', exhausted)
    monkeypatch.setattr(review, '_model_call', nomination_model)
    selected, _, receipt = run(large_pool(129), tmp_path, {'candidate_limit': 3}, review_strategy='auto')
    assert receipt['status'] == 'completed'
    assert receipt['reviewed_molecule_count'] == 129
    assert len(selected) == 3
    assert (tmp_path/'recommendation_review/exploration_fallback.json').exists()


def test_provider_outage_preserves_provisional_pool_without_starting_full_review(tmp_path, monkeypatch):
    def outage(*args):
        raise TimeoutError('fixture provider unavailable')
    monkeypatch.setattr(exploration, '_run_model', outage)
    monkeypatch.setattr(review, '_model_call', lambda *args: (_ for _ in ()).throw(AssertionError('No cascade')))
    selected, _, receipt = run(large_pool(129), tmp_path, {'candidate_limit': 3}, review_strategy='auto')
    assert receipt['status'] == 'unavailable' and len(selected) == 3
    assert receipt['eligible_molecule_count'] == 129


def test_all_eligible_bypasses_narrowing(tmp_path, monkeypatch):
    monkeypatch.setattr(exploration, '_run_model', lambda *args: (_ for _ in ()).throw(AssertionError('All requested')))
    monkeypatch.setattr(review, '_model_call', nomination_model)
    selected, _, receipt = run(large_pool(129), tmp_path,
        {'shortlist_size_preference': 'all_eligible'}, review_strategy='auto')
    assert receipt['status'] == 'completed' and len(selected) == 129


def test_completed_query_components_resume_without_repeating_model_calls(tmp_path, monkeypatch):
    rows = large_pool(200)
    catalog = evidence_catalog(rows)
    configuration = {'screening_reasoning_effort': 'low'}
    monkeypatch.setattr(exploration, '_run_model', querying_model)
    first, receipt = exploration.explore(catalog, {}, tmp_path, configuration, [])
    monkeypatch.setattr(exploration, '_run_model', lambda *a: (_ for _ in ()).throw(AssertionError('Duplicate call')))
    second, resumed = exploration.explore(catalog, {}, tmp_path, configuration, [])
    assert set(first) == set(second) and resumed == receipt


@pytest.mark.parametrize('completion', [
    {'error': 'Max tool-calling steps exceeded'},
    {'error': 'incomplete_response', 'reason': 'max_output_tokens'},
    'The comparison is ready.'])
def test_tool_phase_and_structured_decision_have_separate_provider_contracts(monkeypatch, completion):
    from agents import Runner, _RunResult
    from aso_agents.shortlist_query import CandidateIndex
    index = CandidateIndex(evidence_catalog(large_pool(50)))
    records, calls = [], []
    expected = {'candidate_ids': list(index.ids)[:32], 'basis_set_ids': [],
                'attribute_ids': [], 'summary': 'Compare complementary placements.'}
    def runner(agent, input, max_steps):
        payload = json.loads(input)
        calls.append(agent)
        if len(calls) == 1:
            assert agent.tools and not agent.model_settings.extra_body
            assert payload['output_schema'] == exploration.FINAL_SCHEMA
            return _RunResult(completion, [{'response': {'id': 'exploration'}}])
        assert not agent.tools and max_steps == 0
        assert agent.model_settings.extra_body['text']['format']['type'] == 'json_schema'
        return _RunResult(json.dumps(expected), [{'response': {'id': 'decision'}}])
    monkeypatch.setattr(Runner, '_run_sync', runner)
    value, metadata = exploration._run_model('explore', {'working_limit': 64}, index,
        records, lambda: None, {'screening_reasoning_effort': 'low'})
    assert value == expected and len(calls) == 2
    assert [item['response']['id'] for item in metadata] == ['exploration', 'decision']


def test_query_provenance_keeps_large_sets_outside_comparison_membership(tmp_path, monkeypatch):
    from aso_agents.shortlist_query import CandidateIndex
    index = CandidateIndex(evidence_catalog(large_pool(200)))
    def model(phase, payload, index, records, save, configuration):
        tool = next(tool for tool in exploration._tools(index, records, save)
                    if tool._tool_name == 'filter_candidates')
        result = tool({'input_sets': ['all'], 'combine': 'union', 'predicates': [],
            'join': 'all', 'context_policy': 'any', 'rationale': 'Compare distinct binding sites.',
            'order_by': [], 'offset': 0, 'limit': 32})
        return {'candidate_ids': result['candidate_ids'], 'basis_set_ids': [result['set_id']],
            'attribute_ids': [], 'summary': 'Compare distinct binding sites.'}, []
    monkeypatch.setattr(exploration, '_run_model', model)
    result = exploration._session('explore', {'working_limit': 64}, index, tmp_path,
        {'screening_reasoning_effort': 'low'})
    assert len(exploration._response_ids(result, index)) == 32
    assert len(index.sets[result['basis_set_ids'][0]]) == 200


def test_query_errors_are_recorded_without_becoming_successful_operations():
    from aso_agents.shortlist_query import CandidateIndex
    index = CandidateIndex(evidence_catalog(large_pool(2)))
    records = []
    tool = next(tool for tool in exploration._tools(index, records, lambda: None)
                if tool._tool_name == 'filter_candidates')
    result = tool({'unsupported': 'query'})
    assert result['error'] and records[0]['outcome'] == 'error'
    assert not index.operations


def test_complete_ordinary_finalist_evidence_reaches_ranking_without_eager_page_calls(tmp_path, monkeypatch):
    rows = large_pool(140)
    for row in rows:
        row['source_record'] = 'evidence ' * 6000
    monkeypatch.setattr(exploration, '_run_model', querying_model)
    full_calls = []
    def reviewer(kind, payload, schema):
        assert kind not in {'evidence', 'evidence_audit'}
        if kind in {'rank', 'select', 'nominate'} and 'evidence evidence' in json.dumps(payload):
            full_calls.append(kind)
        return nomination_model(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', reviewer)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 4}, review_strategy='auto')
    assert receipt['status'] == 'completed'
    assert full_calls and receipt['review_coverage']['fully_reviewed_molecule_count'] >= 4


def test_contrary_finalist_evidence_reopens_selection_and_can_leave_empty_panel(tmp_path, monkeypatch):
    rows = large_pool(140)
    monkeypatch.setattr(exploration, '_run_model', querying_model)
    monkeypatch.setattr(review, '_model_call', nomination_model)
    original_rank, original_explain = review._rank, review._explain
    reopened = []
    def rank(catalog, context, *args, **kwargs):
        if context.get('selection_review_findings'):
            reopened.extend(catalog)
            return list(catalog), 0, [{'reviews': [{'preferred_ids': [],
                'summary': 'The recorded effects oppose the requested outcome.'}]}]
        return original_rank(catalog, context, *args, **kwargs)
    def explain(*args, **kwargs):
        if kwargs.get('reconsider'):
            raise review.PanelReconsideration('adverse_evidence', details={
                'issues': [{'candidate_id': molecule_id(rows[0]), 'problem': 'Opposing recorded effect.'}]})
        assert args[1] == [] and args[3]['selection_review_findings']
        return original_explain(*args, **kwargs)
    monkeypatch.setattr(review, '_rank', rank)
    monkeypatch.setattr(review, '_explain', explain)
    selected, selection, receipt = run(rows, tmp_path, {'candidate_limit': 4}, review_strategy='auto')
    assert receipt['status'] == 'completed'
    assert len(reopened) == 140 and selected == []
    assert selection['status'] == 'unfulfilled' and selection['selected_count'] == 0
    assert (tmp_path/'recommendation_review/selection_reconsideration.json').exists()
