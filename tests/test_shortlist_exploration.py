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
        response.update(candidate_ids=ids, query_slices=[], basis_set_ids=[result['set_id']])
    else:
        response.update(additional_candidate_ids=ids, additional_query_slices=[], strategy_supported=True,
            filter_decisions=[{'set_id': key, 'keep': True} for key in payload['proposed_filter_ids']],
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
    expected = {'candidate_ids': list(index.ids)[:32], 'query_slices': [], 'basis_set_ids': [],
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
        return {'candidate_ids': [], 'query_slices': [{'set_id': result['set_id'], 'offset': 0, 'take': 32}],
            'basis_set_ids': [result['set_id']],
            'attribute_ids': [], 'summary': 'Compare distinct binding sites.'}, []
    monkeypatch.setattr(exploration, '_run_model', model)
    result = exploration._session('explore', {'working_limit': 64}, index, tmp_path,
        {'screening_reasoning_effort': 'low'})
    assert len(exploration._response_ids(result, index)) == 32
    assert len(index.sets[result['basis_set_ids'][0]]) == 200


@pytest.mark.parametrize('preloaded,corrections,needs_queries,steps', [
    (True, [], False, [1, 0]), (False, [], False, [2, 0]),
    (True, ['Choose a shorter slice.'], False, [0]),
    (True, ['Inspect missing evidence.'], True, [2, 0])])
def test_challenger_reuses_observations_for_membership_repairs(monkeypatch, preloaded, corrections, needs_queries, steps):
    from agents import Runner, _RunResult
    from aso_agents.shortlist_query import CandidateIndex
    index = CandidateIndex(evidence_catalog(large_pool(40)))
    key = index.ids[10]
    response = {'additional_candidate_ids': [], 'additional_query_slices': [], 'filter_decisions': [],
                'attribute_ids': [], 'summary': 'Compare the alternative site.', 'strategy_supported': True,
                'blocking_issues': [], 'caveats': [], 'reviewed_outsider_ids': [key]}
    calls = []
    def runner(agent, input, max_steps):
        calls.append(max_steps)
        if agent.tools:
            return _RunResult({'error': 'Max tool-calling steps exceeded'})
        request = json.loads(input)
        assert 'resumed_operations' not in request
        assert request['previous_decision'] == response
        return _RunResult(json.dumps(response))
    monkeypatch.setattr(Runner, '_run_sync', runner)
    out, _ = exploration._run_model('challenge', {'working_limit': 64, 'outsider_ids': [key],
        'outsider_evidence': {'ids': [key] if preloaded else []}, 'resumed_operations': [],
        'previous_decision': response, 'corrections': corrections, 'needs_more_queries': needs_queries},
        index, [], lambda: None, {'screening_reasoning_effort': 'low'})
    assert out == response and calls == steps


def test_query_errors_are_recorded_without_becoming_successful_operations():
    from aso_agents.shortlist_query import CandidateIndex
    index = CandidateIndex(evidence_catalog(large_pool(2)))
    records = []
    tool = next(tool for tool in exploration._tools(index, records, lambda: None)
                if tool._tool_name == 'filter_candidates')
    result = tool({'unsupported': 'query'})
    assert result['error'] and records[0]['outcome'] == 'error'
    assert not index.operations


def test_withdrawn_filter_cannot_supply_membership_or_report_rationale(tmp_path, monkeypatch):
    rows = large_pool(160)
    sets, corrections = {}, []
    def challenge_filter(phase, payload, index, records, save, configuration):
        call = next(tool for tool in exploration._tools(index, records, save)
                    if tool._tool_name == 'filter_candidates')
        rank = next(key for key, field in index.fields.items() if field['path'].endswith('/selection_rank'))
        def query(parent, operator, threshold, rationale):
            result = call({'input_sets': [parent], 'combine': 'union',
                'predicates': [{'field_id': rank, 'scope_ids': [], 'operator': operator,
                    'value': threshold, 'aggregation': 'any', 'missing': 'exclude'}],
                'join': 'all', 'context_policy': 'any', 'rationale': rationale,
                'order_by': [], 'offset': 0, 'limit': 32})
            return result['set_id']
        if phase == 'explore':
            sets['original'] = query('all', 'lte', 40, 'An unsupported preliminary priority.')
            sets['child'] = query(sets['original'], 'lte', 32, 'A subset of that preliminary choice.')
            return {'candidate_ids': [], 'query_slices': [{'set_id': sets['child'], 'offset': 0, 'take': 32}],
                    'basis_set_ids': [], 'attribute_ids': [rank], 'summary': 'Initial comparison.'}, []
        assert set(payload['proposed_filter_ids']) == {sets['original'], sets['child']}
        replacement = sets['child']
        if payload['corrections']:
            corrections.extend(payload['corrections'])
            sets['independent'] = query('all', 'gte', 80, 'Compare independently supported sites.')
            replacement = sets['independent']
        return {'additional_candidate_ids': [], 'additional_query_slices': [
                    {'set_id': replacement, 'offset': 0, 'take': 73 if corrections else 32}],
            'filter_decisions': [{'set_id': key, 'keep': key != sets['original']}
                                 for key in payload['proposed_filter_ids']],
            'attribute_ids': [rank], 'strategy_supported': True, 'summary': 'A revised comparison.',
            'blocking_issues': [], 'caveats': [], 'reviewed_outsider_ids': payload['outsider_ids']}, []
    monkeypatch.setattr(exploration, '_run_model', challenge_filter)
    monkeypatch.setattr(review, '_model_call', nomination_model)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 4}, review_strategy='auto')
    assert receipt['status'] == 'completed' and any('descendants' in text for text in corrections)
    decision = json.loads((tmp_path/'recommendation_review/exploration.json').read_text())
    assert decision['basis_set_ids'] == [sets['independent']]
    assert decision['withdrawn_set_ids'] == [sets['original']]
    assert len(decision['proposed_ids']) >= 73
    assert set(decision['proposed_ids']) <= set(decision['matched_ids']) | {molecule_id(row) for row in rows[:4]}
    assert [item['rationale'] for item in receipt['selection_process']['steps']] == ['Compare independently supported sites.']


@pytest.mark.parametrize('valid_repair', [True, False])
def test_first_completed_decision_can_correct_membership_after_truncation(tmp_path, monkeypatch, valid_repair):
    phases = []

    def reviewer(phase, payload, index, records, save, configuration):
        if phase == 'explore':
            return querying_model(phase, payload, index, records, save, configuration)
        phases.append(payload)
        if len(phases) == 1:
            raise exploration.ExplorationExhausted('Truncated final decision.')
        take = 120
        if len(phases) == 3:
            assert any('limit is 32' in error for error in payload['corrections'])
            assert not payload['needs_more_queries']
            assert payload['previous_decision']['additional_query_slices'][0]['take'] == take
            if valid_repair:
                take = 8
        return {'additional_candidate_ids': [],
            'additional_query_slices': [{'set_id': 'all', 'offset': 0, 'take': take}],
            'filter_decisions': [{'set_id': key, 'keep': True} for key in payload['proposed_filter_ids']],
            'attribute_ids': [], 'summary': 'Compare alternatives at additional positions.',
            'strategy_supported': True, 'blocking_issues': [], 'caveats': [],
            'reviewed_outsider_ids': payload['outsider_ids']}, []

    monkeypatch.setattr(exploration, '_run_model', reviewer)
    catalog = evidence_catalog(large_pool(160))
    config = {'screening_reasoning_effort': 'low'}
    if valid_repair:
        _, receipt = exploration.explore(catalog, {}, tmp_path, config, [])
        assert len(receipt['proposed_ids']) <= 40
    else:
        with pytest.raises(exploration.ExplorationExhausted) as error:
            exploration.explore(catalog, {}, tmp_path, config, [])
        assert 'could not be validated' in str(error.value)
    assert len(phases) == 3


def test_compact_finalist_group_receives_a_single_complete_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(exploration, '_run_model', querying_model)
    monkeypatch.setattr(review, '_model_call', nomination_model)
    _, _, receipt = run(large_pool(140), tmp_path, {'candidate_limit': 4}, review_strategy='auto')
    assert receipt['status'] == 'completed'
    subset = json.loads((tmp_path/'recommendation_review/exploration.json').read_text())['proposed_ids']
    assert set(subset) == set(receipt['fully_reviewed_ids'])
    levels = [level for level in receipt['review_levels'] if level['method'] == 'final_selection']
    assert len(levels) == 1


def test_complete_ordinary_finalist_evidence_reaches_ranking_without_eager_page_calls(tmp_path, monkeypatch):
    rows = large_pool(140)
    for row in rows:
        row['provenance'] = {'source_record': 'evidence ' * 6000}
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


def test_incomplete_exploration_resume_retains_attempts_and_reports_each_query_once(tmp_path, monkeypatch):
    from aso_agents.shortlist_query import CandidateIndex
    catalog = evidence_catalog(large_pool(50))
    progress = []
    monkeypatch.setattr(review, '_progress', lambda *a, **k: progress.append((a, k)))
    def interrupted(*args):
        response = querying_model(*args)
        raise TimeoutError('Fixture interruption after a completed query')
    monkeypatch.setattr(exploration, '_run_model', interrupted)
    with pytest.raises(TimeoutError):
        exploration._session('explore', {'working_limit': 64}, CandidateIndex(catalog), tmp_path,
            {'screening_reasoning_effort': 'low'})
    assert len(progress) == 1
    monkeypatch.setattr(exploration, '_run_model', querying_model)
    exploration._session('explore', {'working_limit': 64}, CandidateIndex(catalog), tmp_path,
        {'screening_reasoning_effort': 'low'})
    receipt = json.loads(next((tmp_path/'recommendation_review/exploration').glob('*.json')).read_text())
    assert receipt['status'] == 'accepted'
    assert receipt['attempts'][0]['error_type'] == 'TimeoutError'
    assert len(receipt['attempts']) == 2
    assert len(progress) == 2


def test_distinct_fill_uses_only_the_chosen_order_and_reports_exhaustion():
    from types import SimpleNamespace
    index = SimpleNamespace(ordered_sets={'ranked': list('abcdef')})
    proposal = {'candidate_ids': ['d', 'c'], 'query_slices': [
        {'set_id': 'ranked', 'offset': 0, 'take': 2}, {'set_id': 'ranked', 'offset': 2, 'fill_to_distinct': 5}]}
    assert exploration._response_ids(proposal, index) == list('dcabe')
    proposal['query_slices'][-1]['fill_to_distinct'] = 7
    with pytest.raises(ValueError) as exhausted:
        exploration._response_ids(proposal, index)
    assert 'reached 6 distinct' in str(exhausted.value)
    proposal['query_slices'].reverse()
    with pytest.raises(ValueError) as misplaced:
        exploration._response_ids(proposal, index)
    assert 'final slice' in str(misplaced.value)


@pytest.mark.parametrize('approve', [True, False])
def test_one_duplicate_in_replacement_requires_approval_of_exact_extension(tmp_path, monkeypatch, approve):
    from aso_agents.shortlist_query import CandidateIndex
    catalog = evidence_catalog(large_pool(140))
    index = CandidateIndex(catalog, [])
    kept = list(index.ordered_sets['all'])
    index.ordered_sets.update(kept=kept, withdrawn=list(reversed(kept)))
    index.sets.update(kept=set(kept), withdrawn=set(kept))
    individual = kept[:11] + kept[48:49]
    submitted = {'additional_candidate_ids': individual, 'additional_query_slices': [
        {'set_id': 'kept', 'offset': 48, 'take': 20}], 'filter_decisions': [
        {'set_id': 'withdrawn', 'keep': False}, {'set_id': 'kept', 'keep': True}],
        'attribute_ids': [], 'summary': 'Keep the approved ranking and individually reviewed alternatives.',
        'strategy_supported': True, 'blocking_issues': [], 'caveats': [], 'reviewed_outsider_ids': individual}
    monkeypatch.setattr(exploration, '_run_model', lambda *args: (submitted, []))
    confirmations = []
    def confirm(kind, payload, schema):
        assert kind == 'membership_confirmation'
        confirmations.append(payload)
        assert payload['introduced_by_overlap_repair'] == kept[68:69]
        assert payload['ordered_members'] == list(dict.fromkeys(individual + kept[48:69]))
        return {'approve': approve, 'reason': 'Reviewed the exact comparison membership.'}, []
    monkeypatch.setattr(review, '_model_call', confirm)
    payload = {'goal': {}, 'outsider_ids': individual, 'outsider_evidence': {'ids': individual},
        'proposed_filter_ids': ['withdrawn', 'kept'], 'proposed_ids': [], 'working_limit': 64,
        'additional_candidate_limit': 32, 'replacement_candidate_limit': 96}
    invoke = lambda: exploration._session('challenge', payload, index, tmp_path, {})
    if approve:
        decision = invoke()
        assert decision['additional_query_slices'] == [{'set_id': 'kept', 'offset': 48, 'fill_to_distinct': 32}]
        assert invoke() == decision
        assert submitted['additional_query_slices'][0]['take'] == 20
    else:
        with pytest.raises(exploration.ExplorationExhausted):
            invoke()
    assert len(confirmations) == 1


def test_duplicate_shortfall_cannot_revive_a_withdrawn_source(tmp_path, monkeypatch):
    from aso_agents.shortlist_query import CandidateIndex
    index = CandidateIndex(evidence_catalog(large_pool(140)), [])
    ordered = list(index.ordered_sets['all'])
    index.ordered_sets['withdrawn'] = ordered
    index.sets['withdrawn'] = set(ordered)
    payload = {'goal': {}, 'outsider_ids': [], 'outsider_evidence': {'ids': []}, 'proposed_ids': [],
        'proposed_filter_ids': ['withdrawn'], 'replacement_candidate_limit': 96, 'additional_candidate_limit': 32}
    response = {'additional_candidate_ids': [], 'additional_query_slices': [
        {'set_id': 'withdrawn', 'offset': 0, 'fill_to_distinct': 32}],
        'filter_decisions': [{'set_id': 'withdrawn', 'keep': False}], 'attribute_ids': [], 'summary': 'Initial choice.',
        'strategy_supported': True, 'blocking_issues': [], 'caveats': [], 'reviewed_outsider_ids': []}
    monkeypatch.setattr(exploration, '_run_model', lambda *args: (response, []))
    monkeypatch.setattr(review, '_model_call', lambda *args: pytest.fail('A withdrawn source cannot reach confirmation'))
    with pytest.raises(exploration.ExplorationExhausted):
        exploration._session('challenge', payload, index, tmp_path, {})


@pytest.mark.parametrize('repairs_query', [True, False])
def test_failed_replacement_filter_has_one_bounded_tool_repair(tmp_path, monkeypatch, repairs_query):
    from aso_agents.shortlist_query import CandidateIndex
    index = CandidateIndex(evidence_catalog(large_pool(140)), [])
    ids = list(index.ordered_sets['all'])
    index.ordered_sets['withdrawn'] = ids[:40]
    index.sets['withdrawn'] = set(ids[:40])
    field = next(key for key, item in index.fields.items() if item['path'].endswith('/selection_rank'))
    payload = {'goal': {}, 'outsider_ids': [], 'outsider_evidence': {'ids': []}, 'proposed_ids': ids[:40],
               'proposed_filter_ids': ['withdrawn'], 'replacement_candidate_limit': 96, 'additional_candidate_limit': 32}
    attempts = []
    def reviewer(phase, request, index, records, save, configuration):
        tool = next(t for t in exploration._tools(index, records, save) if t._tool_name == 'filter_candidates')
        attempts.append(request['needs_more_queries'])
        scopes = [] if repairs_query and len(attempts) == 3 else ['invalid_scope']
        result = tool({'input_sets': ['all'], 'combine': 'union', 'join': 'all', 'context_policy': 'any',
            'predicates': [{'field_id': field, 'scope_ids': scopes, 'operator': 'gte', 'value': 20,
                            'aggregation': 'any', 'missing': 'exclude'}], 'order_by': [], 'offset': 0, 'limit': 32,
            'rationale': 'Compare candidates from an independent query.'})
        return {'additional_candidate_ids': [], 'additional_query_slices': [] if 'error' in result else [
                {'set_id': result['set_id'], 'offset': 0, 'take': 32}],
            'filter_decisions': [{'set_id': 'withdrawn', 'keep': False}], 'attribute_ids': [field],
            'strategy_supported': True, 'blocking_issues': [], 'caveats': [], 'reviewed_outsider_ids': [],
            'summary': 'Withdraw the earlier filter and review a replacement set.'}, []
    monkeypatch.setattr(exploration, '_run_model', reviewer)
    def run_challenge():
        return exploration._session('challenge', payload, index, tmp_path, {})
    if repairs_query:
        result = run_challenge()
        assert len(exploration._response_ids(result, index)) == 32
        assert result['additional_query_slices'][0]['set_id'] != 'withdrawn'
    else:
        with pytest.raises(exploration.ExplorationExhausted):
            run_challenge()
    assert attempts == [False, True, True]
