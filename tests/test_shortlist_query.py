"""Exact queries, correlated records and adversarial filtering on saved evidence."""
from copy import deepcopy
import pytest

from aso_agents.shortlist_evidence import evidence_catalog, molecule_id
from aso_agents.shortlist_query import CandidateIndex
from test_shortlist_review import pool


def index(rows):
    return CandidateIndex(evidence_catalog(rows))


def field(data, suffix):
    return next(key for key, item in data.fields.items() if item['path'].endswith(suffix))


def predicate(data, suffix, value, **extra):
    return {'field_id': field(data, suffix), 'operator': 'gte', 'value': value,
            'scope_ids': [], 'aggregation': 'any', 'missing': 'exclude', **extra}


def query(data, predicates, **kwargs):
    return data.filter(input_sets=['all'], combine='union', predicates=predicates,
        join='all', context_policy='any', rationale='Compare the recorded measurements.', **kwargs)


def test_ranges_reversible_sets_sorting_and_original_evidence():
    rows = pool(7)
    for i, row in enumerate(rows):
        row['new_score'] = i if i < 6 else None
    original = deepcopy(rows)
    data = index(rows)
    p = predicate(data, '/new_score', 3)
    result = query(data, [p], order_by=[{'field_id': p['field_id'], 'scope_ids': [],
        'aggregation': 'min', 'direction': 'descending'}])
    assert result['candidate_ids'] == [molecule_id(rows[i]) for i in (5, 4, 3)]
    assert (result['matched_count'], result['unknown_count']) == (3, 1)
    outside = data.filter(['all', result['set_id']], 'difference', [], 'all', 'any', 'Inspect excluded alternatives.')
    assert set(outside['candidate_ids']) == {molecule_id(rows[i]) for i in (0, 1, 2, 6)}
    assert rows == original
    assert query(data, [dict(p, missing='only')])['candidate_ids'] == [molecule_id(rows[6])]


def test_scope_selection_and_dose_mixing_are_explicit():
    rows = pool(2)
    for i, row in enumerate(rows):
        row['oligoai_evaluation'] = {'status': 'complete', 'conditions': {'dosage_nm': 10+i*90},
            'contexts': [{'transcript_id': 'TX.1', 'predicted_inhibition_percent': 70+i*20}]}
    data = index(rows)
    p = predicate(data, '/predicted_inhibition_percent', 60)
    with pytest.raises(ValueError, match='multiple conditions'):
        query(data, [p])
    scopes = data.column(p['field_id'])['scopes']
    with pytest.raises(ValueError, match='different units, doses'):
        query(data, [dict(p, scope_ids=list(scopes))])
    low = next(key for key, conditions in scopes.items() if 10 in conditions.values())
    result = query(data, [dict(p, scope_ids=[low])])
    assert result['candidate_ids'] == [molecule_id(rows[0])]
    assert result['unknown_count'] == 1


def test_filters_cannot_join_measurements_from_different_nested_records():
    rows = pool(2)
    rows[0]['future_assay'] = [{'response': 90, 'toxicity': 50}, {'response': 10, 'toxicity': 0}]
    rows[1]['future_assay'] = [{'response': 90, 'toxicity': 0}]
    data = index(rows)
    result = query(data, [predicate(data, '/response', 80),
                         predicate(data, '/toxicity', 5, operator='lte')])
    assert result['candidate_ids'] == [molecule_id(rows[1])]


def test_incomplete_numeric_measurement_stays_unknown_and_full_source_is_readable():
    rows = pool(2)
    for row, status in zip(rows, ('complete', 'failed')):
        row['future_assay'] = {'status': status, 'response': 90}
    data = index(rows)
    result = query(data, [predicate(data, '/response', 80)])
    assert result['candidate_ids'] == [molecule_id(rows[0])]
    assert result['unknown_count'] == 1
    page = data.read_evidence(molecule_id(rows[1]))
    assert page['page_count'] >= 1
    assert '90' in str(page['records'])


def test_minimum_requires_every_selected_transcript_scope():
    rows = pool(2)
    rows[0]['assay'] = [{'transcript_id': 'TX.1', 'response': 90}]
    rows[1]['assay'] = [{'transcript_id': 'TX.1', 'response': 90}, {'transcript_id': 'TX.2', 'response': 80}]
    data = index(rows)
    p = predicate(data, '/response', 70, aggregation='min')
    p['scope_ids'] = list(data.column(p['field_id'])['scopes'])
    result = query(data, [p])
    assert result['candidate_ids'] == [molecule_id(rows[1])]
    assert result['unknown_count'] == 1


def test_scalar_array_indices_keep_mismatch_bins_distinct():
    data = index(pool(2))
    assert field(data, '/counts_by_mismatch/0') != field(data, '/counts_by_mismatch/2')


def test_challenge_includes_excluded_leaders_and_unknowns():
    rows = pool(7)
    for i, row in enumerate(rows):
        row['response'] = [99, 80, 70, 50, 30, 10, None][i]
    data = index(rows)
    narrowed = query(data, [predicate(data, '/response', 80, operator='lt')])
    outside = data.outsiders(narrowed['candidate_ids'])
    assert molecule_id(rows[0]) in outside
    assert molecule_id(rows[6]) in outside


def test_comparison_coverage_tracks_returned_pages_only():
    rows = pool(7)
    data = index(rows)
    result = data.compare(list(data.ids), [], limit=2)
    assert len(data.compared_ids) == 2 and result['next_offset'] == 2
    data.survey()
    query(data, [])
    assert len(data.compared_ids) == 2


def test_ineligible_context_does_not_supply_a_passing_measurement():
    rows = pool(1)
    rows[0]['response'] = 20
    other = deepcopy(rows[0]); other['response'] = 99
    other['candidate_assessment']['selection_eligible'] = False
    data = index(rows+[other])
    assert query(data, [predicate(data, '/response', 80)])['matched_count'] == 0


def test_explicit_nested_tool_schema_preserves_existing_registration():
    from agents import function_tool, _build_function_schema
    @function_tool
    def query_tool(query):
        return query
    schema = {'type': 'object', 'properties': {'query': {'type': 'object',
        'properties': {'ids': {'type': 'array', 'items': {'type': 'string'}}},
        'required': ['ids'], 'additionalProperties': False}},
        'required': ['query'], 'additionalProperties': False}
    query_tool._tool_parameters_schema = schema
    assert _build_function_schema(query_tool)['parameters'] == schema
    @function_tool
    def previous_tool(name: str):
        return name
    assert _build_function_schema(previous_tool)['parameters']['properties']['name'] == {'type': 'string'}


def test_exists_false_finds_absent_values():
    rows = pool(2)
    rows[0]['optional_measurement'] = 0
    data = index(rows)
    assert query(data, [predicate(data, '/optional_measurement', False, operator='exists')])['candidate_ids'] == [molecule_id(rows[1])]


def test_raw_count_alias_keeps_search_scope_and_incomplete_status():
    rows = pool(2)
    for row in rows:
        row['Off_target_exact_count'] = 0
    rows[1]['candidate_assessment']['off_target']['status'] = 'incomplete_or_unavailable'
    data = index(rows)
    p = predicate(data, '/Off_target_exact_count', 0, operator='eq')
    result = query(data, [p])
    assert result['candidate_ids'] == [molecule_id(rows[0])]
    scopes = data.column(p['field_id'])['scopes']
    assert all('genes' in scope.values() and 2 in scope.values() for scope in scopes.values())


@pytest.mark.parametrize('count', [129, 4057, 10000])
def test_large_libraries_retain_known_leaders_with_paged_query_results(count):
    from test_shortlist_scaling import large_pool
    rows = large_pool(count)
    data = index(rows)
    p = predicate(data, '/selection_rank', count-10)
    result = query(data, [p], limit=3)
    assert result['matched_count'] == 11
    assert result['next_offset'] == 3
    assert len(result['candidate_ids']) == 3
    assert molecule_id(rows[-1]) in data.sets[result['set_id']]
    assert len(data._columns) == 1


def test_queries_have_no_executable_or_pattern_operators():
    data = index(pool(1))
    with pytest.raises(ValueError, match='Unknown comparison'):
        query(data, [predicate(data, '/selection_rank', '.*', operator='regex')])
    with pytest.raises(ValueError, match='Unknown candidate'):
        data.read_evidence('/etc/passwd')


def test_usage_metadata_is_bounded_and_excludes_unrelated_provider_payloads():
    from agents import _compact_history_entry
    result = _compact_history_entry({'usage': {'input_tokens': 100, 'output_tokens': 20,
        'cost': .001, 'access_token': 'private'}}, {})
    assert result['response']['usage']['input_tokens'] == 100
    assert result['response']['usage']['cost'] == .001
    assert 'private' not in str(result)


def test_csv_numeric_values_remain_queryable_and_identifiers_stay_literal():
    rows = pool(2)
    rows[0].update(future_score='0.12345', transcript_id='12345')
    rows[1].update(future_score='0.6', transcript_id='98765')
    data = index(rows)
    p = predicate(data, '/future_score', .5)
    p['scope_ids'] = list(data.column(p['field_id'])['scopes'])
    assert query(data, [p])['candidate_ids'] == [molecule_id(rows[1])]
    f = field(data, '/transcript_id')
    assert data.fields[f]['kinds'] == {'category'}
    assert data.catalog[molecule_id(rows[0])]['facts']['/contexts/0/future_score'] == '0.12345'


def test_missing_numeric_array_positions_keep_their_bin_identity():
    rows = pool(2)
    for row in rows:
        row['candidate_assessment']['off_target']['counts_by_mismatch'] = [0, None, 2]
    data = index(rows)
    p = predicate(data, '/counts_by_mismatch/1', 0, missing='only')
    assert query(data, [p])['matched_count'] == 2


def test_exists_false_finds_a_missing_requested_transcript_scope():
    rows = pool(2)
    rows[0]['assay'] = [{'transcript_id': 'TX.1', 'response': 90}]
    rows[1]['assay'] = [{'transcript_id': 'TX.1', 'response': 90}, {'transcript_id': 'TX.2', 'response': 80}]
    data = index(rows)
    p = predicate(data, '/response', False, operator='exists')
    p['scope_ids'] = list(data.column(p['field_id'])['scopes'])
    assert query(data, [p])['candidate_ids'] == [molecule_id(rows[0])]


def test_rejected_oversized_result_does_not_create_a_hidden_query_set(monkeypatch):
    from aso_agents import shortlist_query
    data = index(pool(2))
    monkeypatch.setattr(shortlist_query, 'MAX_RESULT_CHARACTERS', 1)
    with pytest.raises(ValueError, match='response limit'):
        query(data, [])
    assert set(data.sets) == {'all'}
    assert not data.operations


def test_numeric_object_keys_retain_separate_named_mismatch_bins():
    rows = pool(2)
    rows[0]['future_match_bins'] = {'0': 0, '1': 9}
    rows[1]['future_match_bins'] = {'0': 2, '1': 0}
    data = index(rows)
    assert field(data, '/future_match_bins/0') != field(data, '/future_match_bins/1')
    assert query(data, [predicate(data, '/future_match_bins/0', 0, operator='eq')])['candidate_ids'] == [molecule_id(rows[0])]
