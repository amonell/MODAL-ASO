"""Large libraries keep complete evidence with bounded nomination output."""
from copy import deepcopy
import json

from aso_agents import shortlist_review as review
from aso_agents.shortlist_evidence import evidence_catalog, screening_packet, molecule_id
from test_shortlist_review import pool, model, run

REAL_MODEL_CALL = review._model_call


def large_pool(count):
    rows = pool(count)
    for i, row in enumerate(rows):
        row['synthesized_antisense_sequence'] = 'ACGTACGT' + ''.join('ACGT'[(i // 4**j) % 4] for j in range(8))
    return rows


def nomination_model(kind, payload, schema):
    if kind == 'nominate':
        return {'reviewed_count': len(payload['evidence']['ids']),
                'nominated_ids': list(reversed(payload['evidence']['ids']))[:payload['promotion_limit']],
                'summary': 'These candidates retain useful binding regions and recorded screening evidence.'}, []
    if kind == 'select':
        return {'reviewed_count': len(payload['evidence']['ids']),
                'nominated_ids': list(reversed(payload['evidence']['ids']))[:payload['promotion_limit']],
                'suggested_size': min(3, len(payload['evidence']['ids'])),
                'summary': 'These candidates cover useful binding regions with recorded screening evidence.'}, []
    return model(kind, payload, schema)


def expand_packet(packet):
    def expand(value):
        if isinstance(value, dict) and set(value) == {'value_ref'}:
            return packet['shared_values'][value['value_ref']]
        return value
    shared = {path: expand(fact['value']) for fact in packet['shared_facts'] for path in fact['paths']}
    rows = {key: deepcopy(shared) for key in packet['ids']}
    for row in packet['rows']:
        for label, value in row['values'].items():
            paths = packet['fields'][label]
            for path in paths:
                rows[row['id']][path] = expand(value)
    return rows


def test_screening_roundtrip_preserves_every_fact_context_and_missing_value():
    rows = pool(6)
    rows[0]['new_measurement'] = 0.123456789
    rows[1]['new_measurement'] = None
    for row in rows:
        row['future_chemistry'] = ['DNA', 'MOE'] * 20
        row['future_activity_conditions'] = {'dose': 50, 'units': 'nM', 'delivery': 'transfection'}
    rows.append({**deepcopy(rows[0]), 'program_id': 'other', 'new_measurement': -0.75})
    entries = list(evidence_catalog(rows).values())
    before = deepcopy(entries)
    packet = screening_packet(entries, include_provenance=True)
    assert expand_packet(packet) == {i: entry['facts'] for i, entry in enumerate(entries)}
    assert packet['shared_values']
    assert entries == before


def test_nomination_keeps_scientific_fields_and_future_measurements():
    rows = pool(2)
    for row in rows:
        row['source_path'] = '/fixture/data.json'
        row['oligoai_evaluation'] = {'contexts': [{'rna_context': 'ACGT' * 25,
            'predicted_inhibition_percent': 83.123, 'species': 'human', 'transcript_id': 'TX.1'}],
            'conditions': {'dosage_nm': 50, 'transfection_method': 'Lipofection'}}
        row['new_future_assay'] = {'conditions': {'dose_nm': 12}, 'response': -0.75}
    entries = list(evidence_catalog(rows).values())
    facts = expand_packet(screening_packet(entries))[0]
    assert facts['/contexts/0/oligoai_evaluation/contexts/0/predicted_inhibition_percent'] == 83.123
    assert facts['/contexts/0/oligoai_evaluation/conditions/dosage_nm'] == 50
    assert facts['/contexts/0/new_future_assay/response'] == -0.75
    assert facts['/contexts/0/candidate_assessment/off_target/status'] == 'complete_within_declared_scope'
    assert '/contexts/0/source_path' not in facts
    assert '/contexts/0/oligoai_evaluation/contexts/0/rna_context' not in facts
    assert '/contexts/0/oligoai_evaluation/contexts/0/rna_context' in entries[0]['facts']


def test_species_scope_chemistry_and_unknown_measurements_remain_visible():
    rows = pool(2)
    for row in rows:
        row['off_target_by_species'] = {'human': {'status': 'complete', 'off_target_gene_ids': ['GENE1'],
            'off_target_count_scope': {'assembly': 'GRCh38', 'limitations': 'Exon and adjacent junction scope'}},
            'mouse': {'status': 'incomplete', 'new_species_measurement': 0.75}}
        row['molecule_specification'] = {'custom_linkage_measurement': 0.123,
            'nucleotide_chemistry_5to3': ['custom'] * 16}
    facts = expand_packet(screening_packet(list(evidence_catalog(rows).values())))[0]
    assert facts['/contexts/0/off_target_by_species/mouse/status'] == 'incomplete'
    assert facts['/contexts/0/off_target_by_species/mouse/new_species_measurement'] == 0.75
    assert facts['/contexts/0/off_target_by_species/human/off_target_gene_ids'] == ['GENE1']
    assert facts['/contexts/0/off_target_by_species/human/off_target_count_scope/assembly'] == 'GRCh38'
    assert facts['/contexts/0/molecule_specification/nucleotide_chemistry_5to3'] == ['custom'] * 16


def test_only_available_chemical_encoding_and_new_assessment_fields_survive():
    rows = pool(2)
    rows[1]['IDT'] = 'different-chemistry'
    for row in rows:
        row['candidate_assessment']['off_target']['species_evidence'] = {'human': {'new_field': 'important result'}}
    facts = expand_packet(screening_packet(list(evidence_catalog(rows).values())))
    assert facts[0]['/contexts/0/IDT'] == 'fixture-chemistry'
    assert facts[1]['/contexts/0/IDT'] == 'different-chemistry'
    assert facts[0]['/contexts/0/candidate_assessment/off_target/species_evidence/human/new_field'] == 'important result'


def test_supported_audit_fields_can_omit_duplicate_prose(tmp_path, monkeypatch):
    def concise(kind, payload, schema):
        response, metadata = nomination_model(kind, payload, schema)
        if kind == 'audit':
            for field in response['field_reviews']:
                field['evidence_statement'] = None
        return response, metadata
    monkeypatch.setattr(review, '_model_call', concise)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed'


def test_panel_audits_run_concurrently_with_full_comparison_context(tmp_path, monkeypatch):
    from threading import Barrier
    barrier = Barrier(2, timeout=5)
    audits = []
    def parallel(kind, payload, schema):
        if kind == 'audit':
            audits.append(payload)
            barrier.wait()
        return nomination_model(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', parallel)
    selected, _, receipt = run(pool(6), tmp_path, {'candidate_limit': 6})
    assert receipt['status'] == 'completed' and len(selected) == 6
    assert len(audits) == 2
    ids = [key for payload in audits for key in payload['selected_ids']]
    assert len(ids) == len(set(ids)) == 6
    assert set(ids) == set(receipt['selected_ids'])
    assert sum(payload['audit_scope']['include_panel_fields'] for payload in audits) == 1
    assert all(len(payload['complete_panel']) == 6 and len(payload['draft']['recommendations']) == 6
               and len(payload['evidence']['ids']) == 6 for payload in audits)
    expected_facts = {entry['id']: entry['facts'] for entry in evidence_catalog(pool(6)).values()}
    assert all(expand_packet(payload['evidence']) == expected_facts for payload in audits)
    checkpoint = json.loads(next((tmp_path/'recommendation_review/editorial').glob('*.json')).read_text())
    assert set(checkpoint['audit']['reviewed_ids']) == set(ids)
    assert len(checkpoint['audit']['field_reviews']) == 20
    assert checkpoint['audit']['issues'] == []
    for path in (tmp_path/'recommendation_review/editorial').glob('*.json'):
        path.unlink()
    saved_path = tmp_path/'recommendation_review.json'
    saved = json.loads(saved_path.read_text())
    saved['input_sha256'] = 'earlier-source'
    saved_path.write_text(json.dumps(saved))
    def offline(*args):
        raise AssertionError('Accepted scoped audit components should reopen offline')
    monkeypatch.setattr(review, '_model_call', offline)
    _, _, resumed = run(pool(6), tmp_path, {'candidate_limit': 6})
    assert resumed['status'] == 'completed'


def test_missing_field_in_one_parallel_audit_blocks_panel_approval(tmp_path, monkeypatch):
    def incomplete(kind, payload, schema):
        response, metadata = nomination_model(kind, payload, schema)
        if kind == 'audit' and payload['audit_scope']['include_panel_fields']:
            key = payload['selected_ids'][0]
            response['field_reviews'] = [item for item in response['field_reviews']
                if (item['candidate_id'], item['field']) != (key, 'why_chosen')]
        return response, metadata
    monkeypatch.setattr(review, '_model_call', incomplete)
    selected, _, receipt = run(pool(6), tmp_path, {'candidate_limit': 6})
    assert len(selected) == 6 and receipt['status'] == 'unavailable'
    assert receipt['failure_code'] == 'audit_review_incomplete'
    assert receipt['reviewed_molecule_count'] == 0


def test_selectivity_uncertainty_is_checked_semantically(tmp_path, monkeypatch):
    calls = []
    def uncertainty(kind, payload, schema):
        calls.append(kind)
        response, metadata = nomination_model(kind, payload, schema)
        if kind == 'explain':
            response['summary'] = 'These ASOs target the requested isoform; cellular isoform selectivity remains unresolved.'
        return response, metadata
    monkeypatch.setattr(review, '_model_call', uncertainty)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed' and calls == ['rank', 'explain', 'audit']


def test_host_findings_populate_null_audit_observations_without_extra_generation(tmp_path, monkeypatch):
    calls = []
    def wording(kind, payload, schema):
        calls.append(kind)
        response, metadata = nomination_model(kind, payload, schema)
        if kind == 'explain':
            response['recommendations'][0]['panel_role'] = 'It is the second ASO in this panel.'
        if kind == 'audit':
            for field in response['field_reviews']:
                field['evidence_statement'] = None
            if payload['draft']['recommendations'][0]['panel_role'] == 'It is the second ASO in this panel.':
                response['quantitative_comparisons'] = [{'candidate_id': payload['selected_ids'][0],
                    'field': 'panel_role', 'path': '/panel_order', 'order': 'ascending',
                    'claimed_rank': 2, 'scope_ids': payload['selected_ids']}]
        if kind == 'repair':
            for item in response['recommendations']:
                item['panel_role'] = 'It is the first ASO in this panel.'
        return response, metadata
    monkeypatch.setattr(review, '_model_call', wording)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed'
    assert calls == ['rank', 'explain', 'audit', 'repair', 'audit']


def test_unsupported_audit_field_still_requires_a_specific_finding(tmp_path, monkeypatch):
    count = []
    def audit(kind, payload, schema):
        response, metadata = nomination_model(kind, payload, schema)
        if kind == 'audit':
            count.append(payload)
            if len(count) == 1:
                field = response['field_reviews'][0]
                field.update(supported=False, evidence_statement=None)
                response['issues'] = [{'candidate_id': field['candidate_id'], 'problem': 'A correction needs supporting evidence.'}]
        return response, metadata
    monkeypatch.setattr(review, '_model_call', audit)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed' and len(count) == 2
    assert any('concrete evidence_statement' in item for item in count[1]['corrections_needed'])


def test_four_thousand_candidates_are_reviewed_with_bounded_outputs(tmp_path, monkeypatch):
    calls = []
    def recorded(kind, payload, schema):
        calls.append((kind, payload))
        return nomination_model(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', recorded)
    rows = large_pool(4057)
    selected, _, receipt = run(rows, tmp_path, {'candidate_limit': 4})
    assert receipt['status'] == 'completed'
    assert receipt['reviewed_molecule_count'] == 4057
    assert len(selected) == 4
    assert set(receipt['ranked_ids']) == {molecule_id(row) for row in rows}
    first = receipt['review_levels'][0]
    reviewed_ids = [key for batch in first['reviews'] for key in batch['input_ids']]
    assert len(reviewed_ids) == 4057 and len(set(reviewed_ids)) == 4057
    assert len(calls) < 80
    assert all(len(payload['evidence']['ids']) <= review.SCREEN_BATCH_MOLECULES
               and payload['promotion_limit'] <= 24 for kind, payload in calls if kind == 'nominate')
    assert all(len(payload['evidence']['ids']) <= review.SCREEN_THRESHOLD
               for kind, payload in calls if kind == 'rank')
    assert 'remaining reviewed candidates retain input order' in receipt['ordering_scope']
    progress = json.loads((tmp_path/'recommendation_review/progress.json').read_text())
    assert progress['phase'] == 'completed' and progress['reviewed'] == 4057


def test_invalid_nominees_are_repaired_before_promotion(tmp_path, monkeypatch):
    attempts = []
    def invalid_once(kind, payload, schema):
        response, metadata = nomination_model(kind, payload, schema)
        if kind == 'nominate':
            attempts.append(payload)
            if len(attempts) == 1:
                response['reviewed_count'] = 1
                response['nominated_ids'][0] = len(payload['evidence']['ids'])
        return response, metadata
    monkeypatch.setattr(review, '_model_call', invalid_once)
    _, _, receipt = run(large_pool(129), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed' and len(attempts) == 4
    assert any('corrections_needed' in attempt for attempt in attempts)


def test_fitting_finalists_skip_another_nomination_round(tmp_path, monkeypatch):
    calls = []
    def recorded(kind, payload, schema):
        calls.append((kind, payload))
        return nomination_model(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', recorded)
    _, _, receipt = run(large_pool(2000), tmp_path, {'candidate_limit': 6})
    assert receipt['status'] == 'completed'
    assert receipt['reviewed_molecule_count'] == 2000
    assert [level['method'] for level in receipt['review_levels']] == ['nomination', 'final_selection']
    finalist_calls = [payload for kind, payload in calls if kind == 'select']
    assert len(finalist_calls) == 1
    assert review.SCREEN_THRESHOLD < len(finalist_calls[0]['evidence']['ids']) <= review.FINAL_BATCH_MOLECULES


def test_incompatible_batches_can_nominate_none_without_losing_coverage(tmp_path, monkeypatch):
    calls = []
    def incompatible(kind, payload, schema):
        calls.append(kind)
        response, metadata = nomination_model(kind, payload, schema)
        if kind == 'nominate':
            response['nominated_ids'] = []
            response['summary'] = 'The supplied binding sites fall outside the explicitly required target region.'
        return response, metadata
    monkeypatch.setattr(review, '_model_call', incompatible)
    rows = large_pool(129)
    selected, _, receipt = run(rows, tmp_path)
    assert selected == [] and receipt['status'] == 'completed'
    assert receipt['reviewed_molecule_count'] == 129
    assert set(receipt['ranked_ids']) == {molecule_id(row) for row in rows}
    assert len(receipt['review_levels']) == 1
    assert calls.count('nominate') == 3
    assert 'select' not in calls and 'rank' not in calls
    assert calls[-2:] == ['explain', 'audit']


def test_nomination_outage_stops_after_current_wave(tmp_path, monkeypatch):
    calls = []
    def unavailable(kind, payload, schema):
        calls.append((kind, payload['evidence']['ids']))
        raise ConnectionError('fixture transport failure')
    monkeypatch.setattr(review, '_model_call', unavailable)
    monkeypatch.setattr(review, 'SCREEN_BATCH_MOLECULES', 32)
    _, _, receipt = run(large_pool(1000), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'unavailable'
    assert 0 < len(calls) <= review.SCREEN_WORKERS * review.ATTEMPTS
    assert {kind for kind, _ in calls} == {'nominate'}


def test_nomination_token_limit_splits_instead_of_repeating_a_large_generation(tmp_path, monkeypatch):
    sizes = []
    def bounded(kind, payload, schema):
        if kind == 'nominate':
            size = len(payload['evidence']['ids'])
            sizes.append((size, payload['_generation_token_budget']))
            if size > 32:
                raise review.ReviewTokenLimit('fixture budget exhausted')
        return nomination_model(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', bounded)
    rows = large_pool(129)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed' and receipt['reviewed_molecule_count'] == 129
    assert all(budget == review.NOMINATION_OUTPUT_TOKENS for _, budget in sizes)
    assert sum(size == 64 for size, _ in sizes) == 2
    ids = [key for batch in receipt['review_levels'][0]['reviews'] for key in batch['input_ids']]
    assert len(ids) == 129 and len(set(ids)) == 129
    # A restarted review reuses accepted sub-batches and the recorded split.
    sizes.clear()
    review._nominate(list(evidence_catalog(rows).values()), review.review_context({'candidate_limit': 3}),
        tmp_path, receipt['reviewer'])
    assert sizes == []


def test_nomination_workers_keep_run_trace_and_preserve_full_pool_for_constraints(tmp_path, monkeypatch):
    from aso_agents.telemetry import evaluation_trace, current_evaluation_trace
    seen = []
    def traced(kind, payload, schema):
        seen.append(current_evaluation_trace().trace_id)
        return nomination_model(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', traced)
    rows = large_pool(150)
    # Nominees are later rows; the custom hard constraint needs the first row.
    def selector(ordered, goal):
        return [next(row for row in ordered if row['selection_rank'] == 1)], {'status': 'complete'}
    with evaluation_trace({'case': 'nomination coverage'}) as trace:
        selected, _, receipt = review.recommend_panel(rows, {'candidate_limit': 1}, tmp_path,
            select_panel=selector, allow_model_calls=True)
    assert receipt['status'] == 'completed'
    assert selected[0]['selection_rank'] == 1
    assert set(seen) == {trace.trace_id}


def test_nomination_model_settings_use_bounded_reasoning_and_minified_input(monkeypatch):
    from types import SimpleNamespace
    from agents import Runner
    seen = []
    def response(agent, input, **kwargs):
        seen.append((agent.model_settings, input))
        return SimpleNamespace(final_output='{}', responses=[])
    monkeypatch.setattr(Runner, '_run_sync', response)
    monkeypatch.setenv('ASO_LLM_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_MODEL', 'deepseek/deepseek-v4.1-flash-1017')
    monkeypatch.delenv('OPENROUTER_REASONING_EFFORT', raising=False)
    REAL_MODEL_CALL('nominate', {'evidence': {'ids': [0, 1]}, '_generation_token_budget': 8192}, {})
    REAL_MODEL_CALL('audit', {}, {})
    assert seen[0][0].reasoning == {'effort': 'low'}
    assert seen[1][0].reasoning == {'effort': 'high'}
    assert seen[0][0].max_output_tokens == 8192
    assert seen[0][1] == '{"evidence":{"ids":[0,1]}}'


def test_wire_ids_roundtrip_without_rewriting_evidence_or_user_settings(monkeypatch):
    from types import SimpleNamespace
    from agents import Runner
    import aso_agents.llm_provider as provider
    monkeypatch.setenv('ASO_LLM_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_MODEL', 'deepseek/deepseek-v4.1-flash-1017')
    monkeypatch.setenv('OPENROUTER_REASONING_EFFORT', 'high')
    keys = ['aso_1234567890abcdef', 'aso_abcdef1234567890']
    payload = {'evidence': {'ids': keys, 'rows': [[keys[0], 0.2], [keys[1], 0.5]]}}
    before = deepcopy(payload)
    def response(agent, input, **kwargs):
        wire = json.loads(input)
        assert wire['evidence']['ids'] == ['m0', 'm1']
        assert wire['evidence']['rows'] == [['m0', 0.2], ['m1', 0.5]]
        assert agent.model_settings.reasoning == {'effort': 'high'}
        return SimpleNamespace(final_output=json.dumps({'ordered_ids': ['m1', 'm0'],
            'preferred_ids': ['m1'], 'suggested_size': 1, 'summary': 'm1 leads m0; m10 is an unrelated label.'}), responses=[])
    monkeypatch.setattr(Runner, '_run_sync', response)
    result, _ = REAL_MODEL_CALL('rank', payload, review.RANK_SCHEMA)
    assert result['ordered_ids'] == list(reversed(keys))
    assert result['summary'] == keys[1] + ' leads ' + keys[0] + '; m10 is an unrelated label.'
    assert payload == before
    assert review._review_configuration(provider.get_provider_config())['screening_reasoning_effort'] == 'high'
