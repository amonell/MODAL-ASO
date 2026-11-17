"""Oversized molecular evidence is fully visited before bounded comparison."""
from collections import defaultdict
from copy import deepcopy
import json

import pytest

from aso_agents import shortlist_review as review
from aso_agents.shortlist_evidence import (citation, compact_packet, evidence_catalog,
                                         fact_value, indexed_packet, nested_packet, screening_packet)
from aso_agents.shortlist_paging import (COMPARISON_CHARACTERS, encoded_size, evidence_fragments,
                                      evidence_pages, projected_facts, review_large_entry, bundle_fragments)
from test_shortlist_review import model, pool, run


def select_evidence(kind, payload, schema):
    if kind == 'evidence_audit':
        return {'reviewed_count': len(payload['records']), 'issues': []}, []
    if kind != 'evidence':
        return model(kind, payload, schema)
    # Retain source-backed fields for the ordinary explanation fixture while
    # still visiting every record, including large unknown future measurements.
    prioritized = sorted(payload['records'], key=lambda item: (
        'complete_within_declared_scope' not in str(item),
        'target_RNA_accessibility' not in str(item), item['id']))
    return {'reviewed_count': len(payload['records']),
            'selected_ids': [item['id'] for item in prioritized[:payload['selection_limit']]]}, []


def large_rows(count=3):
    rows = pool(count)
    for index, row in enumerate(rows):
        row['future_assay'] = [{'transcript': f'TX.{i}', 'response': -(i + index) / 1000007,
                               'condition': f'condition {i}'} for i in range(900)]
        row['match_identifiers'] = [f'GENE.{i}' for i in range(2400)]
        row['long_text_evidence'] = 'recorded context: α; newline\n' * 700
    return rows


def test_fragments_roundtrip_preserves_precision_order_missingness_and_unicode():
    facts = {'/contexts/0/text': 'quoted " α 😀\n' * 300,
             '/contexts/0/matches': [None, 'duplicate', 'duplicate', -0.123456789123] * 200,
             '/contexts/0/empty': [], '/contexts/0/missing': None,
             '/contexts/0/measurement': 0.12345678912345678}
    before = deepcopy(facts)
    fragments = evidence_fragments(facts, limit=400)
    assert facts == before
    assert all(encoded_size(record) <= 400 for record in fragments)
    assert [record['id'] for record in fragments] == list(range(len(fragments)))
    grouped = defaultdict(list)
    for record in fragments:
        grouped[record['path']].append(record)
    for path, original in facts.items():
        parts = grouped[path]
        if 'kind' not in parts[0]:
            assert parts[0]['value'] == original
        else:
            assert parts[0]['start'] == 0 and parts[-1]['stop'] == len(original)
            assert all(a['stop'] == b['start'] for a, b in zip(parts, parts[1:]))
            rebuilt = ''.join(part['value'] for part in parts) if isinstance(original, str) else [
                value for part in parts for value in part['value']]
            assert rebuilt == original
    assert projected_facts(fragments, facts) == facts


def test_pages_cover_every_fragment_once_and_obey_encoded_budget():
    records = evidence_fragments({'/contexts/0/text': 'α' * 6000}, limit=800)
    pages = evidence_pages(records, limit=2100)
    assert len(pages) > 1
    assert [record for page in pages for record in page] == records
    assert all(encoded_size(page) <= 2100 for page in pages)


def test_nested_large_array_items_keep_exact_resolvable_source_paths():
    entry = {'facts': {'/contexts/0/observations': [{'a/b~': ['long measured text' * 100]}]}}
    records = evidence_fragments(entry['facts'], limit=400)
    assert records
    for item in records:
        source = fact_value(entry, item['path'])
        expected = source[item['start']:item['stop']] if 'kind' in item else source
        assert expected == item['value']


def test_partial_values_are_explicit_excerpts_and_citations_keep_original():
    entry = next(iter(evidence_catalog(large_rows(1)).values()))
    fragments = evidence_fragments(entry['facts'])
    chosen = [item for item in fragments if item['path'].endswith('long_text_evidence')][:1]
    view = projected_facts(chosen, entry['facts'])
    path = chosen[0]['path']
    assert view[path]['representation'] == 'selected_source_excerpts'
    assert view[path]['source_length'] == len(entry['facts'][path])
    assert citation(entry, path)['value'] == entry['facts'][path]
    assert view[path]['slices'][0]['value'] == entry['facts'][path][:chosen[0]['stop']]


def test_source_pages_all_reviewed_and_projection_bounded():
    entry = next(iter(evidence_catalog(large_rows(1)).values()))
    before = deepcopy(entry)
    calls = []
    def call(kind, payload, schema, check, directory, configuration):
        response, _ = select_evidence(kind, payload, schema)
        assert check(response) == []
        calls.append(payload)
        return response
    result = review_large_entry(entry, {}, None, {}, call, page_limit=12000, comparison_limit=6000)
    source = [record for payload in calls if payload['review_scope']['phase'] == 'source'
              for record in payload['records']]
    assert source == bundle_fragments(evidence_fragments(entry['facts']))
    assert result['evidence_review']['source_fragment_count'] == len(source)
    assert result['evidence_review']['source_pages_reviewed'] == sum(
        payload['review_scope']['phase'] == 'source' for payload in calls)
    assert all(encoded_size(payload['records']) <= 12000 for payload in calls)
    assert encoded_size(result['review_facts']) <= 6000
    assert entry == before and result['facts'] == entry['facts']
    for packet in (compact_packet([result]), screening_packet([result]), nested_packet([result]), indexed_packet([result])):
        assert packet['source_page_reviews'][0]['source_fragment_count'] == len(source)
        assert 'not the complete source record' in packet['source_page_interpretation']
    assert result['facts'] == before['facts']


def test_large_context_overviews_do_not_reenter_bounded_comparison_packets():
    entry = next(iter(evidence_catalog(large_rows(1)).values()))
    entry['review_facts'] = {'/contexts/0/length': 16}
    entry['evidence_review'] = {'method': 'paged_source_evidence_selection',
        'source_sha256': 'fixture', 'source_fact_count': len(entry['facts']),
        'source_fragment_count': 100, 'source_pages_reviewed': 10, 'selected_fragment_count': 1}
    entry['overview'] = [{'context': index, 'recorded_limits': 'context details ' * 1000}
                         for index in range(100)]
    before = deepcopy(entry)
    compact = compact_packet([entry])
    for packet in (compact, screening_packet([entry]), nested_packet([entry]),
                   indexed_packet([entry]), indexed_packet([entry], packet=compact)):
        assert packet['source_page_overviews'] == []
        assert encoded_size(packet) < 10000
    assert entry == before


def test_bundles_preserve_all_source_facts_and_reduce_repeated_paths():
    entry = next(iter(evidence_catalog(large_rows(1)).values()))
    fragments = evidence_fragments(entry['facts'])
    bundled = bundle_fragments(fragments)
    assert encoded_size(bundled) < encoded_size(fragments) * .75
    assert projected_facts(bundled, entry['facts']) == entry['facts']


@pytest.mark.parametrize('bad', ['count', 'foreign_id', 'duplicate', 'quota'])
def test_incomplete_or_invalid_source_review_cannot_publish_recommendations(tmp_path, monkeypatch, bad):
    def invalid(kind, payload, schema):
        response, metadata = select_evidence(kind, payload, schema)
        if kind == 'evidence':
            if bad == 'count':
                response['reviewed_count'] -= 1
            elif bad == 'foreign_id':
                response['selected_ids'] = [100000000]
            elif bad == 'duplicate':
                response['selected_ids'] = [payload['records'][0]['id']] * 2
            else:
                response['selected_ids'] = [item['id'] for item in payload['records']]
        return response, metadata
    monkeypatch.setattr(review, '_model_call', invalid)
    _, _, receipt = run(large_rows(1), tmp_path)
    assert receipt['status'] == 'unavailable'
    assert receipt['reviewed_molecule_count'] == 0
    assert receipt['failure_code'] == 'evidence_review_incomplete'


def test_oversized_records_finish_selection_explanation_and_audit(tmp_path, monkeypatch):
    calls = []
    def recording(kind, payload, schema):
        calls.append((kind, deepcopy(payload)))
        return select_evidence(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', recording)
    rows = large_rows(4)
    before = deepcopy(rows)
    selected, _, receipt = run(rows, tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed', receipt
    assert len(selected) == 3 and receipt['reviewed_molecule_count'] == 4
    assert rows == before
    assert len(receipt['source_evidence_reviews']) == 4
    assert {'evidence', 'rank', 'explain', 'audit', 'evidence_audit'} <= {kind for kind, _ in calls}
    for kind, payload in calls:
        if 'evidence' in payload:
            assert encoded_size(payload['evidence']) <= review.MAX_PACKET_CHARACTERS
    for item in receipt['recommendations'].values():
        assert item['evidence'][0]['value'] == 'complete_within_declared_scope'
        assert item['evidence'][0]['source_pointer'].startswith('/rows/')
    for key in receipt['selected_ids']:
        checked = receipt['source_evidence_reviews'][key]
        assert checked['explanation_audit']['source_pages_checked'] == checked['source_pages_reviewed']
    saved = json.loads((tmp_path / 'recommendation_review.json').read_text())
    assert saved['result']['source_evidence_reviews'] == receipt['source_evidence_reviews']
    def offline(*args):
        raise AssertionError('A completed review must reopen without more model calls')
    monkeypatch.setattr(review, '_model_call', offline)
    _, _, reopened = run(rows, tmp_path, {'candidate_limit': 3})
    assert reopened['status'] == 'completed'


def test_ordinary_reviews_keep_original_call_sequence(tmp_path, monkeypatch):
    calls = []
    def recording(kind, payload, schema):
        calls.append(kind)
        return select_evidence(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', recording)
    _, _, receipt = run(pool(4), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed'
    assert calls == ['rank', 'explain', 'audit']
    assert receipt['source_evidence_reviews'] == {}


def test_final_source_audit_repairs_counterevidence_omitted_from_comparison(tmp_path, monkeypatch):
    rows = large_rows(1)
    rows[0]['late_counterevidence'] = {'response': -0.99, 'condition': 'opposing perturbation'}
    calls = []
    def counterevidence(kind, payload, schema):
        calls.append(kind)
        response, metadata = select_evidence(kind, payload, schema)
        if kind == 'explain':
            response['recommendations'][0]['why_chosen'] = 'Every recorded assay supports the desired change.'
        if kind == 'evidence_audit' and payload['draft']['why_chosen'].startswith('Every recorded assay'):
            record = next((item for item in payload['records'] if 'late_counterevidence' in str(item)), None)
            if record:
                response['issues'] = [{'field': 'why_chosen', 'record_ids': [record['id']],
                    'problem': 'The saved late counterevidence records an opposing response of -0.99.'}]
        if kind == 'repair':
            assert payload['source_page_findings']
            item = response['recommendations'][0]
            item['why_chosen'] = 'A recorded assay shows an opposing response.'
            item['evidence_paths'].append('/contexts/0/late_counterevidence/response')
        return response, metadata
    monkeypatch.setattr(review, '_model_call', counterevidence)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 1})
    assert receipt['status'] == 'completed', receipt
    assert 'repair' in calls
    item = next(iter(receipt['recommendations'].values()))
    assert item['why_chosen'] == 'A recorded assay shows an opposing response.'
    assert any(ref['value'] == -0.99 for ref in item['evidence'])
    assert all(part['issue_count'] == 0 for part in next(iter(receipt['source_evidence_reviews'].values()))[
        'explanation_audit']['passes'])
