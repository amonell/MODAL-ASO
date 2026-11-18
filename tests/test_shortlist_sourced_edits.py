"""Evidence handles, targeted edits, and independent final review."""
from copy import deepcopy

import pytest

from aso_agents import shortlist_review as review
from aso_agents.shortlist_evidence import citation, evidence_catalog, panel_binding_geometry
from aso_agents.shortlist_facts import render_phrases, sourced_audit_edit, verified_phrases
from test_shortlist_review import model, pool, run


@pytest.mark.parametrize('strand,edge', [('+', '8 nt upstream and 5 nt downstream'),
                                         ('-', '5 nt upstream and 8 nt downstream')])
def test_verified_geometry_and_counts_resolve_to_original_rows(strand, edge):
    rows = pool(2)
    for i, row in enumerate(rows):
        row['strand'] = strand
        row['source_annotations'] = {'binding_site_membership': [{'label': 'declared target',
            'genomic_segments_bed0_half_open': [['chr1', 108 + i*20, 111 + i*20, 'target', 0, strand]]}]}
    catalog = evidence_catalog(rows)
    ids = list(catalog)
    phrases = verified_phrases(catalog, ids, panel_binding_geometry(catalog, ids))
    own = {ref: fact for ref, fact in phrases.items() if fact['candidate_id'] == ids[1]}
    extension = next(ref for ref, fact in own.items() if edge in fact['text'])
    relative = next(ref for ref, fact in own.items() if 'of ASO 1' in fact['text'])
    matches = next(ref for ref, fact in own.items() if 'exactly 2 mismatches' in fact['text'])
    item = {'candidate_id': ids[1], 'why_chosen': 'This ASO ' + extension + '.', 'tradeoff': matches + '.',
            'panel_role': 'It ' + relative + '.', 'evidence_paths': [], 'comparison_evidence': []}
    assert render_phrases(item, phrases) == []
    assert item['tradeoff'] == 'The screen found 1 gene with exactly 2 mismatches.'
    assert '20 nt ' + ('downstream' if strand == '+' else 'upstream') in item['panel_role']
    assert any(path.endswith('/counts_by_mismatch/2') for path in item['evidence_paths'])
    assert any(path.endswith('/genomic_segments_bed0_half_open') for path in item['evidence_paths'])
    for pointer in item['evidence_paths']:
        assert citation(catalog[ids[1]], pointer)['source_pointer'].startswith('/rows/1/')
    assert item['comparison_evidence'] and all(ref['candidate_id'] == ids[0] for ref in item['comparison_evidence'])
    for invalid in (matches, '[[F9999]]', '[[F1', 'F1]]'):
        other = dict(item, candidate_id=ids[0], why_chosen=invalid)
        assert render_phrases(other, phrases)


def test_missing_or_incomparable_screening_stays_in_original_evidence():
    rows = pool(3)
    rows[0]['candidate_assessment']['off_target']['status'] = 'incomplete'
    rows[1]['candidate_assessment']['off_target']['counts_by_mismatch'] = [0, None, None, 10]
    rows.append({**deepcopy(rows[2]), 'program_id': 'different-context'})
    catalog = evidence_catalog(rows)
    phrases = verified_phrases(catalog, list(catalog), {})
    assert len(phrases) == 1
    assert next(iter(phrases.values()))['text'] == 'the screen found 0 genes with an exact sequence match'


@pytest.mark.parametrize('rounds', [1, 6])
def test_sourced_auditor_edit_rechecks_complete_text_and_replaces_wrong_citation(tmp_path, monkeypatch, rounds):
    monkeypatch.setattr(review, 'EDITORIAL_ROUNDS', rounds)
    path = '/contexts/0/candidate_assessment/off_target/counts_by_mismatch/'
    drafts, calls = [], []
    def reviewer(kind, payload, schema):
        calls.append(kind)
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            item = result['recommendations'][0]
            item['why_chosen'] = 'The screen found a two-mismatch match.'
            item['evidence_paths'] = [path + '1']
        if kind == 'audit':
            drafts.append(deepcopy(payload['draft']))
            item = payload['draft']['recommendations'][0]
            cid = item['candidate_id']
            if len(drafts) == 1:
                handle = next(key for key, v in payload['verified_phrases'].items()
                              if v['candidate_id'] == cid and 'exactly 2 mismatches' in v['text'])
                result['issues'] = [{'candidate_id': cid, 'problem': 'The citation points to the one-mismatch bin.',
                    'field': 'why_chosen', 'selection_impact': 'text_only', 'replacement': handle + '.',
                    'sources': [{'candidate_id': cid, 'path': path + '2'}],
                    'superseded_sources': [{'candidate_id': cid, 'path': path + '1'}]}]
                result['checks'][0]['facts_and_citations_match'] = False
                result['mismatch_count_claims'] = [{'candidate_id': cid, 'source_candidate_id': cid,
                    'context': 0, 'mismatches': 2, 'count': 2, 'field': 'why_chosen'}]
                field = next(r for r in result['field_reviews'] if r['candidate_id'] == cid and r['field'] == 'why_chosen')
                field.update(supported=False, evidence_statement='Use the two-mismatch bin.')
        return result, metadata
    monkeypatch.setattr(review, '_model_call', reviewer)
    _, _, result = run(pool(3), tmp_path, {'candidate_limit': 2})
    assert result['status'] == 'completed' and calls.count('audit') == 2
    assert 'repair' not in calls and 'selection_reconsideration' not in calls
    assert drafts[0]['recommendations'][1] == drafts[1]['recommendations'][1]
    item = drafts[1]['recommendations'][0]
    assert '[[' not in item['why_chosen'] and 'exactly 2 mismatches' in item['why_chosen']
    assert path + '1' not in item['evidence_paths'] and path + '2' in item['evidence_paths']


@pytest.mark.parametrize('defect', ['source', 'contradiction', 'selection', 'uncertain', 'missing_field', 'second_edit'])
def test_incomplete_or_conflicting_edits_require_further_review(defect):
    catalog = evidence_catalog(pool(1))
    cid = next(iter(catalog))
    draft = {'recommendations': [{'candidate_id': cid, 'tradeoff': 'Old.', 'evidence_paths': ['/contexts/0/length'],
                                 'comparison_evidence': []}]}
    field = {'candidate_id': cid, 'field': 'tradeoff', 'supported': False}
    issue = {'candidate_id': cid, 'field': 'tradeoff', 'selection_impact': 'text_only', 'replacement': 'Revised.',
             'sources': [{'candidate_id': cid, 'path': '/contexts/0/length'}]}
    audit = {'issues': [issue], 'field_reviews': [field]}
    if defect == 'source':
        issue['sources'][0]['path'] = '/contexts/0/absent'
    elif defect == 'contradiction':
        audit['field_reviews'].append({**field, 'supported': True})
    elif defect in ('selection', 'uncertain'):
        issue['selection_impact'] = 'reconsider' if defect == 'selection' else defect
    elif defect == 'missing_field':
        audit['field_reviews'].append({**field, 'field': 'panel_role'})
    else:
        audit['issues'].append({**issue, 'replacement': 'Different.'})
    assert sourced_audit_edit(draft, audit, catalog, lambda value: value) is None
    assert draft['recommendations'][0]['tradeoff'] == 'Old.'


@pytest.mark.parametrize('subject', ['owner', 'comparison'])
def test_independently_extracted_one_mismatch_claim_cannot_pass_on_a_two_mismatch_citation(subject):
    from aso_agents.shortlist_facts import mismatch_citation_findings
    rows = pool(2)
    # Deliberately identical values in two different bins and two different molecules.
    for row in rows:
        row['candidate_assessment']['off_target']['counts_by_mismatch'] = [0, 2, 2]
        row['Off_target_1mm_count'] = row['Off_target_2mm_count'] = 2
    catalog = evidence_catalog(rows)
    owner, other = list(catalog)
    source = owner if subject == 'owner' else other
    item = {'candidate_id': owner, 'tradeoff': 'Two loci have one mismatch.',
            'evidence_paths': ['/contexts/0/Off_target_2mm_count'],
            'comparison_evidence': [{'candidate_id': other, 'path': '/contexts/0/Off_target_2mm_count'}]}
    claim = {'candidate_id': owner, 'source_candidate_id': source, 'context': 0, 'mismatches': 1,
             'count': 2, 'field': 'tradeoff'}
    draft = {'recommendations': [item]}
    finding, = mismatch_citation_findings(draft, [claim], catalog)
    assert finding['selection_impact'] == 'text_only'
    assert finding['sources'] == [{'candidate_id': source, 'path': '/contexts/0/candidate_assessment/off_target/counts_by_mismatch/1'}]
    if source == owner:
        item['evidence_paths'].append('/contexts/0/Off_target_1mm_count')
    else:
        item['comparison_evidence'].extend(finding['sources'])
    assert mismatch_citation_findings(draft, [claim], catalog) == []
    assert mismatch_citation_findings(draft, [{**claim, 'count': 1}], catalog)[0]['selection_impact'] == 'uncertain'


def test_large_final_comparison_gets_the_existing_bounded_allowance_on_first_read(tmp_path, monkeypatch):
    budgets = []
    def model_call(kind, payload, schema):
        budgets.append(payload['_generation_token_budget'])
        return {'count': 1}, []
    monkeypatch.setattr(review, '_model_call', model_call)
    schema = review._object({'count': {'type': 'integer'}})
    for size in (1, review.SCREEN_PACKET_CHARACTERS):
        review._call('select', {'evidence': {'source': 'a' * size}}, schema, lambda value: [], tmp_path, {})
    assert budgets == [review.DEFAULT_OUTPUT_TOKENS, review.MAX_OUTPUT_TOKENS]


def test_terminal_audit_blocks_a_still_unsupported_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'EDITORIAL_ROUNDS', 1)
    audits = []
    def disagree(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            item = payload['draft']['recommendations'][0]
            key = item['candidate_id']
            audits.append(item['tradeoff'])
            field = next(row for row in result['field_reviews'] if row['candidate_id'] == key and row['field'] == 'tradeoff')
            field.update(supported=False, evidence_statement='The original record leaves the activity unresolved.')
            result['issues'] = [{'candidate_id': key, 'field': 'tradeoff', 'selection_impact': 'text_only',
                'problem': field['evidence_statement'], 'replacement': 'This activity remains unresolved.' if len(audits) == 1
                    else 'This revised statement still needs evidence.',
                'sources': [{'candidate_id': key, 'path': '/contexts/0/candidate_assessment/unavailable_evidence'}]}]
        return result, metadata
    monkeypatch.setattr(review, '_model_call', disagree)
    _, _, result = run(pool(2), tmp_path, {'candidate_limit': 1})
    assert len(audits) == 2 and audits[-1] == 'This activity remains unresolved.'
    assert result['status'] == 'unavailable' and result['failure_code'] == 'explanation_evidence_review_incomplete'


def test_sourced_replacement_with_poor_wording_gets_a_focused_edit_before_the_full_audit(tmp_path, monkeypatch):
    calls, repair_fields, drafts = [], [], []
    def edit(kind, payload, schema):
        calls.append(kind)
        result, meta = model(kind, payload, schema)
        if kind == 'audit':
            drafts.append(deepcopy(payload['draft']))
            if len(drafts) == 1:
                key = payload['selected_ids'][0]
                field = next(row for row in result['field_reviews'] if row['candidate_id'] == key and row['field'] == 'tradeoff')
                field.update(supported=False, evidence_statement='Describe the recorded limitation.')
                result['issues'] = [{'candidate_id': key, 'selection_impact': 'text_only', 'field': 'tradeoff',
                    'problem': 'State the actual evidence gap.', 'replacement': 'Accessibility is predicted rather than measured.',
                    'sources': [{'candidate_id': key, 'path': '/contexts/0/candidate_assessment/unavailable_evidence'}]}]
        if kind == 'repair':
            assert payload['repair_mode'] == 'wording'
            repair_fields.append(payload['repair_fields'])
            result['recommendations'][0]['tradeoff'] = 'Target RNA accessibility remains unmeasured.'
        return result, meta
    monkeypatch.setattr(review, '_model_call', edit)
    _, _, result = run(pool(3), tmp_path, {'candidate_limit': 2})
    assert result['status'] == 'completed' and len(drafts) == 2
    assert repair_fields == [{result['selected_ids'][0]: ['tradeoff']}]
    assert 'selection_reconsideration' not in calls and calls.count('repair') == 1
    assert drafts[0]['recommendations'][1] == drafts[1]['recommendations'][1]
    assert drafts[1]['recommendations'][0]['tradeoff'] == 'Target RNA accessibility remains unmeasured.'
