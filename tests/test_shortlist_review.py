"""Adversarial recommendation responses and real typed panel selection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aso_agents import shortlist_review as review
from aso_agents.shortlist_evidence import evidence_catalog, molecule_id


def pool(count=7):
    return [{'synthesized_antisense_sequence': 'ACGTACGTACGT' + ''.join('ACGT'[(i//4**j)%4] for j in range(4)),
        'IDT': 'fixture-chemistry', 'sequence': 'ACGTACGTACGTACGT',
        'selection_rank': i + 1, 'length': 16, 'program_id': 'p',
        'binding_intervals_bed0_half_open': [['chr1', 100+i*20, 116+i*20]],
        'candidate_assessment': {'selection_eligible': True, 'recommendation_status': 'provisional',
            'off_target': {'status': 'complete_within_declared_scope', 'counts_by_mismatch': [0, 0, i],
                'mismatch_budget': 2, 'unit': 'genes'}, 'unavailable_evidence': ['target_RNA_accessibility']}}
        for i in range(count)]


def selector(directory):
    from aso_agents.pipeline_tools import _select_generated_design_panel
    return lambda rows, goal: _select_generated_design_panel(rows, goal, directory=directory)


def model(kind, payload, schema):
    if kind == 'ranking_intent':
        return {'fewer_isoform_matches_used_as_preference': 'without weighting' not in payload['summary'],
                'reason': 'The stated ranking criterion was checked.'}, []
    if kind == 'summarize':
        return {'summary': 'Together these ASOs cover the requested binding regions.',
                'size_reason': 'The complete panel follows the requested count.'}, []
    if kind == 'audit_summary':
        return {'supported': True, 'issues': []}, []
    if kind == 'rank':
        ids = list(reversed(payload['evidence']['ids']))
        return {'ordered_ids': ids, 'preferred_ids': ids[:payload['promotion_limit']],
                'suggested_size': min(3, len(ids)), 'summary': 'Three candidates cover complementary binding sites.'}, []
    if kind == 'repair':
        previous = payload['previous_draft']
        return {'summary': previous['summary'] if payload['repair_panel'] else None,
                'size_reason': previous['size_reason'] if payload['repair_panel'] else None,
                'recommendations': [dict(item) for item in previous['recommendations']
                                    if item['candidate_id'] in payload['repair_candidate_ids']]}, []
    if kind == 'explain':
        return {'summary': 'This panel covers distinct binding sites.', 'size_reason': 'The panel follows the requested count.',
            'recommendations': [{'candidate_id': key, 'why_chosen': 'This ASO has complete scoped off-target evidence.',
                'tradeoff': 'Target RNA accessibility remains unresolved.', 'panel_role': 'Adds another recorded binding site.',
                'evidence_paths': ['/contexts/0/candidate_assessment/off_target/status',
                                   '/contexts/0/candidate_assessment/unavailable_evidence'],
                'comparison_evidence': []} for key in payload['selected_ids']]}, []
    return {'reviewed_ids': list(payload['selected_ids']),
            'quantitative_comparisons': [],
            'checks': [{'candidate_id': key, **{check: True for check in review.AUDIT_CHECKS}}
                       for key in payload['selected_ids']], 'issues': [],
            'field_reviews': [{'candidate_id': key, 'field': field, 'supported': True,
                              'evidence_statement': 'The recorded off-target search covers two mismatches.'}
                             for key, fields in [*[(key, ('why_chosen', 'tradeoff', 'panel_role'))
                                                  for key in payload['selected_ids']],
                                                 ('panel', ('summary', 'size_reason'))]
                             for field in fields]}, []


@pytest.fixture
def fake_model(monkeypatch):
    calls = []
    def call(*args):
        calls.append(args)
        return model(*args)
    monkeypatch.setattr(review, '_model_call', call)
    return calls


def run(rows, tmp_path, goal=None, **kwargs):
    return review.recommend_panel(rows, goal or {}, tmp_path, select_panel=selector(tmp_path),
                                  allow_model_calls=True, **kwargs)


def test_model_changes_membership_preserves_pool_and_exports_sources(tmp_path, fake_model):
    rows = pool(); before = deepcopy(rows)
    selected, selection, receipt = run(rows, tmp_path, {'candidate_limit': 3})
    assert [r['selection_rank'] for r in selected] == [7, 6, 5]
    assert receipt['status'] == 'completed'
    assert receipt['reviewed_molecule_count'] == 7
    assert rows == before
    assert [call[0] for call in fake_model] == ['rank', 'explain', 'audit']
    for entry in receipt['recommendations'].values():
        assert entry['evidence'][0]['value'] == 'complete_within_declared_scope'
        assert entry['evidence'][0]['source_pointer'].startswith('/rows/')
    assert all('_recommendation_order' not in row for row in selected)


def test_legacy_metric_labels_are_defined_before_evidence_audit(tmp_path, monkeypatch):
    def legacy_label(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            result['recommendations'][0]['tradeoff'] = 'Expression-weighted off-target risk is 0.25 in blood; the risk of an unintended RNA effect remains unresolved.'
        if kind == 'audit':
            assert payload['draft']['recommendations'][0]['tradeoff'] == 'Weighted expression of matching RNAs is 0.25 in blood; the risk of an unintended RNA effect remains unresolved.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', legacy_label)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert 'Weighted expression of matching RNAs' in receipt['recommendations'][receipt['selected_ids'][0]]['tradeoff']


def test_conservation_metric_uses_plain_label_before_evidence_audit(tmp_path, monkeypatch):
    def named_metric(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            result['recommendations'][0]['why_chosen'] = 'Predicted activity is 76.92%, with phastcons of 0.6942.'
        if kind == 'audit':
            assert payload['draft']['recommendations'][0]['why_chosen'] == (
                'Predicted activity is 76.92%, with sequence conservation score of 0.6942.')
        return result, metadata
    monkeypatch.setattr(review, '_model_call', named_metric)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert 'sequence conservation score of 0.6942' in receipt['recommendations'][receipt['selected_ids'][0]]['why_chosen']


def test_automatic_size_does_not_alter_request(tmp_path, fake_model):
    goal = {'candidate_minimum': 2}; before = deepcopy(goal)
    selected, _, receipt = run(pool(), tmp_path, goal)
    assert len(selected) == 3 and goal == before
    assert receipt['suggested_size'] == 3


def test_plural_legacy_metric_names_keep_correct_grammar_and_values():
    assert review._metric_labels('Expression-weighted risks range from 87 to 323 in brain.') == (
        'Weighted expression sums for matching RNAs range from 87 to 323 in brain.')
    assert review._metric_labels('The weighted expression of matching RNAss range from 87 to 323.') == (
        'The weighted expression sums for matching RNAs range from 87 to 323.')


@pytest.mark.parametrize('text', [
    'Blocks donor@123456 with the strongest recorded prediction.',
    'This is a computational prediction, not an activity measurement.',
    'The search found three 2-mm matches.',
    'The off-target effect is unavoidable.',
    'Provides clean donor coverage.',
    'Weighted expression is unavailable (null).',
    'It has isoform_hit_count=1 and target_only binding.',
    'Off_target_expression_weighted_risk is 0.0 in blood.',
    'Off_target_2mm_count is 2.',
    'The ISOFORM_HIT_COUNT is 1.',
    'Best selection rank (294) among the selected panel.',
    'This ASO has the highest native ranking.',
])
def test_visible_notation_and_contrastive_boilerplate_require_repair(tmp_path, monkeypatch, text):
    repaired = []
    def wording(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            result['recommendations'][0]['why_chosen'] = text
        if kind == 'repair':
            repaired.extend(payload['repair_candidate_ids'])
            for item in result['recommendations']:
                item['why_chosen'] = 'This ASO has complete scoped off-target evidence.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', wording)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed' and len(repaired) == 1
    assert all(entry['why_chosen'] != text for entry in receipt['recommendations'].values())
    assert review._prose_issue('Target RNA accessibility was not measured.') is None
    assert review._prose_issue('Uses NM_018238.4 with c.1047_1049del.') is None
    assert review._prose_issue('Uses NM_018238 as the reference.') is None
    assert review._prose_issue('Targets HLA_A.', context={'gene_symbol': 'HLA_A'}) is None


@pytest.mark.parametrize('summary', [
    'Four ASOs tile 0–3 nt downstream of the splice donor.',
    'ASO1 has the highest score and ASO4 has the fewest matches.',
    'The directional mean ranges from 0.039 to 0.053.',
    'Four ASOs cover the target. Two bind the donor. Activity remains unresolved.',
])
def test_summary_keeps_scores_and_relative_offsets_in_candidate_details(tmp_path, monkeypatch, summary):
    final_summary = 'Four ASOs target the selected splice region; cellular activity remains unresolved.'
    def wording(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            result['summary'] = summary
        if kind == 'repair':
            assert payload['repair_panel'] and not payload['repair_candidate_ids']
            result['summary'] = final_summary
        return result, metadata
    monkeypatch.setattr(review, '_model_call', wording)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed' and receipt['summary'] == final_summary


def test_canonical_target_alone_cannot_support_an_isoform_selectivity_preference(tmp_path, monkeypatch):
    def wording(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            result['recommendations'][0]['why_chosen'] = 'Matches two isoforms for greater selectivity.'
        if kind == 'repair':
            for item in result['recommendations']:
                item['why_chosen'] = 'This ASO has complete scoped off-target evidence.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', wording)
    goal = {'candidate_limit': 2, 'target_scope': 'canonical protein-coding transcript',
            'splice_isoform_objectives': [{'objective_scope': 'specific_isoform', 'desired_abundance': 'decrease'}]}
    _, _, receipt = run(pool(), tmp_path, goal)
    assert receipt['status'] == 'completed'
    assert all('selectiv' not in item['why_chosen'] for item in receipt['recommendations'].values())
    explicit = {'preservation_targets': [{'action': 'preserve', 'transcript_id': 'TX.2'}]}
    assert review._prose_issue('Matches the target isoform and spares TX.2 for selectivity.', context=explicit) is None


def test_ranking_reconsiders_unsupported_isoform_preference_before_selecting(tmp_path, monkeypatch):
    ranks = []
    def ranked(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'rank':
            ranks.append(payload)
            if len(ranks) == 1:
                result['ordered_ids'] = list(payload['evidence']['ids'])
                result['preferred_ids'] = result['ordered_ids'][:payload['promotion_limit']]
                result['summary'] = 'Ranked by isoform specificity, then predicted inhibition.'
            else:
                assert 'same-gene isoform matches' in ' '.join(payload['corrections_needed'])
        return result, metadata
    monkeypatch.setattr(review, '_model_call', ranked)
    selected, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed' and len(ranks) == 2
    assert [row['selection_rank'] for row in selected] == [7, 6]


def test_new_checks_feed_rejected_saved_response_into_first_repair(tmp_path, fake_model):
    context = {'candidate_limit': 2}
    catalog = evidence_catalog(pool())
    configuration = {'provider': 'fixture', 'model': 'fixture', 'reasoning_effort': 'high'}
    review._rank(catalog, context, tmp_path, configuration)
    component = next((tmp_path/'recommendation_review/components').glob('*.json'))
    saved = json.loads(component.read_text())
    saved['response']['summary'] = 'Ranked by isoform specificity.'
    saved['response_sha256'] = review.digest(saved['response'])
    component.write_text(json.dumps(saved))
    fake_model.clear()
    order, _, _ = review._rank(catalog, context, tmp_path, configuration)
    assert [call[0] for call in fake_model] == ['ranking_intent', 'rank']
    assert 'same-gene isoform matches' in ' '.join(fake_model[1][1]['corrections_needed'])
    assert fake_model[1][1]['previous_response'] == saved['response']
    assert len(order) == len(pool())


def test_ranking_can_explicitly_exclude_isoform_specificity(tmp_path, monkeypatch):
    kinds = []
    def reviewed(kind, payload, schema):
        kinds.append(kind)
        result, metadata = model(kind, payload, schema)
        if kind == 'rank':
            result['summary'] = 'Ranked by predicted inhibition, without weighting isoform-specificity.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', reviewed)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert kinds == ['rank', 'ranking_intent', 'explain', 'audit']


def test_ranking_criterion_outage_stops_instead_of_splitting_batches(tmp_path, monkeypatch):
    kinds = []
    def reviewed(kind, payload, schema):
        kinds.append(kind)
        if kind == 'ranking_intent':
            raise RuntimeError('provider unavailable')
        result, metadata = model(kind, payload, schema)
        if kind == 'rank':
            result['summary'] = 'Ranked by isoform specificity.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', reviewed)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'unavailable'
    assert receipt['failure_code'] == 'ranking_criterion_review_unavailable'
    assert kinds.count('rank') == 1
    assert kinds.count('ranking_intent') == review.ATTEMPTS


def test_explicit_retry_resumes_last_audited_draft_with_a_fresh_bounded_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'EDITORIAL_ROUNDS', 2)
    calls, repairs = [], []
    def editing(kind, payload, schema):
        calls.append(kind)
        result, metadata = model(kind, payload, schema)
        if kind == 'repair':
            repairs.append(True)
            result['recommendations'][0]['why_chosen'] = 'Revised explanation ' + str(len(repairs)) + '.'
        if kind == 'audit' and payload['draft']['recommendations'][0]['why_chosen'] != 'Revised explanation 2.':
            key = payload['selected_ids'][0]
            result['issues'] = [{'candidate_id': key, 'problem': 'Clarify the evidence behind this choice.'}]
            for item in result['field_reviews']:
                if item['candidate_id'] == key and item['field'] == 'why_chosen':
                    item['supported'] = False
        return result, metadata
    monkeypatch.setattr(review, '_model_call', editing)
    rows = pool()
    _, _, first = run(rows, tmp_path, {'candidate_limit': 2})
    assert first['status'] == 'unavailable' and len(repairs) == 1
    before = list(calls)
    _, _, ordinary_reopen = run(rows, tmp_path, {'candidate_limit': 2})
    assert ordinary_reopen['status'] == 'unavailable' and calls == before
    _, _, completed = run(rows, tmp_path, {'candidate_limit': 2}, retry_failed=True)
    assert completed['status'] == 'completed' and len(repairs) == 2
    assert calls[len(before):] == ['repair', 'audit']


def test_unchanged_repair_gets_actionable_feedback_instead_of_replaying_forever(tmp_path, monkeypatch):
    attempts = []
    def editing(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            result['recommendations'][0]['why_chosen'] = 'Matches donor@123.'
        if kind == 'repair':
            attempts.append(payload.get('corrections_needed'))
            if payload.get('corrections_needed'):
                result['recommendations'][0]['why_chosen'] = 'Targets the recorded splice donor.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', editing)
    _, _, result = run(pool(), tmp_path, {'candidate_limit': 2})
    assert result['status'] == 'completed' and len(attempts) == 2
    assert any('unchanged' in issue for issue in attempts[1])


def test_pre_checkpoint_receipts_resume_the_last_checked_draft(tmp_path, fake_model, monkeypatch):
    rows = pool()
    _, _, first = run(rows, tmp_path, {'candidate_limit': 2})
    assert first['status'] == 'completed'
    for path in (tmp_path/'recommendation_review/editorial').glob('*.json'):
        path.unlink()
    root = tmp_path/'recommendation_review.json'
    receipt = json.loads(root.read_text()); receipt['input_sha256'] = 'earlier-source'
    root.write_text(json.dumps(receipt))
    original, kinds = review._call, []
    def calls(kind, *args):
        kinds.append(kind)
        return original(kind, *args)
    monkeypatch.setattr(review, '_call', calls)
    monkeypatch.setattr(review, '_model_call', lambda *args: pytest.fail('Saved checked draft should reopen offline'))
    _, _, reopened = run(rows, tmp_path, {'candidate_limit': 2})
    assert reopened['status'] == 'completed' and kinds == ['rank', 'audit']


def test_review_receipt_identifies_the_loaded_worker_version(tmp_path, fake_model, monkeypatch):
    rows = pool()
    run(rows, tmp_path, {'candidate_limit': 2})
    path = tmp_path/'recommendation_review.json'
    first = json.loads(path.read_text())['input_sha256']
    original_read = Path.read_text
    def changed_source(source, *args, **kwargs):
        if source.name == 'shortlist_review.py':
            return 'on-disk edits after this worker imported its implementation'
        return original_read(source, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', changed_source)
    run(rows, tmp_path, {'candidate_limit': 2})
    assert json.loads(path.read_text())['input_sha256'] == first
    monkeypatch.setattr(review, '_IMPLEMENTATION_FINGERPRINT', {'loaded_worker': 'next-version'})
    run(rows, tmp_path, {'candidate_limit': 2})
    assert json.loads(path.read_text())['input_sha256'] != first


def test_audit_upgrade_rechecks_saved_draft_without_restarting_explanation(tmp_path, fake_model, monkeypatch):
    rows = pool()
    _, _, first = run(rows, tmp_path, {'candidate_limit': 2})
    for path in (tmp_path/'recommendation_review/editorial').glob('*.json'):
        path.unlink()
    original_instructions = review._instructions
    monkeypatch.setattr(review, '_instructions', lambda kind: original_instructions(kind)
                        + ('\nApply the updated tradeoff review.' if kind == 'audit' else ''))
    monkeypatch.setattr(review, '_IMPLEMENTATION_FINGERPRINT', {'loaded_worker': 'updated-audit'})
    original_call, kinds = review._call, []
    def calls(kind, *args):
        kinds.append(kind)
        return original_call(kind, *args)
    monkeypatch.setattr(review, '_call', calls)
    fake_model.clear()
    _, _, reopened = run(rows, tmp_path, {'candidate_limit': 2})
    assert reopened['status'] == 'completed' and kinds == ['rank', 'audit']
    assert [call[0] for call in fake_model] == ['audit']
    assert fake_model[0][1]['draft']['recommendations'][0]['why_chosen'] == (
        first['recommendations'][first['selected_ids'][0]]['why_chosen'])


def test_all_eligible_and_explicit_count_win_over_suggested_size(tmp_path, fake_model):
    for i, (goal, count) in enumerate([({'shortlist_size_preference': 'all_eligible'}, 7),
                                     ({'candidate_limit': 5, 'candidate_minimum': 5}, 5)]):
        selected, _, receipt = run(pool(), tmp_path/str(i), goal)
        assert len(selected) == count and receipt['status'] == 'completed'


@pytest.mark.parametrize('corruption', ['unknown_id', 'duplicate_id', 'missing_id', 'unknown_path', 'opaque_id_in_prose'])
def test_invalid_response_is_repaired_before_publication(tmp_path, monkeypatch, corruption):
    seen = []
    def broken_once(kind, payload, schema):
        result, meta = model(kind, payload, schema)
        if corruption == 'unknown_path' and kind == 'explain' and kind not in seen:
            result['recommendations'][0]['evidence_paths'] = ['/invented/potency']
        elif corruption == 'opaque_id_in_prose' and kind == 'explain' and kind not in seen:
            result['recommendations'][0]['why_chosen'] = 'Complements aso_6bb8463.'
        elif kind == 'rank' and kind not in seen:
            if corruption == 'unknown_id': result['ordered_ids'][0] = 'invented'
            if corruption == 'duplicate_id': result['ordered_ids'][0] = result['ordered_ids'][1]
            if corruption == 'missing_id': result['ordered_ids'].pop()
        seen.append(kind)
        return result, meta
    monkeypatch.setattr(review, '_model_call', broken_once)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert seen.count('explain' if corruption in {'unknown_path', 'opaque_id_in_prose'} else 'rank') == 2


def test_unsupported_rationale_requires_independent_repair(tmp_path, monkeypatch):
    audits = []
    def critic(kind, payload, schema):
        result, meta = model(kind, payload, schema)
        if kind == 'audit':
            if not audits:
                result['issues'] = [{'candidate_id': payload['selected_ids'][0],
                                     'problem': 'Explain the scoped uncertainty more clearly.'}]
            audits.append(result)
        if kind == 'repair':
            result['recommendations'][0]['tradeoff'] = 'The target-site accessibility prediction is unavailable.'
        return result, meta
    monkeypatch.setattr(review, '_model_call', critic)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed' and len(audits) == 2


def test_repairs_preserve_individually_approved_fields(tmp_path, monkeypatch):
    audits = []
    def repair_one_field(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit' and not audits:
            first = payload['selected_ids'][0]
            result['issues'] = [{'candidate_id': first, 'problem': 'Use the exact search scope in why_chosen.'}]
            for item in result['field_reviews']:
                if item['candidate_id'] == first and item['field'] == 'why_chosen':
                    item['supported'] = False
            audits.append(result)
        if kind == 'repair':
            result['recommendations'][0].update(why_chosen='The search covers two mismatches.',
                tradeoff='An unsolicited new inference.', panel_role='An unrelated new comparison.')
        return result, metadata
    monkeypatch.setattr(review, '_model_call', repair_one_field)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    entry = receipt['recommendations'][receipt['selected_ids'][0]]
    assert entry['why_chosen'] == 'The search covers two mismatches.'
    assert entry['tradeoff'] == 'Target RNA accessibility remains unresolved.'
    assert entry['panel_role'] == 'Adds another recorded binding site.'


def test_repairs_keep_exact_citations_and_audits_keep_complete_evidence(tmp_path, monkeypatch):
    rows = pool(2)
    rows[0]['uncited_future_analysis'] = {'score': 0.12345}
    audit_paths, repair_paths = [], []
    def paths(packet):
        return {path for fact in packet['facts'] for path in fact['paths'].values()}
    def focused_repair(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            audit_paths.append(paths(payload['evidence']))
            if len(audit_paths) == 1:
                result['issues'] = [{'candidate_id': payload['selected_ids'][0], 'problem': 'Clarify the search scope.'}]
        if kind == 'repair':
            repair_paths.append(paths(payload['evidence']))
            assert len(payload['evidence']['candidate_overviews']) == len(rows)
            assert '/contexts/0/candidate_assessment/off_target/status' in repair_paths[0]
            result['recommendations'][0]['why_chosen'] = 'The search covers two mismatches.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', focused_repair)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert len(repair_paths) == 1 and audit_paths[0] == audit_paths[1]
    assert '/contexts/0/uncited_future_analysis/score' in audit_paths[0]
    assert '/contexts/0/uncited_future_analysis/score' not in repair_paths[0]
    assert repair_paths[0] < audit_paths[0]


def test_panel_only_repair_ignores_unsolicited_candidate_rewrites(tmp_path, monkeypatch):
    audits = []
    def panel_repair(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit' and not audits:
            result['issues'] = [{'candidate_id': 'panel', 'problem': 'Describe the panel count directly.'}]
            for item in result['field_reviews']:
                if item['candidate_id'] == 'panel' and item['field'] == 'summary':
                    item['supported'] = False
            audits.append(result)
        if kind == 'repair':
            result['summary'] = 'Three ASOs cover distinct binding sites.'
            result['recommendations'] = deepcopy(payload['previous_draft']['recommendations'])
            for item in result['recommendations']:
                item['why_chosen'] = 'An unsolicited replacement.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', panel_repair)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed'
    assert receipt['summary'] == 'Three ASOs cover distinct binding sites.'
    assert all(item['why_chosen'] != 'An unsolicited replacement.' for item in receipt['recommendations'].values())


@pytest.mark.parametrize('number_type', [int, str])
def test_arithmetic_overrides_a_model_approving_an_incorrect_comparison(tmp_path, monkeypatch, number_type):
    audits = []
    def incorrect_rank(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit' and not audits:
            result['quantitative_comparisons'] = [{'candidate_id': payload['selected_ids'][0],
                'field': 'why_chosen', 'path': '/contexts/0/selection_rank', 'order': 'ascending',
                'claimed_rank': 1, 'scope_ids': payload['selected_ids']}]
            audits.append(result)
        if kind == 'repair':
            assert 'give rank 3' in str(payload['review_corrections'])
            result['recommendations'][0]['why_chosen'] = 'The off-target search is complete within its recorded scope.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', incorrect_rank)
    rows = pool()
    for row in rows:
        row['selection_rank'] = number_type(row['selection_rank'])
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed'
    assert receipt['recommendations'][receipt['selected_ids'][0]]['why_chosen'] == 'The off-target search is complete within its recorded scope.'


def test_provider_failure_is_explicit_and_keeps_a_working_panel(tmp_path, monkeypatch):
    def unavailable(*args): raise ConnectionError('secret must never enter receipts')
    monkeypatch.setattr(review, '_model_call', unavailable)
    selected, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'unavailable' and len(selected) == 2
    assert 'secret' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))


def test_generation_limit_retries_with_a_bounded_larger_allowance(tmp_path, monkeypatch):
    budgets = []
    def needs_more_room(kind, payload, schema):
        if kind == 'rank':
            budgets.append(payload['_generation_token_budget'])
            if len(budgets) == 1:
                raise review.ReviewTokenLimit('generation_budget_exhausted')
        return model(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', needs_more_room)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert budgets == [review.DEFAULT_OUTPUT_TOKENS, review.MAX_OUTPUT_TOKENS]


def test_failure_is_bounded_across_report_stages_and_explicit_retry_resumes(tmp_path, monkeypatch):
    calls = []
    def unavailable(*args): calls.append(args); raise ConnectionError()
    monkeypatch.setattr(review, '_model_call', unavailable)
    rows = pool()
    run(rows, tmp_path, {'candidate_limit': 2})
    first_count = len(calls)
    assert first_count == review.ATTEMPTS
    _, _, repeated = run(rows, tmp_path, {'candidate_limit': 2})
    assert len(calls) == first_count and repeated['status'] == 'unavailable'
    monkeypatch.setattr(review, '_model_call', model)
    _, _, resumed = run(rows, tmp_path, {'candidate_limit': 2}, retry_failed=True)
    assert resumed['status'] == 'completed'


def test_completed_review_reopens_offline_and_evidence_change_invalidates_it(tmp_path, fake_model, monkeypatch):
    rows = pool(); first, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    monkeypatch.setattr(review, '_model_call', lambda *a: pytest.fail('Must reuse accepted result'))
    again, _, saved = review.recommend_panel(rows, {'candidate_limit': 2}, tmp_path,
        select_panel=selector(tmp_path), allow_model_calls=False)
    assert again == first and saved['status'] == 'completed'
    rows[-1]['candidate_assessment']['off_target']['counts_by_mismatch'][0] = 10
    _, _, changed = review.recommend_panel(rows, {'candidate_limit': 2}, tmp_path,
        select_panel=selector(tmp_path), allow_model_calls=False)
    assert changed['status'] == 'pending'


def test_all_candidates_enter_bounded_review_and_global_comparison(tmp_path, fake_model, monkeypatch):
    monkeypatch.setattr(review, 'MAX_BATCH_MOLECULES', 12)
    rows = pool(31)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert set(receipt['review_levels'][0]['input_ids']) == {molecule_id(row) for row in rows}
    assert len(receipt['review_levels']) > 1
    assert receipt['reviewed_molecule_count'] == 31


def test_multiple_explanation_batches_produce_one_checked_introduction(tmp_path, fake_model, monkeypatch):
    monkeypatch.setattr(review, 'MAX_BATCH_MOLECULES', 2)
    _, _, receipt = run(pool(5), tmp_path, {'candidate_limit': 4})
    assert receipt['status'] == 'completed'
    assert receipt['summary'] == 'Together these ASOs cover the requested binding regions.'
    summaries = [args[1] for args in fake_model if args[0] == 'summarize']
    assert len(summaries) == 1 and summaries[0]['panel_count'] == 4
    assert len(summaries[0]['checked_notes']) == 2
    assert [args[0] for args in fake_model][-1] == 'audit_summary'


def test_parallel_ranking_retains_run_telemetry_context(tmp_path, monkeypatch):
    from aso_agents.telemetry import evaluation_trace, current_evaluation_trace, record_resource_acquisition
    monkeypatch.setattr(review, 'MAX_BATCH_MOLECULES', 4)
    seen = []
    def traced(kind, payload, schema):
        active = current_evaluation_trace()
        assert active is not None
        seen.append(active.trace_id)
        record_resource_acquisition({'resource': 'shortlist-test', 'wait_seconds': 0, 'slot': 0, 'limit': 4})
        return model(kind, payload, schema)
    monkeypatch.setattr(review, '_model_call', traced)
    with evaluation_trace({'case': 'parallel-shortlist'}) as trace:
        _, _, receipt = run(pool(9), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert len(seen) > 3 and set(seen) == {trace.trace_id}
    assert len(trace.resource_acquisitions) == len(seen)


def test_repeated_missing_ids_reduce_batch_size_without_losing_candidates(tmp_path, monkeypatch):
    batch_sizes = []
    def incomplete_large_batch(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'rank':
            size = len(payload['evidence']['ids'])
            batch_sizes.append(size)
            if size > 4:
                result['ordered_ids'].pop()
        return result, metadata
    monkeypatch.setattr(review, '_model_call', incomplete_large_batch)
    rows = pool(11)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert receipt['reviewed_molecule_count'] == 11
    assert set(receipt['ranked_ids']) == {molecule_id(row) for row in rows}
    assert min(batch_sizes) <= 4 and max(batch_sizes) == 11


def test_field_audit_requires_coverage_and_actionable_corrections(tmp_path, monkeypatch):
    attempts = []
    def incomplete_audit(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            attempts.append(payload)
            if len(attempts) == 1:
                result['field_reviews'].pop()
            elif len(attempts) == 2:
                result['field_reviews'][0]['supported'] = False
        return result, metadata
    monkeypatch.setattr(review, '_model_call', incomplete_audit)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed' and len(attempts) == 3
    assert 'every candidate field' in ' '.join(attempts[1]['corrections_needed'])
    assert 'actionable issue' in ' '.join(attempts[2]['corrections_needed'])


def test_extra_audit_metadata_cannot_approve_unreviewed_fields(tmp_path, monkeypatch):
    def repeats_panel(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            result['reviewed_ids'].append('another_panel_member')
            result['checks'].append({'candidate_id': 'another_panel_member',
                                    **{key: True for key in review.AUDIT_CHECKS}})
        return result, metadata
    monkeypatch.setattr(review, '_model_call', repeats_panel)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert len(receipt['recommendations']) == 2


def test_each_explanation_batch_owns_its_fields_when_auditor_repeats_other_members(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'MAX_BATCH_MOLECULES', 2)
    audited = []
    def repeats_other_members(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            audited.extend(payload['selected_ids'])
            other = next(item['id'] for item in payload['complete_panel'] if item['id'] not in payload['selected_ids'])
            result['field_reviews'].append({'candidate_id': other, 'field': 'why_chosen',
                'supported': False, 'evidence_statement': 'This candidate belongs to another review batch.'})
            result['issues'].append({'candidate_id': other, 'problem': 'Review this candidate in its own batch.'})
        return result, metadata
    monkeypatch.setattr(review, '_model_call', repeats_other_members)
    _, _, receipt = run(pool(4), tmp_path, {'candidate_limit': 4})
    assert receipt['status'] == 'completed'
    assert sorted(audited) == sorted(receipt['selected_ids'])


@pytest.mark.parametrize('pointer', ['panel_order', '/panel_order', '/complete_panel/0/panel_order'])
def test_panel_numbers_are_checked_as_metadata(tmp_path, monkeypatch, pointer):
    def panel_number(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            result['quantitative_comparisons'] = [{'candidate_id': payload['selected_ids'][0],
                'field': 'panel_role', 'path': pointer, 'order': 'ascending',
                'claimed_rank': 1, 'scope_ids': payload['selected_ids']}]
        return result, metadata
    monkeypatch.setattr(review, '_model_call', panel_number)
    _, _, receipt = run(pool(), tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed'


@pytest.mark.parametrize('strand,shift', [('+', 20), ('-', -20)])
def test_panel_binding_geometry_uses_transcript_direction_and_skips_ambiguous_placements(strand, shift):
    from aso_agents.shortlist_evidence import panel_binding_geometry
    rows = pool(3)
    for row in rows:
        row['strand'] = strand
    catalog = evidence_catalog(rows)
    ids = [molecule_id(row) for row in rows]
    geometry = panel_binding_geometry(catalog, ids)
    assert geometry[ids[0]]['start_shift_nt_in_RNA'] == 0
    assert geometry[ids[1]]['start_shift_nt_in_RNA'] == shift
    assert geometry[ids[2]]['start_shift_nt_in_RNA'] == shift * 2
    assert geometry[ids[1]]['start_direction_from_anchor'] == ('downstream' if strand == '+' else 'upstream')
    rows.append({**deepcopy(rows[1]), 'binding_intervals_bed0_half_open': [['chr1', 800, 816]]})
    assert ids[1] not in panel_binding_geometry(evidence_catalog(rows), ids)


def test_rna_position_comparisons_use_recorded_strand(tmp_path, monkeypatch):
    rows = pool(3)
    for row in rows:
        row['strand'] = '-'
    def position(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            first = payload['selected_ids'][0]
            result['quantitative_comparisons'] = [{'candidate_id': first, 'field': 'panel_role',
                'path': '/binding_geometry/' + first + '/start_shift_nt_in_RNA', 'order': 'ascending',
                'claimed_rank': 1, 'scope_ids': payload['selected_ids']}]
            assert payload['binding_geometry'][payload['selected_ids'][1]]['start_direction_from_anchor'] == 'downstream'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', position)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed'


def test_a_small_batch_cannot_discard_candidates_for_a_global_count_shortfall(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'MAX_BATCH_MOLECULES', 3)
    corrected = []
    def misread_count(kind, payload, schema):
        result, meta = model(kind, payload, schema)
        if kind == 'rank':
            # This is a global count, including when this batch has one ASO.
            result['suggested_size'] = 4
        if kind == 'rank' and not payload['review_scope']['is_final_comparison']:
            assert payload['review_scope']['complete_pool_molecule_count'] == 5
            if not payload.get('corrections_needed'):
                result.update(preferred_ids=[], suggested_size=0,
                              summary='This subset has fewer candidates than the requested global count.')
            else:
                corrected.append(payload['evidence']['ids'])
        return result, meta
    monkeypatch.setattr(review, '_model_call', misread_count)
    selected, _, receipt = run(pool(5), tmp_path, {'candidate_limit': 4})
    assert len(selected) == 4 and receipt['status'] == 'completed'
    assert len(corrected) == 2


def test_ineligible_motif_rows_cannot_be_recommended(tmp_path, fake_model):
    rows = pool(); rows[-1]['candidate_assessment']['selection_eligible'] = False
    selected, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert len(selected) == 2 and selected[0]['selection_rank'] == 6
    assert receipt['reviewed_molecule_count'] == 6


def test_no_eligible_candidates_needs_no_model(tmp_path, fake_model):
    rows = pool();
    for row in rows: row['candidate_assessment']['selection_eligible'] = False
    selected, _, receipt = run(rows, tmp_path)
    assert not selected and not fake_model and receipt['status'] == 'no_eligible_candidates'


def test_large_pool_outage_stops_unsent_review_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'MAX_BATCH_MOLECULES', 2)
    attempts = []
    def unavailable(kind, payload, schema):
        attempts.append(tuple(payload['evidence']['ids']))
        raise ConnectionError('simulated provider outage')
    monkeypatch.setattr(review, '_model_call', unavailable)
    rows = pool(20)
    selected, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'unavailable' and len(selected) == 2
    assert len(set(attempts)) <= review.MAX_WORKERS
    assert len(attempts) <= review.MAX_WORKERS * review.ATTEMPTS


def test_hard_nonoverlap_is_enforced_after_model_preference(tmp_path, fake_model):
    rows = pool(4)
    rows[-1]['binding_intervals_bed0_half_open'] = [['chr1', 140, 156]]
    selected, summary, receipt = run(rows, tmp_path, {'candidate_limit': 3, 'candidate_minimum': 3,
        'candidate_selection_policy': 'nonoverlapping_target_sites'})
    assert receipt['status'] == 'completed' and summary['status'] == 'complete'
    assert len(selected) == 3
    starts = [r['binding_intervals_bed0_half_open'][0][1] for r in selected]
    assert len(set(starts)) == 3


def test_same_sequence_with_different_chemistry_stays_distinct(tmp_path, fake_model):
    rows = pool(2); rows[1]['synthesized_antisense_sequence'] = rows[0]['synthesized_antisense_sequence']
    rows[1]['IDT'] = 'second-chemistry'
    selected, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert len(selected) == 2 and receipt['reviewed_molecule_count'] == 2


def test_chemistries_without_ordering_notation_keep_physical_identity(tmp_path, fake_model):
    from aso_molecule import molecule_record
    rows = pool(2)
    sequence = rows[0]['synthesized_antisense_sequence']
    for row, architecture in zip(rows, ('steric-pmo', 'unspecified')):
        row.pop('IDT')
        row['synthesized_antisense_sequence'] = sequence
        row['molecule_specification'] = molecule_record(sequence, architecture, '')
    selected, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed' and receipt['reviewed_molecule_count'] == 2
    assert len({molecule_id(row) for row in selected}) == 2


def test_duplicate_contexts_do_not_fill_a_molecule_count(tmp_path, fake_model):
    rows = pool(4); rows.append({**deepcopy(rows[-1]), 'job_id': 'second-context'})
    selected, _, receipt = run(rows, tmp_path, {'candidate_limit': 3})
    assert len({molecule_id(row) for row in selected}) == 3
    assert receipt['reviewed_binding_record_count'] == 5


def test_full_evidence_keeps_new_attributes_and_bounds_bulk_sequence():
    rows = pool(1); rows[0]['new_future_measurement'] = 0.123456789
    rows[0]['reference_sequence'] = 'ACGT' * 10000
    catalog = evidence_catalog(rows)
    facts = next(iter(catalog.values()))['facts']
    assert facts['/contexts/0/new_future_measurement'] == 0.123456789
    assert facts['/contexts/0/reference_sequence']['length'] == 40000


def test_array_elements_keep_exact_source_pointers_and_comparison_values(tmp_path, monkeypatch):
    from aso_agents.shortlist_evidence import citation, fact_value
    rows = pool(3)
    entry = evidence_catalog(rows)[molecule_id(rows[2])]
    pointer = '/contexts/0/candidate_assessment/off_target/counts_by_mismatch/2'
    assert fact_value(entry, pointer) == 2
    assert citation(entry, pointer)['source_pointer'] == '/rows/2/candidate_assessment/off_target/counts_by_mismatch/2'
    for suffix in ('02', '-1', '99', 'unavailable'):
        with pytest.raises(KeyError):
            fact_value(entry, pointer.rsplit('/', 1)[0] + '/' + suffix)
    def array_reference(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            for item in result['recommendations']:
                item['evidence_paths'].append(pointer)
        if kind == 'audit':
            result['quantitative_comparisons'] = [{'candidate_id': payload['selected_ids'][0],
                'field': 'why_chosen', 'path': pointer, 'order': 'descending', 'claimed_rank': 1,
                'scope_ids': payload['selected_ids']}]
        return result, metadata
    monkeypatch.setattr(review, '_model_call', array_reference)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 3})
    assert receipt['status'] == 'completed'
    assert receipt['recommendations'][molecule_id(rows[2])]['evidence'][-1]['value'] == 2


def test_sparse_tree_encoding_preserves_every_original_fact():
    from aso_agents.shortlist_evidence import nested_packet, _leaves
    rows = pool(3)
    rows[0]['extra/measurement'] = {'~unit': [1, 2, 3]}
    rows[1]['contexts_with_arrays'] = [{'value': 0.123456789}, {'value': None}]
    entries = list(evidence_catalog(rows).values())
    packet = nested_packet(entries)
    shared = dict(_leaves(packet['shared_facts']))
    for entry, saved in zip(entries, packet['candidate_facts']):
        own = dict(_leaves(saved['facts']))
        shared.pop('', None); own.pop('', None)
        assert {**shared, **own} == entry['facts']


def test_indexed_encoding_preserves_nulls_mixed_shapes_arrays_and_escaped_paths():
    from aso_agents.shortlist_evidence import indexed_packet
    rows = pool(3)
    rows[0]['optional'] = None
    rows[1]['optional'] = {'nested/key': {'~unit': [1, 2, 3]}}
    rows[0]['sometimes_empty'] = {}
    rows[1]['sometimes_empty'] = {'child': 0.123456789}
    entries = list(evidence_catalog(rows).values())
    packet = indexed_packet(entries)
    paths = {}
    def walk(node, path=''):
        if isinstance(node, str):
            paths[path] = node
            return
        if isinstance(node, list):
            paths[path], node = node
        for name, child in node.items():
            walk(child, path + '/' + name.replace('~', '~0').replace('/', '~1'))
    walk(packet['path_index'])
    for index, entry in enumerate(entries):
        restored = {}
        for path, key in paths.items():
            fact = packet['fact_values'][key]
            if index not in fact.get('available_for', range(len(entries))):
                continue
            restored[path] = (fact.get('overrides', {}).get(str(index), fact['shared']) if 'shared' in fact
                              else fact['values_in_id_order'][index])
        assert restored == entry['facts']


@pytest.mark.parametrize('strand,upstream,downstream,covered', [('+', 8, 9, [1, 3]), ('-', 9, 8, [1, 3])])
def test_coordinate_observations_follow_rna_direction(strand, upstream, downstream, covered):
    from aso_agents.shortlist_evidence import coordinate_observations
    row = {'source_annotations': {'binding_site_membership': [{'label': 'declared target',
        'genomic_segments_bed0_half_open': [['chr1', 108, 111, 'target', 0, strand]]}]}}
    geometry = {'strand': strand, 'binding_intervals_1based': [{'chrom': 'chr1', 'start': 101, 'end': 120}]}
    observation = coordinate_observations(row, geometry)[0]
    assert observation['bases_beyond_target_upstream_in_RNA'] == upstream
    assert observation['bases_beyond_target_downstream_in_RNA'] == downstream
    assert observation['target_bases_covered_1based_in_RNA_order'] == covered
    geometry['binding_intervals_1based'][0]['end'] = 110
    assert coordinate_observations(row, geometry)[0]['target_bases_covered_1based_in_RNA_order'] == ([1, 2] if strand == '+' else [2, 3])
    geometry['strand'] = '-' if strand == '+' else '+'
    assert coordinate_observations(row, geometry) == []


@pytest.mark.parametrize('request_pointer,wrong_target', [(False, False), (True, False), (False, True)])
def test_coordinate_comparisons_use_verified_geometry(tmp_path, monkeypatch, request_pointer, wrong_target):
    rows = pool(2)
    for index, row in enumerate(rows):
        row['strand'] = '-'
        row['binding_intervals_bed0_half_open'] = [['chr1', 100 + index, 116 + index]]
        row['source_annotations'] = {'binding_site_membership': [{'label': 'declared target',
            'genomic_segments_bed0_half_open': [['chr1', 108, 111 + int(wrong_target and index == 1), 'target', 0, '-']]}]}
    calls = []
    def geometry_claim(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            calls.append(payload)
            field = 'bases_beyond_target_upstream_in_RNA'
            pointer = ('/coordinate_observations/0/observations/0/' if request_pointer
                       else '/coordinate_observations/0/') + field
            if len(calls) == 1:
                result['quantitative_comparisons'] = [{'candidate_id': payload['selected_ids'][0],
                    'field': 'panel_role', 'path': pointer, 'order': 'descending', 'claimed_rank': 1,
                    'scope_ids': payload['selected_ids']}]
        if kind == 'repair':
            assert wrong_target and 'comparable numerical values are unavailable' in str(payload['review_corrections'])
            result['recommendations'][0]['panel_role'] = 'Covers a different declared target.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', geometry_claim)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    if wrong_target:
        assert receipt['status'] == 'completed' and len(calls) == 2
        assert receipt['recommendations'][molecule_id(rows[1])]['panel_role'] == 'Covers a different declared target.'
    else:
        assert receipt['status'] == 'completed' and len(calls) == 1
        comparison = next(item for item in calls[0]['numeric_comparisons']
            if item['path'].endswith('/bases_beyond_target_upstream_in_RNA'))
        assert comparison['values'] == {molecule_id(rows[0]): 5, molecule_id(rows[1]): 6}


def test_missing_comparison_value_repairs_prose_without_retrying_the_auditor(tmp_path, monkeypatch):
    rows = pool(2)
    rows[0]['expression_score'] = 10
    rows[1]['expression_score'] = None
    calls = []
    def missing_value(kind, payload, schema):
        calls.append(kind)
        result, metadata = model(kind, payload, schema)
        if kind == 'audit' and calls.count('audit') == 1:
            result['quantitative_comparisons'] = [{'candidate_id': payload['selected_ids'][0],
                'field': 'why_chosen', 'path': '/contexts/0/expression_score', 'order': 'ascending',
                'claimed_rank': 1, 'scope_ids': payload['selected_ids']}]
        if kind == 'repair':
            result['recommendations'][0]['why_chosen'] = 'The requested binding region is covered; the expression score is unavailable.'
        return result, metadata
    monkeypatch.setattr(review, '_model_call', missing_value)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed'
    assert calls == ['rank', 'explain', 'audit', 'repair', 'audit']
    assert 'expression score is unavailable' in receipt['recommendations'][molecule_id(rows[1])]['why_chosen']


@pytest.mark.parametrize('order,claim', [('absolute_ascending', 1), ('ascending', 2)])
def test_signed_changes_and_magnitudes_have_distinct_rankings(tmp_path, monkeypatch, order, claim):
    rows = pool(2)
    rows[0]['site_change'] = -0.9
    rows[1]['site_change'] = -0.01
    calls = []
    def signed_claim(kind, payload, schema):
        calls.append(kind)
        result, metadata = model(kind, payload, schema)
        if kind == 'explain':
            result['recommendations'][0]['evidence_paths'].append('/contexts/0/site_change')
        if kind == 'audit':
            result['quantitative_comparisons'] = [{'candidate_id': payload['selected_ids'][0],
                'field': 'tradeoff', 'path': '/contexts/0/site_change', 'order': order,
                'claimed_rank': claim, 'scope_ids': payload['selected_ids']}]
            comparison = next(item for item in payload['numeric_comparisons'] if item['path'] == '/contexts/0/site_change')
            assert comparison['absolute_ascending_rank'][molecule_id(rows[1])] == 1
            assert comparison['ascending_rank'][molecule_id(rows[1])] == 2
        return result, metadata
    monkeypatch.setattr(review, '_model_call', signed_claim)
    _, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert receipt['status'] == 'completed' and calls == ['rank', 'explain', 'audit']


def test_large_fact_paths_do_not_prevent_cross_batch_comparison(tmp_path, fake_model, monkeypatch):
    from aso_agents.shortlist_evidence import compact_packet, nested_packet
    rows = pool(3)
    for i, row in enumerate(rows):
        row['long_context_name_' * 7] = {'measurement_' + str(j): i + j for j in range(150)}
    entries = list(evidence_catalog(rows).values())
    size = lambda packet: len(json.dumps(packet, separators=(',', ':')))
    cap = max(size(compact_packet([entry])) for entry in entries) + 100
    assert size(nested_packet(entries)) < cap
    assert size(compact_packet(entries[:2])) > cap
    monkeypatch.setattr(review, 'MAX_PACKET_CHARACTERS', cap)
    order, _, _ = review._rank({entry['id']: entry for entry in entries}, {}, tmp_path, {})
    assert set(order) == {entry['id'] for entry in entries}
    assert any('shared_facts' in args[1]['evidence'] or 'path_index' in args[1]['evidence'] for args in fake_model)


def test_empty_model_recommendation_is_preserved_when_reopened(tmp_path, monkeypatch):
    def none_supported(kind, payload, schema):
        if kind != 'rank':
            return model(kind, payload, schema)
        return {'ordered_ids': payload['evidence']['ids'], 'preferred_ids': [], 'suggested_size': 0,
                'summary': 'All recorded perturbations favor the opposite splicing change.'}, []
    monkeypatch.setattr(review, '_model_call', none_supported)
    rows = pool()
    selected, _, receipt = run(rows, tmp_path, {'candidate_limit': 2})
    assert not selected and receipt['status'] == 'completed'
    monkeypatch.setattr(review, '_model_call', lambda *a: pytest.fail('Reuse the empty recommendation'))
    selected, _, again = run(rows, tmp_path, {'candidate_limit': 2})
    assert not selected and again['status'] == 'completed'
