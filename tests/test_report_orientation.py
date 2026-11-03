"""Novice summaries must stay tied to their specific saved source records."""
from copy import deepcopy

import pytest

from aso_agents.readable_report import _sections, readable_markdown_html
from aso_agents.report_orientation import build_orientation, orientation_html, section_tables


def audit_text(total=3, kept=1, excluded=1, unknown=1):
    return f'''## Outcome summary
- Total input rows: {total}
- Retained: {kept}
- Excluded: {excluded}
- Indeterminate: {unknown}

## Complete audit table
| id | sequence | recomputed_length | retention_status |
|---|---|---|---|
| a | ACGT | 4 | retained |
| b | AACCGG | 6 | excluded |
| c | NA | NA | indeterminate |
'''


def test_sequence_checks_explain_eligibility_without_claiming_biological_success():
    sections = _sections(audit_text()); before = deepcopy(sections)
    summary = build_orientation(sections)
    assert summary['kind'] == 'sequence_checks'
    assert [fact['value'] for fact in summary['facts']] == [3, 1, 1]
    assert summary['notices'] == ['1 entry needs a complete check result.']
    assert summary['purpose'] == 'Sequences checked against the supplied rules.'
    assert all(fact['source'] == 'reader-section-1' for fact in summary['facts'])
    assert sections == before


@pytest.mark.parametrize('text', [audit_text(total=7), audit_text(kept=0, excluded=2),
                                  audit_text().replace('excluded |', 'mystery |'),
                                  audit_text().replace('- Retained: 1', '- Retained: 1\n- Retained: 1')])
def test_inconsistent_or_ambiguous_counts_are_not_given_authoritative_summary_cards(text):
    summary = build_orientation(_sections(text))
    assert summary['facts'] == []
    assert 'could not be reconciled' in summary['notices'][0]


def test_saved_passed_failed_statuses_agree_with_retained_flags_before_summarizing():
    text = audit_text().replace('retention_status |', 'retention_status | retained |')
    text = text.replace('|---|---|---|---|', '|---|---|---|---|---|')
    text = text.replace('| 4 | retained |', '| 4 | passed | true |')
    text = text.replace('| 6 | excluded |', '| 6 | failed | false |')
    text = text.replace('| NA | indeterminate |', '| NA | indeterminate | |')
    summary = build_orientation(_sections(text))
    assert [fact['value'] for fact in summary['facts']] == [3, 1, 1]
    conflict = build_orientation(_sections(text.replace('passed | true', 'passed | false')))
    assert conflict['facts'] == []
    duplicate = build_orientation(_sections(text + text))
    assert duplicate['facts'] == []


def candidate_text(total, shown):
    rows = ''.join(f'| c{i} | ACGT |\n' for i in range(shown))
    return f'''## Selected physical panel
Selected candidate rows: {total}. Distinct molecules (antisense sequence plus chemistry): {total}. Preview: {shown} molecules in recorded rank order.
| Candidate | Antisense oligo (5′→3′) |
|---|---|
{rows}'''


def test_partial_candidate_preview_does_not_become_a_claim_of_complete_delivery():
    summary = build_orientation(_sections(candidate_text(7, 2)), {'kind': 'design', 'gene': 'GENE'})
    assert summary['title'] == '7 ASOs for GENE'
    assert 'previews 2 of 7' in summary['meaning']
    assert summary['facts'] == []
    assert 'successful' not in str(summary)


def test_empty_main_panel_is_not_reported_as_no_asos_anywhere():
    text = candidate_text(0, 0) + '\n## Another candidate list\n| ASO (5′–3′) | score |\n|---|---|\n| CCGG | 0 |'
    summary = build_orientation(_sections(text), {'kind': 'design'})
    assert summary['title'] == 'No ASOs selected'
    assert 'Other tables contain ASO entries' in summary['notices'][0]
    assert 'failed' not in str(summary)


def test_activity_unavailable_remains_distinct_from_zero_activity():
    evaluation = {'candidate_count': 2, 'predicted_count': 0,
                  'candidates': [{'status': 'error'}, {'status': 'not_predicted'}]}
    context = {'kind': 'design', 'evaluations': [evaluation]}; before = deepcopy(context)
    summary = build_orientation(_sections(candidate_text(2, 2)), context)
    assert summary['facts'] == []
    assert summary['notices'] == ['OligoAI activity predictions are unavailable for 2 of 2 ASOs.']
    assert context == before


@pytest.mark.parametrize('evaluations', [
    [{'candidate_count': 2, 'predicted_count': 2, 'candidates': [{'status': 'error'}, {'status': 'error'}]}],
    [{'candidate_count': 2, 'predicted_count': 0, 'candidates': []}],
    [{'candidate_count': 1, 'predicted_count': 0, 'candidates': [{'status': 'unknown'}]}],
    [{'candidate_count': 1, 'predicted_count': 0, 'candidates': [{'status': []}]}],
    {'unexpected': 'format'},
    [{'candidate_count': 1, 'predicted_count': 0, 'candidates': [{'status': 'error'}]}] * 2,
])
def test_activity_summaries_do_not_guess_or_pool_inconsistent_contexts(evaluations):
    summary = build_orientation(_sections(candidate_text(1, 1)), {'kind': 'design', 'evaluations': evaluations})
    assert not any(fact['label'] == 'Activity estimates available' for fact in summary['facts'])


def test_comparison_counts_preserve_observations_without_reranking_or_deduplicating_asos():
    text = '''## Full observation membership
| id | set | sequence | pooled_nondominated_frontier |
|---|---|---|---|
| a | A | ACGT | true |
| b | A | ACGT | false |
| c | B | AACCGG | true |
'''
    summary = build_orientation(_sections(text))
    assert summary['kind'] == 'comparison'
    assert [fact['value'] for fact in summary['facts']] == [3, 2, 2]
    assert 'at least as good on every measure and better on at least one' in summary['meaning']
    assert 'winner' not in str(summary)
    invalid = build_orientation(_sections(text.replace('false |', 'unknown |')))
    assert invalid['facts'] == []


def test_fenced_and_malformed_tables_cannot_create_summary_evidence():
    text = '## Results\n\n```text\n' + audit_text() + '\n```'
    assert all(not section_tables(section) for section in _sections(text))
    malformed = '## Results\n| A | B |\n|---|---|\n| a | b | extra |'
    assert not section_tables(_sections(malformed)[0])


def test_unknown_analysis_receives_orientation_but_no_invented_counts_or_success():
    summary = build_orientation(_sections('## Unexpected finding\n\nImportant result: unavailable.'))
    assert summary['kind'] == 'analysis' and summary['facts'] == []
    assert 'successful' not in str(summary)
    page = readable_markdown_html('## Unexpected finding\n\nImportant result: unavailable.')
    assert 'Important result: unavailable.' in page and '<h2>Unexpected finding</h2>' in page


def test_evaluation_counts_entries_without_interpreting_scientific_assessments():
    text = '''## Assessment
| ASO | Recomputed properties | Assessment |
|---|---|---|
| A | 4 nt | No exact match |
| A | 4 nt | Matching region found; binding unknown |
'''
    summary = build_orientation(_sections(text))
    assert summary['kind'] == 'evaluation'
    assert summary['facts'][0]['value'] == 2
    assert summary['facts'][0]['label'] == 'Entries assessed'
    assert 'Off-targets are other sequences' in summary['meaning']
    assert 'No exact match' not in str(summary)
    ambiguous = build_orientation(_sections(text + text))
    assert ambiguous['facts'] == []


def test_combined_reports_count_reports_not_distinct_targets_or_a_shared_rank():
    summary = build_orientation([], {'kind': 'combined', 'design_labels': ['GENE · gapmer', 'GENE · splice']})
    assert summary['facts'][0]['value'] == 2
    assert summary['facts'][0]['label'] == 'Design reports available'
    assert 'sequences, predictions and checks' in summary['meaning']


def test_orientation_escapes_source_labels_and_links_to_existing_sections():
    sections = _sections(candidate_text(1, 1))
    summary = build_orientation(sections, {'kind': 'design', 'gene': '<script>alert(1)</script>'})
    page = orientation_html(summary, findings='reader-findings', downloads='reader-section-1')
    assert '<script>alert(1)</script>' not in page
    assert 'href="#reader-section-0"' in page and 'href="#reader-section-1"' in page


def test_file_paths_move_to_details_without_hiding_scientific_context():
    text = '# Results\n\nInput: `/input/file.csv`\nAudit table: `/output/table.csv`\nA critical measurement is missing.\n\n## Method\n\nExact procedure.'
    page = readable_markdown_html(text)
    assert '<summary>Input and output files</summary>' in page
    assert '/input/file.csv' in page and '/output/table.csv' in page
    assert page.index('A critical measurement is missing.') < page.index('<summary>Input and output files</summary>')


def test_file_metadata_inside_examples_or_after_prose_stays_in_context():
    text = '# Results\n\n```text\nInput: `/input/example.csv`\n```\n\nImportant context.\nInput: `/input/evidence.csv`'
    page = readable_markdown_html(text)
    assert '<summary>Input and output files</summary>' not in page
    assert '/input/example.csv' in page and '/input/evidence.csv' in page
    repeated = readable_markdown_html('# Results\nInput: `/input/a.csv`\n\nImportant context.\nInput: `/input/a.csv`')
    assert repeated.count('/input/a.csv') == 2


def test_identifier_and_unavailable_rank_are_not_reinterpreted_as_a_candidate_or_zero():
    text = '| id | Native rank / scope | value |\n|---|---|---|\n| x | unavailable / scope_a | 0 |'
    page = readable_markdown_html(text)
    assert 'title="id">ID</th>' in page
    assert 'No rank supplied · group details' in page
    assert 'data-recorded-value="unavailable / scope_a"' in page
    assert '<td>0</td>' in page


def test_failures_and_unavailable_files_are_visible_before_folded_evidence():
    text = '''## OligoAI gapmer efficacy prediction
| Candidate | Predicted inhibition (%) | Status |
|---|---|---|
| a | — | error: runtime unavailable |
| a | — | error: another context unavailable |
## Requested output files
- Candidate table: file unavailable in this report bundle.
## Selected-Panel Off-Target Search Budget
Selected panel: every retained candidate has a recorded off-target search budget of 0 mismatch(es), covering exact matches (0 mismatches) only; one- and higher-mismatch bins were not searched and are unknown.
'''
    summary = build_orientation(_sections(text))
    assert len(summary['notices']) == 2
    assert not summary['facts']  # two contexts are not two ASOs
    page = readable_markdown_html(text)
    assert page.index('Activity prediction failed for every entry') < page.index('<summary>Predicted gapmer activity')
    assert '1 requested file is unavailable' in page
    assert 'Off-target search limit:' not in page
    assert 'exact matches (0 mismatches) only' in page


def test_arbitrary_error_prose_and_configured_search_limits_do_not_become_status_claims():
    text = '## Unfamiliar result\nAn error may occur. Search budget configured: 2.\n\n## Requested output files\n- [Available](panel.csv)'
    assert not build_orientation(_sections(text))['notices']


def test_native_evaluation_match_counts_describe_recorded_references_not_activity():
    text = '''## Molecule overview
| ASO | Evaluated molecule (5′→3′) | Exact antisense records | Screening status |
|---|---|---|---|
| A | ACGT | 1 | complete |
| B | GGTT | 0 | unavailable |
'''
    summary = build_orientation(_sections(text))
    assert summary['kind'] == 'evaluation'
    assert [fact['value'] for fact in summary['facts']] == [2, '1 of 2']
    assert summary['notices'] == ['Off-target search results are unavailable or incomplete for 1 of 2 entries.']
    complete = build_orientation(_sections(text.replace('0 | unavailable', '0 | complete')))
    assert '1 entry has no recorded exact match in the completed search' in complete['notices'][0]
    unknown = build_orientation(_sections(text.replace('| 0 |', '| NA |')))
    assert len(unknown['facts']) == 1
    assert unknown['notices'] == ['Off-target search results are unavailable or incomplete for 1 of 2 entries.']


def test_design_summary_explains_intent_and_actual_incomplete_checks_without_boilerplate():
    from aso_agents.report_orientation import design_reader_context
    text = candidate_text(2, 2) + '''\n## Recommended shortlist
2 distinct molecules selected; 17 in the assessed eligible pool.
| Order | Off-target screen |
|---|---|
| 1 | incomplete_or_unavailable |
| 2 | complete_within_declared_scope |
'''
    context = design_reader_context({'intent':'knockdown'})
    summary = build_orientation(_sections(text), {**context,'kind':'design','gene':'GENE'})
    assert summary['title'] == '2 ASOs for GENE'
    assert summary['purpose'] == 'Goal: lower GENE RNA.'
    assert summary['meaning'] == ''
    assert summary['notices'] == ['Off-targets are other sequences an ASO may also bind. The search is incomplete for 1 of 2 ASOs.']
    page = orientation_html(summary, findings='reader-findings')
    for phrase in ['not measurements', 'does not establish', 'Activity estimates available', 'Entries in the main candidate table']:
        assert phrase not in page


def test_low_confidence_splicing_remains_visible_and_exon_name_comes_from_typed_goal():
    from aso_agents.report_orientation import design_reader_context
    context = design_reader_context({'intent':'splice_modulation','desired_effect':'inclusion','exon_label':7},
        {'goals':[{'rank':1,'ranking_confidence':'low'}]})
    result = build_orientation(_sections(candidate_text(2, 2)),{**context,'kind':'design','gene':'GENE'})
    assert result['purpose'] == 'Goal: help GENE RNA retain exon 7 during splicing.'
    assert result['notices'][0] == 'Confidence in the leading splicing approach is low.'


def test_splicing_summary_uses_the_recorded_direction_comparison_without_calling_it_exon_inclusion():
    text = candidate_text(2, 2) + '''\n## Predicted Splice-Response Candidate Sets
| rank | directional delta | opposite delta | confidence |
|---|---|---|---|
| 1 | 0.0259 | 0.9688 | low |
'''
    context = {'kind':'design','routes':[{'rank':1,'ranking_confidence':'low'}]}
    result = build_orientation(_sections(text), context)
    assert 'strongest splice-site prediction favors the opposite direction' in result['notices'][0]
    for changed in [text.replace('0.9688','0.001'), text.replace('0.9688','unknown'), text + text]:
        assert 'opposite direction' not in str(build_orientation(_sections(changed),context)['notices'])
