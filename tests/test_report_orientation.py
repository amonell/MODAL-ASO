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
    assert summary['notices'] == ['1 could not be classified.']
    assert 'does not establish how well an ASO will work' in summary['meaning']
    assert all(fact['source'] == 'reader-section-0' for fact in summary['facts'])
    assert sections == before


@pytest.mark.parametrize('text', [audit_text(total=7), audit_text(kept=0, excluded=2),
                                  audit_text().replace('excluded |', 'mystery |'),
                                  audit_text().replace('- Retained: 1', '- Retained: 1\n- Retained: 1')])
def test_inconsistent_or_ambiguous_counts_are_not_given_authoritative_summary_cards(text):
    summary = build_orientation(_sections(text))
    assert summary['facts'] == []
    assert 'could not be summarized reliably' in summary['notices'][0]


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
    assert summary['facts'][0]['value'] == 2
    assert '2 are shown here out of 7' in summary['facts'][0]['note']
    assert 'requested effect' in summary['meaning']
    assert 'successful' not in str(summary)


def test_empty_main_panel_is_not_reported_as_no_asos_anywhere():
    text = candidate_text(0, 0) + '\n## Another candidate list\n| ASO (5′–3′) | score |\n|---|---|\n| CCGG | 0 |'
    summary = build_orientation(_sections(text), {'kind': 'design'})
    assert summary['facts'][0]['value'] == 0
    assert 'Other tables contain ASO entries' in summary['notices'][0]
    assert 'failed' not in str(summary)


def test_activity_unavailable_remains_distinct_from_zero_activity():
    evaluation = {'candidate_count': 2, 'predicted_count': 0,
                  'candidates': [{'status': 'error'}, {'status': 'not_predicted'}]}
    context = {'kind': 'design', 'evaluations': [evaluation]}; before = deepcopy(context)
    summary = build_orientation(_sections(candidate_text(2, 2)), context)
    assert summary['facts'][1]['label'] == 'Activity estimates available'
    assert summary['facts'][1]['value'] == '0 of 2'
    assert 'do not mean zero activity' in summary['notices'][0]
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
    assert 'does not show that the ASO binds or works' in summary['meaning']
    assert 'No exact match' not in str(summary)
    ambiguous = build_orientation(_sections(text + text))
    assert ambiguous['facts'] == []


def test_combined_reports_count_reports_not_distinct_targets_or_a_shared_rank():
    summary = build_orientation([], {'kind': 'combined', 'design_labels': ['GENE · gapmer', 'GENE · splice']})
    assert summary['facts'][0]['value'] == 2
    assert summary['facts'][0]['label'] == 'Design reports available'
    assert 'not a shared ranking' in summary['meaning']


def test_orientation_escapes_source_labels_and_links_to_existing_sections():
    sections = _sections(candidate_text(1, 1))
    summary = build_orientation(sections, {'kind': 'design', 'gene': '<script>alert(1)</script>'})
    page = orientation_html(summary, findings='reader-findings', downloads='reader-section-1')
    assert '<script>alert(1)</script>' not in page
    assert 'href="#reader-section-0"' in page and 'href="#reader-findings"' in page


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
