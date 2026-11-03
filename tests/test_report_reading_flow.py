"""Reading shortcuts retain exact records and keep missing evidence visible."""
import json
import re
from copy import deepcopy

import pytest

from aso_agents.readable_report import _sections, friendly_tables, readable_markdown_html
from aso_agents.report_candidate_view import candidate_reading_view, prediction_reading_view, screening_fields
from aso_agents.report_presentation import markdown_html


def panel(sequence='ACGT', screening='complete; searched ≤2 mismatch(es); exact=0 genes; 1mm=3 recorded hits'):
    return f'''## Selected physical panel
Selected candidate rows: 1. Distinct molecules (antisense sequence plus chemistry): 1. Preview: 1 molecules in panel order.
| Candidate | Antisense oligo (5′→3′) | Physical binding interval(s), BED0 | Genomic screening |
|---|---|---|---|
| 1 | {sequence} | [["chr1", 10, 14]] | {screening} |
'''


def test_sequence_view_preserves_strand_positions_zero_and_missing_results():
    section = _sections(panel())[0]; before = deepcopy(section)
    view = candidate_reading_view(section)
    assert 'ACGT' in view and 'chr1:10–14 (zero-based; end excluded)' in view
    assert '<dt>Exact matches</dt><dd>0 genes</dd>' in view
    assert 'Matches with 1 base difference' in view
    assert 'Matches with 2 base differences' not in view
    assert section == before
    missing = candidate_reading_view(_sections(panel(screening='unknown; searched ≤2 mismatch(es)'))[0])
    assert 'This report does not say whether the off-target search finished.' in missing
    assert 'Search for other sequences this ASO may bind' in missing and 'Exact matches' not in missing
    assert 'Recorded search limit' in missing


@pytest.mark.parametrize('sequence', ['unverified', 'target: ACGT', '<script>', '', 'A*CGT'])
def test_reading_view_never_guesses_an_aso_from_an_unverified_record(sequence):
    assert candidate_reading_view(_sections(panel(sequence))[0]) == ''


def test_unknown_screening_fields_remain_available_verbatim():
    value = 'complete; alternate evidence field: 4'
    assert screening_fields(value) is None
    assert value in candidate_reading_view(_sections(panel(screening=value))[0])


def test_missing_completion_record_is_distinct_from_missing_results_or_a_failed_search():
    unknown = screening_fields('unknown; searched ≤2 mismatch(es); exact=0 genes')
    assert unknown[0][1] == 'This report does not say whether the off-target search finished.'
    assert ('Exact matches', '0 genes') in unknown
    assert screening_fields('unavailable')[0][1] == 'This report is missing the off-target search result.'
    assert screening_fields('failed')[0][1] == 'The off-target search failed.'
    assert screening_fields('complete')[0][1] == 'The off-target search finished.'


def test_grouping_keeps_every_table_once_and_each_source_anchor():
    text = panel() + '''\n## Physical chemistry
Unresolved chemistry.
## Recommended shortlist
| Order | ASO (5′→3′) | Off-target screen |
|---|---|---|
| 1 | ACGT | incomplete_or_unavailable |
## Top Normal-Ranking Candidates
| candidate | score |
|---|---|
| candidate_1 | 0 |
## Unexpected result
An unrecognized finding remains visible.
'''
    page = readable_markdown_html(text, context={'kind':'design'})
    assert page.count('<table>') == markdown_html(text).count('<table>')
    assert all(page.count(f'id="reader-section-{i}"') == 1 for i in range(5))
    assert 'id="reader-rationale"' in page and 'id="reader-other-designs"' in page
    assert 'Unresolved chemistry.' in page and '<h2>Unexpected result</h2>' in page
    assert page.index('The search is incomplete') < page.index('class="reader-candidate"')


def test_missing_counts_and_motif_codes_are_explained_only_in_their_columns():
    text = '''| id | exact off-target count | Motif flags |
|---|---|---|
| None | None | ['lna_383_tgc', 'custom_rule'] |
| lna_383_tgc | 0 | [] |
'''
    page = friendly_tables(markdown_html(text))
    assert '<td>None</td>' in page and '<td>lna_383_tgc</td>' in page
    assert 'data-recorded-value="None"' in page and '>Not reported</abbr>' in page
    assert '<td>0</td>' in page
    assert 'TGC motif (study in LNA 3–8–3 gapmers); custom_rule' in page
    assert 'None flagged' in page


def test_critical_report_identity_and_alerts_exist_without_javascript():
    from aso_agents.html_report import render_goal_intake_html
    payload = {'meta':{'gene_symbol':'GENE','gene_id':'ID.1','user_request':'Recorded request <literal>'},
               'overview':{'alerts':['Missing required reference <literal>'], 'report_markdown':panel()}}
    before = deepcopy(payload)
    page = render_goal_intake_html(payload)
    visible = re.sub(r'<script\b[^>]*>.*?</script>', '', page, flags=re.S)
    assert '<h1 id="hero-title">GENE · ASO results</h1>' in visible
    assert 'Recorded request &lt;literal&gt;' in visible
    assert 'Missing required reference &lt;literal&gt;' in visible
    assert payload == before
    saved = json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S)[1])
    assert saved == payload


def test_portable_files_distinguish_missing_files_from_large_available_files(tmp_path):
    from aso_agents.standalone_report import standalone_html
    (tmp_path/'genome.fa').write_text('reference')
    source = tmp_path/'report.html'
    page, _ = standalone_html('<html><body><a href="genome.fa">Genome</a><a href="absent.csv">Missing data</a></body></html>',
                              source_path=source, source_root=tmp_path)
    assert 'genome.fa — Reference or model file' in page
    assert 'absent.csv — File unavailable' in page
    assert page.count('<summary>Files and download details</summary>') == 1


def test_prediction_contexts_are_grouped_without_averaging_or_merging_conditions():
    text = '''## OligoAI gapmer efficacy prediction
| Candidate | ASO (5′→3′) | Reference / context | Dose (nM) / delivery | Predicted inhibition (%) | Status |
|---|---|---|---|---|---|
| candidate_1 | ACGT | reference_a | 100 / Lipofection (dose assumed reference scenario; delivery assumed reference scenario) | 0.00 | predicted |
| candidate_1 | ACGT | reference_b | 100 / Lipofection (dose assumed reference scenario; delivery assumed reference scenario) | 0.00 | predicted |
| candidate_1 | ACGT | reference_a | 50 / Lipofection (dose specified; delivery specified) | 20.00 | predicted |
'''
    section = _sections(text)[0]; before = deepcopy(section)
    view = prediction_reading_view(section)
    assert view.count('class="reader-prediction"') == 2
    assert '0.00% predicted RNA inhibition' in view and '20.00% predicted RNA inhibition' in view
    assert '100 nM with lipofection; reference assay conditions assumed' in view
    assert '2 RNA reference records.' in view
    assert section == before
    page = readable_markdown_html(text)
    assert page.count('<table>') == 1 and 'reference_a' in page and 'reference_b' in page
    changed = prediction_reading_view(_sections(text.replace('| 0.00 | predicted |', '| 8.00 | predicted |', 1))[0])
    assert 'Predictions differ across reference contexts.' in changed
    assert '4.00%' not in changed


def test_explicit_finding_leads_evaluation_without_generic_preamble():
    text = '''## Main finding
The second input needs a complete search.
## Molecule overview
| ASO | Evaluated molecule (5′→3′) | Exact antisense records | Screening status |
|---|---|---|---|
| A | ACGT | 1 | complete |
| B | TGCA | 0 | unavailable |
'''
    page = readable_markdown_html(text)
    assert 'This report assesses' not in page
    assert page.index('The second input needs') < page.index('Counts and search coverage')
    assert 'Off-target search results are unavailable or incomplete' in page


def test_older_assessment_table_with_no_search_status_still_renders():
    text = '''## Assessment
| ASO | Recomputed properties | Assessment | Exact antisense records |
|---|---|---|---|
| A | 4 nt | Search completion was not recorded. | 0 |
'''
    page = readable_markdown_html(text)
    assert 'Search completion was not recorded.' in page
    assert '<td>0</td>' in page
    assert 'no recorded exact match in the completed search' not in page
