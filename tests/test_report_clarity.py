"""Keep a simpler reading path faithful to identities, scopes and missing data."""
from copy import deepcopy
import json

from aso_agents.report_assessment import evidence_gaps, motif_notes, prediction_note, render_shortlist_markdown
from aso_agents.report_candidate_view import candidate_reading_view, panel_check_records, screening_fields
from aso_agents.report_orientation import build_orientation, design_reader_context
from aso_agents.readable_report import _sections, design_downloads, readable_markdown_html


PANEL = '''## Selected physical panel
Selected candidate rows: 1. Distinct molecules (antisense sequence plus chemistry): 1. Preview: 1 molecules in panel order.
| Candidate | Antisense oligo (5′→3′) | Physical binding interval(s), BED0 | Genomic screening |
|---|---|---|---|
| 1 | ACGT | [["chr1", 10, 14]] | complete; searched ≤2 mismatch(es); exact=0 loci; 1mm=0 recorded hits; 2mm=7 recorded hits |
'''


def record():
    return {'recommendation_order': 1, 'antisense_5to3': 'ACGT', 'recommendation_status': 'provisional',
            'off_target_status': 'complete_within_declared_scope', 'safeguard_status': 'not_requested',
            'reference_tm_c': 0, 'reference_delta_g_kcal_mol': None, 'motif_flags': ['lna_383_tgc'],
            'assessment': {'unavailable_evidence': ['target_RNA_accessibility'],
                'motifs': {'flags': [{'rule_id': 'lna_383_tgc', 'applicability': 'outside_studied_architecture'}]}}}


def test_shortlist_names_missing_checks_and_motif_scope_without_changing_records():
    source = {'records': [record()]}; before = deepcopy(source)
    text = render_shortlist_markdown(source, 103)
    assert 'prediction of how exposed the target RNA site is' in text
    assert 'Some evidence unavailable' in text and 'Completed' in text
    assert 'TGC motif (study in LNA 3–8–3 gapmers); this ASO uses a different chemistry' in text
    assert '1 distinct molecules selected; 103 in the assessed eligible pool.' in text
    assert '| 0 | unavailable |' in text
    assert source == before


def test_missing_assessment_is_distinct_from_an_explicit_empty_gap_list():
    assert evidence_gaps({}) == 'Missing-check details unavailable'
    assert evidence_gaps({'assessment': {'unavailable_evidence': []}}) == 'None recorded'
    assert evidence_gaps({'assessment': {'unavailable_evidence': ['custom_check']}}) == 'custom_check'
    assert motif_notes({}) == 'Motif results unavailable'
    assert motif_notes({'motif_flags': []}) == 'None flagged'


def test_selection_checks_keep_evidence_without_repeating_recommendation_prose():
    source = {'review': {'summary': 'The panel samples several locations.'},
              'records': [{**record(), 'recommendation_review': {
                  'why_chosen': 'Reason already shown on the recommendation card.',
                  'tradeoff': 'Saved tradeoff.', 'panel_role': 'Saved role.'}}]}
    before = deepcopy(source)
    full = render_shortlist_markdown(source, 103)
    checks = render_shortlist_markdown(source, 103, include_rationale=False)
    assert checks.startswith('## Selection checks\n')
    assert 'Checks for 1 recommended ASO.' in checks
    assert 'Reason already shown' in full and 'Reason already shown' not in checks
    assert 'The panel samples' not in checks
    assert [line for line in checks.splitlines() if line.startswith('|')] == [
        line for line in full.splitlines() if line.startswith('|')]
    assert 'prediction of how exposed the target RNA site is' in checks
    assert '1 M sodium' in checks
    assert source == before


def test_prediction_review_uses_recorded_failure_and_keeps_other_predictions_distinct():
    source = record()
    source['assessment']['native_prediction'] = {'directional_status': 'opposing_predictions'}
    source['assessment']['splice_safeguards'] = {'status': 'failed'}
    note = prediction_note(source)
    assert 'opposite direction' in note and 'exceed the specified limits' in note
    assert prediction_note(record()) == ''


def test_selected_sequence_and_its_checks_have_one_unambiguous_panel_order():
    sections = _sections(PANEL + render_shortlist_markdown({'records': [record()]}, 103).replace('# Recommended', '## Recommended'))
    before = deepcopy(sections)
    checks = panel_check_records(sections[0], sections)
    view = candidate_reading_view(sections[0], checks=checks)
    assert 'Missing: prediction of how exposed the target RNA site is.' in view
    assert 'Matches with 2 base differences' in view and '7 recorded hits' in view
    assert 'Exact matches: 0 loci.' in view and '0 genes' not in view
    assert 'different chemistry' in view
    assert sections == before
    # A different order or sequence cannot silently borrow this evidence.
    for replacement in ['| 2 | ACGT |', '| 1 | TGCA |']:
        changed = _sections(PANEL + render_shortlist_markdown({'records': [record()]}, 103).replace('# Recommended', '## Recommended').replace('| 1 | ACGT |', replacement))
        assert panel_check_records(changed[0], changed) == []


def test_old_summary_marks_absent_bins_without_inventing_zero_or_a_search_failure():
    fields = dict(screening_fields('complete; searched ≤2 mismatch(es); exact=0 loci; 1mm=3 recorded hits'))
    assert fields['Matches with 2 base differences'] == 'Count absent from this summary'
    assert fields['Search for other sequences this ASO may bind'] == 'The off-target search finished.'
    assert fields['Exact matches'] == '0 loci'
    assert screening_fields('complete; unknown evidence=2') is None


def test_intron_goal_uses_typed_event_and_alternative_goal_keeps_full_objective():
    goal = {'design_programs': [{'intent': 'splice_modulation', 'splice_event': {
        'event_type': 'intron_retention', 'intron_number': 3, 'reference_transcript_id': 'NM_TEST.1'}}]}
    before = deepcopy(goal)
    context = design_reader_context(goal)
    orientation = build_orientation(_sections(PANEL), {**context, 'kind': 'design', 'gene': 'TEST'})
    assert orientation['purpose'] == 'Goal: help TEST RNA retain intron 3 during splicing.'
    assert goal == before
    goal['design_programs'].append({'intent': 'splice_modulation'})
    assert 'splice_event' not in design_reader_context(goal)
    context = {'kind': 'design', 'intent': 'splice_modulation', 'desired_effect': 'inclusion',
               'isoform_label': 'alternate product', 'objective': 'Favor the alternate acceptor and preserve the donor.'}
    assert build_orientation(_sections(PANEL), context)['purpose'] == 'Goal: Favor the alternate acceptor and preserve the donor.'


def test_incomplete_shortlist_remains_an_opening_notice_with_human_status_labels():
    source = record(); source['off_target_status'] = 'incomplete_or_unavailable'
    text = PANEL + render_shortlist_markdown({'records': [source]}, 103).replace('# Recommended', '## Recommended')
    page = readable_markdown_html(text, context={'kind': 'design'})
    assert page.index('The search is incomplete for all 1 ASOs') < page.index('class="reader-candidate"')


def test_missing_consequence_is_specific_to_that_calculation_and_links_to_its_source():
    text = PANEL + '\n## Conditional exon-skipping sequence consequences\n' + json.dumps([
        {'status': 'unresolved_reference', 'reason': 'Transcript consequence lookup failed'}])
    summary = build_orientation(_sections(text), {'kind': 'design'})
    assert summary['notices'] == ['Exon-skipping sequence consequences are unavailable for 1 requested scenario.']
    assert summary['notice_sources'][summary['notices'][0]] == 'reader-section-1'
    assert 'No ASOs' not in summary.get('title', '')


def test_panel_opens_with_chemistry_once_and_keeps_unknown_findings_visible():
    text = PANEL + '\n## Physical chemistry\nSugars: 4 DNA.\n\nFull chemistry information.\n\n## Unexpected finding\nA substantive finding.'
    page = readable_markdown_html(text, context={'kind': 'design'})
    assert 'id="reader-section-0" open' in page
    assert page.count('id="reader-section-1"') == 1
    assert page.count('Sugars: 4 DNA.') == 1 and page.count('Full chemistry information.') == 1
    assert '<h2>Unexpected finding</h2>' in page


def test_downloads_ignore_archived_copies_and_keep_current_output_links_static():
    from aso_agents.html_report import _select_primary_artifacts, render_goal_intake_html
    paths = ['old.md.assets/hash/files/artifacts/candidates/design_panel.csv',
             'artifacts/candidates/design_panel.csv', 'report.md',
             'artifacts/candidates/generated_aso_set.json']
    items = [{'relative_path': path, 'href': path, 'size_label': '2 KB'} for path in paths]
    selected = _select_primary_artifacts({'items': items})
    assert len(selected) == 3 and all('.assets/' not in item['href'] for item in selected)
    payload = {'overview': {'report_markdown': PANEL, 'primary_artifacts': selected}}
    page = render_goal_intake_html(payload)
    assert 'href="artifacts/candidates/design_panel.csv"' in page
    assert 'id="reader-output-files"' in page and 'href="#reader-output-files"' in page
    assert 'artifactsRoot.innerHTML' not in page
    assert '2 KB' in page
    payload['overview']['primary_artifacts'] = [{'href': 'javascript:alert(1)', 'label': 'Bad'}, selected[0], selected[0]]
    fragment = design_downloads(payload)
    assert 'javascript:' not in fragment and fragment.count('href=') == 1


def test_design_panel_exports_every_available_mismatch_bin(tmp_path):
    from aso_agents.design_delivery import write_design_panel
    import csv
    row = {'synthesized_antisense_sequence': 'ACGT', 'sequence_role': 'antisense_5to3',
           'off_target_search_status': 'complete', 'Off_target_mismatch_budget': 2,
           'Off_target_exact_count': 0, 'Off_target_1mm_count': 3, 'Off_target_2mm_count': 9,
           'off_target_exact_count_unit': 'loci'}
    original = deepcopy(row)
    write_design_panel({'rows': [row]}, {}, tmp_path)
    text = (tmp_path/'design_panel.md').read_text()
    assert 'exact=0 loci; 1mm=3 recorded hits; 2mm=9 recorded hits' in text
    assert '3mm=' not in text
    with (tmp_path/'design_panel.csv').open() as handle:
        saved = list(csv.DictReader(handle))[0]
    assert saved['off_target_2mm_count'] == '9'
    assert row == original


def test_repeated_binding_records_do_not_make_a_complete_molecule_preview_look_partial():
    text = PANEL.replace('Selected candidate rows: 1.', 'Selected candidate rows: 7.')
    summary = build_orientation(_sections(text), {'kind': 'design'})
    assert summary['title'] == '1 ASO'
    assert not summary['meaning']


def test_equal_card_summaries_are_shown_once_and_different_results_stay_per_aso():
    text = PANEL.replace('rows: 1.', 'rows: 2.').replace('chemistry): 1.', 'chemistry): 2.').replace('Preview: 1', 'Preview: 2')
    text += '| 2 | TGCA | [["chr1", 20, 24]] | complete; searched ≤2 mismatch(es); exact=0 loci; 1mm=0 recorded hits; 2mm=7 recorded hits |\n'
    page = candidate_reading_view(_sections(text)[0])
    assert page.count('For each ASO shown:') == 1
    assert page.count('Exact matches: 0 loci.') == 1
    assert page.count('<dt>Exact matches</dt><dd>0 loci</dd>') == 2
    different = candidate_reading_view(_sections(text.replace('exact=0 loci', 'exact=1 loci', 1))[0])
    assert 'For each ASO shown:' not in different
    assert 'Exact matches: 0 loci.' in different and 'Exact matches: 1 loci.' in different


def test_binding_region_and_response_site_keep_their_separate_roles():
    text = PANEL.replace('| Genomic screening |', '| Physical splice element | Predicted response site | Genomic screening |')
    text = text.replace('|---|---|---|---|', '|---|---|---|---|---|---|')
    text = text.replace('| complete;', '| exon boundary | {"site_kind": "acceptor", "coordinate": 40} | complete;')
    view = candidate_reading_view(_sections(text)[0])
    assert 'chr1:10–14' in view and 'RNA region covered' in view and 'exon boundary' in view
    assert 'Splice site whose effect was scored' in view and 'acceptor' in view and '40' in view
    assert '{&quot;site_kind&quot;' not in view


def test_mixed_chemistry_panel_does_not_show_one_composition_as_the_panel_identity():
    text = PANEL + '\n## Physical chemistry\nSugars (5′→3′): 4 DNA.\n\nSugars (5′→3′): 4 2′-MOE.\n'
    page = readable_markdown_html(text, context={'kind': 'design'})
    assert 'This panel includes multiple chemistry specifications.' in page
    details = page.index('<summary>View the chemistry specifications</summary>')
    assert page.index('Sugars (5′→3′): 4 DNA.') > details
    assert page.index('Sugars (5′→3′): 4 2′-MOE.') > details
