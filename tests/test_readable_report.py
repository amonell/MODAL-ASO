"""Reader changes must preserve scientific content, uncertainty and exports."""
from copy import deepcopy
from html.parser import HTMLParser
import json
import re

from aso_agents.readable_report import (
    _sections, basic_document, composite_document, design_results, readable_markdown_html,
)
from aso_agents.report_presentation import markdown_html


class Content(HTMLParser):
    def __init__(self, page):
        super().__init__(convert_charrefs=True)
        self.cells, self.code, self.tags, self.stack = [], [], [], []
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag in {'td', 'code'}:
            self.stack.append([tag, ''])

    def handle_data(self, data):
        for entry in self.stack:
            entry[1] += data

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1][0] == tag:
            _, value = self.stack.pop()
            (self.cells if tag == 'td' else self.code).append(value)


def test_body_values_and_chemistry_remain_exact_with_plainer_headers():
    text = '''# Sandbox summary

No activity was measured. Score 0 is a result; unavailable is missing.

## Method

Do not compare ranks across contexts.

```text
/5MOErA/*T*C*G
## NOT A SECTION
<script>literal only</script>
```

## Outcome summary

| id | sequence | source_row | target_score | offtarget_hits |
|---|---|---|---|---|
| first | /5MOErA/*T*C*G | 2 | 0 | unavailable |
| second | AACCGG | 3 | -0.25 | 0 |

## Caveats

The reference is incomplete; zero matches does not prove specificity.
'''
    original = markdown_html(text)
    page = readable_markdown_html(text)
    assert Content(page).cells == Content(original).cells
    assert Content(page).code == Content(original).code
    assert page.index('Results at a glance</h2>') < page.index('<summary>Method</summary>')
    assert '<h2>Limitations</h2>' in page
    assert 'zero matches does not prove specificity.' in page
    assert '<script>literal only</script>' not in page
    assert 'title="target_score">Target score</th>' in page


def test_unknown_sections_and_unknown_fields_are_visible_not_reinterpreted():
    text = '# Results\n\n## Unfamiliar scientific finding\n\nNever infer a dose from these results.\n\n| id | new_measure |\n|---|---|\n| unknown | NA |'
    page = readable_markdown_html(text)
    assert '<h2>Unfamiliar scientific finding</h2>' in page
    assert '<th>new_measure</th>' in page and '<td>NA</td>' in page
    assert '<summary>Unfamiliar scientific finding' not in page


def test_status_and_reason_codes_are_explained_only_in_their_own_columns():
    text = '| id | outcome | Exclusion reason | score |\n|---|---|---|---|\n| completed_no_candidates | completed_no_candidates | length_not_exactly_37_nt | 0 |'
    page = readable_markdown_html(text)
    assert '<td>completed_no_candidates</td>' in page  # identity is untouched
    assert 'data-recorded-value="completed_no_candidates"' in page
    assert '>No candidates produced</abbr>' in page
    assert '>Length differs from 37 bases</abbr>' in page  # no benchmark-specific threshold
    assert '<td>0</td>' in page
    assert 'Prediction unavailable' not in page


def test_rank_stays_bound_to_its_recorded_group_without_a_large_internal_id():
    text = '| Candidate | Native rank / scope | ASO |\n|---|---|---|\n| a | 1 / group_alpha | ACGT |\n| b | 1 / group_beta | GGTT |'
    page = readable_markdown_html(text)
    assert 'Rank within design group' in page
    assert 'data-recorded-value="1 / group_alpha"' in page
    assert 'data-recorded-value="1 / group_beta"' in page
    assert page.count('<summary>1 · group details</summary>') == 2
    assert 'Design group: <code>group_alpha</code>' in page
    assert '<td>ACGT</td>' in page and '<td>GGTT</td>' in page


def test_all_limitations_and_missing_results_are_retained():
    from aso_agents.html_report import render_goal_intake_html
    payload = {'meta': {'gene_symbol': 'NEUTRAL'}, 'overview': {'alerts': [f'Unavailable observation {i}' for i in range(8)]}}
    page = render_goal_intake_html(payload)
    assert 'alerts.slice(' not in page
    assert 'Limitations and missing results' in page
    for i in range(8):
        assert f'Unavailable observation {i}' in page
    assert 'Final results are not recorded.' in page


def test_design_renderer_preserves_payload_exactly_including_literal_template_tokens():
    from aso_agents.html_report import render_goal_intake_html
    payload = {'meta': {'user_request': '__READER_SCRIPT__ </script><script>UNTRUSTED</script>'},
               'overview': {'report_markdown': '# Results\n\nA zero is 0; prediction unavailable.'},
               'results_html': '<p>Supporting value: 0</p><h2>Report and interpretation</h2><p>Saved assessment</p>'}
    before = deepcopy(payload)
    page = render_goal_intake_html(payload)
    embedded = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S)[1]
    assert json.loads(embedded) == before == payload
    assert '</script><script>UNTRUSTED</script>' not in page
    assert 'Supporting value: 0' in page and 'prediction unavailable.' in page
    assert 'ASOs produced during design' in page


def test_legacy_design_without_markdown_never_drops_its_results():
    assert design_results({'results_html': '<p>Unexpected finding: 0</p>'}) == '<p>Unexpected finding: 0</p>'
    page = design_results({'results_html': '<p>Legacy result</p>', 'overview': {'report_markdown': '# New layout\n\nPartial.'}})
    assert 'Legacy result' in page and 'Partial.' in page


def test_request_context_and_download_links_are_preserved(tmp_path):
    (tmp_path/'panel.csv').write_text('id,value\na,0\n')
    text = '# Run Output Summary\n\nOutput folder: `neutral`.\nContext: reference unavailable.\n\nOriginal request:\n\nCompare without reranking.\n\n## Requested output files\n\n[Panel](panel.csv)'
    page = basic_document('X'*130, text, tmp_path)
    assert '<h1>ASO analysis</h1>' in page
    assert 'Context: reference unavailable.' in page
    assert 'Compare without reranking.' in page
    assert 'href="panel.csv"' in page
    assert 'Download your files' in page
    assert (tmp_path/'panel.csv').read_text() == 'id,value\na,0\n'


def test_composite_tabs_have_distinct_labels_and_portable_links():
    entries = [{'tab_label': 'GENE · Gapmer ASOs', 'title': 'A full requested scope',
                'html_href': f'child_runs/{i}/report.html', 'md_href': f'child_runs/{i}/report.md'} for i in range(2)]
    before = deepcopy(entries)
    page = composite_document('A long request ' * 20, '# Results\n\nIncomplete for the second panel.', entries, None)
    assert '<h1>Combined ASO report</h1>' in page
    assert 'GENE · Gapmer ASOs · 1</label>' in page and 'GENE · Gapmer ASOs · 2</label>' in page
    assert "src='child_runs/0/report.html'" in page and "src='child_runs/1/report.html'" in page
    assert 'Incomplete for the second panel.' in page
    assert entries == before
    assert 'cross-run synthesis' not in page and 'scoped child' not in page


def test_fenced_headings_and_unknown_nested_sections_remain_in_their_parent():
    source = '## Method\n\n~~~~text\n## Result inside code\n~~~\nStill fenced\n~~~~\n\n### Coordinate convention\n\n0-based.\n\n## Caveats\n\nUncertain.'
    sections = _sections(source)
    assert [section['heading'] for section in sections] == ['Method', 'Caveats']
    assert '### Coordinate convention' in sections[0]['lines']
