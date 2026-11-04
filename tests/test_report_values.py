"""Evidence stays exact while every reading depth uses human notation."""
from copy import deepcopy
from html.parser import HTMLParser
import json
import shutil
import subprocess

import pytest

from aso_agents.readable_report import readable_markdown_html
from aso_agents.report_values import (
    bed_positions, browser_value_renderer, parse_collection, readable_prose,
    value_html, value_text,
)


class Reading(HTMLParser):
    def __init__(self, page):
        super().__init__(convert_charrefs=True)
        self.text, self.recorded, self.tags = [], [], []
        self.feed(page)

    def handle_data(self, data):
        self.text.append(data)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        if 'data-recorded-value' in attrs:
            self.recorded.append(attrs['data-recorded-value'])


def test_full_tables_render_positions_and_response_records_without_json():
    position = '[["chr2", 0, 18], ["chr2", 30, 48]]'
    response = '{"site_kind": "donor", "position": 18, "score": 0, "verified": false}'
    page = readable_markdown_html(
        '| Physical binding interval(s), BED0 | Predicted response site |\n'
        '|---|---|\n' + f'| {position} | {response} |\n')
    reading = Reading(page)
    text = ' '.join(reading.text)
    assert 'chr2:0–18; chr2:30–48 (zero-based; end excluded)' in text
    assert 'Site kind' in text and 'donor' in text and 'Verified' in text and 'No' in text
    assert '[' not in text and '{' not in text
    assert position in reading.recorded and response in reading.recorded


def test_coordinates_require_explicit_convention_and_keep_bed6_metadata():
    value = [['chr3', 0, 5, 'supplied region', 0, '-']]
    assert bed_positions(value, 'intervals_bed0_half_open') == 'chr3:0–5; name: supplied region; score: 0; strand: - (zero-based; end excluded)'
    assert bed_positions(value, 'unknown interval type') is None
    assert bed_positions([['chr3', 5, 0]], 'bed0') is None
    assert bed_positions([['chr3', True, 5]], 'bed0') is None
    assert 'chr3:0–5' not in value_text(value, 'unknown interval type')


@pytest.mark.parametrize('wrapper', ['<p>{}</p>', '<p><code>{}</code></p>', '<pre><code>{}</code></pre>'])
def test_nested_evidence_is_expandable_and_values_are_preserved(wrapper):
    raw = json.dumps({'status': 'native_reference', 'target_id': 'full_containment',
                      'evidence': [{'score': 0, 'checked': False, 'missing': None, 'sites': ['A', 'A']} ]})
    from html import escape
    page = readable_prose(wrapper.format(escape(raw)))
    reading = Reading(page)
    text = ' '.join(reading.text)
    assert 'Already on the requested reference assembly' in text
    assert 'full_containment' in text  # identity must never become a status label
    assert 'Not reported' in text and 'No' in text
    assert reading.text.count('A') == 2
    assert 'details' in reading.tags and 'dl' in reading.tags
    assert '[' not in text and '{' not in text
    assert raw in reading.recorded and json.loads(reading.recorded[0]) == json.loads(raw)


def test_inline_lists_use_real_punctuation_for_copy_and_screen_readers():
    page = readable_prose('<p>Sites: ["alpha", "beta"].</p>')
    assert ''.join(Reading(page).text) == 'Sites: alpha; beta.'


def test_empty_records_missing_values_and_false_remain_distinct():
    assert value_text([]) == 'None listed'
    assert value_text({}) == 'No details recorded'
    assert value_text(None) == 'Not reported'
    assert value_text(0) == '0'
    assert value_text(False) == 'No'
    assert value_text('None', 'transcript_id') == 'None'


def test_prose_codes_and_file_paths_are_readable_with_originals_retained():
    page = readable_prose('<p>Rule: <code>any_overlap</code>. '
                          'Evidence: <code>artifacts/candidates/generated_aso_set.json</code>. '
                          'Genome: /mnt/references/assembly.fasta.</p>')
    reading = Reading(page)
    text = ''.join(reading.text)
    assert 'ASO binding site overlaps the region' in text
    assert 'Complete ASO records (JSON)' in text and 'assembly.fasta' in text
    assert '/mnt/' not in text and 'artifacts/' not in text
    assert 'any_overlap' in reading.recorded
    assert '/mnt/references/assembly.fasta' in reading.recorded
    policy = 'top_ranked_after_required_evidence_and_constraints_per_site'
    reading = Reading(readable_prose('<p>Selection policy: ' + policy + '. File: ' + policy + '.json.</p>'))
    assert 'Highest-ranked ASOs at each site after the required checks and filters.' in ''.join(reading.text)
    assert policy + '.json.' in ''.join(reading.text)


def test_chemical_notation_code_examples_and_malformed_records_stay_literal():
    source = '<p><code>/52MOErA/*A*T*/32MOErG/</code> [LNA] A[5mC]G [0, 5)</p><pre><code>print({"status": "unknown"})</code></pre>'
    assert readable_prose(source) == source
    assert parse_collection('[{"invalid":') is None
    assert parse_collection('[__import__("os").getcwd()]') is None
    raw = '{"label": "<img src=x onerror=alert(1)>"}'
    assert '<img' not in value_html(parse_collection(raw))


def test_unknown_fields_get_labels_without_rewriting_identifiers_or_assuming_units():
    page = readable_markdown_html('| transcript_id | new_score | status |\n|---|---|---|\n'
                                  '| native_reference | 0.000001 | native_reference |')
    reading = Reading(page)
    assert 'Transcript ID' in reading.text and 'New score' in reading.text
    assert 'native_reference' in reading.text and '0.000001' in reading.text
    assert 'Already on the requested reference assembly' in reading.text


def test_python_and_browser_preserve_nested_values_with_the_same_labels():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to check the browser formatter')
    cases = [
        {'status': 'native_reference', 'target_id': 'full_containment', 'values': [0, False, None, 'A', 'A']},
        {'intervals_bed0_half_open': [['chr3', 0, 5, 'region', 0, '-']]},
        {'coordinates': [['chr3', 0, 5]], 'novel_value': 0.000001},
        {'matches': "['GENE_A', 'GENE_B']", 'transcript_id': '["literal"]'},
        {'small_delta': -1.5e-12, 'zero': -0.0, 'whole': 1.0, 'toString': 'toString'},
        [], {}, {'html': '<script>untrusted</script>'},
    ]
    before = deepcopy(cases)
    script = ('const escapeHtml = value => String(value).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");\n'
              + browser_value_renderer(cases) + '\nconst cases = ' + json.dumps(cases) + ';\n'
              + 'console.log(JSON.stringify(cases.map(value => ({text:formatRecordValue(value), html:renderRecordValue(value)}))));')
    result = subprocess.run([node, '-e', script], text=True, capture_output=True, check=True)
    for source, rendered in zip(cases, json.loads(result.stdout)):
        assert rendered['text'] == value_text(source)
        assert '[object Object]' not in rendered['html'] and '<script>' not in rendered['html']
    assert cases == before


def test_evaluation_evidence_page_has_readable_records_and_exact_json(tmp_path):
    from aso_agents.evaluation_report import write_evaluation_evidence
    response = {'evaluation_summary': {'provided_asos': [
        {'name': 'A', 'input_row_id': 'source_A', 'score': 0, 'gene_names': "['GENE_A', 'GENE_B']"}]}}
    before = deepcopy(response)
    paths = write_evaluation_evidence(tmp_path, response)
    from pathlib import Path
    page = Path(paths['evaluation_evidence_html']).read_text()
    record = json.loads(Path(paths['evaluation_evidence_json']).read_text())
    assert record['provided_asos'] == before['evaluation_summary']['provided_asos']
    assert '<section id="row-source_A">' in page
    assert '<pre>' not in page and '<dl class="reader-value-record">' in page
    text = ' '.join(Reading(page).text)
    assert "['GENE_A'" not in text and 'GENE_A' in text and 'GENE_B' in text
    assert response == before
