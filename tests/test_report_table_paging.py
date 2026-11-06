import base64
import csv
from html import escape
import io
import json
import re

from aso_agents.report_tables import PAGE_SIZE, page_large_tables


def table(count=80):
    rows = [f'<tr id="candidate-{i}"><td>candidate_{i}</td><td>{i}</td>'
            '<td><abbr data-recorded-value="unknown">Not reported</abbr></td></tr>'
            for i in range(count)]
    return '<table><thead><tr><th>Candidate</th><th>Matches</th><th>Status</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>', rows


def data(page):
    return json.loads(re.search(r'<script type="application/json" class="reader-table-data">(.*?)</script>',page,re.S)[1])


def test_large_table_preserves_all_records_order_zero_and_source_values():
    source, rows = table(1000)
    page = page_large_tables(source)
    assert [record['html'] for record in data(page)] == rows
    visible = re.sub(r'<script.*?</script>', '', page, flags=re.S)
    assert visible.count('<tr id=') == PAGE_SIZE
    assert 'Rows 1–25 of 1,000' in visible
    assert data(page)[999]['anchors'] == ['candidate-999']
    assert 'Not reported' in data(page)[999]['search']
    assert 'unknown' in data(page)[999]['search']
    download = re.search(r'href="data:text/csv;base64,([^"]+)"',page)[1]
    records = list(csv.reader(io.StringIO(base64.b64decode(download).decode('utf-8-sig'))))
    assert len(records) == 1001
    assert records[1] == ['candidate_0','0','unknown']
    assert records[-1] == ['candidate_999','999','unknown']
    assert page_large_tables(page) == page


def test_small_and_merged_tables_retain_the_original_view():
    small, _ = table(PAGE_SIZE)
    assert page_large_tables(small) == small
    merged, _ = table(80)
    merged = merged.replace('<td>0</td>', '<td rowspan="2">0</td>')
    assert page_large_tables(merged) == merged


def test_embedded_records_cannot_close_the_data_script():
    source, _ = table(80)
    literal = '</script><script>alert("test")</script>'
    source = source.replace('candidate_79', escape(literal))
    page = page_large_tables(source)
    assert literal not in page
    assert literal in data(page)[-1]['search']
    assert data(page)[-1]['html'].startswith('<tr id="candidate-79">')


def test_repeated_tables_have_independent_controls():
    source, _ = table()
    page = page_large_tables(source + source)
    identifiers = re.findall(r'class="reader-paged-table" id="([^"]+)"', page)
    assert len(identifiers) == len(set(identifiers)) == 2
    assert page.count('class="reader-table-download"') == 2
