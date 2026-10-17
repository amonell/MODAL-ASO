"""Exercise the rendered table with distinct genomic and transcript counts."""
import json
import re
import subprocess

import pytest

from aso_agents.html_report import render_goal_intake_html


def render_table(rows):
    html = render_goal_intake_html({})
    names = ['escapeHtml', 'humanizeKey', 'formatValue', 'formatExonObjectiveEvidence', 'getAsoColumns', 'renderAsoTables']
    functions = []
    for name in names:
        match = re.search(r'      function ' + name + r'\(.*?\n      \}', html, re.S)
        assert match, name
        functions.append(match.group())
    target = {'recommended_asos': rows}
    script = '\n'.join(functions) + '\nprocess.stdout.write(renderAsoTables(' + json.dumps(target) + '));'
    return subprocess.check_output(['node', '-'], input=script, text=True)


def test_mapping_count_alone_does_not_assert_isoform_coverage():
    html = render_table([{'sequence': 'AACCTT', 'times_mapped': 3}])
    assert '<th>Mappings (source reference)</th>' in html
    assert 'Target isoforms' not in html
    assert '>3</td>' in html


@pytest.mark.parametrize('key', ['isoform_hit_count', 'isoform_uniqueness_count'])
def test_distinct_reference_mapping_and_isoform_counts_remain_distinct(key):
    html = render_table([{'sequence': 'AACCTT', 'times_mapped': 3, key: 8}])
    assert '<th>Mappings (source reference)</th>' in html
    assert '<th>Target isoforms</th>' in html
    assert '>3</td>' in html and '>8</td>' in html


def test_zero_mapping_is_not_replaced_with_a_missing_value():
    html = render_table([{'sequence': 'AACCTT', 'times_mapped': 0}])
    assert '>0</td>' in html
    assert 'N/A' not in html


def test_explicit_isoform_count_needs_no_legacy_mapping_count():
    html = render_table([{'sequence': 'AACCTT', 'isoform_hit_count': 8}])
    assert '<th>Target isoforms</th>' in html
    assert 'Mappings (source reference)' not in html


def test_unknown_counts_are_not_invented_from_sequence_or_coordinates():
    html = render_table([{'sequence': 'AACCTT', 'start': 17, 'end': 23}])
    assert 'Target isoforms' not in html and 'Mappings (source reference)' not in html
