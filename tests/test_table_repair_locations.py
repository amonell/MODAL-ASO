from copy import deepcopy

import pytest

from aso_agents.tabular_reports import render_report_tables, _has_manual_markdown_table


def test_error_identifies_tables_in_requested_report_not_only_summary(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("id,value\na,1\nb,2\n")
    export = tmp_path / "measurements.csv"
    export.write_text(source.read_text())
    summary = tmp_path / "summary.md"
    summary.write_text("# Summary\n\n{{table:values}}\n")
    report = tmp_path / "requested_report.md"
    report.write_text("# Report\n\n| Record | Value |\n|---|---|\n| a | 1 |\n| b | 2 |\n"
                      "\n| Label | Measurement |\n|---|---|\n| a | 1 |\n| b | 2 |\n")
    result = {"report_tables": [{"id": "values", "source_path": str(export),
                                  "columns": ["id", "value"]}]}
    originals = {p: p.read_bytes() for p in [source, export, report, summary]}
    with pytest.raises(ValueError, match="unbound") as caught:
        render_report_tables(deepcopy(result), [export, report], summary,
                             required=True, source_tables=[source])
    message = str(caught.value)
    assert str(report) in message
    assert '"header_line_1based": 3' in message
    assert '"header_line_1based": 8' in message
    assert 'Record' in message and 'Measurement' in message
    assert '"row_count": 2' in message
    assert "{{table:id}}" in message and "every" in message
    assert all(p.read_bytes() == data for p, data in originals.items())
    # Correct the actual offending destinations, preserving all observations.
    report.write_text("# Report\n\n{{table:values}}\n")
    render_report_tables(result, [export, report], summary, required=True, source_tables=[source])
    assert "| a | 1 |" in report.read_text()
    assert "| b | 2 |" in report.read_text()
    assert "{{table:" not in summary.read_text()
    assert export.read_bytes() == originals[export]


def test_source_only_branch_reports_every_unbound_destination(tmp_path):
    source = tmp_path / "native.csv"
    source.write_text("record,value\na,1\n")
    summary = tmp_path / "summary.md"
    report = tmp_path / "details.md"
    table = "| Feature | Observation |\n|---|---|\n| region | measured |\n"
    summary.write_text(table)
    report.write_text("# Details\n\n" + table)
    before = {p: p.read_bytes() for p in [summary, report]}
    with pytest.raises(ValueError, match="unbound") as caught:
        render_report_tables({}, [report], summary, required=True, source_tables=[source])
    message = str(caught.value)
    assert str(summary) in message and str(report) in message
    assert '"header_line_1based": 1' in message
    assert '"header_line_1based": 3' in message
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("text, expected", [
    ("| A | B |\n|---|---|\n| 1 | 2 |\n", True),
    ("  | A | B |\n  |:--|--:|\n  | 1 | 2 |\n", True),
    ("| A | B |\n|---|---|\n", False),
    ("A | B\n1 | 2\n", False),
    ("{{table:bound}}\n", False),
])
def test_existing_table_detection_boundary_is_preserved(text, expected):
    assert _has_manual_markdown_table(text) is expected
