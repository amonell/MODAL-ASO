from pathlib import Path

import pytest

from aso_agents.metric_report_integrity import canonicalize_metric_notes
from aso_agents.tabular_reports import render_report_tables


def _fixture(tmp_path: Path):
    source = tmp_path / "audit.csv"
    source.write_text(
        "id,sequence,reported_length,reported_gc_fraction,computed_length,computed_gc_fraction\n"
        "A,GGGGGGGGGGGGGAAAAAAA,20,0.1,20,0.65\n"
        "B,ACGTACGTACGTACGTACG,19,0.526316,19,0.52631578947368421052631578947368421052631578947368\n"
    )
    return source


def test_gc_comparison_note_uses_exact_sequence_fraction_and_keeps_reported_values(tmp_path):
    source = _fixture(tmp_path)
    report = tmp_path / "report.md"
    summary = tmp_path / "summary.md"
    report.write_text("## Notes\n\n- Reported vs computed GC: A reports 0.1 but computed is 0.6; "
                      "B reports 0.526316 but computed is 0.5.\n")
    summary.write_text("Completed analysis.\n")
    result = {}
    render_report_tables(result, [source, report], summary, required=True)
    actual = report.read_text()
    assert "computed is 0.6" not in actual
    assert "| A | GC fraction | 0.1 | 13/20 (0.65) | different |" in actual
    assert "| B | GC fraction | 0.526316 | 10/19 | rounded |" in actual
    assert result["metric_report_canonicalization"][0]["replaced_lines"]


def test_metric_repair_uses_superset_and_leaves_unrelated_note_unchanged(tmp_path):
    source = _fixture(tmp_path)
    subset = tmp_path / "retained.csv"
    subset.write_text(source.read_text().splitlines()[0] + "\n" + source.read_text().splitlines()[1] + "\n")
    untouched = "- GC calculation was performed using uppercase bases; no row comparison here.\n"
    texts, receipts = canonicalize_metric_notes({"report.md": untouched},
                                                {str(source): source, str(subset): subset})
    assert texts["report.md"] == untouched
    assert receipts == []


def test_computed_gc_must_match_sequence_before_any_narrative_repair(tmp_path):
    source = _fixture(tmp_path)
    source.write_text(source.read_text().replace("20,0.65", "20,0.6"))
    with pytest.raises(ValueError, match="Computed GC .* disagrees with the sequence"):
        canonicalize_metric_notes({"report.md": "- Reported vs computed GC: A is 0.6.\n"},
                                  {str(source): source})


def test_manual_full_and_retained_tables_can_share_columns_without_false_conflict(tmp_path):
    full = tmp_path / "full.csv"
    retained = tmp_path / "retained.csv"
    report = tmp_path / "report.md"
    summary = tmp_path / "summary.md"
    full.write_text("id,score,status\nA,1,pass\nB,0,fail\n")
    retained.write_text("id,score,status\nA,1,pass\n")
    report.write_text("| id | score | status |\n|---|---|---|\n"
                      "| A | 1 | pass |\n| B | 0 | fail |\n\n"
                      "| id | score | status |\n|---|---|---|\n| A | 1 | pass |\n")
    summary.write_text("Analysis complete.\n")
    render_report_tables({}, [full, retained, report], summary, required=True)
    assert "| B | 0 | fail |" in report.read_text()

    report.write_text("| id | score | status |\n|---|---|---|\n| A | 9 | pass |\n")
    summary.write_text("Analysis complete.\n")
    with pytest.raises(ValueError, match="cannot be verified against the export"):
        render_report_tables({}, [full, retained, report], summary, required=True)
