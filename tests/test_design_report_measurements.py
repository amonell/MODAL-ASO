from copy import deepcopy
import json

import pytest

from aso_agents import pipeline_tools, report_explainer


@pytest.mark.parametrize("score", [0, 0.1, 1.0])
def test_normalized_candidate_reports_retained_score_without_changing_data(score):
    original = {"Name": "candidate", "Sequence": "AACCGGTT", "Synthesis": "AACCGGTT",
                "selection_rank": 3, "isoform_uniqueness_score": score,
                "isoform_uniqueness_class": "target_without_beneficial"}
    candidate = pipeline_tools._normalized_generated_candidate_row(original)
    before = deepcopy(candidate)
    assert report_explainer._candidate_isoform_score(candidate) == score
    table = "\n".join(report_explainer._render_candidate_table([candidate]))
    assert f"| {score:.4f} |" in table
    assert candidate == before and candidate["selection_rank"] == 3


@pytest.mark.parametrize("value", [None, "", [], {}])
def test_explicit_current_unknown_does_not_fall_back_to_source_score(value):
    row = {"isoform_uniqueness_score": value, "source_annotations": {"isoform_uniqueness_score": 0.9}}
    assert report_explainer._candidate_isoform_score(row) is None


def test_report_uses_recorded_eligibility_without_inventing_zero_cutoff(tmp_path):
    run = tmp_path / "run"; candidates = run / "artifacts/candidates"; candidates.mkdir(parents=True)
    source = {"Name": "candidate", "Sequence": "AACCGGTT", "Synthesis": "AACCGGTT",
              "selection_rank": 1, "off_target_gene_count": 2, "isoform_uniqueness_score": 0.25}
    scored = candidates / "gapmer_scored_uniqueness.json"
    scored.write_text(json.dumps({"rows": [source]}))
    (candidates / "generated_aso_set.json").write_text(json.dumps({
        "rows": [pipeline_tools._normalized_generated_candidate_row(source)],
        "row_count": 1, "unique_sequence_count": 1, "excluded_row_count": 0, "eligibility_filters": []}))
    report = report_explainer._build_design_output_summary_markdown({
        "structured_goal": {"gene_symbol": "example", "intent": "knockdown"},
        "gapmer_tiling": {"scored_tiles_path": str(scored), "row_count": 1}})
    assert "| 0.2500 |" in report
    assert "off-target gene count 0" not in report
    assert "requested eligibility filters" in report
    assert "| 2 |" in report
