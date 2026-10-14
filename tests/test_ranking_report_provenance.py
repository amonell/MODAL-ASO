import json

import pytest

import run_goal_intake as native
from aso_agents.html_report import _build_gapmer_explanation
from aso_agents.ranking_contracts import recorded_ranking_explanation


def test_materialized_rank_and_report_share_the_executed_criteria(tmp_path):
    common = {
        "isoform_uniqueness_class": "target_without_beneficial",
        "isoform_uniqueness_score": 0.25,
        "beneficial_isoform_match_count": 0,
        "other_isoform_match_count": 3,
        "isoform_uniqueness_count": 4,
        "worst_species_off_target_gene_count": 0,
        "Phastcon": 0.4,
        "PhyloP": 0.0,
    }
    path = tmp_path / "tiles.json"
    path.write_text(json.dumps({"rows": [
        {**common, "Name": "higher_gc", "GC": 0.7, "Phastcon": 0.9},
        {**common, "Name": "center_gc", "GC": 0.5},
        {**common, "Name": "other_gene", "GC": 0.5, "worst_species_off_target_gene_count": 1},
    ]}))
    ranking = native._annotate_gapmer_selection_ranks(str(path))
    rows = json.loads(path.read_text())["rows"]
    assert [r["Name"] for r in rows] == ["center_gc", "higher_gc", "other_gene"]
    goal = {"gapmer_tiling_results": {"selection_ranking": ranking, "row_count": 3}}
    report = _build_gapmer_explanation(tmp_path, {}, goal, {})
    section = next(s for s in report["sections"] if s["title"] == "How the final candidates were ranked")
    assert section["bullets"][:-1] == [item["explanation"] for item in ranking["ordering"]]
    assert "closer to 0.50" in " ".join(section["bullets"])
    assert "primarily by uniqueness" not in json.dumps(report)
    assert "then higher GC" not in json.dumps(report)


@pytest.mark.parametrize("ranking", [None, {}, {"status": "failed"}, {"status": "complete", "method": "legacy"}])
def test_report_does_not_invent_missing_ranking_policy(tmp_path, ranking):
    assert recorded_ranking_explanation(ranking) is None
    report = _build_gapmer_explanation(
        tmp_path, {}, {"gapmer_tiling_results": {"row_count": 2, "selection_ranking": ranking}}, {}
    )
    section = next(s for s in report["sections"] if s["title"] == "How the final candidates were ranked")
    assert "not recorded" in section["summary"]
    assert section["bullets"] == []


def test_equal_sort_keys_retain_input_order(tmp_path):
    path = tmp_path / "ties.json"
    path.write_text(json.dumps({"rows": [{"Name": "second"}, {"Name": "first"}]}))
    result = native._annotate_gapmer_selection_ranks(str(path))
    assert [r["Name"] for r in json.loads(path.read_text())["rows"]] == ["second", "first"]
    assert "input order" in result["tie_policy"]
