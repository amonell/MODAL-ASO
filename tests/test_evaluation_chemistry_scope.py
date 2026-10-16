import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.provided_aso_ranking import evaluation_ranking_policy, suppress_incomparable_ranks


@pytest.mark.parametrize("chemistries,allowed", [
    (["steric-hindrance", "5-10-5"], False),
    (["5-10-5", "2ome:5-10-5"], False),
    (["custom chemistry A", "custom chemistry A"], True),
    (["5-10-5", None], False),
    ([None, None], False),
    (["unknown", "unknown"], False),
    ([{"platform": "gapmer", "length": 20}, {"length": 20, "platform": "gapmer"}], True),
    (["5-10-5"], True),
])
def test_only_one_recorded_chemistry_supports_shared_proxy_order(chemistries, allowed):
    rows = [{"chemistry": value} for value in chemistries]
    before = deepcopy(rows)
    assert evaluation_ranking_policy(rows)["allow_shared_ranking"] is allowed
    assert rows == before


def test_stale_numeric_ranks_are_removed_but_measurements_remain():
    rows = [{"chemistry": "steric-hindrance", "selection_rank": 1,
             "knockdown_proxy_rank": 2, "GC": 0.5, "Off_target_exact_count": 0,
             "selection_score_components": {"generic_gapmer_proxy_rank": 3, "gc_fraction": 0.5},
             "stage_results": {"ranking": {"status": "pass"}, "mapping": {"status": "pass"}}},
            {"chemistry": "5-10-5"}]
    suppress_incomparable_ranks(rows, evaluation_ranking_policy(rows))
    assert "selection_rank" not in rows[0] and "knockdown_proxy_rank" not in rows[0]
    assert rows[0]["selection_score_components"] == {"gc_fraction": 0.5}
    assert rows[0]["Off_target_exact_count"] == 0
    assert rows[0]["stage_results"]["mapping"] == {"status": "pass"}
    assert rows[0]["stage_results"]["ranking"]["status"] == "not_run"


@pytest.mark.parametrize("splice_related", [False, True])
def test_mixed_panel_preserves_all_molecules_and_exports_without_shared_rank(tmp_path, splice_related):
    class Runtime:
        @staticmethod
        def _annotate_tile_rows_with_rnafold(rows):
            for row in rows:
                row["RNAfold_MFE"] = -2.5
            return len(rows)

        @staticmethod
        def _annotate_gapmer_selection_ranks(path):
            raise AssertionError("Incomparable chemistries must never reach a shared ranker.")

    rows = [{"name": f"molecule-{i}", "sequence": seq, "sequence_orientation": "antisense",
             "chemistry": chem, "mapping_status": "unmapped", "selection_rank": i + 1}
            for i, (seq, chem) in enumerate([
                ("AACCGGTA", "steric-hindrance"), ("AACCGGTA", "2-4-2"),
                ("ATCCGGTA", "steric-hindrance"), ("ATCCGGTA", "2-4-2")])]
    result = pipeline._score_provided_asos_with_existing_pipeline(
        rows=rows, eval_goal={}, runtime=Runtime(), run_dir=str(tmp_path),
        task_id="evaluation", cdna_map={}, iso_label=None, splice_related=splice_related)
    assert result["selection_ranking"] == {"status": "not_run", "reason": "mixed_or_unresolved_recorded_chemistries"}
    assert len(rows) == 4 and len({row["name"] for row in rows}) == 4
    for row in rows:
        assert row["ranking_status"] == "not_comparable"
        assert "selection_rank" not in row and "knockdown_proxy_rank" not in row
        assert row["RNAfold_MFE"] == -2.5
    for artifact in Path(result["scored_tiles_path"]).parent.glob("provided_aso*.json"):
        content = json.loads(artifact.read_text())
        for row in content["rows"]:
            assert row["ranking_status"] == "not_comparable"
            assert "selection_rank" not in row and "knockdown_proxy_rank" not in row
    summary = {"provided_asos": rows, "provided_aso_scoring": result}
    compact = pipeline._compact_existing_aso_evaluation_for_response(summary)
    assert all(row["ranking_status"] == "not_comparable" for row in compact["provided_asos"])
    table = tmp_path / "panel.csv"
    pipeline._write_v2_response_outputs(run_dir=str(tmp_path / "report"), markdown_text="Evidence by molecule.",
        response_payload={"evaluation_summary": summary}, response_basename="evaluation", title="Evaluation",
        requested_outputs=[{"kind": "evaluation", "path": str(table)}])
    exported = list(csv.DictReader(table.open()))
    assert [row["name"] for row in exported] == [row["name"] for row in rows]
    assert all(row["ranking_status"] == "not_comparable" for row in exported)
    assert all(not row.get("selection_rank") and not row.get("knockdown_proxy_rank") for row in exported)


def test_same_chemistry_retains_existing_proxy_ranking(tmp_path):
    class Runtime:
        @staticmethod
        def _annotate_gapmer_selection_ranks(path):
            payload = json.loads(Path(path).read_text())
            for rank, row in enumerate(payload["rows"], 1):
                row["selection_rank"] = rank
            Path(path).write_text(json.dumps(payload))
            return {"status": "complete"}

    rows = [{"name": "a", "sequence": "AACCGGTA", "chemistry": "2-4-2"},
            {"name": "b", "sequence": "ATCCGGTA", "chemistry": "2-4-2"}]
    result = pipeline._score_provided_asos_with_existing_pipeline(rows=rows, eval_goal={}, runtime=Runtime(),
        run_dir=str(tmp_path), task_id="evaluation", cdna_map={}, iso_label=None, splice_related=False)
    assert result["ranking_policy"]["allow_shared_ranking"]
    assert [row["selection_rank"] for row in rows] == [1, 2]
