from aso_agents.candidate_region_gate import filter_final_rows_by_global_regions


def _intervals(row, *, output_space):
    assert output_space == "genomic"
    return row["intervals"]


def test_final_panel_enforces_user_minimum_overlap_across_ungrouped_rows():
    rows = [
        {"name": "nine", "intervals": [["chr12", 207, 227]]},
        {"name": "ten", "intervals": [["chr12", 206, 226]]},
        {"name": "inside", "intervals": [["chr12", 130, 150]]},
    ]
    goal = {"materialized_region_constraints": {"global": [{
        "constraint_id": "user_interval", "operator": "union",
        "aso_overlap_requirement": "any_overlap", "minimum_overlap_bp": 10,
        "materialized_rows": [["chr12", 100, 216]],
    }]}}
    kept, receipt = filter_final_rows_by_global_regions(rows, goal, _intervals)
    assert [row["name"] for row in kept] == ["ten", "inside"]
    assert receipt["status"] == "complete" and receipt["excluded_count"] == 1


def test_unmaterialized_user_region_does_not_silently_exclude_candidates():
    rows = [{"name": "unresolved", "intervals": [["chr12", 200, 220]]}]
    goal = {"materialized_region_constraints": {"global": [{
        "constraint_id": "user_interval", "materialized_rows": [],
    }]}}
    kept, receipt = filter_final_rows_by_global_regions(rows, goal, _intervals)
    assert kept == rows and receipt["status"] == "materialization_incomplete"
