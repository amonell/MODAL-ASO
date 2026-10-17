"""Empty interval placeholders cannot erase an independently resolved region."""

import run_goal_intake as runtime


BASE = [["chr7", 100, 160, "coding_exon", 0, "+"]]


def _goal(constraint, target_region="cds"):
    return {
        "target_region": target_region,
        "materialized_region_constraints": {"global": [constraint], "by_program": {}},
    }


def test_empty_explicit_placeholder_keeps_resolved_cds_rows():
    goal = _goal({
        "source_type": "explicit_rows", "coordinate_space": "genomic",
        "intervals_bed0_half_open": [], "rows": [], "materialized_rows": [],
        "aso_overlap_requirement": "full_containment",
    })
    selected, meta = runtime._apply_region_constraints_to_rows(
        goal, BASE, output_space="genomic")
    assert selected == BASE
    assert meta["ignored_empty_explicit_constraints"] == 1
    assert meta["constraint_count"] == 0


def test_actual_explicit_rows_remain_authoritative():
    goal = _goal({
        "source_type": "explicit_rows", "coordinate_space": "genomic",
        "aso_overlap_requirement": "full_containment",
        "materialized_rows": [["chr7", 120, 140, "user_interval", 0, "+"]],
    })
    selected, meta = runtime._apply_region_constraints_to_rows(
        goal, BASE, output_space="genomic")
    assert [row[1:3] for row in selected] == [[120, 140]]
    assert meta["ignored_empty_explicit_constraints"] == 0


def test_empty_constraint_without_resolved_typed_region_stays_blocking():
    goal = _goal({"source_type": "explicit_rows", "rows": []}, target_region=None)
    selected, meta = runtime._apply_region_constraints_to_rows(
        goal, BASE, output_space="genomic")
    assert selected == []
    assert meta["constraint_count"] == 1


def test_empty_placeholder_does_not_erase_other_constraints():
    goal = {
        "target_region": "cds",
        "materialized_region_constraints": {"global": [
            {"source_type": "explicit_rows", "rows": []},
            {"source_type": "explicit_rows", "coordinate_space": "genomic",
             "aso_overlap_requirement": "full_containment",
             "materialized_rows": [["chr7", 120, 140, "user_interval", 0, "+"]]},
        ], "by_program": {}},
    }
    selected, meta = runtime._apply_region_constraints_to_rows(
        goal, BASE, output_space="genomic")
    assert [row[1:3] for row in selected] == [[120, 140]]
    assert meta["ignored_empty_explicit_constraints"] == 1
    assert meta["constraint_count"] == 1
