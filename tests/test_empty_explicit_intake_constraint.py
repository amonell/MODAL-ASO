"""Intake placeholders must not become executable region filters."""

import run_goal_intake as runtime


def test_empty_explicit_intervals_preserve_derived_program_region():
    derived_rows = [["chr7", 100, 160, "resolved_exon", 0, "+"]]
    placeholder = {
        "source_type": "explicit_rows",
        "coordinate_space": "genomic",
        "intervals_bed0_half_open": [],
    }
    goal = {
        "region_constraints": [placeholder],
        "design_programs": [{
            "region_rows": derived_rows,
            "region_constraints": [placeholder],
        }],
    }

    runtime._normalize_region_constraints_in_goal(goal)

    assert "region_constraints" not in goal
    assert "region_constraints" not in goal["design_programs"][0]
    assert goal["design_programs"][0]["region_rows"] == derived_rows


def test_nonempty_explicit_intervals_remain_binding():
    interval = {"chrom": "chr7", "start": 120, "end": 140, "strand": "+"}
    goal = {
        "region_constraints": [{
            "source_type": "explicit_rows",
            "intervals_bed0_half_open": [interval],
        }],
    }

    runtime._normalize_region_constraints_in_goal(goal)

    assert goal["region_constraints"][0]["intervals_bed0_half_open"] == [interval]


def test_empty_explicit_placeholder_remains_available_for_typed_exon_recovery():
    goal = {
        "region_constraints": [{
            "source_type": "explicit_rows",
            "intervals_bed0_half_open": [],
        }],
        "design_programs": [{
            "target_type": "exon",
            "focus_exons_only": True,
            "exon_label": 2,
            "transcript_id": "ENSTTEST.4",
        }],
        "evaluation_targets": [{
            "target_type": "exon",
            "exon_label": 2,
            "transcript_id": "ENSTTEST.4",
        }],
    }

    runtime._normalize_region_constraints_in_goal(goal)

    assert goal["region_constraints"][0]["intervals_bed0_half_open"] == []
