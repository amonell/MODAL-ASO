"""Empty interval placeholders must not erase an independently typed exon target."""
from __future__ import annotations

import run_goal_intake as runtime


TRANSCRIPT = {
    "transcript_id": "ENSTTEST",
    "chrom": "chr7",
    "strand": 1,
    "exons_tx": [
        {"index": 1, "start": 100, "end": 149, "cdna_start": 1, "cdna_end": 50},
        {"index": 2, "start": 300, "end": 399, "cdna_start": 51, "cdna_end": 150},
    ],
}


def _goal() -> dict:
    return {
        "design_programs": [{
            "program_id": "design_a", "target_type": "exon", "focus_exons_only": True,
            "transcript_id": "ENSTTEST.4", "exon_label": 2,
        }],
        "evaluation_targets": [{
            "target_type": "exon", "transcript_id": "ENSTTEST.4", "exon_label": 2,
        }],
    }


def _materialize(goal: dict, constraint: dict, tmp_path) -> dict:
    return runtime._materialize_region_constraint_entry(
        structured_goal=goal,
        final_payload={},
        run_dir=str(tmp_path),
        constraint=constraint,
        isoform_transcripts={"ENSTTEST.4": TRANSCRIPT},
        iso_label_primary="ENSTTEST.4",
    )


def test_empty_explicit_rows_recover_only_matching_typed_exon(tmp_path) -> None:
    entry = _materialize(_goal(), {
        "source_type": "explicit_rows", "coordinate_space": "genomic",
        "intervals_bed0_half_open": [], "aso_overlap_requirement": "full_containment",
    }, tmp_path)
    assert entry["source_type"] == "exon"
    assert entry["transcript_id"] == "ENSTTEST.4"
    assert entry["materialized_rows"] == [["chr7", 299, 399, "exon_2", 0, "+"]]
    assert entry["materialized_cdna_rows"] == [["ENSTTEST", 50, 150, "exon_2", 0, "+"]]
    assert entry["aso_overlap_requirement"] == "full_containment"
    assert entry["materialization_summary"]["empty_explicit_rows_recovered_from_typed_exon"] is True


def test_empty_explicit_rows_do_not_guess_when_typed_targets_disagree(tmp_path) -> None:
    goal = _goal()
    goal["evaluation_targets"][0]["exon_label"] = 1
    entry = _materialize(goal, {
        "source_type": "explicit_rows", "coordinate_space": "cdna", "rows": [],
        "aso_overlap_requirement": "full_containment",
    }, tmp_path)
    assert entry["source_type"] == "explicit_rows"
    assert entry["row_count"] == 0


def test_explicit_rows_remain_authoritative_when_present(tmp_path) -> None:
    entry = _materialize(_goal(), {
        "source_type": "explicit_rows", "coordinate_space": "cdna",
        "rows": [["ENSTTEST", 60, 80, "selected", 0, "+"]],
        "aso_overlap_requirement": "full_containment",
    }, tmp_path)
    assert entry["source_type"] == "explicit_rows"
    assert entry["row_count"] == 1
    assert entry["materialized_rows"][0][1:3] == [60, 80]


def test_empty_explicit_rows_do_not_override_conflicting_hint(tmp_path) -> None:
    entry = _materialize(_goal(), {
        "source_type": "explicit_rows", "coordinate_space": "cdna", "rows": [],
        "transcript_id": "ENSTOTHER.1", "exon_label": 2,
        "aso_overlap_requirement": "full_containment",
    }, tmp_path)
    assert entry["source_type"] == "explicit_rows"
    assert entry["row_count"] == 0


def test_scoped_program_materialization_retains_its_exon_evidence(tmp_path, monkeypatch) -> None:
    goal = _goal()
    goal["design_programs"][0]["region_constraints"] = [{
        "source_type": "explicit_rows", "coordinate_space": "genomic",
        "intervals_bed0_half_open": [], "aso_overlap_requirement": "any_overlap",
    }]
    monkeypatch.setattr(runtime, "_ensure_isoform_transcripts_for_target_region", lambda *_: None)
    monkeypatch.setattr(runtime, "_select_constraint_primary_isoform", lambda *_: ("ENSTTEST.4", {"ENSTTEST.4": TRANSCRIPT}))
    summary = runtime._materialize_region_constraints(goal, {}, str(tmp_path))
    assert summary["status"] == "materialized"
    entry = goal["materialized_region_constraints"]["by_program"]["design_a"][0]
    assert entry["source_type"] == "exon"
    assert [row[:3] for row in entry["materialized_rows"]] == [["chr7", 299, 399]]
    assert entry["materialization_summary"]["empty_explicit_rows_recovered_from_typed_exon"] is True


def test_repeated_candidate_slots_with_empty_exon_placeholders_recover(tmp_path, monkeypatch) -> None:
    goal = _goal()
    prototype = goal["design_programs"][0]
    prototype.update(objective="Design a three-molecule panel", candidate_limit=3,
                     notes="Target the same exon.", region_constraints=[{
                         "source_type": "explicit_rows", "intervals_bed0_half_open": [],
                         "aso_overlap_requirement": "any_overlap",
                     }])
    goal["candidate_limit"] = 3
    goal["design_programs"] = [dict(prototype, program_id=f"slot_{index}") for index in range(3)]
    goal["design_programs"][0]["notes"] += " Use the specified chemistry."
    goal["region_constraints"] = [{"source_type": "explicit_rows", "intervals_bed0_half_open": []}]
    runtime._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 1
    monkeypatch.setattr(runtime, "_ensure_isoform_transcripts_for_target_region", lambda *_: None)
    monkeypatch.setattr(runtime, "_select_constraint_primary_isoform", lambda *_: ("ENSTTEST.4", {"ENSTTEST.4": TRANSCRIPT}))
    runtime._materialize_region_constraints(goal, {}, str(tmp_path))
    materialized = goal["materialized_region_constraints"]
    assert materialized["global"][0]["source_type"] == "exon"
    assert materialized["by_program"]["slot_0"][0]["source_type"] == "exon"
    assert materialized["by_program"]["slot_0"][0]["row_count"] == 1
