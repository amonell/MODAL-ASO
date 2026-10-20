"""Exercise physical panel selection through the real annotation reader."""
import json
from pathlib import Path

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.candidate_minimums import belongs_to_region, region_minimum_delivery
from aso_chemistry import encode_idt_oligo


@pytest.mark.parametrize("record", [
    {"binding_intervals_bed0_half_open": [["chrFixture", 10, 30]]},
    {"Genomic_segments": [{"chrom": "chrFixture", "start": 11, "end": 30}]},
    {"Genomic_chrom": "chrFixture", "Genomic_start": 11, "Genomic_end": 30},
    {"Chrom": "chrFixture", "Start": 10, "End": 30},
])
def test_real_reader_preserves_physical_coordinates_in_region_membership(record):
    reader = pipeline._candidate_binding_intervals_for_exon_annotation
    assert reader(record, output_space="genomic") == reader(record) == [("chrFixture", 10, 30)]
    group = {"selectors": [{"rows": [["chrFixture", 10, 30]],
                            "requirement": "full_containment", "filters": []}]}
    assert belongs_to_region(record, group, reader)


@pytest.mark.parametrize("space", ["cdna", "unknown"])
def test_genomic_reader_does_not_relabel_a_requested_reference_frame(space):
    with pytest.raises(ValueError, match="genomic"):
        pipeline._candidate_binding_intervals_for_exon_annotation(
            {"binding_intervals_bed0_half_open": [["chrFixture", 10, 30]]}, output_space=space)


@pytest.mark.parametrize("policy", ["hard_nonoverlap", "joint_objective"])
def test_native_panel_export_combines_region_minimums_with_real_interval_callback(tmp_path, monkeypatch, policy):
    rows = []
    for index, (start, suffix) in enumerate([(0, "ACGA"), (10, "ACGG"), (110, "AGGA")]):
        sequence = "ACGT" * 4 + suffix
        rows.append({"Synthesis": sequence,
                     "Sequence": sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1],
                     "IDT": encode_idt_oligo(sequence, "steric-hindrance"),
                     "Chrom": "chrFixture", "Start": start, "End": start + len(sequence), "Strand": "+",
                     "binding_intervals_bed0_half_open": [["chrFixture", start, start + len(sequence)]],
                     "program_id": "first" if start < 100 else "second",
                     "source_modality": "steric", "selection_rank": index + 1})
    constraints = [{"constraint_id": label, "label": label, "source_type": "explicit_rows",
                    "operator": "union", "coordinate_space": "genomic",
                    "aso_overlap_requirement": "full_containment",
                    "materialized_rows": [["chrFixture", start, start + 60, label, 0, "+"]]}
                   for label, start in [("first", 0), ("second", 100)]]
    goal = {"species": "human", "candidate_minimum": 2, "candidate_limit": 2,
            "candidate_limit_per_site": 1, "min_asos_per_site": 1,
            "materialized_region_constraints": {"by_program": {
                constraint["constraint_id"]: [constraint] for constraint in constraints}},
            "design_programs": [{"program_id": label, "modality": "steric", "candidate_limit": 1}
                                for label in ("first", "second")],
            "steric_design_jobs": [{"program_id": constraint["constraint_id"],
                                    "region_rows": constraint["materialized_rows"]}
                                   for constraint in constraints]}
    if policy == "hard_nonoverlap":
        goal["candidate_selection_policy"] = "nonoverlapping_target_sites"
    else:
        goal["candidate_panel_objectives"] = [{"metric": "gc_fraction", "absolute_distance_from": .5,
                                              "aggregate": "sum", "direction": "minimize"}]
    monkeypatch.setattr(pipeline, "_collect_generated_candidate_rows", lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, "_collect_generated_stage_sets", lambda *a: {})
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda *a: {"structured_goal": goal})
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    panel = json.loads(Path(artifacts["generated_aso_set_json"]).read_text())
    assert panel["candidate_selection"]["status"] == "complete" and len(panel["rows"]) == 2
    delivery = region_minimum_delivery(goal, panel, {}, pipeline._candidate_binding_intervals_for_exon_annotation)
    assert delivery["status"] == "fulfilled"
    assert [group["distinct_molecules"] for group in delivery["groups"]] == [1, 1]
    if policy == "joint_objective":
        assert panel["candidate_selection"]["requested_panel_ranking"]["status"] == "complete"
    else:
        assert panel["candidate_selection"]["hard_nonoverlap"] is True
