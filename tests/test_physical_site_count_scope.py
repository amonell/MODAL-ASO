from copy import deepcopy
import json
from pathlib import Path

import pytest

from aso_agents.candidate_minimums import belongs_to_region, region_minimum_delivery, requested_region_minimums
from aso_chemistry import encode_idt_oligo


def fixture():
    goal = {"species": "human", "genome_assembly": "fixture-assembly",
            "candidate_limit": 6, "candidate_limit_per_site": 2, "min_asos_per_site": 2,
            "steric_design_jobs": [], "design_programs": []}
    rows = []
    for site in range(3):
        start = site * 100
        for length in (12, 16):
            name = f"program_{site}_{length}"
            region = ["chrFixture", start, start + 60, f"site_{site}", 0, "+"]
            core = ["chrFixture", start + 8, start + 12, f"core_{site}", 0, "+"]
            filters = [{"rows": [region], "operator": "union", "aso_overlap_requirement": "full_containment"},
                       {"rows": [core], "operator": "intersect", "aso_overlap_requirement": "contains_feature"}]
            job = {"program_id": name, "job_id": name, "label": name,
                   "transcript_id": "fixture-transcript", "targeting_policy": "explicit_user_region",
                   "chemistry_group_id": f"chemistry_{length}", "length": length,
                   "tile_filter_rows": [core], "region_rows": [region], "tile_overlap_requirement": "mixed",
                   "selection_meta": {"region_constraints": {"constraint_count": 2, "output_space": "genomic",
                                                              "tile_constraint_filters": filters}}}
            goal["steric_design_jobs"].append(job)
            goal["design_programs"].append({"program_id": name, "candidate_limit": 6})
            for offset in range(2):
                sequence = "ACGT" * (length // 4 - 1) + "ACGT"[site] + "ACGT"[offset] + "GC"
                rows.append({"Synthesis": sequence, "Sequence": sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1],
                             "IDT": encode_idt_oligo(sequence, "steric-hindrance"),
                             "program_id": name, "chemistry_group_id": job["chemistry_group_id"],
                             "Chrom": "chrFixture", "Start": start + offset, "End": start + offset + length,
                             "Strand": "+", "selection_rank": len(rows) + 1})
    return goal, rows


def intervals(row, **_kwargs):
    return [(row["Chrom"], row["Start"], row["End"])]


def test_length_and_chemistry_variants_share_physical_minimum_without_mutation():
    goal, rows = fixture()
    original = deepcopy(goal)
    groups = requested_region_minimums(goal)
    assert len(groups) == 3
    assert all(g["site_scope"] == "physical_constraint" and len(g["selectors"]) == 2 for g in groups)
    assert region_minimum_delivery(goal, {"rows": rows[::2]}, {}, intervals)["status"] == "fulfilled"
    assert region_minimum_delivery(goal, {"rows": rows[:6]}, {}, intervals)["status"] == "unfulfilled"
    assert goal == original


def test_native_export_respects_physical_minimum_and_global_cap(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    goal, rows = fixture()
    monkeypatch.setattr(pipeline, "_collect_generated_candidate_rows", lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, "_collect_generated_stage_sets", lambda *a: {})
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda *a: {"structured_goal": goal})
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    panel = json.loads(Path(artifacts["generated_aso_set_json"]).read_text())
    assert len(panel["rows"]) == 6
    assert panel["candidate_selection"]["status"] == "complete"
    delivery = region_minimum_delivery(goal, panel, {}, pipeline._goal_runtime_module()._tile_row_overlap_intervals)
    assert delivery["status"] == "fulfilled"
    assert [g["distinct_molecules"] for g in delivery["groups"]] == [2, 2, 2]


def test_merged_selectors_keep_chemistry_and_spatial_requirements():
    goal, rows = fixture()
    group = requested_region_minimums(goal)[0]
    assert belongs_to_region(rows[0], group, intervals)
    assert belongs_to_region(rows[2], group, intervals)
    assert not belongs_to_region({**rows[0], "chemistry_group_id": "foreign"}, group, intervals)
    assert not belongs_to_region({**rows[0], "Start": 10}, group, intervals)
    assert not belongs_to_region({**rows[0], "End": 61}, group, intervals)
    assert not belongs_to_region(rows[-1], group, intervals)


@pytest.mark.parametrize("difference", ["assembly", "transcript", "species", "strand", "reference", "filter_order", "overlap", "minimum", "blocks"])
def test_distinct_reference_or_predicate_does_not_merge(difference):
    goal, _ = fixture()
    job = goal["steric_design_jobs"][0]
    filters = job["selection_meta"]["region_constraints"]["tile_constraint_filters"]
    if difference in {"assembly", "transcript", "species"}:
        job[{"assembly": "genome_assembly", "transcript": "transcript_id", "species": "species"}[difference]] = "other"
    elif difference == "strand":
        job["tile_filter_rows"][0][5] = "-"
    elif difference == "reference":
        job["tile_filter_rows"][0][0] = "chrOther"
    elif difference == "filter_order":
        filters.reverse()
    elif difference == "overlap":
        filters[1]["aso_overlap_requirement"] = "any_overlap"
    elif difference == "minimum":
        filters[1]["minimum_overlap_bp"] = 2
    else:
        filters[1]["region_groups"] = [[filters[1]["rows"][0]]]
    assert len(requested_region_minimums(goal)) == 4


@pytest.mark.parametrize("missing", ["output_space", "constraint_count", "tile_constraint_filters"])
def test_unverified_compiled_scope_keeps_program_groups(missing):
    goal, _ = fixture()
    for job in goal["steric_design_jobs"]:
        job["selection_meta"]["region_constraints"].pop(missing)
    groups = requested_region_minimums(goal)
    assert len(groups) == 6 and all("site_scope" not in g for g in groups)


def test_display_names_do_not_create_sites_and_fallback_jobs_do_not_count():
    goal, _ = fixture()
    job = goal["steric_design_jobs"][0]
    job["program_id"] = "different_program"
    job["tile_filter_rows"][0][3] = "renamed_display"
    job["region_rows"][0][3] = "renamed_region"
    fallback = deepcopy(job)
    fallback["execution_policy"] = {"role": "fallback"}
    fallback["region_rows"][0][0] = "chrOther"
    goal["steric_design_jobs"].append(fallback)
    assert len(requested_region_minimums(goal)) == 3


def test_invalid_compiled_coordinates_are_not_identity_evidence():
    goal, _ = fixture()
    for job in goal["steric_design_jobs"]:
        job["tile_filter_rows"][0][1] = "not-a-coordinate"
    groups = requested_region_minimums(goal)
    assert len(groups) == 6 and all("site_scope" not in g for g in groups)
