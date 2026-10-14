from copy import deepcopy
import json

from aso_agents.candidate_minimums import (
    requested_region_minimums, select_default_gc_supplements, region_minimum_delivery,
    belongs_to_region,
)
from aso_agents.design_delivery import design_delivery
from aso_agents.binding_region_reports import build_binding_region_report, binding_region_markdown, memberships
from aso_agents.pipeline_tools import _bind_candidate_physical_site_groups, _apply_generated_candidate_limit


def row(sequence, start=0):
    return {"Synthesis": sequence,
        "Sequence": sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1],
        "GC": sum(sequence.count(base) for base in "GC") / len(sequence),
        "IDT": "declared-chemistry:" + sequence,
        "Chrom": "chrFixture", "Start": start, "End": start + len(sequence),
        "Strand": "+", "No_homopolymer": "True", "Score": 5}


def intervals(record, **_kwargs):
    return record.get("binding_intervals_bed0_half_open") or [
        (record["Chrom"], record["Start"], record["End"])]


def goal(minimum=2):
    region = ["chrFixture", 0, 60, "region", 0, "+"]
    return {"min_asos_per_site": minimum, "steric_design_jobs": [
        {"program_id": "region_a", "job_id": "job_a", "label": "Region A",
         "region_rows": [region], "tile_overlap_requirement": "full_containment"}],
        "materialized_region_constraints": {"by_program": {"region_a": [
            {"constraint_id": "region_a", "label": "Region A", "operator": "union",
             "aso_overlap_requirement": "full_containment", "materialized_rows": [region]}]}}}


def test_single_job_multiple_named_physical_sites_require_each_count():
    request = goal()
    request["candidate_limit_per_site"] = 2
    sites = [("west", 0), ("center", 100), ("east", 200)]
    request["steric_design_jobs"][0]["selection_meta"] = {"region_constraints": {
        "tile_constraint_filters": [
            {"rows": [["chrFixture", start, start + 60, name, 0, "+"]],
             "operator": "union", "aso_overlap_requirement": "full_containment"}
            for name, start in sites]}}
    # Input constraints can be in transcript coordinates; compiled filters are
    # the genomic source of truth for physical delivery.
    request["materialized_region_constraints"]["by_program"]["region_a"] = [
        {"constraint_id": f"{name}_containment", "label": name,
         "operator": "union", "source_type": "explicit_rows",
         "coordinate_space": "cdna",
         "materialized_rows": [["ENST00000000001", start, start + 60, name, 0, "+"]]}
        for name, start in sites]
    groups = requested_region_minimums(request)
    assert len(groups) == 3
    assert [entry["label"] for entry in memberships(row("ACGT" * 5, 100), request, intervals)] == ["center"]
    candidates = [row(sequence * 5, start + offset)
                  for _, start in sites
                  for offset, sequence in enumerate(("ACGT", "CGTA", "GTAC"))]
    scoped = _bind_candidate_physical_site_groups(candidates, groups, intervals)
    chosen, selection = _apply_generated_candidate_limit(
        scoped, candidate_limit=2, per_site=True, group_scope="physical_region",
        expected_group_labels=[group["group_id"] for group in groups])
    assert len(chosen) == 6
    assert [group["selected_count"] for group in selection["groups"]] == [2, 2, 2]
    assert region_minimum_delivery(request, {"rows": chosen}, {}, intervals)["status"] == "fulfilled"
    assert region_minimum_delivery(request, {"rows": chosen[:2]}, {}, intervals)["status"] == "unfulfilled"
    appendix = {"sets": [{"rows": candidates}]}
    assert region_minimum_delivery(request, {"rows": chosen[:2]}, appendix, intervals)["status"] == "unfulfilled"
    assert region_minimum_delivery(request, {}, appendix, intervals)["status"] == "fulfilled"


def test_unresolved_named_site_fails_closed():
    request = goal()
    request["materialized_region_constraints"]["by_program"]["region_a"] = [
        {"constraint_id": name, "label": name, "source_type": "explicit_rows", "operator": "union"}
        for name in ("alpha", "beta")]
    request["steric_design_jobs"][0]["selection_meta"] = {"region_constraints": {
        "tile_constraint_filters": [{"rows": [["chrFixture", 0, 60, "alpha"]], "operator": "union"}]}}
    result = region_minimum_delivery(request, {"rows": [row("ACGT" * 5)]}, {}, intervals)
    assert result["status"] == "unfulfilled"
    assert [group["shortfall"] for group in result["groups"]] == [1, 2]
    assert memberships(row("ACGT" * 5), request, intervals) == []


def recover(request, passed, rejected, constraints=None):
    return select_default_gc_supplements(passed, rejected, constraints or {},
        requested_region_minimums(request), intervals, (0.3, 0.65))


PREFERRED = "ACGT" * 5
NEAR = "GCTAGCTATGATATATATAT"
DISTANT = "AT" * 10


def test_minimum_uses_nearest_default_exception_without_mutating_sources():
    inputs = [row(DISTANT, 1), row(NEAR, 2)]
    original = deepcopy(inputs)
    selected, receipt = recover(goal(), [row(PREFERRED)], inputs)
    assert [index for index, _record in selected] == [1]
    assert selected[0][1]["default_gc_preference_exception"]["explicit_constraints_satisfied"]
    assert receipt["groups"][0]["after"] == 2
    assert inputs == original


def test_no_minimum_and_satisfied_minimum_do_not_relax_defaults():
    assert recover({}, [row(PREFERRED)], [row(NEAR)]) == ([], {})
    selected, receipt = recover(goal(1), [row(PREFERRED)], [row(NEAR)])
    assert not selected and receipt["retained_binding_records"] == 0


def test_explicit_gc_bound_cannot_be_relaxed_for_count():
    selected, receipt = recover(goal(), [row(PREFERRED)], [row(NEAR)],
        {"gc_content": {"min": 0.3, "max": 0.65}})
    assert selected == [] and receipt["groups"][0]["shortfall"] == 1


def test_one_sided_bound_preserved_while_unspecified_default_can_yield():
    selected, _ = recover(goal(), [row(PREFERRED)], [row(NEAR)],
        {"gc_content": {"max": 0.65}})
    assert len(selected) == 1
    selected, _ = recover(goal(), [row(PREFERRED)], [row(NEAR)],
        {"gc_content": {"min": row(NEAR)["GC"], "min_inclusive": False}})
    assert selected == []


def test_duplicate_placements_do_not_fill_a_molecule_minimum():
    selected, receipt = recover(goal(3), [row(PREFERRED)], [row(NEAR, 1), row(NEAR, 2)])
    assert len(selected) == 1 and receipt["groups"][0]["shortfall"] == 1


def test_overlapping_regions_share_a_valid_molecule_without_duplicate_padding():
    request = goal()
    second = deepcopy(request["steric_design_jobs"][0])
    second.update(program_id="region_b", job_id="job_b", label="Region B")
    request["steric_design_jobs"].append(second)
    selected, receipt = recover(request, [row(PREFERRED)], [row(NEAR)])
    assert len(selected) == 1 and [group["after"] for group in receipt["groups"]] == [2, 2]


def test_other_constraints_and_unknown_measurements_still_reject():
    candidate = row(NEAR)
    selected, _ = recover(goal(), [row(PREFERRED)], [candidate], {"max_binding_sites": 0})
    assert selected == []
    candidate["GC"] = None
    selected, _ = recover(goal(), [row(PREFERRED)], [candidate])
    assert selected == []
    candidate = row(NEAR, 100)
    selected, _ = recover(goal(), [row(PREFERRED)], [candidate])
    assert selected == []


def test_compiled_exclusions_are_not_bypassed():
    request = goal()
    request["steric_design_jobs"][0]["selection_meta"] = {"region_constraints": {
        "tile_constraint_filters": [{"rows": [["chrFixture", 0, 40]], "operator": "exclude"}]}}
    selected, receipt = recover(request, [], [row(NEAR)])
    assert selected == [] and receipt["groups"][0]["after"] == 0


def test_mixed_union_uses_typed_rules_after_broad_region_screen():
    request = goal(1)
    job = request["steric_design_jobs"][0]
    job["region_rows"] = [["chrFixture", 0, 60], ["chrFixture", 82, 92]]
    job["tile_overlap_requirement"] = "mixed"
    job["selection_meta"] = {"region_constraints": {"tile_constraint_filters": [
        {"rows": [["chrFixture", 0, 60]], "operator": "union",
         "aso_overlap_requirement": "full_containment"},
        {"rows": [["chrFixture", 82, 92]], "operator": "union",
         "aso_overlap_requirement": "contains_feature"},
    ]}}
    group = requested_region_minimums(request)[0]
    assert belongs_to_region(row(PREFERRED, 5), group, intervals)
    assert belongs_to_region(row(PREFERRED, 79), group, intervals)
    assert not belongs_to_region(row(PREFERRED, 55), group, intervals)
    source = row(PREFERRED, 85)
    assert not belongs_to_region(source, group, intervals)


def test_final_delivery_uses_complete_binding_pool_not_response_subset_or_preview_count():
    request = goal()
    candidate_set = {"rows": []}
    binding = {"sets": [{"rows": [row(PREFERRED), row(NEAR)], "candidate_count": 999}]}
    delivery = region_minimum_delivery(request, candidate_set, binding, intervals)
    assert delivery["status"] == "fulfilled"
    binding["sets"][0]["rows"] = [row(PREFERRED), row(PREFERRED, 1)]
    delivery = region_minimum_delivery(request, candidate_set, binding, intervals)
    assert delivery["status"] == "unfulfilled" and delivery["groups"][0]["distinct_molecules"] == 1


def test_completion_detects_missing_region_despite_overall_count():
    request = goal()
    candidate = row(PREFERRED)
    candidate["synthesized_antisense_sequence"] = candidate["Synthesis"]
    candidate["target_binding_sequence"] = candidate["Sequence"]
    result = design_delivery(request, {"rows": [candidate]},
        {"steric": {"requested": True, "status": "outputs_present"}}, intervals_for_row=intervals)
    assert result["status"] == "unfulfilled"
    assert "requested_per_site_candidate_count_not_delivered" in result["errors"]


def test_minimum_with_unresolved_scope_cannot_report_fulfilled():
    result = region_minimum_delivery({"min_asos_per_site": 2}, {"rows": [row(PREFERRED)]}, {}, intervals)
    assert result["errors"] == ["requested_per_site_scope_unresolved"]


def test_report_exports_full_pool_and_count_and_exception_evidence(tmp_path):
    request = goal()
    selected, _ = recover(request, [row(PREFERRED)], [row(NEAR)])
    path = tmp_path / "scored.json"
    path.write_text(json.dumps({"rows": [row(PREFERRED), selected[0][1]]}))
    request["steric_tiling_results_by_job"] = {"job_a": {"scored_tiles_path": str(path)}}
    report = build_binding_region_report(request, {}, str(tmp_path), intervals, {})
    assert report["minimum_delivery"]["status"] == "fulfilled"
    assert report["default_gc_preference_exception_records"] == 1
    text = binding_region_markdown(report)
    assert "minimum 2; shortfall 0" in text and "default GC preference" in text
    import csv
    table = list(csv.DictReader((tmp_path / report["complete_table_relative_path"]).open()))
    assert len(table) == 2
    assert sum(json.loads(record["default_gc_preference_exception"]) is not None for record in table) == 1


def test_minimum_is_not_a_maximum_and_stable_program_ids_share_one_group():
    from aso_agents import pipeline_tools as pipeline
    from aso_agents.candidate_presentation import bind_candidate_program_groups
    request = {"candidate_minimum": 5, "min_asos_per_site": 1,
        "design_programs": [{"program_id": "steric_program_alpha", "label": "Readable independent label"}]}
    assert pipeline._structured_candidate_limit(request) is None
    rows = [{"program_id": "steric_program_alpha", "job_label": "Source-view label", "selection_rank": rank}
            for rank in range(1, 8)]
    rows = bind_candidate_program_groups(rows, request)
    chosen, summary = pipeline._apply_generated_candidate_limit(rows,
        candidate_limit=pipeline._structured_candidate_limit(request), per_site=True,
        expected_group_labels=pipeline._structured_candidate_group_labels(request))
    assert len(chosen) == 7
    assert len(summary["groups"]) == 1 and summary["groups"][0]["eligible_count"] == 7


def test_actual_native_filter_recovery_and_non_gc_failure_buckets(tmp_path):
    import run_goal_intake as runtime
    source = tmp_path / "tiles.json"
    failed_homopolymer = row(DISTANT, 3)
    failed_homopolymer["No_homopolymer"] = "False"
    source.write_text(json.dumps({"schema": {}, "rows": [row(PREFERRED), row(NEAR, 1), failed_homopolymer]}))
    summary = runtime._apply_initial_tiling_filters(str(source), {},
        minimum_regions=requested_region_minimums(goal(3)))
    passed = json.loads(open(summary["passed"]["path"]).read())["rows"]
    assert len(passed) == 2
    assert summary["filtered_no_homopolymer"]["count"] == 1
    assert summary["filtered_gc_out_of_range"]["count"] == 1
    assert summary["default_gc_minimum_recovery"]["groups"][0]["shortfall"] == 1
    assert sum(bool(record.get("default_gc_preference_exception")) for record in passed) == 1
    assert json.loads(source.read_text())["rows"][1].get("default_gc_preference_exception") is None


def test_actual_native_filter_keeps_explicit_gc_and_binding_limits(tmp_path):
    import run_goal_intake as runtime
    source = tmp_path / "tiles.json"
    preferred, near = row(PREFERRED), row(NEAR, 1)
    preferred["off_target_gene_count"] = 0
    near["off_target_gene_count"] = 3
    source.write_text(json.dumps({"schema": {}, "rows": [preferred, near]}))
    for constraints in [{"gc_content": {"min": 0.3, "max": 0.7}}, {"max_binding_sites": 0}]:
        summary = runtime._apply_initial_tiling_filters(str(source), constraints,
            minimum_regions=requested_region_minimums(goal()))
        assert summary["passed"]["count"] == 1
        assert summary["default_gc_minimum_recovery"]["retained_binding_records"] == 0
