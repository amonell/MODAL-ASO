from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import pytest

from aso_agents.candidate_minimums import region_minimum_delivery, requested_region_minimums
from aso_agents.candidate_presentation import molecule_key
from aso_agents.design_delivery import design_delivery
from aso_agents.region_panel_selection import select_region_minimum_panel


def regions(minimum=1):
    return [{"group_id": name, "label": name, "minimum": minimum, "selectors": [
        {"rows": [["chrFixture", i * 100, i * 100 + 40]],
         "requirement": "any_overlap", "filters": []}]} for i, name in enumerate("ABC")]


def intervals(row, **_):
    return row["binding_intervals_bed0_half_open"]


def candidate(index, membership, chemistry="chemistry-one"):
    sequence = "ACGTAC" + "ACGT"[index // 4] + "ACGT"[index % 4]
    return {"Synthesis": sequence, "Sequence": sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1],
            "IDT": chemistry, "selection_rank": index + 1,
            "binding_intervals_bed0_half_open": [["chrFixture", i * 100 + 3, i * 100 + 11]
                                                  for i in membership]}


def test_global_limit_reserves_each_region_without_modifying_native_ranks():
    rows = [candidate(i, {0} if i < 4 else {1}) for i in range(7)]
    original = deepcopy(rows)
    selected, summary = select_region_minimum_panel(rows, regions(2)[:2], intervals, global_limit=4)
    assert [row["selection_rank"] for row in selected] == [1, 2, 5, 6]
    assert summary["status"] == "complete" and summary["selected_count"] == 4
    assert [group["selected_count"] for group in summary["groups"]] == [2, 2]
    assert rows == original


def test_overlapping_memberships_do_not_spend_multiple_panel_slots():
    rows = [candidate(0, {0}), candidate(1, {1}), candidate(2, {0, 1}), candidate(3, {2})]
    selected, summary = select_region_minimum_panel(rows, regions(), intervals, global_limit=2)
    assert [row["selection_rank"] for row in selected] == [3, 4]
    assert summary["status"] == "complete"


def test_duplicate_records_do_not_fill_minimum_but_different_chemistries_do():
    a = candidate(0, {0}); b = deepcopy(a); b["IDT"] = "chemistry-two"
    _, summary = select_region_minimum_panel([a, deepcopy(a)], regions(2)[:1], intervals, global_limit=2)
    assert summary["status"] == "unfulfilled" and summary["selected_count"] == 1
    rows, summary = select_region_minimum_panel([a, deepcopy(a), b], regions(2)[:1], intervals, global_limit=2)
    assert summary["status"] == "complete" and len(rows) == 2


def test_multiple_verified_placements_retain_evidence_without_padding_count():
    a, b = candidate(0, {0}), candidate(0, {1})
    rows, summary = select_region_minimum_panel([a, deepcopy(a), b], regions()[:2], intervals, global_limit=1)
    assert len(rows) == 2 and summary["selected_count"] == 1
    assert [entry["selected_count"] for entry in summary["groups"]] == [1, 1]
    assert {molecule_key(row) for row in rows} == {molecule_key(a)}


@pytest.mark.parametrize("cap", [1, 2, 3, 4])
@pytest.mark.parametrize("minimum", [1, 2])
def test_feasibility_and_rank_cost_match_exhaustive_small_problem(cap, minimum):
    masks = [{0}, {0, 1}, {1}, {2}, {1, 2}, {0, 2}, {0, 1, 2}]
    rows = [candidate(i, mask) for i, mask in enumerate(masks)]
    feasible = [subset for n in range(cap + 1) for subset in combinations(range(len(rows)), n)
                if all(sum(g in masks[i] for i in subset) >= minimum for g in range(3))]
    selected, summary = select_region_minimum_panel(rows, regions(minimum), intervals, global_limit=cap)
    assert (summary["status"] == "complete") == bool(feasible)
    if feasible:
        largest = max(map(len, feasible))
        assert len(selected) == largest
        assert sum(row["selection_rank"] for row in selected) == min(
            sum(i + 1 for i in subset) for subset in feasible if len(subset) == largest)
    assert len(selected) <= cap


def test_per_site_and_program_upper_bounds_still_apply():
    rows = [candidate(i, {0} if i < 3 else {1}) for i in range(6)]
    for row in rows:
        row["program_id"] = "shared"
    selected, summary = select_region_minimum_panel(rows, regions()[:2], intervals,
        global_limit=4, per_site_limit=2, program_limits={"shared": 3})
    assert summary["status"] == "complete" and len(selected) == 3
    assert all(1 <= group["selected_count"] <= 2 for group in summary["groups"])
    _, summary = select_region_minimum_panel(rows, regions(2)[:2], intervals,
        global_limit=3, per_site_limit=2)
    assert summary["status"] == "unfulfilled" and summary["selected_count"] <= 3


def test_selected_panel_shortfall_cannot_be_filled_by_appendix():
    goal = {"min_asos_per_site": 2, "candidate_limit": 2, "candidate_minimum": 2,
            "steric_design_jobs": [{"program_id": "one", "region_rows": [["chrFixture", 0, 40]],
                                    "tile_overlap_requirement": "full_containment"}]}
    rows = [candidate(0, {0}), candidate(1, {0})]
    appendix = {"sets": [{"rows": rows}]}
    result = region_minimum_delivery(goal, {"rows": rows[:1]}, appendix, intervals)
    assert result["status"] == "unfulfilled" and result["count_scope"] == "selected_panel"
    result = design_delivery(goal, {"rows": rows[:1]}, {"steric": {"requested": True}},
                             binding_sets=appendix, intervals_for_row=intervals)
    assert "requested_candidate_count_not_delivered" in result["errors"]
    assert result["selected_panel_distinct_molecules"] == 1
    assert result["complete_pool_distinct_molecules"] == 2
    assert region_minimum_delivery(goal, {}, appendix, intervals)["status"] == "fulfilled"


def test_native_export_applies_minimum_after_required_evidence_filtering(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    rows = [candidate(i, {0} if i < 4 else {1}) for i in range(7)]
    for row in rows:
        chrom, start, end = intervals(row)[0]
        from aso_chemistry import encode_idt_oligo
        row["IDT"] = encode_idt_oligo(row["Synthesis"], "steric-hindrance")
        row.update(Chrom=chrom, Start=start, End=end, Strand="+", cross_species_pass=True)
    # A high-ranked candidate with failed required evidence must remain excluded.
    rows[4]["cross_species_pass"] = False
    constraints = [{"constraint_id": name, "label": name, "source_type": "explicit_rows",
                    "operator": "union", "coordinate_space": "genomic",
                    "aso_overlap_requirement": "full_containment",
                    "materialized_rows": [["chrFixture", i * 100, i * 100 + 40, name, 0, "+"]]}
                   for i, name in enumerate("AB")]
    goal = {"min_asos_per_site": 2, "candidate_limit": 4,
            "species": "human", "cross_species_requirement": {"species": ["human", "mouse"], "required": True},
            "materialized_region_constraints": {"global": constraints},
            "steric_design_jobs": [{"program_id": "one", "region_rows": [["chrFixture", 0, 140]]}]}
    assert len(requested_region_minimums(goal)) == 2
    monkeypatch.setattr(pipeline, "_collect_generated_candidate_rows", lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, "_collect_generated_stage_sets", lambda *a: {})
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda *a: {"structured_goal": goal})
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    panel = json.loads(Path(artifacts["generated_aso_set_json"]).read_text())
    assert len(panel["rows"]) == 4
    assert {row["selection_rank"] for row in panel["rows"]} == {1, 2, 6, 7}
    assert panel["candidate_selection"]["status"] == "complete"


def test_interval_summary_keeps_coordinate_reference_and_strand_scopes_separate():
    from aso_agents.region_interval_summary import nominated_region_interval_summary
    constraints = [{"source_type": "explicit_rows", "coordinate_space": space,
                    "label": label, "operator": "union", "materialized_rows": [[ref, a, b, label, 0, strand]]}
                   for space, ref, strand, label, a, b in [
                       ("cdna", "TX_A", ".", "first", 50, 88),
                       ("cdna", "TX_A", ".", "second", 79, 110),
                       ("genomic", "TX_A", ".", "foreign_space", 0, 500),
                       ("cdna", "TX_B", ".", "foreign_reference", 0, 500),
                       ("cdna", "TX_A", "-", "foreign_strand", 0, 500)]]
    results = nominated_region_interval_summary({"materialized_region_constraints": {"global": constraints}})
    assert len(results) == 1
    assert results[0]["union_intervals"] == [[50, 110]] and results[0]["union_length_nt"] == 60
    assert results[0]["intersection_of_all_regions"] == [[79, 88]] and results[0]["intersection_length_nt"] == 9


def test_interval_summary_merges_duplicate_and_split_blocks_without_bridging_gaps():
    from aso_agents.region_interval_summary import nominated_region_interval_summary
    a = {"source_type": "explicit_rows", "label": "a", "materialized_rows": [
        ["chrFixture", 0, 12, "a"], ["chrFixture", 4, 12, "a"], ["chrFixture", 20, 30, "a"]]}
    b = {"source_type": "explicit_rows", "label": "b", "materialized_rows": [["chrFixture", 10, 25, "b"]]}
    excluded = {"source_type": "explicit_rows", "operator": "exclude", "label": "excluded",
                "materialized_rows": [["chrFixture", 0, 500, "excluded"]]}
    result = nominated_region_interval_summary({"materialized_region_constraints": {"global": [a, b, excluded]}})[0]
    assert result["union_intervals"] == [[0, 30]]
    assert result["intersection_of_all_regions"] == [[10, 12], [20, 25]]
    b["materialized_rows"] = [["chrFixture", 40, 50, "b"]]
    result = nominated_region_interval_summary({"materialized_region_constraints": {"global": [a, b]}})[0]
    assert result["union_intervals"] == [[0, 12], [20, 30], [40, 50]]
    assert result["intersection_of_all_regions"] == [] and result["intersection_length_nt"] == 0
