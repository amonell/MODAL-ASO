"""Explicit joint objectives compared with a separate finite-panel oracle."""
from copy import deepcopy
import asyncio
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from aso_agents.region_panel_selection import select_region_minimum_panel


def row(index, gc_count, start, length=20):
    # GC counts are controlled; distinct placements do not create molecule IDs.
    sequence = "G" * gc_count + "A" * (length - gc_count - 2) + "T" * 2
    if index % 2:
        sequence = sequence[::-1]
    return {"Synthesis": sequence, "IDT": "neutral-chemistry", "selection_rank": index + 1,
            "binding_intervals_bed0_half_open": [["chr2", start, start + length]]}


def blocks(candidate, **_):
    return candidate["binding_intervals_bed0_half_open"]


def criterion_key(rows, ids, objective):
    values = []
    for index in ids:
        sequence = rows[index]["Synthesis"]
        metric = objective["metric"]
        value = (Fraction(sum(b in "GC" for b in sequence), len(sequence)) if metric == "gc_fraction"
                 else Fraction(blocks(rows[index])[0][1] + 1))
        if "absolute_distance_from" in objective:
            value = abs(value - Fraction(str(objective["absolute_distance_from"])))
        values.append(value)
    aggregate = objective["aggregate"]
    if aggregate == "sum":
        value = (sum(values),)
    elif aggregate == "maximum":
        value = (max(values),)
    elif aggregate == "minimum":
        value = (min(values),)
    else:
        value = tuple(sorted(values, reverse=aggregate == "sorted_descending"))
    return tuple(-v for v in value) if objective["direction"] == "maximize" else value


@pytest.mark.parametrize("seed", range(24))
def test_joint_objectives_match_exhaustive_feasible_panel_order(seed):
    rng = random.Random(seed)
    starts = rng.sample(range(160), 7)
    rows = [row(i, i + 4, start) for i, start in enumerate(starts)]
    gap = rng.choice([0, 3, 11])
    limit = rng.randint(2, 4)
    objectives = [{"metric": "gc_fraction", "absolute_distance_from": rng.choice([.3, .45, .6]),
                   "aggregate": rng.choice(["sum", "maximum", "minimum", "sorted_ascending", "sorted_descending"]),
                   "direction": rng.choice(["minimize", "maximize"])},
                  {"metric": "genomic_start_1based", "aggregate": "sorted_ascending", "direction": "minimize"}]
    feasible = []
    for size in range(1, limit + 1):
        for ids in combinations(range(len(rows)), size):
            if all(max(starts[a], starts[b]) - min(starts[a] + 20, starts[b] + 20) >= gap
                   for a, b in combinations(ids, 2)):
                feasible.append(ids)
    key = lambda ids: (-len(ids), *(criterion_key(rows, ids, obj) for obj in objectives), sum(i + 1 for i in ids))
    best = min(feasible, key=key)
    before = deepcopy(rows)
    selected, summary = select_region_minimum_panel(rows, [], blocks,
        global_limit=limit, global_minimum=1, nonoverlap=True,
        minimum_genomic_gap_nt=gap, panel_objectives=objectives)
    actual = tuple(r["selection_rank"] - 1 for r in selected)
    assert key(actual) == key(best)
    assert summary["status"] == "complete"
    assert summary["requested_panel_ranking"]["status"] == "complete"
    assert rows == before


def test_worst_distance_precedes_sum_and_native_rank_sum():
    # Native-rank and minimum-sum both choose 1+2. Minimax chooses 3+4.
    rows = [row(0, 10, 0), row(1, 18, 40), row(2, 5, 1), row(3, 15, 41)]
    for candidate, starts in zip(rows, [(0, 100), (40, 200), (1, 201), (41, 101)]):
        candidate["binding_intervals_bed0_half_open"] = [["chr2", start, start + 20] for start in starts]
    objectives = [{"metric": "gc_fraction", "absolute_distance_from": .5,
                   "aggregate": aggregate, "direction": "minimize"}
                  for aggregate in ["maximum", "sum"]]
    selected, summary = select_region_minimum_panel(rows, [], blocks,
        global_limit=2, global_minimum=2, nonoverlap=True, panel_objectives=objectives)
    # Multiple physical placements forbid cross pairs without conflating them
    # with additional molecules. The exhaustive oracle checks all placements.
    feasible = [(a, b) for a, b in combinations(range(4), 2)
                if all(max(x[1], y[1]) - min(x[2], y[2]) >= 0
                       for x in blocks(rows[a]) for y in blocks(rows[b]))]
    expected = min(feasible, key=lambda ids: (*(criterion_key(rows, ids, obj) for obj in objectives), sum(i + 1 for i in ids)))
    assert tuple(r["selection_rank"] - 1 for r in selected) == expected
    assert expected == (2, 3)
    assert summary["requested_panel_ranking"]["status"] == "complete"


def test_unresolved_requested_coordinate_ranking_cannot_claim_native_optimality():
    rows = [row(0, 8, 0), row(1, 9, 30)]
    objective = {"metric": "mature_start_1based", "reference_transcript_id": "reference.4",
                 "aggregate": "sorted_ascending", "direction": "minimize"}
    selected, summary = select_region_minimum_panel(rows, [], blocks,
        global_limit=2, global_minimum=2, nonoverlap=True, panel_objectives=[objective])
    assert summary["status"] == "unfulfilled"
    assert summary["requested_panel_ranking"]["status"] == "unresolved"
    assert "reference" in summary["requested_panel_ranking"]["reason"]


@pytest.mark.parametrize("strand", ["+", "-"])
def test_requested_mature_start_uses_reference_exons_not_legacy_labels(strand):
    rows = [row(0, 8, 100), row(1, 8, 200)]
    rows[0]["Name"] = "reference.4_cdna999-1018"
    rows[1]["Name"] = "reference.4_cdna1-20"
    catalog = {"reference.4": {"transcript_id": "reference.4", "chrom": "chr2", "strand": strand,
        "exons_tx": [{"start": 101, "end": 140}, {"start": 201, "end": 240}]}}
    objective = {"metric": "mature_start_1based", "reference_transcript_id": "reference.4",
                 "aggregate": "sorted_ascending", "direction": "minimize"}
    selected, summary = select_region_minimum_panel(rows, [], blocks, global_limit=1,
        global_minimum=1, panel_objectives=[objective], reference_catalog=catalog)
    assert [r["selection_rank"] for r in selected] == ([1] if strand == "+" else [2])
    assert summary["requested_panel_ranking"]["objective_values"] == ([[1]] if strand == "+" else [[21]])


def priorities():
    return [{"metric": "gc_percent", "absolute_distance_from": 50,
             "aggregate": aggregate, "direction": "minimize"} for aggregate in ["maximum", "sum"]]


@pytest.mark.parametrize("scope,global_scope,quote_matches", [
    ("joint_panel", True, True), ("individual", True, True),
    ("joint_panel", False, True), ("joint_panel", True, False)])
def test_only_quoted_global_joint_priorities_are_synchronized(monkeypatch, scope, global_scope, quote_matches):
    import run_goal_intake as intake
    from aso_agents.goal_contract import canonical_structured_goal_schema
    from jsonschema import validate
    request = "Among all two-member panels, minimize the worst GC percentage-point distance from 50, then the sum."
    goal = {"candidate_minimum": 2, "candidate_limit": 2, "design_programs": [{
        "program_id": "neutral", "label": "Neutral", "modality": "gapmer",
        "intent": "knockdown", "objective": "Requested panel"}]}
    draft = {"status": "ready", "structured_goal": deepcopy(goal),
             "request_envelope": {"tasks": [{"task_type": "design_new_asos",
                 "context": {"structured_goal": deepcopy(goal)}}]}}
    audit = {"candidate_panel_ranking": {"scope": scope, "applies_to_all_candidates": global_scope,
        "criteria": priorities(), "quote": request if quote_matches else "an invented requirement"}}

    async def fake_run(*args, **kwargs):
        return SimpleNamespace(final_output=json.dumps(audit))

    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    nested = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    for container in [result["structured_goal"], nested]:
        if scope == "joint_panel" and global_scope and quote_matches:
            assert container["candidate_panel_objectives"] == priorities()
            validate(container, canonical_structured_goal_schema())
        else:
            assert "candidate_panel_objectives" not in container


def test_quoted_unsupported_objective_requests_clarification_without_losing_other_questions(monkeypatch):
    import run_goal_intake as intake
    request = "Choose the panel by its predicted potency."
    draft = {"status": "ready", "questions": ["Which chemistry?"], "structured_goal": {"design_programs": []}}
    audit = {"candidate_panel_ranking": {"scope": "joint_panel", "applies_to_all_candidates": True,
        "criteria": [{"metric": "potency", "aggregate": "sum", "direction": "maximize"}], "quote": request}}

    async def fake_run(*args, **kwargs):
        return SimpleNamespace(final_output=json.dumps(audit))

    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    assert result["status"] == "need_clarification"
    assert result["request_obligation_audit"]["status"] == "unresolved"
    assert "Which chemistry?" in result["questions"]
    assert "candidate_panel_objectives" not in result["structured_goal"]


@pytest.mark.parametrize("native_gapmer", [False, True])
@pytest.mark.parametrize("unresolved", [False, True])
def test_requested_ranking_reaches_native_selection_and_export(tmp_path, monkeypatch, native_gapmer, unresolved):
    from aso_agents import pipeline_tools as pipeline
    from aso_chemistry import encode_idt_oligo
    rows = [row(i, gc, i * 10) for i, gc in enumerate([7, 10, 11])]
    objectives = ([{"metric": "mature_start_1based", "reference_transcript_id": "absent.7",
        "aggregate": "sum", "direction": "minimize"}] if unresolved else priorities())
    goal = {"candidate_minimum": 2, "candidate_limit": 2, "species": "human",
            "candidate_panel_objectives": objectives}
    if native_gapmer:
        for item in rows:
            chrom, start, end = item.pop("binding_intervals_bed0_half_open")[0]
            item.update(Genomic_chrom=chrom, Genomic_start=start + 1, Genomic_end=end,
                        Genomic_strand="-", coordinate_system="1based_inclusive")
        path = tmp_path / "pool.json"
        path.write_text(json.dumps({"rows": rows}))
        result = pipeline._apply_gapmer_constraints_to_result(
            gapmer_results={"scored_tiles_path": str(path)}, structured_goal=goal,
            runtime=SimpleNamespace(_annotate_gapmer_selection_ranks=lambda path: {"row_count": 3},
                filter_tiles_json=lambda path, constraints: {"applied": False, "output_path": path}),
            apply_candidate_limit=True)
        assert "error" not in result
        summary = result["gapmer_results"]["candidate_selection"]
        selected = json.loads(Path(result["gapmer_results"]["scored_tiles_path"]).read_text())["rows"]
        omitted = json.loads(Path(summary["unselected_path"]).read_text())["rows"]
        assert sorted(item["selection_rank"] for item in selected + omitted) == [1, 2, 3]
    else:
        for item in rows:
            item["Sequence"] = item["Synthesis"].translate(str.maketrans("ACGT", "TGCA"))[::-1]
            item["IDT"] = encode_idt_oligo(item["Synthesis"], "steric-hindrance")
            chrom, start, end = blocks(item)[0]
            item.update(Chrom=chrom, Start=start, End=end, Strand="-", source_modality="steric")
        monkeypatch.setattr(pipeline, "_collect_generated_candidate_rows", lambda *a, **k: rows)
        monkeypatch.setattr(pipeline, "_collect_generated_stage_sets", lambda *a: {})
        monkeypatch.setattr(pipeline, "_goal_payload_context", lambda *a: {"structured_goal": goal})
        result = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
        panel = json.loads(Path(result["generated_aso_set_json"]).read_text())
        selected, summary = panel["rows"], panel["candidate_selection"]
    assert summary["status"] == ("unfulfilled" if unresolved else "complete")
    assert summary["requested_panel_ranking"]["status"] == ("unresolved" if unresolved else "complete")
    if not unresolved:
        # The optimal pair overlaps: an objective must not invent nonoverlap.
        assert [item["selection_rank"] for item in selected] == [2, 3]
        assert summary["requested_panel_ranking"]["objective_values"] == [5, 5]


@pytest.mark.parametrize("metric", ["genomic_start_1based", "mature_start_1based"])
@pytest.mark.parametrize("other_chrom,other_start", [("chr3", 100), ("chr2", 120)])
def test_multiple_placements_cannot_supply_an_arbitrary_ranking_coordinate(metric, other_chrom, other_start):
    first = row(0, 8, 100)
    second = deepcopy(first)
    second["binding_intervals_bed0_half_open"] = [[other_chrom, other_start, other_start + 20]]
    catalog = {"reference.4": {"transcript_id": "reference.4", "chrom": "chr2", "strand": "+",
        "exons_tx": [{"start": 101, "end": 200}]}}
    _, summary = select_region_minimum_panel([first, second], [], blocks,
        global_limit=1, global_minimum=1, reference_catalog=catalog,
        panel_objectives=[{"metric": metric, "reference_transcript_id": "reference.4",
                           "aggregate": "sum", "direction": "minimize"}])
    assert summary["status"] == "unfulfilled"
    assert summary["requested_panel_ranking"]["status"] == "unresolved"


@pytest.mark.parametrize("strand,expected", [("+", 31), ("-", 91)])
def test_spliced_target_projects_each_exonic_block(strand, expected):
    candidate = row(0, 8, 0)
    candidate["binding_intervals_bed0_half_open"] = [["chr2", 130, 140], ["chr2", 200, 210]]
    catalog = {"reference.4": {"transcript_id": "reference.4", "chrom": "chr2", "strand": strand,
        "exons_tx": [{"start": 101, "end": 140}, {"start": 201, "end": 300}]}}
    _, summary = select_region_minimum_panel([candidate], [], blocks, global_limit=1,
        panel_objectives=[{"metric": "mature_start_1based", "reference_transcript_id": "reference.4",
                           "aggregate": "sum", "direction": "minimize"}], reference_catalog=catalog)
    assert summary["status"] == "complete"
    assert summary["requested_panel_ranking"]["objective_values"] == [expected]


@pytest.mark.parametrize("defect", ["version_conflict", "intronic", "overlapping_exons", "noncontiguous", "missing_blocks"])
def test_reference_and_geometry_defects_fail_closed(defect):
    candidate = row(0, 8, 100)
    reference = {"transcript_id": "reference.4", "chrom": "chr2", "strand": "+",
        "exons_tx": [{"start": 101, "end": 140}, {"start": 201, "end": 240}]}
    if defect == "version_conflict":
        reference["metadata"] = {"versioned_id": "reference.5"}
    elif defect == "intronic":
        candidate["binding_intervals_bed0_half_open"] = [["chr2", 130, 150]]
    elif defect == "overlapping_exons":
        reference["exons_tx"][1]["start"] = 130
    elif defect == "noncontiguous":
        candidate["binding_intervals_bed0_half_open"] = [["chr2", 100, 110], ["chr2", 200, 210]]
    else:
        candidate["binding_intervals_bed0_half_open"] = []
    _, summary = select_region_minimum_panel([candidate], [], blocks, global_limit=1,
        panel_objectives=[{"metric": "mature_start_1based", "reference_transcript_id": "reference.4",
                           "aggregate": "sum", "direction": "minimize"}], reference_catalog={"reference.4": reference})
    assert summary["status"] == "unfulfilled"
    assert summary["requested_panel_ranking"]["status"] == "unresolved"


@pytest.mark.parametrize("minimum", [0, 1])
def test_empty_verified_pool_is_complete_only_when_empty_panel_is_permitted(minimum):
    _, summary = select_region_minimum_panel([], [], blocks, global_limit=2,
        global_minimum=minimum, panel_objectives=priorities())
    assert summary["status"] == ("complete" if minimum == 0 else "unfulfilled")
    if minimum == 0:
        assert summary["requested_panel_ranking"]["empty_panel"]
        assert summary["requested_panel_ranking"]["objective_values"] == [None, 0]
        assert summary["requested_panel_ranking"]["solver_calls"] == 0


@pytest.mark.parametrize("bad", [True, "0.5", float("nan"), float("inf")])
def test_invalid_distance_targets_do_not_fall_back_to_default_ranking(bad):
    objective = {**priorities()[0], "absolute_distance_from": bad}
    _, summary = select_region_minimum_panel([row(0, 8, 0)], [], blocks,
        global_limit=1, panel_objectives=[objective])
    assert summary["status"] == "unfulfilled"
    assert summary["requested_panel_ranking"]["status"] == "unresolved"


def test_joint_objectives_preserve_regional_minimums_and_program_caps():
    rows = [row(i, gc, start) for i, (gc, start) in enumerate([(10, 0), (9, 30), (8, 100), (6, 130)])]
    for candidate in rows[:3]:
        candidate["program_id"] = "limited"
    groups = [{"group_id": "far", "label": "far", "minimum": 1,
        "selectors": [{"rows": [["chr2", 100, 160]], "requirement": "any_overlap", "filters": []}]}]
    selected, summary = select_region_minimum_panel(rows, groups, blocks, global_limit=2,
        global_minimum=2, program_limits={"limited": 1}, per_site_limit=1, panel_objectives=priorities())
    assert [candidate["selection_rank"] for candidate in selected] == [1, 4]
    assert summary["status"] == "complete"
