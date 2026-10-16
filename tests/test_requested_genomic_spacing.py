"""Requested genomic spacing, checked independently of the interval sweep."""
import asyncio
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import random
from types import SimpleNamespace

import jsonschema
import pytest

from aso_agents.region_panel_selection import select_region_minimum_panel


def candidate(index, blocks):
    sequence = "ACGTACGTACGTACGT" + "".join("ACGT"[(index // 4**n) % 4] for n in range(4))
    return {"Synthesis": sequence, "IDT": "5-10-5", "selection_rank": index + 1,
            "binding_intervals_bed0_half_open": blocks}


def intervals(row):
    return row["binding_intervals_bed0_half_open"]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("gap", [0, 1, 7, 23])
def test_requested_gap_matches_independent_exhaustive_panel_oracle(seed, gap):
    rng = random.Random(seed)
    rows = [candidate(i, [["chr2", start, start + 20]])
            for i, start in enumerate(rng.sample(range(120), 8))]
    limit = rng.randint(2, 4)
    feasible = []
    for size in range(limit + 1):
        for indices in combinations(range(len(rows)), size):
            blocks = [intervals(rows[i])[0] for i in indices]
            # Half-open intervals have start_right - end_left untargeted bases.
            if all(max(a[1], b[1]) - min(a[2], b[2]) >= gap
                   for a, b in combinations(blocks, 2)):
                feasible.append(indices)
    best = min(feasible, key=lambda ids: (-len(ids), sum(i + 1 for i in ids)))
    before = deepcopy(rows)
    selected, summary = select_region_minimum_panel(rows, [], intervals,
        global_limit=limit, nonoverlap=True, minimum_genomic_gap_nt=gap)
    assert len(selected) == len(best)
    assert sum(r["selection_rank"] for r in selected) == sum(i + 1 for i in best)
    assert summary["minimum_genomic_gap_nt"] == gap
    assert summary["status"] == "complete"
    assert rows == before


@pytest.mark.parametrize("gap", [0, 1, 4, 13])
def test_exact_spacing_boundary_does_not_require_an_extra_base(gap):
    rows = [candidate(0, [["chr2", 0, 20]]), candidate(1, [["chr2", 20 + gap, 40 + gap]])]
    chosen, summary = select_region_minimum_panel(rows, [], intervals,
        global_limit=2, global_minimum=2, nonoverlap=True, minimum_genomic_gap_nt=gap)
    assert len(chosen) == 2 and summary["status"] == "complete"
    if gap:
        rows[1]["binding_intervals_bed0_half_open"][0][1] -= 1
        chosen, summary = select_region_minimum_panel(rows, [], intervals,
            global_limit=2, global_minimum=2, nonoverlap=True, minimum_genomic_gap_nt=gap)
        assert len(chosen) == 1 and summary["status"] == "unfulfilled"
        assert summary["feasibility"] == "no_feasible_panel_in_verified_pool"


def test_gap_uses_all_placements_and_exonic_blocks_without_padding_molecule_count():
    first = candidate(0, [["chr2", 0, 10], ["chr2", 90, 100]])
    repeat = {**first, "binding_intervals_bed0_half_open": [["2", 150, 170]]}
    rows = [first, repeat, candidate(1, [["2", 174, 194]]),
            candidate(2, [["chr2", 15, 35]]), candidate(3, [["chr3", 0, 20]])]
    chosen, summary = select_region_minimum_panel(rows, [], intervals,
        global_limit=3, global_minimum=3, minimum_genomic_gap_nt=5)
    assert [r["selection_rank"] for r in chosen] == [1, 3, 4]
    assert summary["eligible_count"] == 4 and summary["selected_count"] == 3
    assert summary["hard_nonoverlap"] is True


@pytest.mark.parametrize("bad", [-1, True, 2.5, "3", None])
def test_invalid_spacing_is_never_coerced_to_zero(bad):
    with pytest.raises(ValueError, match="minimum_genomic_gap_nt"):
        select_region_minimum_panel([], [], intervals, minimum_genomic_gap_nt=bad)


@pytest.mark.parametrize("gap", [0, 11])
@pytest.mark.parametrize("applies_to_all", [False, True])
def test_quoted_global_spacing_reaches_scoped_executable_goal(monkeypatch, gap, applies_to_all):
    import run_goal_intake as intake
    from aso_agents.goal_contract import canonical_structured_goal_schema
    request = f"Require at least {gap} untargeted genomic bases between every selected target interval."
    goal = {"candidate_minimum": 2, "candidate_limit": 2, "design_programs": [{
        "program_id": "neutral", "label": "Neutral", "modality": "gapmer",
        "intent": "knockdown", "objective": "Requested panel"}]}
    draft = {"status": "ready", "structured_goal": deepcopy(goal),
             "request_envelope": {"tasks": [{"task_type": "design_new_asos",
                 "context": {"structured_goal": deepcopy(goal)}}]}}
    audit = {"candidate_genomic_spacing": {"minimum_gap_nt": gap,
        "applies_to_all_candidates": applies_to_all, "quote": request}}

    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()

    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    nested = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    for container in [result["structured_goal"], nested]:
        if applies_to_all:
            assert container["candidate_minimum_genomic_gap_nt"] == gap
            assert container["candidate_selection_policy"] == "nonoverlapping_target_sites"
            jsonschema.validate(container, canonical_structured_goal_schema())
        else:
            assert "candidate_minimum_genomic_gap_nt" not in container


def test_export_enforces_spacing_even_if_policy_was_left_ranked(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    from aso_chemistry import encode_idt_oligo
    rows = [candidate(i, [["chr2", start, start + 20]]) for i, start in enumerate([0, 20, 29])]
    for row in rows:
        row["Sequence"] = row["Synthesis"].translate(str.maketrans("ACGT", "TGCA"))[::-1]
        row["IDT"] = encode_idt_oligo(row["Synthesis"], "steric-hindrance")
        chrom, start, end = intervals(row)[0]
        row.update(Chrom=chrom, Start=start, End=end, Strand="-", source_modality="steric")
    goal = {"candidate_minimum": 2, "candidate_limit": 2, "species": "human",
            "candidate_selection_policy": "ranked", "candidate_minimum_genomic_gap_nt": 9}
    monkeypatch.setattr(pipeline, "_collect_generated_candidate_rows", lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, "_collect_generated_stage_sets", lambda *a: {})
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda *a: {"structured_goal": goal})
    result = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    panel = json.loads(Path(result["generated_aso_set_json"]).read_text())
    assert [r["selection_rank"] for r in panel["rows"]] == [1, 3]
    assert panel["candidate_selection"]["minimum_genomic_gap_nt"] == 9


def test_native_gapmer_spacing_survives_one_based_coordinates_and_retains_pool(tmp_path):
    from aso_agents import pipeline_tools as pipeline
    rows = []
    for i, start in enumerate([101, 121, 130]):
        row = candidate(i, [])
        row.pop("binding_intervals_bed0_half_open")
        row.update(Genomic_chrom="chr2", Genomic_start=start, Genomic_end=start + 19,
                   Genomic_strand="-", coordinate_system="1based_inclusive")
        rows.append(row)
    path = tmp_path / "scored.json"
    path.write_text(json.dumps({"rows": rows}))
    result = pipeline._apply_gapmer_constraints_to_result(
        gapmer_results={"scored_tiles_path": str(path)},
        structured_goal={"candidate_minimum": 2, "candidate_limit": 2,
                         "candidate_minimum_genomic_gap_nt": 9},
        runtime=SimpleNamespace(_annotate_gapmer_selection_ranks=lambda path: {"row_count": 3},
            filter_tiles_json=lambda path, constraints: {"applied": False, "output_path": path}),
        apply_candidate_limit=True)
    assert "error" not in result
    selected = json.loads(Path(result["gapmer_results"]["scored_tiles_path"]).read_text())["rows"]
    assert [r["selection_rank"] for r in selected] == [1, 3]
    summary = result["gapmer_results"]["candidate_selection"]
    assert summary["minimum_genomic_gap_nt"] == 9
    omitted = json.loads(Path(summary["unselected_path"]).read_text())["rows"]
    assert [r["selection_rank"] for r in omitted] == [2]
