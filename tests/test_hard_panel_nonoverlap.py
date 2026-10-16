"""Strict physical panel selection checked against an independent finite oracle."""
import asyncio
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import random

import jsonschema
import pytest

from aso_agents.region_panel_selection import select_region_minimum_panel


def row(index, blocks):
    sequence = "ACGTACGTACGTACGT" + "".join("ACGT"[(index // (4 ** n)) % 4] for n in range(4))
    return {"Synthesis": sequence, "IDT": "5-10-5", "selection_rank": index + 1,
            "binding_intervals_bed0_half_open": blocks}


def intervals(candidate, **_):
    return candidate.get("binding_intervals_bed0_half_open", [])


def select(rows, limit=2, minimum=2, groups=()):
    return select_region_minimum_panel(rows, list(groups), intervals,
        global_limit=limit, global_minimum=minimum, nonoverlap=True)


def test_joint_selection_recovers_feasible_pair_that_greedy_ranking_blocks():
    rows = [row(0, [["chr1", 10, 30]]), row(1, [["chr1", 0, 20]]), row(2, [["chr1", 20, 40]])]
    original = deepcopy(rows)
    selected, summary = select(rows)
    assert [r["selection_rank"] for r in selected] == [2, 3]
    assert summary["status"] == "complete" and summary["solver_status"] == "optimal_rank_sum"
    assert rows == original


@pytest.mark.parametrize("seed", range(20))
def test_optimal_count_and_rank_cost_match_independent_exhaustive_search(seed):
    randomizer = random.Random(seed)
    rows = [row(i, [["chr1", start, start + 20]])
            for i, start in enumerate(randomizer.sample(range(90), 7))]
    limit = randomizer.randint(2, 4)
    feasible = []
    for size in range(limit + 1):
        for indices in combinations(range(len(rows)), size):
            blocks = [intervals(rows[i])[0] for i in indices]
            if all(a[2] <= b[1] or b[2] <= a[1] for a, b in combinations(blocks, 2)):
                feasible.append(indices)
    best = min(feasible, key=lambda ids: (-len(ids), sum(i + 1 for i in ids)))
    selected, summary = select(rows, limit=limit, minimum=0)
    assert len(selected) == len(best)
    assert sum(r["selection_rank"] for r in selected) == sum(i + 1 for i in best)
    assert summary["status"] == "complete"


def test_hard_shortfall_never_fills_with_overlapping_targets():
    rows = [row(0, [["chr1", 0, 20]]), row(1, [["1", 1, 21]])]
    selected, summary = select(rows)
    assert len(selected) == 1
    assert summary["status"] == "unfulfilled"
    assert summary["feasibility"] == "no_feasible_panel_in_verified_pool"
    assert "pool" in summary["feasibility_scope"]


def test_spliced_physical_blocks_do_not_cover_the_unbound_intron():
    rows = [row(0, [["chr1", 0, 10], ["chr1", 90, 100]]),
            row(1, [["chr1", 20, 40]]), row(2, [["chr2", 0, 20]])]
    selected, summary = select(rows, limit=3, minimum=3)
    assert len(selected) == 3 and summary["status"] == "complete"


def test_repeated_physical_molecule_and_all_its_placements_remain_one_identity():
    first = row(0, [["chr1", 0, 20]])
    second_placement = {**first, "binding_intervals_bed0_half_open": [["chr1", 50, 70]]}
    competitor = row(1, [["chr1", 55, 75]])
    disjoint = row(2, [["chr1", 100, 120]])
    selected, summary = select([first, deepcopy(first), second_placement, competitor, disjoint])
    assert [r["selection_rank"] for r in selected] == [1, 3]
    assert summary["eligible_count"] == 3 and summary["selected_count"] == 2


@pytest.mark.parametrize("bad", [[], [["chr1", -1, 19]], [["chr1", 20, 20]],
                                 [["", 0, 20]], [["chr1", True, 20]], [["chr1", 0, "20"]]])
def test_unknown_or_invalid_physical_geometry_cannot_establish_nonoverlap(bad):
    selected, summary = select([row(0, [["chr1", 0, 20]]), row(1, bad)])
    assert len(selected) == 1 and summary["status"] == "unfulfilled"
    assert summary["unverified_geometry_rows"] == 1
    assert summary["feasibility"] == "unresolved"


def test_region_counts_are_enforced_alongside_nonoverlap():
    rows = [row(0, [["chr1", 0, 20]]), row(1, [["chr1", 20, 40]]),
            row(2, [["chr1", 100, 120]])]
    groups = [{"group_id": str(start), "label": str(start), "minimum": 1,
               "selectors": [{"rows": [["chr1", start, end]],
                              "requirement": "full_containment", "filters": []}]}
              for start, end in [(0, 50), (100, 150)]]
    selected, summary = select(rows, groups=groups)
    assert [r["selection_rank"] for r in selected] == [1, 3]
    assert [g["selected_count"] for g in summary["groups"]] == [1, 1]


@pytest.mark.parametrize("quote_valid", [True, False])
def test_evidence_bound_hard_policy_reaches_scoped_executable_goal(monkeypatch, quote_valid):
    import run_goal_intake as intake
    from aso_agents.goal_contract import canonical_structured_goal_schema
    request = "Design exactly two gapmers with mutually nonoverlapping target intervals."
    goal = {"candidate_minimum": 2, "candidate_limit": 2, "design_programs": [{
        "program_id": "neutral", "label": "Neutral", "modality": "gapmer",
        "intent": "knockdown", "objective": "Requested panel"}]}
    draft = {"status": "ready", "structured_goal": deepcopy(goal),
             "request_envelope": {"tasks": [{"task_type": "design_new_asos",
                 "context": {"structured_goal": deepcopy(goal)}}]}}
    audit = {"candidate_site_diversity": {"required": True, "hard_nonoverlap": True,
        "quote": "mutually nonoverlapping target intervals" if quote_valid else "No such quote"}}

    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()

    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    if quote_valid:
        assert result["structured_goal"]["candidate_selection_policy"] == "nonoverlapping_target_sites"
        nested = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
        assert nested["candidate_selection_policy"] == "nonoverlapping_target_sites"
        jsonschema.validate({**goal, "candidate_selection_policy": "nonoverlapping_target_sites"},
                            canonical_structured_goal_schema())
    else:
        assert result["structured_goal"].get("candidate_selection_policy") != "nonoverlapping_target_sites"


def test_native_export_selects_from_full_pool_across_duplicate_program_rows(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    from aso_chemistry import encode_idt_oligo
    rows = [row(i, [["chr1", start, start + 20]]) for i, start in enumerate([10, 0, 20])]
    for candidate in rows:
        candidate["Sequence"] = candidate["Synthesis"].translate(str.maketrans("ACGT", "TGCA"))[::-1]
        candidate["IDT"] = encode_idt_oligo(candidate["Synthesis"], "steric-hindrance")
        chrom, start, end = intervals(candidate)[0]
        candidate.update(Chrom=chrom, Start=start, End=end, Strand="+", source_modality="steric")
    rows = [{**candidate, "program_id": program} for program in ["first", "second"] for candidate in rows]
    goal = {"candidate_minimum": 2, "candidate_limit": 2, "species": "human",
            "candidate_selection_policy": "nonoverlapping_target_sites",
            "design_programs": [{"program_id": program, "modality": "steric", "candidate_limit": 2}
                                for program in ["first", "second"]]}
    monkeypatch.setattr(pipeline, "_collect_generated_candidate_rows", lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, "_collect_generated_stage_sets", lambda *a: {})
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda *a: {"structured_goal": goal})
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    panel = json.loads(Path(artifacts["generated_aso_set_json"]).read_text())
    assert len(panel["rows"]) == 2
    assert len({r["synthesized_antisense_sequence"] for r in panel["rows"]}) == 2
    assert {r["selection_rank"] for r in panel["rows"]} == {2, 3}
    assert panel["candidate_selection"]["hard_nonoverlap"] is True
    assert panel["candidate_selection"]["status"] == "complete"


def test_gapmer_final_limit_uses_joint_selection_and_preserves_unselected_evidence(tmp_path):
    from aso_agents import pipeline_tools as pipeline
    from types import SimpleNamespace
    rows = [row(i, [["chr1", start, start + 20]]) for i, start in enumerate([10, 0, 20])]
    path = tmp_path / "scored.json"
    path.write_text(json.dumps({"rows": rows}))

    def rank(_path):
        return {"row_count": 3}

    result = pipeline._apply_gapmer_constraints_to_result(
        gapmer_results={"scored_tiles_path": str(path)},
        structured_goal={"candidate_minimum": 2, "candidate_limit": 2,
                         "candidate_selection_policy": "nonoverlapping_target_sites"},
        runtime=SimpleNamespace(_annotate_gapmer_selection_ranks=rank,
            filter_tiles_json=lambda path, constraints: {"applied": False, "output_path": path}),
        apply_candidate_limit=True)
    assert "error" not in result
    selected = json.loads(Path(result["gapmer_results"]["scored_tiles_path"]).read_text())["rows"]
    assert [r["selection_rank"] for r in selected] == [2, 3]
    summary = result["gapmer_results"]["candidate_selection"]
    assert summary["hard_nonoverlap"] is True
    rejected = json.loads(Path(summary["unselected_path"]).read_text())["rows"]
    assert [r["selection_rank"] for r in rejected] == [1]
