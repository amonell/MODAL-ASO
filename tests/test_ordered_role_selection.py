"""Sequential role selection checked against independent sequence calculations."""
from copy import deepcopy
from fractions import Fraction
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
import asyncio
import csv
import json
import random

import pytest

from aso_agents.role_selection import select_ordered_roles, reconcile_role_obligation


def criterion(metric="gc_percent", **kwargs):
    return {"metric": metric, "direction": "minimize", **kwargs}


def candidate(sequence, program="pool_a", start=0, chemistry="chem_a", rank=1):
    return {"Synthesis": sequence, "IDT": chemistry, "program_id": program,
            "selection_rank": rank,
            "binding_intervals_bed0_half_open": [["chr4", start, start + len(sequence)]]}


def blocks(row, **_):
    return row["binding_intervals_bed0_half_open"]


def goal(policy="distinct_sequences", criteria=None):
    return {"candidate_minimum": 2, "candidate_limit": 2,
            "design_programs": [
                {"program_id": pid, "label": label, "modality": "steric", "intent": "binding",
                 "objective": "A requested scoped candidate pool"}
                for pid, label in [("pool_a", "First"), ("pool_b", "Second"), ("library", "Library")]],
            "candidate_role_selection": {"reuse_policy": policy, "roles": [
                {"role_id": label, "program_id": pid, "count": 1,
                 "criteria": deepcopy(criteria or [criterion(absolute_distance_from=50),
                                                   criterion("longest_homopolymer")])}
                for pid, label in [("pool_a", "First"), ("pool_b", "Second")]]}}


def test_role_order_overrides_global_ranking_and_skips_reused_sequence():
    rows = [candidate("ATATAT", "library", rank=1),
            candidate("ACGTAC", rank=9), candidate("AAAAAC", start=30, rank=2),
            candidate("ACGTAC", "pool_b", rank=3), candidate("CCCGGA", "pool_b", start=60, rank=4)]
    before = deepcopy(rows)
    selected, receipt = select_ordered_roles(rows, goal(), blocks)
    assert [r["Synthesis"] for r in selected] == ["ACGTAC", "CCCGGA"]
    assert [r["selection_rank"] for r in selected] == [9, 4]
    assert receipt["roles"][1]["skipped"][0]["reason"] == "sequence_already_assigned"
    assert receipt["roles"][1]["selected"][0]["role_rank"] == 2
    assert receipt["status"] == "complete" and rows == before
    changed = goal()
    changed["candidate_role_selection"]["roles"].reverse()
    reversed_panel, _ = select_ordered_roles(rows, changed, blocks)
    assert [r["Synthesis"] for r in reversed_panel] == ["ACGTAC", "AAAAAC"]


@pytest.mark.parametrize("policy,expected", [
    ("distinct_sequences", ["chem_a", "chem_c"]),
    ("distinct_molecules", ["chem_a", "chem_b"]),
    ("allow_shared_molecules", ["chem_a"]),
])
def test_reuse_policy_separates_sequences_chemistry_and_duplicate_observations(policy, expected):
    rows = [candidate("ACGTAC"), candidate("ACGTAC"),
            candidate("ACGTAC", "pool_b"), candidate("ACGTAC", "pool_b", chemistry="chem_b"),
            candidate("AAAAAC", "pool_b", chemistry="chem_c")]
    request = goal(policy)
    request["candidate_minimum"] = 1
    selected, result = select_ordered_roles(rows, request, blocks)
    assert [r["IDT"] for r in selected] == expected
    assert result["status"] == "complete"
    if policy == "allow_shared_molecules":
        assert len(selected[0]["role_assignments"]) == 2
    assert result["roles"][0]["eligible_count"] == 1


def test_role_shortfall_is_not_filled_from_another_program_or_duplicate_rows():
    request = goal()
    request["candidate_role_selection"]["roles"][0]["count"] = 2
    request["candidate_limit"] = 3
    rows = [candidate("ACGTAC"), candidate("ACGTAC"), candidate("CCCGGA", "library")]
    chosen, result = select_ordered_roles(rows, request, blocks)
    assert len(chosen) == 1
    assert result["status"] == "unfulfilled"
    assert [r["shortfall_count"] for r in result["roles"]] == [1, 1]


@pytest.mark.parametrize("seed", range(16))
def test_role_selection_matches_independent_sequential_oracle(seed):
    rng = random.Random(seed)
    sequences = []
    while len(sequences) < 10:
        seq = "".join(rng.choice("ACGT") for _ in range(9))
        if seq not in sequences:
            sequences.append(seq)
    rows = [candidate(seq, program, i * 13, rank=11 - i)
            for program in ("pool_a", "pool_b") for i, seq in enumerate(sequences) if rng.random() < .8]
    rng.shuffle(rows)
    request = goal(criteria=[criterion(absolute_distance_from=40), criterion("longest_homopolymer"),
                             criterion("genomic_start_1based", direction="maximize")])
    request["candidate_limit"] = 4
    request["candidate_minimum"] = 4
    for role in request["candidate_role_selection"]["roles"]:
        role["count"] = 2
    expected, used = [], set()
    for program in ("pool_a", "pool_b"):
        local = [r for r in rows if r["program_id"] == program]
        def independent_key(row):
            seq = row["Synthesis"]
            return (abs(Fraction(100 * sum(b in "GC" for b in seq), len(seq)) - 40),
                    max(len(list(g)) for _, g in groupby(seq)), -(blocks(row)[0][1] + 1))
        retained = 0
        for row in sorted(local, key=independent_key):
            if row["Synthesis"] not in used:
                expected.append(row["Synthesis"])
                used.add(row["Synthesis"])
                retained += 1
            if retained == 2:
                break
    selected, result = select_ordered_roles(rows, request, blocks)
    assert [r["Synthesis"] for r in selected] == expected
    assert result["status"] == "complete"
    assert all(len(r["role_assignments"][0]["exact_values"]) == 3 for r in selected)


@pytest.mark.parametrize("strand,expected", [("+", "ACGTAC"), ("-", "ATCGAC")])
def test_role_mature_coordinate_uses_verified_reference_and_strand(strand, expected):
    request = goal(criteria=[criterion("mature_start_1based", reference_transcript_id="ref.8")])
    request["candidate_role_selection"]["roles"] = request["candidate_role_selection"]["roles"][:1]
    request["candidate_minimum"] = 1
    request["isoform_transcripts"] = {"ref.8": {"transcript_id": "ref.8", "chrom": "chr4",
        "strand": strand, "exons_tx": [{"start": 101, "end": 110}, {"start": 201, "end": 210}]}}
    rows = [candidate("ACGTAC", start=100), candidate("ATCGAC", start=200)]
    rows[0]["Name"] = "misleading_cdna999"
    selected, result = select_ordered_roles(rows, request, blocks)
    assert selected[0]["Synthesis"] == expected and result["status"] == "complete"


def test_junction_spanning_target_uses_all_blocks_for_mature_ranking():
    request = goal(criteria=[criterion("mature_start_1based", reference_transcript_id="ref.8")])
    request["candidate_role_selection"]["roles"] = request["candidate_role_selection"]["roles"][:1]
    request["candidate_minimum"] = 1
    request["isoform_transcripts"] = {"ref.8": {"transcript_id": "ref.8", "chrom": "chr4",
        "strand": "-", "exons_tx": [{"start": 101, "end": 110}, {"start": 201, "end": 210}]}}
    row = candidate("ACGTAC")
    row["binding_intervals_bed0_half_open"] = [["chr4", 200, 203], ["chr4", 107, 110]]
    chosen, result = select_ordered_roles([row], request, blocks)
    assert result["status"] == "complete"
    assert chosen[0]["role_assignments"][0]["values"] == [8]


@pytest.mark.parametrize("problem", ["missing_program", "unknown_row", "missing_chemistry", "duplicate_role",
    "missing_reference", "ambiguous_coordinate", "nonfinite", "joint_objective", "per_site", "region_unresolved"])
def test_unresolvable_contract_never_falls_back_to_native_rank(problem):
    request, rows = goal(), [candidate("ACGTAC"), candidate("CCCGGA", "pool_b")]
    region_status = "complete"
    if problem == "missing_program":
        request["candidate_role_selection"]["roles"][0]["program_id"] = "absent"
    elif problem == "unknown_row":
        rows[0]["program_id"] = "absent"
    elif problem == "missing_chemistry":
        rows[0].pop("IDT")
    elif problem == "duplicate_role":
        request["candidate_role_selection"]["roles"][1]["role_id"] = "First"
    elif problem == "missing_reference":
        request["candidate_role_selection"]["roles"][1]["criteria"] = [
            criterion("mature_start_1based", reference_transcript_id="unavailable.2")]
    elif problem == "ambiguous_coordinate":
        request["candidate_role_selection"]["roles"][0]["criteria"] = [criterion("genomic_start_1based")]
        rows.append(candidate("ACGTAC", start=80))
    elif problem == "nonfinite":
        request["candidate_role_selection"]["roles"][0]["criteria"][0]["absolute_distance_from"] = float("nan")
    elif problem == "joint_objective":
        request["candidate_panel_objectives"] = [{**criterion(), "aggregate": "sum"}]
    elif problem == "per_site":
        request["min_asos_per_site"] = 1
    else:
        region_status = "unresolved"
    selected, result = select_ordered_roles(rows, request, blocks, region_status=region_status)
    assert selected == [] and result["status"] == "unresolved"


@pytest.mark.parametrize("constraint,expected_reason", [
    ({"candidate_limit": 1, "candidate_minimum": 1}, "global_maximum"),
    ({"candidate_selection_policy": "nonoverlapping_target_sites"}, "physical_nonoverlap_or_genomic_gap"),
    ({"candidate_minimum_genomic_gap_nt": 3}, "physical_nonoverlap_or_genomic_gap"),
])
def test_hard_bounds_are_not_relaxed_to_fill_later_roles(constraint, expected_reason):
    request = {**goal(), **constraint}
    selected, result = select_ordered_roles(
        [candidate("ACGTAC"), candidate("CCCGGA", "pool_b", start=4)], request, blocks)
    assert len(selected) == 1 and result["status"] == "unfulfilled"
    assert result["roles"][1]["skipped"][0]["reason"] == expected_reason


def test_exact_genomic_gap_passes_and_program_bounds_are_enforced():
    request = {**goal(), "candidate_minimum_genomic_gap_nt": 3}
    rows = [candidate("ACGTAC"), candidate("CCCGGA", "pool_b", start=9)]
    assert select_ordered_roles(rows, request, blocks)[1]["status"] == "complete"
    request["design_programs"][0]["candidate_minimum"] = 2
    assert select_ordered_roles(rows, request, blocks)[1]["program_counts"][0]["shortfall_count"] == 1


def role_obligation(request):
    quote = "Process First then Second, choosing one per role by GC distance from 50, then longest run; skip reused sequences."
    contract = request["candidate_role_selection"]
    return {"quote": quote, "reuse_policy": contract["reuse_policy"],
            "roles": [{k: deepcopy(v) for k, v in role.items() if k != "program_id"} for role in contract["roles"]]}


@pytest.mark.parametrize("bad", ["ambiguous", "missing", "quote", "duplicate", "invalid_identity"])
def test_audit_does_not_guess_role_program_binding(bad):
    request = goal()
    audit = role_obligation(request)
    if bad == "ambiguous":
        request["candidate_role_selection"]["roles"].append({
            **request["candidate_role_selection"]["roles"][0], "program_id": "pool_b"})
    elif bad == "missing":
        request.pop("candidate_role_selection")
        request["design_programs"][0]["label"] = "First-like-but-not-the-role"
    elif bad == "quote":
        audit["quote"] = "unattested instructions"
    elif bad == "duplicate":
        audit["roles"].append(deepcopy(audit["roles"][0]))
    else:
        audit["roles"][0]["role_id"] = ["First"]
    before = deepcopy(request)
    assert reconcile_role_obligation(request, audit, role_obligation(goal())["quote"])["status"] == "unresolved"
    assert request == before


def test_quoted_roles_replace_spurious_global_aggregation_in_nested_goal(monkeypatch):
    import run_goal_intake as intake
    from aso_agents.goal_contract import canonical_structured_goal_schema
    from jsonschema import validate
    request = goal()
    audit = role_obligation(request)
    request["candidate_panel_objectives"] = [{**criterion(), "aggregate": "sum"}]
    request["candidate_limit_per_site"] = 2
    draft = {"status": "ready", "structured_goal": request, "request_envelope": {"tasks": [
        {"task_type": "design_new_asos", "context": {"structured_goal": deepcopy(request)}}]}}
    async def fake_run(*args, **kwargs):
        return SimpleNamespace(final_output=json.dumps({"candidate_role_ranking": audit, "candidate_panel_ranking": None}))
    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, audit["quote"], []))
    assert result["status"] == "ready"
    for value in [result["structured_goal"], result["request_envelope"]["tasks"][0]["context"]["structured_goal"]]:
        assert value["candidate_role_selection"] == goal()["candidate_role_selection"]
        assert "candidate_panel_objectives" not in value and "candidate_limit_per_site" not in value
        validate(value, canonical_structured_goal_schema())


def test_native_gapmer_keeps_full_pool_until_final_role_selection(tmp_path):
    from aso_agents import pipeline_tools as pipeline
    rows = [candidate(seq, pid) for pid in ("pool_a", "pool_b") for seq in ("ACGTAC", "CCCGGA", "AAAAAC")]
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"rows": rows}))
    request = goal()
    request.pop("design_programs")  # No extra biological cohort constraints in this isolated integration.
    result = pipeline._apply_gapmer_constraints_to_result(
        gapmer_results={"scored_tiles_path": str(path)}, structured_goal=request,
        runtime=SimpleNamespace(_annotate_gapmer_selection_ranks=lambda path: {"row_count": 6},
            filter_tiles_json=lambda path, constraints: {"applied": False, "output_path": path}),
        apply_candidate_limit=True)
    assert "error" not in result
    assert len(json.loads(Path(result["gapmer_results"]["scored_tiles_path"]).read_text())["rows"]) == 6
    assert result["gapmer_results"]["candidate_selection"]["status"] == "deferred"


def test_native_export_preserves_roles_full_eligible_pool_and_delivery_gate(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    from aso_agents.design_delivery import design_delivery
    from aso_chemistry import encode_idt_oligo
    sequences = ["ACGT" * 5, "AAAAC" * 4, "CCGGA" * 4]
    rows = [candidate(sequences[0], rank=9), candidate(sequences[1], start=30),
            candidate(sequences[0], "pool_b"), candidate(sequences[2], "pool_b", start=60)]
    request = goal()
    # Use splice programs here so the mature-RNA gate has no extra annotation
    # dependency; exact program identity still binds the two pools.
    for program in request["design_programs"]:
        program["intent"] = "splice_modulation"
    for row in rows:
        row["Sequence"] = row["Synthesis"].translate(str.maketrans("ACGT", "TGCA"))[::-1]
        row["IDT"] = encode_idt_oligo(row["Synthesis"], "steric-hindrance")
        axis, start, end = blocks(row)[0]
        row.update(Chrom=axis, Start=start, End=end, Strand="+", source_modality="steric")
    monkeypatch.setattr(pipeline, "_collect_generated_candidate_rows", lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, "_collect_generated_stage_sets", lambda *a: {})
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda *a: {"structured_goal": request})
    output = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    path = Path(output["generated_aso_set_json"])
    panel = json.loads(path.read_text())
    assert panel["candidate_selection"]["status"] == "complete"
    assert [r["role_assignments"][0]["role_id"] for r in panel["rows"]] == ["First", "Second"]
    assert json.loads(Path(panel["candidate_selection"]["eligible_pool_path"]).read_text())["row_count"] == 4
    exported = list(csv.DictReader((path.parent / "design_panel.csv").open()))
    assert [json.loads(r["role_assignments"])[0]["role_id"] for r in exported] == ["First", "Second"]
    assert "Selection status: complete" in (path.parent / "design_panel.md").read_text()
    assert design_delivery(request, panel, {})["status"] == "fulfilled"
    panel["rows"][1]["role_assignments"] = []
    assert "requested_role_count_not_delivered:Second" in design_delivery(request, panel, {})["errors"]
    panel["candidate_selection"]["status"] = "unresolved"
    assert design_delivery(request, panel, {})["status"] == "unfulfilled"


def test_legacy_prose_count_matcher_cannot_rewrite_typed_role_quotas(monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    request = goal()
    before = deepcopy(request)
    def forbidden(*args, **kwargs):
        raise AssertionError("A prose matcher must not override typed sequential role counts")
    monkeypatch.setattr(pipeline, "_candidate_count_request_from_text", forbidden)
    assert not pipeline._apply_candidate_count_from_request_text_to_goal(
        {"request": "An ordered selection with several counts"}, request, {})
    assert request == before


def test_role_selection_reads_full_job_pools_instead_of_route_preview(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    request = goal()
    request["steric_design_jobs"] = []
    request["steric_tiling_results_by_job"] = {}
    for pid in ("pool_a", "pool_b"):
        path = tmp_path / (pid + ".json")
        rows = [candidate("ACGTAC", pid, chemistry=chem) for chem in ("chem_a", "chem_b")]
        for row in rows:
            row.update(Sequence="GTACGT", Chrom="chr4", Start=0, End=6, Strand="+")
        path.write_text(json.dumps({"rows": rows}))
        request["steric_design_jobs"].append({"job_id": pid, "program_id": pid})
        request["steric_tiling_results_by_job"][pid] = {"uniqueness_scored_path": str(path)}
    preview = tmp_path / "route.csv"
    preview.write_text("Sequence,Chrom,Start,End\nAAAAAA,chr4,100,106\n")
    monkeypatch.setattr(pipeline, "_strategy_route_candidate_paths", lambda *a: [preview])
    monkeypatch.setattr(pipeline, "_canonical_gapmer_candidate_path", lambda *a: None)
    collected = pipeline._collect_generated_candidate_rows(str(tmp_path), request)
    assert len(collected) == 4
    assert {r["program_id"] for r in collected} == {"pool_a", "pool_b"}
    assert {r["IDT"] for r in collected} == {"chem_a", "chem_b"}
    assert all(r["Sequence"] == "GTACGT" for r in collected)
