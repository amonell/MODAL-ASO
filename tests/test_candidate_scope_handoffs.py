"""Native candidate exports retain producers and enforce independently scoped limits."""
from collections import Counter
import hashlib
import json

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.run_layout import resolve_run_path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


def test_job_merge_retains_distinct_producers_and_original_inputs(tmp_path):
    row = {"Sequence": "ACGT", "Synthesis": "ACGT", "selection_rank": 1,
           "program_id": None, "job_id": ""}
    paths = {name: write_json(tmp_path / name / "ranked.json", {"rows": [row]}) for name in ["a", "b"]}
    before = {name: hashlib.sha256(open(path, "rb").read()).hexdigest() for name, path in paths.items()}
    final = {"gapmer_design_jobs": [{"job_id": name, "program_id": "p_" + name} for name in paths],
             "gapmer_tiling_results_by_job": {name: {"json_path": path} for name, path in paths.items()}}
    result = pipeline._merge_design_job_jsons(final_payload=final, modality="gapmer", result_key="json_path",
        output_path=str(tmp_path / "merged.json"), notes="Qualified producers")
    rows = json.loads(open(result["path"]).read())["rows"]
    assert {(r["program_id"], r["job_id"], r["source_modality"]) for r in rows} == {
        ("p_a", "a", "gapmer"), ("p_b", "b", "gapmer")}
    assert {name: hashlib.sha256(open(path, "rb").read()).hexdigest() for name, path in paths.items()} == before
    assert {pipeline._normalized_generated_candidate_row(r)["program_id"] for r in rows} == {"p_a", "p_b"}


@pytest.mark.parametrize("site_limit, total_limit, expected_per_group", [(2, 4, 2), (1, 4, 1), (None, 6, 3)])
def test_actual_export_keeps_site_limit_separate_from_whole_panel_limit(tmp_path, site_limit, total_limit, expected_per_group):
    run = tmp_path / "run"
    goal = {"candidate_limit": total_limit, "min_asos_per_site": 1,
            "design_programs": [{"program_id": "a", "modality": "gapmer"},
                                {"program_id": "b", "modality": "gapmer"}]}
    if site_limit is not None:
        goal["candidate_limit_per_site"] = site_limit
    sequences = ["ACGT", "AACC", "ATGC", "CCAA", "CGTA", "GATC"]
    paths = {}
    for index, name in enumerate(["a", "b"]):
        rows = [{"Sequence": seq, "Synthesis": seq.translate(str.maketrans("ACGT", "TGCA"))[::-1],
                 "Chrom": "chr1", "Start": 10 * index + rank, "End": 10 * index + rank + 3,
                 "selection_rank": rank}
                for rank, seq in enumerate(sequences[index * 3:index * 3 + 3], 1)]
        paths[name] = write_json(run / "jobs" / name / "ranked.json", {"rows": rows})
    final = {"structured_goal": goal,
             "gapmer_design_jobs": [{"job_id": name, "program_id": name} for name in paths],
             "gapmer_tiling_results_by_job": {name: {"scored_tiles_path": path} for name, path in paths.items()}}
    merged = pipeline._merge_design_job_jsons(final_payload=final, modality="gapmer", result_key="scored_tiles_path",
        output_path=str(run / "gapmer_tiling/gapmer_scored_uniqueness.json"), notes="Merged candidates")
    final["gapmer_tiling_results"] = {"scored_tiles_path": merged["path"]}
    write_json(run / "goal_intake.json", {"final_payload": final})
    pipeline._emit_v2_generated_design_artifacts(str(run))
    exported = json.loads(resolve_run_path(run, "generated_aso_set.json").read_text())
    assert Counter(row["program_id"] for row in exported["rows"]) == {"a": expected_per_group, "b": expected_per_group}
    assert exported["candidate_selection"]["candidate_limit"] == site_limit
    assert exported["candidate_selection"]["global_selection"]["candidate_limit"] == total_limit


def test_route_exports_include_completed_gapmer_branch(tmp_path):
    run = tmp_path / "run"
    route = write_json(run / "strategy_routes/route.route.json", {"rows": [
        {"Sequence": "ACGT", "job_id": "steric", "program_id": "s"}]})
    gapmer = write_json(run / "gapmer_tiling/gapmer_scored_uniqueness.json", {"rows": [
        {"Sequence": "TGCA", "job_id": "gapmer", "program_id": "g"}]})
    rows = pipeline._collect_generated_candidate_rows(str(run))
    assert {row["program_id"] for row in rows} == {"s", "g"}
    stages = pipeline._collect_generated_stage_sets(str(run))
    assert set(stages["sources"]["final_output"]) == {route, gapmer}


def test_program_and_site_caps_both_apply_before_total_cap():
    rows = [{"program_id": program, "selection_rank": rank} for program in ["a", "b"] for rank in range(1, 6)]
    selected, metadata = pipeline._apply_generated_candidate_limit(rows, candidate_limit=3, per_site=True,
        group_limits={"a": 1, "b": 4}, global_candidate_limit=3)
    assert Counter(row["program_id"] for row in selected) == {"a": 1, "b": 2}
    assert {g["group_label"]: g["candidate_limit"] for g in metadata["groups"]} == {"a": 1, "b": 3}


def test_merge_rejects_misaligned_producer_records(tmp_path):
    path = write_json(tmp_path / "in.json", {"rows": [{"Sequence": "ACGT"}]})
    result = pipeline._merge_tiles_json_outputs(source_paths=[path], source_annotations=[],
        output_path=str(tmp_path / "out.json"), notes="Invalid source mapping")
    assert result["error"] == "merge_source_annotation_count_mismatch"
    assert not (tmp_path / "out.json").exists()
