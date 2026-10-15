"""Region tools carry independent feature bounds into saved job filters."""
from types import SimpleNamespace

import pytest

import run_goal_intake as native
from aso_agents import pipeline_tools as pipeline
from test_program_feature_boundaries import candidate, filter_pool, reference_goal, region_rule


@pytest.mark.parametrize("modality", ["steric", "gapmer"])
def test_region_tool_preserves_base_predicate_in_job_artifact(tmp_path, monkeypatch, modality):
    goal = reference_goal("+")
    program = {"program_id": "panel", "modality": modality, "length": 20,
               "intent": "binding" if modality == "steric" else "knockdown",
               "target_region": "utr3", "transcript_id": "TXA.3"}
    goal["design_programs"] = [program]
    chrom, start, end, space = (("chr8", 1200, 1300, "genomic") if modality == "steric"
                                else ("TXA.3", 200, 300, "cdna"))
    goal["materialized_region_constraints"] = {"global": [
        region_rule(chrom, start, start + 8, space),
    ]}
    context = {"payload": {}, "structured_goal": goal, "final_payload": {}}
    # Reference geometry and compilation run unchanged. Stage I/O, already
    # completed reference lookup and later chemistry work are outside this test.
    runtime = SimpleNamespace(
        _build_steric_program_region_rows=native._build_steric_program_region_rows,
        _build_gapmer_program_region_rows=native._build_gapmer_program_region_rows,
        _apply_region_constraints_to_rows=native._apply_region_constraints_to_rows,
        _make_steric_bed_rows=lambda *_: [],
        _collect_gapmer_region_rows=lambda *_: ([], {}),
        _select_primary_transcript=lambda *_: ("TXA.3", "primary"),
    )
    monkeypatch.setattr(pipeline, "_goal_runtime_module", lambda: runtime)
    monkeypatch.setattr(pipeline, "_prepare_execution_stage", lambda **_: {
        "status": "ok", "state": {}, "run_dir": str(tmp_path)})
    monkeypatch.setattr(pipeline, "_load_goal_context_from_run_dir", lambda *_: context)
    monkeypatch.setattr(pipeline, "_artifact_markers_for_run", lambda *_: {})
    monkeypatch.setattr(pipeline, "_resolve_intent_steric", lambda *_: True)
    monkeypatch.setattr(pipeline, "_resolve_intent_knockdown", lambda *_: True)
    monkeypatch.setattr(pipeline, "_save_goal_payload", lambda *_: None)
    monkeypatch.setattr(pipeline, "_ensure_program_chemistry_profiles", lambda **_: None)
    monkeypatch.setattr(pipeline, "_promote_atomic_status", lambda **kwargs: kwargs)
    captured = {}
    monkeypatch.setattr(pipeline, "_ensure_design_jobs_metadata", lambda **kwargs: captured.update(kwargs))
    tool = pipeline.tool_build_steric_regions if modality == "steric" else pipeline.tool_build_gapmer_regions
    result = tool(str(tmp_path / "state.json"))
    assert result["status"] == f"{modality}_regions_ready"
    jobs = captured["jobs_override"]
    assert len(jobs) == 1
    job = jobs[0]
    rows = [candidate("outside", chrom, start - 10, start + 10),
            candidate("inside", chrom, start, start + 20)]
    compiled = job["selection_meta"]["region_constraints"]
    assert filter_pool(tmp_path, rows, compiled) == ["inside"]
