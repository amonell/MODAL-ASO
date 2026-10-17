"""Scored steric panels retain the job that produced each molecule."""

import json

from aso_agents import pipeline_tools
from aso_agents.candidate_presentation import bind_candidate_program_groups


def test_per_site_selection_uses_recorded_scored_job_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_tools, "_canonical_gapmer_candidate_path", lambda _: None)
    monkeypatch.setattr(pipeline_tools, "_strategy_route_candidate_paths", lambda _: [])
    jobs = []
    by_job = {}
    for number in range(3):
        job_id = f"job_{number}"
        program_id = f"program_{number}"
        path = tmp_path / "artifacts" / "design" / "steric_jobs" / job_id / "steric_scored_uniqueness.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"rows": [
            {"Sequence": sequence, "Chr": "chr7", "Start": 100 + number * 100 + index,
             "End": 121 + number * 100 + index, "Strand": "+", "Times_mapped": 1}
            for index, sequence in enumerate(("ACGTACGTACGTACGTACGTA",
                                              "CGTACGTACGTACGTACGTAC",
                                              "GTACGTACGTACGTACGTACG"))
        ]}))
        jobs.append({"job_id": job_id, "program_id": program_id, "label": f"Window {number}"})
        # The native sandbox may record a source-relative path while its
        # persisted artifact is mounted under this run's output directory.
        by_job[job_id] = {"uniqueness_scored_path": str(tmp_path / "source" / job_id / path.name)}
    goal = {
        "steric_design_jobs": jobs,
        "steric_tiling_results_by_job": by_job,
        "design_programs": [
            {"program_id": job["program_id"], "label": job["label"]} for job in jobs
        ],
    }
    collected = pipeline_tools._collect_generated_candidate_rows(str(tmp_path), structured_goal=goal)
    assert len(collected) == 9
    assert {row["program_id"] for row in collected} == {job["program_id"] for job in jobs}
    assert all(row["job_id"] in by_job and row["Times_mapped"] == 1 for row in collected)

    grouped = bind_candidate_program_groups(collected, goal)
    selected, summary = pipeline_tools._apply_generated_candidate_limit(
        grouped, candidate_limit=2, per_site=True,
        expected_group_labels=[job["label"] for job in jobs],
    )
    assert len(selected) == 6
    assert {row["candidate_group_label"] for row in selected} == {
        job["label"] for job in jobs
    }
    assert all(group["selected_count"] == 2 and group["shortfall_count"] == 0
               for group in summary["groups"])
