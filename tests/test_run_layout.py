from __future__ import annotations

import json
from pathlib import Path

from aso_agents.run_layout import (
    canonical_run_path,
    resolve_run_path,
    scored_table_output_path,
    write_layout_manifest,
)
from aso_agents.report_explainer import render_report_markdown
from aso_agents import pipeline_tools


def test_canonical_write_paths_use_artifacts_tree(tmp_path: Path) -> None:
    run_dir = tmp_path / "goal_demo"
    run_dir.mkdir()

    assert canonical_run_path(run_dir, "request.txt") == run_dir / "request.txt"
    assert canonical_run_path(run_dir, "report.md") == run_dir / "report.md"
    assert canonical_run_path(run_dir, "goal_intake_report.html") == run_dir / "report.html"
    assert canonical_run_path(run_dir, "goal_intake.json") == run_dir / "artifacts" / "intake" / "goal_intake.json"
    assert canonical_run_path(run_dir, "generated_aso_set.json") == run_dir / "artifacts" / "candidates" / "generated_aso_set.json"
    assert canonical_run_path(run_dir, "gapmer_tiling", "gapmer_uniqueness_ranked.csv") == (
        run_dir / "artifacts" / "candidates" / "gapmer_uniqueness_ranked.csv"
    )
    assert canonical_run_path(run_dir, "gapmer_tiling", "tiles.json") == (
        run_dir / "artifacts" / "design" / "gapmer_tiling" / "tiles.json"
    )
    assert canonical_run_path(run_dir, "steric_jobs", "job_a") == (
        run_dir / "artifacts" / "design" / "steric_jobs" / "job_a"
    )
    assert canonical_run_path(run_dir, "agentic_123") == (
        run_dir / "artifacts" / "controller" / "agentic_123"
    )
    assert canonical_run_path(run_dir, "child_runs", "task_1") == run_dir / "child_runs" / "task_1"


def test_resolve_prefers_new_layout_and_falls_back_to_legacy(tmp_path: Path) -> None:
    run_dir = tmp_path / "goal_legacy"
    run_dir.mkdir()
    (run_dir / "goal_intake.json").write_text("{}", encoding="utf-8")
    (run_dir / "goal_intake_report.html").write_text("<html></html>", encoding="utf-8")
    (run_dir / "generated_aso_set.json").write_text('{"rows": []}', encoding="utf-8")
    (run_dir / "gapmer_tiling").mkdir()
    (run_dir / "gapmer_tiling" / "gapmer_uniqueness_ranked.csv").write_text("Name\n", encoding="utf-8")

    assert resolve_run_path(run_dir, "goal_intake.json") == run_dir / "goal_intake.json"
    assert resolve_run_path(run_dir, "report.html") == run_dir / "goal_intake_report.html"
    assert resolve_run_path(run_dir, "generated_aso_set.json") == run_dir / "generated_aso_set.json"
    assert resolve_run_path(run_dir, "gapmer_tiling", "gapmer_uniqueness_ranked.csv") == (
        run_dir / "gapmer_tiling" / "gapmer_uniqueness_ranked.csv"
    )

    new_intake = canonical_run_path(run_dir, "goal_intake.json")
    new_intake.parent.mkdir(parents=True)
    new_intake.write_text('{"version": 1}', encoding="utf-8")
    assert resolve_run_path(run_dir, "goal_intake.json") == new_intake


def test_save_goal_payload_writes_intake_under_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "goal_write"
    run_dir.mkdir()
    payload = {"request": "design gapmers", "final_payload": {}}
    structured_goal = {"gene_symbol": "MAPT", "species": "human", "intent": "knockdown"}
    final_payload = {"structured_goal": structured_goal, "request_envelope": {"tasks": []}}

    pipeline_tools._save_goal_payload(str(run_dir), payload, structured_goal, final_payload)

    assert not (run_dir / "goal_intake.json").exists()
    assert not (run_dir / "structured_goal.json").exists()
    assert (run_dir / "artifacts" / "intake" / "goal_intake.json").is_file()
    assert (run_dir / "artifacts" / "intake" / "structured_goal.json").is_file()
    assert (run_dir / "artifacts" / "layout.json").is_file()
    loaded = pipeline_tools._load_goal_intake_payload(str(run_dir))
    assert loaded["final_payload"]["structured_goal"]["gene_symbol"] == "MAPT"


def test_save_goal_payload_synchronizes_complete_single_design_goal_in_composite_request(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "goal_composite"
    run_dir.mkdir()
    task_goal = {
        "gene_symbol": "GENE1",
        "species": "human",
        "target_context": "reviewed cell context",
        "design_programs": [
            {
                "program_id": "branch_a",
                "label": "branch A",
                "modality": "steric",
                "intent": "binding",
                "objective": "block the supplied site",
                "desired_delta": "decrease",
                "length": 17,
            }
        ],
        "evaluation_targets": [
            {
                "isoform_label": "user supplied isoform",
                "site_kind": "donor",
                "exon_label": 4,
                "desired_delta": "decrease",
            }
        ],
    }
    request_envelope = {
        "tasks": [
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "context": {"structured_goal": task_goal},
            },
            {
                "task_id": "report",
                "task_type": "build_composite_report",
                "depends_on": ["design"],
                "context": {},
            },
        ]
    }

    pipeline_tools._save_goal_payload(
        str(run_dir),
        {"request": "Use only the supplied design facts.", "request_envelope": request_envelope},
        {"intent": "binding"},
        {"request_envelope": request_envelope},
    )

    saved = json.loads(
        resolve_run_path(run_dir, "structured_goal.json").read_text(encoding="utf-8")
    )
    intake = pipeline_tools._load_goal_intake_payload(str(run_dir))
    saved_task_goal = intake["request_envelope"]["tasks"][0]["context"]["structured_goal"]

    assert saved_task_goal == saved
    assert saved["gene_symbol"] == task_goal["gene_symbol"]
    assert saved["species"] == task_goal["species"]
    assert saved["target_context"] == task_goal["target_context"]
    assert saved["evaluation_targets"] == task_goal["evaluation_targets"]
    assert saved["design_programs"][0]["desired_delta"] == "decrease"
    assert saved["design_programs"][0]["length"] == 17


def test_save_goal_payload_does_not_restore_stale_fields_into_validated_goal(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "validated_goal"
    run_dir.mkdir()
    stale_goal = {
        "intent": "binding",
        "pending_operation": [{"status": "stale"}],
    }
    request_envelope = {
        "tasks": [
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "context": {"structured_goal": stale_goal},
            }
        ]
    }
    authoritative_goal = {
        "intent": "binding",
        "goal_schema_validation": {"ok": True, "missing_fields": []},
    }

    pipeline_tools._save_goal_payload(
        str(run_dir),
        {"request": "Use the validated goal.", "request_envelope": request_envelope},
        authoritative_goal,
        {"request_envelope": request_envelope},
    )

    saved = json.loads(
        resolve_run_path(run_dir, "structured_goal.json").read_text(encoding="utf-8")
    )
    intake = pipeline_tools._load_goal_intake_payload(str(run_dir))
    saved_task_goal = intake["request_envelope"]["tasks"][0]["context"][
        "structured_goal"
    ]

    assert "pending_operation" not in saved
    assert saved_task_goal == saved


def test_quoted_global_panel_count_does_not_regain_stale_per_site_minimum(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "global_panel"
    run_dir.mkdir()
    stale = {"candidate_limit": 6, "candidate_minimum": 4,
             "candidate_limit_per_site": 4, "min_asos_per_site": 4}
    envelope = {"tasks": [{"task_type": "design_new_asos",
                           "context": {"structured_goal": stale.copy()}}]}
    audit = {"status": "accepted", "candidate_count": {
        "scope": "global", "kind": "maximum", "count": 6, "quote": "4–6 distinct"}}
    payload = {"request": "Design a panel of 4–6 distinct candidates.",
               "request_envelope": envelope}
    pipeline_tools._save_goal_payload(str(run_dir), payload, stale.copy(),
                                      {"request_envelope": envelope,
                                       "request_obligation_audit": audit})
    intake = pipeline_tools._load_goal_intake_payload(str(run_dir))
    saved = json.loads(resolve_run_path(run_dir, "structured_goal.json").read_text())
    task = intake["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    for item in (saved, task, intake["final_payload"]):
        assert item.get("candidate_limit_per_site") is None
        assert item.get("min_asos_per_site") is None
    assert saved["candidate_limit"] == 6
    assert saved["candidate_minimum"] == 4


def test_design_report_analysis_uses_candidates_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "goal_chooser"
    candidates = run_dir / "artifacts" / "candidates"
    candidates.mkdir(parents=True)
    scored_path = candidates / "gapmer_scored_uniqueness.json"
    csv_path = candidates / "gapmer_uniqueness_ranked.csv"
    xlsx_path = candidates / "gapmer_scored_uniqueness.xlsx"
    (run_dir / "request.txt").write_text("Design human OTUD4 gapmers.", encoding="utf-8")
    rows = [
        {
            "Name": "candidate_a",
            "Sequence": "ACGTACGTACGTACGTACGT",
            "selection_rank": 1,
            "isoform_uniqueness_score": 1.0,
            "isoform_uniqueness_class": "gene_wide",
            "Off_target_exact_count": 0,
        }
    ]
    scored_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    csv_path.write_text("Name,Sequence\ncandidate_a,ACGTACGTACGTACGTACGT\n", encoding="utf-8")
    xlsx_path.write_text("placeholder", encoding="utf-8")
    (candidates / "generated_aso_set.json").write_text(
        json.dumps(
            {
                "row_count": 1,
                "unique_sequence_count": 1,
                "excluded_row_count": 0,
                "eligibility_filters": [
                    {
                        "name": "cross_species_exact_match",
                        "required": False,
                        "requested_species": ["human"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (candidates / "generated_filter_trace.json").write_text(json.dumps({"row_count": 4}), encoding="utf-8")
    write_layout_manifest(run_dir)

    markdown = render_report_markdown(
        {"summary": {"headline": "Older explainer headline"}},
        {
            "structured_goal": {
                "gene_symbol": "OTUD4",
                "species": "human",
                "intent": "knockdown",
                "target_context": "neurons",
            },
            "gapmer_tiling": {
                "scored_tiles_path": str(scored_path),
                "uniqueness_csv": str(csv_path),
                "row_count": 1,
            },
        },
    )

    assert markdown.startswith("# Run Output Summary")
    assert "artifacts/candidates/gapmer_scored_uniqueness.xlsx" in markdown
    assert "artifacts/candidates/gapmer_uniqueness_ranked.csv" in markdown
    assert "artifacts/candidates/generated_aso_set.json" in markdown
    assert "gapmer_tiling/gapmer_uniqueness_ranked.csv" not in markdown


def test_artifact_index_reads_new_and_legacy_run_layouts(tmp_path: Path) -> None:
    new_run = tmp_path / "goal_new"
    new_run.mkdir()
    (new_run / "report.md").write_text("# New\n", encoding="utf-8")
    candidates = new_run / "artifacts" / "candidates"
    candidates.mkdir(parents=True)
    (candidates / "generated_aso_set.json").write_text(
        json.dumps({"rows": [{"sequence": "AAAACCCCGGGGTTTTAAAA"}]}),
        encoding="utf-8",
    )
    routes = new_run / "artifacts" / "design" / "strategy_routes"
    routes.mkdir(parents=True)
    (routes / "donor_rank_001.route.csv").write_text(
        "primary_label,sequence\ndonor@100,AAAACCCCGGGGTTTTAAAA\n",
        encoding="utf-8",
    )

    new_index = pipeline_tools._build_artifact_context(
        {"request_text": f"Look at {new_run}", "artifact_refs": [{"path": str(new_run)}]}
    )["artifact_index"]
    new_tables = {str(table["rel_path"]): table for table in pipeline_tools._artifact_index_key_tables(new_index)}
    assert new_index["aggregate"]["by_kind"]["generated_aso_set"] == 1
    assert new_tables["artifacts/candidates/generated_aso_set.json"]["row_count"] == 1
    assert new_tables["artifacts/design/strategy_routes/donor_rank_001.route.csv"]["sample_rows"][0]["primary_label"] == "donor@100"

    old_run = tmp_path / "goal_old"
    old_run.mkdir()
    (old_run / "report.md").write_text("# Old\n", encoding="utf-8")
    (old_run / "generated_aso_set.json").write_text(
        json.dumps({"rows": [{"sequence": "TTTTAAAACCCCGGGGTTTT"}]}),
        encoding="utf-8",
    )
    old_index = pipeline_tools._build_artifact_context(
        {"request_text": f"Look at {old_run}", "artifact_refs": [{"path": str(old_run)}]}
    )["artifact_index"]
    assert old_index["aggregate"]["by_kind"]["generated_aso_set"] == 1


def test_uniqueness_annotation_writes_run_level_tables_to_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "goal_write_uniqueness"
    tiling_dir = run_dir / "artifacts" / "design" / "gapmer_tiling"
    tiling_dir.mkdir(parents=True)
    tiles_path = tiling_dir / "tiles.json"
    tiles_path.write_text("{}", encoding="utf-8")
    job_dir = run_dir / "artifacts" / "design" / "gapmer_jobs" / "job_a"
    job_dir.mkdir(parents=True)
    job_tiles = job_dir / "tiles.json"
    job_tiles.write_text("{}", encoding="utf-8")

    run_level = scored_table_output_path(tiles_path, "gapmer_scored_uniqueness.json")
    job_level = scored_table_output_path(job_tiles, "gapmer_scored_uniqueness.json")
    tiles_stay = scored_table_output_path(tiles_path, "tiles.json")

    assert run_level == run_dir / "artifacts" / "candidates" / "gapmer_scored_uniqueness.json"
    assert job_level == job_dir / "gapmer_scored_uniqueness.json"
    assert tiles_stay == tiling_dir / "tiles.json"


def test_design_analysis_ranks_candidates_from_new_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "goal_analysis"
    candidates = run_dir / "artifacts" / "candidates"
    candidates.mkdir(parents=True)
    (candidates / "generated_aso_set.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sequence": "AAAACCCCGGGGTTTTAAAA",
                        "reverse_complement_sequence": "TTTTAAAACCCCGGGGTTTT",
                        "route_label": "donor@100",
                        "route_rank": 1,
                        "chrom": "chr1",
                        "start": "80",
                        "end": "100",
                        "strand": "+",
                        "gc_fraction": 0.4,
                        "repeat_risk_class": "none",
                        "source_annotations": {
                            "primary_direction": "decrease",
                            "primary_delta_deletion": "-0.91",
                            "primary_delta_replacement": "-0.90",
                            "off_target_total_upto_mismatch_count": "0",
                            "off_target_weighted_risk_score": "0",
                            "selected_transcript_exon_labels": "isoform_a:exon2",
                            "tile_score": "800",
                            "no_homopolymer": "True",
                            "no_secondary_structure": "True",
                            "safeguards_status": "pass",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    request_envelope = {
        "artifact_refs": [{"path": str(run_dir)}],
        "request_text": f"Look at {run_dir} and return the top ASOs.",
    }
    artifact_context = pipeline_tools._build_artifact_context(request_envelope)
    response = pipeline_tools._build_generated_aso_artifact_analysis_response(
        artifact_context=artifact_context,
        response_meta={"request_text": "Return the top ASOs."},
        request_envelope=request_envelope,
    )

    assert response is not None
    assert response["analysis_method"] == "deterministic_generated_aso_ranker"
    assert response["top_ranked_asos"][0]["target_site_sequence"] == "AAAACCCCGGGGTTTTAAAA"
    assert artifact_context["artifact_index"]["aggregate"]["by_kind"]["generated_aso_set"] == 1
    key_tables = pipeline_tools._artifact_index_key_tables(artifact_context["artifact_index"])
    rels = {str(table["rel_path"]) for table in key_tables}
    assert "artifacts/candidates/generated_aso_set.json" in rels
