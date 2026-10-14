"""Task-owned evaluation Markdown must bind without borrowing another task's report."""
from __future__ import annotations

import json
from pathlib import Path

from aso_agents import pipeline_tools


def _state() -> dict:
    return {"request_envelope": {"schema_version": "v3", "tasks": [
        {"task_id": "evaluate_a", "task_type": "evaluate_existing_aso_set"},
        {"task_id": "build_report", "task_type": "build_composite_report"},
    ]}}


def test_native_evaluation_markdown_is_self_contained_and_bounded() -> None:
    result = pipeline_tools._native_evaluation_task_markdown({
        "request_summary": "Evaluate the candidate panel.",
        "evaluation_targets": [{"label": "Target A", "transcript_id": "ENSTTEST.2",
                                "exon_label": 4, "exon_identity_resolution": {
                                    "chrom": "chr7", "start": 100, "end": 200,
                                    "coordinate_system": "1based_inclusive"}}],
        "provided_asos": [{"aso_id": "a", "sequence": "ACGT", "mapping_hits": [],
                           "off_target_gene_count": None,
                           "off_target_count_status": "not_screened"}],
        "provided_aso_scoring": {"off_target_count_scope": {
            "metric": "annotated non-target gene count", "assembly": "GRCh38",
            "excluded_gene_id": None, "exclusion_applied": False}},
        "limitations": ["Cellular activity has not been measured."],
    })
    assert "ENSTTEST.2" in result and "chr7:100-200" in result
    assert "| a | ACGT | 0 | not established | not_screened |" in result
    assert "annotated non-target gene count" in result
    assert "Recorded screening gene count" in result
    assert "exclusion applied: False" in result
    assert "Target exclusion was not applied" in result
    assert "not measured activity" in result
    assert "evaluation_evidence.html" not in result


def test_native_export_catalog_binds_only_evaluation_task_own_markdown(tmp_path: Path) -> None:
    state = _state()
    payload = {"provided_asos": [], "request_summary": "No candidates retained."}
    artifacts = pipeline_tools._write_v2_task_result_files(
        run_dir=str(tmp_path), task_id="evaluate_a", basename="provided_aso_evaluation",
        payload=payload, markdown_text=pipeline_tools._native_evaluation_task_markdown(payload),
        request_envelope=state["request_envelope"],
    )
    catalog = pipeline_tools._native_export_catalog(state, str(tmp_path), "evaluate_a")
    assert catalog == {"report_md": artifacts["markdown"]}
    assert pipeline_tools._native_export_catalog(state, str(tmp_path), "build_report") == {}

    manifest = Path(artifacts["manifest_json"])
    altered = json.loads(manifest.read_text())
    altered["artifacts"]["markdown"] = str(tmp_path / "other_task.md")
    (tmp_path / "other_task.md").write_text("another producer")
    manifest.write_text(json.dumps(altered))
    assert pipeline_tools._native_export_catalog(state, str(tmp_path), "evaluate_a") == {}
