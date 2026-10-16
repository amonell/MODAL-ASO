"""A failed earlier delivery receipt cannot invalidate later completed files."""

import json
import pytest

from aso_agents.pipeline_tools import _materialize_requested_final_outputs, _run_path


@pytest.mark.parametrize("completed_status", ["report_ready", "sandbox_analysis_ready", "completed"])
def test_completed_native_outputs_supersede_failed_delivery_receipt(tmp_path, completed_status):
    csv_path = tmp_path / "comparison.csv"
    markdown_path = tmp_path / "comparison.md"
    csv_path.write_text("id,result\nA,ok\n")
    markdown_path.write_text("# Comparison\n\nEvidence is recorded.\n")
    directory = tmp_path / "requested_artifacts"
    directory.mkdir()
    (directory / "delivery.json").write_text(json.dumps({
        "ok": False,
        "attempted_paths": [str(csv_path), str(markdown_path)],
        "error": "prior_generation_failed",
    }))
    task_root = _run_path(str(tmp_path), "v3_tasks", "analysis")
    task_root.mkdir(parents=True)
    (task_root / "task_status.json").write_text(json.dumps({
        "task_id": "analysis", "status": completed_status,
        "artifacts": {"requested_export_1": str(csv_path),
                      "requested_export_2": str(markdown_path)},
    }))
    state = {"request_envelope": {"schema_version": "v3",
        "tasks": [{"task_id": "analysis", "task_type": "sandbox_analysis"}],
        "final_outputs": [
            {"path": str(csv_path), "kind": "custom", "task_id": "analysis"},
            {"path": str(markdown_path), "kind": "custom", "task_id": "analysis"},
        ]}}
    result = _materialize_requested_final_outputs(state, str(tmp_path))
    assert result["ok"] is True
    assert result["already_present"] is True
    assert csv_path.read_text().startswith("id,result")
