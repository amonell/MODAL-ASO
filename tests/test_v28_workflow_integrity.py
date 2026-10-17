"""Integrity requirements follow executed task types, not a controller label."""

import json

from aso_agents import pipeline_tools as pipeline


def _integrity(tmp_path, task_type, *, request_mode="single_task"):
    envelope = {
        "workflow_name": "design_new_asos",  # legacy controller default
        "request_mode": request_mode,
        "response_mode": "report",
        "tasks": [{"task_id": "task_1", "task_type": task_type}],
        "final_outputs": [{"kind": "table", "path": str(tmp_path / "requested.csv")}],
    }
    (tmp_path / "goal_intake.json").write_text(
        json.dumps({"request_envelope": envelope, "final_payload": {}})
    )
    return pipeline._extract_pipeline_integrity(
        str(tmp_path),
        {"exists": {"report_md": True, "report_html": True}},
        skip_router=False,
    )


def test_compare_task_does_not_require_design_goal_or_router(tmp_path):
    (tmp_path / "requested.csv").write_text("candidate_id\nA\n")
    result = _integrity(tmp_path, "compare_aso_sets")
    assert result["ok"] is True
    assert result["blocking_errors"] == []


def test_compare_task_still_requires_declared_export(tmp_path):
    result = _integrity(tmp_path, "compare_aso_sets")
    assert result["ok"] is False
    assert result["blocking_errors"]
    assert "design_goal_missing" not in result["blocking_errors"]


def test_composite_task_does_not_bypass_composite_integrity(tmp_path):
    (tmp_path / "requested.csv").write_text("candidate_id\nA\n")
    result = _integrity(tmp_path, "compare_aso_sets", request_mode="composite")
    assert result["ok"] is False
    assert "composite_report_md_missing" in result["blocking_errors"]


def test_design_task_keeps_design_integrity(tmp_path):
    (tmp_path / "requested.csv").write_text("candidate_id\nA\n")
    result = _integrity(tmp_path, "design_new_asos")
    assert result["ok"] is False
    assert "design_goal_missing" in result["blocking_errors"]


def test_literal_output_paths_must_be_declared_in_task_graph():
    request = (
        "Read /input/pilot.csv and export /output/pilot_audit.csv, "
        "with a matching /output/pilot_report.md."
    )
    required = pipeline._explicit_output_artifact_paths(request)
    assert [str(path) for path in required] == [
        "/output/pilot_audit.csv", "/output/pilot_report.md"
    ]
    envelope = {"tasks": [{"task_id": "audit", "task_type": "sandbox_analysis"}],
                "final_outputs": []}
    issues = pipeline._request_output_graph_issues(request, envelope)
    assert len(issues) == 1 and issues[0]["category"] == "missing_final_output"
    assert "/output/pilot_audit.csv" in issues[0]["reason"]
    envelope["final_outputs"] = [
        {"kind": "table", "path": str(path), "task_id": "audit"}
        for path in required
    ]
    assert pipeline._request_output_graph_issues(request, envelope) == []


def test_integrity_checks_literal_request_path_when_planner_omits_it(tmp_path):
    envelope = {"workflow_name": "design_new_asos", "request_mode": "single_task",
                "response_mode": "report",
                "tasks": [{"task_id": "compare", "task_type": "compare_aso_sets"}],
                "final_outputs": []}
    (tmp_path / "goal_intake.json").write_text(json.dumps({
        "request": "Export /output/missing_comparison.csv.",
        "request_envelope": envelope, "final_payload": {},
    }))
    result = pipeline._extract_pipeline_integrity(
        str(tmp_path),
        {"exists": {"report_md": True, "report_html": True}},
        skip_router=False,
    )
    assert result["ok"] is False
    assert "/output/missing_comparison.csv" in result["requested_output_paths"]
    assert "design_goal_missing" not in result["blocking_errors"]


def test_submitted_set_artifact_retains_source_rows_when_planner_repeats_sequences(tmp_path):
    source = tmp_path / "submitted.csv"
    source.write_text(
        "set_id,candidate_id,molecule_5to3,chemistry\n"
        "A,A1,ACGTACGTACGTACGTACGT,full-PS gapmer\n"
        "A,A2,ACGTACGTACGTACGTACGT,full-PS gapmer\n"
        "B,B1,TGCATGCATGCATGCATGCA,full-PS gapmer\n"
    )
    context = {
        "artifact_refs": [{"path": str(source), "artifact_type": "submitted_sets_csv"}],
        "provided_asos": ["ACGTACGTACGTACGTACGT", "TGCATGCATGCATGCATGCA"],
    }
    sets = pipeline._collect_aso_compare_sets(
        state={}, task={"task_id": "compare", "task_type": "compare_aso_sets"},
        task_context=context, request_envelope=context, evaluated_payload={},
    )
    assert [(item["set_id"], len(item["rows"])) for item in sets] == [("A", 2), ("B", 1)]
    assert [row["name"] for row in sets[0]["rows"]] == ["A1", "A2"]
