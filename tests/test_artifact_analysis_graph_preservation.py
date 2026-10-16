"""No-design constraints must not erase already planned analysis operations."""

from copy import deepcopy

import pytest

from aso_agents import pipeline_tools


def request_and_plan(tmp_path, task_type="sandbox_analysis"):
    source = tmp_path / "observations.csv"
    source.write_text("id,value\na,3\nb,7\n")
    request = (f"Analyze the saved results in {source}. Do not design new ASOs. "
               "Calculate grouped totals and write the requested workbook.")
    task = {"task_id": "calculate", "task_type": task_type,
            "label": "Analyze saved observations", "native": task_type != "sandbox_analysis",
            "can_run_in_parallel": False,
            "context": {"artifact_refs": [{"path": str(source)}]}}
    plan = {"schema_version": "v3", "request_mode": "single_task", "tasks": [task],
            "task_dependencies": [], "request_classifications": [task_type],
            "final_outputs": [{"kind": "file", "format": "xlsx",
                               "path": str(tmp_path / "totals.xlsx"), "task_id": "calculate"}],
            "response_mode": "analysis", "requires_execution": True}
    return request, plan


@pytest.mark.parametrize("task_type", ["sandbox_analysis", "analyze_artifact",
    "compare_aso_sets", "trace_filter_decisions", "answer_question"])
def test_no_design_preserves_canonical_operation_and_deliverable(tmp_path, task_type):
    request, plan = request_and_plan(tmp_path, task_type)
    original = deepcopy(plan)
    envelope = pipeline_tools._build_request_envelope(
        request_text=request, request_id="saved-results", structured_goal={}, raw_envelope=plan)
    assert [task["task_type"] for task in envelope["tasks"]] == [task_type]
    assert envelope["tasks"][0]["task_id"] == "calculate"
    assert envelope["tasks"][0]["native"] == (task_type != "sandbox_analysis")
    assert envelope["tasks"][0]["context"]["artifact_refs"] == plan["tasks"][0]["context"]["artifact_refs"]
    assert envelope["final_outputs"][0]["path"] == str(tmp_path / "totals.xlsx")
    assert envelope["final_outputs"][0]["task_id"] == "calculate"
    assert plan == original
    # Normalization is repeated during state persistence; that must not undo
    # the reviewed plan on a subsequent load.
    reloaded = pipeline_tools._build_request_envelope(
        request_text=request, request_id="saved-results", structured_goal={}, raw_envelope=envelope)
    assert reloaded["tasks"] == envelope["tasks"]
    assert reloaded["final_outputs"] == envelope["final_outputs"]


def test_no_design_preserves_composite_dependencies_and_output_ownership(tmp_path):
    request, plan = request_and_plan(tmp_path)
    plan["tasks"].append({"task_id": "explain", "task_type": "build_composite_report",
                          "label": "Explain the calculated results", "native": True,
                          "can_run_in_parallel": False, "context": {}})
    plan["task_dependencies"] = [{"upstream_task_id": "calculate", "downstream_task_id": "explain"}]
    plan["final_outputs"].append({"kind": "file", "format": "md", "path": str(tmp_path / "explanation.md"),
                                  "task_id": "explain"})
    envelope = pipeline_tools._build_request_envelope(
        request_text=request, request_id="composite-analysis", structured_goal={}, raw_envelope=plan)
    assert [t["task_id"] for t in envelope["tasks"]] == ["calculate", "explain"]
    assert envelope["task_dependencies"] == plan["task_dependencies"]
    assert [o["task_id"] for o in envelope["final_outputs"]] == ["calculate", "explain"]


def test_no_design_does_not_invent_a_workflow_before_semantic_intake(tmp_path):
    request, _ = request_and_plan(tmp_path)
    state = {"request": request, "request_id": "unparsed", "run_dir": str(tmp_path),
             "status": "initialized", "phase": "initialized"}
    envelope = pipeline_tools._ensure_v2_request_metadata(state)
    assert envelope == {}
    assert state["task_graph"]["tasks"] == []


@pytest.mark.parametrize("tasks", [[], [None], [{"task_type": "unknown_operation"}]])
def test_incomplete_plan_is_left_for_intake_validation(tmp_path, tasks):
    request, plan = request_and_plan(tmp_path)
    plan["tasks"] = tasks
    assert pipeline_tools._deterministic_artifact_analysis_envelope(
        request_text=request, request_id="incomplete", raw_envelope=plan) is None
