"""Requested analysis must consume data available before its own output exists."""

import asyncio
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline


def _graph(path, field="artifact_refs"):
    return {
        "tasks": [
            {"task_id": "source", "task_type": "design_new_asos", "context": {}},
            {"task_id": "calculate", "task_type": "sandbox_analysis", "native": False,
             "context": {field: [{"path": str(path)}]}},
        ],
        "task_dependencies": [
            {"upstream_task_id": "source", "downstream_task_id": "calculate"}],
        "final_outputs": [{"kind": "file", "task_id": "calculate", "path": str(path)}],
    }


@pytest.mark.parametrize("field", ["artifact_refs", "provided_aso_files"])
def test_analysis_cannot_consume_its_own_missing_output(tmp_path, field):
    graph = _graph(tmp_path / "summary.csv", field)
    original = deepcopy(graph)
    issues = pipeline._executor_capability_issues(graph)
    assert len(issues) == 1
    assert issues[0]["category"] == "wrong_dependency"
    assert issues[0]["task_ids"] == ["calculate"]
    assert issues[0]["path"] == f"/tasks/1/context/{field}/0"
    assert graph == original
    assert not (tmp_path / "summary.csv").exists()


def test_existing_input_can_be_updated_in_place(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("item,value\na,2\nb,8\n")
    assert pipeline._executor_capability_issues(_graph(path)) == []
    assert path.read_text() == "item,value\na,2\nb,8\n"


@pytest.mark.parametrize("dependency_style", ["direct", "transitive", "task_local"])
def test_declared_upstream_file_producer_can_supply_analysis(tmp_path, dependency_style):
    graph = _graph(tmp_path / "summary.csv")
    graph["final_outputs"].append({**graph["final_outputs"][0], "task_id": "source"})
    if dependency_style == "transitive":
        graph["tasks"].insert(1, {"task_id": "inspect", "task_type": "analyze_artifact", "context": {}})
        graph["task_dependencies"] = [
            {"upstream_task_id": "source", "downstream_task_id": "inspect"},
            {"upstream_task_id": "inspect", "downstream_task_id": "calculate"},
        ]
    elif dependency_style == "task_local":
        graph["task_dependencies"] = []
        graph["tasks"][1]["depends_on"] = ["source"]
    assert pipeline._executor_capability_issues(graph) == []


def test_unconnected_producer_does_not_make_input_available(tmp_path):
    graph = _graph(tmp_path / "summary.csv")
    graph["final_outputs"].append({**graph["final_outputs"][0], "task_id": "source"})
    graph["task_dependencies"] = []
    assert [item["category"] for item in pipeline._executor_capability_issues(graph)] == ["wrong_dependency"]


def test_path_dot_segments_do_not_hide_self_dependency(tmp_path):
    graph = _graph(tmp_path / "summary.csv")
    graph["tasks"][1]["context"]["artifact_refs"][0]["path"] = str(tmp_path) + "/./summary.csv"
    assert [item["category"] for item in pipeline._executor_capability_issues(graph)] == ["wrong_dependency"]


def test_missing_unrelated_input_is_not_a_self_dependency(tmp_path):
    graph = _graph(tmp_path / "summary.csv")
    graph["tasks"][1]["context"]["artifact_refs"][0]["path"] = str(tmp_path / "provided.csv")
    assert pipeline._executor_capability_issues(graph) == []


def test_relative_paths_are_left_to_runtime_root_resolution():
    assert pipeline._executor_capability_issues(_graph("summary.csv")) == []


def test_future_input_is_revised_before_semantic_graph_review(monkeypatch, tmp_path):
    first = {"status": "ready", "questions": [], "reason": None,
             "request_envelope": _graph(tmp_path / "summary.csv")}
    corrected = deepcopy(first)
    corrected["request_envelope"]["tasks"][1]["context"] = {}
    proposals, reviews = [], []

    async def plan(prompt):
        proposals.append(prompt)
        if len(proposals) == 2:
            assert "wrong_dependency" in prompt
            assert str(tmp_path / "summary.csv") in prompt
        return first if len(proposals) == 1 else corrected

    async def review(prompt):
        reviews.append(prompt)
        return {"status": "valid", "issues": []}

    async def unchanged(_prompt, proposal):
        return proposal

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_agent_once", review)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", unchanged)
    monkeypatch.setattr(pipeline, "_finalize_verified_request_plan", unchanged)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once(
        "Calculate summary statistics using the generated source rows."))
    assert result == corrected
    assert len(proposals) == 2
    assert len(reviews) == 2
    assert first["request_envelope"]["tasks"][1]["context"]["artifact_refs"]
