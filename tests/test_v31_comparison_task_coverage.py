"""Typed set-comparison intent requires a corresponding graph operation."""

from aso_agents.pipeline_tools import _executor_capability_issues


def _graph(*, classification=True, compare=False, task_type="evaluate_existing_aso_set"):
    tasks = [{"task_id": "evaluate", "task_type": task_type,
              "context": {"gene_symbol": "GENE", "target_context": "mature RNA"}}]
    if compare:
        tasks.append({"task_id": "compare", "task_type": "compare_aso_sets",
                      "context": {"gene_symbol": "GENE"}})
    return {"request_text": "Assess the submitted groups against a transcript.",
            "request_classifications": ["aso_set_comparison"] if classification else [],
            "tasks": tasks,
            "task_dependencies": ([{"upstream_task_id": "evaluate",
                                   "downstream_task_id": "compare"}] if compare else []),
            "final_outputs": [{"kind": "comparison_table_csv", "task_id":
                               "compare" if compare else "evaluate", "path": "/output/groups.csv"}]}


def test_target_relative_set_comparison_cannot_end_at_evaluation_table():
    issues = _executor_capability_issues(_graph())
    assert [issue["category"] for issue in issues] == ["missing_comparison_operation"]
    assert issues[0]["task_ids"] == ["evaluate"]


def test_qualified_typed_set_comparison_still_requires_comparison_operation():
    graph = _graph()
    graph["request_classifications"] = ["candidate_set_comparison"]
    assert [issue["category"] for issue in _executor_capability_issues(graph)] == [
        "missing_comparison_operation"]


def test_unrelated_typed_comparison_does_not_force_aso_set_operation():
    graph = _graph()
    graph["request_classifications"] = ["gene_comparison"]
    assert _executor_capability_issues(graph) == []


def test_explicit_comparison_task_satisfies_typed_operation():
    assert _executor_capability_issues(_graph(compare=True)) == []


def test_comparison_task_must_consume_evaluation_and_deliver_a_result():
    graph = _graph(compare=True)
    graph["task_dependencies"] = []
    graph["final_outputs"][0]["task_id"] = "evaluate"
    issues = _executor_capability_issues(graph)
    assert {issue["category"] for issue in issues} == {
        "missing_comparison_dependency", "comparison_not_delivered"}


def test_comparison_and_report_can_consume_through_intermediate_tasks():
    graph = _graph(compare=True)
    graph["tasks"].insert(1, {"task_id": "analyze", "task_type": "analyze_artifact",
                               "context": {}})
    graph["task_dependencies"] = [
        {"upstream_task_id": "evaluate", "downstream_task_id": "analyze"},
        {"upstream_task_id": "analyze", "downstream_task_id": "compare"},
    ]
    graph["tasks"].append({"task_id": "assemble", "task_type": "build_composite_report",
                           "context": {}})
    graph["tasks"].append({"task_id": "publish", "task_type": "build_composite_report",
                           "context": {}})
    graph["task_dependencies"].extend([
        {"upstream_task_id": "compare", "downstream_task_id": "assemble"},
        {"upstream_task_id": "assemble", "downstream_task_id": "publish"},
    ])
    graph["final_outputs"][0]["task_id"] = "publish"
    assert _executor_capability_issues(graph) == []


def test_normalized_task_local_dependencies_remain_supported():
    graph = _graph(compare=True)
    graph["task_dependencies"] = []
    graph["tasks"][1]["depends_on"] = ["evaluate"]
    assert _executor_capability_issues(graph) == []


def test_plain_evaluation_does_not_gain_an_unrequested_comparison_task():
    assert _executor_capability_issues(_graph(classification=False)) == []


def test_self_contained_group_analysis_is_not_forced_to_native_comparison():
    assert _executor_capability_issues(_graph(task_type="sandbox_analysis")) == []
