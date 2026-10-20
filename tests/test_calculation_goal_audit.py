from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline


def calculation_graph(tmp_path, with_report):
    supplied = tmp_path / "target.fasta"
    supplied.write_text(">supplied_target\nACGTCAGTAC\n")
    request = "Generate every contiguous substring under the supplied sequence rules."
    tasks = [{
        "task_id": "calculate",
        "task_type": "sandbox_analysis",
        "label": "Compute the supplied sequence transformation",
        "native": False,
        "can_run_in_parallel": False,
        "context": {"artifact_refs": [{"path": str(supplied), "artifact_type": "fasta"}]},
    }]
    dependencies = []
    if with_report:
        tasks.append({
            "task_id": "report",
            "task_type": "build_composite_report",
            "label": "Summarize the completed calculation",
            "native": True,
            "can_run_in_parallel": False,
            "context": {},
        })
        dependencies.append({"upstream_task_id": "calculate", "downstream_task_id": "report"})
    envelope = pipeline._build_request_envelope(
        request_text=request,
        request_id="calculation_control",
        structured_goal=None,
        raw_envelope={
            "schema_version": "v3",
            "request_text": request,
            "request_mode": "multi_task" if with_report else "single_task",
            "request_classifications": ["design_new_asos"],
            "tasks": tasks,
            "task_dependencies": dependencies,
            "final_outputs": [{
                "task_id": "calculate",
                "kind": "file",
                "format": "csv",
                "path": str(tmp_path / "library.csv"),
            }],
            "response_mode": "analysis",
            "requires_execution": True,
        },
        finalize_design_contract=False,
    )
    return request, envelope


@pytest.mark.parametrize("with_report", [False, True])
def test_compiler_checked_calculations_do_not_enter_native_aso_goal_audit(
    tmp_path, monkeypatch, with_report,
):
    request, envelope = calculation_graph(tmp_path, with_report)
    original = deepcopy(envelope)

    def wrong_audit(*args, **kwargs):
        raise AssertionError("A calculation graph has no native ASO goal to synthesize")

    monkeypatch.setattr(pipeline, "_run_goal_audit_sync", wrong_audit)
    result = pipeline._repair_request_envelope_if_needed(
        request_text=request,
        run_dir=str(tmp_path),
        structured_goal={},
        request_envelope=envelope,
        qa_history=[],
        semantic_audit=True,
    )

    assert result["status"] == "ready", result
    assert [t["task_type"] for t in result["request_envelope"]["tasks"]] == [
        t["task_type"] for t in envelope["tasks"]
    ]
    assert all("structured_goal" not in t.get("context", {})
               for t in result["request_envelope"]["tasks"])
    assert result["request_envelope"]["request_text"] == request
    assert result["request_envelope"]["final_outputs"] == envelope["final_outputs"]
    assert envelope == original


def test_native_molecule_tasks_still_enter_semantic_goal_audit(tmp_path, monkeypatch):
    request, envelope = calculation_graph(tmp_path, False)
    envelope["tasks"][0].update(task_type="design_new_asos", native=True)
    envelope["tasks"][0]["context"]["structured_goal"] = {
        "target_context": "supplied computational baseline",
        "design_programs": [{
            "program_id": "native_control",
            "label": "Native design control",
            "modality": "steric",
            "intent": "binding_only",
            "objective": "Evaluate the supplied native design contract",
        }],
    }

    def required_audit(*args, **kwargs):
        raise RuntimeError("native goal audit retained")

    monkeypatch.setattr(pipeline, "_run_goal_audit_sync", required_audit)
    # Isolate the existing semantic-audit boundary from native field repair;
    # native execution/schema controls have their own full regression suite.
    monkeypatch.setattr(pipeline, "_request_envelope_validation_issues", lambda e: [])
    monkeypatch.setattr(pipeline, "_goal_contract_issue_records", lambda *a: [])
    monkeypatch.setattr(pipeline, "_request_validation_summary", lambda e: {"tasks": {}})
    with pytest.raises(RuntimeError, match="native goal audit retained"):
        pipeline._repair_request_envelope_if_needed(
            request_text=request, run_dir=str(tmp_path), structured_goal={},
            request_envelope=envelope, qa_history=[], semantic_audit=True,
        )
