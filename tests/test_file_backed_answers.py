"""Run file-backed question and analysis tools against data outside the intake preview."""
import csv
import json
from pathlib import Path
from textwrap import dedent

import pytest

from aso_agents import pipeline_tools as pipeline


ROUTES = {
    "answer_question": (pipeline.tool_answer_question, "answer_question"),
    "analyze_artifact": (pipeline.tool_analyze_artifact, "artifact_analysis"),
    "sandbox_analysis": (pipeline.tool_run_sandbox_analysis, "sandbox_analysis"),
}


def make_task(tmp_path, kind, request, refs, *, context=None, outputs=()):
    task = {"task_id": "inspect", "task_type": kind, "label": request, "native": True,
            "context": {"artifact_refs": refs, **(context or {})}}
    raw = {"schema_version": "v3", "request_mode": "single_task", "request_text": request,
           "tasks": [task], "final_outputs": list(outputs)}
    envelope = pipeline._build_request_envelope(request_text=request, request_id="lookup",
                                                structured_goal={}, raw_envelope=raw)
    state = {"request": request, "run_dir": str(tmp_path), "status": "request_parsed_ready",
             "phase": "request_parsed_ready", "request_envelope": envelope, "clarification": {},
             "pending_questions": [], "clarification_qa_history": []}
    pipeline._ensure_v2_request_metadata(state)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    return path, envelope


@pytest.mark.parametrize("kind", ROUTES)
@pytest.mark.parametrize("last", [0, 913])
def test_lookup_reads_final_row_and_keeps_response_contract(tmp_path, monkeypatch, kind, last):
    table = tmp_path / "wide readings.tsv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["sample", "signal", "notes"])
        for index, signal in enumerate([18, 30, 21, 19, 222, 10, last], 1):
            writer.writerow([index, signal, "observation details " * 350])
    original = table.read_bytes()
    request = "Looking at this spreadsheet, give the count of readings and the last recorded signal."
    state, _ = make_task(tmp_path, kind, request, [{"path": str(table)}],
                         context={"artifact_operation": "read_records"} if kind == "analyze_artifact" else {})

    def generate(payload):
        declared = payload["declared_inputs"]
        assert payload["request_text"] == request
        assert payload["require_tabular_report_views"] == (kind == "sandbox_analysis")
        assert str(table) in [item["path"] for item in declared["artifact_refs"]]
        if kind != "sandbox_analysis":
            assert "observation details" not in json.dumps(declared)
            assert declared["input_tables"][0]["columns"] == ["sample", "signal", "notes"]
        return {"python_code": dedent("""
            import csv
            import json
            from pathlib import Path

            manifest = json.loads(Path("declared_inputs.json").read_text())
            path = Path(manifest["artifact_refs"][0]["path"])
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=chr(9)))
            count, last = len(rows), int(rows[-1]["signal"])
            Path("sandbox_result.json").write_text(json.dumps({"count": count, "last": last}))
            Path("sandbox_summary.md").write_text(f"Readings: {count}. Last signal: {last}.")
        """)}

    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    monkeypatch.setattr(pipeline, "_run_analysis_contract_review_sync", lambda _: {})
    monkeypatch.setattr(pipeline, "_run_workbench_response_sync", lambda *_: pytest.fail("preview-only response"))
    tool, basename = ROUTES[kind]
    result = tool(str(state))
    assert result["status"] == "ok", result
    assert (tmp_path / "report.md").read_text().strip() == f"Readings: 7. Last signal: {last}."
    response = json.loads(next(tmp_path.rglob(basename + ".json")).read_text())
    assert response["sandbox_result"]["count"] == 7
    assert response["sandbox_result"]["last"] == last
    assert json.loads(state.read_text())["task_statuses"]["inspect"]["status"] == "completed"
    assert table.read_bytes() == original


@pytest.mark.parametrize("kind", ROUTES)
def test_required_optional_and_disappearing_files(tmp_path, kind, monkeypatch):
    table = tmp_path / "first.csv"
    table.write_text("value\n0\n")
    missing = tmp_path / "second.csv"
    refs = [{"path": str(table)}, {"path": str(missing)}, {"path": "https://example.org/source"}]
    request = "Compare the readings in both files."
    state, envelope = make_task(tmp_path, kind, request, refs)
    validation = pipeline._request_validation_summary(envelope)
    assert "inspect" in validation["blocked_task_ids"]
    assert str(missing) in str(validation["clarification_queue"])
    missing.write_text("value\n")
    state, envelope = make_task(tmp_path, kind, request, refs)
    assert not pipeline._request_validation_summary(envelope)["blocked_task_ids"]
    scope = pipeline._task_scoped_child_preseed_envelope

    def remove_after_planning(*args):
        scoped = scope(*args)
        if missing.exists():
            missing.unlink()
        return scoped

    monkeypatch.setattr(pipeline, "_task_scoped_child_preseed_envelope", remove_after_planning)
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", lambda *_: pytest.fail("Missing input"))
    result = ROUTES[kind][0](str(state))
    saved = json.loads(state.read_text())
    # Both planning and execution route to clarification when the file disappears.
    assert "inspect" in pipeline._request_validation_summary(saved["request_envelope"])["blocked_task_ids"]
    assert str(missing) in str(result) or str(missing) in str(saved["pending_questions"])
    assert saved["task_statuses"]["inspect"]["status"] != "completed"
    refs[1]["required"] = False
    _, envelope = make_task(tmp_path, kind, "Use the second file if available.", refs)
    assert not pipeline._request_validation_summary(envelope)["blocked_task_ids"]


@pytest.mark.parametrize("retry", [False, True])
def test_clarification_withholds_partial_export_and_restores_prior_delivery(tmp_path, monkeypatch, retry):
    source = tmp_path / "measurement.csv"
    source.write_text("channel_a,channel_b\n1,2\n")
    target = tmp_path / "calculated.csv"
    target.write_text("prior delivery\n")
    state, _ = make_task(tmp_path, "sandbox_analysis", "Export the requested signal.", [{"path": str(source)}],
                         outputs=[{"task_id": "inspect", "kind": "file", "format": "csv", "path": str(target)}])
    question = "Which channel contains the signal?"
    outcome = {"status": "needs_input", "question": question, "reason": "Two signal channels are present."}

    def generate(payload):
        guard = 'raise ValueError("retry calculation")' if retry and "execution_repair" not in payload else ""
        return {"python_code": dedent(f"""
            from pathlib import Path
            Path({str(target)!r}).write_text("partial export")
            {guard}
            Path("sandbox_result.json").write_text({json.dumps({"request_outcome": outcome})!r})
            Path("sandbox_summary.md").write_text({question!r})
        """)}

    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    result = pipeline.tool_run_sandbox_analysis(str(state))
    saved = json.loads(state.read_text())
    assert question in saved["pending_questions"], result
    assert saved["task_statuses"]["inspect"]["status"] != "completed"
    assert target.read_text() == "prior delivery\n"
    receipts = sorted(tmp_path.rglob("execution.json"))
    assert len(receipts) == (2 if retry else 1)
    assert json.loads(receipts[-1].read_text())["status"] == "needs_input"


def test_file_backed_answer_retains_external_evidence(tmp_path, monkeypatch):
    source = tmp_path / "counts.csv"
    source.write_text("value\n0\n5\n")
    request = "Count my records and cite the supplied source for the definition."
    state, _ = make_task(tmp_path, "answer_question", request, [{"path": str(source)}],
                         context={"web_retrieval": {"needed": True, "required": True,
                                                    "queries": ["official definition"]}})
    evidence = {"used": True, "required": True, "success": True,
                "sources": [{"url": "https://example.org/definition", "title": "Recorded definition"}]}
    monkeypatch.setattr(pipeline, "_run_answer_web_retrieval", lambda **_: evidence)
    monkeypatch.setattr(pipeline, "_run_analysis_contract_review_sync", lambda _: {})

    def generate(payload):
        assert payload["declared_inputs"]["retrieval_context"] == evidence
        return {"python_code": dedent("""
            import csv
            import json
            from pathlib import Path
            inputs = json.loads(Path("declared_inputs.json").read_text())
            with Path(inputs["artifact_refs"][0]["path"]).open() as source:
                count = sum(1 for row in csv.DictReader(source))
            url = inputs["retrieval_context"]["sources"][0]["url"]
            Path("sandbox_result.json").write_text(json.dumps({"count": count}))
            Path("sandbox_summary.md").write_text(f"{count} records. [Recorded definition]({url})")
        """)}

    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    result = pipeline.tool_answer_question(str(state))
    assert result["status"] == "ok"
    assert any(key.startswith("retrieval_") for key in result["artifacts"])
    answer = json.loads(next(tmp_path.rglob("answer_question.json")).read_text())
    assert answer["retrieval"] == evidence
    assert answer["sandbox_result"]["count"] == 2


def test_only_a_declared_upstream_output_can_defer_a_missing_input(tmp_path):
    future = tmp_path / "summary.csv"
    tasks = [
        {"task_id": "make", "task_type": "sandbox_analysis", "context": {}},
        {"task_id": "pass", "task_type": "answer_question", "depends_on": ["make"], "context": {}},
        {"task_id": "read", "task_type": "answer_question", "depends_on": ["pass"],
         "context": {"artifact_refs": [{"path": str(future)}]}},
        {"task_id": "independent", "task_type": "answer_question",
         "context": {"artifact_refs": [{"path": str(future)}]}},
    ]
    envelope = {"schema_version": "v3", "request_text": "Calculate then read the result.", "tasks": tasks,
                "task_dependencies": [{"upstream_task_id": "make", "downstream_task_id": "pass"},
                                      {"upstream_task_id": "pass", "downstream_task_id": "read"}],
                "final_outputs": [{"task_id": "make", "path": str(future), "kind": "file", "format": "csv"}]}
    validation = pipeline._request_validation_summary(envelope)["tasks"]
    assert "artifact_refs_unavailable" not in validation["read"]["issues"]
    assert "artifact_refs_unavailable" in validation["independent"]["issues"]
