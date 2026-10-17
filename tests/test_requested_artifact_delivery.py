"""Requested filenames are executable delivery obligations, not report prose."""
import json
from copy import deepcopy
from pathlib import Path
import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.requested_artifacts import requested_artifact_errors


def envelope(outputs):
    return {"schema_version": "v3", "request_mode": "composite", "request_summary": "Analyze two supplied panels",
        "request_classifications": ["analyze_artifact"], "tasks": [
            {"task_id": name, "task_type": kind, "label": name, "native": True,
             "can_run_in_parallel": False, "context": {}}
            for name, kind in [("a", "analyze_artifact"), ("b", "analyze_artifact"), ("report", "build_composite_report")]],
        "task_dependencies": [], "final_outputs": outputs, "response_mode": "composite",
        "requires_execution": True, "assumptions": []}


def test_transport_can_bind_exact_paths_to_producers_and_rejects_unknown_producer():
    response = {"status": "ready", "questions": [], "reason": None, "request_envelope": envelope([
        {"kind": "candidate_table", "task_id": "b", "path": "panel.csv", "format": "csv"}])}
    assert not pipeline._request_workbench_contract_errors(response)
    response["request_envelope"]["final_outputs"][0]["task_id"] = "unknown"
    assert "unknown output producer" in " ".join(pipeline._request_workbench_contract_errors(response))


def test_unbound_outputs_belong_to_assembled_request_and_explicit_bindings_survive():
    raw = envelope(["summary.md", {"kind": "candidate_table", "task_id": "b", "path": "panel.csv"}])
    normalized = pipeline._build_request_envelope(request_text="Analyze the supplied panels.",
        request_id="request", structured_goal={}, raw_envelope=raw, finalize_design_contract=False)
    assert [o["task_id"] for o in normalized["final_outputs"]] == ["report", "b"]
    first = pipeline._task_scoped_request_envelope(normalized, normalized["tasks"][0])
    assert pipeline._requested_output_paths("run", first["final_outputs"]) == []
    second = pipeline._task_scoped_request_envelope(normalized, normalized["tasks"][1])
    assert pipeline._requested_output_paths("run", second["final_outputs"]) == [Path("run/panel.csv")]


def test_child_relative_output_is_bound_to_parent_request_directory(tmp_path):
    raw = envelope([{"kind": "table", "task_id": "b", "path": "panel.csv"}])
    state = {"run_dir": str(tmp_path), "canonical_request_envelope": deepcopy(raw)}
    child = pipeline._task_scoped_child_preseed_envelope(state, raw, raw["tasks"][1])
    assert child["final_outputs"][0]["path"] == str(tmp_path / "panel.csv")
    assert raw["final_outputs"][0]["path"] == "panel.csv"


def test_physical_artifact_check_rejects_missing_empty_and_invalid_outputs(tmp_path):
    path = tmp_path / "output.json"
    assert requested_artifact_errors([path]) == [f"requested_output_missing:{path}"]
    path.write_text("")
    assert requested_artifact_errors([path]) == [f"requested_output_empty:{path}"]
    path.write_text("not JSON")
    assert requested_artifact_errors([path]) == [f"requested_output_invalid_json:{path}"]
    path.write_text('{"rows": [], "limitation": "No observed result"}')
    assert requested_artifact_errors([path]) == []


def setup_completed_analysis(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    raw = envelope([{"kind": "result", "task_id": "a", "path": str(run / "requested.json")}])
    raw.update(tasks=[raw["tasks"][0]], request_mode="single_task", response_mode="analysis", workflow_name="analyze_artifact")
    for name in ["report.md", "report.html"]:
        path = pipeline._write_run_path(run, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Completed native analysis")
    pipeline._save_goal_payload(str(run), {"request": "Export the supplied observations.", "request_envelope": raw}, {},
        {"status": "ready", "request_envelope": raw})
    state = {"version": "2", "run_dir": str(run), "request": "Export the supplied observations.",
        "request_envelope": raw, "status": "report_ready", "phase": "report_ready", "flags": {},
        "events": [], "task_statuses": {"a": {"status": "completed"}}, "task_artifacts": {}}
    return run, state


def export_program(payload):
    contract = payload["execution_contract"]
    code = "from pathlib import Path\nimport json\n"
    code += "inputs=json.loads(Path('declared_inputs.json').read_text())\nassert inputs['completed_run']['run_dir']\n"
    code += f"Path({contract['result_json']!r}).write_text('{{}}')\n"
    code += f"Path({contract['summary_markdown']!r}).write_text('Exported the recorded observations.')\n"
    for path in payload["requested_output_paths"]:
        code += f"Path({path!r}).write_text(json.dumps({{'score': 0, 'unknown': None}}))\n"
    return {"python_code": code}


def test_actual_finalizer_requires_and_materializes_requested_files(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    before = pipeline._extract_pipeline_integrity(str(run), pipeline._discover_core_artifacts(str(run)), skip_router=True)
    assert before["ok"] is False and before["blocking_errors"] == [f"requested_output_missing:{run / 'requested.json'}"]
    calls = []
    def generate(payload):
        calls.append(payload)
        return export_program(payload)
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    result = pipeline.finalize_run_state(str(state_path))
    assert result["terminal_status"] == "run_complete", result
    assert len(calls) == 1
    assert json.loads((run / "requested.json").read_text()) == {"score": 0, "unknown": None}
    assert pipeline._load_state(str(state_path))["requested_output_delivery"]["ok"]
    again = pipeline.finalize_run_state(str(state_path))
    assert again["terminal_status"] == "run_complete" and len(calls) == 1


def test_requested_files_materialize_even_with_separate_scientific_blocker(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    original = pipeline._extract_pipeline_integrity

    def with_scientific_blocker(*args, **kwargs):
        result = original(*args, **kwargs)
        result["blocking_errors"].append("requested_target_accessibility_not_measured")
        result["ok"] = False
        return result

    monkeypatch.setattr(pipeline, "_extract_pipeline_integrity", with_scientific_blocker)
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", export_program)
    result = pipeline.finalize_run_state(str(state_path))
    assert (run / "requested.json").is_file()
    assert pipeline._load_state(str(state_path))["requested_output_delivery"]["ok"]
    assert "requested_target_accessibility_not_measured" in result["integrity"]["blocking_errors"]
    assert not any(error.startswith("requested_output_missing:") for error in result["integrity"]["blocking_errors"])


@pytest.mark.parametrize("contents,physical_error", [
    ('{"observations": []}', None),
    ("", "requested_output_empty"),
    ("invalid JSON", "requested_output_invalid_json"),
    (None, "requested_output_missing"),
])
@pytest.mark.parametrize("raises", [False, True])
def test_failed_delivery_reports_current_files_and_retains_content_blocker(
    tmp_path, monkeypatch, contents, physical_error, raises,
):
    run, state = setup_completed_analysis(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    destination = run / "requested.json"

    def failed_delivery(_state, _run):
        if contents is not None:
            destination.write_text(contents)
        receipt = {"ok": False, "error": "Required observation provenance unavailable"}
        receipt_path = pipeline._write_run_path(run, "requested_artifacts") / "delivery.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt))
        if raises:
            raise ValueError(receipt["error"])
        return receipt

    monkeypatch.setattr(pipeline, "_materialize_requested_final_outputs", failed_delivery)
    result = pipeline.finalize_run_state(str(state_path))
    assert result["terminal_status"] == "fatal"
    errors = result["integrity"]["blocking_errors"]
    physical = [error for error in errors if error.startswith((
        "requested_output_missing:", "requested_output_empty:", "requested_output_invalid_json:",
    ))]
    assert physical == ([f"{physical_error}:{destination}"] if physical_error else [])
    assert "requested_output_delivery_failed" in errors
    assert "requested_output_delivery_incomplete" in errors
    saved = pipeline._load_state(str(state_path))
    assert saved["requested_output_delivery"]["ok"] is False
    assert saved["integrity"] == result["integrity"]


def test_failed_delivery_keeps_independent_scientific_blockers(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    original = pipeline._extract_pipeline_integrity

    def with_scientific_blocker(*args, **kwargs):
        result = original(*args, **kwargs)
        result["blocking_errors"].append("required_reference_binding_unresolved")
        result["ok"] = False
        return result

    def failed_delivery(_state, _run):
        (run / "requested.json").write_text('{"unknown": null}')
        return {"ok": False, "error": "Content validation failed"}

    monkeypatch.setattr(pipeline, "_extract_pipeline_integrity", with_scientific_blocker)
    monkeypatch.setattr(pipeline, "_materialize_requested_final_outputs", failed_delivery)
    result = pipeline.finalize_run_state(str(state_path))
    assert result["terminal_status"] == "fatal"
    assert set(result["integrity"]["blocking_errors"]) == {
        "required_reference_binding_unresolved", "requested_output_delivery_failed",
    }


@pytest.mark.parametrize("writer_succeeds", [True, False])
def test_finalizer_checks_custom_content_when_all_requested_paths_already_exist(tmp_path, monkeypatch, writer_succeeds):
    run, state = setup_completed_analysis(tmp_path)
    destination = run / "requested.json"
    destination.write_text('{"generic_export": true}')
    before = pipeline._extract_pipeline_integrity(str(run), pipeline._discover_core_artifacts(str(run)), skip_router=True)
    assert before["ok"] is True
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    calls = []

    def generate(payload):
        calls.append(payload)
        return export_program(payload) if writer_succeeds else {"python_code": "raise RuntimeError('Writer failed')"}

    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    result = pipeline.finalize_run_state(str(state_path))
    assert len(calls) == (1 if writer_succeeds else 3)
    if writer_succeeds:
        assert result["terminal_status"] == "run_complete"
        assert json.loads(destination.read_text()) == {"score": 0, "unknown": None}
        assert pipeline.finalize_run_state(str(state_path))["terminal_status"] == "run_complete"
        assert len(calls) == 1
    else:
        assert result["terminal_status"] == "fatal" and result["integrity"]["ok"] is False
        assert "requested_output_delivery_failed" in result["integrity"]["blocking_errors"]


def test_supplied_panel_schema_reaches_analysis_without_treating_outputs_as_inputs(tmp_path):
    source = tmp_path / "observations.csv"
    source.write_text("candidate_id,measured_hits,other_bin\nA,0,\nB,1,2\n")
    requested = tmp_path / "answer.csv"
    envelope = {
        "provided_aso_files": [{"path": str(source), "file_format": "csv"}],
        "artifact_refs": [{"path": str(requested), "artifact_type": "output_file"}],
    }
    declared = pipeline._build_sandbox_declared_inputs({}, envelope)
    supplied = declared["provided_input_files"]
    assert [item["path"] for item in supplied] == [str(source)]
    assert supplied[0]["table_summary"]["columns"] == ["candidate_id", "measured_hits", "other_bin"]
    assert supplied[0]["table_summary"]["sample_rows"][0]["measured_hits"] == "0"
    assert not any(item["path"] == str(requested) for item in supplied)


def test_requested_export_prompt_uses_bounded_inventory_with_real_source_paths(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    original = pipeline._build_artifact_context

    def oversized_index(request_envelope):
        if request_envelope.get("artifact_refs") == [{"path": str(run)}]:
            source_path = str(run / "candidate_table.csv")
            files = [{"path": str(run / f"intermediate_{i}.json"), "sample": "x" * 10000}
                     for i in range(250)]
            return {"artifact_refs": [str(run)], "entries": [], "artifact_index": {
                "refs": [{"root_path": str(run), "files": files,
                          "tables": [{"path": source_path, "rel_path": "candidate_table.csv",
                                      "kind": "ranked_candidate_table", "table_summary": {
                                          "columns": ["id", "score"], "row_count": 2,
                                          "sample_rows": [{"id": "A", "score": 0}]}}],
                          "key_artifacts": [{"rel_path": "candidate_table.csv", "kind": "ranked_candidate_table"}]}],
                "aggregate": {"file_count_indexed": 250, "file_count_truncated": False}}}
        return original(request_envelope)

    monkeypatch.setattr(pipeline, "_build_artifact_context", oversized_index)
    captured = []
    def generate(payload):
        captured.append(payload["declared_inputs"])
        return export_program(payload)
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    assert pipeline._materialize_requested_final_outputs(state, str(run))["ok"]
    completed = captured[0]["completed_run"]["artifact_context"]
    assert completed["indexed_file_count"] == 250
    assert completed["key_tables"][0]["path"] == str(run / "candidate_table.csv")
    assert len(json.dumps(captured[0])) < 100_000


def test_failed_delivery_cannot_complete_or_reset_its_budget_on_reentry(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    calls = []
    def no_outputs(payload):
        calls.append(payload)
        return {"python_code": "raise ValueError('No data was exported')"}
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", no_outputs)
    result = pipeline.finalize_run_state(str(state_path))
    assert result["terminal_status"] == "fatal"
    assert "requested_output_delivery_failed" in result["integrity"]["blocking_errors"]
    assert len(calls) == 3
    result = pipeline.finalize_run_state(str(state_path))
    assert result["terminal_status"] == "fatal" and len(calls) == 3


@pytest.mark.parametrize("producer_ids", [("a",), ("a", "b")])
def test_generated_writer_reads_same_scoped_catalog_as_model(tmp_path, monkeypatch, producer_ids):
    run, state = setup_completed_analysis(tmp_path)
    catalogs = {}
    outputs = []
    originals = {}
    for index, producer in enumerate(producer_ids):
        source = run / f"native_{producer}.csv"
        source.write_text(f"id,measured,unknown\n{producer},0,\n")
        catalogs[producer] = {"design_panel_csv": str(source)}
        originals[source] = source.read_bytes()
        outputs.append({"kind": "file", "task_id": producer,
                        "path": str(run / f"summary_{producer}.json")})
    state["request_envelope"]["tasks"] = [
        {"task_id": producer, "task_type": "analyze_artifact"}
        for producer in producer_ids]
    state["request_envelope"]["final_outputs"] = outputs
    monkeypatch.setattr(pipeline, "_native_export_catalog",
                        lambda _state, _run, producer: catalogs[producer])
    calls = []

    def generate(payload):
        calls.append(payload)
        # Exercise the same runtime boundary as generated report code: only
        # the declared file supplies paths, not a closure over model context.
        code = "import csv,json\nfrom pathlib import Path\n"
        code += "inputs=json.loads(Path('declared_inputs.json').read_text())\n"
        code += "sources=inputs['native_export_sources']\n"
        code += f"assert sources == {payload['native_export_sources']!r}\n"
        for producer, spec in zip(producer_ids, outputs):
            code += f"rows=list(csv.DictReader(Path(sources[{producer!r}]['design_panel_csv']).open()))\n"
            code += f"Path({spec['path']!r}).write_text(json.dumps(rows))\n"
        code += f"Path({payload['execution_contract']['result_json']!r}).write_text('{{}}')\n"
        code += f"Path({payload['execution_contract']['summary_markdown']!r}).write_text('Exported source observations.')\n"
        return {"python_code": code}

    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    receipt = pipeline._materialize_requested_final_outputs(state, str(run))
    assert receipt["ok"], receipt
    assert len(calls) == 1
    assert calls[0]["declared_inputs"]["native_export_sources"] == catalogs
    for producer, spec in zip(producer_ids, outputs):
        assert json.loads(Path(spec["path"]).read_text()) == [
            {"id": producer, "measured": "0", "unknown": ""}]
    assert all(source.read_bytes() == content for source, content in originals.items())


def test_failed_custom_design_writer_delivers_only_same_producer_native_files(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    panel = run / "native_panel.csv"
    panel.write_text("candidate,antisense_5to3\n1,ACGT\n")
    report = run / "report.md"
    csv_out = run / "requested_panel.csv"
    md_out = run / "requested_report.md"
    state["request_envelope"]["tasks"] = [{"task_id": "a", "task_type": "design_new_asos"}]
    state["request_envelope"]["final_outputs"] = [
        {"kind": "file", "task_id": "a", "path": str(csv_out)},
        {"kind": "file", "task_id": "a", "path": str(md_out)},
    ]
    monkeypatch.setattr(pipeline, "_native_export_catalog", lambda *args: {
        "design_panel_csv": str(panel), "report_md": str(report)})
    calls = []

    def broken_writer(payload):
        calls.append(payload)
        return {"python_code": "raise AssertionError('bad coordinate recomputation')"}

    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", broken_writer)
    delivery = pipeline._materialize_requested_final_outputs(state, str(run))
    assert delivery["ok"] and delivery["native_fallback_after_writer_failure"]
    assert len(calls) == 3 and len(delivery["execution_attempts"]) == 3
    assert csv_out.read_bytes() == panel.read_bytes()
    assert md_out.read_bytes() == report.read_bytes()
    assert len(delivery["native_exports"]) == 2
    assert "scientific review" in delivery["limitation"]


def test_only_missing_outputs_are_generated_and_attested_outputs_are_protected(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    existing = run / "already_delivered.json"
    existing.write_text('{"measured_value": 9}')
    state["request_envelope"]["final_outputs"].append({"kind": "result", "task_id": "a", "path": str(existing)})
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", export_program)
    assert pipeline._materialize_requested_final_outputs(state, str(run))["ok"]
    original = existing.read_bytes()
    (run / "requested.json").unlink()
    calls = []
    def generate(payload):
        calls.append(payload)
        assert payload["requested_output_paths"] == [str(run / "requested.json")]
        assert str(existing) in payload["readonly_output_paths"]
        assert str(run / "report.md") in payload["readonly_output_paths"]
        program = export_program(payload)
        if len(calls) == 1:
            program["python_code"] += f"Path({str(existing)!r}).write_text('changed existing result')\n"
        else:
            assert "previously delivered files" in payload["execution_repair"]["error"]
            assert existing.read_bytes() == original
        return program
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    result = pipeline._materialize_requested_final_outputs(state, str(run))
    assert result["ok"] and len(calls) == 2
    assert existing.read_bytes() == original
    assert result["attempted_paths"] == [str(run / "requested.json")]
    assert result["execution_attempts"][0]["restored_output_paths"] == [str(existing)]


def test_partial_files_from_failed_execution_cannot_bypass_failure_receipt(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    calls = []
    def partial(payload):
        calls.append(payload)
        program = export_program(payload)
        program["python_code"] += "raise ValueError('Failed after writing a partial output')\n"
        return program
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", partial)
    first = pipeline.finalize_run_state(str(state_path))
    assert first["terminal_status"] == "fatal" and len(calls) == 3
    assert (run / "requested.json").is_file()
    second = pipeline.finalize_run_state(str(state_path))
    assert second["terminal_status"] == "fatal" and len(calls) == 3
    assert "requested_output_delivery_incomplete" in second["integrity"]["blocking_errors"]


def test_native_source_binding_survives_schema_normalization_and_rejects_unknown_keys():
    spec = {"kind": "file", "task_id": "a", "path": "panel.csv", "source_artifact": "design_panel_csv"}
    response = {"status": "ready", "questions": [], "reason": None, "request_envelope": envelope([spec])}
    assert not pipeline._request_workbench_contract_errors(response)
    normalized = pipeline._normalize_final_outputs([spec], "b")
    assert normalized[0]["source_artifact"] == "design_panel_csv" and normalized[0]["task_id"] == "a"
    spec["source_artifact"] = "invented_artifact"
    assert pipeline._request_workbench_contract_errors(response)


def test_native_reports_preserve_partial_findings_without_regeneration(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    source = run / "report.md"
    source.write_text("# Native result\n\nStatus: partial. Directional evidence is unavailable.\n")
    destination = run / "requested.md"
    state["request_envelope"]["final_outputs"] = [{"kind": "report", "task_id": "a", "path": str(destination)}]
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", lambda *_: pytest.fail("Native reports must not be regenerated"))
    receipt = pipeline._materialize_requested_final_outputs(state, str(run))
    assert receipt["ok"] and receipt["execution_attempts"] == []
    assert destination.read_bytes() == source.read_bytes()
    assert receipt["native_exports"][0]["source"] == str(source)


def test_native_table_exports_bind_to_recorded_children_without_retyping_values(tmp_path, monkeypatch):
    run = tmp_path / "parent"
    outputs = [{"kind": "file", "task_id": name, "path": str(tmp_path / (name + ".csv")),
                "source_artifact": "design_panel_csv"} for name in ("a", "b")]
    env = envelope(outputs)
    for task in env["tasks"][:2]:
        task["task_type"] = "design_new_asos"
    state = {"run_dir": str(run), "request_envelope": env}
    originals = []
    for index, name in enumerate(("a", "b")):
        child = run / "child_runs" / name
        result = pipeline._write_run_path(run, "v3_tasks", name, "design_task_result.json")
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({"task_id": name, "child_run_dir": str(child), "child_status": "run_complete"}))
        panel = pipeline._write_run_path(child, "generated_aso_set.json").with_name("design_panel.csv")
        panel.parent.mkdir(parents=True, exist_ok=True)
        content = f"id,physical_binding_intervals_bed0,measured,unknown\n000{index},[[100;120]],0,\n"
        panel.write_text(content)
        originals.append(panel.read_bytes())
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", lambda *_: pytest.fail("No model needed to export native observations"))
    receipt = pipeline._materialize_requested_final_outputs(state, str(run))
    assert receipt["ok"] and len(receipt["native_exports"]) == 2
    assert [Path(spec["path"]).read_bytes() for spec in outputs] == originals
    assert [entry["source"] for entry in receipt["native_exports"]] == [
        str(pipeline._write_run_path(run / "child_runs" / name, "generated_aso_set.json").with_name("design_panel.csv"))
        for name in ("a", "b")]


def test_canonical_native_output_kinds_copy_scoped_design_artifacts(tmp_path, monkeypatch):
    run = tmp_path / "parent"
    child = run / "child_runs" / "design"
    output_csv = tmp_path / "requested_panel.csv"
    output_md = tmp_path / "requested_report.md"
    env = {
        "schema_version": "v3",
        "tasks": [{"task_id": "design", "task_type": "design_new_asos"}],
        "final_outputs": [
            {"kind": "design_panel_csv", "task_id": "design", "path": str(output_csv)},
            {"kind": "report_md", "task_id": "design", "path": str(output_md)},
        ],
    }
    result = pipeline._write_run_path(run, "v3_tasks", "design", "design_task_result.json")
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({"task_id": "design", "child_run_dir": str(child),
                                  "child_status": "run_complete"}))
    generated = pipeline._write_run_path(child, "generated_aso_set.json")
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text('{"candidates": []}')
    panel = generated.with_name("design_panel.csv")
    panel.write_text("id,sequence\n001,ACGT\n")
    report = pipeline._write_run_path(child, "report.md")
    report.write_text("# Recorded native result\n")
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync",
                        lambda *_: pytest.fail("Typed native outputs must not invoke an LLM"))
    receipt = pipeline._materialize_requested_final_outputs(
        {"request_envelope": env}, str(run))
    assert receipt["ok"] and receipt["execution_attempts"] == []
    assert output_csv.read_bytes() == panel.read_bytes()
    assert output_md.read_bytes() == report.read_bytes()


def test_composite_report_uses_its_own_report_and_unique_design_child_panel(tmp_path, monkeypatch):
    run = tmp_path / "parent"
    child = run / "child_runs" / "design"
    output_csv = tmp_path / "requested_panel.csv"
    output_md = tmp_path / "requested_report.md"
    env = {
        "schema_version": "v3",
        "tasks": [
            {"task_id": "design", "task_type": "design_new_asos"},
            {"task_id": "report", "task_type": "build_composite_report", "depends_on": ["design"]},
        ],
        "final_outputs": [
            {"kind": "design_panel_csv", "task_id": "report", "path": str(output_csv)},
            {"kind": "report_md", "task_id": "report", "path": str(output_md)},
        ],
    }
    result = pipeline._write_run_path(run, "v3_tasks", "design", "design_task_result.json")
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({"task_id": "design", "child_run_dir": str(child),
                                  "child_status": "run_complete"}))
    generated = pipeline._write_run_path(child, "generated_aso_set.json")
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text('{"candidates": []}')
    panel = generated.with_name("design_panel.csv")
    panel.write_text("id,sequence\n001,ACGT\n")
    report = pipeline._write_run_path(run, "report.md")
    report.write_text("# Composite report\n")
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync",
                        lambda *_: pytest.fail("Native outputs must not invoke an LLM"))
    receipt = pipeline._materialize_requested_final_outputs(
        {"request_envelope": env}, str(run))
    assert receipt["ok"] and receipt["execution_attempts"] == []
    assert output_csv.read_bytes() == panel.read_bytes()
    assert output_md.read_bytes() == report.read_bytes()
    report.unlink()
    assert "report_md" not in pipeline._native_export_catalog(
        {"request_envelope": env}, str(run), "report"
    )


def test_composite_with_two_design_children_does_not_guess_panel(tmp_path, monkeypatch):
    run = tmp_path / "parent"
    output_csv = tmp_path / "requested_panel.csv"
    tasks = [{"task_id": name, "task_type": "design_new_asos"} for name in ("a", "b")]
    tasks.append({"task_id": "report", "task_type": "build_composite_report",
                  "depends_on": ["a", "b"]})
    for name in ("a", "b"):
        child = run / "child_runs" / name
        result = pipeline._write_run_path(run, "v3_tasks", name, "design_task_result.json")
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({"task_id": name, "child_run_dir": str(child),
                                      "child_status": "run_complete"}))
        generated = pipeline._write_run_path(child, "generated_aso_set.json")
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text('{"candidates": []}')
        generated.with_name("design_panel.csv").write_text(f"id,sequence\n{name},ACGT\n")
    env = {"schema_version": "v3", "tasks": tasks,
           "final_outputs": [{"kind": "design_panel_csv", "task_id": "report",
                              "path": str(output_csv)}]}
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync",
                        lambda *_: pytest.fail("Ambiguous native panels must not be guessed"))
    with pytest.raises(ValueError, match="requested_output_native_source_unavailable"):
        pipeline._materialize_requested_final_outputs({"request_envelope": env}, str(run))
    assert not output_csv.exists()


def test_composite_exports_distinct_child_panels_by_unique_declared_gene_identity(tmp_path, monkeypatch):
    run = tmp_path / "parent"
    tasks = [
        {"task_id": "design_alpha", "task_type": "design_new_asos",
         "context": {"gene_symbol": "GENEA"}},
        {"task_id": "design_beta", "task_type": "design_new_asos",
         "context": {"gene_symbol": "GENEB"}},
        {"task_id": "report", "task_type": "build_composite_report",
         "depends_on": ["design_alpha", "design_beta"]},
    ]
    expected = {}
    for task_id, symbol in (("design_alpha", "GENEA"), ("design_beta", "GENEB")):
        child = run / "child_runs" / task_id
        result = pipeline._write_run_path(run, "v3_tasks", task_id, "design_task_result.json")
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({"task_id": task_id, "child_run_dir": str(child),
                                      "child_status": "run_complete"}))
        generated = pipeline._write_run_path(child, "generated_aso_set.json")
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text('{"candidates": []}')
        expected[symbol] = f"id,sequence\n{symbol},ACGT\n"
        generated.with_name("design_panel.csv").write_text(expected[symbol])
    report = pipeline._write_run_path(run, "report.md")
    report.write_text("# Two-gene report\n")
    outputs = [
        {"kind": "design_panel_csv", "source_artifact": "design_panel_csv",
         "task_id": "report", "label": f"{symbol} panel", "path": str(tmp_path / f"{symbol}_panel.csv")}
        for symbol in ("GENEA", "GENEB")
    ]
    outputs.append({"kind": "report_md", "task_id": "report",
                    "path": str(tmp_path / "both_genes.md")})
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync",
                        lambda *_: pytest.fail("Native outputs must not invoke an LLM"))
    receipt = pipeline._materialize_requested_final_outputs(
        {"request_envelope": {"schema_version": "v3", "tasks": tasks,
                              "final_outputs": outputs}}, str(run))
    assert receipt["ok"]
    for symbol in ("GENEA", "GENEB"):
        assert (tmp_path / f"{symbol}_panel.csv").read_text() == expected[symbol]
    assert (tmp_path / "both_genes.md").read_text() == report.read_text()


def test_unavailable_bound_source_cannot_fall_back_to_generated_claims(tmp_path, monkeypatch):
    run, state = setup_completed_analysis(tmp_path)
    state["request_envelope"]["final_outputs"] = [{"kind": "file", "task_id": "a", "path": "missing.csv",
                                                    "source_artifact": "design_panel_csv"}]
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", lambda *_: pytest.fail("No source means no generated substitute"))
    with pytest.raises(ValueError, match="requested_output_native_source_unavailable"):
        pipeline._materialize_requested_final_outputs(state, str(run))
    assert not (run / "missing.csv").exists()


def test_native_exports_do_not_guess_custom_tables_or_override_explicit_transformations():
    from aso_agents.requested_artifacts import native_export_key
    assert native_export_key({"kind": "file"}, Path("design_panel.csv")) is None
    assert native_export_key({"kind": "design_panel_csv"}, Path("panel.csv")) == "design_panel_csv"
    assert native_export_key({"kind": "report_md"}, Path("report.md")) == "report_md"
    assert native_export_key({"kind": "report", "source_artifact": None}, Path("custom.md")) is None
    assert native_export_key({"kind": "candidate_table"}, Path("arbitrary_name.csv")) == "design_panel_csv"
    with pytest.raises(ValueError, match="native_format_mismatch"):
        native_export_key({"source_artifact": "design_panel_csv"}, Path("panel.json"))


def test_conflicting_output_producers_are_rejected_before_writing(tmp_path):
    run, state = setup_completed_analysis(tmp_path)
    destination = run / "both.md"
    state["request_envelope"]["final_outputs"] = [
        {"kind": "report", "task_id": name, "path": str(destination)} for name in ("a", "b")]
    with pytest.raises(ValueError, match="requested_output_ambiguous_producer"):
        pipeline._materialize_requested_final_outputs(state, str(run))
    assert not destination.exists()
