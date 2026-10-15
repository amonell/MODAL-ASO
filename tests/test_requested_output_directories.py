"""Directory obligations must produce files without weakening delivery checks."""
from copy import deepcopy
from pathlib import Path
import pytest
from aso_agents import pipeline_tools as pipeline
from aso_agents.requested_artifacts import requested_artifact_errors


def test_directory_exports_keep_formats_and_producers_separate(tmp_path):
    specs = [{"kind": "file", "task_id": task, "path": str(tmp_path), "format": fmt}
             for task, fmt in [("first", "csv"), ("first", "md"), ("second", "csv")]]
    before = deepcopy(specs)
    paths = pipeline._requested_output_paths(str(tmp_path / "run"), specs)
    assert len(set(paths)) == 3
    assert [p.suffix for p in paths] == [".csv", ".md", ".csv"]
    assert all(p.parent == tmp_path for p in paths)
    assert requested_artifact_errors(paths)  # an existing directory is insufficient
    (tmp_path / "unrelated.csv").write_text("unrelated\n")
    assert requested_artifact_errors(paths)
    assert specs == before
    assert paths == [pipeline._requested_output_paths(str(tmp_path / "run"), [s])[0] for s in specs]


def test_explicit_filename_and_extensionless_file_are_not_reinterpreted(tmp_path):
    for name in ["exact_requested.csv", "extensionless"]:
        spec = {"kind": "file", "path": name, "format": "md"}
        assert pipeline._requested_output_paths(str(tmp_path), [spec]) == [tmp_path / name]


def test_future_directory_is_resolved_before_child_scope_changes(tmp_path):
    task = {"task_id": "audit", "task_type": "sandbox_analysis", "context": {}}
    spec = {"kind": "file", "task_id": "audit", "path": "future/", "format": "csv"}
    envelope = {"schema_version": "v3", "tasks": [task], "final_outputs": [spec]}
    state = {"run_dir": str(tmp_path), "canonical_request_envelope": deepcopy(envelope)}
    expected = pipeline._requested_output_paths(str(tmp_path), [spec])[0]
    child = pipeline._task_scoped_child_preseed_envelope(state, envelope, task)
    assert pipeline._requested_output_paths(str(tmp_path / "child"), child["final_outputs"]) == [expected]
    assert expected.parent == tmp_path / "future"
    assert not expected.parent.exists()


def test_directory_native_report_is_copied_without_regenerating_science(tmp_path, monkeypatch):
    run = tmp_path / "run"; run.mkdir()
    dest = tmp_path / "exports"; dest.mkdir()
    report = run / "report.md"; report.write_text("# Findings\n\nMeasured value: 0. Unknown potency.\n")
    spec = {"kind": "report_md", "task_id": "analysis", "path": str(dest)}
    state = {"request_envelope": {"schema_version": "v3", "tasks": [
        {"task_id": "analysis", "task_type": "sandbox_analysis"}], "final_outputs": [spec]}}
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", lambda *_: pytest.fail("No scientific regeneration needed"))
    result = pipeline._materialize_requested_final_outputs(state, str(run))
    exported = pipeline._requested_output_paths(str(run), [spec])[0]
    assert result["ok"] and result["execution_attempts"] == []
    assert exported.read_bytes() == report.read_bytes()
    assert result["paths"] == [str(exported)]
    assert result["native_exports"][0]["source"] == str(report)


def test_evaluation_directory_export_retains_all_supplied_rows(tmp_path):
    spec = {"kind": "file", "task_id": "evaluation", "path": str(tmp_path), "format": "json"}
    payload = {"provided_asos": [{"name": "A", "sequence": "ACGT", "score": None}], "limit": "not measured"}
    result = pipeline._write_requested_evaluation_exports(str(tmp_path / "run"), payload, "report", [spec])
    import json
    assert json.loads(Path(result["requested_export_1"]).read_text()) == payload


def test_directory_needs_format_and_labels_cannot_escape_destination(tmp_path):
    with pytest.raises(ValueError, match="directory_format_required"):
        pipeline._requested_output_paths(str(tmp_path), [{"kind": "file", "path": str(tmp_path)}])
    specs = [{"kind": "file", "task_id": "../parent", "label": label,
              "path": str(tmp_path), "format": "csv"} for label in ["../alpha/beta", ".._alpha_beta"]]
    paths = pipeline._requested_output_paths(str(tmp_path), specs)
    assert len(paths) == 2 and all(p.parent == tmp_path for p in paths)
