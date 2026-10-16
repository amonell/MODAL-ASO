"""Custom reports receive every recorded stage without losing producer scope."""
import hashlib
import json
from pathlib import Path

import pytest
from aso_agents import pipeline_tools as native
from aso_agents.design_export_evidence import design_export_evidence


def write(root, name, value):
    path = native._write_run_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def example(root, modality="steric"):
    rows = [{"id": "a", "Sequence": "AACCGG", "Start": 10, "End": 16, "Off_target_exact_count": 0},
            {"id": "b", "Sequence": "AAAAAA", "Start": 16, "End": 22, "Off_target_exact_count": None},
            {"id": "c", "Sequence": "AACCGG", "Start": 40, "End": 46, "Off_target_exact_count": 2}]
    raw = write(root, f"{modality}_jobs/origin/tiles.json", {"rows": rows})
    filtered = write(root, "scored_tiles.json", {"rows": [rows[0]]})
    trace = write(root, "generated_filter_trace.json", {"rows": [{"sequence": "AACCGG"}, {"sequence": "AAAAAA"}],
        "sources": {"raw_generated": [str(raw)], "scored": [str(filtered)], "expression_annotation": [str(filtered)]}})
    write(root, "structured_goal.json", {"species": "human", "isoform_transcripts": {}, "design_programs": []})
    write(root, "generated_aso_set.json", {"asos": [rows[0]]})
    return raw, filtered, trace


@pytest.mark.parametrize("modality", ["steric", "gapmer"])
def test_inventory_preserves_unfiltered_rows_duplicate_sequences_and_stage_roles(tmp_path, modality):
    raw, filtered, trace = example(tmp_path, modality)
    before = {p: p.read_bytes() for p in (raw, filtered, trace)}
    result = design_export_evidence("design", tmp_path, native._run_path)
    sources = {s["path"]: s for s in result["sources"]}
    assert sources[str(raw)]["row_count"] == 3
    assert sources[str(trace)]["row_count"] == 2  # Trace sequence groups are not physical-site rows.
    assert sources[str(filtered)]["roles"] == ["scored", "expression_annotation"]
    assert sources[str(raw)]["sha256"] == hashlib.sha256(before[raw]).hexdigest()
    assert not result["unavailable_sources"]
    assert all(p.read_bytes() == data for p, data in before.items())


def test_missing_and_cross_producer_sources_are_not_substituted(tmp_path):
    root = tmp_path / "first"; other = tmp_path / "other"; other.mkdir()
    outside = other / "scored.json"; outside.write_text('{"rows":[{"id":"sibling"}]}')
    raw, filtered, trace = example(root)
    link = root / "linked.json"; link.symlink_to(outside)
    payload = json.loads(trace.read_text())
    payload["sources"]["raw_generated"] += [str(root / "missing.json"), str(outside), str(link)]
    trace.write_text(json.dumps(payload))
    result = design_export_evidence("first", root, native._run_path)
    assert [x["reason"] for x in result["unavailable_sources"]] == ["source_missing", "outside_recorded_producer", "outside_recorded_producer"]
    assert not any(x["path"] == str(outside) for x in result["sources"])
    assert next(x for x in result["sources"] if x["path"] == str(raw))["row_count"] == 3


def test_selected_panel_csv_uses_canonical_candidate_directory(tmp_path):
    example(tmp_path)
    panel = native._run_path(tmp_path, "generated_aso_set.json").with_name("design_panel.csv")
    panel.write_text("id,exact,missing\na,0,\n")
    result = design_export_evidence("design", tmp_path, native._run_path)
    selected = next(x for x in result["sources"] if x["roles"] == ["selected_panel"])
    assert selected["path"] == str(panel) and selected["row_count"] == 1


def test_malformed_trace_remains_diagnostic_evidence(tmp_path):
    _, _, trace = example(tmp_path)
    trace.write_text("broken JSON")
    result = design_export_evidence("design", tmp_path, native._run_path)
    assert next(x for x in result["sources"] if x["path"] == str(trace))["table_read_error"] == "JSONDecodeError"
    assert any("resolved_design_contract" in x["roles"] for x in result["sources"])


def test_two_design_producers_remain_separate_and_require_completed_identity(tmp_path):
    tasks = [{"task_id": name, "task_type": "design_new_asos"} for name in ("alpha", "beta")]
    state = {"request_envelope": {"schema_version": "v3", "tasks": tasks}}
    for task in tasks:
        root = tmp_path / "child_runs" / task["task_id"]
        example(root)
        write(tmp_path, f'v3_tasks/{task["task_id"]}/design_task_result.json',
              {"task_id": task["task_id"], "child_status": "run_complete", "child_run_dir": str(root)})
        assert native._native_design_producer(state, str(tmp_path), task["task_id"]) == root
    path = native._run_path(tmp_path, "v3_tasks/beta/design_task_result.json")
    result = json.loads(path.read_text()); result["task_id"] = "alpha"; path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="producer_invalid"):
        native._native_design_producer(state, str(tmp_path), "beta")


def test_custom_writer_receives_complete_sources_and_cannot_replace_them(tmp_path, monkeypatch):
    raw, filtered, trace = example(tmp_path)
    source_bytes = raw.read_bytes()
    destination = tmp_path / "library.json"
    state = {"request": "Export every generated site, retaining screen failures.", "request_envelope": {
        "schema_version": "v3", "tasks": [{"task_id": "design", "task_type": "design_new_asos"}],
        "final_outputs": [{"kind": "custom", "task_id": "design", "path": str(destination)}]}}
    seen = []
    def generate(payload):
        seen.append(payload)
        evidence = payload["declared_inputs"]["completed_run"]["design_evidence"]
        assert evidence[0]["task_id"] == "design"
        assert str(raw) in payload["readonly_output_paths"]
        assert str(filtered) in payload["readonly_output_paths"]
        code = "from pathlib import Path\nimport json\n"
        code += "declared=json.loads(Path('declared_inputs.json').read_text())\n"
        code += "sources=declared['completed_run']['design_evidence'][0]['sources']\n"
        code += "source=next(s for s in sources if 'raw_generated' in s['roles'])\n"
        code += "rows=json.loads(Path(source['path']).read_text())['rows']\nassert len(rows)==source['row_count']\n"
        code += f"Path({str(destination)!r}).write_text(json.dumps({{'rows':rows}}))\n"
        code += "Path('sandbox_result.json').write_text('{}')\nPath('sandbox_summary.md').write_text('All recorded sites exported.')\n"
        return {"python_code": code}
    monkeypatch.setattr(native, "_run_sandbox_analysis_sync", generate)
    result = native._materialize_requested_final_outputs(state, str(tmp_path))
    assert result["ok"] and len(seen) == 1
    rows = json.loads(destination.read_text())["rows"]
    assert len(rows) == 3 and rows[0]["Off_target_exact_count"] == 0 and rows[1]["Off_target_exact_count"] is None
    assert rows[0]["Sequence"] == rows[2]["Sequence"] and rows[0]["Start"] != rows[2]["Start"]
    assert raw.read_bytes() == source_bytes
