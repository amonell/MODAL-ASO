import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

from aso_agents import pipeline_tools as native
from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.design_workbooks import (
    render_design_workbook_plans, validate_design_workbooks, write_design_catalog,
)
from test_design_workbooks import context
from test_design_observation_tables import example


def prepared(tmp_path, *, second=False):
    declared, path, payload = context(tmp_path, second=second)
    meta = write_design_catalog(declared, tmp_path / "catalog.json")
    declared["completed_run"]["design_catalog"] = meta
    declared["requested_design_workbooks"] = [{"path": str(path), "task_id": "design"}]
    rows = json.loads(Path(meta["path"]).read_text())["rows"]
    plan = {"design_workbooks": [{"path": str(path), "tables": [{
        "title": "Library", "task_id": "design", "coverage": "all_raw", "extra_columns": ["ID"],
        "rows": [{"binding_id": row["binding_id"], "values": {"ID": f"candidate-{i}"}}
                 for i, row in enumerate(rows)]}],
        "notes": [{"title": "Methods", "rows": [["Scope", "Recorded source evidence"]]}]}]}
    return declared, path, payload, plan, rows


def test_host_renders_source_values_and_predictions_from_a_table_plan(tmp_path):
    declared, path, payload, plan, rows = prepared(tmp_path)
    meta = declared["completed_run"]["design_catalog"]
    assert hashlib.sha256(Path(meta["path"]).read_bytes()).hexdigest() == meta["sha256"]
    assert rows[0]["off_target_exact_count"] == 0
    render_design_workbook_plans(payload, plan, [path])
    validate_design_workbooks(payload, plan, [path])
    book = openpyxl.load_workbook(path, data_only=True)
    values = list(book["Library"].values)
    row = dict(zip(values[0], values[1]))
    assert row["ID"] == "candidate-0" and row["off_target_exact_count"] == 0
    assert row["antisense_5to3"] == rows[0]["antisense_5to3"]
    assert book["Recorded splice evidence"].max_row == 3
    assert plan["design_workbook_rendering"]["owner"] == "platform"


@pytest.mark.parametrize("fault,match", [
    ("unknown_binding", "binding_id is absent"),
    ("wrong_task", "requested producer"),
    ("extra_source_fields", "only binding_id"),
    ("core_override", "separate from source facts"),
    ("unknown_annotation", "declared extra_columns"),
    ("wrong_path", "requested design scope"),
    ("reserved_sheet", "collides"),
])
def test_plan_cannot_replace_source_identity_or_expand_its_output_scope(tmp_path, fault, match):
    declared, path, payload, plan, rows = prepared(tmp_path)
    workbook = plan["design_workbooks"][0]
    table = workbook["tables"][0]
    if fault == "unknown_binding": table["rows"][0]["binding_id"] = "invented"
    elif fault == "wrong_task": table["task_id"] = "unrelated"
    elif fault == "extra_source_fields": table["rows"][0]["target_5to3"] = "A" * 17
    elif fault == "core_override":
        table["extra_columns"] = ["target_5to3"]
        table["rows"][0]["values"] = {"target_5to3": "A" * 17}
    elif fault == "unknown_annotation": table["rows"][0]["values"]["unlisted"] = 0
    elif fault == "wrong_path": workbook["path"] = str(tmp_path / "unrequested.xlsx")
    elif fault == "reserved_sheet": workbook["notes"][0]["title"] = "Recorded splice evidence"
    with pytest.raises(ValueError, match=match):
        render_design_workbook_plans(payload, plan, [path])


def test_claimed_complete_library_still_requires_every_source_binding(tmp_path):
    declared, path, payload, plan, rows = prepared(tmp_path, second=True)
    plan["design_workbooks"][0]["tables"][0]["rows"].pop()
    render_design_workbook_plans(payload, plan, [path])
    with pytest.raises(ValueError, match="complete source stage"):
        validate_design_workbooks(payload, plan, [path])


def test_execution_accepts_a_program_that_produces_only_the_analysis_plan(tmp_path):
    declared, path, payload, plan, rows = prepared(tmp_path)
    code = ("from pathlib import Path\nimport json\n"
            f"Path('sandbox_result.json').write_text(json.dumps({plan!r}))\n"
            "Path('sandbox_summary.md').write_text('Complete source library with recorded evidence.')\n")
    result = execute_analysis_program(directory=tmp_path, payload=payload, required_outputs=[path],
                                      generate=lambda _: {"python_code": code}, max_attempts=1)
    assert result["ok"], result
    saved = json.loads((tmp_path / "sandbox_result.json").read_text())
    assert saved["design_fact_validation"]["rows_checked"] == 1
    assert saved["design_workbook_rendering"]["owner"] == "platform"


def test_each_requested_workbook_is_rendered_and_checked(tmp_path):
    declared, path, payload, plan, rows = prepared(tmp_path)
    other = tmp_path / "second.xlsx"
    declared["requested_design_workbooks"].append({"path": str(other), "task_id": "design"})
    payload["source_bound_design_workbooks"].append(str(other))
    second = copy.deepcopy(plan["design_workbooks"][0]); second["path"] = str(other)
    plan["design_workbooks"].append(second)
    render_design_workbook_plans(payload, plan, [path, other])
    validate_design_workbooks(payload, plan, [path, other])
    assert plan["design_fact_validation"]["rows_checked"] == 2


def test_native_writer_exposes_protected_catalog_and_renders_analysis_only_code(tmp_path, monkeypatch):
    example(tmp_path)
    destination = tmp_path / "library.xlsx"
    state = {"request": "Export all candidates.", "request_envelope": {
        "schema_version": "v3", "tasks": [{"task_id": "design", "task_type": "design_new_asos"}],
        "final_outputs": [{"kind": "custom", "task_id": "design", "path": str(destination)}]}}
    def generate(payload):
        catalog = payload["declared_inputs"]["completed_run"]["design_catalog"]
        assert catalog["path"] in payload["readonly_output_paths"]
        return {"python_code": "from pathlib import Path\nimport json\n"
            "manifest=json.loads(Path('declared_inputs.json').read_text())\n"
            "catalog=json.loads(Path(manifest['completed_run']['design_catalog']['path']).read_text())['rows']\n"
            "output=manifest['requested_design_workbooks'][0]\n"
            "rows=[r for r in catalog if r['task_id']==output['task_id'] and 'raw_generated' in r['stage_roles']]\n"
            "result={'design_workbooks':[{'path':output['path'],'tables':[{'title':'Candidates',"
            "'task_id':output['task_id'],'coverage':'all_raw','extra_columns':[],"
            "'rows':[{'binding_id':r['binding_id'],'values':{}} for r in rows]}],'notes':[]}]}\n"
            "Path('sandbox_result.json').write_text(json.dumps(result))\n"
            "Path('sandbox_summary.md').write_text('All source candidates exported.')\n"}
    monkeypatch.setattr(native, "_run_sandbox_analysis_sync", generate)
    result = native._materialize_requested_final_outputs(state, str(tmp_path))
    assert result["ok"], result
    assert destination.is_file()


def test_only_source_bound_workbooks_use_the_specialized_analysis_coder(monkeypatch):
    names = []
    async def run(agent, **kwargs):
        names.append(agent.name)
        return SimpleNamespace(final_output=json.dumps({"python_code": "pass"}))
    monkeypatch.setattr(native.Runner, "run", run)
    asyncio.run(native._run_sandbox_analysis_agent_once({"source_bound_design_workbooks": ["library.xlsx"]}))
    asyncio.run(native._run_sandbox_analysis_agent_once({}))
    assert names == ["ASO Design Workbook Analyst", "ASO Sandbox Analysis Coder"]
