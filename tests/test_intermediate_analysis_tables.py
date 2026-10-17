import json
from pathlib import Path

import pytest

from aso_agents.analysis_execution import execute_analysis_program


def table_program(path, *, suffix="xlsx", value=5, append=""):
    setup = "import json\nfrom pathlib import Path\n" + f"p = Path({str(path)!r})\np.parent.mkdir(parents=True, exist_ok=True)\n"
    if suffix == "xlsx":
        setup += (
            "from openpyxl import Workbook\nb = Workbook()\ns = b.active\ns.title = 'Values'\n"
            f"s.append(['id', 'value'])\ns.append(['derived', {value!r}])\nb.save(p)\nb.close()\n"
        )
    else:
        separator = "\t" if suffix == "tsv" else ","
        setup += f"p.write_text({'id' + separator + 'value' + chr(10) + 'derived' + separator + str(value) + chr(10)!r})\n"
    view = {"id": "values", "source_path": str(path)}
    if suffix == "xlsx":
        view["source_sheet"] = "Values"
    return setup + (
        "Path('sandbox_summary.md').write_text('Calculated values\\n{{table:values}}\\n')\n"
        f"Path('sandbox_result.json').write_text(json.dumps({{'report_tables': [{view!r}]}}))\n"
    ) + append


@pytest.mark.parametrize("suffix", ["csv", "tsv", "xlsx"])
def test_intermediate_computed_population_keeps_its_own_rows(tmp_path, suffix):
    native = tmp_path / "native.csv"
    native.write_text("candidate,sequence\nnative-row,ACGT\n")
    previews = []
    def generate(payload):
        root = Path(payload["execution_contract"]["report_table_views"].get(
            "computed_table_directory", tmp_path / "computed_tables"))
        return {"python_code": table_program(root / ("values." + suffix), suffix=suffix)}
    def review(payload):
        previews.append(payload)
        return {"program_sha256": payload["program_sha256"], "checks": [{
            "requirement_quote": "Compute the total.", "obligation": "Compute total", "status": "satisfied",
            "implementation_evidence": "The computed table contains the value 5.",
            "problem": None, "suggested_fix": None}]}
    result = execute_analysis_program(directory=tmp_path,
        payload={"request_text": "Compute the total.", "require_tabular_report_views": True},
        generate=generate, review_program=review, required_outputs=[],
        readonly_outputs=[native], source_tables=[native], max_attempts=1)
    assert result["ok"], result
    assert "| derived | 5 |" in result["summary"] and "native-row" not in result["summary"]
    assert native.read_text() == "candidate,sequence\nnative-row,ACGT\n"
    receipt, = result["result"]["report_table_rendering"]
    assert receipt["row_count"] == 1 and receipt["source_path"] in previews[0]["output_paths"]
    assert Path(receipt["source_path"]).is_relative_to(tmp_path / "computed_tables")
    assert result["attempts"][0]["required_outputs"] == []


def test_computed_workbook_without_upstream_table_is_rendered(tmp_path):
    def generate(payload):
        root = Path(payload["execution_contract"]["report_table_views"].get(
            "computed_table_directory", tmp_path / "computed_tables"))
        return {"python_code": table_program(root / "values.xlsx")}
    result = execute_analysis_program(directory=tmp_path, payload={"require_tabular_report_views": True},
        generate=generate, required_outputs=[], max_attempts=1)
    assert result["ok"] and "| derived | 5 |" in result["summary"]
    assert "{{table:" not in result["summary"]


def test_previous_attempt_table_cannot_be_used_as_fresh_computation(tmp_path):
    first_path = None
    def generate(payload):
        nonlocal first_path
        root = Path(payload["execution_contract"]["report_table_views"].get(
            "computed_table_directory", tmp_path / "computed_tables"))
        if first_path is None:
            first_path = root / "values.xlsx"
            return {"python_code": table_program(first_path, append="raise ValueError('after table write')\n")}
        view = {"id": "values", "source_path": str(first_path), "source_sheet": "Values"}
        return {"python_code": "import json\nfrom pathlib import Path\n"
            + "Path('sandbox_summary.md').write_text('{{table:values}}')\n"
            + f"Path('sandbox_result.json').write_text(json.dumps({{'report_tables': [{view!r}]}}))\n"}
    result = execute_analysis_program(directory=tmp_path, payload={"require_tabular_report_views": True},
        generate=generate, required_outputs=[], max_attempts=2)
    assert not result["ok"]
    assert "computed_table_directory" in result["error"]
    archived = Path(result["attempts"][1]["previous_computed_tables"]["archive"]) / first_path.name
    assert archived.is_file() and not first_path.exists()


def test_repaired_program_reuses_stable_path_and_preserves_prior_table(tmp_path):
    calls = []
    def generate(payload):
        root = Path(payload["execution_contract"]["report_table_views"]["computed_table_directory"])
        calls.append(root)
        return {"python_code": table_program(root / "values.csv", suffix="csv",
            append="raise ValueError('after table write')\n" if len(calls) == 1 else "")}
    result = execute_analysis_program(directory=tmp_path, payload={"require_tabular_report_views": True},
        generate=generate, required_outputs=[], max_attempts=2)
    assert result["ok"] and calls[0] == calls[1]
    archived = Path(result["attempts"][1]["previous_computed_tables"]["archive"]) / "values.csv"
    assert archived.read_bytes() == (calls[1] / "values.csv").read_bytes()


def test_intermediate_symlink_cannot_bind_an_external_table(tmp_path):
    outside = tmp_path / "prior.csv"
    outside.write_text("id,value\nold,99\n")
    def generate(payload):
        root = Path(payload["execution_contract"]["report_table_views"].get(
            "computed_table_directory", tmp_path / "computed_tables"))
        path = root / "values.csv"
        return {"python_code": "import json\nfrom pathlib import Path\n"
            + f"p=Path({str(path)!r})\np.parent.mkdir(parents=True,exist_ok=True)\np.symlink_to({str(outside)!r})\n"
            + "Path('sandbox_summary.md').write_text('{{table:values}}')\n"
            + f"Path('sandbox_result.json').write_text(json.dumps({{'report_tables':[{{'id':'values','source_path':{str(path)!r}}}]}}))\n"}
    result = execute_analysis_program(directory=tmp_path, payload={"require_tabular_report_views": True},
        generate=generate, required_outputs=[], max_attempts=1)
    assert not result["ok"]
    assert outside.read_text() == "id,value\nold,99\n"


def test_computed_table_cannot_replace_missing_requested_deliverable(tmp_path):
    def generate(payload):
        assert not payload["execution_contract"]["report_table_views"].get("computed_table_directory")
        return {"python_code": table_program(tmp_path / "internal.xlsx")}
    requested = tmp_path / "requested.xlsx"
    result = execute_analysis_program(directory=tmp_path, payload={"require_tabular_report_views": True},
        generate=generate, required_outputs=[requested], max_attempts=1)
    assert not result["ok"] and not requested.exists()
    assert "Requested output files were not created" in result["error"]


def test_intermediate_workbook_formula_still_fails_validation(tmp_path):
    def generate(payload):
        root = Path(payload["execution_contract"]["report_table_views"].get(
            "computed_table_directory", tmp_path / "computed_tables"))
        return {"python_code": table_program(root / "values.xlsx", value="=2+3")}
    result = execute_analysis_program(directory=tmp_path, payload={"require_tabular_report_views": True},
        generate=generate, required_outputs=[], max_attempts=1)
    assert not result["ok"] and "formula" in result["error"]
