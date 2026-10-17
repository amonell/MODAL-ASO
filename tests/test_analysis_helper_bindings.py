import json

import pytest

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.analysis_helpers import shared_analysis_helpers, validate_shared_helper_bindings


@pytest.mark.parametrize("code", [
    "def compare_antisense_target(a,b): return 0",
    "if True:\n def summarize_required_criteria(criteria): return {'status':'passed'}",
    "class SequenceCoordinates: pass",
    "compare_antisense_target = lambda a,b: 0",
    "for interval_overlap_summary in []: pass",
    "def f(compare_antisense_target): return compare_antisense_target",
    "from unrelated import compare_antisense_target",
    "from unrelated import compare as compare_antisense_target",
    "import unrelated as compare_antisense_target",
    "from aso_agents.mapping_evidence import compare_antisense_target\ndel compare_antisense_target",
])
def test_named_native_helpers_cannot_be_replaced_in_generated_code(code):
    with pytest.raises(ValueError, match="shared analysis helper"):
        validate_shared_helper_bindings(code)


@pytest.mark.parametrize("code", [
    "from aso_agents.mapping_evidence import compare_antisense_target",
    "from aso_agents.mapping_evidence import compare_antisense_target as compare",
    "import aso_agents.mapping_evidence as mapping",
    "from aso_agents.sequence_coordinates import SequenceCoordinates",
    "from aso_agents.analysis_criteria import summarize_required_criteria",
    "from aso_agents.region_overlap import interval_overlap_summary",
    "def custom_comparison(a,b): return len(a)==len(b)",
    "record={'compare_antisense_target': 'method name'}",
    "class UserObject:\n def compare_antisense_target(self): return None",
])
def test_imports_custom_functions_and_unrelated_field_names_are_allowed(code):
    validate_shared_helper_bindings(code)


def test_runtime_rejects_redefinition_then_unequal_length_and_preserves_attempts(tmp_path):
    contracts = []

    def generate(payload):
        contracts.append(payload)
        contract = payload["execution_contract"]
        assert contract["shared_analysis_helpers"] == shared_analysis_helpers()
        if len(contracts) == 1:
            code = "def compare_antisense_target(a,b): return 0\n"
            code += "from pathlib import Path\nPath('should_not_run').write_text('bad')\n"
        else:
            target = "ACG" if len(contracts) == 2 else "ACGT"
            code = "import json\nfrom pathlib import Path\n"
            code += "from aso_agents.mapping_evidence import compare_antisense_target\n"
            code += f"result=compare_antisense_target('ACGT', {target!r})\n"
            code += f"Path({contract['result_json']!r}).write_text(json.dumps(result))\n"
            code += f"Path({contract['summary_markdown']!r}).write_text('Complete equal-length comparison.')\n"
        return {"python_code": code}

    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[], max_attempts=3)
    assert result["ok"], result
    assert len(contracts) == 3
    assert not (tmp_path / "should_not_run").exists()
    assert "shared analysis helper" in contracts[1]["execution_repair"]["error"]
    assert "length" in contracts[2]["execution_repair"]["error"]
    assert result["result"]["mismatch_count"] == 0
    for number in [1, 2, 3]:
        attempt = tmp_path / "execution_attempts" / f"{number:02d}"
        assert (attempt / "generated_code.py").is_file()
        assert (attempt / "execution.json").is_file()


def test_catalog_is_a_copy_and_each_advertised_import_resolves():
    from importlib import import_module
    catalog = shared_analysis_helpers()
    for spec in catalog:
        assert callable(getattr(import_module(spec["module"]), spec["name"]))
    catalog[0]["name"] = "mutated"
    assert shared_analysis_helpers()[0]["name"] != "mutated"
