import pytest

from aso_agents.analysis_dependencies import unbound_global_names, validate_program_dependencies
from aso_agents.analysis_repair import apply_program_repair, code_digest


@pytest.mark.parametrize("code,missing", [
    ("result = statistics.mean([1, 2, 3])", ["statistics"]),
    ("import math as m\nresult = math.sqrt(4)", ["math"]),
    ("def evaluate(values):\n return helper(values)\nevaluate([1])", ["helper"]),
    ("def evaluate(values):\n return [external(x) for x in values]", ["external"]),
])
def test_missing_dependencies_in_all_lexical_scopes(code, missing):
    assert unbound_global_names(code) == missing
    with pytest.raises(ValueError, match="no import or definition"):
        validate_program_dependencies(code)


@pytest.mark.parametrize("code", [
    "import math as m\nresult = m.sqrt(4)",
    "from statistics import mean\nresult = mean([1,2,3])",
    "def evaluate(values):\n return helper(values)\ndef helper(values):\n return sum(values)\nevaluate([1])",
    "def outer(x):\n def inner(y):\n  return x+y\n return [inner(z) for z in range(3)]",
    "class Box:\n size = 3\n doubled = size*2\n def get(self):\n  return self.size\nBox().get()",
    "def initialize():\n global metric\n metric=1\ninitialize()\nprint(metric)",
    "from pathlib import Path\nprint(Path(__file__).name, __name__)",
    "x: int = 1\nprint(__annotations__)",
    "from __future__ import annotations\ndef identity(x: Later): return x\nclass Later: pass",
])
def test_valid_imports_definitions_closures_and_runtime_globals(code):
    assert unbound_global_names(code) == []
    validate_program_dependencies(code)


@pytest.mark.parametrize("code", [
    "from math import *\nprint(sqrt(4))",
    "exec('created=3')\nprint(created)",
    "globals()['created']=3\nprint(created)",
])
def test_dynamic_namespace_still_requires_execution_validation(code):
    assert unbound_global_names(code) is None
    validate_program_dependencies(code)


def test_repair_cannot_introduce_an_unbound_dependency():
    original = "value = 4\nprint(value)"
    repair = {"base_sha256": code_digest(original),
              "patches": [{"start_line": 1, "end_line": 1,
                           "replacement": "value = sqrt(4)"}],
              "explanation": "Replace the calculation."}
    with pytest.raises(ValueError, match="sqrt"):
        apply_program_repair({}, original, repair)
    repair["patches"][0]["replacement"] = "from math import sqrt\nvalue = sqrt(4)"
    result = apply_program_repair({"assumptions": ["Keep original scope"]}, original, repair)
    assert result["assumptions"] == ["Keep original scope"]
    namespace = {}
    exec(result["python_code"], namespace)
    assert namespace["value"] == 2
