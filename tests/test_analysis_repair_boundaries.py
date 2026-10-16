"""Line edits must not silently copy executable context across their boundary."""
from copy import deepcopy
import asyncio
import json
from types import SimpleNamespace

import pytest

from aso_agents.analysis_repair import apply_program_repair, code_digest


def _repair(code, start, end, replacement):
    return apply_program_repair({"python_code": code}, code, {
        "base_sha256": code_digest(code), "patches": [
            {"start_line": start, "end_line": end, "replacement": replacement}]})


@pytest.mark.parametrize("code,start,end,replacement", [
    ("rows = []\nfor i in range(3):\n    row = {'value': i}\n    rows.append(row)\n",
     3, 3, "    row = {'value': i, 'label': 'x'}\n    rows.append(row)"),
    ("rows = []\nrow = 1\nrows.append(row)\nlabel = 'old'\n",
     4, 4, "rows.append(row)\nlabel = 'new'"),
    ("rows = []\nrow = 1\nrows.append(\n    row\n)\n",
     2, 2, "row = 2\nrows.append(row)"),
])
def test_copied_neighboring_statement_is_rejected_without_changing_the_plan(code, start, end, replacement):
    plan = {"python_code": code, "assumptions": ["Keep the declared number of observations."]}
    original = deepcopy(plan)
    with pytest.raises(ValueError, match="duplicated.*unchanged.*statement"):
        apply_program_repair(plan, code, {"base_sha256": code_digest(code), "patches": [
            {"start_line": start, "end_line": end, "replacement": replacement}]})
    assert plan == original


def test_intentional_repetition_in_one_explicit_span_is_retained():
    code = "rows = []\nfor i in range(3):\n    row = i\n    rows.append(row)\n"
    result = _repair(code, 3, 4, "    row = i + 1\n    rows.append(row)\n    rows.append(row)")
    scope = {}
    exec(result["python_code"], scope)
    assert scope["rows"] == [1, 1, 2, 2, 3, 3]


def test_existing_duplicate_statements_outside_the_edit_are_preserved():
    code = "rows = []\nrow = 1\nrows.append(row)\nrows.append(row)\n"
    result = _repair(code, 2, 2, "row = 2")
    scope = {}
    exec(result["python_code"], scope)
    assert scope["rows"] == [2, 2]


def test_same_operation_with_different_values_is_not_deduplicated():
    code = "rows = []\nrow = 1\nrows.append(row)\n"
    result = _repair(code, 2, 2, "row = 2\nrows.append(0)")
    scope = {}
    exec(result["python_code"], scope)
    assert scope["rows"] == [0, 2]


def test_identical_statements_in_different_control_flow_branches_are_valid():
    code = "rows = []\nrow = 1\nif row:\n    rows.append(row)\nelse:\n    pass\n"
    result = _repair(code, 6, 6, "    rows.append(row)")
    scope = {}
    exec(result["python_code"], scope)
    assert scope["rows"] == [1]


def test_touching_an_existing_duplicate_does_not_create_a_new_duplication():
    code = "rows = []\nrow = 1\nrows.append(row)\nrows.append(row)\n"
    result = _repair(code, 2, 3, "row = 2\nrows.append(row)")
    scope = {}
    exec(result["python_code"], scope)
    assert scope["rows"] == [2, 2]


def test_syntax_errors_can_still_be_repaired_without_an_ast_baseline():
    code = "value = (1 +\nprint(value)\n"
    result = _repair(code, 1, 1, "value = (1 + 2)")
    assert result["python_code"] == "value = (1 + 2)\nprint(value)"


def test_native_repair_uses_existing_bounded_retry_for_copied_context(monkeypatch):
    from aso_agents import pipeline_tools
    from aso_agents.analysis_repair import plan_digest
    code = "rows = []\nfor i in range(3):\n    row = i\n    rows.append(row)\n"
    plan = {"python_code": code}
    proposal = {"base_sha256": code_digest(code), "base_plan_sha256": plan_digest(plan),
        "patches": [{"start_line": 3, "end_line": 3,
                     "replacement": "    row = i + 1\n    rows.append(row)"}],
        "plan_updates": [], "explanation": "Update the row expression."}
    responses = []

    async def run(_agent, *, input, **kwargs):
        reply = deepcopy(proposal)
        if responses:
            assert "duplicated an unchanged neighboring statement" in input
            reply["patches"][0]["end_line"] = 4
        responses.append(deepcopy(reply))
        return SimpleNamespace(final_output=json.dumps(reply))

    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    result = asyncio.run(pipeline_tools._run_sandbox_analysis_agent_once({
        "execution_repair": {"previous_code": code, "previous_plan": plan}}))
    assert len(responses) == 2
    assert responses[0]["patches"][0]["end_line"] == 3
    scope = {}
    exec(result["python_code"], scope)
    assert scope["rows"] == [1, 2, 3]
