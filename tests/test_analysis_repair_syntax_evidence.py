"""A rejected patch must identify the revised code and its original edit span."""
import asyncio
import json

import pytest

from aso_agents.analysis_repair import apply_program_repair, code_digest, plan_digest


def repair_for(code, patches):
    return {"base_sha256": code_digest(code), "base_plan_sha256": plan_digest({}),
            "patches": patches, "plan_updates": [], "explanation": "Repair the calculation."}


def rejected_evidence(code, patches):
    with pytest.raises(ValueError, match="Repaired program syntax error") as error:
        apply_program_repair({}, code, repair_for(code, patches))
    return json.loads(str(error.value).split("\n", 1)[1])


def test_error_in_replacement_identifies_original_span_after_earlier_insertion():
    code = "values = []\nfor item in [1]:\n    values.append(item)\nprint(values)\n"
    patches = [
        {"start_line": 1, "end_line": 1, "replacement": "values = []\n# preserved setup\n# extra context"},
        {"start_line": 3, "end_line": 3,
         "replacement": "    values.append(item)\n        values.append(item + 1)"},
    ]
    evidence = rejected_evidence(code, patches)
    assert evidence["base_sha256"] == code_digest(code)
    assert evidence["lineno"] == 6
    line = next(row for row in evidence["context"] if row["revised_line"] == 6)
    assert line == {"revised_line": 6, "base_line": None, "patch_number": 2,
                    "base_span": [3, 3], "text": "        values.append(item + 1)",
                    "text_truncated": False}
    last = next(row for row in evidence["context"] if row["text"] == "print(values)")
    assert last["base_line"] == 4 and last["patch_number"] is None
    assert patches[1]["replacement"].startswith("    values.append")


def test_edit_breaking_unchanged_neighbor_reports_its_base_line_without_reindenting():
    code = "values = []\nfor item in [1]:\n    values.append(item)\nprint(values)\n"
    evidence = rejected_evidence(code, [
        {"start_line": 2, "end_line": 2, "replacement": "item = 1"}])
    assert evidence["lineno"] == 3
    line = next(row for row in evidence["context"] if row["revised_line"] == 3)
    assert line["base_line"] == 3 and line["patch_number"] is None
    assert line["text"] == "    values.append(item)"
    assert any(row["patch_number"] == 1 and row["base_span"] == [2, 2]
               for row in evidence["context"])


def test_syntax_feedback_is_bounded_and_marks_long_line_truncation():
    code = "\n".join(f"value_{i} = {i}" for i in range(30))
    evidence = rejected_evidence(code, [
        {"start_line": 15, "end_line": 15, "replacement": "    bad = 1 # " + "x" * 3000}])
    assert len(evidence["context"]) <= 7
    broken = next(row for row in evidence["context"] if row["revised_line"] == 15)
    assert broken["text_truncated"] is True and len(broken["text"]) == 1000
    assert "original" in evidence["repair_instruction"]


def test_invalid_original_program_can_still_receive_an_explicit_valid_syntax_repair():
    code = "value = (\nprint(value)"
    repaired = apply_program_repair({}, code, repair_for(code, [
        {"start_line": 1, "end_line": 1, "replacement": "value = 4"}]))
    assert repaired["python_code"] == "value = 4\nprint(value)"


def test_native_validator_delivers_mapped_error_to_existing_structured_repair(monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    code = "value = 1\nprint(value)"
    bad = repair_for(code, [{"start_line": 1, "end_line": 1,
                             "replacement": "value = 2\n    value += 1"}])
    good = repair_for(code, [{"start_line": 1, "end_line": 1,
                              "replacement": "value = 2\nvalue += 1"}])
    calls = []

    async def run(**kwargs):
        errors = kwargs["contract_validator"](bad)
        evidence = json.loads(errors[0].split("\n", 1)[1])
        assert evidence["lineno"] == 2
        assert evidence["context"][1]["base_span"] == [1, 1]
        assert kwargs["contract_validator"](good) == []
        calls.append(True)
        return good

    monkeypatch.setattr(pipeline, "_run_schema_validated_agent_once", run)
    result = asyncio.run(pipeline._run_sandbox_analysis_agent_once({
        "execution_repair": {"previous_code": code, "previous_plan": {}}}))
    assert calls == [True]
    assert result["python_code"] == "value = 2\nvalue += 1\nprint(value)"
