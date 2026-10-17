import json
from itertools import product

import pytest
import numpy as np

from aso_agents.analysis_criteria import summarize_required_criteria
from aso_agents.analysis_execution import execute_analysis_program


def test_all_observation_combinations_preserve_individual_statuses():
    names = ["lower_bound", "upper_bound", "required_coverage", "reference_verified"]
    for values in product([True, False, None], repeat=len(names)):
        inputs = dict(zip(names, values))
        result = summarize_required_criteria(inputs)
        assert result["criteria"] == inputs
        assert result["criteria"] is not inputs
        assert result["passed_criteria"] == [n for n, v in zip(names, values) if v is True]
        assert result["failed_criteria"] == [n for n, v in zip(names, values) if v is False]
        assert result["unverified_criteria"] == [n for n, v in zip(names, values) if v is None]
        expected = "failed" if False in values else "indeterminate" if None in values else "passed"
        assert result["status"] == expected
        assert inputs == dict(zip(names, values))


def test_empty_specification_does_not_establish_eligibility():
    result = summarize_required_criteria({})
    assert result["status"] == "indeterminate"
    assert result["failed_criteria"] == result["unverified_criteria"] == []


def test_boolean_array_scalars_preserve_truth_and_export_as_json():
    values = {"first": np.bool_(False), "second": np.bool_(True), "third": None}
    result = summarize_required_criteria(values)
    assert result["failed_criteria"] == ["first"]
    assert result["passed_criteria"] == ["second"]
    assert result["unverified_criteria"] == ["third"]
    assert json.loads(json.dumps(result))["criteria"] == {"first": False, "second": True, "third": None}
    assert isinstance(values["first"], np.bool_)


@pytest.mark.parametrize("value", [0, 1, 0.0, float("nan"), "False", "unknown", [], {}])
def test_nonboolean_values_are_not_silently_interpreted(value):
    with pytest.raises(TypeError, match="True, False or None"):
        summarize_required_criteria({"measured": value})


@pytest.mark.parametrize("name", ["", " ", 1, None])
def test_criteria_need_explicit_names(name):
    with pytest.raises(ValueError, match="nonempty"):
        summarize_required_criteria({name: False})


def test_required_criteria_need_a_mapping():
    with pytest.raises(TypeError, match="mapping"):
        summarize_required_criteria([False, True])


def test_required_schema_keeps_omitted_conditions_unknown_even_after_an_observed_failure():
    names = ["coverage", "composition", "cross_comparison"]
    for first in [True, False, None]:
        observations = {"coverage": first, "composition": True}
        result = summarize_required_criteria(observations, required_names=names)
        assert result["criteria"] == {**observations, "cross_comparison": None}
        assert "cross_comparison" in result["unverified_criteria"]
        assert result["status"] == ("failed" if first is False else "indeterminate")
        assert "cross_comparison" not in observations
    assert summarize_required_criteria(dict.fromkeys(names, True), required_names=names)["status"] == "passed"


@pytest.mark.parametrize("names", ["coverage", [""], [1], ["coverage", "coverage"], {"coverage"}, frozenset()])
def test_required_schema_rejects_ambiguous_names(names):
    with pytest.raises((ValueError, TypeError)):
        summarize_required_criteria({}, required_names=names)


def test_required_schema_catches_extra_or_misspelled_conditions_and_preserves_schema_order():
    with pytest.raises(ValueError, match="not in required_names"):
        summarize_required_criteria({"covergae": True}, required_names=["coverage"])
    result = summarize_required_criteria({"last": False, "first": False}, required_names=["first", "last"])
    assert result["failed_criteria"] == ["first", "last"]


def test_actual_analysis_runtime_exports_independent_failure_and_unknown_sets(tmp_path):
    observations = [
        {"first": False, "second": True},
        {"first": True, "second": False},
        {"first": False, "second": False},
        {"first": True, "second": None},
        {"first": False, "second": None},
        {"first": True, "second": True},
    ]
    destination = tmp_path / "decisions.json"
    (tmp_path / "declared_inputs.json").write_text(json.dumps({"observations": observations}))

    def generate(payload):
        contract = payload["execution_contract"]
        code = "import json\nfrom pathlib import Path\n"
        code += "from aso_agents.analysis_criteria import summarize_required_criteria\n"
        code += "inputs=json.loads(Path('declared_inputs.json').read_text())\n"
        code += "rows=[summarize_required_criteria(x) for x in inputs['observations']]\n"
        code += f"Path({str(destination)!r}).write_text(json.dumps(rows))\n"
        code += f"Path({contract['result_json']!r}).write_text(json.dumps({{'rows':rows}}))\n"
        code += f"Path({contract['summary_markdown']!r}).write_text('Recorded independent criterion results.')\n"
        return {"python_code": code}

    result = execute_analysis_program(
        directory=tmp_path,
        payload={"declared_inputs": {"observations": observations}},
        generate=generate, required_outputs=[destination], max_attempts=1,
    )
    assert result["ok"], result
    rows = json.loads(destination.read_text())
    assert [row["failed_criteria"] for row in rows] == [
        ["first"], ["second"], ["first", "second"], [], ["first"], [],
    ]
    assert [row["status"] for row in rows] == [
        "failed", "failed", "failed", "indeterminate", "failed", "passed",
    ]
    assert rows[4]["unverified_criteria"] == ["second"]
