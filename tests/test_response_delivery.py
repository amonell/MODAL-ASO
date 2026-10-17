from copy import deepcopy

import pytest

from aso_agents.evaluation_evidence import attach_measurement_references
from aso_agents.response_context import evaluation_response_batches
from aso_agents.response_contracts import response_contract_errors
from aso_agents.response_delivery import normalize_evaluation_delivery, evaluation_delivery_schema


def fixture():
    rows = [{"input_row_id": f"row_{i}", "name": "repeated", "sequence": "ACGT"} for i in range(3)]
    source = {"task_type": "evaluate_existing_aso_set", "evaluation_summary":
              attach_measurement_references({"provided_asos": rows})}
    part = evaluation_response_batches(source, target_context_bytes=1)[1]
    response = {"answer_summary": "Activity is unmeasured.", "markdown": "",
                "evaluation_assessments": [{"input_row_id": r["input_row_id"], "name": r["name"],
                    "mechanistic_plausibility": "Unmeasured for " + r["input_row_id"]} for r in rows]}
    return part, response


@pytest.mark.parametrize("as_map", [False, True])
def test_known_panel_oversupply_preserves_exact_in_scope_assessment(as_map):
    part, response = fixture()
    expected = deepcopy(response["evaluation_assessments"][1])
    if as_map:
        response["evaluation_assessments"] = {r["input_row_id"]: r for r in response["evaluation_assessments"]}
    before = deepcopy(response)
    result = normalize_evaluation_delivery(response, part)
    assert response == before
    assert result["evaluation_assessments"] == [expected]
    assert result["native_delivery_normalization"]["out_of_scope_input_row_ids"] == ["row_0", "row_2"]
    assert not response_contract_errors(result, part)


@pytest.mark.parametrize("fault", ["unknown", "wrong_name", "missing", "duplicate"])
def test_projection_never_hides_a_real_identity_or_coverage_error(fault):
    part, response = fixture()
    if fault == "unknown": response["evaluation_assessments"][0]["input_row_id"] = "unknown"
    if fault == "wrong_name": response["evaluation_assessments"][0]["name"] = "wrong"
    if fault == "missing": response["evaluation_assessments"].pop(1)
    if fault == "duplicate": response["evaluation_assessments"].append(deepcopy(response["evaluation_assessments"][1]))
    result = normalize_evaluation_delivery(response, part)
    assert response_contract_errors(result, part)


def test_map_keys_cannot_reassign_an_assessment_to_another_observation():
    part, response = fixture()
    response["evaluation_assessments"] = {"row_1": response["evaluation_assessments"][0]}
    with pytest.raises(ValueError, match="conflicts"):
        normalize_evaluation_delivery(response, part)


def test_wire_schema_carries_only_the_component_identity():
    part, _ = fixture()
    schema = evaluation_delivery_schema(part)
    items = schema["properties"]["evaluation_assessments"]
    assert items["minItems"] == items["maxItems"] == 1
    assert items["items"]["properties"]["input_row_id"]["enum"] == ["row_1"]


def test_legacy_unique_names_use_existing_identity_resolution_without_crashing():
    rows = [{"input_row_id": "a", "name": "first"}, {"input_row_id": "b", "name": "second"}]
    source = {"task_type": "evaluate_existing_aso_set", "evaluation_summary": {
        "provided_asos": rows[:1], "panel_comparison_overview": rows}}
    assessments = [{"name": "first", "mechanistic_plausibility": "Unmeasured."},
                   {"name": "second", "mechanistic_plausibility": "Unmeasured."}]
    result = normalize_evaluation_delivery({"evaluation_assessments": assessments}, source)
    assert result["evaluation_assessments"] == assessments[:1]
    assert result["native_delivery_normalization"]["out_of_scope_input_row_ids"] == ["b"]
    assert not response_contract_errors(result, source)
