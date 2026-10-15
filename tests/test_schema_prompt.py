from copy import deepcopy
import json

import jsonschema
import pytest

from aso_agents.schema_prompt import compact_schema_for_prompt
from aso_agents import llm_provider


def repeated_schema():
    choices = [f"source_{index:04d}_evidence_span" for index in range(250)]
    return {"type": "object", "additionalProperties": False,
            "required": ["first", "second"],
            "properties": {name: {"type": "string", "enum": choices}
                           for name in ["first", "second", "optional"]}}


def test_repeated_enum_preserves_validation_and_every_value():
    source = repeated_schema()
    original = deepcopy(source)
    compact = compact_schema_for_prompt(source)
    assert source == original
    assert len(json.dumps(compact)) < len(json.dumps(source)) / 2
    original_validator = jsonschema.Draft202012Validator(source)
    compact_validator = jsonschema.Draft202012Validator(compact)
    for value in source["properties"]["first"]["enum"]:
        instance = {"first": value, "second": value}
        assert original_validator.is_valid(instance)
        assert compact_validator.is_valid(instance)
    for instance in [{}, {"first": "invented", "second": "invented"},
                     {"first": 0, "second": None},
                     {"first": "source_0000_evidence_span", "second": "source_0000_evidence_span", "extra": 1}]:
        assert original_validator.is_valid(instance) is False
        assert compact_validator.is_valid(instance) is False


def test_literal_enum_const_and_example_data_are_not_rewritten():
    literal = repeated_schema()["properties"]["first"]
    source = repeated_schema()
    source["properties"]["literal"] = {"const": literal}
    source["examples"] = [literal, literal]
    compact = compact_schema_for_prompt(source)
    assert compact["properties"]["literal"] == {"const": literal}
    assert compact["examples"] == [literal, literal]


@pytest.mark.parametrize("key,value", [
    ("$ref", "#/$defs/item"), ("$id", "https://example.invalid/nested"),
    ("id", "nested"), ("$anchor", "local"), ("$dynamicRef", "#local"),
])
def test_reference_and_resource_scopes_remain_unchanged(key, value):
    source = repeated_schema()
    source["properties"]["nested"] = {key: value}
    assert compact_schema_for_prompt(source) == source


def test_existing_definitions_and_name_collisions_are_preserved():
    source = repeated_schema()
    first = compact_schema_for_prompt(source)
    name = next(iter(first["$defs"]))
    source["$defs"] = {name: {"type": "integer"}}
    compact = compact_schema_for_prompt(source)
    assert compact["$defs"][name] == {"type": "integer"}
    assert len(compact["$defs"]) == 2
    assert compact["properties"]["first"]["$ref"] != "#/$defs/" + name


def test_small_schemas_and_one_off_enums_stay_inline():
    for source in [{"properties": {"a": {"type": "string", "enum": ["yes", "no"]},
                                    "b": {"type": "string", "enum": ["yes", "no"]}}},
                   {"type": "string", "enum": [str(i) for i in range(1000)]}]:
        assert compact_schema_for_prompt(source) == source


def test_nested_union_and_array_validation_is_preserved():
    repeated = repeated_schema()
    source = {"anyOf": [{"type": "array", "items": repeated}, {"type": "null"}]}
    compact = compact_schema_for_prompt(source)
    assert compact != source
    for instance in [None, [], [{"first": "source_0000_evidence_span", "second": "source_0001_evidence_span"}],
                     [{"first": "unknown", "second": "unknown"}], "invalid"]:
        assert jsonschema.Draft202012Validator(source).is_valid(instance) == jsonschema.Draft202012Validator(compact).is_valid(instance)


def test_provider_prompt_shares_values_without_mutating_request_or_hosted_schema():
    source = repeated_schema()
    payload = {
        "model": "gpt-5.2-2025-12-11",
        "input": [{"role": "user", "content": "Return the requested records."}],
        "text": {"format": {"type": "json_schema", "name": "evidence_contract",
                            "strict": True, "schema": source}},
    }
    original = deepcopy(payload)
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter"}, require_credentials=False,
    )
    prepared, _ = llm_provider.prepare_responses_payload(payload, config)
    assert payload == original
    assert prepared["text"]["format"] == {"type": "json_object"}
    assert prepared["input"][1:] == original["input"]
    instruction = prepared["input"][0]["content"][0]["text"]
    prompt_schema = json.loads(instruction.split("Schema name: evidence_contract\n", 1)[1])
    assert len(json.dumps(prompt_schema)) < len(json.dumps(source)) / 2
    for value in source["properties"]["first"]["enum"]:
        assert jsonschema.Draft202012Validator(prompt_schema).is_valid(
            {"first": value, "second": value}
        )
    assert not jsonschema.Draft202012Validator(prompt_schema).is_valid(
        {"first": "invented", "second": "invented"}
    )
    hosted = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_STRUCTURED_OUTPUT_MODE": "json_schema"},
        require_credentials=False,
    )
    hosted_payload, _ = llm_provider.prepare_responses_payload(payload, hosted)
    assert hosted_payload["input"] == original["input"]
    assert "$defs" not in hosted_payload["text"]["format"]["schema"]
    assert hosted_payload["text"]["format"]["schema"]["properties"]["first"]["enum"] == source["properties"]["first"]["enum"]
