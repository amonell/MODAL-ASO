"""Retry hints follow the rejected union branch without changing validation."""
from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from aso_agents.pipeline_tools import (
    _json_schema_validation_errors, _structured_output_retry_prompt,
)


def schema(union='anyOf', reverse=False, referenced=False):
    item = {'type': 'object', 'properties': {'weight': {'type': 'number'}},
            'required': ['weight'], 'additionalProperties': False}
    def branch(path, value):
        return {'type': 'object', 'properties': {'op': {'const': 'add'},
            'path': {'const': path}, 'value': value}, 'required': ['op', 'path', 'value'],
            'additionalProperties': False}
    branches = [branch('/attachment', {'type': 'object', 'properties': {'filename': {'type': 'string'}}}),
                branch('/weights', {'type': 'array', 'minItems': 1, 'items': item})]
    if reverse:
        branches.reverse()
    root = {'type': 'object', 'properties': {'patches': {'type': 'array', 'items': {union: branches}}}}
    if referenced:
        root['$defs'] = {f'branch{i}': value for i, value in enumerate(branches)}
        root['properties']['patches']['items'][union] = [{'$ref': f'#/$defs/branch{i}'} for i in range(len(branches))]
    return root


def contracts(prompt):
    return json.loads(prompt.split('Contracts at the rejected paths (not a replacement for the full schema):\n', 1)[1]
                      .split('\n\nKeep scientific constraints', 1)[0])


@pytest.mark.parametrize('union', ['anyOf', 'oneOf'])
@pytest.mark.parametrize('reverse', [False, True])
def test_array_hint_uses_the_discriminated_branch_in_either_order(union, reverse):
    contract = schema(union, reverse)
    rejected = {'patches': [{'op': 'add', 'path': '/weights', 'value': {'weight': 0.3}}]}
    before = deepcopy((contract, rejected))
    errors = _json_schema_validation_errors(rejected, contract)
    assert len(errors) == 1 and "not of type 'array'" in errors[0]
    prompt = _structured_output_retry_prompt('Original user evidence', errors, contract, rejected)
    hints = contracts(prompt)
    assert hints == [{'path': '$.patches[0].value', 'type': 'array', 'minItems': 1,
                      'items': {'type': 'object', 'allowed_keys': ['weight'], 'required_keys': ['weight']}}]
    assert 'filename' not in prompt
    assert (contract, rejected) == before
    assert not Draft202012Validator(contract).is_valid(rejected)
    assert 'Original user evidence' in prompt


def test_actual_leaf_contract_resolves_local_schema_references():
    contract = schema(referenced=True)
    rejected = {'patches': [{'op': 'add', 'path': '/weights', 'value': {'weight': 0.3}}]}
    errors = _json_schema_validation_errors(rejected, contract)
    # Reference-backed union filtering may report unrelated branch errors too;
    # locate the actual array error without assuming its position in the list.
    relevant = [error for error in errors if error.startswith('$.patches[0].value:') and "not of type 'array'" in error]
    assert relevant
    hints = contracts(_structured_output_retry_prompt('Evidence', relevant, contract, rejected))
    assert len(hints) == 1 and hints[0]['type'] == 'array'


def test_nested_array_member_reports_actual_item_contract():
    contract = schema()
    rejected = {'patches': [{'op': 'add', 'path': '/weights', 'value': [{'other': 2}]}]}
    errors = _json_schema_validation_errors(rejected, contract)
    hints = contracts(_structured_output_retry_prompt('Evidence', errors, contract, rejected))
    assert hints and all(h['path'] == '$.patches[0].value[0]' for h in hints)
    assert all(h['allowed_keys'] == ['weight'] and h['required_keys'] == ['weight'] for h in hints)
    assert not Draft202012Validator(contract).is_valid(rejected)


def test_without_rejected_instance_ambiguous_union_gets_no_invented_hint():
    prompt = _structured_output_retry_prompt('Evidence', ['$.patches[0].value: invalid'], schema())
    assert contracts(prompt) == []


def test_unknown_discriminator_cannot_advertise_one_unrelated_value_contract():
    contract = schema()
    rejected = {'patches': [{'op': 'add', 'path': '/unknown', 'value': False}]}
    errors = [e for e in _json_schema_validation_errors(rejected, contract) if e.startswith('$.patches[0].value:')]
    assert errors
    assert contracts(_structured_output_retry_prompt('Evidence', errors, contract, rejected)) == []


def test_string_rejected_response_keeps_the_same_branch_hint():
    contract = schema()
    rejected = {'patches': [{'op': 'add', 'path': '/weights', 'value': {'weight': 0.3}}]}
    errors = _json_schema_validation_errors(rejected, contract)
    prompt = _structured_output_retry_prompt('Evidence', errors, contract, json.dumps(rejected))
    assert contracts(prompt)[0]['type'] == 'array'


def test_invalid_json_response_does_not_create_union_contract():
    prompt = _structured_output_retry_prompt('Evidence', ['$.patches[0].value: invalid'], schema(), '{broken')
    assert contracts(prompt) == []


def test_validation_keeps_valid_array_and_root_shape_rules():
    contract = schema()
    valid = {'patches': [{'op': 'add', 'path': '/weights', 'value': [{'weight': 0.3}]}]}
    assert _json_schema_validation_errors(valid, contract) == []
    assert Draft202012Validator(contract).is_valid(valid)
    assert _json_schema_validation_errors([], contract) == ['$: expected a JSON object']


@pytest.mark.parametrize('branch_count', [2, 20])
def test_identical_error_messages_do_not_hide_ambiguous_contracts(branch_count):
    contract = {'type': 'object', 'properties': {'value': {'anyOf': [
        {'type': 'object', 'properties': {f'field_{index}': {'type': 'string'}}}
        for index in range(branch_count)
    ]}}}
    rejected = {'value': False}
    errors = _json_schema_validation_errors(rejected, contract)
    assert errors == ["$.value: False is not of type 'object'"]
    assert contracts(_structured_output_retry_prompt('Evidence', errors, contract, rejected)) == []
