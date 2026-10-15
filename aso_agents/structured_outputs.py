"""Strict structured-output contracts for model-facing JSON boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict
from aso_agents.requested_artifacts import NATIVE_EXPORT_FORMATS

from aso_agents.goal_contract import (
    CANONICAL_TASK_TYPES,
    canonical_patch_value_schema,
    canonical_structured_goal_schema,
    canonical_task_context_schema,
)


def _nullable_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    candidate = deepcopy(schema)
    schema_type = candidate.get("type")
    if schema_type == "null" or (
        isinstance(schema_type, list) and "null" in schema_type
    ):
        return candidate
    any_of = candidate.get("anyOf")
    if isinstance(any_of, list):
        if any(
            isinstance(branch, dict) and branch.get("type") == "null"
            for branch in any_of
        ):
            return candidate
        return {"anyOf": [*deepcopy(any_of), {"type": "null"}]}
    if schema_type == "object" or schema_type == "array":
        return {"anyOf": [candidate, {"type": "null"}]}
    if "const" in candidate:
        return {"anyOf": [candidate, {"type": "null"}]}
    if isinstance(schema_type, str):
        candidate["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list):
        candidate["type"] = [*schema_type, "null"]
    else:
        return {"anyOf": [candidate, {"type": "null"}]}
    enum_values = candidate.get("enum")
    if isinstance(enum_values, list) and None not in enum_values:
        candidate["enum"] = [*enum_values, None]
    return candidate


def portable_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Compile a canonical schema to the shared provider grammar subset.

    Provider grammars do not necessarily implement every JSON Schema keyword.
    The canonical schema remains authoritative for local validation; this copy
    contains only constraints that can safely be delegated to hosted grammars.
    """

    def _compile(node: Any) -> Any:
        if not isinstance(node, dict):
            return deepcopy(node)
        # OpenAI's Structured Outputs subset rejects ``uniqueItems`` even
        # though it is valid JSON Schema.  Keep that semantic constraint in
        # the canonical schema used by the local validator, but omit it from
        # the provider-facing grammar.  This compiler is deliberately the
        # single provider boundary so every dynamically derived schema gets
        # the same treatment without weakening deterministic validation.
        compiled = {
            key: deepcopy(value)
            for key, value in node.items()
            if key != "uniqueItems"
        }
        for union_key in ("anyOf", "oneOf"):
            branches = compiled.get(union_key)
            if isinstance(branches, list):
                compiled[union_key] = [_compile(branch) for branch in branches]
        if isinstance(compiled.get("items"), dict):
            compiled["items"] = _compile(compiled["items"])

        schema_types = compiled.get("type")
        is_object = schema_types == "object" or (
            isinstance(schema_types, list) and "object" in schema_types
        )
        if is_object:
            properties = compiled.get("properties")
            if not isinstance(properties, dict):
                raise ValueError("strict output object schemas require properties")
            original_required = {
                str(item) for item in compiled.get("required", [])
            }
            compiled_properties: Dict[str, Any] = {}
            for name, property_schema in properties.items():
                property_compiled = _compile(property_schema)
                if name not in original_required:
                    property_compiled = _nullable_schema(property_compiled)
                compiled_properties[str(name)] = property_compiled
            compiled["properties"] = compiled_properties
            compiled["required"] = list(compiled_properties)
            compiled["additionalProperties"] = False

        is_array = schema_types == "array" or (
            isinstance(schema_types, list) and "array" in schema_types
        )
        if is_array and not isinstance(compiled.get("items"), dict):
            raise ValueError("strict output array schemas require items")
        return compiled

    return _compile(schema)


def openai_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible name for the shared strict-output compiler."""

    return portable_strict_json_schema(schema)


def normalize_strict_output(payload: Any, schema: Dict[str, Any]) -> Any:
    """Remove null placeholders added only to make optional fields strict."""

    def _matching_branch(value: Any, branches: Any) -> Dict[str, Any] | None:
        if not isinstance(branches, list):
            return None
        expected = (
            "object"
            if isinstance(value, dict)
            else "array"
            if isinstance(value, list)
            else "boolean"
            if isinstance(value, bool)
            else "number"
            if isinstance(value, (int, float))
            else "string"
            if isinstance(value, str)
            else "null"
        )
        matching_type = [
            branch
            for branch in branches
            if isinstance(branch, dict)
            and (
                branch.get("type") == expected
                or (
                    isinstance(branch.get("type"), list)
                    and expected in branch["type"]
                )
            )
        ]
        if isinstance(value, dict):
            non_null_keys = {str(key) for key, item in value.items() if item is not None}
            compatible: list[tuple[int, int, Dict[str, Any]]] = []
            for branch in matching_type:
                properties = branch.get("properties")
                if not isinstance(properties, dict):
                    continue
                property_names = {str(key) for key in properties}
                if not non_null_keys.issubset(property_names):
                    continue
                discriminator_mismatch = False
                for key in non_null_keys:
                    property_schema = properties.get(key)
                    if not isinstance(property_schema, dict):
                        continue
                    actual = value.get(key)
                    if "const" in property_schema and actual != property_schema["const"]:
                        discriminator_mismatch = True
                        break
                    allowed = property_schema.get("enum")
                    if isinstance(allowed, list) and actual not in allowed:
                        discriminator_mismatch = True
                        break
                if discriminator_mismatch:
                    continue
                required = {str(item) for item in branch.get("required", [])}
                required_present = len(required.intersection(non_null_keys))
                compatible.append(
                    (required_present, -len(property_names - non_null_keys), branch)
                )
            if compatible:
                compatible.sort(key=lambda item: (item[0], item[1]), reverse=True)
                return compatible[0][2]
        if isinstance(value, list) and value and all(
            isinstance(item, dict) for item in value
        ):
            key_sets = [
                {str(key) for key, nested in item.items() if nested is not None}
                for item in value
            ]
            compatible_arrays: list[tuple[int, int, Dict[str, Any]]] = []
            for branch in matching_type:
                item_schema = branch.get("items")
                properties = (
                    item_schema.get("properties")
                    if isinstance(item_schema, dict)
                    else None
                )
                if not isinstance(properties, dict):
                    continue
                property_names = {str(key) for key in properties}
                if any(not keys.issubset(property_names) for keys in key_sets):
                    continue
                required = {
                    str(item)
                    for item in item_schema.get("required", [])
                }
                required_present = sum(
                    len(required.intersection(keys)) for keys in key_sets
                )
                unused_properties = sum(
                    len(property_names - keys) for keys in key_sets
                )
                compatible_arrays.append(
                    (required_present, -unused_properties, branch)
                )
            if compatible_arrays:
                compatible_arrays.sort(
                    key=lambda item: (item[0], item[1]),
                    reverse=True,
                )
                return compatible_arrays[0][2]
        for branch in matching_type:
            if not isinstance(branch, dict):
                continue
            return branch
        return None

    def _normalize(value: Any, node: Any) -> Any:
        if not isinstance(node, dict):
            return value
        branch = _matching_branch(value, node.get("anyOf") or node.get("oneOf"))
        if branch is not None:
            return _normalize(value, branch)
        if isinstance(value, dict):
            properties = node.get("properties")
            if not isinstance(properties, dict):
                return value
            required = {str(item) for item in node.get("required", [])}
            normalized: Dict[str, Any] = {}
            for key, item in value.items():
                if key in properties and item is None and key not in required:
                    continue
                normalized[key] = _normalize(item, properties.get(key, {}))
            return normalized
        if isinstance(value, list) and isinstance(node.get("items"), dict):
            return [_normalize(item, node["items"]) for item in value]
        return value

    return _normalize(payload, schema)


def strict_json_text_format(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a Responses API ``text.format`` payload for strict JSON Schema."""

    return {
        "text": {
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": deepcopy(schema),
            }
        }
    }


_JSON_VALUE_SCHEMA: Dict[str, Any] = canonical_patch_value_schema()

_EVIDENCE_QUOTE_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": 160,
}


def _canonical_structured_goal_schema() -> Dict[str, Any]:
    return canonical_structured_goal_schema()


def _canonical_task_context_schema() -> Dict[str, Any]:
    return canonical_task_context_schema()


REQUEST_WORKBENCH_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ready", "need_clarification", "out_of_scope"],
        },
        "questions": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": ["string", "null"]},
        "clarification_requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "missing_field": {"type": "string", "minLength": 1},
                    "question": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["missing_field", "question", "reason"],
                "additionalProperties": False,
            },
        },
        "request_envelope": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "schema_version": {"type": "string", "const": "v3"},
                        "request_mode": {
                            "type": "string",
                            "enum": ["single_task", "composite"],
                        },
                        "request_summary": {"type": "string"},
                        "request_classifications": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task_id": {"type": "string"},
                                    "task_type": {
                                        "type": "string",
                                        "enum": sorted(CANONICAL_TASK_TYPES),
                                    },
                                    "label": {"type": "string"},
                                    "native": {"type": "boolean"},
                                    "can_run_in_parallel": {"type": "boolean"},
                                    "context": _canonical_task_context_schema(),
                                },
                                "required": [
                                    "task_id",
                                    "task_type",
                                    "label",
                                    "native",
                                    "can_run_in_parallel",
                                    "context",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "task_dependencies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "upstream_task_id": {"type": "string"},
                                    "downstream_task_id": {"type": "string"},
                                },
                                "required": [
                                    "upstream_task_id",
                                    "downstream_task_id",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "final_outputs": {
                            "type": "array",
                            "items": {"anyOf": [
                                {"type": "string"},
                                {"type": "object", "properties": {
                                    "kind": {"type": "string", "minLength": 1},
                                    "task_id": {"type": ["string", "null"]},
                                    "path": {"type": ["string", "null"], "description": "Preserve an explicitly requested filename exactly. If the user specifies only an output directory, choose a concrete filename within it or retain the directory with an explicit format; an existing directory or trailing slash is resolved to a separate producer-bound file."},
                                    "format": {"type": ["string", "null"], "description": "File extension such as csv, md, json or xlsx; required for a directory destination unless an explicit native artifact role determines it."},
                                    "label": {"type": ["string", "null"]},
                                    "source_artifact": {
                                        "type": ["string", "null"],
                                        "enum": [None, *NATIVE_EXPORT_FORMATS],
                                        "description": "Existing native output of task_id to export unchanged. Use design_panel_csv for the selected candidate table, generated_aso_set_json for full candidate evidence, or report_md/report_html for its completed report only when its existing contents satisfy the requested output. This role does not add requested calculations or annotations. When custom content is required, use null with kind=file and describe the required transformation, or bind an explicit analysis task that produces it. Ordinary native fields do not require a custom transformation.",
                                    },
                                    "source_scope": {
                                        "type": ["object", "null"],
                                        "description": "Optional row selection for design_panel_csv when the requested file contains a particular design branch rather than the full task panel. Match recorded program IDs and/or modalities; both filters intersect when supplied. Omit for a complete panel. Do not infer membership from filenames or sequences.",
                                        "properties": {
                                            "program_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
                                            "modalities": {"type": "array", "items": {"type": "string", "enum": ["gapmer", "steric"]}},
                                        },
                                        "additionalProperties": False,
                                    },
                                }, "required": ["kind"], "additionalProperties": False},
                            ]},
                        },
                        "response_mode": {
                            "type": "string",
                            "enum": ["report", "answer", "analysis", "composite"],
                        },
                        "requires_execution": {"type": "boolean"},
                        "assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "schema_version",
                        "request_mode",
                        "request_summary",
                        "request_classifications",
                        "tasks",
                        "task_dependencies",
                        "final_outputs",
                        "response_mode",
                        "requires_execution",
                        "assumptions",
                    ],
                    "additionalProperties": False,
                },
            ]
        },
    },
    "required": ["status", "questions", "reason", "request_envelope"],
    "additionalProperties": False,
}


REQUEST_GRAPH_REVIEW_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["valid", "revise"],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "missing_task",
                            "spurious_task",
                            "wrong_task_type",
                            "wrong_dependency",
                            "misplaced_input",
                            "unsupported_fact",
                            "missing_fact",
                        ],
                    },
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "path": {"type": ["string", "null"]},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                    "evidence_quote": deepcopy(_EVIDENCE_QUOTE_SCHEMA),
                    "reason": {"type": "string"},
                },
                "required": [
                    "category",
                    "task_ids",
                    "path",
                    "evidence_refs",
                    "evidence_quote",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "issues"],
    "additionalProperties": False,
}


def _synthesized_design_goal_schema() -> Dict[str, Any]:
    schema = _canonical_structured_goal_schema()
    # Every newly compiled design makes its delivery obligation explicit.
    # Keep persisted/legacy goal loading compatible with the default minimum.
    schema["required"] = [*schema["required"], "candidate_minimum", "candidate_limit"]
    return schema


DESIGN_GOAL_SYNTHESIS_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ready", "needs_clarification"],
        },
        "questions": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": ["string", "null"]},
        "structured_goal": {
            "anyOf": [
                {"type": "null"},
                _synthesized_design_goal_schema(),
            ]
        },
    },
    "required": ["status", "questions", "reason", "structured_goal"],
    "additionalProperties": False,
}


GOAL_CONTRACT_AUDIT_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["valid", "repairable", "needs_clarification"],
        },
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["add", "replace", "remove"],
                    },
                    "path": {"type": "string"},
                    "value": deepcopy(_JSON_VALUE_SCHEMA),
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence_quote": deepcopy(_EVIDENCE_QUOTE_SCHEMA),
                    "reason": {"type": "string"},
                },
                "required": [
                    "op",
                    "path",
                    "value",
                    "evidence_refs",
                    "evidence_quote",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "goal_replacements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_index": {"type": "integer", "minimum": 0},
                    "task_id": {"type": "string"},
                    "structured_goal": _canonical_structured_goal_schema(),
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence_quote": deepcopy(_EVIDENCE_QUOTE_SCHEMA),
                    "reason": {"type": "string"},
                },
                "required": [
                    "task_index",
                    "task_id",
                    "structured_goal",
                    "evidence_refs",
                    "evidence_quote",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "field_dispositions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding_index": {"type": "integer", "minimum": 0},
                    "action": {
                        "type": "string",
                        "enum": [
                            "relocate",
                            "remove",
                            "contract_gap",
                            "needs_user_input",
                        ],
                    },
                    "source_path": {"type": "string"},
                    "destination_patches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "value": deepcopy(_JSON_VALUE_SCHEMA),
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence_quote": deepcopy(
                                    _EVIDENCE_QUOTE_SCHEMA
                                ),
                                "reason": {"type": "string"},
                            },
                            "required": [
                                "path",
                                "value",
                                "evidence_refs",
                                "evidence_quote",
                                "reason",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "finding_index",
                    "action",
                    "source_path",
                    "destination_patches",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "unresolved": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "question": {"type": "string"},
                    "reason": {"type": "string"},
                    "resolution": {
                        "type": "string",
                        "enum": ["user_input", "source_retrieval"],
                    },
                },
                "required": ["path", "question", "reason", "resolution"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "status",
        "patches",
        "goal_replacements",
        "field_dispositions",
        "unresolved",
    ],
    "additionalProperties": False,
}


GOAL_CONTRACT_REPAIR_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["applied", "cannot_apply"],
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["add", "replace", "remove"],
                    },
                    "path": {"type": "string"},
                    "value": deepcopy(_JSON_VALUE_SCHEMA),
                    "finding_index": {"type": "integer", "minimum": 0},
                    "destination_index": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "op",
                    "path",
                    "value",
                    "finding_index",
                    "destination_index",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "errors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "operations", "errors"],
    "additionalProperties": False,
}
