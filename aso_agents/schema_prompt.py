"""Lossless sharing of repeated enum schemas in model-facing instructions."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from typing import Any, Iterator


_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_VALUES = (
    "items", "additionalItems", "additionalProperties", "unevaluatedItems",
    "unevaluatedProperties", "contains", "propertyNames", "not", "if", "then",
    "else", "contentSchema",
)
_SCHEMA_ARRAYS = ("allOf", "anyOf", "oneOf", "prefixItems")
_LOCATION_KEYS = {"$ref", "$dynamicRef", "$recursiveRef", "$id", "id", "$anchor",
                  "$dynamicAnchor", "$recursiveAnchor"}


def _schema_nodes(schema: Any) -> Iterator[dict[str, Any]]:
    """Visit schema positions, never literal enum/const/default/example data."""
    if not isinstance(schema, dict):
        return
    yield schema
    for key in _SCHEMA_MAPS:
        values = schema.get(key)
        if isinstance(values, dict):
            for value in values.values():
                yield from _schema_nodes(value)
    for key in _SCHEMA_VALUES:
        value = schema.get(key)
        if isinstance(value, list):  # Legacy tuple-form items.
            for item in value:
                yield from _schema_nodes(item)
        else:
            yield from _schema_nodes(value)
    for key in _SCHEMA_ARRAYS:
        values = schema.get(key)
        if isinstance(values, list):
            for value in values:
                yield from _schema_nodes(value)
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for value in dependencies.values():
            if isinstance(value, dict):
                yield from _schema_nodes(value)


def _encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compact_schema_for_prompt(schema: dict[str, Any]) -> dict[str, Any]:
    """Represent identical large enum schemas once, without dropping any value.

    This changes only the model-facing representation. Callers continue to
    validate against the original canonical schema. Schemas with existing
    references or resource/anchor scopes remain unchanged, avoiding changes to
    their reference resolution. Only location-independent type+enum schemas
    are shared; descriptions, examples and other constraints stay in place.
    """
    result = deepcopy(schema)
    nodes = list(_schema_nodes(result))
    if any(_LOCATION_KEYS.intersection(node) for node in nodes):
        return result
    if "$defs" in result and not isinstance(result["$defs"], dict):
        return result
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if set(node) == {"type", "enum"} and isinstance(node["enum"], list):
            groups[_encoded(node)].append(node)

    definitions = deepcopy(result.get("$defs", {}))
    changed = False
    for encoded, copies in sorted(groups.items()):
        if len(copies) < 2:
            continue
        name = "shared_enum_" + hashlib.sha256(encoded.encode()).hexdigest()[:16]
        suffix = 0
        base = name
        while name in definitions:
            suffix += 1
            name = f"{base}_{suffix}"
        reference = {"$ref": f"#/$defs/{name}"}
        # Sharing is worthwhile only if definitions and references reduce the
        # complete serialization. Small repeated enums stay inline.
        cost = len(encoded) + len(_encoded(name)) + 2 + len(_encoded(reference)) * len(copies)
        if cost >= len(encoded) * len(copies):
            continue
        definitions[name] = deepcopy(copies[0])
        for node in copies:
            node.clear()
            node.update(reference)
        changed = True
    if not changed:
        return result
    # Retain changes to any original definitions visited above, then add only
    # the newly created shared schemas.
    original_names = set(result.get("$defs", {}))
    result.setdefault("$defs", {}).update({key: value for key, value in definitions.items()
                                          if key not in original_names})
    return result if len(_encoded(result)) < len(_encoded(schema)) else deepcopy(schema)
