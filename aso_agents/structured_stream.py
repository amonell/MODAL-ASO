"""Stop structured generations only on proven, irreversible array violations.

This is an early rejection check, not a partial-JSON acceptance mechanism. The
ordinary canonical schema validator still validates every completed answer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any


def root_array_constraints(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Read the canonical schema before provider grammar translation drops rules."""
    fmt = (payload.get("text") or {}).get("format") or {}
    if fmt.get("type") != "json_schema":
        return {}
    schema = fmt.get("schema") or {}
    if schema.get("type") != "object":
        return {}
    return {
        name: rule
        for name, rule in (schema.get("properties") or {}).items()
        if isinstance(rule, dict) and rule.get("type") == "array"
        and (rule.get("uniqueItems") is True or (
            type(rule.get("maxItems")) is int and rule["maxItems"] >= 0
        ))
    }


def _json_identity(value: Any) -> Any:
    # JSON Schema distinguishes booleans from numbers, but not 1 from 1.0.
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, Decimal)):
        return ("number", Decimal(value))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("array", tuple(_json_identity(item) for item in value))
    return ("object", frozenset((key, _json_identity(item)) for key, item in value.items()))


class RootArrayGuard:
    """Incrementally examine completed items in constrained root arrays.

    Numbers are not complete until their delimiter arrives. Strings, escapes,
    nested values and arbitrary chunk boundaries are handled by JSONDecoder.
    Unparseable prefixes are left to final validation; they are never accepted.
    """

    def __init__(self, constraints: dict[str, dict[str, Any]]):
        self.constraints = constraints
        self.buffer = ""
        self.state = "root"
        self.key = ""
        self.seen: dict[Any, int] = {}
        self.count = 0
        self.decoder = json.JSONDecoder(parse_float=Decimal)

    def feed(self, text: str) -> dict[str, Any] | None:
        if self.state in {"done", "unparseable"}:
            return None
        self.buffer += text
        while self.state not in {"done", "unparseable"}:
            self.buffer = self.buffer.lstrip()
            if not self.buffer:
                return None
            if self.state in {"root", "colon", "array"}:
                expected = {"root": "{", "colon": ":", "array": "["}[self.state]
                if self.buffer[0] != expected:
                    self.state = "unparseable"
                    return None
                self.buffer = self.buffer[1:]
                if self.state == "root":
                    self.state = "key"
                elif self.state == "colon":
                    self.state = "array" if self.key in self.constraints else "value"
                else:
                    self.state = "item"
                    self.seen = {}
                    self.count = 0
                continue
            if self.state == "key" and self.buffer[0] == "}":
                self.state = "done"
                return None
            if self.state == "item" and self.buffer[0] == "]":
                self.buffer = self.buffer[1:]
                self.state = "property_end"
                continue
            if self.state in {"property_end", "item_end"}:
                closer = "}" if self.state == "property_end" else "]"
                token = self.buffer[0]
                if token == ",":
                    self.state = "key" if self.state == "property_end" else "item"
                elif token == closer:
                    self.state = "done" if closer == "}" else "property_end"
                else:
                    self.state = "unparseable"
                    return None
                self.buffer = self.buffer[1:]
                continue
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except (ValueError, RecursionError):
                return None
            if self.state == "key":
                if not isinstance(value, str):
                    self.state = "unparseable"
                    return None
                self.key = value
                self.state = "colon"
            else:
                # A raw_decode result of 1 can still become 10 or 1e10.
                suffix = self.buffer[end:].lstrip()
                if not suffix:
                    return None
                if suffix[0] not in ",]}":
                    return None
                if self.state == "item":
                    rule = self.constraints[self.key]
                    identity = _json_identity(value)
                    self.count += 1
                    details = {"property": self.key, "item_index": self.count - 1}
                    if rule.get("uniqueItems") is True and identity in self.seen:
                        return {**details, "keyword": "uniqueItems", "duplicate_of": self.seen[identity]}
                    if type(rule.get("maxItems")) is int and self.count > rule["maxItems"]:
                        return {**details, "keyword": "maxItems", "limit": rule["maxItems"]}
                    self.seen[identity] = self.count - 1
                    self.state = "item_end"
                else:
                    self.state = "property_end"
            self.buffer = self.buffer[end:]
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else value.model_dump(exclude_none=True)


def consume_guarded_response(stream: Any, constraints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Keep complete provider responses; close a rejected generation immediately.

    Cancellation may not stop upstream billing. Missing terminal usage is unknown,
    never zero. Partial output is represented by a digest, never a final answer.
    """
    guards: dict[tuple[int, int], RootArrayGuard] = {}
    digest = hashlib.sha256()
    characters = 0
    response_identity: dict[str, Any] = {}
    terminal_types = {"response.completed", "response.failed", "response.incomplete"}
    try:
        for raw_event in stream:
            event = _as_dict(raw_event)
            event_type = event.get("type")
            response = event.get("response")
            if isinstance(response, Mapping):
                response_identity.update({key: response[key] for key in ("id", "model", "created_at") if key in response})
            if event_type in terminal_types:
                if not isinstance(response, Mapping):
                    raise ValueError("Terminal Responses event has no response object")
                return dict(response)
            if event_type == "error":
                return {**response_identity, "status": "failed", "output": [], "error": event}
            if event_type != "response.output_text.delta":
                continue
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise ValueError("Responses text delta is not a string")
            digest.update(delta.encode("utf-8"))
            characters += len(delta)
            key = (event.get("output_index", 0), event.get("content_index", 0))
            guard = guards.setdefault(key, RootArrayGuard(constraints))
            violation = guard.feed(delta)
            if violation:
                return {
                    **response_identity, "status": "incomplete", "output": [],
                    "incomplete_details": {"reason": "invalid_structured_output"},
                    "structured_output_rejection": {
                        **violation, "observed_text_characters": characters,
                        "observed_text_sha256": digest.hexdigest(),
                        "usage_available": False,
                    },
                }
        raise ValueError("Responses stream ended without a terminal event")
    finally:
        stream.close()
