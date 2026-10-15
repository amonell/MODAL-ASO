"""Apply bound code edits and reconcile descriptive metadata for that revision."""
from copy import deepcopy
import hashlib
import json


_DESCRIPTIVE_FIELDS = ('analysis_plan', 'assumptions', 'limitations', 'expected_outputs')


def code_digest(code):
    return hashlib.sha256(code.encode()).hexdigest()


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def repair_schema(code, previous_plan):
    count = len(code.splitlines())
    fields = {"start_line": {"type": "integer", "minimum": 1, "maximum": count},
              "end_line": {"type": "integer", "minimum": 1, "maximum": count},
              "replacement": {"type": "string"}}
    return {"type": "object", "properties": {
        "base_sha256": {"type": "string", "const": code_digest(code)},
        "base_plan_sha256": {"type": "string", "const": plan_digest(previous_plan)},
        "patches": {"type": "array", "minItems": 1,
                    "items": {"type": "object", "properties": fields,
                              "required": list(fields), "additionalProperties": False}},
        "plan_updates": {"type": "array", "items": {
            "type": "object", "properties": {
                "field": {"type": "string", "enum": list(_DESCRIPTIVE_FIELDS)},
                "value": {"anyOf": [{"type": "string"},
                                     {"type": "array", "items": {"type": "string"}}]}},
            "required": ["field", "value"], "additionalProperties": False}},
        "explanation": {"type": "string"}},
        "required": ["base_sha256", "base_plan_sha256", "patches", "plan_updates", "explanation"],
        "additionalProperties": False}


def apply_program_repair(previous_plan, code, repair):
    if repair.get("base_sha256") != code_digest(code):
        raise ValueError("Code repair is not bound to the failed program.")
    updates = repair.get('plan_updates', [])
    # Legacy code-only records remain readable. New native repairs always
    # carry both bindings and explicitly list any descriptive revisions.
    if 'plan_updates' in repair or 'base_plan_sha256' in repair:
        if repair.get('base_plan_sha256') != plan_digest(previous_plan):
            raise ValueError('Plan repair is not bound to the preceding plan.')
        if not isinstance(updates, list) or 'plan_updates' not in repair:
            raise ValueError('Plan repair must contain a list of descriptive updates.')
    seen = set()
    for update in updates:
        if not isinstance(update, dict) or set(update) != {'field', 'value'}:
            raise ValueError('Invalid descriptive plan update.')
        field, value = update['field'], update['value']
        if field not in _DESCRIPTIVE_FIELDS or field in seen:
            raise ValueError('Plan repair may update each descriptive field once only.')
        valid = (isinstance(value, str) if field == 'analysis_plan' else
                 isinstance(value, list) and all(isinstance(item, str) for item in value))
        if not valid:
            raise ValueError('Plan repair has an invalid descriptive field type.')
        seen.add(field)
    lines = code.splitlines(keepends=True)
    patches = repair.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError("Code repair must contain explicit line replacements.")
    normalized = []
    for patch in patches:
        start, end = patch.get("start_line"), patch.get("end_line")
        if (type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines)
                or not isinstance(patch.get("replacement"), str)):
            raise ValueError("Code repair contains an invalid line span or replacement.")
        normalized.append((start, end, patch["replacement"]))
    normalized.sort()
    if any(left[1] >= right[0] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("Code repair spans must not overlap.")
    for start, end, replacement in reversed(normalized):
        lines[start - 1:end] = [replacement.rstrip("\n") + "\n"] if replacement else []
    revised = "".join(lines).strip()
    if revised == code.strip():
        raise ValueError("The repair did not change the failed program.")
    compile(revised, "generated_code.py", "exec")
    from .analysis_dependencies import validate_program_dependencies
    validate_program_dependencies(revised)
    result = deepcopy(previous_plan)
    result["python_code"] = revised
    for update in updates:
        result[update['field']] = deepcopy(update['value'])
    result["native_code_repair"] = deepcopy(repair)
    return result
