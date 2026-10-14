"""Apply hash-bound line edits to a failed analysis without rewriting its plan."""
from copy import deepcopy
import hashlib


def code_digest(code):
    return hashlib.sha256(code.encode()).hexdigest()


def repair_schema(code):
    count = len(code.splitlines())
    fields = {"start_line": {"type": "integer", "minimum": 1, "maximum": count},
              "end_line": {"type": "integer", "minimum": 1, "maximum": count},
              "replacement": {"type": "string"}}
    return {"type": "object", "properties": {
        "base_sha256": {"type": "string", "const": code_digest(code)},
        "patches": {"type": "array", "minItems": 1,
                    "items": {"type": "object", "properties": fields,
                              "required": list(fields), "additionalProperties": False}},
        "explanation": {"type": "string"}},
        "required": ["base_sha256", "patches", "explanation"], "additionalProperties": False}


def apply_program_repair(previous_plan, code, repair):
    if repair.get("base_sha256") != code_digest(code):
        raise ValueError("Code repair is not bound to the failed program.")
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
    result = deepcopy(previous_plan)
    result["python_code"] = revised
    result["native_code_repair"] = deepcopy(repair)
    return result
