"""Apply bound code edits and reconcile descriptive metadata for that revision."""
from copy import deepcopy
import ast
from collections import Counter
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


def _duplicate_statement_pairs(code):
    pairs = []
    for parent in ast.walk(ast.parse(code)):
        for _field, children in ast.iter_fields(parent):
            if not isinstance(children, list):
                continue
            for left, right in zip(children, children[1:]):
                if not isinstance(left, ast.stmt) or not isinstance(right, ast.stmt):
                    continue
                key = ast.dump(left, include_attributes=False)
                if key == ast.dump(right, include_attributes=False):
                    pairs.append((key, left, right))
    return pairs


def _validate_repair_boundaries(original, revised, origins):
    """Reject copied executable context; never silently remove observations.

    A replacement can accidentally include the unchanged statement immediately
    outside its declared line span. Intentional repetition remains possible by
    explicitly replacing the entire repeated span. Existing repetitions and
    identical statements in different control-flow branches are unaffected.
    """
    try:
        original_pairs = _duplicate_statement_pairs(original)
    except SyntaxError:
        # A syntax repair has no valid AST baseline for this comparison. The
        # revised program still has to compile and pass dependency validation.
        return
    before = Counter(key for key, _left, _right in original_pairs)
    pairs = _duplicate_statement_pairs(revised)
    after = Counter(key for key, _left, _right in pairs)
    for key, left, right in pairs:
        if after[key] <= before[key]:
            continue
        left_sources = set(origins[left.lineno - 1:left.end_lineno])
        right_sources = set(origins[right.lineno - 1:right.end_lineno])
        crosses_boundary = (
            left_sources == {None} and right_sources and None not in right_sources
            or right_sources == {None} and left_sources and None not in left_sources
        )
        if crosses_boundary:
            raise ValueError(
                f"Code repair duplicated an unchanged neighboring statement at lines "
                f"{left.lineno} and {right.lineno}. Replacements contain only their "
                "declared inclusive span, without copied context. Remove the copied "
                "statement, or include the original statement in the explicit edit "
                "span if repetition is intended."
            )


def _validate_repaired_syntax(original, revised, origins, base_lines, patches):
    """Explain syntax failures against the bound edits without changing code."""
    try:
        compile(revised, "generated_code.py", "exec")
    except SyntaxError as exc:
        lines = revised.splitlines()
        center = min(max(exc.lineno or len(lines), 1), len(lines))
        context = []
        for index in range(max(0, center - 4), min(len(lines), center + 3)):
            patch = origins[index]
            context.append({
                "revised_line": index + 1, "base_line": base_lines[index],
                "patch_number": patch + 1 if patch is not None else None,
                "base_span": list(patches[patch][:2]) if patch is not None else None,
                "text": lines[index][:1000], "text_truncated": len(lines[index]) > 1000,
            })
        evidence = {
            "base_sha256": code_digest(original), "message": exc.msg,
            "lineno": exc.lineno, "offset": exc.offset,
            "end_lineno": exc.end_lineno, "end_offset": exc.end_offset,
            "context": context,
            "repair_instruction": "Correct the proposed edits against the original bound "
                "source_lines. Revised line numbers describe the rejected result, not a new "
                "patch base. patch_number follows base-span order. Preserve intended block "
                "structure; no indentation or scientific operation was automatically changed.",
        }
        raise ValueError("Repaired program syntax error\n" + json.dumps(evidence)) from exc


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
    origins = [None] * len(lines)
    base_lines = list(range(1, len(lines) + 1))
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
    for patch_index, (start, end, replacement) in reversed(list(enumerate(normalized))):
        replacement_lines = (replacement.rstrip("\n") + "\n").splitlines(keepends=True) if replacement else []
        lines[start - 1:end] = replacement_lines
        origins[start - 1:end] = [patch_index] * len(replacement_lines)
        base_lines[start - 1:end] = [None] * len(replacement_lines)
    revised = "".join(lines)
    _validate_repaired_syntax(code, revised, origins, base_lines, normalized)
    _validate_repair_boundaries(code, revised, origins)
    revised = revised.strip()
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
