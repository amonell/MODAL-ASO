"""Bind semantic input-role checks to every declared candidate file.

The model interprets the original evidence. The host requires complete coverage
and routes reference-only or unresolved assignments through the existing repair.
No filename, sequence-length or biological heuristic classifies an input here.
"""
from copy import deepcopy
import json


def declared_candidate_files(prompt):
    """Read graph protocol fields, never protocol-looking text inside evidence."""
    seen = set()
    while isinstance(prompt, str) and prompt not in seen:
        seen.add(prompt)
        try:
            payload = json.loads(prompt.partition('\n\n')[2])
        except (ValueError, TypeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        envelope = payload.get('proposed_request_envelope')
        if isinstance(envelope, dict):
            result = {}
            for task_index, task in enumerate(envelope.get('tasks') or []):
                if not isinstance(task, dict):
                    continue
                context = task.get('context') or {}
                if not isinstance(context, dict):
                    continue
                specs = context.get('provided_aso_files') or []
                if not isinstance(specs, list):
                    continue
                for file_index, spec in enumerate(specs):
                    path = spec.get('path') if isinstance(spec, dict) else spec
                    if isinstance(path, str) and path.strip():
                        pointer = f'/tasks/{task_index}/context/provided_aso_files/{file_index}'
                        result[pointer] = {'task_id': task.get('task_id'), 'path': path,
                                           'declaration': deepcopy(spec),
                                           'provided_sequence_role': context.get('provided_sequence_role')}
            return result
        prompt = payload.get('original_review_input')
    return {}


def with_file_role_schema(schema, files):
    if not files:
        return schema
    result = deepcopy(schema)
    evidence = result['properties']['issues']['items']['properties']
    properties = {}
    for pointer, entry in files.items():
        fields = {key: deepcopy(evidence[key])
                  for key in ('evidence_refs', 'evidence_span_id', 'reason')}
        fields['role'] = {'type': 'string', 'enum': [
            'candidate_input', 'reference_or_annotation_only', 'not_established']}
        properties[pointer] = {
            'type': 'object', 'additionalProperties': False,
            'description': 'Classify this proposed file from original user/source evidence, '
                           'not from its placement in the proposal: ' + json.dumps(entry, ensure_ascii=False),
            'properties': fields, 'required': list(fields),
        }
    result['properties']['file_role_checks'] = {
        'type': 'object', 'additionalProperties': False,
        'properties': properties, 'required': list(properties),
    }
    result['required'] = [*result['required'], 'file_role_checks']
    return result


def has_non_candidate_file(review):
    checks = review.get('file_role_checks') or {}
    return any(isinstance(value, dict) and value.get('role') in {
        'reference_or_annotation_only', 'not_established'} for value in checks.values())


def apply_file_role_checks(review, files):
    """Turn cited role judgments into the existing graph-repair interface."""
    result = deepcopy(review)
    checks = result.pop('file_role_checks', {})
    if files and set(checks) != set(files):
        raise ValueError('File-role review does not cover the declared candidate files')
    for pointer, check in checks.items():
        if check['role'] == 'candidate_input':
            continue
        entry = files[pointer]
        reference_only = check['role'] == 'reference_or_annotation_only'
        category = 'misplaced_input' if reference_only else 'missing_fact'
        if any(issue.get('path') == pointer and issue.get('category') == category
               for issue in result['issues']):
            continue
        result['issues'].append({
            'category': category,
            'task_ids': [entry['task_id']] if entry['task_id'] else [],
            'path': pointer, 'evidence_refs': deepcopy(check['evidence_refs']),
            'evidence_quote': check['evidence_quote'],
            'reason': (
                f"The role check for {entry['path']} is {check['role']}: {check['reason']} "
                + ('Keep this evidence file in artifact_refs rather than submitted candidate inputs.'
                   if reference_only else
                   'Resolve its candidate-input role from the original evidence; preserve the file and '
                   'do not infer its role from its name, format or sequence length.')),
        })
    if result['issues']:
        result['status'] = 'revise'
    return result
