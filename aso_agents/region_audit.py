"""Evidence coverage for existing, narrowly scoped region semantic audits."""
from copy import deepcopy
import re


_SCOPE = re.compile(
    r"/tasks/(\d+)/context/structured_goal(?:/design_programs/(\d+))?/region_constraints"
)


def is_focused_region_audit(protocol):
    scopes = protocol.get('semantic_scope_pointers')
    return (
        protocol.get('audit_mode') in {
            'semantic_completeness_and_contract_repair', 'post_repair_semantic_verification'}
        and not protocol.get('canonical_contract_issues')
        and not protocol.get('noncanonical_field_findings')
        and isinstance(scopes, list) and bool(scopes)
        and all(isinstance(p, str) and _SCOPE.fullmatch(p) for p in scopes)
    )


def region_audit_obligations(protocol):
    """Enumerate pointer pairs, without judging scientific intent or changing rules."""
    if not is_focused_region_audit(protocol):
        return []
    tasks = (protocol.get('current_request_envelope') or {}).get('tasks') or []
    obligations = []
    for scope in dict.fromkeys(protocol['semantic_scope_pointers']):
        match = _SCOPE.fullmatch(scope)
        task_index = int(match[1])
        if task_index >= len(tasks) or not isinstance(tasks[task_index], dict):
            continue
        goal = (tasks[task_index].get('context') or {}).get('structured_goal') or {}
        programs = goal.get('design_programs') or []
        root = f'/tasks/{task_index}/context/structured_goal'
        if match[2] is not None:
            index = int(match[2])
            if index >= len(programs) or not isinstance(programs[index], dict):
                continue
            owner = programs[index]
            affected = [(root + f'/design_programs/{index}', owner.get('program_id'))]
        else:
            owner = goal
            affected = [(root + f'/design_programs/{i}', p.get('program_id'))
                        for i, p in enumerate(programs) if isinstance(p, dict)]
            if not affected:
                affected = [(root, None)]
        for index, rule in enumerate(owner.get('region_constraints') or []):
            if not isinstance(rule, dict):
                continue
            for program_pointer, program_id in affected:
                obligations.append({
                    'check_id': f'check_{len(obligations)}',
                    'constraint_pointer': scope + f'/{index}',
                    'program_pointer': program_pointer,
                    'program_id': program_id,
                })
    return obligations


def with_region_audit_coverage(schema, obligations):
    result = deepcopy(schema)
    finding = {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'verdict': {'type': 'string', 'enum': ['supported', 'unsupported', 'unclear']},
            'effective_restriction': {'type': 'string', 'minLength': 1},
            'evidence_refs': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                              'items': {'type': 'string'}},
            'evidence_quote': {'type': 'string', 'minLength': 1},
            'reason': {'type': 'string', 'minLength': 1},
        },
        'required': ['verdict', 'effective_restriction', 'evidence_refs', 'evidence_quote', 'reason'],
    }
    result['properties']['region_scope_checks'] = {
        'type': 'object', 'additionalProperties': False,
        'properties': {o['check_id']: deepcopy(finding) for o in obligations},
        'required': [o['check_id'] for o in obligations],
    }
    result['required'].append('region_scope_checks')
    return result


def region_audit_coverage_errors(payload, obligations, span_map):
    """Unsupported findings must be paired with an effective scoped correction."""
    if not isinstance(payload, dict):
        return []  # The response schema reports malformed payloads.
    checks = payload.get('region_scope_checks')
    if not isinstance(checks, dict):
        return ['$.region_scope_checks: provide every required evidence check before a verdict']
    errors = []
    patches = payload.get('patches') or []
    for obligation in obligations:
        key = obligation['check_id']
        finding = checks.get(key)
        if not isinstance(finding, dict):
            continue  # Required keys and item types are enforced by the schema.
        span = span_map.get(finding.get('evidence_span_id'))
        if not span or span['evidence_id'] not in (finding.get('evidence_refs') or []):
            errors.append(f'$.region_scope_checks.{key}: span must belong to a cited evidence source')
        pointer = obligation['constraint_pointer']
        related = lambda path: isinstance(path, str) and (
            path == pointer or path.startswith(pointer + '/') or pointer.startswith(path + '/'))
        if finding.get('verdict') == 'unsupported' and not any(
            isinstance(p, dict) and related(p.get('path')) for p in patches
        ):
            errors.append(f'$.region_scope_checks.{key}: unsupported restriction at {pointer} '
                          'requires an evidence-bound effective correction; valid is unavailable')
        if finding.get('verdict') == 'unclear' and not patches and not any(
            isinstance(u, dict) and related(u.get('path')) for u in payload.get('unresolved') or []
        ):
            errors.append(f'$.region_scope_checks.{key}: unresolved region meaning requires '
                          'a scoped unresolved item; valid is unavailable')
    return errors
