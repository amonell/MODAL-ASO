"""Evidence-bound finite length coverage in existing broad design-goal audits."""
from copy import deepcopy


PROGRAM_LENGTH_AUDIT_INSTRUCTIONS = """
For each program_length_checks_required entry, fill its check_id under
program_length_checks. Determine the complete explicitly requested oligo-length
domain for that intervention branch from evidence_sources: enumerate a finite set
or every integer of a bounded inclusive range. Preserve branch-specific differences.
Do not silently pick a default length or accept lengths mentioned only in labels,
objectives, chemistry or notes. List the actual same-branch represented_program_ids
and compare their typed design_programs[*].length values with the requested domain.
Use complete only for exactly one program per requested length. Use incomplete for
omitted, duplicated or out-of-domain variants and supply effective canonical patches
in the same response: set missing length scalars and add otherwise-equivalent
program variants with unique IDs, preserving supported branch facts and graph
references. Do not combine unrelated branches to manufacture complete coverage.
If no length domain was explicitly requested, use not_requested with empty domain
and represented-program arrays; do not invent a domain or ask for an optional one.
Use unclear only for a necessary user choice genuinely absent from the evidence.
Every finding needs a supplied evidence_refs entry, exact evidence_span_id and a
short explanation of its branch/domain interpretation. Missing or incomplete
coverage cannot be accepted as valid. This audit adds no biological requirements.
"""


def program_length_audit_obligations(protocol):
    if (protocol.get('audit_mode') not in {
        'semantic_completeness_and_contract_repair', 'post_repair_semantic_verification'}
        or protocol.get('semantic_scope_pointers')
        or protocol.get('canonical_contract_issues')
        or protocol.get('noncanonical_field_findings')):
        return []
    obligations = []
    tasks = (protocol.get('current_request_envelope') or {}).get('tasks') or []
    for task_index, task in enumerate(tasks):
        if not isinstance(task, dict) or task.get('task_type') != 'design_new_asos':
            continue
        goal = (task.get('context') or {}).get('structured_goal') or {}
        programs = goal.get('design_programs') or []
        available = {p['program_id']: p.get('length') for p in programs
                     if isinstance(p, dict) and isinstance(p.get('program_id'), str)}
        root = f'/tasks/{task_index}/context/structured_goal/design_programs'
        for index, program in enumerate(programs):
            if not isinstance(program, dict) or program.get('program_id') not in available:
                continue
            obligations.append({'check_id': f'program_{len(obligations)}',
                'program_pointer': root + f'/{index}', 'program_id': program['program_id'],
                'programs_pointer': root, 'available_program_lengths': deepcopy(available),
                'available_program_pointers': {p['program_id']: root + f'/{i}'
                    for i, p in enumerate(programs) if isinstance(p, dict)
                    and isinstance(p.get('program_id'), str)}})
    return obligations


def with_program_length_coverage(schema, obligations):
    result = deepcopy(schema)
    findings = {}
    for obligation in obligations:
        properties = {
            'verdict': {'type':'string','enum':['not_requested','complete','incomplete','unclear']},
            'requested_lengths': {'type':'array','uniqueItems':True,
                                  'items':{'type':'integer','minimum':1}},
            'represented_program_ids': {'type':'array','uniqueItems':True,
                'items':{'type':'string','enum':list(obligation['available_program_lengths'])}},
            'evidence_refs': {'type':'array','minItems':1,'uniqueItems':True,'items':{'type':'string'}},
            'evidence_quote': {'type':'string','minLength':1},
            'reason': {'type':'string','minLength':1},
        }
        findings[obligation['check_id']] = {'type':'object','additionalProperties':False,
            'properties':properties,'required':list(properties)}
    result['properties']['program_length_checks'] = {
        'type':'object','additionalProperties':False,'properties':findings,'required':list(findings)}
    result['required'].append('program_length_checks')
    return result


def program_length_coverage_errors(payload, obligations, span_map):
    """Call after schema validation; compare findings to actual typed variants."""
    checks = payload['program_length_checks']
    errors = []
    for obligation in obligations:
        finding = checks[obligation['check_id']]
        path = '$.program_length_checks.' + obligation['check_id']
        span = span_map.get(finding.get('evidence_span_id'))
        if not span or span['evidence_id'] not in finding['evidence_refs']:
            errors.append(path + ': span must belong to a cited evidence source')
        domain, ids = finding['requested_lengths'], finding['represented_program_ids']
        verdict = finding['verdict']
        if verdict == 'not_requested':
            if domain or ids:
                errors.append(path + ': not_requested requires empty length and program arrays')
            continue
        if verdict != 'unclear' and not domain:
            errors.append(path + ': an explicit finite requested domain must be nonempty')
        if verdict == 'complete':
            observed = [obligation['available_program_lengths'][identifier] for identifier in ids]
            if (obligation['program_id'] not in ids or len(observed) != len(domain)
                or any(type(value) is not int for value in observed)
                or sorted(observed) != sorted(domain)):
                errors.append(path + ': complete requires the current program and exactly one '
                              'typed program.length for every requested value; notes and defaults do not count')
        root = obligation['programs_pointer']
        group_pointers = {obligation['program_pointer']} | {
            obligation['available_program_pointers'][identifier] for identifier in ids}
        if verdict == 'incomplete' and not any(
            (patch['op'] == 'add' and patch['path'] == root + '/-')
            or (patch['path'] in {p + '/length' for p in group_pointers}
                and patch['op'] in {'add','replace'})
            or (patch['path'] in group_pointers
                and patch['op'] == 'remove')
            for patch in payload.get('patches') or []
        ):
            errors.append(path + ': incomplete length coverage requires effective program-variant '
                          'or length-scalar repairs; valid is unavailable')
        if verdict == 'unclear' and not payload.get('patches') and not any(
            item['path'] == root or item['path'].startswith(root + '/')
            for item in payload.get('unresolved') or []
        ):
            errors.append(path + ': a necessary unresolved length choice requires a scoped question')
    return errors
