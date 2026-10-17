"""Evidence-bound finite length coverage in existing broad design-goal audits."""
from copy import deepcopy


PROGRAM_LENGTH_AUDIT_INSTRUCTIONS = """
Execution contract: one design_program executes one scalar oligo length. A program
named a series, library or comparison does not execute multiple lengths: labels,
objectives and output-table requests cannot expand its length at execution time.
An absent program.length permits a single resolved/default length, never an
implicit length series. Every explicitly requested finite length set or inclusive
range therefore needs one otherwise-equivalent program per value for each
applicable intervention branch. A domain shared across several branches is still
explicitly requested for each branch; it does not qualify as not_requested.
Determine requested_lengths from the user evidence before choosing the verdict.
Do not exclude a length domain because it was phrased as a series or a full-library
requirement instead of as a branch-specific scalar. Those are execution variants.
For each program_length_checks_required entry, fill its check_id under
program_length_checks. Determine the complete explicitly requested oligo-length
domain for that intervention branch from evidence_sources: enumerate a finite set
or every integer of a bounded inclusive range. Preserve branch-specific differences.
When one existing program is a template for a finite length domain, prefer verdict
expand: supply requested_lengths and represented_program_ids=[that program_id].
Do not write length/ID/array-structure patches for that expansion. The executor
keeps the original ID, sets its scalar length and creates exact copies for the
remaining values with unique IDs. Other supported leaf repairs are applied first
and copied to every variant. This avoids retyping or omitting branch constraints.
The same compilation applies to an evidence-bound incomplete finding that names
only its current template and supplies no effective program length, ID or array
structure repair. Identifying the finite missing domain is sufficient; another
command is not needed. Explicit multi-program repairs keep their existing path.
Expansion is explicit, evidence-bound and subject to the ordinary repair/verify
loop; it does not infer a domain from a label. If the branch already has multiple
variants, use complete or incomplete with actual IDs and canonical repairs instead.
Do not silently pick a default length or accept lengths mentioned only in labels,
objectives, chemistry or notes. List the actual same-branch represented_program_ids
and compare their typed design_programs[*].length values with the requested domain.
For complete, assess the proposed envelope AFTER applying your canonical patches:
list every actual same-branch program ID, including any newly added variants.
Those IDs must exist after successful patch application; naming them is not a repair.
Complete requires exactly one program per requested length and preservation of the
branch's executable facts (only IDs, labels, objectives and lengths may differ).
For incomplete, describe the CURRENT envelope: list only its existing same-branch
IDs. Use incomplete for
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
            'verdict': {'type':'string','enum':['not_requested','complete','incomplete','expand','unclear']},
            'requested_lengths': {'type':'array','uniqueItems':True,
                                  'items':{'type':'integer','minimum':1}},
            'represented_program_ids': {'type':'array','uniqueItems':True,
                'description': 'Complete refers to actual IDs after canonical patches; '
                               'incomplete refers to existing IDs before patches. '
                               'Invented or unapplied IDs never establish coverage.',
                'items':{'type':'string','minLength':1}},
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


def compile_program_length_expansions(payload, obligations, span_map, *, projected_programs):
    """Compile an explicit single-template domain without changing branch facts.

    The caller first validates the response schema and previews ordinary patches
    with the canonical evidence/path/value executor. Generated patches then pass
    that same executor and complete-coverage checks. No request prose is parsed.
    """
    result = deepcopy(payload)
    checks = result['program_length_checks']
    occupied = {root: {p.get('program_id') for p in programs if isinstance(p, dict)}
                for root, programs in projected_programs.items()}
    original_patches = payload.get('patches') or []
    for obligation in obligations:
        finding = checks[obligation['check_id']]
        if finding['verdict'] not in {'expand', 'incomplete'}:
            continue
        if finding['verdict'] == 'incomplete':
            root = obligation['programs_pointer']
            if finding['represented_program_ids'] != [obligation['program_id']]:
                continue
            # Respect an explicit variant repair rather than duplicating it.
            # No-op edits were already removed by the ordinary normalizer.
            def changes_program_structure(patch):
                path = patch['path']
                if path == root or root.startswith(path + '/'):
                    return True
                if path.startswith(root + '/'):
                    suffix = path[len(root) + 1:].split('/')
                    return len(suffix) == 1 or suffix[1] in {'length', 'program_id'}
                return False

            if any(changes_program_structure(patch) for patch in original_patches):
                continue
        prefix = '$.program_length_checks.' + obligation['check_id'] + ': '

        def require(condition, message):
            if not condition:
                raise ValueError(prefix + message)

        domain = finding['requested_lengths']
        identity = obligation['program_id']
        root = obligation['programs_pointer']
        span = span_map.get(finding.get('evidence_span_id'))
        require(span and span['evidence_id'] in finding['evidence_refs'],
                'expansion requires a span from its cited evidence source')
        require(domain and all(type(n) is int and n > 0 for n in domain)
                and len(set(domain)) == len(domain),
                'expansion requires a nonempty domain of distinct positive integer lengths')
        require(finding['represented_program_ids'] == [identity],
                'expand represents exactly its existing single template program')
        programs = projected_programs.get(root, [])
        identifiers = [p.get('program_id') for p in programs if isinstance(p, dict)]
        require(len(identifiers) == len(set(identifiers)) and identity in identifiers,
                'the template must still exist among uniquely identified programs')
        # A structural edit can move positional pointers; a length or ID edit
        # competes with this operation. The model must choose one repair form.
        for patch in original_patches:
            path = patch['path']
            if path == root or root.startswith(path + '/'):
                require(False, 'expand cannot accompany an ancestor or collection replacement')
            if path.startswith(root + '/'):
                suffix = path[len(root) + 1:].split('/')
                require(len(suffix) > 1 and suffix[1] not in {'length', 'program_id'},
                        'expand cannot accompany program structure, ID or length patches in its collection')
        index = next(i for i, p in enumerate(programs) if p.get('program_id') == identity)
        template = programs[index]

        def facts(program):
            return {k: v for k, v in program.items()
                    if k not in {'program_id', 'label', 'objective', 'length'}}

        require(not any(p.get('program_id') != identity and facts(p) == facts(template)
                        for p in programs if isinstance(p, dict)),
                'equivalent variants already exist; audit their explicit coverage instead of expanding one')
        proof = {key: deepcopy(finding[key])
                 for key in ('evidence_refs', 'evidence_span_id')}
        proof['reason'] = 'Compile the explicitly audited length domain: ' + finding['reason']
        retained = template.get('length')
        retained = retained if type(retained) is int and retained in domain else domain[0]
        if type(template.get('length')) is not int or template.get('length') != retained:
            result['patches'].append({**proof, 'op': 'replace' if 'length' in template else 'add',
                                      'path': root + f'/{index}/length', 'value': retained})
        represented = [identity]
        for length in domain:
            if length == retained:
                continue
            base = f'{identity}__length_{length}'
            new_id, counter = base, 2
            while new_id in occupied[root]:
                new_id, counter = f'{base}_{counter}', counter + 1
            occupied[root].add(new_id)
            variant = {**deepcopy(template), 'program_id': new_id, 'length': length}
            result['patches'].append({**deepcopy(proof), 'op': 'add', 'path': root + '/-',
                                      'value': variant})
            represented.append(new_id)
        finding.update(verdict='complete', represented_program_ids=represented)
    return result


def program_length_coverage_errors(payload, obligations, span_map, *, projected_programs=None):
    """Compare coverage to actual variants, including successfully previewed repairs.

    The caller must apply canonical patches with the ordinary evidence/path/value
    checks before supplying projected programs. This function never interprets or
    executes model-authored patch dictionaries itself.
    """
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
        if verdict == 'expand':
            errors.append(path + ': expansion must be compiled and validated before it establishes coverage')
            continue
        if verdict == 'not_requested':
            if domain or ids:
                errors.append(path + ': not_requested requires empty length and program arrays')
            continue
        if verdict != 'unclear' and not domain:
            errors.append(path + ': an explicit finite requested domain must be nonempty')
        available = obligation['available_program_lengths']
        projected = None
        if verdict == 'complete' and projected_programs is not None:
            projected = projected_programs.get(obligation['programs_pointer'], [])
            available = {p.get('program_id'): p.get('length') for p in projected
                         if isinstance(p, dict)}
            identifiers = [p.get('program_id') for p in projected if isinstance(p, dict)]
            if len(identifiers) != len(set(identifiers)):
                errors.append(path + ': proposed program IDs must be unique')
        if any(identifier not in available for identifier in ids):
            errors.append(path + ': represented program IDs must exist in the assessed '
                          'envelope; names alone or rejected patches do not create variants')
            continue
        if verdict == 'complete':
            observed = [available[identifier] for identifier in ids]
            if (obligation['program_id'] not in ids or len(observed) != len(domain)
                or any(type(value) is not int for value in observed)
                or sorted(observed) != sorted(domain)):
                errors.append(path + ': complete requires the current program and exactly one '
                              'typed program.length for every requested value; notes and defaults do not count')
            if projected is not None and obligation['program_id'] in available:
                members = {p['program_id']: p for p in projected if isinstance(p, dict)
                           and isinstance(p.get('program_id'), str)}
                def branch_facts(program):
                    return {key: value for key, value in program.items()
                            if key not in {'program_id', 'label', 'objective', 'length'}}
                expected = branch_facts(members[obligation['program_id']])
                if any(branch_facts(members[identifier]) != expected for identifier in ids):
                    errors.append(path + ': complete length variants must preserve the '
                                  'same executable branch facts; unrelated or altered '
                                  'branches cannot supply missing lengths')
        root = obligation['programs_pointer']
        group_pointers = {obligation['program_pointer']} | {
            obligation['available_program_pointers'][identifier] for identifier in ids
            if identifier in obligation['available_program_pointers']}
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
