"""Execute explicit ordered, role-local rankings over an eligible design pool.

Role order is part of the request: an earlier choice is never changed to improve
a later role or a global aggregate. Sequence identity and physical-molecule
identity are separate policies. No request prose is interpreted here.
"""
from copy import deepcopy


INDIVIDUAL_CRITERION_SCHEMA = {
    "type": "object", "required": ["metric", "direction"],
    "properties": {
        "metric": {"type": "string", "enum": ["gc_fraction", "gc_percent", "length_nt",
            "longest_homopolymer", "genomic_start_1based", "mature_start_1based"]},
        "direction": {"type": "string", "enum": ["minimize", "maximize"]},
        "absolute_distance_from": {"type": "number"},
        "reference_transcript_id": {"type": "string", "minLength": 1}},
    "allOf": [{"if": {"properties": {"metric": {"const": "mature_start_1based"}}},
               "then": {"required": ["reference_transcript_id"]}}],
    "additionalProperties": False,
}

ROLE_SELECTION_SCHEMA = {
    "type": "object", "required": ["reuse_policy", "roles"],
    "properties": {
        "reuse_policy": {"type": "string", "enum": ["distinct_sequences",
            "distinct_molecules", "allow_shared_molecules"]},
        "roles": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["role_id", "program_id", "count", "criteria"],
            "properties": {
                "role_id": {"type": "string", "minLength": 1},
                "program_id": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 1},
                "criteria": {"type": "array", "minItems": 1,
                             "items": INDIVIDUAL_CRITERION_SCHEMA}},
            "additionalProperties": False}}},
    "additionalProperties": False,
    "description": (
        "Explicit sequential selection of named roles. Process roles in array order; each role "
        "ranks eligible molecules from its exact design program by its own individual criteria, "
        "in lexicographic priority order, then takes count after the requested reuse rule. "
        "Use the user's role names as role_id and create the corresponding scoped design programs. "
        "distinct_sequences forbids reusing a base sequence even under different chemistry; "
        "distinct_molecules forbids reusing sequence plus chemistry; allow_shared_molecules permits "
        "one molecule to satisfy different roles but never pads a role with duplicate records. "
        "Do not infer role order or reuse policy. Do not translate individual criteria to sums, "
        "maxima, or joint-panel objectives. Enumeration/library programs need not be selection roles."
    ),
}


def selection_scope_conflicts(goal):
    """Expose incompatible final-selection contracts without inferring intent."""
    fields = ("candidate_role_selection", "candidate_panel_objectives")
    if not isinstance(goal, dict) or any(goal.get(field) is None for field in fields):
        return {}
    conflict = {
        "code": "joint_and_sequential_ranking_scope_conflict",
        "conflicting_fields": list(fields),
        "reason": (
            "Sequential role selection commits each role's choice before the next role; "
            "joint panel optimization compares complete panels. They cannot both govern "
            "the same final selection."
        ),
        "repair_requirement": (
            "Use explicit request evidence to resolve the final-selection scope. "
            "Individual ranking for reporting or reference does not establish sequential "
            "role selection. Preserve requested ranking/export information and all hard "
            "constraints; do not discard a requested algorithm to satisfy validation. "
            "If the intended selection scope is ambiguous, request clarification."
        ),
    }
    return {field: deepcopy(conflict) for field in fields}


def reconcile_role_obligation(goal, obligation, source):
    """Bind independently quoted role requirements to existing program identities.

    An auditor does not see the draft goal. Bind only exact role identities from
    the draft's typed contract, or exact program IDs/labels; never guess a scope
    from a gene, a sequence match, a label substring or a candidate location.
    """
    if obligation is None:
        return {"status": "not_requested"}
    from jsonschema import Draft7Validator
    unresolved = {"status": "unresolved", "reason": "role_scope_or_evidence_unverified"}
    if not isinstance(obligation, dict):
        return unresolved
    quote = obligation.get("quote")
    if not isinstance(quote, str) or not quote.strip() or quote.strip().casefold() not in source.casefold():
        return unresolved
    if set(obligation) != {"reuse_policy", "roles", "quote"}:
        return unresolved
    audit_schema = deepcopy(ROLE_SELECTION_SCHEMA)
    audit_schema["properties"]["quote"] = {"type": "string", "minLength": 1}
    audit_role_schema = audit_schema["properties"]["roles"]["items"]
    audit_role_schema["properties"].pop("program_id")
    audit_role_schema["required"].remove("program_id")
    if not Draft7Validator(audit_schema).is_valid(obligation):
        return unresolved
    programs = goal.get("design_programs") or []
    if not isinstance(programs, list) or not all(isinstance(p, dict) for p in programs):
        return unresolved
    previous = goal.get("candidate_role_selection") or {}
    if previous and not Draft7Validator(ROLE_SELECTION_SCHEMA).is_valid(previous):
        return unresolved
    previous_roles = previous.get("roles", []) if isinstance(previous, dict) else []
    bindings = {}
    for role in previous_roles:
        if isinstance(role, dict):
            bindings.setdefault(role.get("role_id"), set()).add(role.get("program_id"))
    roles = obligation.get("roles")
    if not isinstance(roles, list) or not roles:
        return unresolved
    bound = []
    for role in roles:
        if not isinstance(role, dict) or set(role) != {"role_id", "count", "criteria"}:
            return unresolved
        identity = role.get("role_id")
        matches = bindings.get(identity) or {
            p.get("program_id") for p in programs if identity in {p.get("program_id"), p.get("label")}}
        if len(matches) != 1:
            return unresolved
        program = next(iter(matches))
        if sum(p.get("program_id") == program for p in programs) != 1:
            return unresolved
        bound.append({**deepcopy(role), "program_id": program})
    contract = {"reuse_policy": obligation.get("reuse_policy"), "roles": bound}
    if (not Draft7Validator(ROLE_SELECTION_SCHEMA).is_valid(contract)
            or len({r["role_id"] for r in bound}) != len(bound)):
        return unresolved
    goal["candidate_role_selection"] = contract
    return {"status": "applied", "quote": quote, "contract": deepcopy(contract)}


def select_ordered_roles(rows, goal, intervals_for_row, *, region_status="complete"):
    """Select after hard eligibility; return a panel and an auditable assignment.

    Any unresolvable role metric stops selection, rather than silently replacing
    a requested tuple with native scores. Partial feasible choices are retained
    for genuine pool/count shortfalls. The full input pool is never modified.
    """
    from jsonschema import Draft7Validator
    from .candidate_presentation import bind_candidate_program_groups, molecule_key
    from .panel_objectives import compile_individual_criteria, _display
    from .region_panel_selection import _verified_blocks

    contract = goal.get("candidate_role_selection")
    summary = {"status": "unfulfilled", "method": "ordered_role_local_lexicographic_selection",
               "selection_scope": "ordered_roles", "contract": deepcopy(contract),
               "input_row_count": len(rows), "eligible_count": 0, "selected_count": 0,
               "unselected_count": 0, "roles": [],
               "scientific_interpretation": "Requested sequence/coordinate priorities, not experimental potency."}

    def unresolved(reason):
        return [], {**summary, "status": "unresolved", "reason": reason}

    if not Draft7Validator(ROLE_SELECTION_SCHEMA).is_valid(contract):
        return unresolved("invalid_role_selection_contract")
    roles = contract["roles"]
    if len({r["role_id"] for r in roles}) != len(roles):
        return unresolved("duplicate_role_identity")
    programs = goal.get("design_programs") or []
    if (not isinstance(programs, list)
            or not all(isinstance(p, dict) and isinstance(p.get("program_id"), str) and p["program_id"].strip() for p in programs)
            or len({p["program_id"] for p in programs}) != len(programs)):
        return unresolved("program_identity_unresolved")
    program_map = {p["program_id"]: p for p in programs}
    if any(role["program_id"] not in program_map for role in roles):
        return unresolved("role_program_identity_unresolved")
    if region_status not in {"complete", "not_required"}:
        return unresolved("hard_region_eligibility_unresolved")
    if selection_scope_conflicts(goal):
        return unresolved("joint_and_sequential_ranking_scope_conflict")
    if any(goal.get(k) is not None for k in ("candidate_limit_per_site", "per_site_candidate_limit", "min_asos_per_site")):
        return unresolved("physical_site_counts_with_ordered_roles_not_supported")
    if goal.get("candidate_selection_policy") == "diverse_target_sites":
        return unresolved("soft_diversity_and_explicit_role_ranking_scope_conflict")
    minimum, maximum = goal.get("candidate_minimum", 1), goal.get("candidate_limit")
    if (type(minimum) is not int or minimum < 0 or
            (maximum is not None and (type(maximum) is not int or maximum < 1 or minimum > maximum))):
        return unresolved("global_count_contract_invalid")
    gap = goal.get("candidate_minimum_genomic_gap_nt", 0)
    if type(gap) is not int or gap < 0:
        return unresolved("genomic_gap_contract_invalid")
    nonoverlap = (goal.get("candidate_selection_policy") == "nonoverlapping_target_sites"
                  or "candidate_minimum_genomic_gap_nt" in goal)
    wanted_programs = {role["program_id"] for role in roles}
    pools = {pid: {} for pid in wanted_programs}
    global_blocks, memberships = {}, {}
    if not all(isinstance(row, dict) for row in rows):
        return unresolved("candidate_row_invalid")
    for row in bind_candidate_program_groups(rows, goal):
        program = row.get("program_id")
        if program not in program_map:
            return unresolved("candidate_program_identity_unresolved")
        if program not in wanted_programs:
            continue
        key = molecule_key(row)
        if not key[0] or set(key[0]) - set("ACGT") or key[1] == "null":
            return unresolved("candidate_molecule_identity_unresolved")
        entry = pools[program].setdefault(key, {"rows": []})
        entry["rows"].append((row, set()))
        memberships.setdefault(key, set()).add(program)
        if nonoverlap:
            blocks = _verified_blocks(row, intervals_for_row)
            if blocks is None:
                return unresolved("candidate_physical_geometry_unresolved")
            global_blocks.setdefault(key, set()).update(blocks)
    program_bounds = {}
    for pid in wanted_programs:
        p = program_map[pid]
        low, high = p.get("candidate_minimum", 0), p.get("candidate_limit")
        if (type(low) is not int or low < 0 or
                (high is not None and (type(high) is not int or high < 1 or low > high))):
            return unresolved("program_count_contract_invalid")
        program_bounds[pid] = (low, high)
    summary.update(eligible_count=len(memberships), unselected_count=len(memberships))

    # Compile every role before selecting, so an unavailable later criterion
    # cannot be hidden behind a successful earlier role or a native fallback.
    compiled = []
    for role in roles:
        pool = pools[role["program_id"]]
        keys, entries = list(pool), list(pool.values())
        try:
            columns = compile_individual_criteria(keys, entries, role["criteria"],
                goal.get("isoform_transcripts") or {}, lambda row: _verified_blocks(row, intervals_for_row))
        except ValueError as exc:
            return unresolved(str(exc))
        values = [tuple(column[i] for column in columns) for i in range(len(keys))]
        order = sorted(range(len(keys)), key=lambda i: tuple(
            -value if criterion["direction"] == "maximize" else value
            for value, criterion in zip(values[i], role["criteria"])))
        compiled.append((role, keys, entries, values, order))

    selected, used_sequences = {}, set()
    for role, keys, entries, values, order in compiled:
        record = {"role_id": role["role_id"], "program_id": role["program_id"],
                  "requested_count": role["count"], "eligible_count": len(keys),
                  "criteria": deepcopy(role["criteria"]), "selected": [], "skipped": [],
                  "remaining_ties": "stable_input_order_after_all_requested_priorities"}
        for rank, index in enumerate(order, 1):
            key = keys[index]
            reason = None
            if contract["reuse_policy"] == "distinct_sequences" and key[0] in used_sequences:
                reason = "sequence_already_assigned"
            elif contract["reuse_policy"] == "distinct_molecules" and key in selected:
                reason = "molecule_already_assigned"
            elif key not in selected:
                if maximum is not None and len(selected) >= maximum:
                    reason = "global_maximum"
                elif any(high is not None and sum(pid in memberships[k] for k in selected) >= high
                         for pid, (_, high) in program_bounds.items() if pid in memberships[key]):
                    reason = "program_maximum"
                elif nonoverlap and any(
                        a[0] == b[0] and max(a[1], b[1]) - min(a[2], b[2]) < gap
                        for previous in selected for a in global_blocks[key] for b in global_blocks[previous]):
                    reason = "physical_nonoverlap_or_genomic_gap"
            identity = {"sequence": key[0], "chemistry_identity": key[1]}
            if reason:
                record["skipped"].append({**identity, "role_rank": rank, "reason": reason})
                continue
            assignment = {"role_id": role["role_id"], "program_id": role["program_id"],
                          "role_rank": rank, "criteria": deepcopy(role["criteria"]),
                          "values": [_display(v) for v in values[index]],
                          "exact_values": [_display(v, True) for v in values[index]]}
            if key not in selected:
                chosen = deepcopy(entries[index]["rows"][0][0])
                chosen["role_assignments"] = []
                selected[key] = chosen
            selected[key]["role_assignments"].append(deepcopy(assignment))
            used_sequences.add(key[0])
            record["selected"].append({**identity, **assignment})
            if len(record["selected"]) == role["count"]:
                break
        record["selected_count"] = len(record["selected"])
        record["shortfall_count"] = role["count"] - record["selected_count"]
        summary["roles"].append(record)
    summary["program_counts"] = [{"program_id": pid, "minimum": low, "maximum": high,
        "selected_count": sum(pid in memberships[k] for k in selected),
        "shortfall_count": max(0, low - sum(pid in memberships[k] for k in selected))}
        for pid, (low, high) in sorted(program_bounds.items())]
    summary.update(selected_count=len(selected), unselected_count=len(memberships) - len(selected),
                   global_minimum=minimum, global_maximum=maximum,
                   global_shortfall_count=max(0, minimum - len(selected)),
                   reuse_policy=contract["reuse_policy"])
    if (len(selected) >= minimum and not any(r["shortfall_count"] for r in summary["roles"])
            and not any(p["shortfall_count"] for p in summary["program_counts"])):
        summary["status"] = "complete"
    return list(selected.values()), summary
