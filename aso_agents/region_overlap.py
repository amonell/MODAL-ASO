"""Executable overlap semantics for half-open candidate and feature intervals."""
from numbers import Integral
from typing import Any, Dict, Iterable, List, Tuple


def interval_overlap_summary(start: int, end: int, feature_start: int, feature_end: int,
                             *, coordinate_convention: str) -> dict:
    """Report intersection separately from either direction of containment.

    Coordinates must already share a reference and frame. Explicitly choose
    'inclusive' (including signed relative offsets) or 'half_open'. Each input
    must describe a nonempty interval. This does not map reference frames or
    establish biological binding.
    """
    if coordinate_convention not in {'inclusive', 'half_open'}:
        raise ValueError('Choose an explicit inclusive or half_open coordinate convention')
    values = (start, end, feature_start, feature_end)
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
        raise TypeError('Interval coordinates must be integers')
    start, end, feature_start, feature_end = map(int, values)
    adjustment = 1 if coordinate_convention == 'inclusive' else 0
    end_exclusive, feature_end_exclusive = end + adjustment, feature_end + adjustment
    if start >= end_exclusive or feature_start >= feature_end_exclusive:
        raise ValueError('Each interval must be nonempty and ordered in the declared convention')
    left, right = max(start, feature_start), min(end_exclusive, feature_end_exclusive)
    overlap = max(0, right - left)
    return {
        'coordinate_convention': coordinate_convention,
        'overlap_interval': [left, right - adjustment] if overlap else None,
        'overlap_nt': overlap, 'any_overlap': overlap > 0,
        'contains_feature': interval_matches(start, end_exclusive, feature_start,
                                             feature_end_exclusive, 'contains_feature'),
        'within_feature': interval_matches(start, end_exclusive, feature_start,
                                           feature_end_exclusive, 'full_containment'),
    }


def normalize_constraint_operator(value):
    """Candidate predicates also support include/exclude, unlike BED merging."""
    text = str(value or "").strip().lower()
    return text if text in {"union", "include", "intersect", "exclude"} else None


def interval_matches(start: int, end: int, feature_start: int, feature_end: int,
                     requirement: str = "any_overlap", minimum_overlap_bp: int = 1) -> bool:
    overlap = max(0, min(end, feature_end) - max(start, feature_start))
    if overlap < minimum_overlap_bp:
        return False
    if requirement == "full_containment":
        return feature_start <= start and end <= feature_end
    if requirement == "contains_feature":
        return start <= feature_start and feature_end <= end
    if requirement == "any_overlap":
        return True
    raise ValueError(f"Unsupported ASO overlap requirement: {requirement}")


def candidate_matches_regions(intervals, rows, requirement='any_overlap', minimum_overlap_bp=1):
    """Full containment applies to every block in a matching coordinate space."""
    intervals = list(intervals)
    if requirement == 'full_containment':
        groups = {}
        for row in rows:
            groups.setdefault(str(row[0]).removeprefix('chr'), []).append(row)
        return candidate_matches_region_groups(intervals, list(groups.values()), requirement, minimum_overlap_bp)
    for row in rows:
        blocks = {(start, end) for chrom, start, end in intervals
                  if str(chrom).removeprefix('chr') == str(row[0]).removeprefix('chr')}
        if not blocks:
            continue
        matches = [interval_matches(start, end, int(row[1]), int(row[2]), requirement, minimum_overlap_bp)
                   for start, end in blocks]
        if all(matches) if requirement == 'full_containment' else any(matches):
            return True
    return False


def candidate_matches_region_groups(intervals, groups, requirement='any_overlap', minimum_overlap_bp=1):
    """Match one physical source interval, even when exons split its projection.

    Unrelated nominated sites remain separate groups. Coverage is measured on
    unions of bases, so duplicate placements cannot inflate overlap counts.
    """
    def union(blocks):
        merged = []
        for start, end in sorted(set(blocks)):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged
    intervals = list(intervals)
    for group in groups:
        chromosomes = {str(row[0]).removeprefix('chr') for row in group}
        if len(chromosomes) != 1:
            continue
        chrom = next(iter(chromosomes))
        blocks = union((int(a), int(b)) for c, a, b in intervals if str(c).removeprefix('chr') == chrom)
        features = union((int(row[1]), int(row[2])) for row in group)
        overlap = sum(b-a for a,b in union((max(a,c), min(b,d)) for a,b in blocks for c,d in features if max(a,c)<min(b,d)))
        if overlap < minimum_overlap_bp:
            continue
        if requirement == 'full_containment' and overlap != sum(b-a for a,b in blocks):
            continue
        if requirement == 'contains_feature' and overlap != sum(b-a for a,b in features):
            continue
        if requirement not in {'any_overlap', 'full_containment', 'contains_feature'}:
            raise ValueError(f'Unsupported ASO overlap requirement: {requirement}')
        return True
    return False


def matches_constraint_filters(intervals: Iterable[Tuple[str, int, int]],
                               filters: List[Dict[str, Any]]) -> bool:
    """Apply the saved union/intersection/exclusion policy to one candidate."""
    intervals = list(intervals)
    observations = []
    for constraint in filters:
        requirement = constraint.get("aso_overlap_requirement") or "any_overlap"
        minimum = int(constraint.get("minimum_overlap_bp") or 1)
        matches = (candidate_matches_region_groups(intervals, constraint['region_groups'], requirement, minimum)
                   if constraint.get('region_groups') else
                   candidate_matches_regions(intervals, constraint.get('rows') or [], requirement, minimum))
        observations.append(matches)
    return combine_constraint_matches(observations, filters)


def combine_constraint_matches(observations, filters):
    """One Boolean execution policy shared by tiling and intake diagnostics."""
    result = None
    for matches, constraint in zip(observations, filters):
        operator = constraint.get("operator") or "union"
        if result is None:
            result = not matches if operator == "exclude" else matches
        elif operator in {"intersect", "intersection"}:
            result = result and matches
        elif operator == "exclude":
            result = result and not matches
        else:
            result = result or matches
    return True if result is None else result


def _explicit_interval_geometry(constraint, pointer, length):
    """Describe each declared BED0 interval, without inferring final feasibility."""
    result = []
    source = constraint.get('intervals_bed0_half_open') or constraint.get('rows') or []
    if not isinstance(source, list):
        return result
    requirement = constraint.get('aso_overlap_requirement') or 'any_overlap'
    minimum = constraint.get('minimum_overlap_bp', 1)
    for index, row in enumerate(source):
        if isinstance(row, dict):
            start, end = row.get('start'), row.get('end')
            reference = row.get('chrom') or row.get('transcript_id')
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            reference, start, end = row[:3]
        else:
            continue
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            continue
        width = end - start
        known = type(length) is int and length > 0 and type(minimum) is int and minimum > 0
        feasible = None
        if known and requirement in {'full_containment', 'contains_feature', 'any_overlap'}:
            feasible = min(length, width) >= minimum
            if requirement == 'full_containment':
                feasible = feasible and length <= width
            elif requirement == 'contains_feature':
                feasible = feasible and width <= length
        result.append({
            'constraint_pointer': pointer, 'interval_index': index,
            'reference': reference, 'coordinate_space': constraint.get('coordinate_space'),
            'start0': start, 'end0_exclusive': end, 'interval_length_nt': width,
            'candidate_length_nt': length if type(length) is int and length > 0 else None,
            'aso_overlap_requirement': requirement, 'minimum_overlap_bp': minimum,
            'single_interval_relation_possible': feasible,
            'operator': constraint.get('operator') or 'union',
        })
    return result


def constraint_execution_summary(goal):
    """Expose executable consequences to the existing semantic intake audit.

    This describes the parsed logic; it does not infer intent or change any
    operator. A union can be legitimate. The existing evidence-bound audit
    decides whether its actual behavior matches the user's request.
    """
    scopes = [("", goal)] + [(f"/design_programs/{i}", program)
        for i, program in enumerate(goal.get("design_programs") or [])
        if isinstance(program, dict)]
    summaries = []
    for scope, container in scopes:
        filters = container.get("region_constraints") or []
        if not isinstance(filters, list) or not filters or not all(isinstance(c, dict) for c in filters):
            continue
        indices = range(len(filters))
        pointers = [f"{scope}/region_constraints/{i}" for i in indices]
        summaries.append({
            "scope": scope or "/", "constraint_pointers": pointers,
            "allows_candidate_matching_no_constraints": combine_constraint_matches([False]*len(filters), filters),
            "constraints_allowed_to_fail_when_all_others_match": [pointers[i] for i in indices
                if combine_constraint_matches([j != i for j in indices], filters)],
            "constraints_sufficient_alone_for_admission": [pointers[i] for i in indices
                if combine_constraint_matches([j == i for j in indices], filters)],
            "explicit_interval_geometry": [item for i, constraint in enumerate(filters)
                for item in _explicit_interval_geometry(
                    constraint, pointers[i], container.get('length', goal.get('length')))],
            "geometry_scope": "Each calculation concerns one explicit BED0 half-open interval and the declared "
                "candidate length. It does not infer feasibility of the entire design: union alternatives, "
                "exclusions, multi-block projections, unresolved features and coordinate origins still matter. "
                "Check an impossible relation against the quoted request; never relax the request merely to obtain candidates.",
        })
    return summaries


def effective_program_constraint_summary(goal):
    """Show inherited predicate order without changing or inferring constraints.

    A program's local list is appended to the global list by native execution.
    Auditing either list alone hides both unintended global restrictions and
    local unions that broaden earlier restrictions. These are Boolean diagnostics,
    not proof that a sequence or a biological site exists.
    """
    global_filters = goal.get('region_constraints') or []
    if not isinstance(global_filters, list) or not all(isinstance(c, dict) for c in global_filters):
        return []
    result = []
    for index, program in enumerate(goal.get('design_programs') or []):
        if not isinstance(program, dict):
            continue
        local = program.get('region_constraints') or []
        if not isinstance(local, list) or not all(isinstance(c, dict) for c in local):
            continue
        filters = global_filters + local
        if not filters:
            continue
        scope = f'/design_programs/{index}'
        inherited = [f'/region_constraints/{i}' for i in range(len(global_filters))]
        own = [f'{scope}/region_constraints/{i}' for i in range(len(local))]
        pointers = inherited + own
        result.append({
            'program_scope': scope, 'program_id': program.get('program_id'),
            'inherited_constraint_pointers': inherited,
            'local_constraint_pointers': own,
            'ordered_predicates': [
                {'constraint_pointer': pointer, 'operator': rule.get('operator') or 'union',
                 'aso_overlap_requirement': rule.get('aso_overlap_requirement') or 'any_overlap',
                 'minimum_overlap_bp': rule.get('minimum_overlap_bp') or 1}
                for pointer, rule in zip(pointers, filters)],
            'allows_candidate_matching_no_constraints': combine_constraint_matches([False]*len(filters), filters),
            'constraints_allowed_to_fail_when_all_others_match': [pointer
                for i, pointer in enumerate(pointers)
                if combine_constraint_matches([j != i for j in range(len(filters))], filters)],
            'constraints_sufficient_alone_for_admission': [pointer
                for i, pointer in enumerate(pointers)
                if combine_constraint_matches([j == i for j in range(len(filters))], filters)],
            'explicit_interval_geometry': [item
                for pointer, rule in zip(pointers, filters)
                for item in _explicit_interval_geometry(rule, pointer, program.get('length', goal.get('length')))],
        })
    return result
