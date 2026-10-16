"""Apply materialized user region constraints to the final physical panel."""

from .region_overlap import matches_constraint_filters
from .candidate_minimums import genomic_regions_for_constraint
from .candidate_presentation import bind_candidate_program_groups
from .region_projection import genomic_sources


def _filters(goal, program_id, constraints):
    if not isinstance(constraints, list):
        return None
    filters = []
    for constraint in constraints:
        if not isinstance(constraint, dict):
            return None
        regions = genomic_regions_for_constraint(goal, program_id, constraint)
        if not isinstance(regions, list) or not regions:
            return None
        filters.append({"rows": regions,
                        "region_groups": [item["genomic_rows"] for item in genomic_sources(goal, constraint)],
                        "operator": constraint.get("operator") or "union",
                        "aso_overlap_requirement": constraint.get("aso_overlap_requirement") or "any_overlap",
                        "minimum_overlap_bp": constraint.get("minimum_overlap_bp") or 1})
    return filters


def filter_final_rows_by_regions(rows, goal, intervals_for_row):
    """Reapply global and resolved program rules, including on aggregate pools.

    Program identity comes from persisted metadata or a unique possible producer,
    never from the candidate's location. Unknown identity/materialization stays
    explicitly unresolved, rather than being interpreted as no applicable rule.
    """
    materialized = goal.get("materialized_region_constraints") or {}
    source = materialized.get("global") or []
    by_program = {str(key): value for key, value in (materialized.get("by_program") or {}).items() if value}
    receipt = {"input_count": len(rows), "retained_count": len(rows), "excluded_count": 0,
               "scope": "global_and_program"}
    if not source and not by_program:
        return list(rows), {**receipt, "status": "not_required"}
    global_filters = _filters(goal, None, source)
    if global_filters is None:
        return list(rows), {**receipt, "status": "materialization_incomplete"}
    local_filters = {key: _filters(goal, key, value) for key, value in by_program.items()}
    programs = {str(item['program_id']): item for item in goal.get('design_programs') or []
                if isinstance(item, dict) and item.get('program_id')}
    for key, modality in [('steric_design_jobs', 'steric'), ('gapmer_design_jobs', 'gapmer')]:
        for job in goal.get(key) or []:
            if job.get('program_id'):
                programs.setdefault(str(job['program_id']), {'program_id': str(job['program_id']), 'modality': modality})
    for key in by_program:
        programs.setdefault(key, {'program_id': key})
    bound = bind_candidate_program_groups(rows, {**goal, 'design_programs': list(programs.values())}) if by_program else rows
    kept, excluded, unresolved = [], [], []
    for index, row in enumerate(bound):
        program_id = str(row.get('program_id') or '')
        if by_program and not program_id and len(programs) == 1 and not any(
                row.get(key) for key in ('program_id', 'job_id', 'job_label', 'target_region_label')):
            only_id, only = next(iter(programs.items()))
            if not row.get('source_modality') or only.get('modality') == row['source_modality']:
                program_id = only_id
                row = {**row, 'program_id': program_id}
        filters = global_filters
        if by_program and program_id not in programs:
            unresolved.append({'row_index': index, 'reason': 'program_scope_unresolved'})
            kept.append(rows[index])
            continue
        elif program_id in local_filters:
            if local_filters[program_id] is None:
                unresolved.append({'row_index': index, 'program_id': program_id,
                                   'reason': 'materialization_incomplete'})
                kept.append(rows[index])
                continue
            else:
                # Match the compiler's ordered global + program Boolean rules.
                filters = [*global_filters, *local_filters[program_id]]
        intervals = intervals_for_row(row, output_space="genomic")
        if not filters or (intervals and matches_constraint_filters(intervals, filters)):
            kept.append(row)
        else:
            excluded.append(row)
    return kept, {**receipt, "status": "unresolved" if unresolved else "complete",
                  "retained_count": len(kept), "excluded_count": len(excluded),
                  "constraint_ids": [item.get("constraint_id") for item in source],
                  "program_constraint_ids": {key: [item.get('constraint_id') for item in value if isinstance(item, dict)]
                                             if isinstance(value, list) else []
                                             for key, value in by_program.items()},
                  "unresolved_rows": unresolved,
                  "unresolved_scope": "Rows with unknown program scope or incomplete materialization are retained as unverified."}


def filter_final_rows_by_global_regions(rows, goal, intervals_for_row):
    """Compatibility entry point; the gate now includes applicable program rules."""
    return filter_final_rows_by_regions(rows, goal, intervals_for_row)
