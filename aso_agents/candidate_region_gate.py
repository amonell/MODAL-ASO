"""Apply materialized user region constraints to the final physical panel."""

from .region_overlap import matches_constraint_filters
from .candidate_minimums import genomic_regions_for_constraint
from .region_projection import genomic_sources


def filter_final_rows_by_global_regions(rows, goal, intervals_for_row):
    """Keep only candidates that satisfy every compiled global region rule.

    Earlier tiling jobs can include an ungrouped scored pool alongside
    program-scoped pools. The final candidate inventory must still obey the
    global user constraint, including its minimum overlap. Incomplete region
    materialization is reported rather than silently interpreted as no rule.
    """
    materialized = goal.get("materialized_region_constraints") or {}
    source = materialized.get("global") if isinstance(materialized, dict) else None
    if not isinstance(source, list) or not source:
        return list(rows), {"status": "not_required", "input_count": len(rows),
                            "retained_count": len(rows), "excluded_count": 0}
    filters = []
    for constraint in source:
        if not isinstance(constraint, dict):
            return list(rows), {"status": "materialization_incomplete", "input_count": len(rows),
                                "retained_count": len(rows), "excluded_count": 0}
        regions = genomic_regions_for_constraint(goal, None, constraint)
        if not isinstance(regions, list) or not regions:
            return list(rows), {"status": "materialization_incomplete", "input_count": len(rows),
                                "retained_count": len(rows), "excluded_count": 0}
        filters.append({"rows": regions,
                        "region_groups": [item["genomic_rows"] for item in genomic_sources(goal, constraint)],
                        "operator": constraint.get("operator") or "union",
                        "aso_overlap_requirement": constraint.get("aso_overlap_requirement") or "any_overlap",
                        "minimum_overlap_bp": constraint.get("minimum_overlap_bp") or 1})
    kept, excluded = [], []
    for row in rows:
        intervals = intervals_for_row(row, output_space="genomic")
        if intervals and matches_constraint_filters(intervals, filters):
            kept.append(row)
        else:
            excluded.append(row)
    return kept, {"status": "complete", "input_count": len(rows),
                  "retained_count": len(kept), "excluded_count": len(excluded),
                  "constraint_ids": [item.get("constraint_id") for item in source]}
