"""Coordinate-explicit union/intersection of materialized nominated regions."""
from .region_projection import source_intervals, distinct_source_labels


def merge_intervals(intervals):
    result = []
    for start, end in sorted(set(tuple(interval) for interval in intervals)):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def intersect_intervals(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            result.append([start, end])
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return merge_intervals(result)


def nominated_region_interval_summary(goal):
    materialized = goal.get("materialized_region_constraints") or {}
    constraints = [(None, item) for item in materialized.get("global") or []]
    constraints += [(program, item) for program, items in (materialized.get("by_program") or {}).items()
                    for item in items]
    spaces = {}
    for program, constraint in constraints:
        if constraint.get("source_type") != "explicit_rows" or constraint.get("operator") == "exclude":
            continue
        for source in source_intervals(constraint):
            row = source['source_row']
            key = (program, constraint.get("coordinate_space") or "genomic",
                   constraint.get("genome_assembly") or constraint.get("assembly"), str(row[0]),
                   str(row[5]) if len(row) > 5 else ".")
            spaces.setdefault(key, {}).setdefault(source['source_interval_id'], source)
    results = []
    for (program, space, assembly, reference, strand), sources in spaces.items():
        if len(sources) < 2:
            continue
        named = distinct_source_labels(list(sources.values()))
        intervals = {source['label']: [source['source_row'][1:3]] for source in named}
        union = merge_intervals(row for rows in intervals.values() for row in rows)
        intersection = next(iter(intervals.values()))
        for rows in list(intervals.values())[1:]:
            intersection = intersect_intervals(intersection, rows)
        results.append({"program_id": program, "coordinate_space": space, "assembly": assembly,
                        "reference_id": reference, "strand": strand, "coordinate_convention": "0based_half_open",
                        "source_regions": intervals, "union_intervals": union,
                        "source_interval_ids": {source['label']: source['source_interval_id'] for source in named},
                        "union_length_nt": sum(b - a for a, b in union),
                        "intersection_of_all_regions": intersection,
                        "intersection_length_nt": sum(b - a for a, b in intersection),
                        "scope": "Nominated source-region geometry; not candidate counts, genomic projection or binding evidence."})
    return results
