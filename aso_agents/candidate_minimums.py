"""Explicit count requirements versus optional sequence-screening preferences.

Inputs are compiled job geometry and typed constraints, never request prose.
Meeting a count does not establish potency or bypass downstream evidence checks.
"""
from copy import deepcopy
import hashlib
import json
import math

from .candidate_presentation import molecule_key
from .parameter_constraints import evaluate_row_against_constraints
from .region_overlap import candidate_matches_regions, matches_constraint_filters
from .region_projection import genomic_sources, source_intervals, distinct_source_labels


def genomic_regions_for_constraint(goal, program_id, constraint):
    """Resolve a typed constraint to genomic rows without treating cDNA as a chromosome."""
    space = str(constraint.get("coordinate_space") or "genomic").lower()
    if space not in {"cdna", "transcript", "mature_transcript"}:
        return deepcopy(constraint.get("materialized_rows") or [])
    sources = genomic_sources(goal, constraint)
    if sources:
        return [row for source in sources for row in source['genomic_rows']]
    label = str(constraint.get("label") or "")
    if not label:
        return []
    matching = []
    for job in goal.get("steric_design_jobs") or []:
        if program_id and job.get("program_id") != program_id:
            continue
        filters = ((job.get("selection_meta") or {}).get("region_constraints") or {}).get("tile_constraint_filters") or []
        for item in filters:
            if not isinstance(item, dict):
                continue
            rows = item.get("rows") or []
            if rows and all(isinstance(row, (list, tuple)) and len(row) > 3
                            and str(row[3]) == label for row in rows):
                matching.append(rows)
    # Multiple different projections are ambiguous, rather than evidence that
    # any one genomic locus is the requested transcript window.
    unique = {tuple(tuple(row) for row in rows) for rows in matching}
    return deepcopy(matching[0]) if len(unique) == 1 else []


def _compiled_site_id(goal, job, metadata):
    """Share a resolved spatial predicate across program/chemistry variants."""
    filters = metadata.get("tile_constraint_filters") or []
    if (metadata.get("output_space") != "genomic"
            or not metadata.get("constraint_count") or not filters):
        return None
    assembly = (job.get("genome_assembly") or job.get("assembly")
                or goal.get("genome_assembly") or goal.get("assembly"))

    def intervals(rows):
        sources = source_intervals({"materialized_rows": rows, "genome_assembly": assembly})
        if rows and not sources:
            raise ValueError("Unresolved compiled intervals")
        return sorted(source["source_interval_id"] for source in sources)

    rows = job.get("tile_filter_rows") or job.get("region_rows") or []
    if not rows:
        return None
    try:
        predicate = [intervals(rows), job.get("tile_overlap_requirement") or "any_overlap", [
            [intervals(item.get("rows") or []),
             [intervals(blocks) for blocks in item.get("region_groups") or []],
             item.get("operator") or "union",
             item.get("aso_overlap_requirement") or "any_overlap",
             item.get("minimum_overlap_bp") or 1]
            for item in filters]]
    except (TypeError, ValueError, AttributeError):
        return None
    # Preserve reference scope and predicate order. Display labels, program IDs,
    # oligo lengths and chemistry variants do not create extra physical sites.
    reference = [job.get("species") or goal.get("species"), assembly, job.get("transcript_id")]
    return "physical_region:" + hashlib.sha256(
        json.dumps([reference, predicate], sort_keys=True).encode()).hexdigest()


def requested_region_minimums(goal):
    minimum = goal.get("min_asos_per_site")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        return []
    groups = {}
    for job in goal.get("steric_design_jobs") or []:
        if (job.get("execution_policy") or {}).get("role") == "fallback":
            continue
        group_id = job.get("program_id") or job.get("job_id")
        if not group_id:
            continue
        metadata = (job.get("selection_meta") or {}).get("region_constraints") or {}
        materialized = goal.get("materialized_region_constraints") or {}
        constraints = list(materialized.get("global") or [])
        constraints += (materialized.get("by_program") or {}).get(job.get("program_id")) or []
        positive_sites = [constraint for constraint in constraints
                          if isinstance(constraint, dict)
                          and constraint.get("source_type") == "explicit_rows"
                          and constraint.get("operator") != "exclude"]
        if positive_sites:
            # Each source interval is one site, regardless of repeated display
            # names or the number of genomic blocks in its exon projection.
            filters = metadata.get("tile_constraint_filters") or []
            for constraint in positive_sites:
                sites = genomic_sources(goal, constraint)
                if not sites:
                    sites = distinct_source_labels(source_intervals(constraint))
                    regions = genomic_regions_for_constraint(goal, job.get('program_id'), constraint)
                    # A unique compiled projection of a single source interval
                    # can contain several exons. Never distribute an aggregate
                    # projection among multiple unresolvable source intervals.
                    if len(sites) == 1 and regions:
                        sites[0]['genomic_rows'] = regions
                if not sites:
                    label = str(constraint.get('label') or constraint.get('constraint_id') or 'unresolved')
                    sites = [{'source_interval_id': 'unresolved', 'label': label}]
                    matching = [item for item in filters if isinstance(item, dict)
                        and len(item.get('rows') or []) == 1 and len(item['rows'][0]) > 3
                        and str(item['rows'][0][3]) == label]
                    if len(matching) == 1:
                        # Older runs may omit source rows. A sole named compiled
                        # genomic interval is still unambiguous; multiple blocks
                        # without source provenance cannot establish site count.
                        compiled = {**constraint, 'coordinate_space': 'genomic',
                                    'materialized_rows': matching[0]['rows']}
                        sites = genomic_sources(goal, compiled) or sites
                for source in sites:
                    name = source['label']
                    label = str(constraint.get('label') or name)
                    rows = source.get('genomic_rows') or []
                    resolved = bool(rows)
                    site_id = str(constraint.get("constraint_id") or label)
                    site_group_id = f"{group_id}:{site_id}:{source['source_interval_id']}"
                    group = groups.setdefault(site_group_id, {"group_id": site_group_id,
                        "label": name, "minimum": minimum, "site_scope": "physical_constraint",
                        "source_interval_id": source['source_interval_id'],
                        "source_row": source.get('source_row'),
                        "source_coordinate_space": source.get('coordinate_space'),
                        "selectors": []})
                    site_filter = {**constraint, 'rows':rows, 'region_groups':[rows], 'operator':'intersect'}
                    group["selectors"].append({
                        "rows": deepcopy(rows),
                        "requirement": constraint.get("aso_overlap_requirement") or "any_overlap",
                        "filters": deepcopy([*filters, site_filter] if resolved else []),
                        "chemistry_group_id": job.get("chemistry_group_id"),
                        "explicit_scope_resolved": resolved,
                    })
            continue
        explicit_scope_resolved = not (
            job.get("targeting_policy") == "explicit_user_region"
            and not metadata.get("constraint_count")
        )
        if not explicit_scope_resolved:
            from .splice_site_geometry import verified_reference_window_scope
            explicit_scope_resolved = verified_reference_window_scope(goal, job)
        site_id = _compiled_site_id(goal, job, metadata) if explicit_scope_resolved else None
        if site_id:
            group_id = site_id
        group = groups.setdefault(group_id, {"group_id": group_id,
            "label": job.get("label") or group_id, "minimum": minimum, "selectors": []})
        if site_id:
            rows = job.get("tile_filter_rows") or job.get("region_rows") or []
            group.update(site_scope="physical_constraint", label=" | ".join(dict.fromkeys(
                str(row[3]) if len(row) > 3 and row[3] else f"{row[0]}:{row[1]}-{row[2]}"
                for row in rows)))
        group["selectors"].append({
            "rows": deepcopy(job.get("tile_filter_rows") or job.get("region_rows") or []),
            "requirement": job.get("tile_overlap_requirement") or "any_overlap",
            "filters": deepcopy(metadata.get("tile_constraint_filters") or []),
            "chemistry_group_id": job.get("chemistry_group_id"),
            "explicit_scope_resolved": explicit_scope_resolved,
        })
    return list(groups.values())


def belongs_to_region(row, group, intervals_for_row):
    intervals = intervals_for_row(row, output_space="genomic")
    if not intervals:
        return False
    annotations = row.get("source_annotations") or {}
    chemistry = row.get("chemistry_group_id") or annotations.get("chemistry_group_id")
    for selector in group["selectors"]:
        if not selector.get("explicit_scope_resolved", True):
            continue
        expected = selector.get("chemistry_group_id")
        if chemistry and expected and chemistry != expected:
            continue
        # A union of typed region constraints can have different overlap rules.
        # Its aggregate "mixed" marker is not itself an overlap rule: use the
        # broad geometric envelope here, then enforce each original rule below.
        requirement = selector["requirement"]
        if requirement == "mixed":
            requirement = "any_overlap"
        if not selector["rows"] or not candidate_matches_regions(
                intervals, selector["rows"], requirement):
            continue
        if matches_constraint_filters(intervals, selector["filters"]):
            return True
    return False


def select_default_gc_supplements(passed, gc_only_rejected, constraints, groups,
                                  intervals_for_row, preferred_bounds):
    """Return indexed additions; every explicit constraint still applies.

    The caller excludes every row that failed a non-GC initial filter. Rows are
    retained only to fill a declared minimum, preferring the closest GC values.
    Later reference, conservation, screening and scoring stages still run.
    """
    if not groups:
        return [], {}
    populations = [{molecule_key(row) for row in passed
                    if belongs_to_region(row, group, intervals_for_row)} for group in groups]
    before = [len(population) for population in populations]
    ranked = []
    lower, upper = preferred_bounds
    for index, row in enumerate(gc_only_rejected):
        key = molecule_key(row)
        if key[0] == "unverified" or not key[0] or set(key[0]) - set("ACGT"):
            continue
        try:
            gc = float(row["GC"])
        except (KeyError, TypeError, ValueError):
            continue
        gc = gc / 100 if gc > 1 else gc
        if not math.isfinite(gc) or not 0 <= gc <= 1:
            continue
        keep, _ = evaluate_row_against_constraints(row, constraints)
        if not keep:
            continue
        membership = [i for i, group in enumerate(groups)
                      if belongs_to_region(row, group, intervals_for_row)]
        deviation = max(lower - gc, gc - upper, 0)
        ranked.append((deviation, index, gc, key, membership))
    retained = []
    for deviation, index, gc, key, membership in sorted(ranked):
        needed = [i for i in membership if len(populations[i]) < groups[i]["minimum"]
                  and key not in populations[i]]
        if not needed:
            continue
        row = deepcopy(gc_only_rejected[index])
        row["default_gc_preference_exception"] = {
            "reason": "explicit_region_minimum",
            "observed_gc_fraction": gc, "preferred_gc_fraction": [lower, upper],
            "distance_from_preferred_interval": deviation,
            "required_by_groups": [groups[i]["group_id"] for i in needed],
            "explicit_constraints_satisfied": True,
            "interpretation": "Outside a default GC preference; retained to meet a requested count. This does not establish binding or activity. All later evidence checks remain required.",
        }
        retained.append((index, row))
        for i in membership:
            populations[i].add(key)
    return retained, {"method": "nearest_default_gc_preference_for_explicit_minimum",
        "retained_binding_records": len(retained),
        "groups": [{"group_id": group["group_id"], "minimum": group["minimum"],
                    "before": before[i], "after": len(populations[i]),
                    "shortfall": max(0, group["minimum"] - len(populations[i]))}
                   for i, group in enumerate(groups)],
        "scope": "Initial sequence-filter stage only. Final delivery must independently satisfy every requested minimum after all downstream checks."}


def region_minimum_delivery(goal, candidate_set, binding_sets, intervals_for_row):
    minimum = goal.get("min_asos_per_site")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        return {"status": "not_requested", "groups": [], "errors": []}
    groups = requested_region_minimums(goal)
    # When the user requested a bounded selected panel, its count must
    # stand on its own. A larger scored binding-set appendix cannot fill
    # missing entries in that panel. Other binding-set deliverables retain
    # their existing complete-pool semantics.
    selected_panel_required = any(isinstance(goal.get(key), int)
        and not isinstance(goal.get(key), bool) and goal[key] > 0
        for key in ("candidate_limit", "candidate_limit_per_site", "per_site_candidate_limit"))
    if selected_panel_required and "rows" in candidate_set:
        rows = list(candidate_set.get("rows") or [])
    else:
        rows = list(candidate_set.get("rows") or [])
        rows.extend(row for group in (binding_sets or {}).get("sets") or []
                    for row in group.get("rows") or [])
    results = []
    if groups:
        for group in groups:
            keys = {molecule_key(row) for row in rows
                    if belongs_to_region(row, group, intervals_for_row)
                    and molecule_key(row)[0] != "unverified"}
            results.append({"group_id": group["group_id"], "label": group["label"],
                "minimum": group["minimum"], "distinct_molecules": len(keys),
                "shortfall": max(0, group["minimum"] - len(keys))})
    else:
        # Non-physical-region designs retain their declared native grouping.
        selection = candidate_set.get("candidate_selection") or {}
        keys = (("route_label", "site_label", "job_label", "target_region_label", "program_id")
                if selection.get("group_scope") == "route" else
                ("candidate_group_label", "program_id", "job_label", "target_region_label", "site_label", "route_label"))
        for group in selection.get("groups") or []:
            label = group.get("group_label")
            molecules = {molecule_key(row) for row in candidate_set.get("rows") or []
                         if next((row.get(key) for key in keys if row.get(key)), "ungrouped") == label
                         and molecule_key(row)[0] != "unverified"}
            results.append({"group_id": label, "minimum": minimum,
                "distinct_molecules": len(molecules), "shortfall": max(0, minimum-len(molecules))})
    errors = (["requested_per_site_scope_unresolved"] if not results else
              ["requested_per_site_candidate_count_not_delivered"]
              if any(group["shortfall"] for group in results) else [])
    return {"status": "unfulfilled" if errors else "fulfilled", "groups": results, "errors": errors,
            "count_scope": "selected_panel" if selected_panel_required and "rows" in candidate_set else "complete_binding_pool"}
