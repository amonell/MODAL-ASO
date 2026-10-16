"""Compile explicit panel priorities and optimize them without scalar weighting.

Only verified sequence/coordinate values participate. Native ranks break ties
after all requested criteria; they never substitute for an unavailable metric.
"""
from copy import deepcopy
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
import re

from .gapmer_selectivity import resolve_catalog_label


METRICS = {"gc_fraction", "gc_percent", "length_nt", "longest_homopolymer",
           "genomic_start_1based", "mature_start_1based"}
AGGREGATES = {"sum", "maximum", "minimum", "sorted_ascending", "sorted_descending"}


def _number(value):
    if type(value) not in {int, float}:
        raise ValueError("objective_number_invalid")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("objective_number_not_finite") from exc


def _chrom(value):
    value = str(value or "").strip()
    return value[3:] if value.lower().startswith("chr") else value


def _mature_start(blocks, reference, length):
    """Project all target blocks through one exact reference's exon structure."""
    strand = reference.get("strand")
    chrom = _chrom(reference.get("chrom") or (reference.get("metadata") or {}).get("seq_region_name"))
    exons = reference.get("exons_tx") or []
    if not chrom or str(strand) not in {"+", "-", "1", "-1"} or not exons:
        raise ValueError("objective_reference_geometry_unresolved")
    if any(not isinstance(exon, dict) or type(exon.get("start")) is not int
           or type(exon.get("end")) is not int or not 1 <= exon["start"] <= exon["end"]
           or _chrom(exon.get("seq_region_name") or chrom) != chrom for exon in exons):
        raise ValueError("objective_reference_geometry_invalid")
    genomic = sorted((exon["start"] - 1, exon["end"]) for exon in exons)
    if any(left[1] > right[0] for left, right in zip(genomic, genomic[1:])):
        raise ValueError("objective_reference_exons_overlap")
    negative = str(strand) in {"-", "-1"}
    ordered = list(reversed(genomic)) if negative else genomic
    projected = []
    for axis, start, end in blocks:
        if _chrom(axis) != chrom:
            raise ValueError("objective_target_reference_axis_mismatch")
        offset, covered = 0, 0
        for lo, hi in ordered:
            a, b = max(start, lo), min(end, hi)
            if a < b:
                projected.append((offset + hi - b, offset + hi - a) if negative
                                 else (offset + a - lo, offset + b - lo))
                covered += b - a
            offset += hi - lo
        if covered != end - start:
            raise ValueError("objective_target_not_fully_exonic")
    projected.sort()
    if (not projected or sum(b - a for a, b in projected) != length
            or any(left[1] != right[0] for left, right in zip(projected, projected[1:]))):
        raise ValueError("objective_target_mature_placement_not_unique_contiguous")
    return projected[0][0] + 1


def compile_objectives(keys, entries, objectives, reference_catalog, verified_blocks):
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("requested_panel_objectives_empty_or_invalid")
    columns = []
    for objective in objectives:
        if (not isinstance(objective, dict)
                or set(objective) - {"metric", "aggregate", "direction", "absolute_distance_from", "reference_transcript_id"}
                or objective.get("metric") not in METRICS
                or objective.get("aggregate") not in AGGREGATES
                or objective.get("direction") not in {"minimize", "maximize"}):
            raise ValueError("requested_panel_objective_invalid")
        metric = objective["metric"]
        reference = None
        if metric == "mature_start_1based":
            if (not isinstance(reference_catalog, dict)
                    or any(not isinstance(entry, dict) or not isinstance(entry.get("metadata", {}), dict)
                           for entry in reference_catalog.values())):
                raise ValueError("objective_reference_catalog_invalid")
            label = resolve_catalog_label(objective.get("reference_transcript_id"), reference_catalog, reference_catalog)
            if label is None:
                raise ValueError("objective_reference_identity_unresolved")
            reference = reference_catalog[label]
        target = _number(objective["absolute_distance_from"]) if "absolute_distance_from" in objective else None
        values = []
        coordinate_axes = set()
        for key, entry in zip(keys, entries):
            sequence = key[0]
            if metric in {"gc_fraction", "gc_percent"}:
                value = Fraction(sum(base in "GC" for base in sequence), len(sequence))
                if metric == "gc_percent":
                    value *= 100
            elif metric == "length_nt":
                value = Fraction(len(sequence))
            elif metric == "longest_homopolymer":
                value = Fraction(max(len(match.group()) for match in re.finditer(r"(.)\1*", sequence)))
            else:
                placements = set()
                for row, _ in entry["rows"]:
                    blocks = verified_blocks(row)
                    if not blocks:
                        raise ValueError("objective_target_geometry_unresolved")
                    if metric == "genomic_start_1based":
                        coordinate_axes.update(_chrom(axis) for axis, _, _ in blocks)
                        if len(coordinate_axes) != 1:
                            raise ValueError("objective_target_multiple_genomic_axes")
                        placements.add(min(start for _, start, _ in blocks) + 1)
                    else:
                        placements.add(_mature_start(blocks, reference, len(sequence)))
                if len(placements) != 1:
                    raise ValueError("objective_target_coordinate_ambiguous")
                value = Fraction(next(iter(placements)))
            values.append(abs(value - target) if target is not None else value)
        columns.append(values)
    return columns


def _aggregate(values, aggregate):
    if aggregate == "sum":
        return sum(values, Fraction(0))
    if not values and aggregate in {"maximum", "minimum"}:
        return None
    if aggregate == "maximum":
        return max(values)
    if aggregate == "minimum":
        return min(values)
    return sorted(values, reverse=aggregate == "sorted_descending")


def _display(value, exact=False):
    if isinstance(value, list):
        return [_display(item, exact) for item in value]
    if value is None:
        return None
    return str(value) if exact else int(value) if value.denominator == 1 else float(value)


def optimize_objectives(columns, objectives, count, solve):
    """Lexicographic selection with a caller-supplied constrained MILP solver.

    solve(cost, extra_constraints) returns an optimal set, None for proven
    infeasibility, or raises for indeterminacy. Each added constraint is a
    (integer coefficient vector, lower, upper) tuple. Order statistics use a
    logarithmic number of feasibility solves per selected position, avoiding
    enumeration of candidate combinations and arbitrary weighted rank sums.
    """
    n = len(columns[0])
    if not n:
        return set(), _ranking_summary(columns, objectives, set(), 0)
    fixed = []
    selected = solve([0] * n, fixed)
    if selected is None:
        raise RuntimeError("requested_objective_base_panel_infeasible")
    solves = 1
    for values, objective in zip(columns, objectives):
        aggregate, direction = objective["aggregate"], objective["direction"]
        if not count:
            break
        if aggregate == "sum":
            denominator = lcm(*(value.denominator for value in values))
            weights = [int(value * denominator) for value in values]
            divisor = reduce(gcd, weights, 0) or 1
            weights = [weight // divisor for weight in weights]
            # Integer differences must survive the solver's floating storage.
            if sum(abs(weight) for weight in weights) > 2**45:
                raise ValueError("objective_exact_sum_exceeds_solver_precision")
            cost = weights if direction == "minimize" else [-value for value in weights]
            selected = solve(cost, fixed)
            solves += 1
            if selected is None:
                raise RuntimeError("requested_objective_sum_unresolved")
            total = sum(weights[index] for index in selected)
            fixed.append((weights, total, total))
            continue
        positions = ([count] if aggregate == "maximum" else [1] if aggregate == "minimum"
                     else list(range(1, count + 1)))
        if aggregate == "sorted_descending":
            positions.reverse()
        unique = sorted(set(values))
        for position in positions:
            low, high, best = 0, len(unique) - 1, None
            while low <= high:
                middle = (low + high) // 2
                threshold = unique[middle]
                if direction == "minimize":
                    weights = [int(value <= threshold) for value in values]
                    required = position
                else:
                    weights = [int(value >= threshold) for value in values]
                    required = count - position + 1
                constraint = (weights, required, count)
                chosen = solve([0] * n, [*fixed, constraint])
                solves += 1
                if chosen is not None:
                    best = (chosen, constraint)
                if (chosen is not None) == (direction == "minimize"):
                    high = middle - 1
                else:
                    low = middle + 1
            if best is None:
                raise RuntimeError("requested_objective_order_statistic_unresolved")
            selected, constraint = best
            fixed.append(constraint)
    # Any remaining equivalence under all explicit priorities uses native order.
    selected = solve(list(range(1, n + 1)), fixed)
    solves += 1
    if selected is None:
        raise RuntimeError("requested_objective_final_tie_break_unresolved")
    return selected, _ranking_summary(columns, objectives, selected, solves)


def _ranking_summary(columns, objectives, selected, solves):
    values = [_aggregate([column[index] for index in selected], objective["aggregate"])
              for column, objective in zip(columns, objectives)]
    return {"status": "complete", "criteria": deepcopy(objectives),
                      "objective_values": [_display(value) for value in values],
                      "exact_objective_values": [_display(value, True) for value in values],
                      "optimization": "lexicographic_subject_to_all_existing_count_and_physical_constraints",
                      "selected_count": len(selected), "solver_calls": solves,
                      "empty_panel": not selected,
                      "remaining_ties": "native_rank_sum_after_all_requested_priorities",
                      "scientific_interpretation": "Requested deterministic panel priorities, not experimental potency."}
