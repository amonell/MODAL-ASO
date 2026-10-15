"""Select an eligible panel subject to typed physical-region count constraints.

This does not generate candidates, relax eligibility or rescore scientific data.
Counts use sequence plus chemistry; repeated placements never pad a minimum.
"""
from copy import deepcopy

from .candidate_minimums import belongs_to_region
from .candidate_presentation import molecule_key


def select_region_minimum_panel(ranked_rows, groups, intervals_for_row, *,
                                global_limit=None, per_site_limit=None,
                                program_limits=None):
    """Return ranked molecules and the binding records proving their membership.

    Usually the ordinary ranked panel already meets the minima. Otherwise solve
    the binary count constraints, maximizing panel size before minimizing the sum
    of native rank positions. Solver failures leave explicit shortfalls rather
    than substituting the larger pool for the panel.
    """
    molecules = {}
    for row in ranked_rows:
        key = molecule_key(row)
        if not key[0] or set(key[0]) - set("ACGT"):
            continue
        membership = {i for i, group in enumerate(groups)
                      if belongs_to_region(row, group, intervals_for_row)}
        entry = molecules.setdefault(key, {"rows": [], "membership": set(), "programs": set()})
        entry["rows"].append((row, membership))
        entry["membership"].update(membership)
        program = row.get("program_id") or (row.get("source_annotations") or {}).get("program_id")
        if program:
            entry["programs"].add(program)
    keys = list(molecules)
    entries = list(molecules.values())
    minimums = [group["minimum"] for group in groups]
    memberships = [{j for j, entry in enumerate(entries) if i in entry["membership"]}
                   for i in range(len(groups))]
    maximum = min(len(keys), global_limit) if global_limit is not None else len(keys)
    upper = [(set(range(len(keys))), maximum, "global")]
    if per_site_limit is not None:
        upper.extend((members, per_site_limit, groups[i]["group_id"])
                     for i, members in enumerate(memberships))
    upper.extend(({j for j, entry in enumerate(entries) if program in entry["programs"]},
                  limit, "program:" + program)
                 for program, limit in (program_limits or {}).items())

    # Retain the native order whenever it already satisfies the request.
    selected = set()
    for j in range(len(keys)):
        if all(len(selected & members) < limit for members, limit, _ in upper if j in members):
            selected.add(j)
    satisfies = lambda chosen: all(len(chosen & members) >= minimum
                                   for members, minimum in zip(memberships, minimums))
    solver_status = "ranked_panel_already_satisfies_minimums"
    if keys and not satisfies(selected):
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix, vstack

        constraints = memberships + [members for members, _, _ in upper]
        coords = [(i, j) for i, members in enumerate(constraints) for j in sorted(members)]
        matrix = coo_matrix((np.ones(len(coords)),
                            ([i for i, _ in coords], [j for _, j in coords])),
                           shape=(len(constraints), len(keys))).tocsc()
        lower = np.array(minimums + [0] * len(upper), dtype=float)
        higher = np.array([np.inf] * len(groups) + [limit for _, limit, _ in upper], dtype=float)
        options = {"mip_rel_gap": 0.0}
        try:
            size = milp(-np.ones(len(keys)), integrality=np.ones(len(keys)),
                        bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lower, higher),
                        options=options)
            solver_status = "infeasible" if size.status == 2 else "solver_unresolved"
            if size.success:
                count = int(round(sum(size.x)))
                fixed_count = coo_matrix(np.ones((1, len(keys)))).tocsc()
                ranked = milp(np.arange(1, len(keys) + 1, dtype=float),
                              integrality=np.ones(len(keys)), bounds=Bounds(0, 1),
                              constraints=LinearConstraint(vstack([matrix, fixed_count]),
                                  np.append(lower, count), np.append(higher, count)), options=options)
                chosen = {i for i, value in enumerate(ranked.x if ranked.success else size.x)
                          if value > 0.5}
                # Independently check integer membership after numerical solving.
                if satisfies(chosen) and all(len(chosen & members) <= limit for members, limit, _ in upper):
                    selected = chosen
                    solver_status = "optimal_rank_sum" if ranked.success else "feasible_rank_unresolved"
        except (ValueError, RuntimeError) as exc:
            solver_status = "solver_error:" + type(exc).__name__

    rows = []
    for index, entry in enumerate(entries):
        if index not in selected:
            continue
        covered = set()
        for position, (row, membership) in enumerate(entry["rows"]):
            # Keep additional physical placements only when needed to prove a
            # selected molecule's membership in another requested region.
            if position == 0 or membership - covered:
                rows.append(deepcopy(row))
                covered.update(membership)
    summaries = [{"group_label": group["group_id"], "label": group["label"],
                  "minimum": group["minimum"], "candidate_limit": per_site_limit,
                  "eligible_count": len(memberships[i]),
                  "selected_count": len(selected & memberships[i]),
                  "shortfall_count": max(0, group["minimum"] - len(selected & memberships[i]))}
                 for i, group in enumerate(groups)]
    return rows, {"status": "complete" if satisfies(selected) else "unfulfilled",
                  "method": "native_rank_order_subject_to_physical_region_minimums",
                  "selection_scope": "per_site", "group_scope": "physical_region",
                  "count_unit": "distinct_antisense_sequence_plus_chemistry",
                  "candidate_limit": per_site_limit, "global_candidate_limit": global_limit,
                  "eligible_count": len(keys), "selected_count": len(selected),
                  "selected_binding_records": len(rows), "unselected_count": len(keys) - len(selected),
                  "group_count": len(groups), "groups": summaries, "solver_status": solver_status,
                  "source_eligibility_preserved": True}
