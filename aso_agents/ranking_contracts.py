"""Names and explanations bound to the executed gapmer selection key."""

from copy import deepcopy


def scoped_strategy_routing_plan(plan):
    """Keep independent job rankings scoped to the jobs that produced them."""
    if not isinstance(plan, dict):
        return {}
    result = deepcopy(plan)
    jobs = result.get("jobs")
    if result.get("mode") != "per_job" or not isinstance(jobs, list):
        return result
    scoped = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        child = job.get("planner_plan") or {}
        ranking = child.get("ranking") if isinstance(child, dict) else {}
        if not isinstance(ranking, dict):
            continue
        scoped.append({"job_id": job.get("job_id"), "job_label": job.get("job_label") or job.get("label"),
                       "ranking": deepcopy(ranking)})
    summaries = []
    for index, item in enumerate(scoped, 1):
        label = item.get("job_label") or item.get("job_id") or f"Job {index}"
        summary = item["ranking"].get("summary")
        if summary:
            summaries.append(f"{label}: {summary}")
    result["ranking"] = {"mode": "per_job", "scope": "within_each_design_job",
                         "job_rankings": scoped,
                         "summary": "Independently ranked design jobs; order across jobs is not a comparative efficacy ranking."
                                    + ("\n" + "\n".join(summaries) if summaries else "")}
    return result

GAPMER_SELECTION_CRITERIA = (
    ("allele_evidence", "For an allele-specific design, prefer an observed wild-type mismatch with a valid variant position."),
    ("allele_gap", "For an allele-specific design, prefer the variant inside the DNA gap."),
    ("allele_center_distance", "For an allele-specific design, prefer the variant closer to the ASO center."),
    ("wildtype_mismatch", "For an allele-specific design, prefer more wild-type mismatches."),
    ("cross_species_status", "Order cross-species eligibility as passed, unknown, then failed where recorded."),
    ("worst_species_offtargets", "Prefer fewer recorded off-target genes in the worst evaluated species."),
    ("expression_incomplete", "Prefer records without a flagged incomplete off-target expression annotation."),
    ("worst_species_expression", "Prefer lower recorded worst-species off-target expression risk."),
    ("target_accessibility", "When requested, prefer higher measured full-window target unpaired probability in the least accessible required species; this is a computational ranking proxy, not an efficacy threshold."),
    ("structure_status", "Order the recorded no-secondary-structure flag as true, unknown, then false."),
    ("isoform_class", "Order isoform classes as target_only, target_without_beneficial, target_shared_with_beneficial, non_target, then gene_wide."),
    ("isoform_score", "Within the remaining ties, prefer a higher recorded isoform score; interpret that score using its recorded selectivity or catalog-coverage definition."),
    ("beneficial_hits", "For remaining ties, prefer fewer protected-isoform hits."),
    ("other_isoform_hits", "For remaining ties, prefer fewer other-isoform hits."),
    ("total_isoform_hits", "For remaining ties, prefer fewer total isoform hits; gene-wide breadth is already represented by the preceding catalog-coverage score."),
    ("weighted_offtarget_risk", "For remaining ties, prefer lower recorded weighted off-target risk."),
    ("expression_risk", "For remaining ties, prefer lower available off-target expression risk; unavailable expression ranks after finite values at this step."),
    ("gc_distance", "For remaining ties, prefer GC fraction closer to 0.50, rather than monotonically higher GC."),
    ("phastcons", "For remaining ties, prefer higher measured PhastCons; unavailable conservation follows measured values."),
    ("phylop", "For remaining ties, prefer higher measured PhyloP; unavailable conservation follows measured values."),
)


def gapmer_selection_order():
    return [
        {"criterion": name, "explanation": explanation}
        for name, explanation in GAPMER_SELECTION_CRITERIA
    ]


def gapmer_selection_key(components):
    """Use the same ordered criteria that are persisted with the ranking."""
    return tuple(components[name] for name, _ in GAPMER_SELECTION_CRITERIA)


def recorded_ranking_explanation(ranking):
    """Describe only a completed, recorded ordering; never guess a past policy."""
    if not isinstance(ranking, dict) or ranking.get("status") != "complete":
        return None
    order = ranking.get("ordering")
    if not isinstance(order, list) or not order:
        return None
    if not all(isinstance(item, dict) and item.get("criterion") and item.get("explanation") for item in order):
        return None
    return [str(item["explanation"]) for item in order]
