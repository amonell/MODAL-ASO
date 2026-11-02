"""Require complete screening evidence for every delivered design molecule."""
from .candidate_assessment import off_target_evidence
from .mismatch_screening import reference_incomplete_reasons


def screening_delivery(rows, *, required_species, mismatch_budget):
    incomplete = []
    for index, row in enumerate(rows, 1):
        values = {**(row.get("source_annotations") or {}), **row}
        evidence = off_target_evidence(values)
        reasons = list(values.get("off_target_incomplete_reasons") or [])
        if evidence["status"] != "complete_within_declared_scope":
            reasons.append("search_incomplete")
        if evidence["mismatch_budget"] is None or evidence["mismatch_budget"] < mismatch_budget:
            reasons.append("requested_mismatch_budget_unsearched")
        if evidence["unit"] not in {"genes", "loci"}:
            reasons.append("count_unit_missing")
        scope = values.get("off_target_count_scope") or {}
        reasons.extend(reference_incomplete_reasons(scope.get("junction_search_summary") or {},
            values.get("off_target_query_sequence"), values.get("off_target_reference_coverage")))
        for species in required_species:
            reference = evidence["references_by_species"].get(species)
            if not isinstance(reference, dict) or not reference:
                reasons.append(f"reference_identity_missing:{species}")
            status = evidence["species_statuses"].get(species)
            if status and status not in {"complete", "complete_within_declared_scope"}:
                reasons.append(f"species_search_incomplete:{species}")
            species_values = evidence["species_evidence"].get(species)
            if isinstance(species_values, dict) and species_values:
                species_evidence = off_target_evidence(species_values)
                if (species_evidence["status"] != "complete_within_declared_scope"
                        or species_evidence["mismatch_budget"] < mismatch_budget):
                    reasons.append(f"species_search_incomplete:{species}")
            elif len(required_species) > 1:
                reasons.append(f"species_counts_missing:{species}")
        if reasons:
            label = str(row.get("candidate_id") or row.get("name") or index)
            incomplete.append({"candidate": label, "reasons": list(dict.fromkeys(reasons))})
    return {"status": "incomplete" if incomplete else "complete", "checked_candidates": len(rows),
            "required_species": list(required_species), "mismatch_budget": mismatch_budget,
            "incomplete_candidates": incomplete,
            "errors": [f"selected_off_target_screen_incomplete:{item['candidate']}" for item in incomplete]}
