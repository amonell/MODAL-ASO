"""Render planning priorities without turning desired effects into evidence."""


def _text(value):
    return str(value or "").replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def render_splice_plan(context, ranked_targets):
    candidates = {}
    for candidate in context.get("steric_site_candidates") or []:
        label = candidate.get("canonical_label") or candidate.get("label_hint")
        if label in candidates:
            raise ValueError("Ambiguous reference-bound splice label: " + str(label))
        if label:
            candidates[label] = candidate
    lines = ["## Strategy Overview", "",
             "This is a planning priority list. Directions describe desired junction usage, "
             "not measured ASO effects. Candidate-level perturbation scores and their limitations "
             "must be assessed separately.", ""]
    for objective in context.get("isoform_objectives") or []:
        label = objective.get("isoform_label") or objective.get("transcript_id")
        direction = objective.get("desired_abundance")
        if label and direction in {"increase", "decrease", "preserve"}:
            lines.append(f"- {_text(label)}: {direction} — requested objective.")
    protected = context.get("preserved_isoforms") or []
    if protected:
        lines.append("- Preserve: " + ", ".join(_text(value) for value in protected) + ".")
    lines.extend(["", "## Ranked Steric Hindrance Targets", "",
                  "Order records the model's planning choice, not an efficacy ranking. "
                  "Catalog membership supports only the stated reference scope; it does not establish "
                  "transcriptome-wide uniqueness or selective modulation.", ""])
    for rank, ranked in enumerate(ranked_targets, 1):
        bundle = ranked.get("bundle") or []
        if not bundle:
            raise ValueError("A splice priority has no reference-bound target")
        rendered = []
        for item in bundle:
            label, direction = item.get("label"), item.get("desired_direction")
            if label not in candidates or direction not in {"increase", "decrease", "neutral"}:
                raise ValueError("Unresolved splice priority identity or direction: " + str(label))
            rendered.append(f"{_text(label)} ({direction})")
        lines.append(f"{rank}) Desired junction usage: " + ", ".join(rendered))
        lines.append("   - Primary targets: " + ", ".join(rendered))
        for item in bundle:
            candidate = candidates[item["label"]]
            scope = candidate.get("reference_scope") or {}
            compared = scope.get("compared_transcript_ids") or []
            present = scope.get("present_in") or []
            unresolved = scope.get("unresolved_transcript_ids") or []
            if not scope:
                lines.append(f"   - {_text(item['label'])}: reference sharing has not been resolved.")
            else:
                lines.append(f"   - {_text(item['label'])}: {len(present)} matching transcript(s) "
                             f"among {len(compared)} inspected ({_text(scope.get('scope'))}); "
                             "outside-catalog specificity remains unknown.")
                if present:
                    lines.append("   - Matching references: " + ", ".join(_text(value) for value in present))
                if unresolved:
                    lines.append("   - Unresolved references: " + ", ".join(_text(value) for value in unresolved))
            exon_indices = candidate.get("exon_indices") or {}
            if exon_indices:
                lines.append("   - Annotation exon ordinals: " + ", ".join(
                    f"{_text(label)}: {_text(index)}" for label, index in exon_indices.items()))
        guards = ranked.get("safeguards") or []
        lines.append("   - Safeguards (requested directions, not confirmed preservation): " + (
            ", ".join(f"{_text(guard.get('label'))} ({_text(guard.get('guard'))})" for guard in guards)
            if guards else "none recorded"))
        lines.append("")
    if not ranked_targets:
        lines.extend(["No reference-bound junction priorities were produced.", ""])
    lines.extend(["## Mechanism and Evidence Limits", "",
                  "A desired increase in canonical junction usage does not establish that directly "
                  "masking that junction will increase inclusion. Splice-site masking can instead "
                  "impair recognition; inclusion-promoting strategies require evidence for the relevant "
                  "regulatory or competing-site mechanism. The effect of each proposed molecule remains "
                  "to be established. See [Hua et al., 2007](https://doi.org/10.1371/journal.pbio.0050073).",
                  "", "Deletion/replacement model perturbations are computational surrogates. They do "
                  "not simulate ASO occupancy or establish experimental potency, isoform selectivity, "
                  "or successful preservation of another transcript."])
    return "\n".join(lines)
