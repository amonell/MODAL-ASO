"""Identify contradictory declared roles without resolving or changing identities."""


def knockdown_preservation_conflicts(goal):
    if not isinstance(goal, dict):
        return {}

    def identity(value):
        return value.strip().casefold() if isinstance(value, str) else ""

    protected = {}
    gene = identity(goal.get("gene_symbol"))
    for index, target in enumerate(goal.get("preservation_targets") or []):
        if not isinstance(target, dict) or target.get("action") != "preserve":
            continue
        if target.get("preservation_target_type") not in {"transcript", "isoform"}:
            continue
        target_gene = identity(target.get("gene_symbol"))
        if gene and target_gene and gene != target_gene:
            continue
        for field in ("transcript_id", "isoform_label"):
            label = identity(target.get(field))
            if label:
                protected.setdefault(label, []).append(f"preservation_targets[{index}].{field}")

    conflicts = {}
    for index, program in enumerate(goal.get("design_programs") or []):
        if not isinstance(program, dict):
            continue
        if identity(program.get("modality")) != "gapmer" or identity(program.get("intent")) != "knockdown":
            continue
        for field in ("transcript_id", "isoform_label", "transcript_ids"):
            value = program.get(field)
            values = value if isinstance(value, list) else [value]
            matches = [item for item in values if identity(item) in protected]
            if not matches:
                continue
            conflicts[f"design_programs[{index}].{field}"] = {
                "conflicting_target_values": list(dict.fromkeys(matches)),
                "preservation_paths": sorted({path for item in matches for path in protected[identity(item)]}),
                "reason": (
                    "A declared gapmer knockdown target is also an explicitly preserved transcript or isoform. "
                    "Program transcript_ids requires every listed member to bind; it is not a catalog of "
                    "reference or avoided transcripts. Repair the conflicting role from quoted user evidence, "
                    "or request clarification if the user actually requires incompatible roles. Do not "
                    "discard preservation or target requirements merely to produce candidates."
                ),
                "identity_basis": "Exact declared identity after case/whitespace normalization; no alias inference or version stripping.",
            }
    return conflicts
