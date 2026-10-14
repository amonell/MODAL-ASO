"""Stable candidate identity and presentation order; no scientific re-scoring."""
import json
import math


def program_group_labels(goal):
    programs = [program for program in goal.get("design_programs") or []
                if isinstance(program, dict) and program.get("program_id")]
    labels = [str(program.get("label") or program["program_id"]) for program in programs]
    return {str(program["program_id"]): (label if labels.count(label) == 1 else str(program["program_id"]))
            for program, label in zip(programs, labels)}


def bind_candidate_program_groups(rows, goal):
    """Join persisted job/program identities without parsing labels or filenames."""
    labels = program_group_labels(goal)
    aliases = {}
    for program in goal.get("design_programs") or []:
        program_id = str(program.get("program_id") or "")
        if program_id not in labels:
            continue
        for alias in (program_id, program.get("label")):
            if alias:
                aliases.setdefault(str(alias), set()).add(program_id)
    for job in [*(goal.get("steric_design_jobs") or []), *(goal.get("gapmer_design_jobs") or [])]:
        program_id = str(job.get("program_id") or "")
        if program_id in labels and job.get("job_id"):
            aliases.setdefault(str(job["job_id"]), set()).add(program_id)
    bound = []
    for row in rows:
        identity = None
        for key in ("program_id", "job_id", "job_label", "target_region_label"):
            matches = aliases.get(str(row.get(key) or ""), set())
            if len(matches) == 1:
                identity = next(iter(matches))
                break
        if identity is None and not any(row.get(key) for key in
                ("program_id", "job_id", "job_label", "target_region_label")):
            # Aggregate single-producer exports can omit job annotations. The
            # recorded source modality still identifies the only possible
            # producer. Never guess across multiple producers or replace a
            # conflicting explicit identity with this fallback.
            producers = [str(program.get("program_id")) for program in goal.get("design_programs") or []
                         if program.get("program_id") in labels and row.get("source_modality")
                         and program.get("modality") == row["source_modality"]]
            if len(producers) == 1:
                identity = producers[0]
        bound.append({**row, "program_id": identity, "candidate_group_label": labels[identity]}
                     if identity else row)
    return bound


def molecule_key(row):
    annotations = row.get("source_annotations") or {}
    sequence = (row.get("synthesized_antisense_sequence") or row.get("Synthesis")
                or row.get("chemistry_synthesis") or annotations.get("chemistry_synthesis")
                or annotations.get("Synthesis"))
    chemistry = (row.get("IDT") or row.get("chemistry_idt") or annotations.get("IDT")
                 or annotations.get("chemistry_idt") or row.get("chemistry_label") or row.get("chemistry"))
    if not sequence:
        # Unverified sequence identity cannot justify deduplicating molecules.
        return ("unverified", json.dumps(row, sort_keys=True, default=str))
    return (str(sequence).upper(), json.dumps(chemistry, sort_keys=True, default=str))


def ordered_rows(rows, rank_field="selection_rank"):
    def key(item):
        index, row = item
        value = row.get(rank_field)
        try:
            rank = float(value) if not isinstance(value, bool) else float("inf")
        except (TypeError, ValueError):
            rank = float("inf")
        if not math.isfinite(rank) or rank <= 0:
            rank = float("inf")
        return rank, index
    return [row for _, row in sorted(enumerate(rows), key=key)]


def distinct_ranked_rows(rows):
    result, seen = [], set()
    for row in ordered_rows(rows):
        key = molecule_key(row)
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result
