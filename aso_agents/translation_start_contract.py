"""Normalize translation-start windows without interpreting them as splice sites."""

from typing import Any, Dict


_START_FEATURES = {
    "start codon", "start codon aug", "start codon region",
    "translation start", "translation start codon",
    "translation initiation codon", "translation start site",
}


def _feature(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").replace("-", " ").split())


def _is_translation_start_feature(value: Any) -> bool:
    normalized = _feature(value)
    if normalized in _START_FEATURES:
        return True
    tokens = set(normalized.split())
    return (
        {"start", "codon"} <= tokens
        or {"translation", "initiation"} <= tokens
        or {"aug", "translation"} <= tokens
    )


def canonicalize_translation_start_program_windows(
    goal: Dict[str, Any], *, user_evidence_text: str = "",
) -> bool:
    """Move wrongly typed start-codon constraints to transcript-aware windows.

    The typed feature and transcript identity must agree.  No genomic interval,
    splice-modulation program, or conflicting existing target window is changed.
    """
    if not isinstance(goal, dict):
        return False
    programs = goal.get("design_programs")
    if not isinstance(programs, list):
        return False
    changed = False
    repaired_transcripts = set()
    for program in programs:
        if not isinstance(program, dict):
            continue
        if str(program.get("modality") or "").strip().lower() != "steric":
            continue
        if str(program.get("intent") or goal.get("intent") or "").strip().lower() != "binding":
            continue
        if program.get("run_splice_router") is True or program.get("run_splice_eval") is True:
            continue
        if program.get("target_region") not in (None, "", "translation_start"):
            continue
        transcript = str(program.get("transcript_id") or "").strip()
        if not transcript:
            continue
        constraints = program.get("region_constraints")
        if not isinstance(constraints, list) or len(constraints) != 1:
            continue
        constraint = constraints[0]
        if not isinstance(constraint, dict):
            continue
        source_type = constraint.get("source_type")
        if source_type == "splice_relative_window":
            window = constraint.get("splice_relative_window")
        elif source_type == "explicit_rows" and user_evidence_text:
            rows = constraint.get("intervals_bed0_half_open")
            if not isinstance(rows, list) or not rows:
                continue
            # A model claim that coordinates came from the user is not proof.
            # Preserve genuinely supplied intervals; discard an inferred row
            # only when the original evidence lacks both interval endpoints.
            evidence = str(user_evidence_text).replace(",", "").replace("_", "")
            supplied = any(
                isinstance(row, dict)
                and str(row.get("end") or "") in evidence
                and any(str(start) in evidence for start in (
                    row.get("start"),
                    int(row["start"]) + 1 if str(row.get("start") or "").isdigit() else None,
                ) if start is not None)
                for row in rows
            )
            if supplied:
                continue
            targets = [
                target.get("splice_relative_window")
                for target in goal.get("evaluation_targets") or []
                if isinstance(target, dict)
                and str(target.get("transcript_id") or transcript).strip() == transcript
                and isinstance(target.get("splice_relative_window"), dict)
                and _is_translation_start_feature(target["splice_relative_window"].get("feature_name"))
            ]
            if len(targets) != 1:
                continue
            window = targets[0]
        else:
            continue
        if not isinstance(window, dict) or not _is_translation_start_feature(window.get("feature_name")):
            continue
        if str(window.get("reference_transcript_id") or constraint.get("reference_transcript_id") or transcript).strip() != transcript:
            continue
        if str(constraint.get("reference_transcript_id") or transcript).strip() != transcript:
            continue
        if source_type == "splice_relative_window" and (
            constraint.get("intervals_bed0_half_open") or constraint.get("rows")
        ):
            continue
        if constraint.get("aso_overlap_requirement") not in (None, "", "full_containment"):
            continue
        try:
            start = int(window.get("relative_start_nt"))
            end = int(window.get("relative_end_nt"))
        except (TypeError, ValueError):
            continue
        if start > 0 or end < 0 or start >= end:
            continue
        flanks = {"up": -start, "down": end}
        if program.get("flanks") not in (None, {}, flanks):
            continue
        program["target_region"] = "translation_start"
        program["flanks"] = flanks
        program["region_constraints"] = []
        repaired_transcripts.add(transcript)
        changed = True
    if not changed:
        return False
    targets = goal.get("evaluation_targets")
    if isinstance(targets, list):
        for target in targets:
            if not isinstance(target, dict):
                continue
            window = target.get("splice_relative_window")
            if not isinstance(window, dict) or not _is_translation_start_feature(window.get("feature_name")):
                continue
            transcript = str(target.get("transcript_id") or window.get("reference_transcript_id") or "").strip()
            if transcript not in repaired_transcripts:
                continue
            if str(window.get("reference_transcript_id") or transcript).strip() != transcript:
                continue
            target.pop("splice_relative_window", None)
    if len(programs) == len([p for p in programs if isinstance(p, dict) and p.get("target_region") == "translation_start"]):
        goal["target_region"] = "translation_start"
    return True
