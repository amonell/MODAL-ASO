"""Expose coordinate origins to evidence-bound intake, without guessing intent."""
from copy import deepcopy


COORDINATE_ORIGIN_INSTRUCTIONS = (
    "Coordinate origin and indexing are independent facts. Mature-RNA, mature-transcript "
    "or cDNA positions count from the transcript's 5-prime end; a genome assembly or "
    "the transcript's chromosome does not turn those numbers into genomic positions. "
    "Preserve the evidence-supported origin before converting indexing. Explicit mature "
    "intervals belong in region_constraints with coordinate_space='cdna', the exact "
    "transcript_id, and that transcript identifier in each row's reference/chrom slot. "
    "Convert one-based inclusive [s,e] to BED0 [s-1,e) exactly once. The native exon "
    "projection supplies genomic coordinates later, including split and reverse-strand "
    "mappings. A stated transcript plus stated mature endpoints is sufficient to encode "
    "the cDNA constraint now. Do not ask the user for chromosome coordinates or permission "
    "to express their own mature interval as cDNA: this preserves their input and is not "
    "genome-assembly conversion. Reference retrieval and exon projection are subsequent "
    "native operations; their results are not prerequisite intake fields. Do not invent "
    "chromosome coordinates or strand from transcript positions. "
    "evaluation_targets has no transcript-coordinate field: retain the target identity "
    "there, and omit numeric start/end/position and chrom/strand unless genomic mapping "
    "is independently supported. Keep mature endpoints in the executable region rows. "
    "A small coordinate, a transcript lock, or a UTR label alone does not establish "
    "coordinate origin; explicit genomic intervals must remain genomic. In a mixed "
    "request, verify each interval against its own evidence, not a request-wide default. "
    "For typed intervals_bed0_half_open, typed_interval_meaning gives host-computed "
    "equivalent one_based_inclusive bounds and length_nt. Compare these actual covered "
    "bases with the original evidence before accepting or changing an endpoint. The "
    "BED0 end equals the one-based inclusive end; do not add one to that endpoint. "
    "Reindexing must preserve interval length. recorded_coordinate_convention is "
    "source metadata and cannot reinterpret an explicitly typed BED0 field. Do not "
    "change correct interval bases merely to synchronize that metadata. A repair "
    "reason is not proof of correct conversion; check the recomputed interval meaning "
    "after every proposed change.\n"
)


def _typed_interval_meaning(rows):
    """Describe typed BED intervals without modifying them or inferring user intent."""
    result = []
    for index, row in enumerate(rows if isinstance(rows, list) else []):
        meaning = {"row_index": index, "status": "invalid_typed_interval"}
        if isinstance(row, dict):
            start, end, reference = row.get("start"), row.get("end"), row.get("chrom")
            if (type(start) is int and type(end) is int and 0 <= start < end
                    and isinstance(reference, str) and reference.strip()):
                meaning.update(
                    status="interpreted", reference=reference, strand=row.get("strand"),
                    bed0_half_open={"start": start, "end": end},
                    one_based_inclusive={"start": start + 1, "end": end},
                    length_nt=end - start,
                )
        result.append(meaning)
    return result


def coordinate_execution_summary(goal):
    """Describe native interpretation and a narrow, coherent semantic repair surface.

    This never converts or relabels coordinates. The existing model audit must
    cite the original user/source evidence for any repair. A single constraint
    needs this check just as much as several compound constraints do.
    """
    scopes = [("", goal)] + [(f"/design_programs/{i}", program)
        for i, program in enumerate(goal.get("design_programs") or [])
        if isinstance(program, dict)]
    constraints, program_windows, pointers = [], [], []
    for scope, container in scopes:
        for index, constraint in enumerate(container.get("region_constraints") or []):
            if not isinstance(constraint, dict):
                continue
            rows = constraint.get("intervals_bed0_half_open") or constraint.get("rows")
            if not rows:
                continue
            space = constraint.get("coordinate_space") or "genomic"
            constraints.append({
                "json_pointer": f"{scope}/region_constraints/{index}",
                "native_coordinate_space": space,
                "native_reference_kind": "transcript" if space == "cdna" else "chromosome",
                "transcript_id": constraint.get("transcript_id") or container.get("transcript_id"),
                "recorded_coordinate_convention": constraint.get("coordinate_convention"),
                "intervals_bed0_half_open": deepcopy(constraint.get("intervals_bed0_half_open")),
                "typed_interval_meaning": _typed_interval_meaning(constraint.get("intervals_bed0_half_open")),
                "rows": deepcopy(constraint.get("rows")),
                "genome_build": constraint.get("genome_build"),
            })
            base = f"{scope}/region_constraints/{index}"
            pointers.append(f"{base}/coordinate_space")
            # Typed BED0 fields already fix indexing; do not offer redundant
            # metadata edits that can oscillate without changing any bases.
            if constraint.get("rows"):
                pointers.append(f"{base}/coordinate_convention")
            for row_index, row in enumerate(constraint.get("intervals_bed0_half_open") or []):
                if isinstance(row, dict):
                    pointers.extend(f"{base}/intervals_bed0_half_open/{row_index}/{field}"
                                    for field in ("chrom", "start", "end", "strand"))
            for row_index, row in enumerate(constraint.get("rows") or []):
                if isinstance(row, (list, tuple)):
                    pointers.extend(f"{base}/rows/{row_index}/{field}"
                                    for field in (0, 1, 2, 5) if field < len(row))
        if scope and any(field in container for field in ("start", "end", "position")):
            program_windows.append({"json_pointer": scope, **{
                field: deepcopy(container.get(field)) for field in
                ("start", "end", "position", "coordinate_space", "coordinate_convention", "transcript_id")}})
            pointers.extend(f"{scope}/{field}" for field in
                ("start", "end", "position", "coordinate_space", "coordinate_convention", "transcript_id"))
    targets = []
    for index, target in enumerate(goal.get("evaluation_targets") or []):
        if not isinstance(target, dict) or not any(field in target for field in ("start", "end", "position")):
            continue
        targets.append(deepcopy(target))
        pointers.extend(f"/evaluation_targets/{index}/{field}"
                        for field in ("chrom", "start", "end", "position", "strand") if field in target)
    if not constraints and not program_windows and not targets:
        return None
    # Target identity and coordinate constraints must be repaired together;
    # otherwise a repaired cDNA row can leave a contradictory genomic target.
    return {"transcript_lock": deepcopy(goal.get("transcript_lock")),
            "constraints": constraints, "program_windows": program_windows,
            "genomic_evaluation_targets": targets,
            "audit_scope_pointers": list(dict.fromkeys(pointers))}
