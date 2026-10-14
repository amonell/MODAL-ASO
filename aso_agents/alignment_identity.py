"""Conservative identity of recorded alignment windows, not binding predictions.

Exact and approximate searches can retain the same reference window. Preserve
both provenance records and expose their shared identity only when the recorded
reference, orientation, sequence, mismatch pattern and geometry establish it.
"""
from copy import deepcopy
import json


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _interval(value):
    if not isinstance(value, dict) or not value.get("chrom"):
        return None
    start, end = value.get("start"), value.get("end")
    if not (_integer(start) and _integer(end) and 1 <= start <= end):
        return None
    return {"chrom": str(value["chrom"]).removeprefix("chr"), "start": start, "end": end}


def recorded_window_identity(hit, kind):
    """Return comparable complete facts, or None when identity is not established.

    This never substitutes a bounding span for spliced-RNA block evidence. A
    contiguous pre-mRNA window may use its explicit span, provided its length
    equals the recorded ungapped sequence and any supplied segments agree.
    """
    if not isinstance(hit, dict):
        return None
    context = hit.get("rna_context")
    scope = hit.get("mapping_scope")
    if (context, scope) not in {
        ("unspliced_transcript_locus_pre_mrna", "pre_mrna_locus"),
        ("mature_spliced_cdna", "spliced_cdna"),
    }:
        return None
    if (not hit.get("transcript_id") or hit.get("strand") not in {"+", "-"}
            or hit.get("match_orientation") not in {"input_sequence", "reverse_complement"}
            or hit.get("coordinate_system", "1based_inclusive") != "1based_inclusive"):
        return None
    sequences = [hit[k].upper().replace("U", "T") for k in
                 ("matched_target_sequence", "matched_window_sequence")
                 if isinstance(hit.get(k), str) and hit[k]]
    if not sequences or len(set(sequences)) != 1 or set(sequences[0]) - set("ACGTRYSWKMBDHVN"):
        return None
    sequence = sequences[0]
    mismatches = hit.get("mismatch_count", 0 if kind == "exact_mapping" else None)
    positions = hit.get("mismatch_positions_in_supplied_sequence_1based", [] if mismatches == 0 else None)
    if (not _integer(mismatches) or mismatches < 0 or not isinstance(positions, list)
            or len(positions) != mismatches or len(set(str(p) for p in positions)) != len(positions)
            or any(not _integer(p) or not 1 <= p <= len(sequence) for p in positions)):
        return None
    if kind == "exact_mapping" and mismatches != 0:
        return None
    raw_segments = hit.get("genomic_segments")
    segments = [_interval(s) for s in raw_segments] if isinstance(raw_segments, list) else None
    cdna = None
    if scope == "pre_mrna_locus":
        span = _interval(hit.get("genomic_span"))
        if span is None or span["end"] - span["start"] + 1 != len(sequence):
            return None
        if raw_segments is not None and segments != [span]:
            return None
        blocks = [span]
    else:
        if not segments or any(s is None for s in segments):
            return None
        if sum(s["end"] - s["start"] + 1 for s in segments) != len(sequence):
            return None
        start, end = hit.get("cdna_start"), hit.get("cdna_end")
        if not (_integer(start) and _integer(end) and 1 <= start <= end and end-start+1 == len(sequence)):
            return None
        blocks, cdna = segments, [start, end]
    return {
        "reference_identity": {k: deepcopy(hit.get(k)) for k in (
            "transcript_id", "transcript_version", "versioned_transcript_id",
            "reference_build", "reference_source", "reference_annotation", "reference_identity")},
        "rna_context": context, "mapping_scope": scope,
        "match_orientation": hit["match_orientation"], "strand": hit["strand"],
        "matched_sequence": sequence, "mismatch_count": mismatches,
        "mismatch_positions_1based": sorted(positions),
        "ordered_genomic_blocks_1based_inclusive": blocks, "cdna_interval_1based_inclusive": cdna,
    }


def annotate_window_groups(hypotheses):
    """Add row-local identity links without discarding or changing source geometry."""
    groups, identified = [], {}
    unresolved = []
    for hypothesis in hypotheses:
        identity = recorded_window_identity(hypothesis["geometry"], hypothesis["kind"])
        if identity is None:
            hypothesis["window_identity_status"] = "not_established"
            unresolved.append(hypothesis["source_field"])
            continue
        key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        if key not in identified:
            group = {"window_group_id": "window_" + str(len(groups) + 1),
                     "identity": identity, "source_fields": [], "source_record_count": 0}
            groups.append(group)
            identified[key] = group
        group = identified[key]
        group["source_fields"].append(hypothesis["source_field"])
        group["source_record_count"] += 1
        hypothesis["window_identity_status"] = "established_from_recorded_evidence"
        hypothesis["window_group_id"] = group["window_group_id"]
    return {"identified_window_groups": groups, "unresolved_source_fields": unresolved,
            "identified_window_group_count": len(groups),
            "complete_for_retained_records": not unresolved,
            "scope": "Only this input row and its recorded reference context. Multiple source records may represent one identical aligned window; search method is provenance, not a new window. Group counts are not binding, activity or genome-wide counts. Unresolved records remain distinct evidence of unknown equivalence and are not excluded from any scientific assessment. Original records and their segment-count completeness remain unchanged."}
