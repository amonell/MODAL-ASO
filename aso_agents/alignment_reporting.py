"""Render retained alignment geometry without generating reference facts."""
from copy import deepcopy
from .alignment_identity import recorded_window_identity


def alignment_geometry(hit):
    """Keep the identity, RNA substrate and recorded segments in one citation.

    A bounding span cannot establish the number of blocks in spliced RNA.
    Missing or malformed segment evidence therefore leaves the count unknown.
    This is a description of an existing alignment, not a binding decision.
    """
    segments = hit.get("genomic_segments")
    complete = bool(isinstance(segments, list) and segments) and all(
        isinstance(segment, dict)
        and bool(segment.get("chrom"))
        and isinstance(segment.get("start"), int)
        and not isinstance(segment.get("start"), bool)
        and isinstance(segment.get("end"), int)
        and not isinstance(segment.get("end"), bool)
        and segment["start"] <= segment["end"]
        for segment in segments
    )
    fields = (
        "transcript_id", "transcript_version", "versioned_transcript_id",
        "reference_build", "reference_source", "reference_annotation", "reference_identity",
        "coordinate_system", "mapping_scope", "rna_context",
        "match_orientation", "strand", "cdna_start", "cdna_end",
        "target_sequence", "matched_window_sequence", "matched_target_sequence",
        "mismatch_count", "mismatch_positions_in_supplied_sequence_1based",
        "genomic_span", "requested_site_overlaps", "isoform_label", "search_scope",
    )
    result = {key: deepcopy(hit[key]) for key in fields if key in hit}
    result["genomic_segments"] = deepcopy(segments) if isinstance(segments, list) else None
    result["recorded_genomic_segment_count"] = len(segments) if complete else None
    result["segment_count_basis"] = (
        "Number of explicitly recorded genomic segments, in recorded order; "
        "the bounding span is not substituted for spliced-RNA segments."
        if complete else "Unknown: complete genomic segment evidence was not supplied."
    )
    # Raw segment-list completeness and validated locus continuity are distinct
    # facts. The native identity validator already accepts an explicit unspliced
    # window only when its span, sequence length, orientation and mismatch
    # evidence agree. Expose that same evidence at the alignment citation; do
    # not invent raw segments or apply the inference to mature/spliced RNA.
    if hit.get("mapping_scope") == "pre_mrna_locus":
        identity = recorded_window_identity(hit, "near_match")
        if identity is not None:
            blocks = identity["ordered_genomic_blocks_1based_inclusive"]
            result["validated_contiguous_locus_geometry"] = {
                "genomic_segments": deepcopy(blocks),
                "genomic_segment_count": len(blocks),
                "coordinate_system": "1based_inclusive",
                "rna_context": identity["rna_context"],
                "mapping_scope": identity["mapping_scope"],
                "matched_sequence_length": len(identity["matched_sequence"]),
                "basis": "Native recorded_window_identity validation of an explicitly unspliced, ungapped locus window: genomic span length equals the recorded matched sequence length, and any supplied segments agree. This establishes contiguous reference geometry while leaving raw recorded segment-list completeness unchanged; it does not establish biological binding or activity.",
            }
    return result
