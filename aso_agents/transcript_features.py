"""Resolve named mature-RNA features from the exact reference annotation."""
from copy import deepcopy

from .gapmer_selectivity import resolve_catalog_label
from .region_projection import genomic_sources

FEATURES = frozenset({"start_codon", "cds"})


def normalize_feature_name(value):
    name = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"translation_start": "start_codon", "coding_sequence": "cds"}
    name = aliases.get(name, name)
    if name not in FEATURES:
        raise ValueError("transcript_feature_unsupported: " + name)
    return name


def feature_interval(entry, feature):
    """Return BED0 mature-RNA bounds, retaining split codons across exon joins."""
    feature = normalize_feature_name(feature)
    strand = str(entry.get("strand", (entry.get("metadata") or {}).get("strand")))
    if strand not in {"+", "-", "1", "-1"}:
        raise ValueError("transcript_feature_strand_unresolved")
    negative = strand in {"-", "-1"}
    exons = entry.get("exons_tx") or []
    if not exons:
        raise ValueError("transcript_feature_exons_unresolved")
    offset, coding = 0, []
    previous = None
    for exon in exons:
        lo, hi = exon.get("start"), exon.get("end")
        if type(lo) is not int or type(hi) is not int or not 1 <= lo <= hi:
            raise ValueError("transcript_feature_exon_bounds_invalid")
        if previous is not None and (hi >= previous[0] if negative else lo <= previous[1]):
            raise ValueError("transcript_feature_exon_order_invalid")
        previous = (lo, hi)
        a, b = exon.get("coding_start_genomic"), exon.get("coding_end_genomic")
        if exon.get("contains_coding") or a is not None or b is not None:
            if type(a) is not int or type(b) is not int or not lo <= a <= b <= hi:
                raise ValueError("transcript_feature_cds_bounds_unresolved")
            start, end = ((offset + hi-b, offset + hi-a+1) if negative
                          else (offset + a-lo, offset + b-lo+1))
            coding.append((start, end))
        offset += hi-lo+1
    if not coding:
        raise ValueError("transcript_feature_cds_unresolved")
    if any(left[1] != right[0] for left, right in zip(coding, coding[1:])):
        raise ValueError("transcript_feature_cds_discontinuous")
    start, end = coding[0][0], coding[-1][1]
    if feature == "start_codon":
        metadata = entry.get("metadata") or {}
        tags = []
        for source in (entry, metadata):
            for key in ("tag", "tags", "annotation_tags"):
                value = source.get(key) or []
                tags.extend(value if isinstance(value, list) else [value])
        if "cds_start_NF" in tags or entry.get("cds_start_incomplete") or metadata.get("cds_start_incomplete"):
            raise ValueError("transcript_feature_coding_start_incomplete")
        if end-start < 3:
            raise ValueError("transcript_feature_start_codon_incomplete")
        annotation = entry if "start_codon_genomic_intervals" in entry else metadata
        if "start_codon_genomic_intervals" in annotation:
            pieces = annotation["start_codon_genomic_intervals"]
            if not isinstance(pieces, list) or not pieces:
                raise ValueError("transcript_feature_start_codon_not_annotated")
            covered = set()
            for piece in pieces:
                if (not isinstance(piece, (list, tuple)) or len(piece) != 2
                        or any(type(value) is not int for value in piece)
                        or not 1 <= piece[0] <= piece[1] or piece[1]-piece[0] >= 3):
                    raise ValueError("transcript_feature_start_annotation_invalid")
                cursor, matches = 0, []
                for exon in exons:
                    lo, hi = exon["start"], exon["end"]
                    if lo <= piece[0] <= piece[1] <= hi:
                        a, b = ((cursor+hi-piece[1], cursor+hi-piece[0]+1) if negative
                                else (cursor+piece[0]-lo, cursor+piece[1]-lo+1))
                        matches.append((a, b))
                    cursor += hi-lo+1
                if len(matches) != 1:
                    raise ValueError("transcript_feature_start_annotation_unmapped")
                covered.update(range(*matches[0]))
            if covered != set(range(start, start+3)):
                raise ValueError("transcript_feature_start_annotation_conflicts_with_cds")
        return start, start+3
    return start, end


def materialize_feature(constraint, catalog, primary=None):
    """Build reference-bound source rows; never reinterpret user interval numbers."""
    if any(constraint.get(key) not in (None, [], {}) for key in
           ("rows", "intervals_bed0_half_open", "genomic_rows", "cdna_rows")):
        raise ValueError("transcript_feature_conflicting_explicit_intervals")
    feature = normalize_feature_name(constraint.get("feature_name"))
    references = [str(constraint[key]).strip() for key in
                  ("reference_transcript_id", "transcript_id", "isoform_label")
                  if constraint.get(key)]
    labels = [resolve_catalog_label(ref, catalog, catalog) for ref in references]
    if references:
        if None in labels or len(set(labels)) != 1:
            raise ValueError("transcript_feature_reference_unresolved_or_conflicting")
        label = labels[0]
    else:
        label = primary
        if label not in catalog:
            raise ValueError("transcript_feature_reference_unresolved")
    entry = catalog[label]
    reference = (entry.get("metadata") or {}).get("versioned_id") or entry.get("transcript_id") or label
    start, end = feature_interval(entry, feature)
    strand = "-" if str(entry.get("strand", (entry.get("metadata") or {}).get("strand"))) in {"-", "-1"} else "+"
    name = constraint.get("label") or feature
    rows = [[reference, start, end, name, 0, strand]] if end > start else []
    materialized = {**deepcopy(constraint), "source_type": "transcript_feature",
                    "feature_name": feature, "coordinate_space": "cdna",
                    "materialized_rows": rows, "materialized_cdna_rows": deepcopy(rows),
                    "aso_overlap_requirement": constraint.get("aso_overlap_requirement") or "any_overlap"}
    sources = genomic_sources({"isoform_transcripts": catalog}, materialized)
    genomic = [row for source in sources for row in source["genomic_rows"]]
    if rows and sum(row[2]-row[1] for row in genomic) != end-start:
        raise ValueError("transcript_feature_genomic_projection_incomplete")
    materialized["materialized_genomic_rows"] = genomic
    materialized["row_count"] = len(rows)
    materialized["materialization_summary"] = {
        "status": "materialized" if rows else "annotated_feature_empty",
        "method": "locked_transcript_annotated_cds_and_exon_bounds",
        "reference_transcript_id": reference, "catalog_label": label,
        "feature_name": feature, "coordinate_convention": "bed0_half_open",
        "mature_interval": [start, end], "row_count": len(rows),
        "sequence_or_biological_activity_validation_claimed": False,
    }
    return materialized

