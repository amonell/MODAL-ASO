"""Intersect retrieved source peaks with a reference feature, retaining provenance."""
from copy import deepcopy


def restrict_source_rows(rows, feature_rows, feature, *, reference_transcript=None):
    """Return only the BED0 intersections, never relabel an entire source peak.

    Strand-specific source peaks must match the transcript; unstranded source
    peaks may overlap either strand but retain their original strand metadata.
    The complete retrieved input inventory and each projection remain auditable.
    """
    summary = {'target_feature':feature, 'reference_transcript':reference_transcript,
               'source_rows_bed0_half_open':deepcopy(rows),
               'reference_feature_rows_bed0_half_open':deepcopy(feature_rows),
               'source_row_count':len(rows), 'projections':[],
               'policy':'Intersect retrieved source peaks with the resolved transcript feature; preserve original source rows. No feature membership is inferred from a label.'}
    if not feature_rows:
        summary.update(status='unresolved_reference_feature', retained_source_row_count=0,
                       projected_row_count=0)
        return [], summary
    result, retained = [], set()
    for index, row in enumerate(rows):
        intersections = []
        for region in feature_rows:
            if str(row[0]).removeprefix('chr') != str(region[0]).removeprefix('chr'):
                continue
            strand = row[5] if len(row)>5 else None
            reference_strand = region[5] if len(region)>5 else None
            if strand in {'+','-'} and reference_strand in {'+','-'} and strand != reference_strand:
                continue
            start, end = max(int(row[1]),int(region[1])), min(int(row[2]),int(region[2]))
            if end > start:
                intersections.append((start,end))
        merged = []
        for start,end in sorted(set(intersections)):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1],end)
            else:
                merged.append([start,end])
        projected = []
        for start,end in merged:
            item = list(row);item[1:3] = [start,end]
            result.append(item);projected.append(item)
            retained.add(index)
        summary['projections'].append({'source_row_index':index,
            'status':'intersects_feature' if projected else 'outside_feature_or_opposite_strand',
            'projected_rows_bed0_half_open':projected})
    summary.update(status='resolved', retained_source_row_count=len(retained),
                   projected_row_count=len(result), excluded_source_row_count=len(rows)-len(retained))
    return result, summary
