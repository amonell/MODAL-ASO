"""Project typed transcript intervals through attested exon structures."""
from copy import deepcopy

from .gapmer_selectivity import resolve_catalog_label


def genomic_sources(goal, constraint):
    """Keep each source interval's identity when it spans genomic segments."""
    space = str(constraint.get('coordinate_space') or 'genomic').lower()
    rows = constraint.get('materialized_rows') or []
    catalog = goal.get('isoform_transcripts') or {}
    sources = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            return []
        projected = [deepcopy(list(row))]
        if space in {'cdna', 'transcript', 'mature_transcript'}:
            label = resolve_catalog_label(row[0], catalog, catalog)
            if label is None:
                return []
            entry = catalog[label];meta = entry.get('metadata') or {}
            chrom = entry.get('chrom') or meta.get('seq_region_name')
            strand = '-' if str(entry.get('strand', meta.get('strand'))) in {'-', '-1'} else '+'
            exons = entry.get('exons_tx') or []
            start, end = int(row[1]), int(row[2])
            length = sum(int(e['end']) - int(e['start']) + 1 for e in exons)
            if not chrom or not 0 <= start < end <= length:
                return []
            chrom = str(chrom)
            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            projected = [];offset = 0
            for exon in exons:
                left, right = int(exon['start']) - 1, int(exon['end'])
                size = right - left
                a, b = max(start, offset), min(end, offset + size)
                if a < b:
                    lo, hi = ((left + a - offset, left + b - offset) if strand == '+'
                              else (right - (b - offset), right - (a - offset)))
                    projected.append([chrom,lo,hi,row[3] if len(row)>3 else constraint.get('label'),
                                      row[4] if len(row)>4 else 0,strand])
                offset += size
            if sum(x[2]-x[1] for x in projected) != end-start:
                return []
        sources.append({'source_row':deepcopy(list(row)), 'coordinate_space':space,
                        'genomic_rows':projected})
    return sources
