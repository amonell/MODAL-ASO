"""Project displayed target markers without mixing reference coordinate spaces."""
from copy import deepcopy


def attach_genomic_marker(target, catalog, project_range, gene_chrom=None):
    """Preserve the original region and record a separately verified map position.

    BED target intervals are half-open, whereas the transcript projection and the
    gene diagram use one-based genomic positions. A cDNA midpoint is projected
    through exons; the midpoint of a genomic bounding span could lie in an intron.
    """
    target['genomic_map_position'] = None
    target['genomic_map_segments'] = []
    target['genomic_map_status'] = 'unresolved_reference'
    reference = str(target.get('chrom') or '')
    candidates = []
    for label, entry in (catalog or {}).items():
        if not isinstance(entry, dict):
            continue
        ids = {str(label), str(entry.get('transcript_id') or entry.get('id') or ''),
               str((entry.get('metadata') or {}).get('versioned_id') or '')}
        # A specified version must agree with reference metadata.
        if reference and (reference in ids or ('.' not in reference and
                reference in {item.split('.')[0] for item in ids})):
            candidates.append(entry)
    if candidates:
        start, end = target.get('region_start'), target.get('region_end')
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            return
        if target.get('coordinate_system') != 'bed0_half_open':
            return
        projections = []
        for entry in candidates:
            blocks = project_range(start + 1, end, entry)
            point = project_range((start + end + 1) // 2, (start + end + 1) // 2, entry)
            if sum(b-a+1 for _, a, b in blocks) != end-start or len(point) != 1:
                return
            projections.append((blocks, point[0]))
        if any(item != projections[0] for item in projections[1:]):
            target['genomic_map_status'] = 'ambiguous_transcript_projection'
            return
        blocks, (chrom, position, _) = projections[0]
        target['genomic_map_segments'] = [dict(chrom=c, start=a, end=b) for c,a,b in blocks]
        target['genomic_map_position'] = position
        target['genomic_map_chrom'] = chrom
        target['genomic_map_status'] = 'projected_from_transcript_exons'
        target['source_region'] = {key: deepcopy(target.get(key)) for key in
            ('chrom', 'region_start', 'region_end', 'position', 'coordinate_system')}
    elif reference and gene_chrom and reference.removeprefix('chr') != str(gene_chrom).removeprefix('chr'):
        return
    elif gene_chrom and (reference or target.get('site_kind') in {'acceptor', 'donor'}):
        # Native splice-site plans already use genomic one-based anchors.
        position = target.get('position')
        if not isinstance(position, int):
            return
        if target.get('coordinate_system') == 'bed0_half_open':
            start, end = target.get('region_start'), target.get('region_end')
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
                return
            position = (start + end + 1) // 2
            target['genomic_map_segments'] = [dict(chrom=reference, start=start+1, end=end)]
        target['genomic_map_position'] = position
        target['genomic_map_chrom'] = reference or gene_chrom
        target['genomic_map_status'] = 'native_genomic_reference'
    target['genomic_map_coordinate_system'] = '1based_inclusive'
