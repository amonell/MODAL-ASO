"""Strand-aware annotated geometry for resolved feature checks, without names."""


def explicit_window_geometry(record, geometry):
    """Prove stated offsets from an independently resolved anchor, not a name."""
    if not geometry:
        return None
    requested = record.get('requested_target') or {}
    resolved = record.get('resolved_target') or {}
    offsets = [requested.get('relative_start_nt'), requested.get('relative_end_nt')]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in offsets):
        return None
    if offsets[1] < offsets[0] or requested.get('anchor_site') != resolved.get('anchor_site'):
        return None
    if requested.get('reference_transcript_id') and str(requested['reference_transcript_id']).split('.')[0] != str(geometry['transcript_id']).split('.')[0]:
        return None
    direction = 1 if geometry['strand'] == '+' else -1
    anchor = geometry['selected_feature']['position']
    positions = [anchor + direction * value for value in offsets]
    if min(positions) < 1:
        return None  # A clipped interval is not the requested complete window.
    rows = resolved.get('materialized_rows') or []
    expected = [min(positions) - 1, max(positions)]
    if len(rows) != 1 or len(rows[0]) < 6 or list(rows[0][1:3]) != expected:
        return None
    if str(rows[0][0]).removeprefix('chr') != str(geometry['chrom']).removeprefix('chr') or rows[0][5] != geometry['strand']:
        return None
    return {'reference_transcript_id': geometry['transcript_id'], 'anchor_1based': anchor,
            'anchor_site': resolved['anchor_site'], 'relative_start_nt': offsets[0],
            'relative_end_nt': offsets[1], 'genomic_window_bed0_half_open': list(rows[0]),
            'relative_coordinate_system': 'Both endpoints included; zero is the annotated exon boundary; offsets follow transcript 5-prime to 3-prime direction.',
            'scope': 'Verifies the stated operational search window after anchor identity is resolved. It does not assert that a named regulatory feature spans this entire window, is present, binds a factor or has activity. Preserve this distinction in all recommendations.'}


def transcript_feature_geometry(record, catalog):
    resolved = record.get('resolved_target') or {}
    tid = str(resolved.get('transcript_id') or '')
    anchor = resolved.get('anchor_position')
    site = resolved.get('anchor_site')
    rows = resolved.get('materialized_rows') or []
    if not tid or not isinstance(anchor, int) or isinstance(anchor, bool) or site not in {'acceptor', 'donor'} or not rows:
        return None
    proofs = []
    for label, entry in catalog.items():
        meta = entry.get('metadata') or {}
        source_id = str(entry.get('transcript_id') or meta.get('id') or label)
        if source_id.split('.')[0] != tid.split('.')[0] or meta.get('reference_source', entry.get('reference_source')) != 'local_gtf':
            continue
        versioned_id = str(meta.get('versioned_id') or entry.get('versioned_id') or
                           (f"{source_id.split('.')[0]}.{meta.get('version', entry.get('version'))}" if meta.get('version',entry.get('version')) is not None else source_id))
        requested_id = str((record.get('requested_target') or {}).get('reference_transcript_id') or tid)
        if '.' in requested_id and requested_id != versioned_id:
            continue
        annotation = meta.get('reference_annotation') or entry.get('reference_annotation')
        strand = {1:'+', -1:'-', '1':'+', '-1':'-'}.get(entry.get('strand'), entry.get('strand'))
        chrom = entry.get('chrom') or meta.get('seq_region_name')
        if not annotation or strand not in {'+', '-'} or not chrom:
            continue
        if any(len(row) < 6 or str(row[0]).removeprefix('chr') != str(chrom).removeprefix('chr') or row[5] != strand for row in rows):
            continue
        features = []
        exons = sorted(entry.get('exons_tx') or [], key=lambda exon:int(exon.get('index') or 0))
        for index, exon in enumerate(exons):
            start, end = exon.get('start'), exon.get('end')
            if any(not isinstance(v, int) or isinstance(v, bool) for v in (start, end)) or start < 1 or end < start:
                continue
            for kind, position in [('acceptor', start if strand == '+' else end), ('donor', end if strand == '+' else start)]:
                offset = (position - anchor) * (1 if strand == '+' else -1)
                features.append({'exon_index':exon.get('index'), 'exon_start':start, 'exon_end':end,
                                 'site':kind, 'position':position, 'transcript_direction_offset_from_anchor':offset,
                                 'is_splice_junction':index > 0 if kind == 'acceptor' else index < len(exons)-1,
                                 'relation':'at_anchor' if offset == 0 else 'upstream' if offset < 0 else 'downstream'})
        selected = [f for f in features if f['site'] == site and f['position'] == anchor]
        if len(selected) != 1:
            continue
        proofs.append({'source':'local_gtf', 'reference_annotation':annotation,
                       'transcript_id':versioned_id, 'chrom':chrom, 'strand':strand,
                       'coordinate_system':'1based_inclusive', 'selected_feature':selected[0],
                       'annotated_junctions':[feature for feature in features if feature['is_splice_junction']],
                       'annotated_exon_boundaries':features,
                       'scope':'Annotated transcript geometry only. Regulatory-element names or user intent require their own mapping evidence.'})
    if not proofs or any(p != proofs[0] for p in proofs[1:]):
        return None
    return proofs[0]
