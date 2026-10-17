"""Translate explicit transcript-preservation constraints into reference junctions."""
from copy import deepcopy


def required_routing_safeguards(safeguards, protected_labels, *, default_tolerance=0.1):
    """Complete both directions of explicit preservation without shrinking caps."""
    result = deepcopy(safeguards)
    for label in sorted(protected_labels):
        guard = next((g for g in result if g.get('label') == label), None)
        if guard is None:
            guard = {'label':label, 'max_abs_delta':default_tolerance}
            result.append(guard)
        elif guard.get('max_abs_delta') is None:
            for key in ('max_increase','max_decrease'):
                if guard.get(key) is None:
                    guard[key] = default_tolerance
        guard.update(required=True, note='Explicit transcript preservation; recorded perturbation cap is a computational screening default, not proven cellular neutrality.')
    return result


def include_preserved_exon_boundaries(goal, sites, *, window_bp):
    """Measure alternative boundaries of preserved exons in the targeted feature.

    A target's ordinal does not identify its counterpart in another isoform.
    Intersect the physical exon spans instead. Monitoring sites never enter the
    design/tiling plan and never become requests to change preserved abundance.
    """
    result = deepcopy(sites)
    protected = protected_junctions(goal)
    if not sites or not protected:
        return result
    catalog = {**(goal.get('all_transcripts_full') or {}), **(goal.get('isoform_transcripts') or {})}
    def chromosome(value):
        return str(value or '').removeprefix('chr')
    def strand(value):
        return '-' if str(value) in {'-', '-1'} else '+'
    spans, existing = [], set()
    for site in sites:
        acceptor, donor = site.get('acceptor') or {}, site.get('donor') or {}
        a, d = acceptor.get('position'), donor.get('position')
        if not isinstance(a, int) or not isinstance(d, int):
            continue
        chrom, sign = chromosome(acceptor.get('chrom')), strand(acceptor.get('strand'))
        if (chrom, sign) != (chromosome(donor.get('chrom')), strand(donor.get('strand'))):
            continue
        spans.append((chrom, sign, min(a, d), max(a, d)))
        kind = site.get('site_kind')
        if kind in {'acceptor', 'donor'}:
            existing.add((chrom, sign, kind, site[kind]['position']))
    for label, references in protected.items():
        for ref in references:
            chrom, sign = chromosome(ref['chrom']), strand(ref['strand'])
            key = (chrom, sign, ref['site'], ref['position'])
            if key in existing:
                continue
            entry = catalog[ref['isoform_label']]
            exon = next((e for e in entry.get('exons_tx') or [] if e.get('index') == ref['exon_index']), None)
            if not exon or not any(c == chrom and s == sign and exon['start'] <= hi and exon['end'] >= lo
                                   for c, s, lo, hi in spans):
                continue
            acceptor, donor = (exon['end'], exon['start']) if sign == '-' else (exon['start'], exon['end'])
            def window(position):
                return {'chrom': ref['chrom'], 'strand': sign, 'position': position,
                    'start': max(1, position-window_bp), 'end': position+window_bp, 'window_bp': window_bp}
            result.append({'label': label, 'display_label': label, 'canonical_label': label,
                'site_kind': ref['site'], 'exon_index': ref['exon_index'],
                'isoform_label': ref['isoform_label'], 'present_in': sorted({r['isoform_label'] for r in references}),
                'delta_direction': 'context', 'window_bp': window_bp,
                'monitoring_purpose': 'preservation', 'acceptor': window(acceptor), 'donor': window(donor),
                'preservation_reference_evidence': deepcopy(references)})
            existing.add(key)
    return result


def protected_junctions(goal):
    from .preservation_context import preservation_reference_context
    if not isinstance(goal, dict):
        return {}
    targets = [target for container in [goal, *(goal.get('design_programs') or [])] if isinstance(container, dict)
               for target in container.get('preservation_targets') or []
               if isinstance(target, dict) and target.get('action') == 'preserve']
    if not targets:
        return {}
    catalog = {**(goal.get('all_transcripts_full') or {}), **(goal.get('isoform_transcripts') or {})}
    references = preservation_reference_context({**goal, 'preservation_targets':targets}, catalog)
    result = {}
    for label, entry in catalog.items():
        metadata = entry.get('metadata') or {}
        matches = [reference['requested_target'] for reference in references
                   if reference['resolution_status'] == 'resolved' and label in reference['resolved_labels']]
        if not matches:
            continue
        strand = str(entry.get('strand') or metadata.get('strand'))
        if strand not in {'+', '-', '1', '-1'}:
            continue
        exons = sorted(entry.get('exons_tx') or [], key=lambda x:int(x.get('index') or 0))
        for index, exon in enumerate(exons):
            # Transcript-terminal boundaries are not splice junctions.
            sites = [('acceptor', exon.get('start') if strand in {'+', '1'} else exon.get('end'))] if index else []
            if index < len(exons)-1:
                sites.append(('donor', exon.get('end') if strand in {'+', '1'} else exon.get('start')))
            for site, position in sites:
                if not isinstance(position, int) or isinstance(position, bool):
                    continue
                key = f'{site}@{position}'
                result.setdefault(key, []).append({'isoform_label':label,'transcript_id':entry.get('transcript_id'),
                    'chrom':entry.get('chrom') or metadata.get('seq_region_name'), 'strand':strand,
                    'exon_index':exon.get('index'), 'site':site, 'position':position, 'preservation_targets':matches})
    return result
