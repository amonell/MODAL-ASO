"""Reference-derived monitoring around design sites, without widening design.

These are perturbation-score safeguards, not predictions of cellular neutrality.
The neighborhood is bounded in genomic coordinates and in the available catalog.
"""
from copy import deepcopy

from .design_event_consistency import _structure

DEFAULT_GUARD_TOLERANCE = 0.05
DEFAULT_CONTEXT_RADIUS_BP = 2000


def physical_site(site):
    kind = site.get('site') or site.get('site_kind')
    point = site.get(kind) or {} if kind in {'acceptor', 'donor'} else {}
    position = site.get('position', point.get('position'))
    chrom = str(site.get('chrom') or point.get('chrom') or '').removeprefix('chr')
    strand = {'+': '+', '1': '+', '-': '-', '-1': '-'}.get(
        str(site.get('strand') or point.get('strand')))
    if kind not in {'acceptor', 'donor'} or type(position) is not int or position < 1 or not chrom or not strand:
        return None
    return chrom, strand, kind, position


def neighborhood_guard_sites(goal, anchors, *, radius_bp=DEFAULT_CONTEXT_RADIUS_BP):
    """Find genuine annotated junctions near an anchor, on its chromosome/strand.

    First-exon starts and last-exon ends are transcript termini, not junctions.
    A shared physical junction is measured once; all catalog memberships remain.
    """
    if type(radius_bp) is not int or radius_bp < 0:
        raise ValueError('Splice neighborhood radius must be a nonnegative integer')
    anchor_keys = {key for site in anchors if isinstance(site, dict)
                   if (key := physical_site(site)) is not None}
    catalog = {**(goal.get('all_transcripts_full') or {}), **(goal.get('isoform_transcripts') or {})}
    grouped = {}
    for label, entry in catalog.items():
        structure = _structure(entry)
        if not structure:
            continue
        chrom, strand, exons = structure
        for exon in exons:
            for kind in ('acceptor', 'donor'):
                if (kind == 'acceptor' and exon['index'] == 1
                        or kind == 'donor' and exon['index'] == len(exons)):
                    continue
                endpoint = 'start' if (kind == 'acceptor') == (strand == '+') else 'end'
                position = exon[endpoint]
                key = chrom, strand, kind, position
                if key in anchor_keys or not any(c == chrom and s == strand and abs(p-position) <= radius_bp
                                                for c, s, _, p in anchor_keys):
                    continue
                item = grouped.setdefault(key, {'site': kind, 'position': position,
                    'chrom': entry.get('chrom') or chrom, 'strand': strand,
                    'label': f'{kind}@{position}', 'canonical_label': f'{kind}@{position}',
                    'delta_direction': 'context', 'monitoring_purpose': 'splice_neighborhood',
                    'neighborhood_radius_bp': radius_bp, 'exon_indices': {}, 'present_in': []})
                item['exon_indices'][label] = exon['index']
                item['present_in'].append(label)
    return [grouped[key] for key in sorted(grouped)]


def include_neighborhood_measurements(goal, sites, *, window_bp):
    """Add measured context without changing the plan, targets, or tile windows."""
    result = deepcopy(sites)
    # Preservation additions must not recursively expand the monitoring radius.
    anchors = [site for site in sites if not site.get('monitoring_purpose')]
    existing = {physical_site(site) for site in sites}
    catalog = {**(goal.get('all_transcripts_full') or {}), **(goal.get('isoform_transcripts') or {})}
    selected = goal.get('isoform_transcripts') or {}
    for site in neighborhood_guard_sites(goal, anchors):
        if physical_site(site) in existing:
            continue
        label = min(site['present_in'], key=lambda name: (name not in selected, name))
        index = site['exon_indices'][label]
        exon = next(e for e in catalog[label]['exons_tx'] if e['index'] == index)
        acceptor, donor = (exon['end'], exon['start']) if site['strand'] == '-' else (exon['start'], exon['end'])
        def window(position):
            return {'chrom': site['chrom'], 'strand': site['strand'], 'position': position,
                'start': max(1, position-window_bp), 'end': position+window_bp, 'window_bp': window_bp}
        result.append({**site, 'display_label': site['label'], 'site_kind': site['site'],
            'isoform_label': label, 'exon_index': index, 'window_bp': window_bp,
            'acceptor': window(acceptor), 'donor': window(donor)})
    return result
