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


def target_exon_pairs(goal, anchors):
    """Bind the current exon goal to exact paired boundaries in its reference.

    The local plan identifies the physical target. Catalog exon geometry checks
    that identity independently of biological exon names and nearby coordinates.
    Different candidate endpoints or contradictory plan records remain unresolved.
    """
    job = goal.get('routing_job')
    source = job if isinstance(job, dict) else goal
    effect = source.get('desired_effect')
    direction = {'inclusion': 'increase', 'skipping': 'decrease'}.get(effect)
    if (not direction or source.get('splice_event') or source.get('intent') not in {None, 'splice_modulation'}
            or source.get('desired_delta') not in {None, direction}):
        return {}
    catalog = goal.get('isoform_transcripts') or {}
    from .evaluation_feature_identity import transcript_identifier
    from .splice_reference_scope import measurement_site_exon_membership
    result = {}
    anchors = [site for site in anchors if isinstance(site, dict) and not site.get('monitoring_purpose')]
    for site in anchors:
        key = physical_site(site)
        if not key or (site.get('decision_direction') or site.get('delta_direction')) != direction:
            continue
        memberships = site.get('exon_indices')
        if not isinstance(memberships, dict) or not memberships:
            memberships = measurement_site_exon_membership(
                {'chrom': key[0], 'strand': key[1], 'site': key[2], 'position': key[3]}, catalog)
        preferred = source.get('isoform_label')
        if preferred in catalog:
            memberships = {preferred: memberships[preferred]} if preferred in memberships else {}
        matches = []
        for label, ordinal in memberships.items():
            entry = catalog.get(label)
            structure = _structure(entry)
            if not structure or type(ordinal) is not int or not 1 < ordinal < len(structure[2]):
                break
            chrom, strand, exons = structure
            exon = exons[ordinal - 1]
            positions = {'acceptor': exon['start' if strand == '+' else 'end'],
                         'donor': exon['end' if strand == '+' else 'start']}
            if key != (chrom, strand, key[2], positions[key[2]]):
                break
            # A stale second boundary in this local plan cannot supply a new
            # exon interpretation. Validate every declared side for this exon.
            declarations = [other for other in anchors
                if (other.get('exon_indices') or {}).get(label) == ordinal]
            if any(physical_site(other) != (chrom, strand, other.get('site') or other.get('site_kind'),
                       positions.get(other.get('site') or other.get('site_kind')))
                   or (other.get('decision_direction') or other.get('delta_direction')) != direction
                   for other in declarations):
                break
            partner_kind = 'donor' if key[2] == 'acceptor' else 'acceptor'
            matches.append({'partner_key': (chrom, strand, partner_kind, positions[partner_kind]),
                'isoform_label': label, 'reference_transcript_id': transcript_identifier(entry, label),
                'exon_index': ordinal, 'chrom': chrom, 'strand': strand,
                'acceptor_position': positions['acceptor'], 'donor_position': positions['donor']})
        else:
            if matches and len({match['partner_key'] for match in matches}) == 1:
                result[key] = {'direction': direction, 'desired_effect': effect,
                    'partner_key': matches[0]['partner_key'], 'references': matches}
    return result


def paired_exon_guard(candidate, primary, pairs, *, protected_sites=()):
    """Allow intended movement at the partner while retaining opposing checks."""
    context = pairs.get(physical_site(primary))
    key = physical_site(candidate)
    if (not context or key != context['partner_key'] or key in protected_sites
            or candidate.get('required') or candidate.get('preserved_isoforms')
            or candidate.get('conflict_between_objectives')
            or any(candidate.get(field) is not None for field in ('max_abs_delta', 'max_increase', 'max_decrease'))):
        return candidate
    increase = context['direction'] == 'increase'
    note = ('Paired boundary of the targeted exon: increases support the inclusion goal; decreases are checked against the limit.'
            if increase else 'Paired boundary of the targeted exon: decreases support the skipping goal; increases are checked against the limit.')
    return {**candidate, 'suggested_guard': 'no_decrease' if increase else 'no_increase',
        'suggested_guard_reason': 'target_exon_paired_boundary', 'target_guard_reason': note,
        'guard_policy_source': 'reference_paired_exon_goal', 'paired_exon_objective': deepcopy(context)}


def neighborhood_guard_sites(goal, anchors, *, radius_bp=DEFAULT_CONTEXT_RADIUS_BP):
    """Find genuine annotated junctions near an anchor, on its chromosome/strand.

    First-exon starts and last-exon ends are transcript termini, not junctions.
    A shared physical junction is measured once; all catalog memberships remain.
    The paired boundary of a targeted exon is included at any distance.
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
    # A targeted exon's paired boundary remains measurable even for a long
    # exon extending beyond the surrounding-junction search radius.
    for context in target_exon_pairs(goal, anchors).values():
        key = context['partner_key']
        if key in anchor_keys or key in grouped:
            continue
        chrom, strand, kind, position = key
        grouped[key] = {'site': kind, 'position': position, 'chrom': chrom, 'strand': strand,
            'label': f'{kind}@{position}', 'canonical_label': f'{kind}@{position}',
            'delta_direction': 'context', 'monitoring_purpose': 'target_exon_boundary',
            'exon_indices': {ref['isoform_label']: ref['exon_index'] for ref in context['references']},
            'present_in': [ref['isoform_label'] for ref in context['references']]}
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
