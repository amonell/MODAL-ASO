"""Bind a requested exon conversion to the structures resolved for this run.

No gene names, paper panels, downloaded evidence cache or free-text inference
are used. A cassette event requires the observed bypass junction in the other
transcript, not merely the absence of an exon annotation.
"""
import hashlib
import json

from .gapmer_selectivity import resolve_catalog_label


class ExonConversionConflict(ValueError):
    def __init__(self, reference, requested_label, candidates, reason):
        self.reference = reference
        self.requested_label = requested_label
        self.candidates = candidates
        self.reason = reason
        super().__init__(f'exon_conversion_reference_conflict: {reference} exon {requested_label}: {reason}')

    def question(self):
        intervals = ', '.join(f"{r['chrom']}:{r['start_1based']}-{r['end_1based']}"
                              for r in self.candidates)
        return (f"Which physical exon should define exon {self.requested_label} on {self.reference}? "
                f"The transcript-conversion evidence and exon naming do not identify one consistent target"
                + (f" (candidate intervals: {intervals})" if intervals else '')
                + ". Provide the intended genomic interval or an exact reference with the desired exon definition.")


def _structure(entry):
    if not isinstance(entry, dict):
        return None
    metadata = entry.get('metadata') or {}
    chrom = entry.get('chrom') or metadata.get('seq_region_name')
    strand = str(entry.get('strand') or metadata.get('strand') or '')
    exons = entry.get('exons_tx') or []
    if not chrom or strand not in {'+', '-', '1', '-1'} or not exons:
        return None
    if any(not isinstance(e, dict) or type(e.get('index')) is not int
           or type(e.get('start')) is not int or type(e.get('end')) is not int
           or not 1 <= e['start'] <= e['end'] for e in exons):
        return None
    ordered = sorted(exons, key=lambda e: e['index'])
    if [e['index'] for e in ordered] != list(range(1, len(ordered) + 1)):
        return None
    minus = strand in {'-', '-1'}
    if any((left['start'] <= right['end'] if minus else left['end'] >= right['start'])
           for left, right in zip(ordered, ordered[1:])):
        return None
    return str(chrom).removeprefix('chr'), '-' if minus else '+', [
        {key: exon[key] for key in ('index', 'start', 'end')} for exon in ordered]


def validate_reference_sites(catalog):
    """Reject a producer's site coordinates that contradict its own exon chain."""
    for label, entry in catalog.items():
        structure = _structure(entry)
        if not structure:
            continue
        chrom, strand, exons = structure
        by_index = {e['index']: e for e in exons}
        for site in entry.get('sites') or []:
            if not isinstance(site, dict):
                continue
            exon = by_index.get(site.get('exon_index'))
            if exon is None:
                continue
            site_chrom = str(site.get('chrom') or chrom).removeprefix('chr')
            site_strand = '-' if str(site.get('strand') or strand) in {'-', '-1'} else '+'
            for kind, endpoint in [('acceptor', 'start' if strand == '+' else 'end'),
                                   ('donor', 'end' if strand == '+' else 'start')]:
                # The outer transcript endpoints are not splice junctions.
                if (kind == 'acceptor' and exon['index'] == 1
                        or kind == 'donor' and exon['index'] == len(exons)):
                    continue
                position = site.get(kind)
                if position is not None and (position != exon[endpoint]
                        or site_chrom != chrom or site_strand != strand):
                    raise ValueError(f'splice_site_reference_conflict: {label} exon {exon["index"]} {kind}')


def _cassette_events(included, excluded):
    left, right = _structure(included), _structure(excluded)
    if not left or not right or left[:2] != right[:2]:
        return []
    from .gapmer_selectivity import _reference_context_compatible
    if not _reference_context_compatible(included, excluded):
        return []
    chrom, strand, exons = left
    donor = 'end' if strand == '+' else 'start'
    acceptor = 'start' if strand == '+' else 'end'
    junctions = {(a[donor], b[acceptor]) for a, b in zip(right[2], right[2][1:])}
    result = []
    for upstream, exon, downstream in zip(exons, exons[1:], exons[2:]):
        bridge = upstream[donor], downstream[acceptor]
        if bridge not in junctions:
            continue
        if any(e['start'] <= exon['end'] and e['end'] >= exon['start'] for e in right[2]):
            continue
        result.append({'chrom': 'chr' + chrom, 'strand': strand,
            'start_1based': exon['start'], 'end_1based': exon['end'],
            'resolved_transcript_ordinal': exon['index'],
            'acceptor_1based': exon[acceptor], 'donor_1based': exon[donor],
            'bypass_junction_1based': list(bridge)})
    return result


def structural_exon_resolution(entry, requested_label, goal):
    """Return event evidence only for a typed, opposing isoform conversion."""
    if not isinstance(goal, dict) or type(requested_label) is not int:
        return None
    catalog = goal.get('isoform_transcripts') or {}
    reference = entry.get('transcript_id') or (entry.get('metadata') or {}).get('id')
    label = resolve_catalog_label(reference, catalog, catalog)
    if label is None:
        return None
    scopes = [goal] if goal.get('exon_label') == requested_label else []
    for program in goal.get('design_programs') or []:
        if not isinstance(program, dict):
            continue
        if program.get('exon_label') != requested_label:
            continue
        identity = program.get('transcript_id') or program.get('isoform_label')
        if not identity or resolve_catalog_label(identity, catalog, catalog) == label:
            scopes.append(program)
    effects = {scope.get('desired_effect') for scope in scopes
               if scope.get('desired_effect') in {'inclusion', 'skipping'}}
    if len(effects) != 1:
        return None
    effect = next(iter(effects))
    included_direction = 'increase' if effect == 'inclusion' else 'decrease'
    directions = {}
    for objective in goal.get('splice_isoform_objectives') or []:
        identity = objective.get('transcript_id') or objective.get('isoform_label')
        key = resolve_catalog_label(identity, catalog, catalog)
        direction = objective.get('desired_abundance')
        if key is not None and direction in {'increase', 'decrease'}:
            directions.setdefault(key, set()).add(direction)
    if directions.get(label) != {included_direction}:
        return None
    counterparts = sorted(key for key, values in directions.items()
                          if values == ({'decrease'} if effect == 'inclusion' else {'increase'}))
    events = {}
    for other in counterparts:
        for event in _cassette_events(entry, catalog[other]):
            key = event['start_1based'], event['end_1based']
            record = events.setdefault(key, {**event, 'excluded_in': []})
            record['excluded_in'].append(other)
    if not events:
        return None
    candidates = list(events.values())
    identities = {key: {'transcript_id': catalog[key].get('transcript_id'),
        'versioned_id': (catalog[key].get('metadata') or {}).get('versioned_id'),
        'structure': _structure(catalog[key])} for key in [label, *counterparts]}
    binding = {'version': 1, 'source': 'resolved_transcript_cassette_conversion',
        'status': 'verified_structural_event' if len(candidates) == 1 else 'ambiguous_structural_events',
        'reference_transcript_id': reference, 'catalog_label': label,
        'requested_exon_label': requested_label, 'desired_effect': effect,
        'explicit_transcript_ordinals': sorted({scope['exon_ordinal'] for scope in scopes
                                               if type(scope.get('exon_ordinal')) is int}),
        'reference_structure_sha256': hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
        'candidates': candidates,
        'interpretation': 'Physical event inferred from the requested isoform conversion and an observed bypass junction. This is not independent verification of historical exon nomenclature or a predicted abundance effect.'}
    if len(candidates) == 1:
        binding['resolved_transcript_ordinal'] = candidates[0]['resolved_transcript_ordinal']
    return binding


def record_event_binding(goal, binding):
    records = goal.setdefault('exon_conversion_bindings', [])
    key = binding['reference_transcript_id'], binding['requested_exon_label']
    records[:] = [r for r in records if (r.get('reference_transcript_id'), r.get('requested_exon_label')) != key]
    records.append(binding)
