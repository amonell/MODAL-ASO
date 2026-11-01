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
    scopes = list(binding.get('request_scopes') or [])
    for record in records:
        if ((record.get('reference_transcript_id'), record.get('requested_exon_label')) == key
                and record.get('reference_structure_sha256') == binding.get('reference_structure_sha256')
                and record.get('desired_effect') == binding.get('desired_effect')):
            scopes.extend(scope for scope in record.get('request_scopes') or [] if scope not in scopes)
    if scopes:
        binding = {**binding, 'request_scopes': scopes}
    records[:] = [r for r in records if (r.get('reference_transcript_id'), r.get('requested_exon_label')) != key]
    records.append(binding)


def conversion_requests(goal):
    """Normalize typed exon requests without changing the declared objectives."""
    programs = [p for p in goal.get('design_programs') or [] if isinstance(p, dict)]
    objectives = [o for o in goal.get('splice_isoform_objectives') or [] if isinstance(o, dict)]
    effects = {o.get('desired_effect') for o in objectives
               if o.get('desired_effect') in {'inclusion', 'skipping'}}
    default_effect = goal.get('desired_effect') or (next(iter(effects)) if len(effects) == 1 else None)
    lock = goal.get('transcript_lock') or {}
    reference = lock.get('value') if lock.get('kind') == 'explicit_id' else None
    scopes = [('goal', goal, reference, None)]
    scopes += [('program:' + str(p.get('program_id') or i), p,
                p.get('transcript_id') or p.get('isoform_label') or reference, p.get('program_id'))
               for i, p in enumerate(programs) if p.get('modality') != 'gapmer'
               and not (p.get('splice_event') or {}).get('event_type')]
    scopes += [('objective:' + str(i), o, o.get('transcript_id') or o.get('isoform_label'), None)
               for i, o in enumerate(objectives)]
    result = []
    for scope_id, scope, identity, program_id in scopes:
        if scope_id == 'goal' and programs and all(p.get('modality') == 'gapmer' for p in programs):
            continue
        inherit = scope_id.startswith('program:')
        if inherit and scope.get('targeting_policy') == 'exploratory_derived':
            # A derived program is an execution decision. Retain its original
            # request's site obligations while revalidating physical evidence.
            ancestors = {program_id, scope.get('parent_program_id')} - {None, ''}
            saved = [request for binding in [
                *(goal.get('_prior_exon_conversion_bindings') or []),
                *(goal.get('exon_conversion_bindings') or [])]
                for request in binding.get('request_scopes') or []
                if request.get('program_id') in ancestors
                and request.get('exon_label') == scope.get('exon_label', goal.get('exon_label'))
                and request.get('desired_effect') == (scope.get('desired_effect') or default_effect)
                and (not identity or not request.get('reference') or resolve_catalog_label(
                    identity, goal.get('isoform_transcripts') or {}, goal.get('isoform_transcripts') or {})
                    == resolve_catalog_label(request['reference'], goal.get('isoform_transcripts') or {},
                                             goal.get('isoform_transcripts') or {}))]
            if saved:
                for request in saved:
                    request = dict(request)
                    if request['minimum_program_id'] is None:
                        request['minimum'] = goal.get('min_asos_per_site')
                    if request not in result:
                        result.append(request)
                continue
        labels = [scope.get('exon_label'), *(scope.get('exon_labels') or [])]
        if inherit and not any(type(label) is int for label in labels):
            labels = [goal.get('exon_label'), *(goal.get('exon_labels') or [])]
        effect = scope.get('desired_effect') or default_effect
        if effect not in {'inclusion', 'skipping'}:
            continue
        kind = scope.get('site_kind') or scope.get('target_type') or goal.get('site_kind')
        kind = {'splice_donor': 'donor', 'splice_acceptor': 'acceptor'}.get(kind, kind)
        kinds = [kind] if kind in {'donor', 'acceptor'} else ['acceptor', 'donor']
        for label in dict.fromkeys(v for v in labels if type(v) is int and v > 0):
            scoped_programs = [p for p in programs if p.get('modality') != 'gapmer'
                and not (p.get('splice_event') or {}).get('event_type')
                and p.get('exon_label', goal.get('exon_label')) == label
                and (p.get('desired_effect') or default_effect) == effect]
            if (scope_id == 'goal' and not identity and scoped_programs
                    and goal.get('site_kind') is None and goal.get('exon_ordinal') is None
                    and all(p.get('transcript_id') or p.get('isoform_label') for p in scoped_programs)):
                # Global defaults belong to the explicitly scoped programs;
                # they do not create an extra unscoped cross-program request.
                continue
            minimum = scope.get('min_asos_per_site')
            own_minimum = minimum is not None
            if minimum is None:
                minimum = goal.get('min_asos_per_site')
            ordinal = scope.get('exon_ordinal')
            if ordinal is None and inherit and label == goal.get('exon_label'):
                ordinal = goal.get('exon_ordinal')
            result.append({'scope_id': scope_id, 'program_id': program_id,
                'reference': identity, 'exon_label': label, 'desired_effect': effect,
                'exon_ordinal': ordinal, 'site_kinds': kinds,
                'minimum': minimum,
                'minimum_program_id': program_id if own_minimum else None})
    return result


def _request_candidates(goal, request):
    catalog = goal.get('isoform_transcripts') or {}
    reference = request['reference']
    label = resolve_catalog_label(reference, catalog, catalog) if reference else None
    labels = [label] if label is not None else [] if reference else list(catalog)
    scoped = {**goal, 'exon_label': request['exon_label'],
        'desired_effect': request['desired_effect'], 'exon_ordinal': request['exon_ordinal'],
        'design_programs': [], 'exon_conversion_bindings': []}
    return scoped, [(label, event) for label in labels
        if (event := structural_exon_resolution(catalog[label], request['exon_label'], scoped))]


def _physical_event_key(event):
    return tuple(event[key] for key in ('chrom', 'strand', 'start_1based', 'end_1based'))


def bind_requested_exon_events(goal, resolve_ordinal):
    """Resolve conversions before route discovery can narrow their junctions."""
    catalog = goal.get('isoform_transcripts') or {}
    for request in conversion_requests(goal):
        scoped, candidates = _request_candidates(goal, request)
        bindings = []
        for label, _ in candidates:
            resolve_ordinal(catalog[label], request['exon_label'], scoped)
            bindings.extend(binding for binding in scoped['exon_conversion_bindings']
                if binding['catalog_label'] == label)
        physical = {_physical_event_key(event) for binding in bindings for event in binding['candidates']
                    if event['resolved_transcript_ordinal'] == binding.get('resolved_transcript_ordinal')}
        if len(physical) > 1:
            raise ExonConversionConflict(request['reference'] or 'resolved isoforms', request['exon_label'],
                [event for binding in bindings for event in binding['candidates']],
                'more than one physical cassette fits this scoped request')
        for binding in bindings:
            record_event_binding(goal, {**binding, 'request_scopes': [request]})


def program_event_bindings(goal, program):
    """Find current event references applicable to one typed design program."""
    if program.get('modality') == 'gapmer' or (program.get('splice_event') or {}).get('event_type'):
        return []
    catalog = goal.get('isoform_transcripts') or {}
    identity = program.get('transcript_id') or program.get('isoform_label')
    label = resolve_catalog_label(identity, catalog, catalog) if identity else None
    if identity and label is None:
        return []
    exon = program.get('exon_label')
    if type(exon) is not int:
        exon = goal.get('exon_label')
    effect = program.get('desired_effect') or goal.get('desired_effect')
    programs = [p for p in goal.get('design_programs') or [] if p.get('modality') != 'gapmer']
    bindings = []
    for binding in goal.get('exon_conversion_bindings') or []:
        if label is not None and binding.get('catalog_label') != label:
            continue
        if exon is not None and binding.get('requested_exon_label') != exon:
            continue
        if effect is not None and binding.get('desired_effect') != effect:
            continue
        scopes = binding.get('request_scopes') or []
        if not scopes:
            continue
        if exon is None and identity is None and len(programs) > 1 and not any(
                scope.get('program_id') in {program.get('program_id'), program.get('parent_program_id')} for scope in scopes
                if scope.get('program_id')):
            continue
        bindings.append(binding)
    return bindings


def requested_response_minimums(goal):
    """Derive expected response sites from requests and structures, never jobs.

    Missing event receipts remain explicit requirements that cannot be filled.
    This protects delivery even if a generated route or its job disappears.
    """
    groups = {}
    for request in conversion_requests(goal):
        minimum = request['minimum']
        if type(minimum) is not int or minimum <= 0:
            continue
        _, candidates = _request_candidates(goal, request)
        for label, event in candidates:
            bound = next((b for b in goal.get('exon_conversion_bindings') or []
                if b.get('catalog_label') == label
                and b.get('requested_exon_label') == request['exon_label']
                and b.get('desired_effect') == request['desired_effect']
                and b.get('reference_structure_sha256') == event['reference_structure_sha256']
                and b.get('status') in {'verified_structural_event', 'verified_nomenclature_disambiguated_event'}), None)
            ordinal = bound.get('resolved_transcript_ordinal') if bound else event.get('resolved_transcript_ordinal')
            selected = [e for e in event['candidates'] if e['resolved_transcript_ordinal'] == ordinal]
            if len(selected) != 1:
                groups['unresolved:' + request['scope_id']] = {
                    'group_id': 'unresolved:' + request['scope_id'], 'label': 'Unresolved exon response sites',
                    'minimum': minimum, 'site_scope': 'splice_response', 'reference_verified': False}
                continue
            physical = selected[0]
            for kind in request['site_kinds']:
                position = physical[kind + '_1based']
                key = json.dumps([goal.get('species'), goal.get('genome_assembly'),
                    physical['chrom'], physical['strand'], position, kind,
                    request['desired_effect'], request['minimum_program_id']])
                identifier = 'splice_response:' + hashlib.sha256(key.encode()).hexdigest()
                prior = groups.get(identifier)
                groups[identifier] = {'group_id': identifier,
                    'label': f"{kind}@{physical['chrom']}:{position} ({physical['strand']})",
                    'site_scope': 'splice_response', 'site_kind': kind,
                    'chrom': physical['chrom'], 'strand': physical['strand'], 'position': position,
                    'minimum': max(minimum, prior['minimum'] if prior else 0),
                    'program_id': request['minimum_program_id'],
                    'program_ids': [request['minimum_program_id'], *[
                        p.get('program_id') for p in goal.get('design_programs') or []
                        if p.get('parent_program_id') == request['minimum_program_id']]],
                    'reference_verified': bool(bound) and (prior['reference_verified'] if prior else True)}
    return list(groups.values())


def matches_response_site(row, group):
    if not group.get('reference_verified'):
        return False
    if (row.get('candidate_assessment') or {}).get('selection_eligible') is False:
        return False
    annotations = row.get('source_annotations') or {}
    response = row.get('predicted_response_site') or {}
    def value(key, response_key):
        return next((v for v in (row.get(key), annotations.get(key), response.get(response_key))
                     if v is not None and v != ''), None)
    position = value('selected_site_position', 'position')
    if isinstance(position, str) and position.isdigit():
        position = int(position)
    if type(position) is not int or position != group['position']:
        return False
    chrom = str(value('selected_site_chrom', 'chrom') or '').removeprefix('chr')
    strand = value('selected_site_strand', 'strand')
    strand = {'1': '+', '-1': '-'}.get(str(strand), strand)
    program = row.get('program_id') or annotations.get('program_id')
    return (chrom == group['chrom'].removeprefix('chr') and strand == group['strand']
            and value('selected_site_kind', 'site_kind') == group['site_kind']
            and (not group.get('program_id') or program in group.get('program_ids', [group['program_id']])))
