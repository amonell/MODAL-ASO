"""Reference-aware reconciliation of explicit splice targets.

Coordinates identify a physical feature; display labels are not identity.
Clarifications may supersede inferred interpretations, never unrelated targets.
"""
from copy import deepcopy
from .evaluation_feature_identity import transcript_identifier, transcript_matches


def strand_symbol(value):
    return {'+': '+', '1': '+', '-': '-', '-1': '-'}.get(str(value))


def chromosome(value):
    value = str(value or '').strip().casefold()
    return value[3:] if value.startswith('chr') else value


def target_references(target):
    values = set()
    for field in ('transcript_id', 'reference_transcript_id', 'exon_numbering_transcript_id'):
        value = str(target.get(field) or '').strip()
        if value:
            values.add(value.split('::', 1)[0])
    if not values and target.get('base_isoform_label'):
        values.add(str(target['base_isoform_label']).split('::', 1)[0])
    if not values and target.get('isoform_label'):
        values.add(str(target['isoform_label']).split('::', 1)[0])
    return values


def compatible_references(left, right):
    # Two specified versions must agree; one unversioned selector can bind
    # the corresponding version. A different reference is not a clarification.
    return transcript_matches(left, right) or transcript_matches(right, left)


def same_physical_feature(left, right):
    try:
        if any(isinstance(t.get(k), bool) for t in (left, right) for k in ('start', 'end')):
            return False
        a, b = (tuple(sorted((int(t['start']), int(t['end'])))) for t in (left, right))
    except (KeyError, TypeError, ValueError):
        return False
    if a != b or a[0] < 1 or a[0] == a[1]:
        return False
    if not chromosome(left.get('chrom')) or chromosome(left.get('chrom')) != chromosome(right.get('chrom')):
        return False
    refs_a, refs_b = target_references(left), target_references(right)
    if not refs_a or not refs_b:
        return False
    if not all(any(compatible_references(a, b) for b in refs_b) for a in refs_a):
        return False
    if not all(any(compatible_references(a, b) for a in refs_a) for b in refs_b):
        return False
    directions = {str(t.get('desired_delta') or '').lower() for t in (left, right)} - {''}
    return len(directions) <= 1


def inferred_target(target):
    return str(target.get('source') or '').lower() in {'', 'parsed', 'inferred', 'model_inferred'}


def apply_clarified_target(clarified, inferred):
    result = deepcopy(inferred)
    # These fields are consequences of the old interpretation, not user facts.
    for key in ('strand', 'strand_source', 'isoform_label', 'base_isoform_label',
                'mapped_start', 'mapped_end', 'mapping_mode', 'exon_label',
                'synthetic_exon_created', 'synthetic_exon_label', 'overlapped_exon_labels'):
        result.pop(key, None)
    result.update(deepcopy(clarified))
    old = deepcopy(inferred)
    history = old.pop('superseded_target_interpretations', [])
    result['superseded_target_interpretations'] = [*history, old]
    result['target_reconciliation'] = 'Clarified physical feature supersedes the earlier inferred interpretation; strand omitted by the user is resolved from its reference.'
    return result


def reconcile_clarified_targets(targets):
    items = [deepcopy(t) for t in targets if isinstance(t, dict)]
    replacements, consumed = {}, set()
    for index, item in enumerate(items):
        if not inferred_target(item):
            continue
        matches = [j for j, candidate in enumerate(items)
                   if candidate.get('source') == 'clarification_answers'
                   and same_physical_feature(item, candidate)]
        if len(matches) != 1:
            continue
        j = matches[0]
        replacements[j] = apply_clarified_target(replacements.get(j, items[j]), item)
        consumed.add(index)
    return [replacements.get(i, item) for i, item in enumerate(items) if i not in consumed]


def reconcile_declared_exon_spans(targets, spans):
    """Use a declared exon span instead of treating its boundaries as tiny exons.

    Only matching labels, references and exact endpoints bind. The complete span
    is supplied by a typed exon target, never inferred from prose or proximity.
    Every consumed input is retained as provenance.
    """
    items = [deepcopy(t) for t in targets if isinstance(t, dict)]
    spans = [deepcopy(s) for s in spans if isinstance(s, dict)]
    assigned = {}
    for index, target in enumerate(items):
        matches = []
        for slot, span in enumerate(spans):
            label = target.get('requested_exon_label', target.get('exon_label'))
            span_label = span.get('requested_exon_label', span.get('exon_label'))
            if label is None or str(label).casefold() != str(span_label).casefold():
                continue
            if chromosome(target.get('chrom')) != chromosome(span.get('chrom')):
                continue
            a,b = target.get('start'),target.get('end')
            left,right = span.get('start'),span.get('end')
            if not all(type(v) is int for v in (a,b,left,right)) or not 0 < left < right:
                continue
            if not ((a,b)==(left,right) or a==b and a in {left,right}):
                continue
            refs_a,refs_b=target_references(target),target_references(span)
            if not refs_b or (refs_a and not all(any(compatible_references(a,b) for b in refs_b) for a in refs_a)):
                continue
            strands={strand_symbol(t.get('strand')) for t in (target,span)}-{None}
            directions={t.get('desired_delta') for t in (target,span)}-{None,''}
            if len(strands)>1 or len(directions)>1:
                continue
            matches.append(slot)
        if len(matches)==1:
            assigned[index]=matches[0]
    result=[];emitted=set()
    for index,target in enumerate(items):
        slot=assigned.get(index)
        if slot is None:
            result.append(target)
        elif slot not in emitted:
            members=[items[i] for i,s in assigned.items() if s==slot]
            if len(members)==1 and members[0].get('target_reconciliation') == 'Exact boundaries reconciled with the declared complete exon span.':
                result.append(members[0]);emitted.add(slot)
                continue
            resolved=deepcopy(spans[slot])
            provenance=[]
            for member in members:
                for record in member.get('coalesced_boundary_targets') or [member]:
                    if record not in provenance:
                        provenance.append(record)
            resolved['coalesced_boundary_targets']=provenance
            resolved['target_reconciliation']='Exact boundaries reconciled with the declared complete exon span.'
            result.append(resolved);emitted.add(slot)
    return result


def bind_target_to_reference(target, entry):
    """Resolve only missing orientation; reject contradictory explicit geometry."""
    metadata = entry.get('metadata') or {}
    actual_strand = strand_symbol(entry.get('strand', metadata.get('strand')))
    requested_strand = strand_symbol(target.get('strand'))
    actual_chrom = chromosome(entry.get('chrom') or metadata.get('seq_region_name'))
    requested_chrom = chromosome(target.get('chrom'))
    reference = transcript_identifier(entry)
    if actual_chrom and requested_chrom and actual_chrom != requested_chrom:
        raise ValueError(f'Explicit splice target chromosome {target.get("chrom")} conflicts with reference {reference} chromosome {actual_chrom}; resolve the target reference before evaluation.')
    if actual_strand and requested_strand and actual_strand != requested_strand:
        raise ValueError(f'Explicit splice target strand {requested_strand} conflicts with reference {reference} strand {actual_strand}; clarify or repair the target interpretation before evaluation.')
    if actual_strand and requested_strand is None:
        target['strand'] = actual_strand
        target['strand_source'] = 'resolved_reference'
    return target


def prediction_geometry_status(site, entry):
    """Validate pair orientation without relabeling a cryptic site as annotation."""
    metadata = entry.get('metadata') or {}
    strand = strand_symbol(entry.get('strand', metadata.get('strand')))
    chrom = chromosome(entry.get('chrom') or metadata.get('seq_region_name'))
    points = [site.get(k) for k in ('acceptor', 'donor')]
    if not strand or not chrom or not all(isinstance(p, dict) for p in points):
        return 'unresolved'
    try:
        acceptor, donor = [int(p['position']) for p in points]
    except (KeyError, TypeError, ValueError):
        return 'unresolved'
    if any(strand_symbol(p.get('strand')) != strand or chromosome(p.get('chrom')) != chrom for p in points):
        return 'conflicting'
    if (strand == '+' and acceptor > donor) or (strand == '-' and acceptor < donor):
        return 'conflicting'
    # Arbitrary cryptic boundaries can be valid model evaluation sites. Only a
    # claimed exon ordinal implies equality with that model's exon boundaries.
    index = site.get('exon_index')
    exons = [e for e in entry.get('exons_tx') or [] if isinstance(e, dict) and str(e.get('index')) == str(index)] if index is not None else []
    if len(exons) == 1:
        try:
            span = sorted((int(exons[0]['start']), int(exons[0]['end'])))
        except (KeyError, TypeError, ValueError):
            return 'unresolved'
        expected = span if strand == '+' else span[::-1]
        if [acceptor, donor] != expected:
            return 'conflicting'
    elif index is not None:
        return 'unresolved'
    return 'consistent'
