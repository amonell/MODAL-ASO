"""Pause consequential exon requests until their physical identity is known.

Uses typed goals and reference geometry. No request-text interpretation and no
implicit conversion of a biological exon name to transcript order.
"""
from copy import deepcopy

from .curated_exon_nomenclature import reference_context, reference_fingerprint


def _assembly(value):
    value = str(value or '').casefold().split('.p', 1)[0]
    return {'hg38': 'grch38', 'hg19': 'grch37', 'mm10': 'grcm38', 'mm39': 'grcm39'}.get(value, value)


def coordinate_exon(physical, entry):
    """Match a complete user-supplied exon in the selected reference space."""
    context = reference_context(entry)
    assembly = context['assembly']
    if not assembly and context['modified']:
        assembly = ((entry.get('metadata') or {}).get('base_reference_snapshot') or {}).get('assembly_name')
    if (str(physical.get('chrom') or '').removeprefix('chr') != str(context['chrom'] or '').removeprefix('chr')
            or physical.get('strand') not in (None, '.', context['strand'])
            or (physical.get('assembly') and _assembly(physical['assembly']) != _assembly(assembly))):
        return None
    if any(type(physical.get(key)) is not int for key in ('start', 'end')):
        return None
    matches = [exon for exon in entry.get('exons_tx') or []
               if (exon.get('start'), exon.get('end')) == (physical['start'], physical['end'])]
    return matches[0] if len(matches) == 1 else None


def _identity_key(item):
    return (item.get('reference_transcript_id') or item.get('requested_reference'),
            str(item['requested_exon_label']) if item.get('requested_exon_label') is not None
            else ('annotation_order', item.get('explicit_exon_index')))


def share_explicit_identities(items):
    """Share an answer across repeated declarations of the same scoped feature."""
    explicit = [deepcopy(item) for item in items if item['status'] in {
        'explicit_transcript_ordinal', 'explicit_feature_coordinates'}]
    for item in items:
        reference, feature = _identity_key(item)
        matches = [other for other in explicit if _identity_key(other)[1] == feature
                   and (reference is None or reference == _identity_key(other)[0]
                        or reference in (other.get('identity_reference_aliases') or []))
                   and (not other.get('program_id') or not item.get('program_id')
                        or other['program_id'] == item['program_id'])]
        coordinates = {(other['reference_transcript_id'], other['start'], other['end']) for other in matches}
        if len(coordinates) > 1:
            item.update(status='unresolved', biological_label_verified=False,
                        reason='The supplied definitions identify different physical exons.')
            for key in ('start', 'end', 'resolved_transcript_ordinal'):
                item.pop(key, None)
        elif (matches and not item.get('identity_conflict') and item.get('explicit_exon_index') is None
              and item.get('explicit_exon_coordinates') is None):
            other = matches[0]
            for key in ('status', 'biological_label_verified', 'resolved_transcript_ordinal',
                        'reference_transcript_id', 'chrom', 'strand', 'start', 'end', 'coordinate_system'):
                item[key] = deepcopy(other[key])
            item['identity_answer_path'] = other['request_path']
            item.pop('reason', None)


class ExonIdentityClarification(ValueError):
    def __init__(self, goal, unresolved):
        self.features = deepcopy(unresolved)
        self.reason = 'The requested exon needs a confirmed physical location before this run can continue.'
        self.questions = []
        seen = set()
        gene = goal.get('gene_symbol') or goal.get('gene_id') or 'the target gene'
        for item in unresolved:
            key = _identity_key(item)
            if key in seen:
                continue
            seen.add(key)
            label = item.get('requested_exon_label')
            reference = item.get('reference_transcript_id') or item.get('requested_reference')
            feature = f'exon {label}' if label is not None else f'the requested transcript-order exon {item.get("explicit_exon_index")}'
            location = f' on {reference}' if reference else ''
            self.questions.append(
                f'Which physical exon do you mean by {feature} in {gene}{location}? '
                'Please provide its complete genomic coordinates and genome assembly, '
                'or explicitly identify its exon number counted from the 5′ end of a specified transcript.')
        super().__init__('exon_identity_clarification_required: ' + self.reason)


def require_resolved_exon_identities(goal):
    """Gate design and review, preserving offline browsing of any exon chain."""
    from .exon_nomenclature_context import resolved_exon_requests, selected_reference
    items = resolved_exon_requests(goal)
    by_identity = {}
    for item in items:
        if item['status'] != 'unresolved':
            by_identity.setdefault(_identity_key(item), set()).add((item['start'], item['end']))
    for item in items:
        if len(by_identity.get(_identity_key(item), set())) > 1:
            item.update(status='unresolved', reason='The same named exon has conflicting physical definitions.')
    unresolved = [item for item in items if item['status'] == 'unresolved']
    if unresolved:
        raise ExonIdentityClarification(goal, unresolved)
    # Store execution identities separately from public biological nomenclature.
    # Consumers that only receive a reference can use an explicitly answered
    # location without changing the user's original requested name.
    catalog = goal.get('isoform_transcripts') or {}
    for entry in catalog.values():
        if isinstance(entry, dict):
            entry.setdefault('metadata', {}).pop('execution_exon_identities', None)
    for item in items:
        _, entry = selected_reference(catalog, item.get('reference_transcript_id'))
        if entry is None:
            continue
        context = reference_context(entry)
        saved = {**deepcopy(item), 'reference_fingerprint': reference_fingerprint(
            context['transcript_id'], entry.get('exons_tx') or [], context)}
        entry.setdefault('metadata', {}).setdefault('execution_exon_identities', []).append(saved)


def execution_exon_identity(entry, label):
    """Read one unambiguous, still-current execution identity from a reference."""
    context = reference_context(entry)
    fingerprint = reference_fingerprint(context['transcript_id'], entry.get('exons_tx') or [], context)
    matches = [item for item in (entry.get('metadata') or {}).get('execution_exon_identities') or []
               if str(item.get('requested_exon_label')) == str(label)
               and item.get('reference_fingerprint') == fingerprint and item.get('status') != 'unresolved']
    return matches[0] if len({item.get('resolved_transcript_ordinal') for item in matches}) == 1 else None
