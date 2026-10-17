"""Resolve protected references without turning preservation into upregulation."""
from copy import deepcopy


def preservation_resolution_objectives(goal):
    objectives = []
    gene = str(goal.get('gene_symbol') or '').casefold()
    seen = set()
    for target in goal.get('preservation_targets') or []:
        if not isinstance(target, dict) or target.get('action') != 'preserve':
            continue
        target_gene = str(target.get('gene_symbol') or '').casefold()
        if gene and target_gene and gene != target_gene:
            continue  # Another gene must not resolve against this gene's catalog.
        kind = target.get('preservation_target_type')
        if kind not in {'transcript', 'isoform'}:
            continue
        label = str(target.get('isoform_label') or target.get('transcript_id') or '').strip()
        if not label or label in seen:
            continue
        seen.add(label)
        item = {'isoform_label': label, 'objective_scope': 'specific_transcript' if kind == 'transcript' else 'specific_isoform',
                'role': 'beneficial', 'resolution_purpose': 'preservation_reference'}
        if target.get('transcript_id'):
            item['transcript_id'] = target['transcript_id']
        objectives.append(item)
    return objectives


def preservation_reference_context(goal, catalog):
    """Match only attested catalog aliases, preserving unresolved/version conflicts."""
    references = []
    gene = str(goal.get('gene_symbol') or '').casefold()
    for target in goal.get('preservation_targets') or []:
        if not isinstance(target, dict):
            continue
        identities = [str(target[k]).casefold() for k in ('isoform_label', 'transcript_id') if target.get(k)]
        matches = []
        target_gene = str(target.get('gene_symbol') or '').casefold()
        same_gene = not (gene and target_gene and gene != target_gene)
        for label, entry in catalog.items() if same_gene else []:
            metadata = entry.get('metadata') or {}
            aliases = {str(v).casefold() for v in (label, entry.get('transcript_id'), entry.get('versioned_id'),
                       metadata.get('id'), metadata.get('versioned_id')) if v}
            if identities and all(identity in aliases for identity in identities):
                matches.append(label)
        references.append({'requested_target': deepcopy(target), 'resolved_labels': matches,
                           'resolution_status': 'resolved' if len(matches) == 1 else 'unresolved',
                           'interpretation': 'Preservation is a constraint, not a request to increase abundance. Shared sequence/site geometry requires a comparative mechanism assessment; it does not prove activity or harm.'})
    return references


def preserved_catalog_labels(goal, catalog):
    return {label for ref in preservation_reference_context(goal, catalog)
            if ref['resolution_status'] == 'resolved' for label in ref['resolved_labels']}
