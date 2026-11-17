"""Resolve typed transcript groups once, before scoring candidate sequences.

An accession anchors a feature; it does not replace the requested group.
Only reference-bound exon identities and transcript structures assign members.
No gene names, biological aliases, request prose, or model calls are used here.
"""
from copy import deepcopy

from .curated_exon_nomenclature import reference_context
from .exon_nomenclature_context import resolved_exon_requests, selected_reference
from .species_config import normalize_species


def group_objectives(goal):
    return [(index, objective) for index, objective in enumerate(goal.get('splice_isoform_objectives') or [])
            if isinstance(objective, dict) and (objective.get('objective_scope') == 'aggregate_class'
                                               or objective.get('aggregate_class'))]


def objective_roles(objective):
    direction = objective.get('desired_abundance') or objective.get('direction') or objective.get('goal')
    role = objective.get('role')
    roles = []
    if direction == 'decrease' or role == 'pathogenic':
        roles.append('target')
    if role == 'pathogenic':
        roles.append('pathogenic')
    if direction == 'increase' or role == 'beneficial':
        roles.append('beneficial')
    return roles


def _assembly(value):
    text = str(value or '').lower().split('.p', 1)[0]
    return {'hg38': 'grch38', 'hg19': 'grch37', 'mm10': 'grcm38', 'mm39': 'grcm39'}.get(text, text)


def exon_membership(entry, reference, feature):
    """Exact exon = included; an annotated bypass junction = excluded.

    Partial transcripts, retained introns, incompatible references and absent
    annotation are unknown, never evidence that an exon was skipped.
    """
    if not isinstance(entry, dict):
        return 'unknown'
    actual, expected = reference_context(entry), reference_context(reference)
    if (not actual['chrom'] or actual['strand'] is None
            or str(actual['chrom']).removeprefix('chr') != str(expected['chrom']).removeprefix('chr')
            or actual['strand'] != expected['strand']):
        return 'unknown'
    for field, normalize in (('assembly', _assembly), ('species', normalize_species)):
        if expected[field] and (not actual[field] or normalize(actual[field]) != normalize(expected[field])):
            return 'unknown'
    if actual['gene_symbol'] and expected['gene_symbol'] and actual['gene_symbol'].casefold() != expected['gene_symbol'].casefold():
        return 'unknown'
    exons = entry.get('exons_tx') or []
    if not exons or any(not isinstance(e, dict) or type(e.get('start')) is not int
                        or type(e.get('end')) is not int or not 1 <= e['start'] <= e['end'] for e in exons):
        return 'unknown'
    spans = sorted((e['start'], e['end']) for e in exons)
    if any(left[1] >= right[0] for left, right in zip(spans, spans[1:])):
        return 'unknown'
    start, end = feature['start'], feature['end']
    if (start, end) in spans:
        return 'included'
    if any(left <= end and right >= start for left, right in spans):
        return 'unknown'
    if any(left[1] < start <= end < right[0] for left, right in zip(spans, spans[1:])):
        return 'excluded'
    return 'unknown'


def resolve_groups(goal, catalog, resolved):
    """Return membership evidence for every declared group, in catalog space.

    Multiple exon predicates mean all requested exons must have the requested
    presence. Unsupported role/residual classes remain explicit unresolved
    requests; their names never become aliases for the reference transcript.
    """
    objectives = [(index, objective) for index, objective in group_objectives(goal) if objective_roles(objective)]
    if not objectives:
        return None
    from .gapmer_selectivity import resolve_catalog_label
    request_goal = {**goal, 'isoform_transcripts': resolved,
                    'splice_isoform_objectives': deepcopy(goal.get('splice_isoform_objectives') or [])}
    # External accession aliases require the same attested catalog join as
    # single-transcript roles. Never strip an accession version to force a match.
    for _, objective in group_objectives(request_goal):
        for key in ('reference_transcript_id', 'transcript_id', 'reference_accession', 'isoform_label'):
            if objective.get(key):
                label = resolve_catalog_label(objective[key], resolved, resolved)
                if label is not None:
                    objective[key] = reference_context(resolved[label])['transcript_id']
    requests = resolved_exon_requests(request_goal)
    groups = []
    for index, objective in objectives:
        specification = objective.get('aggregate_class') or {}
        presence = specification.get('presence')
        path = f'/splice_isoform_objectives/{index}'
        group = {'request_path': path, 'label': objective.get('isoform_label') or specification.get('label'),
                 'roles': objective_roles(objective), 'definition': deepcopy(specification),
                 'status': 'unresolved', 'members': [], 'nonmembers': [], 'unknown': sorted(catalog),
                 'features': []}
        groups.append(group)
        if specification.get('basis') != 'exon_presence' or presence not in {'included', 'excluded'}:
            group['reason'] = 'This group has no supported structural membership predicate.'
            continue
        features = [r for r in requests if r['request_path'] == path + '/aggregate_class']
        # Older typed declarations can put the exon on the objective itself.
        if not features:
            features = [r for r in requests if r['request_path'] == path]
        if not features or any(r['status'] == 'unresolved' for r in features):
            group['reason'] = 'The requested exon identity is unresolved; its reference cannot substitute for the group.'
            continue
        references = [selected_reference(resolved, f['reference_transcript_id'])[1] for f in features]
        if any(entry is None for entry in references):
            group['reason'] = 'The exon numbering reference is unavailable or ambiguous.'
            continue
        group['features'] = deepcopy(features)
        group['unknown'] = []
        for label, entry in catalog.items():
            states = [exon_membership(entry, reference, feature) for reference, feature in zip(references, features)]
            if all(state == presence for state in states):
                group['members'].append(label)
            elif any(state not in {presence, 'unknown'} for state in states):
                group['nonmembers'].append(label)
            else:
                group['unknown'].append(label)
        for key in ('members', 'nonmembers', 'unknown'):
            group[key].sort()
        group['status'] = 'partial' if group['unknown'] else 'resolved'
        if not group['members']:
            group.update(status='unresolved', reason='No sequence-backed catalog member satisfies this group definition.')
    return {'status': 'unresolved' if any(g['status'] == 'unresolved' for g in groups)
            else 'partial' if any(g['status'] == 'partial' for g in groups) else 'resolved',
            'catalog_size': len(catalog), 'groups': groups,
            'basis': 'Exact resolved exon spans and annotated exon-bypass junctions in the sequence-backed transcript catalog. '
                     'Partial or incompatible structures remain unknown; catalog coverage is not expression-weighted.'}


def apply_group_roles(roles, resolution):
    """Merge declared group members with independently resolved single RNAs."""
    if not resolution:
        return roles
    roles = deepcopy(roles)
    for role in ('target', 'beneficial', 'pathogenic'):
        groups = [g for g in resolution['groups'] if role in g['roles']]
        if not groups:
            continue
        record = roles[role]
        record['catalog_labels'] = sorted(set(record['catalog_labels']).union(*(g['members'] for g in groups)))
        record['group_requests'] = [g['request_path'] for g in groups]
        if any(g['status'] == 'unresolved' for g in groups):
            record['status'] = 'unresolved'
    conflicts = set(roles['target']['catalog_labels']) & set(roles['beneficial']['catalog_labels'])
    if conflicts:
        roles['target']['status'] = roles['beneficial']['status'] = 'unresolved'
        roles['conflicting_target_and_beneficial_labels'] = sorted(conflicts)
    roles['status'] = 'unresolved' if any(roles[r]['status'] == 'unresolved' for r in ('target', 'beneficial', 'pathogenic')) else 'resolved'
    return roles


def group_summary(resolution):
    """Compact row/export evidence; the full membership table is saved once."""
    if not resolution:
        return None
    return [{key: deepcopy(g[key]) for key in ('request_path', 'label', 'roles', 'definition', 'status', 'features', 'reason') if key in g}
            | {key + '_count': len(g[key]) for key in ('members', 'nonmembers', 'unknown')}
            for g in resolution['groups']]
