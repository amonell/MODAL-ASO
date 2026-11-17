"""Plain, explicit target-set provenance for current and historical scores."""


def target_context(source):
    roles = source.get('isoform_role_resolution') or {}
    definitions = source.get('isoform_group_definitions') or []
    target = roles.get('target') or {}
    labels = target.get('catalog_labels') or []
    counts = {key: source.get(field) for key, field in (
        ('target', 'target_isoform_match_count'), ('protected', 'beneficial_isoform_match_count'),
        ('other', 'other_isoform_match_count'), ('unresolved', 'unclassified_isoform_match_count'))}
    if definitions:
        groups = [g for g in definitions if 'target' in (g.get('roles') or [])]
        names = [g.get('label') or 'Requested transcript group' for g in groups]
        text = 'Target group: ' + '; '.join(names) + '.' if names else 'Saved transcript-group definitions apply.'
        size = source.get('target_isoform_catalog_size')
        if size is not None:
            text += f' {size} confirmed target transcripts in the catalog.'
    elif source.get('isoform_uniqueness_class') == 'gene_wide':
        text = 'Gene-wide coverage across the saved transcript catalog.'
    elif labels:
        text = ('Single-transcript selectivity against ' if len(labels) == 1 else 'Selectivity for the saved target set: ')
        text += ', '.join(labels[:5]) + (f' and {len(labels)-5} more transcripts' if len(labels)>5 else '') + '.'
    elif source.get('isoform_uniqueness_score') is not None:
        text = 'The saved score has no recorded target-set definition.'
    else:
        return None
    parts = [f'{count} {key}' for key, count in counts.items() if count is not None]
    if parts:
        text += ' Matching transcripts: ' + ', '.join(parts) + '.'
    if roles.get('status') == 'unresolved':
        text += 'Target-set assignment remains unresolved.'
    return {'text': text, 'roles': roles, 'groups': definitions, 'match_counts': counts,
            'basis': source.get('isoform_score_definition'), 'ranking_status': source.get('isoform_ranking_status')}
