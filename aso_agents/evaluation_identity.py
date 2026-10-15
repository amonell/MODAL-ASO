"""Bind evaluation screening to the primary transcript's reference identity."""
from copy import deepcopy

from .species_config import normalize_species
from .transcript_reference import transcript_reference


def declared_evaluation_context(state, envelope, task_id):
    """Separate declared task facts from defaults added by runtime normalization.

    Without the canonical task, keep the supplied envelope authoritative. This
    conservative fallback must not silently repair an explicit contradiction.
    """
    canonical = state.get('canonical_request_envelope') or {}
    matches = [task for task in canonical.get('tasks') or [] if isinstance(task, dict)
               and (task.get('task_id') == task_id
                    or task_id in (task.get('merged_from_task_ids') or []))]
    if len(matches) == 1:
        task = matches[0]
        context = deepcopy(task)
        context.update(deepcopy(task.get('context') or {}))
        if not any(context.get(key) for key in ('species', 'gene_id', 'gene_symbol', 'transcript_id', 'transcript_lock')):
            inherited = deepcopy(canonical.get('default_context') or {})
            inherited.update({key: deepcopy(canonical[key]) for key in
                              ('species', 'gene_id', 'gene_symbol', 'transcript_id', 'transcript_lock')
                              if canonical.get(key)})
            inherited.update(context)
            context = inherited
        return context, 'canonical_task'
    return deepcopy(envelope), 'supplied_evaluation_envelope'


def _gene(entry):
    metadata = entry.get('metadata') or {}
    values = {str(value).strip() for value in (entry.get('gene_id'), metadata.get('gene_id')) if value}
    if len({value.split('.')[0] for value in values}) > 1:
        raise ValueError('conflicting_transcript_reference_genes')
    return str(metadata.get('gene_id') or entry.get('gene_id') or '').strip()


def bind_primary_evaluation_identity(goal, entry, declared):
    """Mutate only the evaluation copy; reject conflicts with explicit task facts.

    Organism and gene come from resolved reference metadata, never accession
    prefixes, gene-name casing, prose, or the preceding design task.
    """
    if not entry:
        return {'status': 'not_resolved', 'reason': 'primary_transcript_unavailable'}
    previous = {key: deepcopy(goal.get(key)) for key in ('species', 'gene_id', 'gene_symbol')}
    reference = transcript_reference(entry, goal.get('species'))
    species = reference['species']
    gene_id = _gene(entry)
    if (reference['species_basis'] == 'reference_metadata' and declared.get('species')
            and normalize_species(declared['species']) != species):
        raise ValueError('evaluation_species_conflicts_with_primary_transcript')
    if (gene_id and declared.get('gene_id')
            and str(declared['gene_id']).split('.')[0] != gene_id.split('.')[0]):
        raise ValueError('evaluation_gene_conflicts_with_primary_transcript')
    species_changed = normalize_species(goal.get('species')) != species
    gene_changed = bool(gene_id
                        and str(previous['gene_id'] or '').split('.')[0] != gene_id.split('.')[0])
    rebound = species_changed or gene_changed
    if species_changed and goal.get('patient_reference') and declared.get('patient_reference'):
        if not declared.get('species') or normalize_species(declared['species']) != species:
            raise ValueError('evaluation_patient_reference_species_unverified')
    cleared = []
    if rebound:
        # These are derived from the old primary identity. Keep user-supplied
        # task fields, but never reuse a previous gene catalog as a complete one.
        for key in ('gene_id_resolution', 'knockdown_isoform_catalog'):
            if key in goal:
                cleared.append(key)
                goal.pop(key)
        if not gene_id:
            goal.pop('gene_id', None)
            cleared.append('gene_id')
        if not declared.get('gene_symbol'):
            goal.pop('gene_symbol', None)
            cleared.append('gene_symbol')
        if species_changed and goal.get('patient_reference'):
            if declared.get('patient_reference'):
                goal['patient_reference'] = deepcopy(declared['patient_reference'])
            else:
                goal.pop('patient_reference')
                cleared.append('patient_reference')
    goal['species'] = species
    if gene_id:
        goal['gene_id'] = gene_id
    if declared.get('gene_symbol'):
        goal['gene_symbol'] = declared['gene_symbol']
    return {
        'status': 'bound', 'reference': reference, 'gene_id': gene_id or None,
        'gene_basis': 'reference_metadata' if gene_id else 'evaluation_context',
        'previous_context': previous, 'inherited_identity_changed': rebound,
        'cleared_inherited_fields': cleared,
    }


def screening_catalog_labels(catalog, primary_label, goal, binding):
    """Keep foreign references for mapping, outside primary isoform scoring."""
    primary_gene = str(goal.get('gene_id') or '').split('.')[0]
    rebound = binding.get('inherited_identity_changed', False)
    labels = set()
    for label, entry in catalog.items():
        if label == primary_label:
            labels.add(label)
            continue
        try:
            reference = transcript_reference(entry, goal.get('species'))
            gene_id = _gene(entry).split('.')[0]
        except ValueError:
            continue
        if reference['species'] != normalize_species(goal.get('species')):
            continue
        if primary_gene and gene_id and primary_gene != gene_id:
            continue
        if rebound and (reference['species_basis'] != 'reference_metadata'
                        or not primary_gene or gene_id != primary_gene):
            continue
        labels.add(label)
    return labels
