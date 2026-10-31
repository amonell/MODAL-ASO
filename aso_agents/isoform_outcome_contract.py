"""Keep every declared isoform outcome in the executable objective catalog."""
from copy import deepcopy


def _identities(record):
    aliases = {('identity', str(record.get(key)).strip().casefold()) for key in
            ('isoform_label', 'transcript_id', 'reference_transcript_id',
             'reference_accession')
            if record.get(key)}
    # Filesystem paths are case-sensitive and are not transcript aliases.
    return aliases | {('artifact', str(record[key]).strip()) for key in
                      ('artifact_ref_path', 'snapgene_path') if record.get(key)}


def complete_isoform_outcomes(goal):
    """Reconcile typed outcomes without inferring direction from prose or filenames.

    A descriptive label on an explicitly typed isoform target is an unresolved
    isoform identity, not permission to drop that target. Its identity must
    still pass the ordinary transcript/artifact resolver before design.
    """
    objectives = deepcopy(goal.get('splice_isoform_objectives') or [])
    targets = deepcopy(goal.get('evaluation_targets') or [])
    changed = False
    gene = str(goal.get('gene_symbol') or '').casefold()
    for target in targets:
        if (not isinstance(target, dict) or target.get('target_type') not in {'isoform', 'transcript'}
                or target.get('objective_type') != 'primary_outcome'
                or target.get('desired_delta') not in {'increase', 'decrease'}):
            continue
        target_gene = str(target.get('gene_symbol') or '').casefold()
        if gene and target_gene and gene != target_gene:
            continue
        label = (target.get('isoform_label') or target.get('transcript_id')
                 or target.get('reference_transcript_id') or target.get('label')
                 or target.get('artifact_ref_path'))
        if not isinstance(label, str) or not label.strip():
            raise ValueError('declared_isoform_outcome_identity_missing')
        candidate = {'isoform_label': label.strip(),
                     'desired_abundance': target['desired_delta'],
                     'objective_scope': 'specific_isoform'}
        for key in ('transcript_id', 'artifact_ref_path'):
            if target.get(key):
                candidate[key] = target[key]
        if not candidate.get('transcript_id') and target.get('reference_transcript_id'):
            candidate['transcript_id'] = target['reference_transcript_id']
        if candidate.get('transcript_id') or candidate.get('artifact_ref_path'):
            candidate['objective_scope'] = 'specific_transcript'
        matches = [item for item in objectives if isinstance(item, dict)
                   and _identities(item) & _identities(candidate)]
        if not target.get('isoform_label'):
            target['isoform_label'] = ((matches[0].get('isoform_label') if len(matches) == 1 else None)
                                      or candidate['isoform_label'])
            changed = True
        if matches:
            for item in matches:
                if item.get('desired_abundance') != candidate['desired_abundance']:
                    raise ValueError('declared_isoform_outcome_direction_conflict: ' + label)
                for key in ('transcript_id', 'artifact_ref_path'):
                    if not candidate.get(key):
                        continue
                    if item.get(key) and item[key] != candidate[key]:
                        raise ValueError('declared_isoform_outcome_reference_conflict: ' + label)
                    if not item.get(key):
                        item[key] = candidate[key]
                        changed = True
            continue
        objectives.append(candidate)
        changed = True
    if changed:
        goal['splice_isoform_objectives'] = objectives
        goal['evaluation_targets'] = targets
    return changed
