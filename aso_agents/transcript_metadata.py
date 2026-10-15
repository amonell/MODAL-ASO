"""Bind structural and annotation metadata to the resolved reference record."""
from copy import deepcopy


_ANNOTATION_FLAG_ALIASES = {
    'is_canonical': ('is_canonical',),
    'is_mane_select': ('is_mane_select', 'mane_select', 'MANE_Select'),
    'is_mane_plus_clinical': ('is_mane_plus_clinical', 'mane_plus_clinical', 'MANE_Plus_Clinical'),
}


def _annotation_boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return {'true': True, 'false': False, '1': True, '0': False}.get(value.strip().lower())
    return None


def reference_annotation_flags(payload):
    """Preserve explicit annotation flags; absence and conflicts remain unknown."""
    result = {}
    for name, aliases in _ANNOTATION_FLAG_ALIASES.items():
        values = [_annotation_boolean(payload[key]) for key in aliases
                  if key in payload and payload[key] is not None]
        result[name] = values[0] if values and None not in values and len(set(values)) == 1 else None
    return result


def resolved_transcript_metadata(structure, selection_record=None):
    previous = {key: deepcopy(value) for key, value in (selection_record or {}).items() if key != 'Exon'}
    metadata = deepcopy(previous)
    structural_fields = {'id', 'gene_id', 'gene_symbol', 'gene_name', 'transcript_id', 'version', 'versioned_id', 'reference_source',
        'reference_annotation', 'species', 'assembly_name', 'seq_region_name', 'strand', 'start', 'end', 'length',
        'exon_count', 'cdna_length', 'coding_bp_length', 'translation_id', 'protein_length_aa'}
    annotation_aliases = {key for aliases in _ANNOTATION_FLAG_ALIASES.values() for key in aliases}
    for key in list(metadata):
        if key in structural_fields or key in annotation_aliases or key.startswith(('five_prime_terminal_exon_', 'three_prime_terminal_exon_')):
            metadata.pop(key)
    for key in structural_fields:
        if structure.get(key) is not None:
            metadata[key] = deepcopy(structure[key])
    annotation_source = structure.get('annotation_features', structure)
    metadata.update(reference_annotation_flags(annotation_source if isinstance(annotation_source, dict) else {}))
    metadata['annotation_features_basis'] = ('Annotation flags from the same reference record as the resolved structure. '
        'Missing, malformed or conflicting flags are unknown, not false. Earlier lookup flags remain only in selection_snapshot.')
    exons = structure.get('exons_tx') or []
    valid = bool(exons) and all(isinstance(exon, dict) and all(
        isinstance(exon.get(key), int) and not isinstance(exon.get(key), bool) for key in ('start', 'end'))
        and 1 <= exon['start'] <= exon['end'] for exon in exons)
    strand = {1: '+', -1: '-', '1': '+', '-1': '-'}.get(structure.get('strand'), structure.get('strand'))
    if valid:
        start = min(exon['start'] for exon in exons)
        end = max(exon['end'] for exon in exons)
        metadata.update(start=start, end=end, length=end-start+1, exon_count=len(exons),
                        cdna_length=sum(exon['end']-exon['start']+1 for exon in exons))
        if strand in {'+', '-'}:
            ordered = sorted(exons, key=lambda exon: exon['start'], reverse=strand == '-')
            for name, exon in [('five_prime', ordered[0]), ('three_prime', ordered[-1])]:
                metadata.update({name+'_terminal_exon_start': exon['start'],
                                 name+'_terminal_exon_end': exon['end'],
                                 name+'_terminal_exon_length': exon['end']-exon['start']+1})
            metadata['three_prime_terminal_exon_signature'] = f"{ordered[-1]['start']}-{ordered[-1]['end']}"
        coding = [exon for exon in exons if exon.get('coding_start_genomic') is not None
                  and exon.get('coding_end_genomic') is not None]
        if coding:
            metadata['coding_bp_length'] = sum(int(exon['coding_end_genomic'])-int(exon['coding_start_genomic'])+1 for exon in coding)
    if previous:
        metadata['selection_snapshot'] = {'scope': 'Earlier transcript-selection metadata, retained as provenance. Structural measurements and annotation flags of the resolved reference are the current top-level fields and exons_tx; earlier values must not be attributed to that reference.',
                                          'record': previous}
    metadata['structural_metadata_basis'] = 'Resolved exons_tx and its reference identity; not the earlier selection lookup.'
    return metadata
