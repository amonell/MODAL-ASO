"""Bound response-composer inputs while retaining full comparison artifacts."""
from copy import deepcopy
import math


_METRIC_PATHS = {
    'isoform_hit_count': ('isoform_hit_count', 'recorded_scientific_fields.isoform_hit_count'),
    'isoform_coverage_fraction': ('recorded_scientific_fields.isoform_coverage_fraction',
        'recorded_scientific_fields.selection_score_components.isoform_coverage_fraction'),
    'off_target_gene_count': ('recorded_scientific_fields.off_target_gene_count',),
    'phastcons': ('phastcons', 'recorded_scientific_fields.selection_score_components.phastcons'),
    'phylop': ('phylop', 'recorded_scientific_fields.selection_score_components.phylop'),
}


def _measured(row, paths):
    for path in paths:
        value = row
        for key in path.split('.'):
            value = value.get(key) if isinstance(value, dict) else None
        if value is None or isinstance(value, bool):
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue
        if math.isfinite(value):
            return value
    return None


def annotate_comparison_attributes(payload):
    """Summarize actual examined rows, including exceptions and missing values."""
    for entry in payload.get('candidate_evidence_by_set') or []:
        rows = entry.get('top_rows_in_source_order')
        if not isinstance(rows, list):
            continue  # Preserve other evidence schemas without invented scope.
        metrics = {}
        for name, paths in _METRIC_PATHS.items():
            values = [_measured(row, paths) for row in rows]
            measured = [v for v in values if v is not None]
            metrics[name] = {'examined_rows': len(rows), 'measured_rows': len(measured),
                'unknown_rows': len(rows) - len(measured),
                'minimum': min(measured) if measured else None,
                'maximum': max(measured) if measured else None,
                'all_examined_rows_same_measured_value': bool(measured) and len(measured) == len(rows)
                    and min(measured) == max(measured), 'source_field_paths': list(paths)}
        entry['recorded_attribute_summaries'] = metrics
        if entry.get('source_type') == 'generated':
            entry['attribute_definitions'] = {
                'isoform_hit_count': 'Number of annotated target-gene isoforms containing the recorded match within this set\'s catalog. This raw count is distinct from the fraction of the catalog; a higher fraction does not imply a higher count when catalog sizes differ.',
                'phastcons_and_phylop': 'Conservation of the genomic bases covered by each target tile, projected through its exon blocks for mature RNA; not a separate flanking window. Means over available annotated bases do not imply full coverage.',
                'isoform_coverage_fraction': 'Fraction of the recorded per-gene catalog with the reported sequence match. Catalog universes can differ across sets; this is not measured knockdown.',
                'off_target_gene_count': 'Recorded distinct-gene screening metric within its source search scope; do not substitute locus or transcript counts, or infer safety from a zero.',
            }
    return payload


def response_comparison_context(payload, sample_limit=20):
    """The full unique/shared row inventories are exports, not prompt context.

    Numerical summaries and candidate evidence per set remain intact. Samples
    carry explicit counts/scope so the composer cannot describe them as a census.
    """
    result = {}
    inventory_fields = {'unique_by_set', 'shared_membership', 'provided_matches', 'provided_aso_matches'}
    for key, value in payload.items():
        if key not in inventory_fields:
            result[key] = deepcopy(value)
        elif isinstance(value, dict):
            result[key] = {name: _inventory_summary(rows, sample_limit) for name, rows in value.items()}
        else:
            result[key] = _inventory_summary(value, sample_limit)
    result['evidence_scope'] = ('Full row inventories remain in the saved comparison artifact. '
        'The response receives counts and bounded identifier samples plus the recorded top-row evidence for every set. '
        'Do not describe an examined subset as all candidates. Compare sets under explicit shared criteria; '
        'cross-gene catalog coverage and within-set ranks are not calibrated potency.')
    return annotate_comparison_attributes(result)


def _inventory_summary(rows, limit):
    if not isinstance(rows, list):
        return deepcopy(rows)
    identifiers = []
    for row in rows[:limit]:
        if isinstance(row, dict):
            identifiers.append({key: deepcopy(row[key]) for key in
                ('name', 'tile_id', 'sequence', 'Sequence', 'Synthesis', 'set_id', 'set_ids') if key in row})
        else:
            identifiers.append(deepcopy(row))
    return {'row_count':len(rows), 'identifier_sample':identifiers,
            'sample_limit':limit, 'sample_scope':'first rows in recorded source order; not a new ranking'}


def comparison_binding_evidence(row):
    """Keep strong and weak alignment evidence separately from string overlap.

    Each reference/organism, RNA context and orientation retains its numerical
    range and best recorded example. Equal examples are counted, not padded into
    a larger sample. Original indexes identify the complete saved observations.
    """
    groups = {}
    for field in ('mapping_hits', 'near_match_candidates'):
        for index, hit in enumerate(row.get(field) or []):
            identity = hit.get('reference_identity') or {}
            reference = identity.get('sequence_evidence') or identity
            key = (reference.get('species'), reference.get('assembly'),
                   hit.get('rna_context') or hit.get('mapping_scope'), hit.get('match_orientation'), field)
            group = groups.setdefault(key, {'species': key[0], 'assembly': key[1],
                'rna_context': key[2], 'match_orientation': key[3], 'source_field': field,
                'recorded_alignment_count': 0, 'transcript_ids': [], 'minimum_mismatches': None,
                'maximum_mismatches': None, 'minimum_example_count': 0, 'representative': None})
            group['recorded_alignment_count'] += 1
            if hit.get('transcript_id') not in group['transcript_ids']:
                group['transcript_ids'].append(hit.get('transcript_id'))
            mismatches = hit.get('mismatch_count', 0 if field == 'mapping_hits' else None)
            if isinstance(mismatches, bool) or not isinstance(mismatches, int):
                continue
            group['maximum_mismatches'] = max(group['maximum_mismatches'] or 0, mismatches)
            if group['minimum_mismatches'] is None or mismatches < group['minimum_mismatches']:
                group['minimum_mismatches'] = mismatches
                group['minimum_example_count'] = 1
                identity = hit.get('reference_identity') or {}
                representative_fields = (
                    'transcript_id', 'mapping_scope', 'rna_context', 'match_orientation',
                    'binding_interpretation', 'matched_target_sequence', 'matched_window_sequence',
                    'cdna_start', 'cdna_end', 'genomic_segments', 'genomic_span', 'strand',
                    'mismatch_count', 'mismatch_positions_in_supplied_sequence_1based',
                    'search_scope', 'requested_site_overlaps',
                )
                representative = {field: deepcopy(hit[field]) for field in representative_fields
                                  if field in hit}
                representative['reference_identity'] = {
                    field: deepcopy(identity[field]) for field in
                    ('transcript_id', 'versioned_id', 'species', 'assembly', 'reference_source')
                    if field in identity}
                group['representative'] = {'source_index': index, **representative}
            elif mismatches == group['minimum_mismatches']:
                group['minimum_example_count'] += 1
    return {'search': deepcopy(row.get('near_match_search') or {}),
            'groups': list(groups.values()),
            'orientation_interpretation': deepcopy(row.get('orientation_interpretation') or {}),
            'scope': 'Recorded reference alignments only. Best examples are grouped by organism, assembly, RNA context and orientation; full records remain at source_field/source_index in the supplied-ASO evaluation. A full-length Hamming match is distinct from a short contiguous overlap with sampled generated candidates. Neither proves activity; an absent exact match is not absence of near binding, and an unsearched reference remains unknown.'}
