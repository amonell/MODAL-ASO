"""Explicit sequence roles and count units for generated-candidate reports.

These helpers never alter scientific measurements or manufacture a molecule from
an ambiguous source string. Legacy sequence fields retain their original bases.
"""
import re
from typing import Any, Dict


def _plain_bases(value: Any) -> str:
    value = str(value or '').strip().upper().replace('U', 'T')
    return value if value and re.fullmatch('[ACGTN]+', value) else ''


def generated_sequence_fields(row: Dict[str, Any]) -> Dict[str, str]:
    annotations = row.get('source_annotations') or {}
    sequence = _plain_bases(row.get('Sequence') or row.get('sequence') or row.get('seq'))
    synthesis = _plain_bases(row.get('synthesized_antisense_sequence') or row.get('Synthesis') or row.get('chemistry_synthesis') or annotations.get('chemistry_synthesis') or annotations.get('Synthesis'))
    result = {'sequence_role': 'unverified_source_sequence', 'sequence_direction': "5prime_to_3prime"}
    if not sequence or not synthesis:
        return result
    complement = synthesis.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]
    if sequence == complement:
        result.update(sequence_role='transcript_oriented_target_binding_site', target_binding_sequence=sequence, synthesized_antisense_sequence=synthesis)
    elif sequence == synthesis:
        result.update(sequence_role='synthesized_antisense_oligo', target_binding_sequence=complement, synthesized_antisense_sequence=synthesis)
    else:
        result['sequence_role'] = 'inconsistent_with_synthesis'
    return result


def sequence_notes(row: Dict[str, Any]):
    fields = generated_sequence_fields(row)
    raw = row.get('Sequence') or row.get('sequence') or ''
    notes = [f"Source sequence ({fields['sequence_role']}, 5' to 3'): {raw}"]
    if fields.get('synthesized_antisense_sequence'):
        notes.append(f"Synthesized antisense ASO (5' to 3'): {fields['synthesized_antisense_sequence']}")
        notes.append(f"Transcript-oriented target binding site (5' to 3'): {fields['target_binding_sequence']}")
    return notes


def exact_count_unit(row: Dict[str, Any]) -> str:
    explicit = row.get('off_target_exact_count_unit')
    if explicit:
        return str(explicit)
    if 'Off_target_0mm_loci' in row or 'off_target_0mm_loci' in row:
        return 'loci'
    return 'source_defined_hits; do_not_interpret_as_distinct_genes'


def exact_gene_count(row: Dict[str, Any]):
    """Read or derive a distinct-gene count, keeping unknown searches unknown."""
    if str(row.get('off_target_count_status') or '').lower() in {'incomplete','unavailable','not_run','failed'}:
        return None
    for key in ('off_target_gene_count','Times_mapped_off_target_genes','times_mapped_off_target_genes','worst_species_off_target_gene_count'):
        value = row.get(key)
        if value is not None and value != '' and not isinstance(value, bool):
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
    genes = row.get('Off_target_0mm_genes')
    if genes is None and isinstance(row.get('Off_target_genes_by_mismatch'),dict):
        genes = row['Off_target_genes_by_mismatch'].get('0mm')
    # An explicit mismatch-bin list is a measurement; no list is not zero.
    if isinstance(genes, list):
        return len({str(gene) for gene in genes if gene not in (None,'')})
    if row.get('off_target_exact_count_unit') == 'genes':
        value = row.get('Off_target_exact_count', row.get('off_target_exact_count'))
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            pass
    return None


ISOFORM_SCORE_DEFINITION = ('isoform_uniqueness_score depends on its class: gene_wide is the fraction of the loaded catalog containing the site; '
    'without a beneficial set, a target-specific class is target matches / all matching isoforms; with a beneficial set it is target matches / (target matches + other matches + 4 * beneficial matches), using the saved role classifications. '
    'It is not pan-isoform coverage unless class=gene_wide. isoform_coverage_fraction separately reports matching isoforms / catalog size. '
    'Same-gene sharing supports broad knockdown but limits isoform-selective knockdown; neither metric measures experimental efficacy.')
