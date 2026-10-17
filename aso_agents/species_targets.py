"""Keep an organism's explicit references bound through multi-species execution."""
from copy import deepcopy

from .species_config import normalize_requested_species


def anchor_conflict(context, goal):
    """Conflicting scalar identities must be reconciled, never overwritten."""
    return bool(context.get('species') and goal.get('species')
                and normalize_requested_species(context['species']) != normalize_requested_species(goal['species']))


def scope_contract_errors(goal):
    """Pure identity checks, before finalization can add inferred target entries."""
    targets = [item for item in goal.get('species_targets') or [] if isinstance(item, dict)]
    declared = {normalize_requested_species(item.get('species')) for item in targets if item.get('species')}
    anchor = normalize_requested_species(goal['species']) if goal.get('species') else None
    errors = {}
    if len(declared) > 1 and anchor and anchor not in declared:
        errors['species'] = 'The single anchor organism must be a member of species_targets.'
    lock = goal.get('transcript_lock') or {}
    if isinstance(lock, dict) and lock.get('value') and lock.get('kind') in (None, 'explicit_id'):
        anchored_ids = [str(item['transcript_id']) for item in targets
                        if item.get('transcript_id') and normalize_requested_species(item.get('species')) == anchor]
        if anchored_ids and str(lock['value']) not in {value for tid in anchored_ids for value in (tid, tid.split('.')[0])}:
            errors['transcript_lock.value'] = 'A scalar transcript lock must identify one transcript of the anchor organism; other organisms retain their own species_targets references.'
    return errors


def explicit_targets(goal, species):
    key = normalize_requested_species(species)
    return [deepcopy(item) for item in goal.get('species_targets') or []
            if isinstance(item, dict) and normalize_requested_species(item.get('species')) == key
            and (item.get('transcript_id') or item.get('gene_id'))]


def reference_identity_error(structure, transcript_id, species, gene_id=None):
    """A lookup hit must establish identity, organism, version and parent gene."""
    if structure.get('error'):
        return str(structure['error'])
    requested = str(transcript_id)
    actual = str(structure.get('versioned_id') or structure.get('id') or '')
    if actual.split('.')[0] != requested.split('.')[0]:
        return 'transcript_identity_mismatch'
    if '.' in requested and actual != requested:
        return 'transcript_version_unresolved_or_mismatched'
    if not structure.get('species') or normalize_requested_species(structure['species']) != normalize_requested_species(species):
        return 'transcript_species_unresolved_or_mismatched'
    parent = structure.get('gene_id') or structure.get('Parent')
    if not parent:
        return 'transcript_parent_gene_unresolved'
    # NCBI numeric GeneIDs and Ensembl gene IDs are different namespaces.
    # Comparing their strings would reject a valid transcript-parent pair.
    # The foreign alias is checked against the resolved gene symbol below.
    if (gene_id and (not str(gene_id).isdigit() or str(parent).isdigit())
            and str(parent).split('.')[0] != str(gene_id).split('.')[0]):
        return 'transcript_parent_gene_mismatch'
    return None


def resolve_explicit_target(goal, species, fetch_structure, lookup_gene):
    """Use explicit references first; errors must never trigger a homology fallback."""
    requests = explicit_targets(goal, species)
    if not requests:
        return None
    genes, transcripts, evidence = set(), [], []
    numeric_alias_symbols = []
    error = None
    for item in requests:
        gene_id = item.get('gene_id')
        tid = item.get('transcript_id')
        if tid:
            if gene_id and str(gene_id).isdigit():
                numeric_alias_symbols.append(str(item.get('gene_symbol') or '').strip())
            try:
                structure = fetch_structure(tid)
                error = reference_identity_error(structure, tid, species, gene_id)
            except Exception as exc:
                error = f'transcript_reference_unavailable:{exc}'
            if error:
                break
            gene_id = structure.get('gene_id') or structure.get('Parent')
            if tid not in transcripts:
                transcripts.append(tid)
            evidence.append({key: structure.get(key) for key in ('id', 'versioned_id', 'species', 'gene_id', 'assembly_name', 'reference_source', 'reference_annotation')})
        genes.add(str(gene_id).split('.')[0])
    if not error and len(genes) != 1:
        error = 'explicit_species_targets_require_one_gene_per_design_task'
    metadata = {}
    if not error:
        try:
            metadata = lookup_gene(next(iter(genes)))
            if metadata.get('error'):
                error = str(metadata['error'])
            elif not metadata.get('species') or normalize_requested_species(metadata['species']) != normalize_requested_species(species):
                error = 'gene_species_unresolved_or_mismatched'
            elif numeric_alias_symbols and (
                not all(numeric_alias_symbols)
                or any(symbol.casefold() != str(metadata.get('display_name') or '').strip().casefold()
                       for symbol in numeric_alias_symbols)
            ):
                error = 'numeric_gene_alias_symbol_unresolved_or_mismatched'
        except Exception as exc:
            error = f'gene_reference_unavailable:{exc}'
    result = {'status': 'error' if error else 'resolved', 'species': species,
              'resolution_source': 'explicit_species_reference', 'requested_references': requests,
              'reference_evidence': evidence, 'transcript_ids': transcripts}
    if error:
        result['error'] = error
    else:
        result.update(gene_id=next(iter(genes)), gene_symbol=metadata.get('display_name') or next(iter(genes)))
        if numeric_alias_symbols:
            result['numeric_gene_alias_status'] = 'foreign_namespace_not_crosswalk_verified; transcript_parent_and_symbol_verified'
    return result
