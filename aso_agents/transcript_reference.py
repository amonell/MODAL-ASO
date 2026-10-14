"""Bind reference coordinates to their organism and assembly before extraction."""
from pathlib import Path

from .species_config import get_species_config, normalize_species


def transcript_reference(entry, default_species, reference_fasta=None, *, config_getter=None):
    """Resolve a catalog entry without inferring organism from its identifier.

    Old single-organism catalogs may inherit the request organism. Resolved
    reference metadata always takes precedence. A patient FASTA belongs only
    to the request organism, never to every organism in a mixed catalog.
    """
    config_getter = config_getter or get_species_config
    metadata = entry.get('metadata') or {}
    organisms = {normalize_species(value) for value in
                 (entry.get('species'), metadata.get('species')) if value}
    if len(organisms) > 1:
        raise ValueError('conflicting_transcript_reference_species')
    organism = next(iter(organisms)) if organisms else normalize_species(default_species)
    config = config_getter(organism)
    assemblies = {str(value) for value in
                  (entry.get('assembly_name'), metadata.get('assembly_name')) if value}
    if len(assemblies) > 1:
        raise ValueError('conflicting_transcript_reference_assemblies')
    configured_assembly = getattr(config, 'ensembl_assembly', None)
    assembly = next(iter(assemblies)) if assemblies else configured_assembly
    if assembly and configured_assembly and assembly != configured_assembly:
        raise ValueError('transcript_assembly_differs_from_configured_genome')
    custom = bool(reference_fasta and organism == normalize_species(default_species))
    path = reference_fasta if custom else config.genome_fasta
    return {
        'species': organism, 'assembly': assembly,
        'genome_fasta': str(Path(path).expanduser()),
        'reference_annotation': entry.get('reference_annotation') or metadata.get('reference_annotation'),
        'species_basis': 'reference_metadata' if organisms else 'single_organism_request_context',
        'assembly_basis': 'reference_metadata' if assemblies else 'configured_reference',
        'patient_reference_applied': custom,
    }


def catalog_genomes(catalog, default_species, runtime, reference_fasta=None):
    """Return a complete label map; unavailable foreign genomes remain None."""
    from pyfaidx import Fasta
    by_path, genomes = {}, {}
    for label, entry in catalog.items():
        genomes[label] = None
        try:
            binding = transcript_reference(entry, default_species, reference_fasta)
            entry['reference_binding'] = binding
            path = binding['genome_fasta']
            if path not in by_path:
                try:
                    loader = getattr(runtime, '_load_genome_fasta', None)
                    by_path[path] = (Fasta(path) if binding['patient_reference_applied'] else
                                     loader(binding['species']) if callable(loader) else None)
                except (OSError, ValueError):
                    by_path[path] = None
            genomes[label] = by_path[path]
            if genomes[label] is None:
                entry['genomic_reference_resolution_error'] = 'bound_reference_genome_unavailable'
        except (OSError, ValueError) as exc:
            entry['genomic_reference_resolution_error'] = str(exc)
    return genomes
