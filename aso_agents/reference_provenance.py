"""Persist the actual reference catalog alongside its interpretation limits."""
from pathlib import Path
import hashlib
import json


def design_reference_provenance(goal, run_dir=None):
    from .species_config import get_species_config, normalize_species
    config = get_species_config(normalize_species(goal.get('species') or 'human'))
    catalog = goal.get('all_transcripts_full') or goal.get('isoform_transcripts') or {}
    serialized = json.dumps(catalog, sort_keys=True, ensure_ascii=False, separators=(',',':'))
    snapshot = None
    region_snapshot = None
    region_hash = None
    if run_dir:
        path = Path(run_dir) / 'reference_transcript_catalog.json'
        path.write_text(serialized)
        snapshot = str(path)
        region_evidence = {'requested_constraints':goal.get('region_constraints'),
                           'materialized_constraints':goal.get('materialized_region_constraints'),
                           'mirna_binding_result':goal.get('mirna_binding_result'),
                           'rbp_binding_result':goal.get('rbp_binding_result')}
        if any(region_evidence.values()):
            region_text = json.dumps(region_evidence, sort_keys=True, ensure_ascii=False, separators=(',',':'))
            region_path = Path(run_dir) / 'region_constraint_evidence.json'
            region_path.write_text(region_text)
            region_snapshot = str(region_path)
            region_hash = hashlib.sha256(region_text.encode()).hexdigest()
    identities = []
    for label, entry in catalog.items():
        meta = entry.get('metadata') or {}
        tid = str(entry.get('transcript_id') or meta.get('id') or label)
        versioned = meta.get('versioned_id') or (f"{tid.split('.')[0]}.{meta['version']}" if meta.get('version') is not None else tid)
        identities.append({'label':label,'transcript_id':versioned,
            'source':meta.get('reference_source') or 'resolved transcript catalog; inspect saved snapshot',
            'annotation':meta.get('reference_annotation'), 'version_attested':'.' in versioned})
    from .exon_skip_consequences import requested_skip_reports
    try:
        skipping = requested_skip_reports(goal, catalog, config.genome_fasta)
    except (OSError, KeyError, ValueError) as exc:
        skipping = [{'status':'reference_consequence_unavailable', 'reason':str(exc), 'nmd_predicted':None}]
    return {'assembly':config.ensembl_assembly, 'genome_fasta':str(Path(config.genome_fasta).resolve()),
        'gene_annotation':str(Path(config.gtf_path).resolve()), 'bowtie_index':str(config.bowtie_prefix),
        'transcript_catalog_snapshot':snapshot, 'transcript_catalog_sha256':hashlib.sha256(serialized.encode()).hexdigest(),
        'transcript_count':len(catalog), 'transcript_identities':identities,
        'exon_skip_consequences':skipping,
        'region_evidence_snapshot':region_snapshot, 'region_evidence_sha256':region_hash,
        'coordinate_conversion_evidence':goal.get('materialized_region_constraints') or {},
        'hard_mapping_filter_definition':'The native max_binding_sites hard filter first rejects explicitly incomplete or unavailable counts. Otherwise it uses the row\'s recorded exact-match count: an explicit off-target mapping lower bound if present, otherwise annotated other-gene count, otherwise the steric mapper\'s exact genomic placement count (including intended placements). It never substitutes gapmer target-isoform coverage. Missing count evidence cannot pass. Reported one- or higher-mismatch bins are separate secondary screens and are not summed into this exact-match hard threshold. State the actual field and unit; this default does not override a user-specified mismatch-inclusive screen.',
        'reference_scope':'The saved transcript catalog is the actual mature-RNA isoform universe for these scores. It may contain REST-resolved structures from a different annotation release than the configured GTF. Do not call these one annotation universe or infer unsearched isoforms are absent.',
        'mapping_scope':'Bowtie screening uses the configured genomic index and ungapped mismatch bins. Inspect each recorded mismatch budget and completeness flag; unsearched or missing bins are unknown. Genomic matching searches both strands; this is a sequence-screening proxy, not evidence that every hit binds transcribed RNA. Gene counts depend on the configured gene annotation and exclude the intended target gene; loci, transcripts and genes are different units. Gapmer exact-gene screening additionally includes adjacent mature-RNA exon-junction windows. It does not test arbitrary multi-junction or gapped alignments.',
        'efficacy_scope':'Sequence complementarity, catalog coverage, genomic screening and perturbation scores do not measure cellular occupancy, knockdown, specificity or safety.'}


def reference_markdown(provenance):
    if not provenance:
        return ''
    text = '\n'.join(['## References and search scope','',
        f"Assembly: `{provenance['assembly']}`. Genome: `{provenance['genome_fasta']}`. Gene annotation: `{provenance['gene_annotation']}`.",
        f"Transcript catalog: {provenance['transcript_count']} entries; SHA-256 `{provenance['transcript_catalog_sha256']}`. Full structures: `{provenance.get('transcript_catalog_snapshot')}`.",
        provenance['reference_scope'], provenance['mapping_scope'], provenance['hard_mapping_filter_definition'], provenance['efficacy_scope']])
    conversions = provenance.get('coordinate_conversion_evidence') or {}
    entries = [*(conversions.get('global') or []), *[item for group in (conversions.get('by_program') or {}).values() for item in group]]
    records = [entry.get('materialization_summary') for entry in entries if (entry.get('materialization_summary') or {}).get('coordinate_conversions')]
    if records:
        text += '\n\n## Original intervals and assembly conversion evidence\n\n' + json.dumps(records, ensure_ascii=False, indent=2)
    if provenance.get('region_evidence_snapshot'):
        text += '\n\nFull regulatory-source records, requested regions and projection evidence: ' + f"`{provenance['region_evidence_snapshot']}` (SHA-256 `{provenance['region_evidence_sha256']}`)."
    if provenance.get('exon_skip_consequences'):
        text += '\n\n## Conditional exon-skipping sequence consequences\n\n' + json.dumps(provenance['exon_skip_consequences'], ensure_ascii=False, indent=2)
    return text


def evaluation_catalog_provenance(catalog, sequences, rows, output_directory):
    """Save actual structures/sequences and identify every reported mapping hit."""
    entries, identities = {}, {}
    for label, entry in catalog.items():
        metadata = entry.get('metadata') or {}
        tid = str(entry.get('transcript_id') or entry.get('id') or label)
        version = metadata.get('version', entry.get('version'))
        versioned = metadata.get('versioned_id') or entry.get('versioned_id')
        if not versioned and version is not None:
            versioned = f"{tid.split('.')[0]}.{version}"
        identity = {'transcript_id':versioned or tid, 'version_attested':'.' in str(versioned or tid),
            'reference_source':metadata.get('reference_source') or entry.get('reference_source'),
            'reference_annotation':metadata.get('reference_annotation') or entry.get('reference_annotation'),
            **(entry.get('reference_binding') or {}),
            'sequence_evidence':entry.get('sequence_evidence'),
            'sequence_resolution_error':entry.get('sequence_resolution_error'),
            'genomic_reference_resolution_error':entry.get('genomic_reference_resolution_error')}
        sequence = sequences.get(label)
        entries[label] = {'reference_identity':identity, 'structure':entry,
            'mature_cdna_sequence':sequence, 'mature_cdna_length':len(sequence) if sequence is not None else None,
            'mature_cdna_sha256':hashlib.sha256(sequence.encode()).hexdigest() if sequence is not None else None}
        identities[label] = {**identity, 'mature_cdna_length':entries[label]['mature_cdna_length'],
            'mature_cdna_sha256':entries[label]['mature_cdna_sha256']}
    for row in rows:
        for match in [*(row.get('mapping_hits') or []), *(row.get('near_match_candidates') or [])]:
            label = match.get('isoform_label')
            if label not in identities:
                matching = [key for key,value in catalog.items()
                    if str(value.get('transcript_id') or key) == str(match.get('transcript_id'))]
                label = matching[0] if len(matching)==1 else None
            if label in identities:
                match['reference_identity'] = identities[label].copy()
    path = Path(output_directory)/'evaluation_reference_catalog.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps({'entries':entries}, sort_keys=True, ensure_ascii=False, separators=(',',':'))
    path.write_text(serialized)
    return {'snapshot_path':str(path), 'snapshot_sha256':hashlib.sha256(serialized.encode()).hexdigest(),
        'transcript_count':len(entries), 'transcript_identities':identities,
        'scope':'These are the actual structures and mature-RNA sequences searched in this execution. Each transcript has its own species, assembly, genome binding and sequence-extraction evidence; the primary request organism does not apply to every catalog entry. Missing reference sequence is unknown, not evidence of an absent binding site. Version identity comes from the saved metadata, not from a versionless alias or the requested primary transcript. The configured genomic annotation may be a different release. Coverage is relative to this explicit catalog; it is not efficacy or a complete isoform census.'}
