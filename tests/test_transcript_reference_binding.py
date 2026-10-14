from copy import deepcopy
from types import SimpleNamespace

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline
from aso_agents import transcript_reference as binding
from aso_agents.reference_provenance import evaluation_catalog_provenance


@pytest.fixture
def references(tmp_path, monkeypatch):
    configs = {}
    for organism, sequence in [('human', 'AACCGGTTAACCGGTT'), ('mouse', 'AATCGGTTAATCGGTT')]:
        fasta = tmp_path / (organism + '.fa')
        fasta.write_text('>chrA\n' + sequence + '\n')
        configs[organism] = SimpleNamespace(genome_fasta=fasta, ensembl_assembly=organism + '-assembly')
    monkeypatch.setattr(runtime, 'get_species_config', configs.__getitem__)
    monkeypatch.setattr(binding, 'get_species_config', configs.__getitem__)
    monkeypatch.setattr(pipeline, 'get_species_config', configs.__getitem__)
    catalog = {label: {'transcript_id': label, 'chrom': 'chrA', 'strand': '+',
               'metadata': {'species': organism, 'assembly_name': organism + '-assembly', 'versioned_id': label + '.2'},
               'exons_tx': [{'index': 1, 'start': 1, 'end': 4}, {'index': 2, 'start': 9, 'end': 12}]}
               for label, organism in [('TX_ALPHA', 'human'), ('TX_BETA', 'mouse')]}
    return configs, catalog


def test_same_coordinates_use_each_transcripts_own_genome(references, tmp_path):
    configs, catalog = references
    sequences = runtime._build_isoform_cdna_sequences(catalog, 'human')
    assert sequences == {'TX_ALPHA': 'AACCAACC', 'TX_BETA': 'AATCAATC'}
    for label, entry in catalog.items():
        species = entry['metadata']['species']
        assert entry['sequence_evidence']['genome_fasta'] == str(configs[species].genome_fasta)
        assert entry['sequence_evidence']['species'] == species
    result = evaluation_catalog_provenance(catalog, sequences, [], tmp_path)
    assert result['transcript_identities']['TX_BETA']['assembly'] == 'mouse-assembly'


def test_patient_reference_only_overrides_its_own_organism(references, tmp_path):
    _, catalog = references
    patient = tmp_path / 'patient.fa'
    patient.write_text('>chrA\nTTTTGGTTTTTTGGTT\n')
    sequences = runtime._build_isoform_cdna_sequences(catalog, 'human', str(patient))
    assert sequences == {'TX_ALPHA': 'TTTTTTTT', 'TX_BETA': 'AATCAATC'}
    genomes = binding.catalog_genomes(catalog, 'human', runtime, str(patient))
    assert str(genomes['TX_ALPHA']['chrA'][:4]) == 'TTTT'
    assert str(genomes['TX_BETA']['chrA'][:4]) == 'AATC'


def test_missing_patient_reference_cannot_fall_back_to_wildtype(references, tmp_path):
    _, catalog = references
    result = runtime._build_isoform_cdna_sequences(catalog, 'human', str(tmp_path / 'absent.fa'))
    assert result == {'TX_BETA': 'AATCAATC'}
    assert catalog['TX_ALPHA']['sequence_resolution_error'] == 'provided_reference_genome_unavailable'


def test_unavailable_foreign_reference_is_unknown_never_primary(references, monkeypatch):
    configs, catalog = references
    configs['mouse'].genome_fasta.unlink()
    from aso_agents import transcript_sequence
    def unavailable(*_):
        raise ValueError('reference service unavailable')
    monkeypatch.setattr(transcript_sequence, 'reference_cdna', unavailable)
    result = runtime._build_isoform_cdna_sequences(catalog, 'human')
    assert result == {'TX_ALPHA': 'AACCAACC'}
    genomes = binding.catalog_genomes(catalog, 'human', runtime)
    assert genomes['TX_ALPHA'] is not None and genomes['TX_BETA'] is None
    assert catalog['TX_BETA']['sequence_resolution_error']


def test_conflicting_assembly_fails_before_extraction(references):
    _, catalog = references
    catalog['TX_BETA']['metadata']['assembly_name'] = 'older-assembly'
    assert runtime._build_isoform_cdna_sequences(catalog, 'human') == {'TX_ALPHA': 'AACCAACC'}
    assert catalog['TX_BETA']['sequence_resolution_error'] == 'transcript_assembly_differs_from_configured_genome'
    assert binding.catalog_genomes(catalog, 'human', runtime)['TX_BETA'] is None


def test_negative_strand_and_junction_geometry_survive_per_reference_extraction(references):
    _, catalog = references
    entry = catalog['TX_BETA']
    entry['strand'] = '-'
    entry['exons_tx'] = [{'index': 1, 'start': 9, 'end': 12}, {'index': 2, 'start': 1, 'end': 4}]
    assert runtime._build_isoform_cdna_sequences(catalog, 'human')['TX_BETA'] == 'GATTGATT'
    assert runtime._map_cdna_range_to_genomic_segments(3, 6, entry) == [('chrA', 9, 10), ('chrA', 3, 4)]


def test_exact_and_near_locus_searches_do_not_reuse_primary_genome(references, monkeypatch):
    _, catalog = references
    sequences = runtime._build_isoform_cdna_sequences(catalog, 'human')
    genomes = binding.catalog_genomes(catalog, 'human', runtime)
    monkeypatch.setattr(pipeline, '_mapping_conservation_summary', lambda **_: {})
    molecule = {'sequence': 'GGTT', 'reverse_complement_sequence': 'AACC'}
    hits = pipeline._build_sequence_mapping_hits(entry=molecule, isoform_transcripts=catalog,
        cdna_map=sequences, runtime=runtime, species='human', preferred_transcript_id='TX_ALPHA',
        genome=genomes['TX_ALPHA'], genomes_by_label=genomes)
    antisense_hits = [hit for hit in hits if hit['match_orientation'] == 'reverse_complement']
    assert antisense_hits and {hit['transcript_id'] for hit in antisense_hits} == {'TX_ALPHA'}
    near = pipeline._best_locus_near_matches(entry=molecule, isoform_transcripts=catalog,
        runtime=runtime, genome=genomes['TX_ALPHA'], genomes_by_label=genomes, preferred_transcript_id='TX_BETA')
    foreign = [hit for hit in near if hit['transcript_id'] == 'TX_BETA' and hit['match_orientation'] == 'reverse_complement']
    assert foreign and foreign[0]['mismatch_count'] == 1
    # Missing per-label genomes cannot silently use the non-None default.
    hits = pipeline._build_sequence_mapping_hits(entry=molecule, isoform_transcripts={'TX_BETA':catalog['TX_BETA']},
        cdna_map={}, runtime=runtime, species='human', preferred_transcript_id=None,
        genome=genomes['TX_ALPHA'], genomes_by_label={'TX_BETA':None})
    assert hits == []


def test_authoritative_organism_survives_structure_and_metadata_resolution():
    from aso_agents.transcript_chooser_agent import _transcript_structure_from_payload
    from aso_agents.transcript_metadata import resolved_transcript_metadata
    payload = {'id':'TX_BETA', 'version':2, 'species':'mus_musculus', 'assembly_name':'GRCm39',
               'seq_region_name':'A', 'strand':1, 'start':1, 'end':4, 'Exon':[{'start':1,'end':4}]}
    structure = _transcript_structure_from_payload(payload, 'TX_BETA.2')
    assert structure['species'] == 'mus_musculus' and structure['assembly_name'] == 'GRCm39'
    metadata = resolved_transcript_metadata(structure, {'species':'human', 'assembly_name':'GRCh38'})
    assert metadata['species'] == 'mus_musculus' and metadata['assembly_name'] == 'GRCm39'


def test_evaluation_payload_keeps_per_reference_search_and_primary_context(references, tmp_path, monkeypatch):
    _, catalog = references
    monkeypatch.setattr(pipeline, '_goal_runtime_module', lambda: runtime)
    monkeypatch.setattr(pipeline, '_candidate_run_dirs_for_task', lambda *_: [])
    monkeypatch.setattr(pipeline, '_reference_goal_context_for_request', lambda *_: {})
    monkeypatch.setattr(pipeline, '_ensure_eval_transcript_catalog', lambda *_: (catalog, catalog, []))
    monkeypatch.setattr(pipeline, '_choose_eval_transcript', lambda *_: ('TX_BETA', 'TX_BETA', deepcopy(catalog['TX_BETA'])))
    monkeypatch.setattr(pipeline, '_score_provided_asos_with_existing_pipeline', lambda **_: {})
    monkeypatch.setattr(pipeline, '_mapping_conservation_summary', lambda **_: {})
    result = pipeline._evaluate_existing_asos_payload(state={'request':'Evaluate the supplied antisense molecule.'},
        request_envelope={'provided_asos':[{'name':'probe','sequence':'GGTTGGTT','sequence_role':'aso'}]},
        run_dir=str(tmp_path / 'output'),task_id='neutral')
    assert result['context_summary']['reference_build'] == 'mouse-assembly'
    identities = result['reference_catalog']['transcript_identities']
    assert identities['TX_ALPHA']['species'] == 'human' and identities['TX_BETA']['species'] == 'mouse'
    assert all(search['mature_cdna_searched'] and search['premrna_searched']
               for search in result['mapping_strategy']['reference_searches'].values())
    exact = result['provided_asos'][0]['mapping_hits']
    assert {hit['transcript_id'] for hit in exact if hit['match_orientation']=='reverse_complement'} == {'TX_ALPHA'}


def test_plural_dependency_evaluation_loads_each_requested_catalog(references, tmp_path, monkeypatch):
    _, catalog = references
    contexts = {}
    for label, species, gene in [('TX_ALPHA', 'human', 'GENE_A'), ('TX_BETA', 'mouse', 'GeneB')]:
        directory = tmp_path / species
        directory.mkdir()
        goal = {'species': species, 'gene_symbol': gene,
                'isoform_transcripts': {label: catalog[label]}, 'all_transcripts_full': {label: catalog[label]}}
        (directory / 'goal_intake.json').write_text(__import__('json').dumps({'final_payload': {'structured_goal': goal}}))
        contexts[species] = str(directory)
    monkeypatch.setattr(pipeline, '_goal_runtime_module', lambda: runtime)
    monkeypatch.setattr(pipeline, '_candidate_run_dirs_for_task', lambda *_: list(contexts.values()))
    monkeypatch.setattr(pipeline, '_choose_eval_transcript', lambda *_: ('TX_ALPHA', 'TX_ALPHA', deepcopy(catalog['TX_ALPHA'])))
    scored_catalogs = []
    monkeypatch.setattr(pipeline, '_score_provided_asos_with_existing_pipeline',
                        lambda **kwargs: scored_catalogs.append(list(kwargs['cdna_map'])) or {})
    monkeypatch.setattr(pipeline, '_mapping_conservation_summary', lambda **_: {})
    result = pipeline._evaluate_existing_asos_payload(state={'request': 'Evaluate the supplied molecule against both reference sets.'},
        request_envelope={'provided_asos': [{'name': 'probe', 'sequence': 'GGTTGGTT', 'sequence_role': 'aso'}],
                          'task_context': {'species_targets': [{'species': 'human', 'gene_symbol': 'GENE_A'},
                                                               {'species': 'mouse', 'gene_symbol': 'GeneB'}]}},
        run_dir=str(tmp_path / 'output'), task_id='neutral')
    assert set(result['mapping_strategy']['reference_searches']) == {'TX_ALPHA', 'TX_BETA'}
    assert all(item['status'] == 'resolved' for item in result['reference_context_coverage'])
    assert scored_catalogs == [['TX_ALPHA']]  # Primary off-target/uniqueness scope is not silently widened.
    row = result['provided_asos'][0]
    assert {hit['transcript_id'] for hit in row['mapping_hits'] if hit['match_orientation'] == 'reverse_complement'} == {'TX_ALPHA'}
    assert any(hit['transcript_id'] == 'TX_BETA' and hit['match_orientation'] == 'reverse_complement'
               and hit['mismatch_count'] == 2 for hit in row['near_match_candidates'])
