from copy import deepcopy
from types import SimpleNamespace

import pytest
from pyfaidx import Fasta

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline
from aso_agents import transcript_reference
from aso_agents.evaluation_identity import (
    bind_primary_evaluation_identity, declared_evaluation_context, screening_catalog_labels,
)


def entry(species, gene, transcript='TX_BETA'):
    return {'transcript_id': transcript, 'chrom': 'chrA', 'strand': '+',
            'exons_tx': [{'index': 1, 'start': 1, 'end': 8}],
            'metadata': {'species': species, 'gene_id': gene,
                         'versioned_id': transcript + '.2'}}


@pytest.fixture
def references(tmp_path, monkeypatch):
    configs = {}
    for species in ('human', 'mouse'):
        fasta = tmp_path / (species + '.fa')
        fasta.write_text('>chrA\nAACCGGTTAACCGGTT\n')
        configs[species] = SimpleNamespace(genome_fasta=fasta, ensembl_assembly=species + '-fixture')
    monkeypatch.setattr(transcript_reference, 'get_species_config', configs.__getitem__)
    monkeypatch.setattr(runtime, 'get_species_config', configs.__getitem__)
    monkeypatch.setattr(pipeline, 'get_species_config', configs.__getitem__)
    monkeypatch.setattr(runtime, '_load_genome_fasta', lambda species: Fasta(str(configs[species].genome_fasta)))
    return {'TX_ALPHA': entry('human', 'GENE_ALPHA', 'TX_ALPHA'),
            'TX_BETA': entry('mouse', 'GENE_BETA')}


def test_declared_context_distinguishes_runtime_defaults_and_preserves_global_facts():
    envelope = {'species': 'human', 'gene_id': 'GENE_ALPHA', 'transcript_id': 'TX_BETA.2'}
    state = {'canonical_request_envelope': {'species': 'human', 'tasks': [
        {'task_id': 'evaluate', 'context': {'transcript_id': 'TX_BETA.2'}}]}}
    context, basis = declared_evaluation_context(state, envelope, 'evaluate')
    assert basis == 'canonical_task' and 'species' not in context and 'gene_id' not in context
    state['canonical_request_envelope']['tasks'][0]['context'] = {}
    assert declared_evaluation_context(state, envelope, 'evaluate')[0]['species'] == 'human'
    assert declared_evaluation_context({}, envelope, 'evaluate')[0] == envelope


@pytest.mark.parametrize('declared,error', [
    ({'species': 'human'}, 'evaluation_species_conflicts'),
    ({'gene_id': 'GENE_ALPHA'}, 'evaluation_gene_conflicts'),
])
def test_explicit_conflicts_fail_before_screening(references, declared, error):
    goal = {'species': 'human', 'gene_id': 'GENE_ALPHA'}
    before = deepcopy(goal)
    with pytest.raises(ValueError, match=error):
        bind_primary_evaluation_identity(goal, references['TX_BETA'], declared)
    assert goal == before


def test_rebinding_drops_stale_gene_cache_and_patient_reference(references):
    goal = {'species': 'human', 'gene_id': 'GENE_ALPHA', 'gene_symbol': 'EarlierGene',
            'gene_id_resolution': {'source': 'earlier'}, 'knockdown_isoform_catalog': {'status': 'complete'},
            'patient_reference': {'fasta_path': '/earlier/patient.fa'}, 'chemistry': '2MOE'}
    receipt = bind_primary_evaluation_identity(goal, references['TX_BETA'], {'species': 'mus_musculus'})
    assert goal == {'species': 'mouse', 'gene_id': 'GENE_BETA', 'chemistry': '2MOE'}
    assert receipt['inherited_identity_changed'] and receipt['gene_basis'] == 'reference_metadata'
    assert receipt['previous_context']['gene_id'] == 'GENE_ALPHA'


def test_same_species_new_gene_excludes_old_and_unattested_catalogs(references):
    primary = entry('human', 'GENE_BETA')
    goal = {'species': 'human', 'gene_id': 'GENE_ALPHA'}
    receipt = bind_primary_evaluation_identity(goal, primary, {})
    catalog = {**references, 'TX_BETA': primary,
               'TX_UNKNOWN': {'transcript_id': 'TX_UNKNOWN'},
               'TX_SIBLING': entry('human', 'GENE_BETA', 'TX_SIBLING')}
    assert screening_catalog_labels(catalog, 'TX_BETA', goal, receipt) == {'TX_BETA', 'TX_SIBLING'}


def test_missing_identity_is_not_invented_from_accession(references):
    primary = {'transcript_id': 'ENSMUST_fixture', 'metadata': {}}
    goal = {'species': 'human', 'gene_id': 'GENE_ALPHA'}
    receipt = bind_primary_evaluation_identity(goal, primary, {})
    assert goal['species'] == 'human'
    assert receipt['reference']['species_basis'] == 'single_organism_request_context'
    assert receipt['gene_id'] is None and receipt['gene_basis'] == 'evaluation_context'


def test_reference_gene_identity_survives_explicit_version_resolution():
    structure = {'id': 'TX_BETA', 'versioned_id': 'TX_BETA.2', 'version': 2,
                 'species': 'mus_musculus', 'gene_id': 'GENE_BETA', 'seq_region_name': 'chrA',
                 'strand': '+', 'exons_tx': [{'start': 1, 'end': 8}]}
    native = SimpleNamespace(get_transcript_structure=lambda _: deepcopy(structure))
    goal = {'transcript_lock': {'value': 'TX_BETA.2'}}
    isoforms, _, notes = pipeline._ensure_eval_transcript_catalog(goal, native)
    assert not notes and isoforms['TX_BETA']['metadata']['gene_id'] == 'GENE_BETA'


@pytest.mark.parametrize('conflict', [False, True])
def test_evaluator_uses_reference_for_screening_and_preserves_foreign_mapping(
        references, tmp_path, monkeypatch, conflict):
    catalog = deepcopy(references)
    earlier = {'species': 'human', 'gene_id': 'GENE_ALPHA', 'gene_symbol': 'EarlierGene'}
    monkeypatch.setattr(pipeline, '_goal_runtime_module', lambda: runtime)
    monkeypatch.setattr(pipeline, '_candidate_run_dirs_for_task', lambda *_: [])
    monkeypatch.setattr(pipeline, '_reference_goal_context_for_request', lambda *_: {'structured_goal': earlier})
    monkeypatch.setattr(pipeline, '_ensure_eval_transcript_catalog', lambda *_: (catalog, catalog, []))
    monkeypatch.setattr(pipeline, '_choose_eval_transcript',
                        lambda *_: ('TX_BETA', 'TX_BETA', deepcopy(catalog['TX_BETA'])))
    monkeypatch.setattr(pipeline, '_mapping_conservation_summary', lambda **_: {})
    refreshed = []
    def refresh(goal):
        refreshed.append(deepcopy(goal))
        return {'TX_SIBLING': entry('mouse', 'GENE_BETA', 'TX_SIBLING')}
    monkeypatch.setattr(runtime, '_load_all_transcripts_for_gene', refresh)
    scores = []
    monkeypatch.setattr(pipeline, '_score_provided_asos_with_existing_pipeline',
                        lambda **kwargs: scores.append({key: deepcopy(kwargs[key])
                                                       for key in ('eval_goal', 'cdna_map')}) or {})
    context = {'transcript_id': 'TX_BETA.2', 'gene_symbol': 'CurrentGene'}
    if conflict:
        context['species'] = 'human'
    state = {'request': 'Evaluate the supplied molecule.', 'canonical_request_envelope': {
        'tasks': [{'task_id': 'evaluation', 'context': context}]}}
    envelope = {'species': 'human', 'gene_id': 'GENE_ALPHA', 'gene_symbol': 'CurrentGene',
                'transcript_id': 'TX_BETA.2', 'provided_asos': [
                    {'name': 'probe', 'sequence': 'AACCGGTT', 'sequence_role': 'aso'}]}
    result = pipeline._evaluate_existing_asos_payload(state=state, request_envelope=envelope,
        run_dir=str(tmp_path / 'output'), task_id='evaluation')
    if conflict:
        assert result['error'] == 'evaluation_species_conflicts_with_primary_transcript'
        assert not scores and not refreshed
        return
    assert len(scores) == 1
    assert scores[0]['eval_goal']['species'] == 'mouse'
    assert scores[0]['eval_goal']['gene_id'] == 'GENE_BETA'
    assert scores[0]['eval_goal']['gene_symbol'] == 'CurrentGene'
    assert set(scores[0]['cdna_map']) == {'TX_BETA', 'TX_SIBLING'}
    assert set(result['mapping_strategy']['reference_searches']) == {'TX_ALPHA', 'TX_BETA', 'TX_SIBLING'}
    assert result['context_summary']['species'] == 'mouse'
    assert result['context_summary']['gene_id'] == 'GENE_BETA'
    assert result['provided_asos'][0]['species'] == 'mouse'
    assert refreshed[0]['gene_id'] == 'GENE_BETA' and 'isoform_transcripts' not in refreshed[0]
    assert earlier == {'species': 'human', 'gene_id': 'GENE_ALPHA', 'gene_symbol': 'EarlierGene'}
    assert envelope['species'] == 'human'  # Caller inputs remain unchanged.
