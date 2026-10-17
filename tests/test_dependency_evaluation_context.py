from copy import deepcopy
import pytest

from aso_agents.dependency_references import (
    requested_dependency_contexts, merge_reference_catalog, recorded_trace_markdown,
)
from aso_agents.comparison_evidence import comparison_binding_evidence
from aso_agents.species_config import normalize_species
from aso_agents import pipeline_tools as pipeline


def context(species, gene, directory):
    return {'run_dir': directory, 'structured_goal': {'species': species, 'gene_symbol': gene}}


def test_context_union_is_bound_to_requested_typed_identity():
    contexts = {'a': context('human', 'GENE_A', 'a'), 'b': context('mouse', 'GeneA', 'b'),
                'c': context('mouse', 'OtherGene', 'c')}
    envelope = {'task_context': {'species_targets': [
        {'species': 'human', 'gene_symbol': 'GENE_A'},
        {'species': 'mouse', 'gene_symbol': 'GeneA'},
        {'species': 'rat', 'gene_symbol': 'Missing'}]}}
    found, coverage = requested_dependency_contexts(['b', 'c', 'a', 'b'], envelope,
        lambda dirs: contexts[dirs[0]], normalize_species)
    assert [item['run_dir'] for item in found] == ['b', 'a']
    assert [item['status'] for item in coverage] == ['resolved', 'resolved', 'unavailable']


def test_no_plural_target_keeps_legacy_single_context_scope():
    def unexpected(_):
        raise AssertionError('Should not inspect unrelated dependencies')
    assert requested_dependency_contexts(['a', 'b'], {}, unexpected, normalize_species) == ([], [])


def test_catalog_merge_keeps_reference_identity_and_never_mutates_inputs():
    entry = {'transcript_id': 'TX_A', 'chrom': 'chrA', 'strand': '+',
             'exons_tx': [{'start': 1, 'end': 20}]}
    original = deepcopy(entry)
    merged = merge_reference_catalog({}, {'TX_A': entry}, {'species': 'mouse'}, normalize_species)
    assert merged['TX_A']['metadata']['species'] == 'mouse'
    assert entry == original
    with pytest.raises(ValueError, match='identity_conflict'):
        merge_reference_catalog(merged, {'TX_A': {**entry, 'chrom': 'chrB'}}, {'species': 'mouse'}, normalize_species)
    with pytest.raises(ValueError, match='species_conflict'):
        merge_reference_catalog(merged, {'TX_A': entry}, {'species': 'human'}, normalize_species)


def hit(distance, orientation='reverse_complement', species='mouse', context='mature_spliced_cdna'):
    return {'mismatch_count': distance, 'match_orientation': orientation,
            'rna_context': context, 'transcript_id': 'TX_B',
            'reference_identity': {'species': species, 'assembly': species + '-fixture'},
            'genomic_segments': [{'chrom': 'chrA', 'start': 12, 'end': 31}],
            'matched_window_sequence': 'ACGT' * 5}


def test_reference_near_evidence_survives_without_generated_set_overlap():
    row = {'sequence': 'ATGC' * 5, 'near_match_search': {'status': 'performed'},
           'near_match_candidates': [hit(6), hit(2), hit(2), hit(1, orientation='input_sequence'),
                                     hit(4, species='human'), hit(3, context='unspliced_transcript_locus_pre_mrna')]}
    result = pipeline._compact_provided_aso_for_ranking(row, [], None)
    summary = result['reference_binding_evidence']
    assert len(summary['groups']) == 4
    group = summary['groups'][0]
    assert group['minimum_mismatches'] == 2 and group['maximum_mismatches'] == 6
    assert group['minimum_example_count'] == 2 and group['recorded_alignment_count'] == 3
    assert group['representative']['source_index'] == 1
    assert group['representative']['genomic_segments'] == row['near_match_candidates'][1]['genomic_segments']
    assert 'nearest_generated_candidates' not in result


def test_not_searched_reference_never_becomes_zero_mismatch_measurement():
    summary = comparison_binding_evidence({'near_match_search': {'status': 'not_run'}})
    assert summary['search']['status'] == 'not_run' and summary['groups'] == []


def test_trace_renderer_separates_never_generated_from_actual_rejection():
    rows = [{'name': 'outside', 'entered_pipeline': False, 'first_drop_stage': 'raw_generated',
             'terminal_classification': 'not_generated_by_search_space'},
            {'name': 'inside', 'entered_pipeline': True, 'first_drop_stage': 'initial_filter',
             'terminal_classification': 'removed_by_initial_filter'}]
    text = recorded_trace_markdown([{'reference_traces': [{'reference_identity': {'species': 'mouse', 'gene_symbol': 'GeneA'},
                                                         'trace_rows': rows}]}])
    assert 'mouse / GeneA' in text and '| outside | no |' in text
    assert '| inside | yes | removed_by_initial_filter | initial_filter |' in text
    assert 'Not an observed downstream rejection' in text


def test_composite_renders_current_comparison_field_and_scoped_trace(monkeypatch):
    evaluation = {'provided_asos': [{'name': 'probe', 'sequence': 'ATGC', 'mapping_status': 'unmapped'}]}
    comparison = {'task_type': 'compare_aso_sets', 'provided_count': 1, 'generated_count': 2,
        'overlap_count': 0, 'comparison_assessment': {'markdown': 'Recorded complementary near evidence remains conditional.',
                                                   'limitations': ['Activity is unmeasured.']}}
    trace = {'reference_traces': [{'reference_identity': {'species': 'mouse', 'gene_symbol': 'GeneA'},
                'trace_rows': [{'name': 'probe', 'entered_pipeline': False,
                                'terminal_classification': 'not_generated_by_search_space'}]}]}
    monkeypatch.setattr(pipeline, '_task_payloads_by_type',
        lambda _tasks, kind: {'compare_aso_sets': [comparison], 'trace_filter_decisions': [trace]}.get(kind, []))
    monkeypatch.setattr(pipeline, '_find_completed_context_payload', lambda *_: evaluation)
    result = pipeline._build_deterministic_composite_response({}, {})
    assert comparison['comparison_assessment']['markdown'] in result['markdown']
    assert 'mouse / GeneA' in result['markdown']
    assert 'Activity is unmeasured.' in result['limitations']


def test_missing_inventory_is_unknown_but_recorded_empty_inventory_is_rejection():
    row = {'sequence': 'ACGTCAGT'}
    stages = {'raw_generated': {'ACGTCAGT'}, 'initial_filter': set()}
    unknown = pipeline._sequence_stage_trace(row, stages, stage_sources={'raw_generated': ['raw.json']})
    assert unknown['terminal_classification'] == 'trace_incomplete'
    assert unknown['first_drop_stage'] == '' and unknown['stage_presence']['initial_filter'] is None
    assert unknown['stage_results']['sequence_filters']['status'] == 'unknown'
    excluded = pipeline._sequence_stage_trace(row, stages,
        stage_sources={'raw_generated': ['raw.json'], 'initial_filter': ['empty.json']})
    assert excluded['terminal_classification'] == 'filtered_during_initial_filter'
    assert excluded['stage_presence']['initial_filter'] is False


def test_actual_delivered_final_inventory_survives_missing_intermediate_files(tmp_path):
    import json
    path = tmp_path / 'artifacts/candidates/generated_aso_set.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'rows': [{'sequence': 'ACGTCAGT'}]}))
    stages = pipeline._collect_generated_stage_sets(str(tmp_path))
    trace = pipeline._sequence_stage_trace({'sequence': 'ACGTCAGT'}, stages['stage_sets'], stage_sources=stages['sources'])
    assert trace['terminal_classification'] == 'final_output'
    assert trace['entered_pipeline'] is True and trace['stage_presence']['raw_generated'] is None
    assert trace['first_drop_stage'] == ''


def test_available_empty_final_inventory_does_not_revert_to_scored_pool(tmp_path):
    import json
    candidates = tmp_path / 'artifacts/candidates'
    candidates.mkdir(parents=True)
    (candidates / 'generated_aso_set.json').write_text(json.dumps({'rows': []}))
    (candidates / 'gapmer_scored_uniqueness.json').write_text(json.dumps({'rows': [{'Sequence': 'ACGTCAGT'}]}))
    stages = pipeline._collect_generated_stage_sets(str(tmp_path))
    assert stages['sources']['final_output'] and not stages['stage_sets']['final_output']
    trace = pipeline._sequence_stage_trace({'sequence': 'ACGTCAGT'}, stages['stage_sets'], stage_sources=stages['sources'])
    assert trace['terminal_classification'] == 'not_selected_final_output'


def test_malformed_table_cannot_establish_a_negative_result(tmp_path):
    from aso_agents.dependency_references import sequence_inventory_available
    path = tmp_path / 'rows.json'
    path.write_text('{"error": "could not execute"}')
    assert not sequence_inventory_available(path)
    path.write_text('{"rows": []}')
    assert sequence_inventory_available(path)
    path.write_text('{invalid')
    assert not sequence_inventory_available(path)
