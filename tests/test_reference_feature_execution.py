"""Exercise reference identity and region compilation across stage boundaries."""
from copy import deepcopy
import importlib

import pytest
import run_goal_intake as native
from aso_agents.declared_reference_catalog import complete_target_references
from aso_agents.gapmer_selectivity import resolve_catalog_label
from aso_agents.mature_steric import compile_regions
from aso_agents.preservation_context import preserved_catalog_labels
from aso_agents.transcript_features import materialize_feature


def test_exact_accession_receipt_selects_feature_without_version_substitution():
    from test_reference_feature_constraints import reference
    entry = reference()
    entry['metadata']['reference_accession_resolution'] = {
        'requested_accession': 'NM_123456.7', 'exact_accession_verified': True}
    goal = {'isoform_transcripts': {'target': entry}, 'transcript_policy': 'user_locked',
            'transcript_lock': {'value': 'NM_123456.7'}}
    assert native._build_utr_bed_rows(goal, 'utr3')
    goal['transcript_lock']['value'] = 'NM_123456.8'
    assert native._build_utr_bed_rows(goal, 'utr3') == []
    goal['transcript_lock']['value'] = 'NM_123456.7'
    entry['metadata']['reference_accession_resolution']['exact_accession_verified'] = False
    assert native._build_utr_bed_rows(goal, 'utr3') == []


def test_constraint_free_gapmer_completes_exact_reference_before_region_build(tmp_path, monkeypatch):
    from test_declared_reference_catalog import structure
    chooser = importlib.import_module('aso_agents.transcript_chooser_agent')
    reference = 'NM_123456.7'
    physical = 'ENST00000000002.3'
    monkeypatch.setattr(chooser, 'resolve_external_transcript_accession', lambda *a, **k: {
        'transcript_id': physical, 'requested_accession': reference, 'method': 'local_gencode_refseq_xref'})
    monkeypatch.setattr(native, '_fetch_transcript_structure', lambda _: structure(physical))
    goal = {'gene_id': 'ENSG00000000001', 'species': 'human',
            'design_programs': [{'program_id': 'one', 'modality': 'gapmer',
                                 'intent': 'knockdown', 'transcript_id': reference}]}
    result = native._materialize_region_constraints(goal, {}, str(tmp_path))
    assert result['constraint_count'] == 0
    rows, _ = native._build_gapmer_program_region_rows(
        goal, goal['isoform_transcripts'], None, goal['design_programs'][0])
    assert rows and rows[0][0] == physical


def test_external_alias_keeps_existing_label_and_resolves_preservation(monkeypatch):
    from test_declared_reference_catalog import structure
    chooser = importlib.import_module('aso_agents.transcript_chooser_agent')
    physical = 'ENST00000000002.3'
    monkeypatch.setattr(native, '_fetch_transcript_structure', lambda _: structure(physical))
    goal = {'gene_id': 'ENSG00000000001', 'species': 'human',
            'design_programs': [{'program_id': 'one', 'transcript_id': physical}]}
    assert complete_target_references(native, goal)
    goal['isoform_transcripts']['chosen'] = goal['isoform_transcripts'].pop(physical)
    for accession in ('NM_123456.7', 'NM_654321.2'):
        monkeypatch.setattr(chooser, 'resolve_external_transcript_accession', lambda *a, **k: {
            'transcript_id': physical, 'requested_accession': accession, 'method': 'local_gencode_refseq_xref'})
        goal['design_programs'][0]['transcript_id'] = accession
        assert complete_target_references(native, goal)
    catalog = goal['isoform_transcripts']
    assert set(catalog) == {'chosen'}
    for accession in ('NM_123456.7', 'NM_654321.2'):
        assert resolve_catalog_label(accession, catalog, catalog) == 'chosen'
    assert resolve_catalog_label('NM_123456.8', catalog, catalog) is None
    goal['preservation_targets'] = [{'action': 'preserve', 'preservation_target_type': 'transcript',
                                    'isoform_label': 'chosen', 'transcript_id': 'NM_123456.7'}]
    assert preserved_catalog_labels(goal, catalog) == {'chosen'}


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('with_flanks', [False, True])
def test_annotated_feature_constraints_seed_mature_rna_and_keep_flank_window(strand, with_flanks):
    from test_reference_feature_constraints import reference, constraint
    entry = reference(strand)
    source = constraint(aso_overlap_requirement='any_overlap', minimum_overlap_bp=1)
    materialized = materialize_feature(source, {'chosen': entry}, 'chosen')
    goal = {'isoform_transcripts': {'chosen': entry}, 'materialized_region_constraints': {
        'by_program': {'one': [materialized, deepcopy(materialized)]}}}
    program = {'program_id': 'one', 'modality': 'steric', 'intent': 'binding',
               'transcript_id': 'TXA.3', 'coordinate_space': 'cdna', 'length': 10,
               'region_constraints': [source]}
    if with_flanks:
        program['flanks'] = {'up': 4, 'down': 15}
    base, _ = native._build_steric_program_region_rows(goal, program)
    rows, compiled = compile_regions(native, goal, program, base)
    assert rows
    assert compiled['mature_rna']['search_rows_cdna'][0][1:3] == ([1, 21] if with_flanks else [0, 50])
    assert len(compiled['mature_rna']['constraint_filters_cdna']) == 2
