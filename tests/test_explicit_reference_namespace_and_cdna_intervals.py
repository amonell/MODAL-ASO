"""Explicit transcript intervals and gene identifiers retain their reference spaces."""

import run_goal_intake as native
import pytest
from aso_agents.species_targets import resolve_explicit_target


def test_explicit_cdna_reference_binds_matching_chromosome_without_changing_bounds(tmp_path):
    from aso_agents.region_projection import genomic_sources
    transcript = {'transcript_id': 'TX.4', 'chrom': 'chrFixture', 'strand': '-',
                  'exons_tx': [{'index': 1, 'start': 1001, 'end': 1200}]}
    goal = {'design_programs': [{'program_id': 'region', 'transcript_id': 'TX.4'}],
            'isoform_transcripts': {'TX.4': transcript}}
    constraint = {'source_type': 'explicit_rows', 'coordinate_space': 'cdna',
                  'intervals_bed0_half_open': [{'chrom': 'Fixture', 'start': 19, 'end': 60}]}
    result = native._materialize_region_constraint_entry(
        structured_goal=goal, final_payload={}, run_dir=str(tmp_path),
        constraint=constraint, isoform_transcripts=goal['isoform_transcripts'])
    assert result['materialized_rows'][0][:3] == ['TX.4', 19, 60]
    assert result['materialization_summary']['transcript_reference_binding'][0]['original_reference'] == 'Fixture'
    assert genomic_sources(goal, result)[0]['genomic_rows'][0][:3] == ['chrFixture', 1140, 1181]
    assert constraint['intervals_bed0_half_open'][0]['chrom'] == 'Fixture'


@pytest.mark.parametrize('defect', ['ambiguous', 'no_declared_transcript', 'wrong_chromosome', 'wrong_version', 'out_of_bounds'])
def test_cdna_reference_binding_does_not_guess_through_conflicts(tmp_path, defect):
    transcript = {'transcript_id': 'TX.4', 'chrom': 'chrFixture', 'strand': '+',
                  'exons_tx': [{'start': 1001, 'end': 1200}]}
    goal = {'design_programs': [{'program_id': 'region', 'transcript_id': 'TX.4'}]}
    constraint = {'source_type': 'explicit_rows', 'coordinate_space': 'cdna',
                  'intervals_bed0_half_open': [{'chrom': 'Fixture', 'start': 19, 'end': 60}]}
    if defect == 'ambiguous':
        goal['design_programs'].append({'program_id': 'other', 'transcript_id': 'TX.5'})
    elif defect == 'no_declared_transcript':
        goal.clear()
    elif defect == 'wrong_chromosome':
        constraint['intervals_bed0_half_open'][0]['chrom'] = 'Different'
    elif defect == 'wrong_version':
        goal['design_programs'][0]['transcript_id'] = 'TX.8'
    else:
        constraint['intervals_bed0_half_open'][0]['end'] = 201
    result = native._materialize_region_constraint_entry(
        structured_goal=goal, final_payload={}, run_dir=str(tmp_path),
        constraint=constraint, isoform_transcripts={'TX.4': transcript})
    assert result['materialized_rows'] == []


def test_mature_transcript_interval_is_not_treated_as_a_genomic_chromosome(tmp_path):
    transcript = {
        'transcript_id': 'ENST_NEUTRAL.2', 'chrom': 'chr7', 'strand': '-',
        'exons_tx': [{'start': 10001, 'end': 10100, 'cdna_start': 1, 'cdna_end': 100}],
    }
    constraint = {
        'source_type': 'explicit_rows', 'coordinate_space': 'genomic',
        'coordinate_convention': 'mature_transcript_1based_inclusive',
        'aso_overlap_requirement': 'full_containment',
        'intervals_bed0_half_open': [{'chrom': 'ENST_NEUTRAL.2', 'start': 19, 'end': 60}],
    }
    result = native._materialize_region_constraint_entry(
        structured_goal={'gene_symbol': 'GENE'}, final_payload={}, run_dir=str(tmp_path),
        constraint=constraint, isoform_transcripts={'ENST_NEUTRAL.2': transcript},
    )
    assert result['coordinate_space'] == 'cdna'
    assert result['materialized_cdna_rows'][0][:3] == ['ENST_NEUTRAL.2', 19, 60]
    assert result['aso_overlap_requirement'] == 'full_containment'


def test_mature_transcript_interval_rejects_unresolved_bounds(tmp_path):
    result = native._materialize_region_constraint_entry(
        structured_goal={}, final_payload={}, run_dir=str(tmp_path),
        constraint={'source_type': 'explicit_rows',
                    'coordinate_convention': 'mature_transcript_1based_inclusive',
                    'intervals_bed0_half_open': [{'chrom': 'TX.1', 'start': 90, 'end': 120}]},
        isoform_transcripts={'TX.1': {'transcript_id': 'TX.1', 'exons_tx': [
            {'start': 1, 'end': 100, 'cdna_start': 1, 'cdna_end': 100}]}},
    )
    assert result['materialized_rows'] == []
    assert result['materialization_summary']['status'] == 'unresolved'


def test_cdna_windows_inherit_only_unambiguous_program_transcript(tmp_path):
    transcript = {
        'transcript_id': 'TX.1', 'chrom': 'chr7', 'strand': '-',
        'exons_tx': [{'start': 10001, 'end': 10200, 'cdna_start': 1, 'cdna_end': 200}],
    }
    constraint = {
        'source_type': 'explicit_rows', 'coordinate_space': 'genomic',
        'coordinate_convention': 'mature_transcript_1based_inclusive',
        'aso_overlap_requirement': 'full_containment',
        'intervals_bed0_half_open': [{'label': 'region_a', 'start': 19, 'end': 80}],
    }
    goal = {'design_programs': [{'program_id': 'design_a', 'transcript_id': 'TX.1'}]}
    result = native._materialize_region_constraint_entry(
        structured_goal=goal, final_payload={}, run_dir=str(tmp_path),
        constraint=constraint, isoform_transcripts={'TX.1': transcript},
    )
    assert result['materialized_cdna_rows'][0][:4] == ['TX.1', 19, 80, 'region_a']
    assert result['row_count'] == 1
    assert result['materialization_summary']['validated_reference_rows'] == 1

    ambiguous = {'design_programs': [
        {'program_id': 'design_a', 'transcript_id': 'TX.1'},
        {'program_id': 'design_b', 'transcript_id': 'TX.2'},
    ]}
    unresolved = native._materialize_region_constraint_entry(
        structured_goal=ambiguous, final_payload={}, run_dir=str(tmp_path),
        constraint=constraint, isoform_transcripts={'TX.1': transcript},
    )
    assert unresolved['materialized_cdna_rows'] == []
    assert unresolved['row_count'] == 0


def test_cdna_window_does_not_inherit_program_reference_for_genomic_interval(tmp_path):
    result = native._materialize_region_constraint_entry(
        structured_goal={'design_programs': [{'program_id': 'design_a', 'transcript_id': 'TX.1'}]},
        final_payload={}, run_dir=str(tmp_path),
        constraint={'source_type': 'explicit_rows', 'coordinate_convention': 'genomic_1based_inclusive',
                    'intervals_bed0_half_open': [{'start': 19, 'end': 80}]},
    )
    assert result['materialized_rows'] == []


def test_numeric_external_gene_alias_uses_parent_and_symbol_without_false_mismatch():
    goal = {'species_targets': [{'species': 'human', 'gene_symbol': 'GENE',
                                 'transcript_id': 'TX.2', 'gene_id': '12345'}]}
    fetch = lambda _: {'id': 'TX', 'versioned_id': 'TX.2', 'species': 'human',
                       'gene_id': 'ENSG00000000001'}
    lookup = lambda _: {'species': 'human', 'display_name': 'GENE'}
    result = resolve_explicit_target(goal, 'human', fetch, lookup)
    assert result['status'] == 'resolved'
    assert result['gene_id'] == 'ENSG00000000001'
    assert 'not_crosswalk_verified' in result['numeric_gene_alias_status']
    assert resolve_explicit_target(goal, 'human', fetch,
                                   lambda _: {'species': 'human', 'display_name': 'OTHER'})['status'] == 'error'
    goal['species_targets'][0]['gene_id'] = 'ENSG00000000002'
    assert resolve_explicit_target(goal, 'human', fetch, lookup)['error'] == 'transcript_parent_gene_mismatch'
