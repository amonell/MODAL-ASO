import pytest
from aso_agents.design_qualification import strategy_contract, qualify_panel


@pytest.mark.parametrize('program,expected', [
    ({'modality': 'gapmer'}, 'gapmer_knockdown'),
    ({'modality': 'gapmer', 'preservation_targets': [{'transcript_id': 'TX'}]}, 'isoform_selective_gapmer'),
    ({'desired_effect': 'inclusion', 'intent': 'knockdown'}, 'splice_inclusion'),
    ({'desired_effect': 'skipping'}, 'splice_skipping'),
    ({'splice_event': {'event_type': 'intron_retention'}}, 'intron_retention'),
    ({'splice_event': {'event_type': 'intron_excision'}}, 'intron_excision'),
    ({'rbp_binding': {'rbp': 'neutral'}}, 'regulatory_site_blockade'),
    ({'mirna_binding': {'mirnas': ['neutral']}}, 'regulatory_site_blockade'),
    ({'region_constraints': [{'source_type': 'rbp_binding', 'rbp_binding': {'rbp': 'neutral'}}]}, 'regulatory_site_blockade'),
    ({'region_constraints': [{'source_type': 'mirna_binding', 'mirna_binding': {'mirnas': ['neutral']}}]}, 'regulatory_site_blockade'),
    ({'region_constraints': [{'source_type': 'transcript_feature', 'feature_name': 'start_codon'}]}, 'translation_start_blockade'),
])
def test_typed_mechanisms_have_distinct_requirements(program, expected):
    result = strategy_contract(program)
    assert result['strategy'] == expected
    assert result['biological_validation'] == 'not_established'
    assert 'declared_off_target_scope' in result['required_evidence']


def test_panel_receipt_does_not_hide_unknowns_or_confuse_binding_records_with_molecules():
    row = {'sequence': 'ACGT', 'program_id': 'one',
           'candidate_assessment': {'recommendation_status': 'exploratory', 'unavailable_evidence': ['complete_off_target_screen']}}
    goal = {'design_programs': [{'program_id': 'one', 'modality': 'steric', 'desired_effect': 'inclusion'}]}
    result = qualify_panel(goal, [row, row], [row], {})
    assert result['pool']['binding_record_count'] == 2
    assert result['pool']['distinct_molecule_count'] == 1
    assert result['selected']['recommendation_status_by_binding_record'] == {'exploratory': 1}
    assert result['programs'][0]['reference_status'] == 'unresolved'
    assert result['selected']['unavailable_evidence_by_binding_record'] == {'complete_off_target_screen': 1}
