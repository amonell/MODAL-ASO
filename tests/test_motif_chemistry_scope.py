"""Published motif associations retain their documented chemistry scope."""
from copy import deepcopy

import pytest

from aso_agents.candidate_assessment import assess_candidates, motif_screen, rank_candidates


def molecule(sequence='ATGC' + 'A' * 10, links='PS'):
    return {'antisense_5to3': sequence,
            'nucleotide_sugars_5to3': ['LNA'] * 3 + ['DNA'] * 8 + ['LNA'] * 3,
            'linkages_5to3': [links] * 13, 'base_modifications_5to3': ['none'] * 14}


@pytest.mark.parametrize('backbone,expected', [
    (['PS'] * 13, 'matches_studied_chemistry'),
    (['PO'] * 13, 'outside_studied_chemistry'),
    (['PS'] * 12 + ['PO'], 'outside_studied_chemistry'),
    (None, 'chemistry_unresolved'),
    (['PS'], 'chemistry_unresolved'),
])
def test_backbone_qualifies_evidence_without_removing_sequence_flag(backbone, expected):
    value = molecule()
    value['linkages_5to3'] = backbone
    screen = motif_screen(value['antisense_5to3'], value)
    flag, = screen['flags']
    assert flag['rule_id'] == 'lna_383_tgc' and flag['positions_1based'] == [2]
    assert flag['applicability'] == expected
    assert not screen['excluded']
    assert 'not evidence of safety' in screen['interpretation']


def test_study_cpg_methylation_is_required_and_unknown_is_not_assumed():
    value = molecule('ATGCAACGAAAAAA')
    assert motif_screen(value['antisense_5to3'], value)['flags'][0]['applicability'] == 'outside_studied_chemistry'
    value['base_modifications_5to3'][6] = '5mC'
    screen = motif_screen(value['antisense_5to3'], value)
    assert screen['flags'][0]['applicability'] == 'matches_studied_chemistry'
    assert screen['flags'][1]['cytosines_5mC_at_all_hits'] is True
    value.pop('base_modifications_5to3')
    assert motif_screen(value['antisense_5to3'], value)['flags'][0]['applicability'] == 'chemistry_unresolved'


@pytest.mark.parametrize('backbone,expected_first', [('PS', 'clear'), ('PO', 'flagged')])
def test_only_matching_study_chemistry_changes_the_motif_tiebreak(backbone, expected_first):
    rows = []
    for label, sequence in [('flagged', 'ATGC' + 'A' * 10), ('clear', 'ATGA' + 'A' * 10)]:
        rows.append({'label': label, 'program_id': 'same', 'recommendation_rank_key': [4, 0.01],
            'molecule_specification': molecule(sequence, backbone),
            'off_target_count_status': 'complete_within_declared_scope',
            'off_target_exact_count_unit': 'genes', 'Off_target_mismatch_budget': 2,
            'Off_target_exact_count': 0, 'Off_target_1mm_count': 0, 'Off_target_2mm_count': 0,
            'off_target_search_provenance_by_species': {'human': {'assembly': 'fixture'}}})
    original = deepcopy(rows)
    assert rank_candidates(assess_candidates(rows, {}))[0]['label'] == expected_first
    assert rows == original
