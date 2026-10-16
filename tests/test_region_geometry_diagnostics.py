from copy import deepcopy

import pytest

from aso_agents.region_overlap import constraint_execution_summary


@pytest.mark.parametrize('relation,length,width,possible', [
    ('full_containment', 21, 8, False), ('contains_feature', 21, 8, True),
    ('full_containment', 17, 40, True), ('contains_feature', 17, 40, False),
    ('full_containment', 19, 19, True), ('contains_feature', 19, 19, True),
    ('any_overlap', 18, 2, True),
])
def test_geometry_describes_direction_of_containment_without_changing_goal(relation, length, width, possible):
    goal = {'design_programs': [{'length': length, 'region_constraints': [{
        'coordinate_space': 'cdna', 'aso_overlap_requirement': relation,
        'intervals_bed0_half_open': [{'chrom': 'TX.2', 'start': 11, 'end': 11 + width}],
    }]}]}
    before = deepcopy(goal)
    result = constraint_execution_summary(goal)
    item = result[0]['explicit_interval_geometry'][0]
    assert item['single_interval_relation_possible'] is possible
    assert item['interval_length_nt'] == width
    assert item['candidate_length_nt'] == length
    assert goal == before


def test_minimum_overlap_unknown_lengths_and_exclusion_are_not_final_design_claims():
    goal = {'length': 15, 'region_constraints': [
        {'rows': [['chr2', 100, 105]], 'minimum_overlap_bp': 6, 'operator': 'exclude'},
        {'rows': [['chr2', 100, 130]], 'operator': 'union'},
    ]}
    result = constraint_execution_summary(goal)[0]
    assert [item['single_interval_relation_possible'] for item in result['explicit_interval_geometry']] == [False, True]
    assert result['allows_candidate_matching_no_constraints'] is True
    goal.pop('length')
    assert all(item['single_interval_relation_possible'] is None
               for item in constraint_execution_summary(goal)[0]['explicit_interval_geometry'])


def test_feature_without_explicit_coordinates_remains_unresolved():
    goal = {'length': 20, 'region_constraints': [{'source_type': 'transcript_feature', 'feature_name': 'utr3'}]}
    assert constraint_execution_summary(goal)[0]['explicit_interval_geometry'] == []
