from copy import deepcopy

import pytest

from aso_agents.response_assembly import assembly_source, feature_check_outcomes
from aso_agents.evaluation_evidence import row_evidence_inventory


@pytest.mark.parametrize('feature', [None, {}, {'observations': None}])
def test_absent_checks_and_outcomes_remain_unknown(feature):
    result = feature_check_outcomes(feature)
    assert result['records_present'] is False
    assert result['recorded_check_count'] is None
    assert result['overlap_true_record_count'] is None
    assert result['overlap_false_record_count'] is None
    assert result['overlap_unrecorded_record_count'] is None


def test_only_exact_booleans_are_positive_or_negative_and_contexts_stay_separate():
    context = {'mapping_transcript_id': 'TX_A.2', 'mapping_scope': 'pre_mrna_locus',
               'match_orientation': 'reverse_complement', 'mismatch_count': 2}
    observations = [{**context, 'overlaps': value} for value in [False, False, True, None, 0, 'false']]
    observations.append({**context, 'match_orientation': 'input_sequence', 'overlaps': True})
    observations.append({**context, 'mapping_scope': 'spliced_cdna', 'overlaps': False})
    observations.append({**context, 'mapping_transcript_id': 'TX_B.1', 'overlaps': True})
    observations.append({**context, 'mismatch_count': 0, 'overlaps': True})
    feature = {'observations': observations}
    before = deepcopy(feature)
    result = feature_check_outcomes(feature)
    assert result['recorded_check_count'] == 10
    assert result['overlap_true_record_count'] == 4
    assert result['overlap_false_record_count'] == 3
    assert result['overlap_unrecorded_record_count'] == 3
    assert len(result['context_groups']) == 5
    first = result['context_groups'][0]
    assert first['overlap_true_record_count'] == 1
    assert first['overlap_false_record_count'] == 2
    assert first['overlap_unrecorded_record_count'] == 3
    assert feature == before


def test_empty_is_distinct_from_absent_and_negative_checks_are_not_positive():
    empty = feature_check_outcomes({'observations': []})
    assert empty['records_present'] is True and empty['recorded_check_count'] == 0
    assert empty['overlap_true_record_count'] == 0
    negative = feature_check_outcomes({'observations': [{'overlaps': False}] * 4})
    assert negative['recorded_check_count'] == 4
    assert negative['overlap_true_record_count'] == 0
    assert negative['overlap_false_record_count'] == 4


def test_assembly_joins_only_explicit_same_row_source_identity_and_preserves_inventory():
    rows = [{'input_row_id': label, 'name': label,
             'requested_feature_membership': [{'observations': [{'overlaps': outcome}] * 3}]}
            for label, outcome in [('alpha', False), ('beta', True)]]
    for row in rows:
        row['evidence_inventory'] = row_evidence_inventory(row)
    # A stale or unavailable source pointer must not borrow an existing feature.
    rows[1]['evidence_inventory']['feature_evidence_availability']['requested_feature_membership']['features'][0]['source_field'] = 'requested_feature_membership[99]'
    payload = {'evaluation_summary': {'provided_asos': rows}}
    response = {'evaluation_assessments': [
        {key: row[key] for key in ('input_row_id', 'name')} for row in rows]}
    before = deepcopy(payload)
    source = assembly_source(payload, response)
    features = [row['recorded_inventory']['feature_evidence_availability']['requested_feature_membership']['features'][0]
                for row in source['evidence_scope']['observations']]
    assert features[0]['recorded_check_outcomes']['overlap_false_record_count'] == 3
    assert features[0]['recorded_check_outcomes']['overlap_true_record_count'] == 0
    assert features[1]['recorded_check_outcomes']['records_present'] is False
    assert features[1]['recorded_check_outcomes']['overlap_true_record_count'] is None
    assert payload == before
