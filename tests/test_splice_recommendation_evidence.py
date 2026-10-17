from copy import deepcopy
import json

import pytest

from aso_agents.splice_recommendations import (
    annotate_route_recommendation, objective_prediction_evidence,
    prediction_evidence, preserve_splice_site_evidence, recommendation_sort_key,
    report_recommendation_fields, resolve_routing_exon_objectives,
)


def goal(effect='inclusion', strand='+', reference='TX_NEUTRAL.2'):
    return {'routing_job': {'desired_effect': effect, 'exon_index': 2, 'transcript_id': reference},
        'isoform_transcripts': {'neutral_label': {'transcript_id': 'TX_NEUTRAL',
            'metadata': {'versioned_id': 'TX_NEUTRAL.2'}, 'chrom': 'chr8', 'strand': strand,
            'exons_tx': [{'index': 1, 'start': 101, 'end': 120},
                         {'index': 2, 'start': 201, 'end': 240}]}}}


def site(values=(0.03, 0.02), strand='+', **updates):
    result = {'isoform_label': 'neutral_label', 'reference_transcript_id': 'TX_NEUTRAL.2',
        'evaluation_status': 'evaluated', 'window_bp': 10,
        'acceptor': {'chrom': 'chr8', 'position': 240 if strand == '-' else 201, 'strand': strand},
        'donor': {'chrom': 'chr8', 'position': 201 if strand == '-' else 240, 'strand': strand},
        'delta_inclusion_deletion': values[0], 'delta_inclusion_replacement': values[1]}
    result.update(updates)
    return result


def annotated(values, site_values, effect='inclusion'):
    direction = 'increase' if effect == 'inclusion' else 'decrease'
    row = {'splice_site_deltas': [site(values)], 'Score': 100}
    return annotate_route_recommendation(row, [{'label': 'acceptor@201', 'direction': direction,
        'delta_deletion': site_values[0], 'delta_replacement': site_values[1]}],
        resolve_routing_exon_objectives(goal(effect)))


@pytest.mark.parametrize('effect,sign', [('inclusion', 1), ('skipping', -1)])
def test_large_opposing_signal_cannot_outrank_supported_requested_direction(effect, sign):
    good = annotated((sign*.03, sign*.02), (sign*.02, sign*.02), effect)
    bad = annotated((-sign*.9, -sign*.8), (sign*.8, sign*.01), effect)
    assert recommendation_sort_key(good) > recommendation_sort_key(bad)
    assert good['recommendation_status'] == 'direction_supported'
    assert bad['recommendation_status'] == 'opposing_predictions'
    assert bad['exon_objective_evidence'][0]['measurements'][0]['values'] == [-sign*.9, -sign*.8]


def test_disagreement_remains_an_exploratory_result_not_an_eligibility_failure():
    row = annotated((.2, -.1), (.5, -.1))
    assert row['recommendation_status'] == 'mixed_or_partial_predictions'
    assert row['exon_objective_evidence'][0]['measurements'][0]['prediction']['state'] == 'discordant'
    assert 'exploratory' in row['recommendation_note']


@pytest.mark.parametrize('values', [(None, None), ('NaN', float('inf')), (True, False)])
def test_missing_or_nonfinite_predictions_never_become_measured_zeros(values):
    evidence = prediction_evidence(values, 'increase')
    assert evidence['state'] == 'unavailable' and evidence['mean_delta'] is None


def test_preservation_does_not_average_opposing_predictions_into_neutrality():
    evidence = prediction_evidence([.8, -.8], 'preserve')
    assert evidence['state'] == 'nonzero_predicted_change'
    assert evidence['directional_mean'] == -.8


def test_unknown_direction_uses_quality_not_absolute_splice_magnitude():
    weak = {'mean_splice_delta': .01, 'score': 2}
    large = {'mean_splice_delta': -.9, 'score': 1}
    assert recommendation_sort_key(weak, 'binding') > recommendation_sort_key(large, 'binding')


def test_job_effect_does_not_inherit_another_jobs_exon_outcome():
    payload = goal()
    payload['desired_effect'] = 'skipping'
    assert resolve_routing_exon_objectives(payload)[0]['direction'] == 'increase'
    del payload['routing_job']['desired_effect']
    payload['routing_job']['desired_delta'] = 'increase'
    assert resolve_routing_exon_objectives(payload) == []


def test_version_mismatch_stays_unresolved_even_when_coordinates_match():
    objective = resolve_routing_exon_objectives(goal(reference='TX_NEUTRAL.1'))[0]
    assert objective['status'] == 'unresolved'
    result = objective_prediction_evidence({'splice_site_deltas': [site()]}, objective)
    assert result['status'] == 'unresolved_identity' and not result['measurements']


def test_verified_alternate_label_uses_resolved_ordinal_without_gene_rules():
    payload = goal()
    payload['routing_job'].pop('exon_index')
    payload['routing_job']['exon_label'] = 7
    payload['isoform_transcripts']['neutral_label']['metadata']['exon_nomenclature_resolutions'] = {
        '7': {'status': 'verified_alternate_mapping', 'resolved_transcript_ordinal': 2}}
    result = resolve_routing_exon_objectives(payload)[0]
    assert result['status'] == 'resolved' and result['exon_index'] == 2
    assert result['acceptor_position'] == 201 and result['donor_position'] == 240


@pytest.mark.parametrize('strand', ['+', '-'])
def test_pair_binding_respects_reference_coordinates_and_strand(strand):
    objective = resolve_routing_exon_objectives(goal(strand=strand))[0]
    record = site(strand=strand)
    assert objective_prediction_evidence({'splice_site_deltas': [record]}, objective)['status'] == 'recorded_pair_predictions'
    for changed in (
        site(strand=strand, reference_transcript_id='TX_OTHER.2'),
        site(strand=strand, donor={'chrom': 'chr9', 'position': objective['donor_position']}),
        site(strand='-' if strand == '+' else '+'),
        site(strand=strand, reference_binding_status='conflicting'),
    ):
        assert not objective_prediction_evidence({'splice_site_deltas': [changed]}, objective)['measurements']


def test_unlabelled_summary_and_unevaluated_zero_are_not_pair_evidence():
    objective = resolve_routing_exon_objectives(goal())[0]
    row = {'delta_inclusion_deletion': .9, 'delta_inclusion_replacement': .9,
        'splice_site_deltas': [site((0, 0), evaluation_status='outside_evaluated_window')]}
    assert objective_prediction_evidence(row, objective)['status'] == 'paired_measurement_unavailable'


def test_duplicate_site_views_do_not_double_count_and_conflicts_stay_unknown():
    objective = resolve_routing_exon_objectives(goal())[0]
    row = {'splice_site_deltas': [site(), site()]}
    result = objective_prediction_evidence(row, objective)
    assert len(result['measurements']) == 1
    assert result['measurements'][0]['source_site_indices'] == [0, 1]
    row['splice_site_deltas'][1]['delta_inclusion_deletion'] = -.8
    assert objective_prediction_evidence(row, objective)['status'] == 'conflicting_pair_measurements'


def test_raw_scoring_cache_remains_free_of_job_specific_outcomes():
    raw = {'splice_site_deltas': [site(reference_transcript_id=None)]}
    before = deepcopy(raw)
    saved = preserve_splice_site_evidence(raw, goal()['isoform_transcripts'])
    assert raw == before and saved[0]['reference_transcript_id'] == 'TX_NEUTRAL.2'
    assert 'desired_effect' not in saved[0]
    increasing = objective_prediction_evidence({'splice_site_deltas': saved}, resolve_routing_exon_objectives(goal())[0])
    decreasing = objective_prediction_evidence({'splice_site_deltas': saved}, resolve_routing_exon_objectives(goal('skipping'))[0])
    assert increasing['rank_key'] > decreasing['rank_key']


def test_csv_normalization_preserves_complete_recommendation_evidence():
    original = annotated((.03, -.02), (.1, .02))
    encoded = {key: json.dumps(value) if isinstance(value, (list, dict)) else value
               for key, value in report_recommendation_fields(original).items()}
    assert report_recommendation_fields(encoded) == report_recommendation_fields(original)


def test_multiple_explicit_exon_objectives_keep_their_own_pair_evidence():
    payload = goal()
    payload['routing_job'].pop('exon_index')
    payload['routing_job']['exon_labels'] = [1, 2]
    # Labels without a numbering source are retained but must stay unresolved.
    assert all(item['status'] == 'unresolved' for item in resolve_routing_exon_objectives(payload))
    import hashlib
    source_sha = hashlib.sha256(b'neutral transcript annotation').hexdigest()
    payload['isoform_transcripts']['neutral_label']['metadata']['exon_nomenclature_resolutions'] = {
        str(index): {'status': 'ordinal_default', 'source': 'transcript_exon_index',
                    'requested_exon_label': str(index), 'evidence_refs': ['local:neutral'],
                    'source_sha256': source_sha, 'xref_sha256': source_sha}
        for index in (1, 2)}
    objectives = resolve_routing_exon_objectives(payload)
    assert [item['exon_index'] for item in objectives] == [1, 2]
    row = annotate_route_recommendation({'splice_site_deltas': [site()]}, [], objectives)
    assert row['recommendation_status'] == 'incomplete_prediction_evidence'
    assert row['exon_objective_evidence'][0]['measurements'] == []
    assert row['exon_objective_evidence'][1]['measurements']


def test_pair_with_missing_strand_does_not_establish_objective_identity():
    pair = site()
    pair['acceptor'].pop('strand')
    pair['donor'].pop('strand')
    objective = resolve_routing_exon_objectives(goal())[0]
    assert not objective_prediction_evidence({'splice_site_deltas': [pair]}, objective)['measurements']
