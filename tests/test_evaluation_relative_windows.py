from copy import deepcopy

import jsonschema
import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.evaluation_feature_identity import resolve_evaluation_target_identities
from aso_agents.goal_contract import canonical_structured_goal_schema, canonical_task_context_schema
from aso_agents.mapping_evidence import requested_feature_membership


def catalog(strand='+'):
    intervals = [(101, 130), (301, 330), (501, 530)]
    if strand == '-':
        intervals.reverse()
    return {'fixture': {'transcript_id': 'TX_NEUTRAL', 'chrom': 'chr9', 'strand': strand,
        'metadata': {'versioned_id': 'TX_NEUTRAL.4'},
        'exons_tx': [{'index': i, 'start': s, 'end': e} for i, (s, e) in enumerate(intervals, 1)]}}


def target(site='donor', start=3, end=11):
    return {'label': 'named feature', 'transcript_id': 'TX_NEUTRAL.4',
        'splice_relative_window': {'reference_transcript_id': 'TX_NEUTRAL.4',
            'exon_label': 3, 'exon_index': 2, 'anchor_site': site,
            'relative_start_nt': start, 'relative_end_nt': end}}


def test_schema_and_normalizer_preserve_distinct_numbering_and_relative_frames():
    targets = [{'exon_label': 3, 'exon_index': 2, 'reference_transcript_id': 'TX_NEUTRAL.4'}, target()]
    targets = [{**item, 'objective_type': 'identity_anchor', 'target_type': 'exon' if 'exon_index' in item else 'regulatory_region'} for item in targets]
    goal = {'design_programs': [{'program_id': 'neutral', 'label': 'neutral', 'modality': 'steric',
        'intent': 'binding', 'objective': 'neutral feature'}], 'evaluation_targets': targets}
    jsonschema.validate(goal, canonical_structured_goal_schema())
    jsonschema.validate({'evaluation_targets': targets}, canonical_task_context_schema())
    assert pipeline._normalize_evaluation_targets(targets) == targets
    bad = target(); bad['splice_relative_window']['relative_start_nt'] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'evaluation_targets': [bad]}, canonical_task_context_schema())


@pytest.mark.parametrize('strand,site,expected', [
    ('+', 'donor', (333, 341)), ('-', 'donor', (290, 298)),
    ('+', 'acceptor', (304, 312)), ('-', 'acceptor', (319, 327)),
])
def test_relative_features_use_exact_annotation_ordinal_and_transcript_orientation(strand, site, expected):
    refs = catalog(strand); original = target(site); before = deepcopy((refs, original))
    result = resolve_evaluation_target_identities([original], refs, 'TX_NEUTRAL.4')[0]
    identity = result['relative_window_resolution']
    assert identity['status'] == 'resolved'
    assert identity['anchor_identity_resolution']['resolved_transcript_ordinal'] == 2
    assert (result['start'], result['end']) == expected
    assert result['end'] - result['start'] + 1 == 9
    assert (refs, original) == before
    row = {'mapping_hits': [{'transcript_id': 'TX_NEUTRAL.4', 'match_orientation': 'reverse_complement',
        'genomic_segments': [{'chrom': '9', 'start': expected[0] - 4, 'end': expected[0] + 4}]}]}
    feature = requested_feature_membership(row, [result], refs['fixture'], 'TX_NEUTRAL.4', catalog=refs)[0]
    assert feature['feature'] == 'interval'
    assert feature['observations'][0]['overlap_bp'] == 5
    assert not feature['observations'][0]['fully_contained']
    sites = pipeline._filter_splice_sites_by_targets(pipeline._build_transcript_splice_sites_for_eval(refs['fixture']), [result])
    assert len(sites) == 1 and sites[0]['exon_index'] == 2 and sites[0]['site_kind'] == site


@pytest.mark.parametrize('strand,expected', [('+', (295, 304)), ('-', (327, 336))])
def test_offsets_crossing_zero_are_inclusive_without_count_reinterpretation(strand, expected):
    result = resolve_evaluation_target_identities([target('acceptor', -6, 3)], catalog(strand), 'TX_NEUTRAL.4')[0]
    assert (result['start'], result['end']) == expected


@pytest.mark.parametrize('change', [
    {'exon_index': 77}, {'reference_transcript_id': 'TX_NEUTRAL.3'},
    {'relative_start_nt': 12}, {'relative_end_nt': None}, {'relative_start_nt': False},
    {'anchor_site': 'both'}, {'exon_index': None, 'exon_label': 8},
    {'exon_index': 3, 'anchor_site': 'donor'}, {'exon_index': 1, 'anchor_site': 'acceptor'},
    {'relative_start_nt': -400},
])
def test_unresolved_relative_windows_cannot_become_exon_membership_or_splice_targets(change):
    refs = catalog(); original = target(); original['splice_relative_window'].update(change)
    result = resolve_evaluation_target_identities([original], refs, 'TX_NEUTRAL.4')[0]
    assert result['relative_window_resolution']['status'] == 'unresolved'
    feature = requested_feature_membership({}, [result], refs['fixture'], 'TX_NEUTRAL.4', catalog=refs)[0]
    assert not feature['requested_feature_identity_resolved'] and 'start' not in feature
    assert feature['observations'] == []
    assert pipeline._filter_splice_sites_by_targets(pipeline._build_transcript_splice_sites_for_eval(refs['fixture']), [result]) == []


def test_explicit_coordinates_cannot_override_conflicting_relative_identity():
    refs = catalog(); original = {**target(), 'chrom': 'chr9', 'start': 304, 'end': 312, 'coordinate_system': '1based_inclusive'}
    result = resolve_evaluation_target_identities([original], refs, 'TX_NEUTRAL.4')[0]
    assert result['relative_window_resolution']['status'] == 'unresolved'
    feature = requested_feature_membership({}, [result], refs['fixture'], 'TX_NEUTRAL.4', catalog=refs)[0]
    assert not feature['requested_feature_identity_resolved'] and 'start' not in feature


def test_absolute_interval_and_exon_without_relative_window_keep_existing_semantics():
    refs = catalog()
    targets = [{'chrom': 'chr9', 'start': 310, 'end': 319, 'coordinate_system': '1based_inclusive'}, {'exon_index': 2}]
    result = resolve_evaluation_target_identities(targets, refs, 'TX_NEUTRAL.4')
    features = requested_feature_membership({}, result, refs['fixture'], 'TX_NEUTRAL.4', catalog=refs)
    assert [(f['feature'], f['start'], f['end']) for f in features] == [('interval', 310, 319), ('exon', 301, 330)]


def test_nested_label_does_not_inherit_a_different_parent_ordinal():
    refs = catalog(); original = {**target(), 'exon_index': 1}
    del original['splice_relative_window']['exon_index']
    result = resolve_evaluation_target_identities([original], refs, 'TX_NEUTRAL.4')[0]
    assert result['relative_window_resolution']['status'] == 'unresolved'
