from copy import deepcopy

import pytest

from aso_agents.evaluation_feature_identity import resolve_evaluation_target_identities
from aso_agents.mapping_evidence import requested_feature_membership
from aso_agents.evaluation_evidence import attach_measurement_references
from aso_agents import pipeline_tools as pipeline


def catalog(strand='+'):
    skipped = {'transcript_id': 'TX_A', 'chrom': 'chr6', 'strand': strand,
        'metadata': {'versioned_id': 'TX_A.2'}, 'exons_tx': [
            {'index': 1, 'start': 101, 'end': 120},
            {'index': 2, 'start': 301, 'end': 320},
            {'index': 3, 'start': 401, 'end': 420}]}
    full = {'transcript_id': 'TX_B', 'chrom': 'chr6', 'strand': strand,
        'metadata': {'versioned_id': 'TX_B.5'}, 'exons_tx': [
            {'index': 1, 'start': 101, 'end': 120},
            {'index': 2, 'start': 201, 'end': 220},
            {'index': 3, 'start': 301, 'end': 320},
            {'index': 4, 'start': 401, 'end': 420}]}
    return {'TX_A': skipped, 'TX_B': full}


def row(start=306):
    return {'input_row_id': 'supplied_1', 'name': 'probe', 'mapping_hits': [{
        'transcript_id': 'TX_A.2', 'mapping_scope': 'spliced_cdna',
        'match_orientation': 'reverse_complement', 'mismatch_count': 0,
        'genomic_segments': [{'chrom': '6', 'start': start, 'end': start + 9}]}]}


@pytest.mark.parametrize('strand', ['+', '-'])
def test_verified_biological_label_maps_to_its_attested_ordinal(strand):
    refs = catalog(strand)
    refs['TX_A']['metadata']['exon_nomenclature_resolutions'] = {'3': {
        'status': 'verified_alternate_mapping', 'resolved_transcript_ordinal': 2,
        'requested_exon_label': 3, 'evidence_refs': ['local_authoritative_fixture']}}
    targets = [{'exon_label': 3, 'transcript_id': 'TX_A.2'}]
    before = deepcopy((refs, targets))
    resolved = resolve_evaluation_target_identities(targets, refs, 'TX_A.2')
    features = requested_feature_membership(row(), resolved, refs['TX_A'], 'TX_A.2', catalog=refs)
    feature = features[0]
    assert feature['requested_exon_label'] == 3 and feature['exon_index'] == 2
    assert (feature['start'], feature['end']) == (301, 320)
    assert feature['observations'][0]['fully_contained']
    assert (refs, targets) == before
    sites = pipeline._build_transcript_splice_sites_for_eval(refs['TX_A'])
    selected = pipeline._filter_splice_sites_by_targets(sites, resolved)
    assert {s['exon_index'] for s in selected} == {2}
    assert {s['site_kind'] for s in selected} == {'acceptor', 'donor'}


@pytest.mark.parametrize('status', ['unresolved', 'ordinal_default', 'not_attempted'])
def test_unverified_biological_label_cannot_become_a_wrong_membership_claim(status):
    refs = catalog(); targets = [{'exon_label': 3, 'transcript_id': 'TX_A.2'}]
    resolved = resolve_evaluation_target_identities(targets, refs, 'TX_A.2', lambda *_: {'status': status})
    observation = row()
    observation['requested_feature_membership'] = requested_feature_membership(observation, resolved, refs['TX_A'], 'TX_A.2', catalog=refs)
    feature = observation['requested_feature_membership'][0]
    assert not feature['requested_feature_identity_resolved']
    assert feature['observations'] == [] and 'start' not in feature
    # The alternative ordinal candidate is visible as a hypothesis, not a fact.
    assert feature['identity_resolution']['start'] == 401
    assert pipeline._filter_splice_sites_by_targets(pipeline._build_transcript_splice_sites_for_eval(refs['TX_A']), resolved) == []
    summary = attach_measurement_references({'provided_asos': [observation]})
    assert not any('feature_' in key for key in summary['provided_asos'][0]['reference_tokens'])


def test_explicit_numbering_reference_can_differ_from_evaluated_transcript():
    refs = catalog(); targets = [{'exon_index': 3, 'transcript_id': 'TX_A.2',
                                 'exon_numbering_reference_transcript_id': 'TX_B.5'}]
    resolved = resolve_evaluation_target_identities(targets, refs, 'TX_A.2')
    feature = requested_feature_membership(row(), resolved, refs['TX_A'], 'TX_A.2', catalog=refs)[0]
    assert feature['transcript_id'] == 'TX_B.5'
    assert feature['evaluated_transcript_id'] == 'TX_A.2'
    assert feature['start'] == 301 and feature['observations'][0]['fully_contained']
    sites = pipeline._filter_splice_sites_by_targets(pipeline._build_transcript_splice_sites_for_eval(refs['TX_A']), resolved)
    assert {s['exon_index'] for s in sites} == {2}
    off = requested_feature_membership(row(405), resolved, refs['TX_A'], 'TX_A.2', catalog=refs)[0]
    assert not off['observations'][0]['overlaps']


def test_reference_version_mismatch_and_unavailable_ordinals_stay_unresolved():
    refs = catalog(); refs['TX_B.4'] = refs['TX_B']
    for target in [{'exon_index': 3, 'reference_transcript_id': 'TX_B.4'}, {'exon_index': 99}]:
        resolved = resolve_evaluation_target_identities([target], refs, 'TX_A.2')[0]
        assert resolved['exon_identity_resolution']['status'] == 'unresolved'


def test_normalization_keeps_coordinate_convention_and_numbering_reference():
    raw = {'exon_index': 3, 'exon_numbering_reference_transcript_id': 'TX_B.5',
           'transcript_id': 'TX_A.2', 'start': 300, 'end': 320, 'chrom': 'chr6', 'coordinate_system': 'bed'}
    normalized = pipeline._normalize_evaluation_targets([raw])
    assert normalized == [raw]
    refs = catalog()
    feature = requested_feature_membership(row(), normalized, refs['TX_A'], 'TX_A.2', catalog=refs)[0]
    assert feature['feature'] == 'interval' and feature['start'] == 301
    assert feature['observations'][0]['fully_contained']


@pytest.mark.parametrize('explicit', [
    {'position': 401}, {'chrom': 'chr7'}, {'strand': '-'},
])
def test_explicit_boundary_constraints_are_not_replaced_by_an_exon_label(explicit):
    refs = catalog()
    targets = [{'exon_index': 2, 'transcript_id': 'TX_A.2', 'site_kind': 'acceptor', **explicit}]
    resolved = resolve_evaluation_target_identities(targets, refs, 'TX_A.2')
    sites = pipeline._build_transcript_splice_sites_for_eval(refs['TX_A'])
    assert pipeline._filter_splice_sites_by_targets(sites, resolved) == []


def test_consistent_explicit_boundary_and_reference_identity_select_one_site():
    refs = catalog()
    targets = [{'exon_index': 2, 'transcript_id': 'TX_A.2', 'site_kind': 'acceptor',
                'position': 301, 'chrom': '6', 'strand': '+'}]
    resolved = resolve_evaluation_target_identities(targets, refs, 'TX_A.2')
    sites = pipeline._filter_splice_sites_by_targets(
        pipeline._build_transcript_splice_sites_for_eval(refs['TX_A']), resolved)
    assert len(sites) == 1 and sites[0]['exon_index'] == 2
