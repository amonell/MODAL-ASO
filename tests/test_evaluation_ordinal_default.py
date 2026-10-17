"""Neutral regression controls for the native ordinal-resolution repair."""
from copy import deepcopy
import hashlib

import pytest

from aso_agents.evaluation_feature_identity import resolve_evaluation_target_identities
from aso_agents.mapping_evidence import requested_feature_membership
from aso_agents.evaluation_evidence import attach_measurement_references
from aso_agents import pipeline_tools as pipeline


def source_resolution(label=3):
    return {
        'status': 'ordinal_default', 'source': 'transcript_exon_index',
        'requested_exon_label': str(label),
        'evidence_refs': ['local:neutral-reference', 'local:neutral-cross-reference'],
        'source_sha256': hashlib.sha256(b'neutral-reference').hexdigest(),
        'xref_sha256': hashlib.sha256(b'neutral-cross-reference').hexdigest(),
        'reason': 'The source uses ordinary one-based transcript exon order.',
    }


def reference(strand='+', resolution=None):
    starts = [101, 201, 301, 401]
    if strand == '-':
        starts.reverse()
    return {
        'transcript_id': 'TX_REFERENCE', 'chrom': 'chr8', 'strand': strand,
        'metadata': {'versioned_id': 'TX_REFERENCE.7',
                     'exon_nomenclature_resolutions': {'3': source_resolution() if resolution is None else resolution}},
        'exons_tx': [{'index': index, 'start': start, 'end': start + 19}
                     for index, start in enumerate(starts, 1)],
    }


@pytest.mark.parametrize('strand', ['+', '-'])
def test_source_backed_ordinary_numbering_supports_membership_and_sites(strand):
    entry = reference(strand)
    catalog = {'TX_REFERENCE': entry}
    target = {'exon_label': 3, 'transcript_id': 'TX_REFERENCE.7'}
    original = deepcopy((catalog, target))
    resolved = resolve_evaluation_target_identities([target], catalog, 'TX_REFERENCE.7')
    identity = resolved[0]['exon_identity_resolution']
    exon = entry['exons_tx'][2]
    assert identity['status'] == 'resolved_reference_ordinal'
    assert identity['numbering_convention'] == 'one_based_transcript_ordinal'
    assert identity['biological_label_verified'] is False
    row = {'input_row_id': 'probe_1', 'name': 'neutral_probe', 'mapping_hits': [{
        'transcript_id': 'TX_REFERENCE.7', 'mapping_scope': 'spliced_cdna',
        'match_orientation': 'reverse_complement', 'mismatch_count': 0,
        'genomic_segments': [{'chrom': '8', 'start': exon['start'] + 2, 'end': exon['start'] + 11}],
    }]}
    feature = requested_feature_membership(row, resolved, entry, 'TX_REFERENCE.7', catalog=catalog)[0]
    assert feature['requested_feature_identity_resolved']
    assert feature['observations'][0]['fully_contained']
    row['requested_feature_membership'] = [feature]
    summary = attach_measurement_references({'provided_asos': [row]})
    assert summary['provided_asos'][0]['reference_tokens']['feature_1']
    sites = pipeline._filter_splice_sites_by_targets(pipeline._build_transcript_splice_sites_for_eval(entry), resolved)
    assert {site['exon_index'] for site in sites} == {3}
    assert {site['site_kind'] for site in sites} == {'acceptor', 'donor'}
    assert (catalog, target) == original


@pytest.mark.parametrize('change', [
    {'status': 'unresolved'}, {'status': 'not_attempted'},
    {'source': 'unknown'}, {'source_sha256': None}, {'source_sha256': 'unverified'},
    {'xref_sha256': None}, {'evidence_refs': []}, {'requested_exon_label': '2'},
])
def test_incomplete_or_conflicting_ordinal_provenance_stays_unresolved(change):
    resolution = {**source_resolution(), **change}
    entry = reference(resolution=resolution)
    catalog = {'TX_REFERENCE': entry}
    targets = resolve_evaluation_target_identities([{'exon_label': 3}], catalog, 'TX_REFERENCE.7')
    feature = requested_feature_membership({}, targets, entry, 'TX_REFERENCE.7', catalog=catalog)[0]
    assert not feature['requested_feature_identity_resolved']
    assert feature['observations'] == []
    assert pipeline._filter_splice_sites_by_targets(pipeline._build_transcript_splice_sites_for_eval(entry), targets) == []


def test_ordinary_source_does_not_satisfy_a_different_requested_version():
    entry = reference()
    resolved = resolve_evaluation_target_identities(
        [{'exon_label': 3, 'transcript_id': 'TX_REFERENCE.6'}],
        {'TX_REFERENCE': entry}, 'TX_REFERENCE.7',
    )
    assert resolved[0]['exon_identity_resolution']['status'] == 'unresolved'


def test_verified_alternate_name_still_selects_its_attested_ordinal():
    resolution = {**source_resolution(), 'status': 'verified_alternate_mapping',
                  'resolved_transcript_ordinal': 2}
    entry = reference(resolution=resolution)
    target = resolve_evaluation_target_identities([{'exon_label': 3}], {'TX_REFERENCE': entry}, 'TX_REFERENCE.7')[0]
    assert target['exon_identity_resolution']['status'] == 'verified_biological_label'
    assert target['exon_identity_resolution']['start'] == 201


def test_resolved_ordinal_preserves_explicit_conflicting_boundary_constraint():
    entry = reference()
    target = {'exon_label': 3, 'site_kind': 'acceptor', 'position': 201}
    targets = resolve_evaluation_target_identities([target], {'TX_REFERENCE': entry}, 'TX_REFERENCE.7')
    assert targets[0]['exon_identity_resolution']['status'] == 'resolved_reference_ordinal'
    assert pipeline._filter_splice_sites_by_targets(pipeline._build_transcript_splice_sites_for_eval(entry), targets) == []


def test_ordinary_numbering_reference_is_not_replaced_by_evaluated_isoform_order():
    full = reference()
    skipped = deepcopy(full)
    skipped['transcript_id'] = 'TX_SKIPPED'
    skipped['metadata'] = {'versioned_id': 'TX_SKIPPED.2'}
    skipped['exons_tx'] = [
        {'index': 1, 'start': 101, 'end': 120},
        {'index': 2, 'start': 301, 'end': 320},
        {'index': 3, 'start': 401, 'end': 420},
    ]
    catalog = {'TX_REFERENCE': full, 'TX_SKIPPED': skipped}
    targets = resolve_evaluation_target_identities([{
        'exon_label': 3, 'transcript_id': 'TX_SKIPPED.2',
        'exon_numbering_reference_transcript_id': 'TX_REFERENCE.7',
    }], catalog, 'TX_SKIPPED.2')
    feature = requested_feature_membership({}, targets, skipped, 'TX_SKIPPED.2', catalog=catalog)[0]
    assert feature['requested_feature_identity_resolved']
    assert feature['transcript_id'] == 'TX_REFERENCE.7'
    assert (feature['start'], feature['end']) == (301, 320)
    sites = pipeline._filter_splice_sites_by_targets(pipeline._build_transcript_splice_sites_for_eval(skipped), targets)
    assert {site['exon_index'] for site in sites} == {2}
