from copy import deepcopy
import pytest
from aso_agents.target_geometry import transcript_feature_geometry, explicit_window_geometry
from aso_agents import pipeline_tools


def fixture(strand):
    anchor = 230 if strand == '+' else 201
    bounds = [anchor+1-1, anchor+60] if strand == '+' else [anchor-60-1, anchor-1]
    record = {'record_index': 0, 'requested_target': {'reference_transcript_id': 'TX.4',
        'anchor_site': 'donor', 'relative_start_nt': 1, 'relative_end_nt': 60, 'feature_name': 'user regulatory label'},
        'resolved_target': {'transcript_id': 'TX.4', 'anchor_site': 'donor', 'anchor_position': anchor,
        'materialized_rows': [['chr8', *bounds, 'window', 0, strand]]}}
    exons = [{'index': i+1, 'start': a, 'end': b} for i, (a, b) in enumerate(
        [(101, 130), (201, 230), (301, 330)] if strand == '+' else [(301, 330), (201, 230), (101, 130)])]
    catalog = {'TX': {'transcript_id': 'TX.4', 'chrom': '8', 'strand': strand, 'exons_tx': exons,
                     'metadata': {'reference_source': 'local_gtf', 'reference_annotation': 'neutral.gtf'}}}
    return record, catalog


@pytest.mark.parametrize('strand', ['+', '-'])
def test_inclusive_operational_window_uses_verified_anchor_and_strand(strand):
    record, catalog = fixture(strand)
    proof = explicit_window_geometry(record, transcript_feature_geometry(record, catalog))
    assert proof and proof['relative_start_nt'] == 1 and proof['relative_end_nt'] == 60
    start, end = proof['genomic_window_bed0_half_open'][1:3]
    assert end-start == 60
    record['resolved_target']['evidence_capabilities'] = ['explicit_window_geometry']
    finding = {'record_index': 0, 'status': 'satisfied', 'requirement_kind': 'explicit_window_definition'}
    assert not pipeline_tools._resolved_target_finding_capability_errors([finding], [record], positive_status='satisfied')
    for requirement in ['positive_mapping', 'spatial_exclusion', 'isoform_distinction', 'reference_identity_lock']:
        finding['requirement_kind'] = requirement
        assert pipeline_tools._resolved_target_finding_capability_errors([finding], [record], positive_status='satisfied')


@pytest.mark.parametrize('defect', ['position', 'version', 'offset', 'window', 'strand', 'missing_offset'])
def test_geometry_never_repairs_a_conflicting_window_or_unresolved_anchor(defect):
    record, catalog = fixture('+')
    if defect == 'position': record['resolved_target']['anchor_position'] += 1
    if defect == 'version': record['requested_target']['reference_transcript_id'] = 'TX.3'
    if defect == 'offset': record['requested_target']['relative_start_nt'] = 2
    if defect == 'window': record['resolved_target']['materialized_rows'][0][1] += 1
    if defect == 'strand': record['resolved_target']['materialized_rows'][0][5] = '-'
    if defect == 'missing_offset': record['requested_target'].pop('relative_start_nt')
    assert explicit_window_geometry(record, transcript_feature_geometry(record, catalog)) is None
