from copy import deepcopy
import json

import pytest
import run_goal_intake as runtime
from aso_agents.rna_substrate import transcript_loci, candidate_rna_support, filter_target_rna_candidates
from aso_agents.binding_region_reports import build_binding_region_report, binding_region_markdown


def catalog(strand='+'):
    return {'neutral': {'transcript_id': 'TX.4', 'chrom': 'chr8', 'strand': strand,
            'exons_tx': [{'start': 101, 'end': 120}, {'start': 201, 'end': 230}],
            'metadata': {'reference_source': 'local_gtf', 'reference_annotation': 'neutral.gtf'}}}


@pytest.mark.parametrize('strand', ['+', '-'])
def test_intronic_and_boundary_candidates_have_rna_substrate(strand):
    loci = transcript_loci(catalog(strand))
    for start, end in [(140, 160), (110, 130), (100, 120), (210, 230)]:
        row = {'Chrom': '8', 'Start': start, 'End': end, 'Strand': strand}
        assert candidate_rna_support(row, loci, runtime._tile_row_overlap_intervals)['status'] == 'supported_transcript_locus'


@pytest.mark.parametrize('start,end,strand', [(80, 100, '+'), (90, 110, '+'), (220, 240, '+'), (140, 160, '-')])
def test_a_coordinate_or_partial_overlap_does_not_establish_target_rna(start, end, strand):
    row = {'Chrom': '8', 'Start': start, 'End': end, 'Strand': strand}
    assert candidate_rna_support(row, transcript_loci(catalog()), runtime._tile_row_overlap_intervals)['status'] == 'outside_recorded_target_rna_loci'


def test_discontinuous_candidate_requires_all_blocks_in_one_target_locus():
    row = {'Strand': '+', 'Genomic_segments': json.dumps([
        {'chrom': 'chr8', 'start': 101, 'end': 110}, {'chrom': 'chr8', 'start': 221, 'end': 230}])}
    assert candidate_rna_support(row, transcript_loci(catalog()), runtime._tile_row_overlap_intervals)['status'] == 'supported_transcript_locus'
    row['Genomic_segments'] = json.dumps([
        {'chrom': 'chr8', 'start': 101, 'end': 110}, {'chrom': 'chr9', 'start': 221, 'end': 230}])
    assert candidate_rna_support(row, transcript_loci(catalog()), runtime._tile_row_overlap_intervals)['status'] == 'outside_recorded_target_rna_loci'


def test_missing_reference_or_strand_is_unknown_not_a_negative_result():
    row = {'Chrom': '8', 'Start': 140, 'End': 160, 'Strand': '+'}
    assert candidate_rna_support(row, [], runtime._tile_row_overlap_intervals)['status'] == 'unverified'
    row.pop('Strand')
    assert candidate_rna_support(row, transcript_loci(catalog()), runtime._tile_row_overlap_intervals)['status'] == 'unverified'


def test_excluded_rows_are_preserved_and_site_report_does_not_recommend_them(tmp_path):
    rows = [{'name': name, 'Chrom': 'chr8', 'Start': start, 'End': start + 20, 'Strand': '+',
             'Sequence': 'ACGTTACGATCGACTAGCAT', 'Synthesis': 'ATGCTAGTCGATCGTAACGT'} for name, start in [('outside', 50), ('intronic', 140)]]
    source = tmp_path/'rows.json'; source.write_text(json.dumps({'rows': rows})); original = source.read_bytes()
    result = filter_target_rna_candidates(source, catalog(), runtime._tile_row_overlap_intervals)
    assert source.read_bytes() == original
    assert result['eligible_count'] == result['excluded_count'] == 1
    assert json.loads(open(result['excluded_path']).read())['rows'][0]['name'] == 'outside'
    goal = {'materialized_region_constraints': {'global': [
        {'label': name, 'materialized_rows': [['chr8', start, start+20, name, 0, '+']],
         'aso_overlap_requirement': 'full_containment'} for name, start in [('outside', 50), ('intronic', 140)]]}}
    # Binding-only jobs finish with uniqueness scoring and no splice-score file.
    final = {'tiling_results': {'uniqueness_scored_path': result['eligible_path'], 'target_rna_evidence_filtering': result}}
    report = build_binding_region_report(goal, final, tmp_path, runtime._tile_row_overlap_intervals, {})
    assert [group['candidate_count'] for group in report['sets']] == [0, 1]
    assert report['sets'][0]['status'] == 'no_supported_target_rna_candidates'
    assert report['sets'][0]['excluded_outside_target_rna_count'] == 1
    assert 'excluded for lack of target-RNA' in binding_region_markdown(report)
