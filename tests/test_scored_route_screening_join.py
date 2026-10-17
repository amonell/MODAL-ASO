import csv
import json

from aso_agents.pipeline_tools import _collect_generated_candidate_rows, _normalized_generated_candidate_row


def _write_route(root):
    path = root / 'artifacts' / 'design' / 'strategy_routes' / 'rank01_donor_500.route.csv'
    path.parent.mkdir(parents=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['chrom', 'start_bp', 'end_bp', 'strand', 'sequence'])
        writer.writeheader()
        writer.writerow({'chrom': 'chr1', 'start_bp': 490, 'end_bp': 510,
                         'strand': '+', 'sequence': 'ACGTACGTACGTACGTACGT'})


def _write_scored(root, rows):
    path = root / 'artifacts' / 'candidates' / 'scored_tiles.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'rows': rows}))


def _scored(start=490, exact=2):
    return {'Chrom': 'chr1', 'Start': start, 'End': start + 20, 'Strand': '+',
            'Sequence': 'ACGTACGTACGTACGTACGT', 'Off_target_exact_count': exact,
            'Off_target_1mm_count': 4, 'Off_target_mismatch_budget': 1,
            'off_target_search_status': 'complete'}


def test_route_keeps_screening_measured_for_the_same_genomic_candidate(tmp_path):
    _write_route(tmp_path)
    _write_scored(tmp_path, [_scored()])
    rows = _collect_generated_candidate_rows(str(tmp_path))
    assert len(rows) == 1
    row = _normalized_generated_candidate_row(rows[0])
    assert (row['Off_target_exact_count'], row['Off_target_1mm_count'],
            row['Off_target_mismatch_budget'], row['off_target_search_status']) == (2, 4, 1, 'complete')
    assert row['screening_evidence_source'].endswith('scored_tiles.json')


def test_different_placement_or_conflicting_scored_rows_are_not_joined(tmp_path):
    _write_route(tmp_path)
    _write_scored(tmp_path, [_scored(start=800)])
    assert 'Off_target_exact_count' not in _collect_generated_candidate_rows(str(tmp_path))[0]
    _write_scored(tmp_path, [_scored(exact=0), _scored(exact=2)])
    assert 'Off_target_exact_count' not in _collect_generated_candidate_rows(str(tmp_path))[0]
