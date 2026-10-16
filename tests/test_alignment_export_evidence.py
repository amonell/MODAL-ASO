from copy import deepcopy
import hashlib
import json
from pathlib import Path

from aso_agents import pipeline_tools as pipeline
from aso_agents.reference_report_projection import evaluation_projection


def sample():
    near = [{
        'transcript_id': 'TX.8', 'mapping_scope': 'spliced_cdna', 'rna_context': 'mature_spliced_cdna',
        'match_orientation': 'reverse_complement', 'strand': '-', 'coordinate_system': '1based_inclusive',
        'cdna_start': 21 + index, 'cdna_end': 24 + index, 'matched_window_sequence': 'ACGT',
        'genomic_segments': [{'chrom': '8', 'start': 90, 'end': 91}, {'chrom': '8', 'start': 70, 'end': 71}],
        'mismatch_count': 1 if index == 0 else 2,
        'mismatch_positions_in_supplied_sequence_1based': [3] if index == 0 else [2, 3],
        'mismatch_positions_in_target_sequence_1based': [2] if index == 0 else [2, 3],
        'mismatch_details_in_target_orientation': (
            [{'target_position_1based': 2, 'reference_base': 'C', 'aligned_query_base': 'A'}]
            + ([{'target_position_1based': 3, 'reference_base': 'G', 'aligned_query_base': 'A'}] if index else [])),
        'search_scope': 'recorded search only; capped, not exhaustive',
        'reference_identity': {'sequence_evidence': {'large_reference': 'A' * 10000}},
    } for index in range(7)]
    return {'provided_asos': [{'name': 'sample', 'input_row_id': 'source-row-7',
        'source_file': 'panel.csv', 'source_row': 7, 'sequence': 'ACGT',
        'mapping_hits': [], 'near_match_candidates': near}]}


def test_projection_retains_nonbest_alignments_and_oriented_evidence():
    source = sample(); before = deepcopy(source)
    row = evaluation_projection(source)[0]
    group = row['reference_alignment_groups'][0]
    assert group['near_hit_count'] == 7
    assert len(group['best_near_alignments']) == 1
    assert len(group['near_alignments']) == 7
    assert row['input_row_id'] == 'source-row-7' and row['source_row'] == 7
    for index, observed in enumerate(group['near_alignments']):
        original = source['provided_asos'][0]['near_match_candidates'][index]
        assert observed['source_pointer'] == f'/near_match_candidates/{index}'
        for field in ['strand', 'coordinate_system', 'matched_window_sequence', 'genomic_segments',
                      'mismatch_positions_in_target_sequence_1based', 'mismatch_details_in_target_orientation', 'search_scope']:
            assert observed[field] == original[field]
        assert 'reference_identity' not in observed
    assert source == before


def test_export_preview_is_bounded_but_complete_projection_keeps_all_records(monkeypatch):
    source = sample()
    monkeypatch.setattr(pipeline, '_collect_completed_task_outputs', lambda _: ({}, [{
        'task_type': 'evaluate_existing_aso_set', 'task_id': 'eval', 'payload': source}]))
    evidence = pipeline._bounded_completed_evaluation_evidence({})[0]
    row = evidence['rows'][0]
    assert row['input_row_id'] == 'source-row-7'
    assert len(row['near_match_candidates']) == 3
    assert row['near_match_candidates_total'] == 7 and row['near_match_candidates_preview_truncated']
    assert row['mapping_hits_total'] == 0 and not row['mapping_hits_preview_truncated']
    assert row['near_match_candidates'][0]['mismatch_details_in_target_orientation'] == source['provided_asos'][0]['near_match_candidates'][0]['mismatch_details_in_target_orientation']
    assert len(evidence['reference_projection'][0]['reference_alignment_groups'][0]['near_alignments']) == 7
    assert 'exhaustive' in evidence['reference_projection_scope']


def test_complete_projection_file_preserves_all_records_outside_prompt(tmp_path, monkeypatch):
    source = sample()
    source['provided_asos'][0]['near_match_candidates'] *= 100
    before = deepcopy(source)
    monkeypatch.setattr(pipeline, '_collect_completed_task_outputs', lambda _: ({}, [{
        'task_type': 'evaluate_existing_aso_set', 'task_id': 'eval', 'payload': source}]))
    state = {'task_artifacts': {'eval': {'response_json': '/source/completed.json'}}}
    entry = pipeline._bounded_completed_evaluation_evidence(state, evidence_directory=tmp_path)[0]
    assert 'reference_projection' not in entry
    record = entry['reference_projection_file']; path = Path(record['path'])
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    assert json.loads(path.read_text()) == evaluation_projection(source)
    assert len(json.loads(path.read_text())[0]['reference_alignment_groups'][0]['near_alignments']) == 700
    assert len(json.dumps(entry)) < path.stat().st_size / 10
    assert entry['source_json_path'] == '/source/completed.json'
    assert len(entry['rows'][0]['near_match_candidates']) == 3
    assert entry['rows'][0]['near_match_candidates_total'] == 700
    assert source == before
    again = pipeline._bounded_completed_evaluation_evidence(state, evidence_directory=tmp_path)[0]
    assert again['reference_projection_file'] == record


def test_changed_evidence_file_is_not_silently_reused(tmp_path):
    from aso_agents.output_evidence import store_reference_projection
    import pytest
    projection = evaluation_projection(sample())
    record = store_reference_projection(projection, tmp_path)
    Path(record['path']).write_text('[]')
    with pytest.raises(ValueError, match='evidence changed'):
        store_reference_projection(projection, tmp_path)
