from copy import deepcopy
from types import SimpleNamespace

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline
from aso_agents.evaluation_evidence import attach_measurement_references, row_evidence_inventory


def test_exact_binding_in_one_isoform_does_not_skip_alternative_binding_search(tmp_path, monkeypatch):
    catalog = {label: {'transcript_id': tid, 'chrom': 'chr8', 'strand': '+',
               'exons_tx': [{'index': 1, 'start': 101, 'end': 104}, {'index': 2, 'start': 201, 'end': 204}]}
               for label, tid in [('A', 'TX.1'), ('B', 'TY.1')]}
    sequences = {'A': 'GGTTACGT', 'B': 'GGTTACGA'}
    native = SimpleNamespace(_build_isoform_cdna_sequences=lambda *_a, **_k: sequences,
             _map_cdna_range_to_genomic_segments=runtime._map_cdna_range_to_genomic_segments,
             _continuous_genomic_span_from_segments=runtime._continuous_genomic_span_from_segments)
    monkeypatch.setattr(pipeline, '_goal_runtime_module', lambda: native)
    monkeypatch.setattr(pipeline, '_candidate_run_dirs_for_task', lambda *_: [])
    monkeypatch.setattr(pipeline, '_reference_goal_context_for_request', lambda *_: {})
    monkeypatch.setattr(pipeline, '_ensure_eval_transcript_catalog', lambda *_: (catalog, catalog, []))
    monkeypatch.setattr(pipeline, '_choose_eval_transcript', lambda *_: ('TX.1', 'A', catalog['A']))
    monkeypatch.setattr(pipeline, '_score_provided_asos_with_existing_pipeline', lambda **_: {})
    result = pipeline._evaluate_existing_asos_payload(state={'request': 'Evaluate this supplied antisense molecule.'},
        request_envelope={'provided_asos': [{'name': 'physical molecule', 'sequence': 'ACGTAACC',
                                            'sequence_role': 'aso'}]}, run_dir=str(tmp_path), task_id='neutral')
    row = result['provided_asos'][0]
    assert row['mapping_status'] == 'mapped'
    assert row['transcript_id'] == 'TX.1' and row['sequence'] == 'ACGTAACC'
    assert row['near_match_search']['status'] == 'performed'
    assert row['near_match_search']['alternative_transcripts_without_exact_binding'] == ['TY']
    alternative = [hit for hit in row['near_match_candidates'] if hit['transcript_id'] == 'TY.1'
                   and hit['match_orientation'] == 'reverse_complement']
    assert alternative[0]['mismatch_count'] == 1 and len(alternative[0]['genomic_segments']) == 2
    assert alternative[0]['rna_context'] == 'mature_spliced_cdna'


def test_row_inventory_keeps_absence_and_other_rows_evidence_distinct():
    rows = [{'name': 'first', 'input_row_id': 'r1', 'mapping_hits': [], 'near_match_candidates': [],
             'near_match_search': {'status': 'not_run', 'reason': 'scope not searched'}},
            {'name': 'second', 'input_row_id': 'r2', 'mapping_hits': [], 'near_match_candidates': [
                {'transcript_id': 'TY.1', 'mapping_scope': 'spliced_cdna',
                 'match_orientation': 'reverse_complement', 'requested_site_overlaps': [{'position': 200}]}]}]
    before = deepcopy(rows)
    inventory = row_evidence_inventory(rows[0])
    assert rows == before and inventory['retained_near_match_count'] == 0 and not inventory['retained_hypotheses']
    summary = {'provided_asos': rows}
    attach_measurement_references(summary)
    assert rows[0]['evidence_inventory']['retained_near_match_count'] == 0
    assert rows[1]['evidence_inventory']['retained_near_match_count'] == 1
    assert rows[0]['reference_tokens']['evidence_inventory'] != rows[1]['reference_tokens']['evidence_inventory']
