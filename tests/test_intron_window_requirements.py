from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
import run_goal_intake as native
from aso_agents import pipeline_tools as pipeline
from aso_agents.intron_window_requirements import bind_missing_intron_windows


def goal():
    return {'species': 'human', 'gene_symbol': 'NEUTRAL', 'design_programs': [{
        'program_id': 'event', 'modality': 'steric', 'intent': 'splice_modulation',
        'targeting_policy': 'explicit_user_region',
        'splice_event': {'event_type': 'intron_retention', 'intron_number': 1,
                         'reference_transcript_id': 'ENST000TEST'}}]}


def request(width):
    return (f'Use both annotated boundaries. Tile binding windows extending {width} nt on each side of each junction; '
            'require overlap with the corresponding junction window.')


@pytest.mark.parametrize('width', [13, 37])
@pytest.mark.parametrize('strand', ['+', '-'])
def test_user_window_materializes_both_boundaries_on_either_strand(tmp_path, width, strand):
    g = goal()
    spans = [(101, 150), (301, 350), (501, 550)]
    if strand == '-':
        spans.reverse()
    g['isoform_transcripts'] = {'ENST000TEST': {'transcript_id': 'ENST000TEST',
        'chrom': 'chr2', 'strand': strand, 'exons_tx': [
            {'index': i, 'start': start, 'end': end, 'cdna_start': (i-1)*50+1, 'cdna_end': i*50}
            for i, (start, end) in enumerate(spans, 1)]}}
    result = bind_missing_intron_windows(g, request(width))
    assert result['changed'] and result['width_each_side_nt'] == width
    native._normalize_region_constraints_in_goal(g)
    native._materialize_region_constraints(g, {}, str(tmp_path))
    constraints = g['materialized_region_constraints']['by_program']['event']
    assert len(constraints) == 2
    donor, acceptor = (150, 301) if strand == '+' else (501, 350)
    assert [c['materialized_rows'][0][1:3] for c in constraints] == [
        [donor-1-width, donor+width], [acceptor-1-width, acceptor+width]]
    assert all(c['aso_overlap_requirement'] == 'any_overlap' for c in constraints)
    before = deepcopy(g)
    assert not bind_missing_intron_windows(g, request(width))['changed']
    assert g == before


def test_plus_minus_window_and_explicit_containment():
    g = goal()
    result = bind_missing_intron_windows(g, 'Use both boundaries, windows +/-17 nt; full containment.', 'full_containment')
    assert result['changed'] and result['aso_overlap_requirement'] == 'full_containment'


@pytest.mark.parametrize('text', [
    'Use both boundaries; windows +/-13 nt or windows +/-37 nt; any overlap.',
    'Use both boundaries; do not use windows +/-13 nt; any overlap.',
    'Use both boundaries; 13 nt upstream and 37 nt downstream; any overlap.',
    request(13) + ' Each ASO must cross the junction.',
    request(13) + ' Require 5 nt overlap.',
])
def test_unclear_or_different_predicate_is_not_silently_guessed(text):
    g = goal(); before = deepcopy(g)
    result = bind_missing_intron_windows(g, text, 'any_overlap')
    assert result['status'] == 'needs_clarification' and g == before


def test_multiple_programs_cannot_borrow_a_global_window():
    g = goal(); g['design_programs'].append(deepcopy(g['design_programs'][0]))
    assert bind_missing_intron_windows(g, request(13))['status'] == 'needs_clarification'


def test_resolution_stops_for_missing_window_before_other_execution(tmp_path, monkeypatch):
    g = goal(); state_path = tmp_path/'state.json'
    state_path.write_text(json.dumps({'run_dir': str(tmp_path), 'request': 'Use both intron boundaries',
        'status': 'goal_schema_ready', 'phase': 'goal_schema_ready', 'force_rerun_stages': []}))
    context = {'payload': {}, 'structured_goal': g, 'final_payload': {'structured_goal': g}}
    monkeypatch.setattr(pipeline, 'provider_credentials_available', lambda: False)
    monkeypatch.setattr(pipeline, '_prepare_execution_stage', lambda **kw: {'status': 'ok', 'state': pipeline._load_state(str(state_path)), 'run_dir': str(tmp_path)})
    monkeypatch.setattr(pipeline, '_load_goal_context_from_run_dir', lambda _: context)
    monkeypatch.setattr(pipeline, '_save_goal_payload', lambda *args: None)
    monkeypatch.setattr(pipeline, '_goal_runtime_module', lambda: SimpleNamespace())
    result = pipeline.tool_resolve_isoforms_and_sites(str(state_path))
    assert result.get('needs_user_input'), result
    planning = pipeline._derive_next_actions(pipeline._load_state(str(state_path)))
    assert planning['clarification_request']['restart_stage'] == 'intake'
