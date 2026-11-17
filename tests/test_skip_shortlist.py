"""Skip only recommendation models, with durable execution policy and evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aso_agents import agentic_runner, pipeline_tools as pipeline, shortlist_review
from test_generated_export_reuse import run  # Native export fixture.
from test_shortlist_review import pool, selector


def forbid_review(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Skipped shortlist must not prepare or call recommendation models')
    for name in ('evidence_catalog', '_model_call', '_page_large_evidence', '_rank', '_explain'):
        monkeypatch.setattr(shortlist_review, name, unexpected)
    monkeypatch.setattr('aso_agents.llm_provider.get_provider_config', unexpected)


@pytest.mark.parametrize('allow_calls', [False, True])
@pytest.mark.parametrize('retry', [False, True])
def test_skip_precedes_review_configuration_and_receipts(tmp_path, monkeypatch, allow_calls, retry):
    rows = pool()
    rows[-1]['candidate_assessment']['selection_eligible'] = False
    rows[0]['recommendation_review'] = {'why_chosen': 'An earlier model explanation'}
    before = deepcopy(rows)
    receipt = tmp_path / 'recommendation_review.json'
    receipt.write_text('{"status":"completed","original_evidence":true}')
    prior = receipt.read_bytes()
    forbid_review(monkeypatch)
    selected, constraints, review = shortlist_review.recommend_panel(
        rows, {'candidate_limit': 3}, tmp_path, select_panel=selector(tmp_path),
        skip_shortlist=True, allow_model_calls=allow_calls, retry_failed=retry)
    assert len(selected) == 3
    assert [r['selection_rank'] for r in selected] == [1, 2, 3]
    assert all('recommendation_review' not in row for row in selected)
    assert review['status'] == 'skipped' and review['reviewed_molecule_count'] == 0
    assert review['eligible_molecule_count'] == 6
    assert 'provisional' in review['summary']
    assert json.loads(Path(review['receipt_path']).read_text())['reason'] == 'skip_shortlist'
    assert receipt.read_bytes() == prior and rows == before


def test_skip_with_empty_pool_remains_explicit(tmp_path, monkeypatch):
    forbid_review(monkeypatch)
    selected, _, review = shortlist_review.recommend_panel(
        [], {}, tmp_path, select_panel=selector(tmp_path), skip_shortlist=True, allow_model_calls=True)
    assert selected == [] and review['status'] == 'skipped'
    assert review['eligible_molecule_count'] == 0
    assert 'No ASOs met' in review['summary']


def test_skipped_native_exports_keep_full_pool_and_reopen_offline(run, monkeypatch):
    root, context, source, source_rows, _ = run
    context['final_payload']['execution_options'] = {'skip_shortlist': True}
    before = source.read_bytes()
    # Other scientific analysis remains enabled; only shortlist entry points
    # are forbidden here. Provider config is also used by unrelated analyses.
    for name in ('evidence_catalog', '_model_call', '_page_large_evidence', '_rank', '_explain'):
        monkeypatch.setattr(shortlist_review, name, lambda *a, **kw: pytest.fail('Shortlist called'))
    paths = pipeline._emit_v2_generated_design_artifacts(str(root), review_recommendations=True)
    read = lambda key: json.loads(Path(paths[key]).read_text())
    generated = read('generated_aso_set_json')
    assert generated['recommendation_review']['status'] == 'skipped'
    assert generated['row_count'] == 2  # Explicit requested count is preserved.
    assert read('candidate_assessments_json')['pool_row_count'] == len(source_rows) == 4
    assert read('recommended_shortlist_json')['review']['status'] == 'skipped'
    assert 'review was skipped' in Path(paths['recommended_shortlist_md']).read_text()
    assert Path(paths['recommendation_review_json']).name == 'recommendation_review_skipped.json'
    assert source.read_bytes() == before
    again = pipeline._emit_v2_generated_design_artifacts(str(root))
    assert json.loads(Path(again['generated_aso_set_json']).read_text())['rows'] == generated['rows']
    assert json.loads(Path(again['recommended_shortlist_json']).read_text())['review']['status'] == 'skipped'


@pytest.mark.parametrize('tool', ['tool_build_report_payload', 'tool_write_markdown_report'])
@pytest.mark.parametrize('skip', [False, True])
def test_report_stages_persist_policy_before_export(tmp_path, monkeypatch, tool, skip):
    state = {'run_dir': str(tmp_path), 'flags': {'skip_shortlist': skip}}
    final = {}
    context = {'payload': {}, 'structured_goal': {}, 'final_payload': final}
    saved = []
    monkeypatch.setattr(pipeline, '_prepare_execution_stage', lambda **kw: {
        'status': 'ok', 'state': state, 'run_dir': str(tmp_path)})
    monkeypatch.setattr(pipeline, '_load_goal_context_from_run_dir', lambda _: context)
    monkeypatch.setattr(pipeline, '_prepare_exon_identity_for_execution', lambda *a: None)
    monkeypatch.setattr(pipeline, '_save_goal_payload', lambda *a: saved.append(deepcopy(a[-1])))
    class ReachedExport(Exception):
        pass
    def export(*args, **kwargs):
        assert bool((final.get('execution_options') or {}).get('skip_shortlist')) == skip
        if skip:
            assert saved[-1]['execution_options']['skip_shortlist'] is True
        raise ReachedExport
    monkeypatch.setattr(pipeline, '_emit_v2_generated_design_artifacts', export)
    with pytest.raises(ReachedExport):
        getattr(pipeline, tool)(str(tmp_path / 'state.json'))


def test_new_skip_run_is_initialized_before_controller_and_survives_resume(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    root.mkdir()
    monkeypatch.setattr(pipeline, '_make_managed_run_dir', lambda _: str(root))
    monkeypatch.setattr(agentic_runner.Runner, '_run_sync', lambda *a, **kw: pytest.fail('Controller called'))
    states = []
    def planned(path, **kwargs):
        state = json.loads(path.read_text())
        assert state['flags']['skip_shortlist'] is True
        states.append(str(path))
        return {'status': 'needs_user_input', 'state_path': str(path)}
    monkeypatch.setattr(agentic_runner, '_run_planned_next_action_once', planned)
    first = agentic_runner.run_agentic_pipeline(user_request='Design ASOs', skip_shortlist=True)
    agentic_runner.run_agentic_pipeline(user_request='Design ASOs', existing_state_path=first['state_path'])
    assert len(states) == 2 and states[0] == states[1]


def test_resume_can_enable_skip_before_report_and_failed_save_stops(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    original = {'run_dir': str(tmp_path), 'flags': {}}
    path.write_text(json.dumps(original))
    seen = []
    monkeypatch.setattr(agentic_runner, '_run_planned_next_action_once',
                        lambda p, **kw: seen.append(json.loads(p.read_text())) or {'status': 'in_progress'})
    agentic_runner.run_agentic_pipeline(user_request='Design', existing_state_path=str(path), skip_shortlist=True)
    assert seen[0]['flags']['skip_shortlist'] is True
    path.write_text(json.dumps(original))
    monkeypatch.setattr(agentic_runner, '_mutate_runner_state', lambda *a: None)
    result = agentic_runner.run_agentic_pipeline(user_request='Design', existing_state_path=str(path), skip_shortlist=True)
    assert result['error']['kind'] == 'skip_shortlist_not_recorded' and len(seen) == 1


def test_child_runner_receives_saved_skip_policy():
    calls = []
    def runner(**kwargs):
        calls.append(kwargs)
        return {'status': 'needs_user_input'}
    pipeline._run_agentic_pipeline_to_terminal(
        run_agentic_pipeline=runner, user_request='Child design', existing_state_path=None,
        clarification_answers=None, clarification_answer_summary='',
        flags={'skip_shortlist': True}, max_steps=1)
    assert calls[0]['skip_shortlist'] is True


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('tty', [False, True])
def test_public_cli_forwards_flag_on_both_output_paths(tmp_path, monkeypatch, skip, tty):
    from aso_agents import cli
    from test_cli_progress import Terminal
    calls = []
    monkeypatch.setattr('sys.stdout', Terminal(tty=tty))
    def runner(**kwargs):
        calls.append(kwargs)
        return {'status': 'run_complete', 'run_dir': str(tmp_path)}
    monkeypatch.setattr(agentic_runner, 'run_agentic_pipeline', runner)
    args = ['--output-dir', str(tmp_path), *(['--skip-shortlist'] if skip else []), 'Design ASOs']
    assert cli.main(args) == 0
    assert calls[0]['skip_shortlist'] is skip
    assert calls[0]['user_request'] == 'Design ASOs'
