"""Public CLI behavior for completion, pausing and execution errors."""
import pytest
import json
import os

from aso_agents import cli


@pytest.mark.parametrize(('status', 'exit_status'), [
    ('run_complete', 0), ('completed', 0), ('needs_user_input', 0),
    ('failed', 1), ('fatal_error', 1), ('model_failure', 1),
])
def test_cli_exit_status_describes_the_native_outcome(monkeypatch, status, exit_status):
    import run_goal_intake
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    monkeypatch.setenv('ASO_LLM_PROVIDER', 'openai')
    def finish(*, argv, result_callback):
        assert argv == ['A neutral request']
        result_callback({'status': status})
    monkeypatch.setattr(run_goal_intake, 'main', finish)
    assert cli.main(['A neutral request']) == exit_status


@pytest.mark.parametrize('override', [None, 'qwen/qwen3.8-27b'])
def test_startup_names_resolved_default_model(monkeypatch, capsys, override):
    import run_goal_intake
    monkeypatch.delenv('ASO_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('OPENROUTER_MODEL', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only')
    if override:
        monkeypatch.setenv('OPENROUTER_MODEL', override)
    monkeypatch.setattr(run_goal_intake, 'main', lambda **kwargs: None)
    assert cli.main(['A neutral request']) == 0
    expected = override or 'deepseek/deepseek-v4.1-flash-1017'
    assert f'Default language model: {expected} (openrouter).' in capsys.readouterr().out


def test_startup_names_separate_intake_override(monkeypatch, capsys):
    import run_goal_intake
    import importlib
    workbench = importlib.import_module('aso_agents.request_workbench_agent')
    monkeypatch.delenv('ASO_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('OPENROUTER_MODEL', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only')
    monkeypatch.setattr(workbench, 'REQUEST_WORKBENCH_MODEL', 'openai/gpt-5.2')
    monkeypatch.setattr(run_goal_intake, 'main', lambda **kwargs: None)
    assert cli.main(['A neutral request']) == 0
    output = capsys.readouterr().out
    assert 'deepseek/deepseek-v4.1-flash-1017' in output
    assert 'Intake language model: openai/gpt-5.2.' in output


@pytest.mark.parametrize('explicit_output', [False, True])
def test_resume_preserves_custom_output_root_across_shells(monkeypatch, tmp_path, explicit_output):
    import run_goal_intake
    from aso_agents import pipeline_tools
    run = tmp_path / 'chosen results' / 'goal_neutral'
    run.mkdir(parents=True)
    state = run / 'run_state.json'
    state.write_text(json.dumps({'run_dir': str(run)}))
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    monkeypatch.setenv('ASO_LLM_PROVIDER', 'openai')
    monkeypatch.setenv('ASO_RUNS_DIR', str(tmp_path / 'other shell'))
    output = run.parent
    def resume(*, argv, result_callback):
        assert argv == ['--agentic-state-path', str(state)]
        assert os.environ['ASO_RUNS_DIR'] == str(output)
        if not explicit_output:
            child = pipeline_tools._child_task_run_dir(str(run), 'neutral')
            accepted, rejected = pipeline_tools._trusted_run_dir_override(child)
            assert accepted == child and not rejected
        result_callback({'status': 'needs_user_input'})
    monkeypatch.setattr(run_goal_intake, 'main', resume)
    arguments = ['--agentic-state-path', str(state)]
    if explicit_output:
        arguments = ['--output-dir', str(output)] + arguments
    assert cli.main(arguments) == 0


def test_resume_rejects_a_different_output_root(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv('ASO_RUNS_DIR', str(tmp_path / 'initial root'))
    state = tmp_path / 'run_state.json'
    state.write_text(json.dumps({'run_dir': str(tmp_path / 'original' / 'goal_neutral')}))
    assert cli.main(['--output-dir', str(tmp_path / 'different'),
                     '--agentic-state-path', str(state)]) == 1
    assert 'Omit --output-dir when resuming' in capsys.readouterr().err
