"""Fresh processes use the documented model and retain explicit overrides."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agents import Agent, Runner
from aso_agents import llm_provider
from aso_agents.model_defaults import DEFAULT_MODEL, DEFAULT_SMALL_MODEL


FLASH = "deepseek/deepseek-v4.1-flash-1017"
ROOT = Path(__file__).resolve().parents[1]


def clean_env():
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('OPENROUTER_') or key == 'ASO_LLM_PROVIDER' or (
                key.startswith('ASO_') and key.endswith('_MODEL')):
            env.pop(key)
    env['OPENROUTER_API_KEY'] = 'test-only'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


@pytest.mark.parametrize('role', [DEFAULT_MODEL, DEFAULT_SMALL_MODEL])
@pytest.mark.parametrize('settings', [{}, {'ASO_LLM_PROVIDER': 'openrouter'},
                                     {'OPENROUTER_MODEL': ' '}])
def test_default_roles_reach_transport_as_exact_flash_snapshot(role, settings):
    config = llm_provider.get_provider_config(settings, require_credentials=False)
    payload, metadata = llm_provider.prepare_responses_payload(
        {'model': role, 'input': 'Neutral request'}, config)
    assert config.name == 'openrouter'
    assert payload['model'] == FLASH
    assert metadata['resolved_model'] == FLASH


@pytest.mark.parametrize(('env', 'requested', 'expected'), [
    ({}, 'gpt-5.2-2025-12-11', 'openai/gpt-5.2'),
    ({}, 'openai/gpt-5-mini', 'openai/gpt-5-mini'),
    ({}, 'anthropic/claude-sonnet-4', 'anthropic/claude-sonnet-4'),
    ({'OPENROUTER_MODEL': 'qwen/qwen3.8-27b'}, DEFAULT_MODEL, 'qwen/qwen3.8-27b'),
    ({'OPENROUTER_MODEL': 'qwen/qwen3.8-27b'}, 'openai/gpt-5.2', 'qwen/qwen3.8-27b'),
    ({'ASO_LLM_PROVIDER': 'openai'}, DEFAULT_MODEL, 'gpt-5.2-2025-12-11'),
    ({'ASO_LLM_PROVIDER': 'openai'}, DEFAULT_SMALL_MODEL, 'gpt-5-mini-2025-08-07'),
])
def test_explicit_models_and_provider_keep_their_meaning(env, requested, expected):
    config = llm_provider.get_provider_config(env, require_credentials=False)
    assert config.resolve_model(requested) == expected


def test_every_builtin_agent_uses_flash_in_a_fresh_process():
    script = '''
import ast, importlib, json
from pathlib import Path
from agents import Agent
from aso_agents.llm_provider import get_provider_config
config = get_provider_config()
agents = {'Agent default': config.resolve_model(Agent(name='neutral', instructions='').model)}
paths = sorted(Path('aso_agents').glob('*.py')) + [Path('run_goal_intake.py')]
for path in paths:
    tree = ast.parse(path.read_text())
    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Agent'
               for n in ast.walk(tree)):
        continue
    module = importlib.import_module('aso_agents.' + path.stem if path.parent.name == 'aso_agents' else path.stem)
    for name, value in vars(module).items():
        if isinstance(value, Agent):
            agents[module.__name__ + '.' + name] = config.resolve_model(value.model)
print(json.dumps(agents))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                            env=clean_env(), capture_output=True, text=True,
                            check=True, timeout=60)
    models = json.loads(result.stdout)
    assert len(models) > 40
    assert set(models.values()) == {FLASH}, models
    assert 'aso_agents.pipeline_controller_agent.pipeline_controller_agent' in models
    assert 'aso_agents.request_workbench_agent.request_workbench_agent' in models
    assert 'aso_agents.chemistry_agent.chemistry_agent' in models
    assert 'aso_agents.report_explainer_agent.report_explainer_agent' in models


@pytest.mark.parametrize(('role_model', 'global_model', 'expected'), [
    ('gpt-5.2-2025-12-11', None, 'openai/gpt-5.2'),
    ('openai/gpt-5-mini', None, 'openai/gpt-5-mini'),
    ('openai/gpt-5-mini', FLASH, FLASH),
])
def test_intake_override_is_preserved_in_a_fresh_process(role_model, global_model, expected):
    env = clean_env()
    env['ASO_REQUEST_WORKBENCH_MODEL'] = role_model
    if global_model:
        env['OPENROUTER_MODEL'] = global_model
    script = '''
from aso_agents.llm_provider import get_provider_config
from aso_agents.request_workbench_agent import request_workbench_agent
print(get_provider_config().resolve_model(request_workbench_agent.model))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                            env=env, capture_output=True, text=True, check=True, timeout=30)
    assert result.stdout.strip() == expected


def test_runner_defaults_send_flash_and_record_actual_model(monkeypatch):
    monkeypatch.delenv('ASO_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('OPENROUTER_MODEL', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only')
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['base_url'] == 'https://openrouter.ai/api/v1'
            self.responses = self
        def create(self, **payload):
            calls.append(payload)
            return {'id': 'resp-neutral', 'model': FLASH, 'status': 'completed',
                    'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Ready'}]}]}
    llm_provider.reset_provider_runtime_for_tests()
    monkeypatch.setattr(llm_provider, 'OpenAI', Client)
    try:
        result = Runner._run_sync(Agent(name='neutral', instructions='Return Ready.'), 'Neutral request')
        assert result.final_output == 'Ready'
        assert calls[0]['model'] == FLASH
        assert result.responses[0]['response']['resolved_model'] == FLASH
    finally:
        llm_provider.reset_provider_runtime_for_tests()


def test_opt_in_diagnostic_accepts_default_provider():
    from scripts.check_llm_provider import openrouter_smoke_ready
    env = {'OPENROUTER_API_KEY': 'fixture-credential', 'ASO_RUN_OPENROUTER_SMOKE': '1'}
    assert openrouter_smoke_ready(env)
    assert not openrouter_smoke_ready({'OPENROUTER_API_KEY': 'fixture-credential'})
    assert not openrouter_smoke_ready({**env, 'ASO_LLM_PROVIDER': 'openai'})
