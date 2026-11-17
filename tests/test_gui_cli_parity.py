"""Compare actual CLI and GUI boundaries using deterministic model responses."""
import copy
import os
import sys
from queue import Queue
from types import SimpleNamespace as NS
import pytest

from aso_agents.gui.config import Settings, SetupError, environment_for


def test_default_means_inherit_and_explicit_choices_use_only_cli_variables():
    base = {"OPENROUTER_MODEL": "vendor/inherited", "OPENROUTER_REASONING_EFFORT": "high",
            "OPENROUTER_PROVIDER_SORT": "price", "OPENROUTER_ZDR": "true",
            "OPENROUTER_TIMEOUT_SECONDS": "90", "OPENROUTER_MAX_RETRIES": "3"}
    inherited = environment_for(Settings("openrouter", "key"), base)
    assert all(inherited[key] == value for key, value in base.items())
    explicit = environment_for(Settings("openrouter", "key", "vendor/selected", "low", 120, 0), base)
    assert explicit == {**base, "ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "key",
                        "OPENROUTER_MODEL": "vendor/selected", "OPENROUTER_REASONING_EFFORT": "low",
                        "OPENROUTER_TIMEOUT_SECONDS": "120", "OPENROUTER_MAX_RETRIES": "0"}


def test_openai_global_overrides_are_rejected_instead_of_rewriting_requests():
    with pytest.raises(SetupError, match="existing CLI per-agent settings"):
        environment_for(Settings("openai", "key", "gpt-5.2", "high"), {})


@pytest.mark.parametrize("case", ["openai", "router_defaults", "router_inherited", "router_explicit"])
def test_cli_and_gui_send_identical_requests_and_preserve_sdk_behavior(monkeypatch, case):
    from agents import Agent, ModelSettings, Runner, function_tool
    from openai._models import BaseModel as SDKModel
    from aso_agents import cli, llm_provider
    from aso_agents.gui.worker import run

    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith(("OPENAI_", "OPENROUTER_", "ASO_"))}
    inherited["OPENROUTER_PROVIDER_SORT"] = "price"
    inherited["OPENROUTER_STRUCTURED_OUTPUT_MODE"] = "json_schema"
    if case == "openai":
        selected = Settings("openai", "fake-key")
        inherited.update(ASO_REQUEST_WORKBENCH_MODEL="gpt-5-mini-2025-08-07",
                         ASO_REQUEST_WORKBENCH_REASONING_EFFORT="minimal",
                         OPENAI_BASE_URL="https://example.invalid/v1")
    elif case == "router_explicit":
        selected = Settings("openrouter", "fake-key", "vendor/selected", "low", 120, 0)
    else:
        selected = Settings("openrouter", "fake-key")
        if case == "router_inherited":
            inherited.update(OPENROUTER_MODEL="vendor/inherited", OPENROUTER_REASONING_EFFORT="high")

    # Build the CLI's exports independently of the GUI mapping function.
    cli_env = {**inherited, "ASO_LLM_PROVIDER": selected.provider,
               ("OPENAI_API_KEY" if selected.provider == "openai" else "OPENROUTER_API_KEY"): "fake-key"}
    if case == "router_explicit":
        cli_env.update(OPENROUTER_MODEL="vendor/selected", OPENROUTER_REASONING_EFFORT="low",
                       OPENROUTER_TIMEOUT_SECONDS="120", OPENROUTER_MAX_RETRIES="0")

    sdk_methods = (SDKModel.model_dump, SDKModel.model_dump_json)
    provider_method = llm_provider.prepare_responses_payload
    captured, executions = [], []

    @function_tool
    def measure(value: int):
        executions.append(value)
        return {"value": value}

    responses = [
        {"id": "resp_1", "model": "deterministic-model", "status": "completed", "output": [
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "measure", "arguments": '{"value":2}'}]},
        {"id": "resp_2", "model": "deterministic-model", "status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": '{"ok":true}'}]}]},
    ]

    class Client:
        def __init__(self, config):
            self.responses = self
            self.config = config

        def create(self, **payload):
            captured.append((copy.deepcopy(payload), self.config))
            return copy.deepcopy(responses[(len(captured) - 1) % 2])

    monkeypatch.setattr(llm_provider, "_get_client", Client)

    def controller(*, argv, clarification_resolver=None, result_callback=None):
        assert argv == ["--", "Same ASO request"]
        assert llm_provider.prepare_responses_payload is provider_method
        assert (SDKModel.model_dump, SDKModel.model_dump_json) == sdk_methods
        agent = Agent(name="Parity agent", instructions="Use measure, then return a result.", tools=[measure],
            model=os.environ.get("ASO_REQUEST_WORKBENCH_MODEL", "gpt-5.2-2025-12-11"),
            model_settings=ModelSettings(
                reasoning={"effort": os.environ.get("ASO_REQUEST_WORKBENCH_REASONING_EFFORT", "max"), "summary": "auto"},
                extra_body={"text": {"format": {"type": "json_schema", "name": "result", "strict": True,
                    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                               "required": ["ok"], "additionalProperties": False}}}}))
        outcome = Runner._run_sync(agent, "Same ASO request", max_steps=2)
        assert outcome.final_output == '{"ok":true}'
        result_callback({"status": "run_complete"})

    monkeypatch.setitem(sys.modules, "run_goal_intake", NS(main=controller))
    monkeypatch.setattr(os, "environ", cli_env)
    assert cli._run(["--", "Same ASO request"]) == 0
    cli_calls = captured[:]
    captured.clear()
    monkeypatch.setattr(os, "environ", dict(inherited))
    events = Queue()
    run(selected, "Same ASO request", events, Queue())
    gui_calls = captured[:]
    assert len(cli_calls) == len(gui_calls) == 2
    assert gui_calls == cli_calls  # full prompts, schemas, tools, reasoning and history
    assert executions == [2, 2]
    assert (SDKModel.model_dump, SDKModel.model_dump_json) == sdk_methods
    assert llm_provider.prepare_responses_payload is provider_method
