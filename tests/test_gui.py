"""GUI contracts use deterministic providers; no credentials or live spending."""
from types import SimpleNamespace as NS
import pytest
from aso_agents.gui.config import Settings, SetupError, efforts, environment_for, api_error, validate, list_models


def settings(**changes):
    return Settings(**({"provider": "openrouter", "api_key": "test-key", "model": "openai/gpt-5.2", "effort": "low"} | changes))


class Client:
    def __init__(self, error=None, output=None):
        self.responses = self
        self.error = error
        self.output = output if output is not None else [NS(type="function_call", name="gui_check", arguments='{"ok":true}')]
        self.payloads = []

    def create(self, **payload):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return NS(output=self.output)


@pytest.mark.parametrize("provider,model,effort,expected", [("openai", "default", "default", "gpt-5.2-2025-12-11"), ("openrouter", "openai/gpt-5.2", "low", "openai/gpt-5.2")])
def test_validation_uses_framework_transport(provider, model, effort, expected):
    client = Client()
    validate(settings(provider=provider, model=model, effort=effort), [model], client, base={})
    assert client.payloads[0]["model"] == expected
    if provider == "openrouter":
        assert client.payloads[0]["reasoning"]["effort"] == "low"
    else:
        assert "reasoning" not in client.payloads[0]
    assert client.payloads[0]["tools"][0]["name"] == "gui_check"


@pytest.mark.parametrize("changes,field", [({"provider": "other"}, "Provider"), ({"api_key": " "}, "API key"),
    ({"model": "typo"}, "Model"), ({"effort": "max"}, "Reasoning effort")])
def test_invalid_fields_block_requests(changes, field):
    client = Client()
    with pytest.raises(SetupError) as caught:
        validate(settings(**changes), ["openai/gpt-5.2"], client)
    assert caught.value.field == field
    assert not client.payloads


@pytest.mark.parametrize("status,field", [(401, "API key"), (402, "Billing"), (403, "Access"),
    (404, "Model"), (429, "Rate limit"), (500, "Connection"), (400, "Model compatibility")])
def test_provider_errors_are_focused_and_redacted(status, field):
    exc = RuntimeError("secret-key-must-not-leak")
    exc.status_code = status
    client = Client(error=exc)
    with pytest.raises(SetupError) as caught:
        validate(settings(), ["openai/gpt-5.2"], client)
    assert caught.value.field == field
    assert "secret-key" not in str(caught.value)


def test_reasoning_error_identification():
    exc = RuntimeError()
    exc.status_code = 400
    exc.body = {"error": {"param": "reasoning.effort"}}
    assert api_error(exc).field == "Reasoning effort"


def test_timeout_has_retry_guidance_without_blaming_credentials():
    error = api_error(TimeoutError())
    assert error.field == "Timeout"
    assert "40-second" in str(error)
    assert "No automatic retry" in str(error)
    assert "does not establish" in str(error)


def test_successful_http_response_without_tool_call_is_rejected():
    with pytest.raises(SetupError, match="structured tool call"):
        validate(settings(), ["openai/gpt-5.2"], Client(output=[]))


def test_environment_mapping_preserves_existing_cli_settings():
    base = {"OPENROUTER_PROVIDER_SORT": "price", "OPENROUTER_ZDR": "true", "ASO_GOAL_INTAKE_MODEL": "existing"}
    result = environment_for(settings(), base)
    assert all(result[key] == value for key, value in base.items())
    assert result["OPENROUTER_MODEL"] == "openai/gpt-5.2"
    assert result["OPENROUTER_REASONING_EFFORT"] == "low"
    assert "OPENROUTER_MODEL" not in base


def test_unknown_model_never_gets_guessed_efforts():
    assert efforts("vendor/new-model") == ["default"]
    assert "max" not in efforts("openai/gpt-5.2")


def test_openai_catalog_excludes_unsupported_modalities(monkeypatch):
    monkeypatch.setattr("aso_agents.gui.config.client_for", lambda *args: NS(models=NS(list=lambda: [NS(id=x) for x in
        ["gpt-5.2", "gpt-5-mini-2025-08-07", "gpt-image-1", "gpt-5.2-pro", "whisper-1"]])))
    assert list(list_models("openai", "key")) == ["gpt-5-mini-2025-08-07", "gpt-5.2"]


def test_openrouter_catalog_requires_tools_and_structured_outputs(monkeypatch):
    monkeypatch.setattr("aso_agents.gui.config.client_for", lambda *args: NS())
    import requests
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: NS(status_code=200, json=lambda: {"data": [
        {"id": "vendor/compatible", "supported_parameters": ["tools", "structured_outputs"]},
        {"id": "vendor/text-only", "supported_parameters": []}]}))
    assert list_models("openrouter", "key") == {"vendor/compatible": ["default"]}


def test_openrouter_efforts_follow_live_metadata():
    from aso_agents.gui.config import catalog_efforts
    assert catalog_efforts({"reasoning": {"supported_efforts": ["none", "low", "max"], "mandatory": True}}) == ["default", "low", "max"]
    assert "max" in catalog_efforts({"reasoning": {"supported_efforts": None}})
    assert catalog_efforts({}) == ["default"]


def test_controller_adapter_contract():
    # Detect interface drift without importing or changing the scientific engine.
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).parents[1] / "run_goal_intake.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert {"argv", "clarification_resolver", "result_callback"}.issubset({arg.arg for arg in main.args.kwonlyargs})


def test_gui_missing_extra_is_actionable(monkeypatch):
    from aso_agents.gui import launcher
    monkeypatch.setattr(launcher.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="--extra gui"):
        launcher.launch([])


def test_worker_delegates_to_unchanged_controller_and_isolates_settings(monkeypatch):
    import os
    import sys
    from queue import Queue
    from aso_agents.gui.worker import run
    from aso_agents import llm_provider
    original = llm_provider.prepare_responses_payload
    monkeypatch.setattr(llm_provider, "prepare_responses_payload", original)
    monkeypatch.setattr(os, "environ", dict(os.environ) | {"OPENROUTER_MODEL": "stale/model", "OPENAI_BASE_URL": "https://stale.invalid"})
    events, answers = Queue(), Queue()
    answers.put([{"question": "How many?", "answer": "10"}])

    def controller(*, argv, clarification_resolver, result_callback):
        assert argv == ["--", "Design ASOs"]
        assert os.environ["OPENROUTER_MODEL"] == "openai/gpt-5.2"
        assert os.environ["OPENAI_BASE_URL"] == "https://stale.invalid"
        assert clarification_resolver(questions=["How many?"], context={}) == [{"question": "How many?", "answer": "10"}]
        config = llm_provider.get_provider_config()
        assert llm_provider.prepare_responses_payload is original
        payload, _ = llm_provider.prepare_responses_payload({"model": "gpt-5.2-2025-12-11", "input": "unchanged prompt", "reasoning": {"effort": "max", "summary": "auto"}}, config)
        assert payload["model"] == "openai/gpt-5.2"
        assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
        assert payload["input"] == "unchanged prompt"
        result_callback({"status": "run_complete", "run_dir": "/tmp/example"})

    monkeypatch.setitem(sys.modules, "run_goal_intake", NS(main=controller))
    run(settings(), "Design ASOs", events, answers)
    kinds = []
    while not events.empty():
        kinds.append(events.get_nowait()[0])
    assert [kind for kind in kinds if kind != "activity"] == ["questions", "result", "done"]
    assert kinds.count("activity") == 1


def test_changed_controller_contract_fails_in_gui_adapter(monkeypatch):
    import os
    import sys
    from queue import Queue
    from aso_agents import llm_provider
    from aso_agents.gui.worker import run
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(llm_provider, "prepare_responses_payload", llm_provider.prepare_responses_payload)
    monkeypatch.setitem(sys.modules, "run_goal_intake", NS(main=lambda new_interface: None))
    events = Queue()
    run(settings(), "Request", events, Queue())
    kind, message = events.get_nowait()
    assert kind == "error" and "GUI adapter" in message and "worker.py" in message
    assert events.get_nowait()[0] == "done"


def test_chat_stream_buffers_narration_and_redacts_keys():
    from queue import Queue
    from aso_agents.gui.worker import ChatStream
    events = Queue()
    stream = ChatStream(events, "secret-key")
    stream.write("Choosing ")
    stream.flush()
    assert events.empty()
    stream.write("transcript.\nsecret-")
    stream.write("key\nFinal message")
    stream.finish()
    assert [events.get_nowait() for _ in range(3)] == [
        ("message", "Choosing transcript."), ("message", "[redacted]"), ("message", "Final message")]


def test_machine_result_is_not_shown_as_agent_narration():
    from queue import Queue
    from aso_agents.gui.worker import ChatStream
    events = Queue()
    stream = ChatStream(events, "key")
    stream.write('AGENTIC_RUN_RESULT: {"status":"fatal"}\n')
    assert events.empty()


def test_sdk_warnings_are_displayed_as_diagnostics_without_serialization_changes():
    from queue import Queue
    from aso_agents.gui.worker import ChatStream
    events = Queue()
    stream = ChatStream(events, "secret-key")
    stream.write("PydanticSerializationUnexpectedValue: secret-key\n")
    stream.finish()
    kind, details = events.get_nowait()
    assert kind == "diagnostic"
    assert details["type"] == "SDK serialization warning"
    assert "secret-key" not in str(details)


def test_failed_controller_exit_does_not_invent_connection_error(monkeypatch):
    import os
    import sys
    from queue import Queue
    from aso_agents import llm_provider
    from aso_agents.gui.worker import run
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(llm_provider, "prepare_responses_payload", llm_provider.prepare_responses_payload)
    def controller(*, argv, clarification_resolver, result_callback):
        result_callback({"status": "fatal", "failed_stage": "intake", "error": {"message": "HTTP 400"}})
        raise SystemExit(1)
    monkeypatch.setitem(sys.modules, "run_goal_intake", NS(main=controller))
    events = Queue()
    run(settings(), "Request", events, Queue())
    assert events.get_nowait()[0] == "activity"
    assert events.get_nowait()[0] == "result"
    assert events.get_nowait()[0] == "done"
    assert events.empty()


def test_failure_summary_preserves_http_rejection():
    from aso_agents.gui.presentation import result_message
    message = result_message({"status": "fatal", "failed_stage": "intake", "error": {"message": "OpenRouter HTTP 400"}})
    assert "intake" in message and "HTTP 400" in message
    assert "Connection" not in message


def test_worker_retains_execution_exception_in_redacted_diagnostic(monkeypatch):
    import os
    import sys
    from queue import Queue
    from aso_agents import llm_provider
    from aso_agents.gui.worker import run
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(llm_provider, "prepare_responses_payload", llm_provider.prepare_responses_payload)
    def controller(*, argv, clarification_resolver, result_callback):
        raise TypeError("unexpected field for test-key")
    monkeypatch.setitem(sys.modules, "run_goal_intake", NS(main=controller))
    events = Queue()
    run(settings(), "Request", events, Queue())
    output = []
    while not events.empty():
        output.append(events.get_nowait())
    error = next(value for kind, value in output if kind == "error")
    details = next(value for kind, value in output if kind == "diagnostic")
    assert "TypeError" in error and "unexpected field" in error
    assert details["type"] == "TypeError"
    assert "controller" in details["traceback"]
    assert "test-key" not in str(output)
    assert output[-1][0] == "done"


def test_gui_default_model_validation_without_catalog():
    from aso_agents.gui.config import DEFAULT_OPENROUTER_MODEL
    client = Client()
    validate(settings(model=DEFAULT_OPENROUTER_MODEL, effort="default"), {}, client, base={})
    assert client.payloads[0]["model"] == DEFAULT_OPENROUTER_MODEL

def test_default_settings_preserve_global_and_per_agent_environment():
    base = {"OPENROUTER_MODEL": "stale/model", "ASO_GOAL_INTAKE_MODEL": "role/model"}
    result = environment_for(settings(model="default", effort="default"), base)
    assert result["OPENROUTER_MODEL"] == "stale/model"
    assert result["ASO_GOAL_INTAKE_MODEL"] == "role/model"
    assert base["OPENROUTER_MODEL"] == "stale/model"
