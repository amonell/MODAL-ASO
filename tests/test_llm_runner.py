from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest

from agents import Agent, ModelSettings, Runner, _RunResult, function_tool
from aso_agents import llm_provider


class _FakeModelResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = copy.deepcopy(data)

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(self._data)


class _FakeOpenAI:
    instances: list["_FakeOpenAI"] = []
    queued_responses: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = copy.deepcopy(kwargs)
        self.calls: list[dict[str, Any]] = []
        self.responses = self
        type(self).instances.append(self)

    def create(self, **kwargs: Any) -> _FakeModelResponse:
        self.calls.append(copy.deepcopy(kwargs))
        if not type(self).queued_responses:
            raise AssertionError("Runner made an unexpected Responses request")
        return _FakeModelResponse(type(self).queued_responses.pop(0))


@pytest.fixture(autouse=True)
def _reset_provider_runtime() -> None:
    llm_provider.reset_provider_runtime_for_tests()
    yield
    llm_provider.reset_provider_runtime_for_tests()


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    responses: list[dict[str, Any]],
) -> type[_FakeOpenAI]:
    _FakeOpenAI.instances = []
    _FakeOpenAI.queued_responses = copy.deepcopy(responses)
    monkeypatch.setattr(llm_provider, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("ASO_LLM_PROVIDER", provider)
    if provider == "openrouter":
        monkeypatch.setenv("OPENROUTER_API_KEY", "router-test")
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    return _FakeOpenAI


def _message_response(
    text: str,
    *,
    response_id: str = "resp_final",
    model: str = "gpt-5.2-2025-12-11",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg_final",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
    }


def _runner_settings() -> ModelSettings:
    return ModelSettings(
        reasoning={"effort": "high"},
        verbosity="medium",
        max_output_tokens=321,
        extra_body={"text": {"format": {"type": "json_object"}}},
    )


@function_tool
def double(value: int) -> dict[str, int]:
    """Double one integer."""

    return {"doubled": value * 2}


@function_tool
def label(value: str) -> str:
    """Return a stable label."""

    return f"label:{value}"


def test_openai_second_turn_is_stateful_and_contains_only_function_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_output = [
        {
            "id": "reasoning_1",
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "encrypted_content": "must-not-enter-compact-history",
        },
        {
            "id": "fc_item_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_double",
            "name": "double",
            "arguments": '{"value": 6}',
        },
    ]
    fake = _install_fake_client(
        monkeypatch,
        "openai",
        [
            {
                "id": "resp_openai_1",
                "status": "completed",
                "model": "gpt-5.2-2025-12-11",
                "output": first_output,
            },
            _message_response('{"answer":12}'),
        ],
    )
    agent = Agent(
        "runner-test",
        "Follow the tool protocol.",
        tools=[double],
        model="gpt-5.2-2025-12-11",
        model_settings=_runner_settings(),
    )

    result = Runner._run_sync(agent, input="Double six.", max_steps=3)

    assert result.final_output == '{"answer":12}'
    calls = fake.instances[0].calls
    assert len(calls) == 2
    assert calls[1] == {
        "model": "gpt-5.2-2025-12-11",
        "previous_response_id": "resp_openai_1",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_double",
                "output": '{"doubled": 12}',
            }
        ],
        "tools": calls[0]["tools"],
        "reasoning": {"effort": "high"},
        "verbosity": "medium",
        "max_output_tokens": 321,
        "text": {"format": {"type": "json_object"}},
    }
    assert first_output[0] not in calls[1]["input"]
    first_trace = result.responses[0]["response"]
    assert first_trace["provider"] == "openai"
    assert first_trace["requested_model"] == "gpt-5.2-2025-12-11"
    assert first_trace["resolved_model"] == "gpt-5.2-2025-12-11"
    assert first_trace["model"] == "gpt-5.2-2025-12-11"
    assert first_trace["output_types"] == ["reasoning", "function_call"]
    assert "must-not-enter-compact-history" not in repr(result.responses)


def test_openrouter_second_turn_accumulates_complete_raw_output_without_response_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_output = [
        {
            "id": "reasoning_router_1",
            "type": "reasoning",
            "status": "completed",
            "phase": "analysis",
            "summary": [{"type": "summary_text", "text": "private reasoning summary"}],
            "encrypted_content": "encrypted-reasoning-payload",
        },
        {
            "id": "fc_router_item_1",
            "type": "function_call",
            "status": "completed",
            "phase": "analysis",
            "call_id": "call_router_double",
            "name": "double",
            "arguments": '{"value": 7}',
        },
    ]
    fake = _install_fake_client(
        monkeypatch,
        "openrouter",
        [
            {
                "status": "completed",
                "model": "openai/gpt-5.2",
                "output": first_output,
            },
            _message_response('{"answer":14}', model="openai/gpt-5.2"),
        ],
    )
    agent = Agent(
        "runner-test",
        "Follow the tool protocol.",
        tools=[double],
        model="gpt-5.2-2025-12-11",
        model_settings=_runner_settings(),
    )

    result = Runner._run_sync(agent, input="Double seven.", max_steps=3)

    assert result.final_output == '{"answer":14}'
    calls = fake.instances[0].calls
    assert len(calls) == 2
    assert calls[1] == {
        "model": "openai/gpt-5.2",
        "input": calls[0]["input"]
        + first_output
        + [
            {
                "type": "function_call_output",
                "call_id": "call_router_double",
                "output": '{"doubled": 14}',
            }
        ],
        "tools": calls[0]["tools"],
        "reasoning": {"effort": "high"},
        "verbosity": "medium",
        "max_output_tokens": 321,
        "text": {"format": {"type": "json_object"}},
        "extra_body": {"provider": {"sort": "throughput"}},
    }
    assert "previous_response_id" not in calls[1]
    assert calls[1]["input"][2:4] == first_output
    first_trace = result.responses[0]["response"]
    assert first_trace["provider"] == "openrouter"
    assert first_trace["requested_model"] == "gpt-5.2-2025-12-11"
    assert first_trace["resolved_model"] == "openai/gpt-5.2"
    assert first_trace["model"] == "openai/gpt-5.2"
    assert "private reasoning summary" not in repr(result.responses)
    assert "encrypted-reasoning-payload" not in repr(result.responses)


def test_openrouter_global_model_override_applies_to_nested_agent_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_client(
        monkeypatch,
        "openrouter",
        [
            _message_response("strong", model="qwen/qwen3.8-27b"),
            _message_response("fast", model="qwen/qwen3.8-27b"),
        ],
    )
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b")

    strong_result = Runner._run_sync(
        Agent("strong-agent", "Answer.", model="gpt-5.2-2025-12-11"),
        input="First.",
    )
    fast_result = Runner._run_sync(
        Agent("fast-agent", "Answer.", model="gpt-5-mini-2025-08-07"),
        input="Second.",
    )

    assert strong_result.final_output == "strong"
    assert fast_result.final_output == "fast"
    assert [instance.calls[0]["model"] for instance in fake.instances] == [
        "qwen/qwen3.8-27b",
    ]
    assert [call["model"] for call in fake.instances[0].calls] == [
        "qwen/qwen3.8-27b",
        "qwen/qwen3.8-27b",
    ]
    for result in (strong_result, fast_result):
        assert result.responses[0]["response"]["resolved_model"] == "qwen/qwen3.8-27b"


def test_openrouter_parallel_function_calls_keep_order_and_stable_call_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function_calls = [
        {
            "id": "fc_a",
            "type": "function_call",
            "call_id": "call_a",
            "name": "label",
            "arguments": '{"value":"alpha"}',
        },
        {
            "id": "fc_b",
            "type": "function_call",
            "call_id": "call_b",
            "name": "double",
            "arguments": '{"value":4}',
        },
    ]
    fake = _install_fake_client(
        monkeypatch,
        "openrouter",
        [
            {
                "id": "resp_parallel",
                "status": "completed",
                "parallel_tool_calls": True,
                "model": "openai/gpt-5.2",
                "output": function_calls,
            },
            _message_response("done", model="openai/gpt-5.2"),
        ],
    )
    events: list[dict[str, Any]] = []
    agent = Agent(
        "parallel-test",
        "Use both tools.",
        tools=[label, double],
        model="gpt-5.2-2025-12-11",
    )

    result = Runner._run_sync(agent, input="Run both.", max_steps=3, event_callback=events.append)

    assert result.final_output == "done"
    second_input = fake.instances[0].calls[1]["input"]
    assert second_input[-2:] == [
        {
            "type": "function_call_output",
            "call_id": "call_a",
            "output": "label:alpha",
        },
        {
            "type": "function_call_output",
            "call_id": "call_b",
            "output": '{"doubled": 8}',
        },
    ]
    assert [event["type"] for event in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final_output",
    ]


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_incomplete_output_preserves_structured_error(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    response_model = "openai/gpt-5-mini" if provider == "openrouter" else "gpt-5-mini-2025-08-07"
    _install_fake_client(
        monkeypatch,
        provider,
        [
            {
                "id": "resp_incomplete",
                "status": "incomplete",
                "model": response_model,
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            }
        ],
    )
    agent = Agent("incomplete-test", "Return JSON.")

    result = Runner._run_sync(agent, input="Answer.")

    assert result.final_output == {
        "error": "incomplete_response",
        "status": "incomplete",
        "reason": "max_output_tokens",
    }
    assert result.responses[0]["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_failed_provider_response_raises_without_empty_turn_retries(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    response_model = "openai/gpt-5-mini" if provider == "openrouter" else "gpt-5-mini-2025-08-07"
    fake = _install_fake_client(
        monkeypatch,
        provider,
        [
            {
                "id": "resp_failed",
                "status": "failed",
                "model": response_model,
                "error": {
                    "code": "provider_error",
                    "type": "upstream_error",
                    "message": "Structured output was rejected.",
                },
                "output": [],
            }
        ],
    )
    agent = Agent("failed-test", "Return JSON.")

    with pytest.raises(RuntimeError, match="LLM API response failed") as exc_info:
        Runner._run_sync(agent, input="Answer.")

    assert "provider_error" in str(exc_info.value)
    assert "Structured output was rejected" in str(exc_info.value)
    assert len(fake.instances[0].calls) == 1


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_empty_turn_nudge_uses_provider_continuation_contract(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    empty_output = [
        {
            "id": "reasoning_empty",
            "type": "reasoning",
            "status": "completed",
            "summary": [],
        }
    ]
    response_model = "openai/gpt-5-mini" if provider == "openrouter" else "gpt-5-mini-2025-08-07"
    fake = _install_fake_client(
        monkeypatch,
        provider,
        [
            {
                "id": "resp_empty",
                "status": "completed",
                "model": response_model,
                "output": empty_output,
            },
            _message_response("finished", model=response_model),
        ],
    )
    settings = ModelSettings(reasoning={"effort": "low"}, max_output_tokens=99)
    agent = Agent("empty-test", "Finish the answer.", model_settings=settings)

    result = Runner._run_sync(agent, input="Begin.", max_steps=3)

    assert result.final_output == "finished"
    calls = fake.instances[0].calls
    nudge = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Continue. Either call the next tool or return your final JSON answer.",
                }
            ],
        }
    ]
    expected_model = "openai/gpt-5-mini" if provider == "openrouter" else "gpt-5-mini-2025-08-07"
    if provider == "openai":
        assert calls[1] == {
            "model": expected_model,
            "previous_response_id": "resp_empty",
            "input": nudge,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 99,
        }
    else:
        assert calls[1] == {
            "model": expected_model,
            "input": calls[0]["input"] + empty_output + nudge,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 99,
            "extra_body": {"provider": {"sort": "throughput"}},
        }
        assert "previous_response_id" not in calls[1]


def test_openai_requires_response_id_for_tool_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_client(
        monkeypatch,
        "openai",
        [
            {
                "status": "completed",
                "model": "gpt-5-mini-2025-08-07",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_without_response_id",
                        "name": "double",
                        "arguments": '{"value":2}',
                    }
                ],
            }
        ],
    )
    agent = Agent("missing-id-test", "Use the tool.", tools=[double])

    result = Runner._run_sync(agent, input="Double two.")

    assert result.final_output["error"] == "missing_response_id"
    assert len(_FakeOpenAI.instances[0].calls) == 1


def test_missing_function_call_id_is_a_hard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_client(
        monkeypatch,
        "openrouter",
        [
            {
                "status": "completed",
                "model": "openai/gpt-5-mini",
                "output": [
                    {
                        "id": "function_item_without_call_id",
                        "type": "function_call",
                        "name": "double",
                        "arguments": '{"value":2}',
                    }
                ],
            }
        ],
    )
    agent = Agent("missing-call-id-test", "Use the tool.", tools=[double])

    with pytest.raises(RuntimeError, match="Missing tool call id"):
        Runner._run_sync(agent, input="Double two.")


def test_run_delegates_to_legacy_run_sync_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = _RunResult("patched")

    def fake_run_sync(
        agent: Agent,
        input: str,
        temperature: float = 0.2,
        max_steps: int = 5,
        event_callback: Any = None,
    ) -> _RunResult:
        calls.append(
            {
                "agent": agent,
                "input": input,
                "temperature": temperature,
                "max_steps": max_steps,
                "event_callback": event_callback,
            }
        )
        return sentinel

    monkeypatch.setattr(Runner, "_run_sync", fake_run_sync)
    agent = Agent("compatibility-test", "No network.")
    callback = lambda _event: None

    result = asyncio.run(
        Runner.run(
            agent,
            input="legacy call",
            temperature=0.4,
            max_steps=7,
            event_callback=callback,
        )
    )

    assert result is sentinel
    assert calls == [
        {
            "agent": agent,
            "input": "legacy call",
            "temperature": 1.0,
            "max_steps": 7,
            "event_callback": callback,
        }
    ]


def test_run_method_can_still_be_monkeypatched_with_existing_async_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run(agent: Agent, input: str, max_steps: int = 5) -> _RunResult:
        return _RunResult({"agent": agent.name, "input": input, "max_steps": max_steps})

    monkeypatch.setattr(Runner, "run", fake_run)

    result = asyncio.run(Runner.run(Agent("patched-run", "No network."), input="hello", max_steps=4))

    assert result.final_output == {
        "agent": "patched-run",
        "input": "hello",
        "max_steps": 4,
    }
