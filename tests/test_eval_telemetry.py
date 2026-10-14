from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents import Agent, Runner, function_tool
from aso_agents import llm_provider
from aso_agents.telemetry import (
    capture_eval_trace,
    current_eval_trace,
    evaluation_trace,
    record_tool_execution,
)
from aso_agents.resource_limits import acquire_resource


class _FakeResponsesClient:
    def __init__(self, responses: list[Any]):
        self.responses = self
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _openrouter_config() -> llm_provider.ProviderConfig:
    return llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test-only",
            "OPENROUTER_MODEL": "deepseek/deepseek-v4-flash-0731",
        }
    )


def test_capture_normalizes_usage_and_safely_records_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "sk-or-v1-this-must-never-be-written"
    response = {
        "id": "gen-123",
        "status": "completed",
        "model": "deepseek/deepseek-v4-flash-0731",
        "output": [
            {
                "type": "reasoning",
                "summary": "private chain of thought",
                "encrypted_content": "encrypted-private-thought",
            },
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": f"safe answer; accidental {secret}"}
                ],
            },
            {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_123",
                "name": "resolve_target",
                "arguments": json.dumps(
                    {
                        "gene": "MAPT",
                        "api_key": secret,
                        "reasoning": "hidden tool rationale",
                    }
                ),
            },
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 9},
            "cost": 0.00125,
        },
        "openrouter_metadata": {
            "requested": "deepseek/deepseek-v4-flash-0731",
            "strategy": "direct",
            "region": "iad",
            "attempt": 2,
            "summary": "selected DeepSeek",
            "is_byok": False,
            "attempts": [
                {"provider": "DeepSeek", "model": "deepseek-v4", "status": 200}
            ],
            "unsafe_custom_field": secret,
            "pipeline": [
                {
                    "type": "context_compression",
                    "name": "middle-out",
                    "data": {"raw_prompt": secret},
                }
            ],
        },
    }
    client = _FakeResponsesClient([response])
    monkeypatch.setattr(llm_provider, "_get_client", lambda _config: client)
    output_path = tmp_path / "trace.json"

    with capture_eval_trace(
        eval_run_id="eval-001",
        case_id="targeting.mapt",
        output_path=output_path,
    ) as collector:
        assert current_eval_trace() is collector
        result = llm_provider.create_response(
            {"model": "gpt-5-mini-2025-08-07", "input": []},
            _openrouter_config(),
        )
        record_tool_execution(
            agent="targeting-agent",
            tool="resolve_target",
            call_id="call_123",
            arguments={"gene": "MAPT", "authorization": f"Bearer {secret}"},
            result={"transcript": "ENST00000334239", "secret": secret},
        )

        assert result == response
        assert len(collector.events) == 1
        event = collector.events[0]
        assert event["provider"] == "openrouter"
        assert event["requested_model"] == "gpt-5-mini-2025-08-07"
        assert event["resolved_model"] == "deepseek/deepseek-v4-flash-0731"
        assert event["response_model"] == "deepseek/deepseek-v4-flash-0731"
        assert event["response_id"] == "gen-123"
        assert event["status"] == "completed"
        assert event["output_types"] == ["message", "function_call"]
        assert event["usage"] == {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "cached_tokens": 80,
            "cache_write_tokens": 4,
            "reasoning_tokens": 9,
            "cost": 0.00125,
        }
        assert event["parsed_output"]["tool_calls"][0]["arguments"] == {
            "gene": "MAPT",
            "api_key": "[REDACTED]",
            "reasoning": "[REDACTED]",
        }
        assert event["tool_executions"][0]["result"] == {
            "transcript": "ENST00000334239",
            "secret": "[REDACTED]",
        }
        assert event["openrouter_metadata"]["attempt"] == 2
        assert "unsafe_custom_field" not in event["openrouter_metadata"]

        sent = client.calls[0]
        assert sent["extra_headers"]["X-OpenRouter-Metadata"] == "enabled"
        assert sent["extra_body"]["trace"]["trace_id"] == collector.trace_id
        assert sent["extra_body"]["trace"]["trace_name"] == "eval-001"
        assert secret not in json.dumps(collector.snapshot())
        assert "private chain of thought" not in json.dumps(collector.snapshot())
        assert "encrypted-private-thought" not in json.dumps(collector.snapshot())

    assert current_eval_trace() is None
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["summary"]["call_count"] == 1
    assert persisted["summary"]["usage"]["cost"] == 0.00125
    assert persisted["summary"]["usage"]["cached_tokens"] == 80


def test_inactive_capture_does_not_change_openrouter_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {"id": "gen-inactive", "status": "completed", "output": []}
    client = _FakeResponsesClient([response])
    monkeypatch.setattr(llm_provider, "_get_client", lambda _config: client)

    result = llm_provider.create_response(
        {"model": "gpt-5-mini-2025-08-07", "input": []},
        _openrouter_config(),
    )

    assert result == response
    assert client.calls == [
        {
            "model": "deepseek/deepseek-v4-flash-0731",
            "input": [],
            "extra_body": {"provider": {"sort": "throughput"}},
        }
    ]
    assert current_eval_trace() is None


def test_trace_records_requested_reasoning_controls_without_reasoning_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "id": "gen-none",
        "status": "completed",
        "model": "deepseek/deepseek-v4-flash-0731",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    client = _FakeResponsesClient([response])
    monkeypatch.setattr(llm_provider, "_get_client", lambda _config: client)
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test-only",
            "OPENROUTER_MODEL": "deepseek/deepseek-v4-flash-0731",
            "OPENROUTER_REASONING_EFFORT": "none",
        }
    )

    with evaluation_trace({"case_id": "reasoning-controls"}) as trace:
        llm_provider.create_response(
            {"model": "gpt-5-mini-2025-08-07", "input": []}, config
        )

    assert trace.calls[0]["requested_reasoning_effort"] == "none"
    assert trace.calls[0]["requested_reasoning_enabled"] is False
    assert trace.summary()["requested_reasoning_efforts"] == ["none"]


def test_input_output_token_names_and_failed_calls_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successful_client = _FakeResponsesClient(
        [
            {
                "id": "resp-openai",
                "status": "completed",
                "model": "gpt-5-mini-2025-08-07",
                "output": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
            }
        ]
    )
    monkeypatch.setattr(llm_provider, "_get_client", lambda _config: successful_client)
    config = llm_provider.get_provider_config({"OPENAI_API_KEY": "openai-test"})

    with evaluation_trace({"case_id": "usage-aliases"}) as trace:
        llm_provider.create_response(
            {"model": "gpt-5-mini-2025-08-07", "input": []}, config
        )

    assert trace.calls[0]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "cached_tokens": 3,
        "cache_write_tokens": 0,
        "reasoning_tokens": 2,
        "cost": None,
    }

    transport_secret = "sk-transport-message-must-not-leak"
    failing_client = _FakeResponsesClient([RuntimeError(transport_secret)])
    monkeypatch.setattr(llm_provider, "_get_client", lambda _config: failing_client)
    with evaluation_trace({"case_id": "transport"}) as failed_trace:
        with pytest.raises(RuntimeError, match="API transport"):
            llm_provider.create_response(
                {"model": "gpt-5-mini-2025-08-07", "input": []}, config
            )

    assert failed_trace.calls[0]["status"] == "transport_error"
    assert failed_trace.calls[0]["error_type"] == "RuntimeError"
    assert transport_secret not in json.dumps(failed_trace.snapshot())


def test_context_isolation_restores_outer_trace() -> None:
    assert current_eval_trace() is None


def test_trace_separately_records_resource_admission_wait(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ASO_RESOURCE_LOCK_ROOT", str(tmp_path))
    monkeypatch.setenv("ASO_RESOURCE_LIMIT_GPU", "1")

    with evaluation_trace({"case_id": "resource-admission"}) as trace:
        with acquire_resource("gpu"):
            pass

    snapshot = trace.snapshot()
    assert len(snapshot["resource_acquisitions"]) == 1
    assert snapshot["resource_acquisitions"][0]["resource"] == "gpu"
    summary = trace.summary()
    assert summary["resource_acquisition_count"] == 1
    assert summary["total_resource_wait_ms"] >= 0
    assert summary["resource_wait_ms_by_resource"]["gpu"] >= 0
    with evaluation_trace({"case_id": "outer"}) as outer:
        assert current_eval_trace() is outer
        with evaluation_trace({"case_id": "inner"}) as inner:
            assert current_eval_trace() is inner
        assert current_eval_trace() is outer
    assert current_eval_trace() is None


def test_runner_attaches_executed_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @function_tool
    def resolve_gene(gene: str) -> dict[str, str]:
        return {"gene": gene, "transcript": "ENST00000334239"}

    client = _FakeResponsesClient(
        [
            {
                "id": "gen-tool",
                "status": "completed",
                "model": "deepseek/deepseek-v4-flash-0731",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-gene",
                        "name": "resolve_gene",
                        "arguments": '{"gene":"MAPT"}',
                    }
                ],
            },
            {
                "id": "gen-final",
                "status": "completed",
                "model": "deepseek/deepseek-v4-flash-0731",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
        ]
    )
    config = _openrouter_config()
    monkeypatch.setattr(llm_provider, "_get_client", lambda _config: client)
    monkeypatch.setattr(llm_provider, "get_provider_config", lambda: config)
    agent = Agent(name="goal-intake", instructions="Use the tool.", tools=[resolve_gene])

    with evaluation_trace({"case_id": "runner-tool"}) as trace:
        result = Runner._run_sync(agent, "Resolve MAPT")

    assert result.final_output == "done"
    assert len(trace.calls) == 2
    assert trace.calls[0]["tool_executions"] == [
        {
            "agent": "goal-intake",
            "tool": "resolve_gene",
            "call_id": "call-gene",
            "arguments": {"gene": "MAPT"},
            "result": {"gene": "MAPT", "transcript": "ENST00000334239"},
        }
    ]
