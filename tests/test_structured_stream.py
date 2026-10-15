from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from agents import Agent, ModelSettings, Runner
from aso_agents import llm_provider
from aso_agents.structured_stream import RootArrayGuard, consume_guarded_response, root_array_constraints


def payload(rule=None):
    return {"model": "gpt-5-mini-2025-08-07", "input": "Return JSON.", "text": {"format": {
        "type": "json_schema", "name": "finite_list", "strict": True,
        "schema": {"type": "object", "properties": {"items": {
            "type": "array", "items": {"type": "integer"}, **(rule or {"uniqueItems": True}),
        }}, "required": ["items"], "additionalProperties": False},
    }}}


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 256])
@pytest.mark.parametrize("items,duplicate", [
    ([1, 10, 100], False), ([True, 1, False, 0, None, "1"], False),
    ([1, 1.0], True), ([0, -0.0], True),
    ([{"a": [1, 2], "b": "x"}, {"b": "x", "a": [1, 2]}], True),
    ([[1, 2], [2, 1]], False), (["quote\"slash\\brace]", "quote\"slash\\brace]"], True),
    (["é", "e"], False), ([None, None], True),
])
def test_fragmentation_and_json_schema_equality(chunk_size, items, duplicate):
    guard = RootArrayGuard({"items": {"uniqueItems": True}})
    text = json.dumps({"ignored": {"nested": [2, 2]}, "items": items, "tail": [3, 3]})
    violation = None
    for index in range(0, len(text), chunk_size):
        violation = guard.feed(text[index:index + chunk_size])
        if violation:
            break
    assert bool(violation) is duplicate
    if duplicate:
        assert violation["keyword"] == "uniqueItems"
        assert violation["duplicate_of"] == 0


def test_numeric_prefix_waits_for_delimiter():
    guard = RootArrayGuard({"items": {"uniqueItems": True}})
    assert guard.feed('{"items":[1, 1') is None
    assert guard.feed('e') is None
    assert guard.feed('1') is None
    assert guard.feed(']}') is None


@pytest.mark.parametrize("limit", [0, 1, 5])
def test_max_items_allows_exact_limit_and_rejects_extra(limit):
    guard = RootArrayGuard({"items": {"maxItems": limit}})
    assert guard.feed(json.dumps({"items": list(range(limit))})) is None
    guard = RootArrayGuard({"items": {"maxItems": limit}})
    violation = guard.feed(json.dumps({"items": list(range(limit + 1))}))
    assert violation == {"property": "items", "item_index": limit, "keyword": "maxItems", "limit": limit}


def test_constraints_require_explicit_canonical_root_array_rules():
    assert root_array_constraints(payload()) == {"items": payload()["text"]["format"]["schema"]["properties"]["items"]}
    assert root_array_constraints(payload({"maxItems": 0}))["items"]["maxItems"] == 0
    assert not root_array_constraints(payload({"uniqueItems": False}))
    assert not root_array_constraints(payload({"maxItems": True}))
    assert not root_array_constraints({"text": {"format": {"type": "json_object"}}})
    nested = payload()
    nested["text"]["format"]["schema"]["properties"]["items"] = {"type": "object", "properties": {"nested": {"type": "array", "uniqueItems": True}}}
    assert not root_array_constraints(nested)


class Stream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = False
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        result = next(self.events)
        self.consumed += 1
        return result

    def close(self):
        self.closed = True


def events(text, terminal=None):
    yield {"type": "response.created", "response": {"id": "resp_1", "model": "test", "status": "in_progress"}}
    for char in text:
        yield {"type": "response.output_text.delta", "delta": char, "output_index": 0, "content_index": 0}
    if terminal is not None:
        yield {"type": "response." + terminal["status"], "response": terminal}


def completed(text='{"items":[1,2]}'):
    return {"id": "resp_1", "status": "completed", "model": "test",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20, "cost": 0.0001},
            "openrouter_metadata": {"provider_name": "test"}}


@pytest.mark.parametrize("status", ["completed", "failed", "incomplete"])
def test_preserves_terminal_response_and_usage(status):
    response = completed()
    response["status"] = status
    stream = Stream(events('{"items":[1,2]}', response))
    assert consume_guarded_response(stream, {"items": {"uniqueItems": True}}) == response
    assert stream.closed


def test_closes_invalid_generation_without_reading_remainder_or_inventing_cost():
    text = '{"items":[1,1,2,3,4,5,6]}'
    stream = Stream(events(text, completed(text)))
    response = consume_guarded_response(stream, {"items": {"uniqueItems": True}})
    assert stream.closed and stream.consumed < len(text)
    assert response["status"] == "incomplete" and response["output"] == []
    assert response["id"] == "resp_1"
    assert "usage" not in response
    assert response["structured_output_rejection"]["usage_available"] is False
    assert response["structured_output_rejection"]["duplicate_of"] == 0


def test_missing_terminal_event_is_never_success():
    stream = Stream(events('{"items":[1,2]}'))
    with pytest.raises(ValueError, match="without a terminal"):
        consume_guarded_response(stream, {"items": {"uniqueItems": True}})
    assert stream.closed


def test_stream_exception_closes_connection():
    def broken():
        yield {"type": "response.output_text.delta", "delta": "{"}
        raise OSError("connection lost")
    stream = Stream(broken())
    with pytest.raises(OSError, match="connection lost"):
        consume_guarded_response(stream, {"items": {"uniqueItems": True}})
    assert stream.closed


def test_sdk_event_dump_and_nontext_events():
    response = completed()
    stream = Stream([
        SimpleNamespace(model_dump=lambda **kw: {"type": "response.reasoning.delta", "delta": "ignored"}),
        SimpleNamespace(model_dump=lambda **kw: {"type": "response.completed", "response": response}),
    ])
    assert consume_guarded_response(stream, {"items": {"uniqueItems": True}}) == response


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_provider_streams_canonical_constraints_and_retains_telemetry(monkeypatch, provider):
    requests, recorded = [], []
    response = completed()
    stream = Stream(events('{"items":[1,2]}', response))
    def create(**kwargs):
        requests.append(kwargs)
        return stream
    monkeypatch.setattr(llm_provider, "_get_client", lambda config: SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setattr("aso_agents.telemetry.complete_llm_call", lambda pending, data: recorded.append(copy.deepcopy(data)))
    config = llm_provider.get_provider_config({"ASO_LLM_PROVIDER": provider, "OPENAI_API_KEY": "test", "OPENROUTER_API_KEY": "test"})
    assert llm_provider.create_response(payload(), config) == response
    assert requests[0]["stream"] is True
    assert recorded == [response]
    if provider == "openrouter":
        assert requests[0]["text"]["format"]["type"] == "json_object"


def test_rejected_stream_uses_existing_bounded_schema_repair(monkeypatch):
    from aso_agents.pipeline_tools import _run_schema_validated_agent_once
    calls = []
    streams = [Stream(events('{"items":[1,1,2]}', completed())),
               Stream(events('{"items":[1,2]}', completed()))]
    def create(**kwargs):
        calls.append(kwargs)
        return streams[len(calls) - 1]
    monkeypatch.setenv("ASO_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(llm_provider, "_get_client", lambda config: SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent = Agent("unique-list", "Return unique integers.", tools=[], model_settings=ModelSettings(extra_body={"text": payload()["text"]}))
    result = asyncio.run(_run_schema_validated_agent_once(
        agent=agent, prompt="Return the list.", schema=payload()["text"]["format"]["schema"], error_prefix="test", max_attempts=2,
    ))
    assert result == {"items": [1, 2]}
    assert len(calls) == 2 and all(stream.closed for stream in streams)
    assert "uniqueItems" in json.dumps(calls[1]["input"])


def test_repair_budget_cannot_be_extended_by_repeated_invalid_streams(monkeypatch):
    from aso_agents.pipeline_tools import _run_schema_validated_agent_once
    streams = []
    def create(**kwargs):
        stream = Stream(events('{"items":[9,9]}', completed()))
        streams.append(stream)
        return stream
    monkeypatch.setenv("ASO_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(llm_provider, "_get_client", lambda config: SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent = Agent("unique-list", "Return unique integers.", tools=[], model_settings=ModelSettings(extra_body={"text": payload()["text"]}))
    result = asyncio.run(_run_schema_validated_agent_once(
        agent=agent, prompt="Return the list.", schema=payload()["text"]["format"]["schema"], error_prefix="test", max_attempts=2,
    ))
    assert result["error"] == "invalid_structured_output"
    assert len(streams) == 2 and all(stream.closed for stream in streams)
