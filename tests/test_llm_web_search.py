from __future__ import annotations

import copy
import importlib
import json
from typing import Any, Callable

import pytest

from aso_agents import llm_provider, pipeline_tools


transcript_chooser_agent = importlib.import_module("aso_agents.transcript_chooser_agent")


@pytest.fixture(params=("openai", "openrouter"))
def hosted_search_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    provider = str(request.param)
    if provider == "openai":
        config = llm_provider.get_provider_config(
            {"ASO_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "openai-test"}
        )
        expected_tool = {"type": "web_search"}
        citation = {
            "type": "url_citation",
            "title": "Ensembl transcript record",
            "url": "https://www.ensembl.org/id/ENST00000334239",
        }
        expected_evidence = {
            "title": "Ensembl transcript record",
            "url": "https://www.ensembl.org/id/ENST00000334239",
        }
        citation_placement = "nested_output_text"
    else:
        config = llm_provider.get_provider_config(
            {"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "router-test"}
        )
        expected_tool = {
            "type": "openrouter:web_search",
            "parameters": {"max_uses": 3},
        }
        citation = {
            "type": "url_citation",
            "source_name": "NCBI",
            "title": "NCBI transcript record",
            "url": "https://www.ncbi.nlm.nih.gov/nuccore/NM_001123",
        }
        expected_evidence = {
            "source": "NCBI",
            "title": "NCBI transcript record",
            "url": "https://www.ncbi.nlm.nih.gov/nuccore/NM_001123",
        }
        citation_placement = "top_level_output_text"
    return {
        "provider": provider,
        "config": config,
        "expected_tool": expected_tool,
        "citation": citation,
        "expected_evidence": expected_evidence,
        "citation_placement": citation_placement,
    }


def _completed_response(case: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    output_text = json.dumps(body)
    citation = copy.deepcopy(case["citation"])
    if case["citation_placement"] == "nested_output_text":
        output = [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [citation],
                    }
                ],
            }
        ]
    else:
        output = [
            {
                "type": "output_text",
                "text": output_text,
                "annotations": [citation],
            }
        ]
    return {
        "id": f"resp-{case['provider']}",
        "status": "completed",
        "output": output,
    }


def _install_fake_facade(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    case: dict[str, Any],
    responder: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"config": [], "build": [], "payloads": [], "response_config": []}
    real_tool_builder = llm_provider.build_hosted_web_search_tool

    def fake_get_provider_config() -> llm_provider.ProviderConfig:
        calls["config"].append(True)
        return case["config"]

    def fake_build_hosted_web_search_tool(
        config: llm_provider.ProviderConfig,
        *,
        max_uses: int = 3,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        calls["build"].append((config, max_uses, max_results))
        return real_tool_builder(config, max_uses=max_uses, max_results=max_results)

    def fake_create_response(
        payload: dict[str, Any],
        config: llm_provider.ProviderConfig | None = None,
    ) -> dict[str, Any]:
        calls["payloads"].append(copy.deepcopy(payload))
        calls["response_config"].append(config)
        return copy.deepcopy(responder(payload))

    monkeypatch.setattr(module, "get_provider_config", fake_get_provider_config)
    monkeypatch.setattr(module, "build_hosted_web_search_tool", fake_build_hosted_web_search_tool)
    monkeypatch.setattr(module, "create_response", fake_create_response)
    return calls


def test_transcript_search_uses_facade_tool_and_normalizes_provider_citations(
    monkeypatch: pytest.MonkeyPatch,
    hosted_search_case: dict[str, Any],
) -> None:
    response = _completed_response(
        hosted_search_case,
        {
            "transcript_id": "ENST00000334239",
            "confidence": "high",
            "rationale": "The primary record identifies the requested transcript.",
            "requires_confirmation": False,
            "evidence": [],
        },
    )
    calls = _install_fake_facade(
        monkeypatch,
        transcript_chooser_agent,
        hosted_search_case,
        lambda _payload: response,
    )

    def fail_direct_post(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("transcript hosted search must not issue a direct HTTP POST")

    monkeypatch.setattr(transcript_chooser_agent.requests, "post", fail_direct_post)

    result = transcript_chooser_agent.web_search("Resolve the requested isoform")

    assert result["extracted_transcript_id"] == "ENST00000334239"
    assert result["confidence"] == "high"
    assert result["requires_confirmation"] is False
    assert result["evidence"] == [hosted_search_case["expected_evidence"]]
    assert result["citations"] == [hosted_search_case["citation"]]
    assert calls["config"] == [True]
    assert calls["build"] == [(hosted_search_case["config"], 3, None)]
    assert calls["response_config"] == [hosted_search_case["config"]]
    assert len(calls["payloads"]) == 1
    payload = calls["payloads"][0]
    assert payload["tools"] == [hosted_search_case["expected_tool"]]
    assert payload["max_tool_calls"] == 3
    assert payload["parallel_tool_calls"] is False
    assert payload["max_output_tokens"] == 8192
    assert payload["model"] == "gpt-5.2-2025-12-11"
    assert "Resolve the requested isoform" in payload["input"]


def test_answer_workbench_search_uses_facade_tool_and_normalizes_provider_citations(
    monkeypatch: pytest.MonkeyPatch,
    hosted_search_case: dict[str, Any],
) -> None:
    response = _completed_response(
        hosted_search_case,
        {
            "summary": "Source-backed workbench answer.",
            "answerable": True,
            "evidence": [],
            "unresolved_questions": [],
        },
    )
    calls = _install_fake_facade(
        monkeypatch,
        pipeline_tools,
        hosted_search_case,
        lambda _payload: response,
    )

    def fail_runner_transport(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("answer workbench hosted search must not use Runner._responses")

    monkeypatch.setattr(pipeline_tools.Runner, "_responses", fail_runner_transport)

    result = pipeline_tools._run_answer_web_search_once(
        query="What evidence supports this design choice?",
        request_text="Explain the design choice with sources.",
        source_policy="official_first",
        reason="The user requested current references.",
    )

    assert result == {
        "query": "What evidence supports this design choice?",
        "status": "ok",
        "response_id": f"resp-{hosted_search_case['provider']}",
        "summary": "Source-backed workbench answer.",
        "answerable": True,
        "evidence": [hosted_search_case["expected_evidence"]],
        "unresolved_questions": [],
    }
    assert calls["config"] == [True]
    assert calls["build"] == [(hosted_search_case["config"], 3, None)]
    assert calls["response_config"] == [hosted_search_case["config"]]
    assert len(calls["payloads"]) == 1
    payload = calls["payloads"][0]
    assert payload["tools"] == [hosted_search_case["expected_tool"]]
    assert payload["max_tool_calls"] == 3
    assert payload["parallel_tool_calls"] is False
    assert payload["max_output_tokens"] == 8192
    assert payload["model"] == "gpt-5.2-2025-12-11"
    assert "official vendor" in payload["input"]


def test_transcript_search_retries_only_token_limit_incomplete_output(
    monkeypatch: pytest.MonkeyPatch,
    hosted_search_case: dict[str, Any],
) -> None:
    responses = iter(
        [
            {
                "id": "resp-incomplete",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
            _completed_response(
                hosted_search_case,
                {
                    "transcript_id": None,
                    "confidence": "none",
                    "rationale": "Primary sources were ambiguous.",
                    "requires_confirmation": False,
                    "evidence": [],
                },
            ),
        ]
    )
    calls = _install_fake_facade(
        monkeypatch,
        transcript_chooser_agent,
        hosted_search_case,
        lambda _payload: next(responses),
    )

    result = transcript_chooser_agent.web_search("Ambiguous isoform")

    assert result["confidence"] == "none"
    assert [payload["max_output_tokens"] for payload in calls["payloads"]] == [8192, 16384]
    assert all(
        payload["tools"] == [hosted_search_case["expected_tool"]]
        and payload["max_tool_calls"] == 3
        for payload in calls["payloads"]
    )


def test_transcript_search_preserves_non_token_incomplete_output(
    monkeypatch: pytest.MonkeyPatch,
    hosted_search_case: dict[str, Any],
) -> None:
    incomplete = {
        "id": "resp-filtered",
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "output": [],
    }
    calls = _install_fake_facade(
        monkeypatch,
        transcript_chooser_agent,
        hosted_search_case,
        lambda _payload: incomplete,
    )

    result = transcript_chooser_agent.web_search("Filtered query")

    assert result == {
        "query": "Filtered query",
        "output_text": "",
        "citations": [],
        "raw": incomplete,
        "error": "web_search incomplete: {'reason': 'content_filter'}",
    }
    assert len(calls["payloads"]) == 1


def test_answer_workbench_search_preserves_retry_and_incomplete_behavior(
    monkeypatch: pytest.MonkeyPatch,
    hosted_search_case: dict[str, Any],
) -> None:
    responses = iter(
        [
            {
                "id": "resp-token-limit",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
            {
                "id": "resp-still-incomplete",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
            {
                "id": "resp-final-incomplete",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
        ]
    )
    calls = _install_fake_facade(
        monkeypatch,
        pipeline_tools,
        hosted_search_case,
        lambda _payload: next(responses),
    )

    result = pipeline_tools._run_answer_web_search_once(
        query="Current evidence",
        request_text="Summarize current evidence.",
        source_policy="primary_sources_only",
        reason="Evidence is required.",
    )

    assert result == {
        "query": "Current evidence",
        "error": "answer_web_search_incomplete: {'reason': 'max_output_tokens'}",
        "response_id": "resp-final-incomplete",
    }
    assert [payload["max_output_tokens"] for payload in calls["payloads"]] == [
        8192,
        16384,
        32768,
    ]


def test_transcript_search_credential_gate_is_provider_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcript_chooser_agent, "provider_credentials_available", lambda: False)

    def fail_search(_query: str) -> None:
        raise AssertionError("search should not run without the selected provider credential")

    monkeypatch.setattr(transcript_chooser_agent, "web_search", fail_search)

    result = transcript_chooser_agent._attempt_transcript_via_web_search(
        "GENE1",
        "isoform a",
        [{"id": "ENST00000334239"}],
    )

    assert result is None
    assert not hasattr(transcript_chooser_agent, "OPENAI_API_KEY")
    assert not hasattr(transcript_chooser_agent, "OPENAI_BASE_URL")


def test_transcript_search_preserves_primary_source_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcript_chooser_agent,
        "provider_credentials_available",
        lambda: True,
    )
    monkeypatch.setattr(
        transcript_chooser_agent,
        "web_search",
        lambda _query: {
            "extracted_transcript_id": "ENST00000334239",
            "confidence": "high",
            "rationale": "The primary record supports this transcript.",
            "evidence_summary": "Primary transcript record.",
            "evidence": [
                {
                    "source": "primary database",
                    "url": "https://primary.example/transcript/ENST00000334239",
                    "note": "Reviewed transcript record.",
                }
            ],
        },
    )

    result = transcript_chooser_agent._attempt_transcript_via_web_search(
        "GENE1",
        "requested isoform",
        [{"id": "ENST00000334239"}],
    )

    assert result is not None
    assert result["evidence_refs"] == [
        "https://primary.example/transcript/ENST00000334239"
    ]
