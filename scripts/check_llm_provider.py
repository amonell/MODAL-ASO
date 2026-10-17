#!/usr/bin/env python3
"""Run an explicitly opted-in, content-safe OpenRouter diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aso_agents.llm_provider import ProviderConfig, create_response, get_provider_config  # noqa: E402


SMOKE_FLAG = "ASO_RUN_OPENROUTER_SMOKE"
WEB_SMOKE_FLAG = "ASO_RUN_OPENROUTER_WEB_SMOKE"
OPENROUTER_CREDENTIAL = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class LogicalModelTarget:
    logical_model: str
    resolved_model: str
    accepted_response_models: frozenset[str]


DEFAULT_LOGICAL_MODEL_TARGETS = (
    LogicalModelTarget(
        logical_model="gpt-5.2-2025-12-11",
        resolved_model="openai/gpt-5.2",
        accepted_response_models=frozenset(
            {
                "openai/gpt-5.2",
                "openai/gpt-5.2-2025-12-11",
            }
        ),
    ),
    LogicalModelTarget(
        logical_model="gpt-5-mini-2025-08-07",
        resolved_model="openai/gpt-5-mini",
        accepted_response_models=frozenset(
            {
                "openai/gpt-5-mini",
                "openai/gpt-5-mini-2025-08-07",
            }
        ),
    ),
)


@dataclass(frozen=True)
class DiagnosticRecord:
    check: str
    provider: str
    requested_model: str | None
    resolved_model: str | None
    response_model: str | None
    status: str
    output_types: tuple[str, ...]
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        """Return the complete, content-free representation safe to print."""

        data = asdict(self)
        data["output_types"] = list(self.output_types)
        return data


class DiagnosticFailure(RuntimeError):
    """A diagnostic failure carrying only content-safe metadata."""

    def __init__(self, code: str, record: DiagnosticRecord | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.record = record


_SAFE_OUTPUT_TYPE = re.compile(r"[a-z][a-z0-9_.:-]{0,47}\Z")
_SAFE_COMPACT_OUTPUT_TYPE = re.compile(r"[a-z][a-z0-9_.:,\[\]-]{0,95}\Z")
_SAFE_MODEL_ID = re.compile(r"[~a-zA-Z0-9][a-zA-Z0-9_.:@+~-]*/[a-zA-Z0-9][a-zA-Z0-9_.:@/+~-]{0,191}\Z")
_RESPONSE_STATUSES = frozenset(
    {
        "cancelled",
        "completed",
        "failed",
        "in_progress",
        "incomplete",
        "queued",
    }
)
_PLACEHOLDER_KEYS = frozenset(
    {
        "changeme",
        "dummy",
        "placeholder",
        "test-key",
        "test-only",
    }
)
_STRICT_JSON_PROMPT = (
    "Return a JSON object with the single property ok set to true. "
    "Do not include any other properties or text."
)
_STRICT_JSON_FORMAT = {
    "type": "json_schema",
    "name": "modal_aso_provider_canary",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"ok": {"type": "boolean", "const": True}},
        "required": ["ok"],
        "additionalProperties": False,
    },
}


def _safe_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in _RESPONSE_STATUSES else "unknown"


def _safe_output_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if _SAFE_OUTPUT_TYPE.fullmatch(text) else "unknown"


def _safe_compact_output_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if _SAFE_COMPACT_OUTPUT_TYPE.fullmatch(text) else "unknown"


def _safe_response_model(value: Any, target: LogicalModelTarget) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and _SAFE_MODEL_ID.fullmatch(value):
        # Aliases such as openrouter/auto and ~provider/model-latest may return
        # the concrete routed model rather than the requested alias. Keep any
        # syntactically safe catalog ID while never exposing arbitrary content.
        return value
    return "unexpected"


def _output_types(response: Mapping[str, Any]) -> tuple[str, ...]:
    output = response.get("output")
    if not isinstance(output, list):
        return ()

    types: list[str] = []
    for item in output[:24]:
        if not isinstance(item, Mapping):
            types.append("unknown")
            continue
        item_type = _safe_output_type(item.get("type"))
        if item_type != "message":
            types.append(item_type)
            continue
        content = item.get("content")
        segment_types: list[str] = []
        if isinstance(content, list):
            for segment in content[:8]:
                if isinstance(segment, Mapping):
                    segment_types.append(_safe_output_type(segment.get("type")))
        if segment_types:
            types.append(f"message[{','.join(segment_types)}]")
        else:
            types.append("message")
    return tuple(types)


def _extract_output_text(response: Mapping[str, Any]) -> str:
    top_level = response.get("output_text")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()

    text_parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") in {"output_text", "text"}:
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for segment in content:
            if not isinstance(segment, Mapping):
                continue
            if segment.get("type") not in {"output_text", "text"}:
                continue
            text = segment.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return "".join(text_parts).strip()


def _validate_strict_json_content(response: Mapping[str, Any]) -> None:
    text = _extract_output_text(response)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        raise DiagnosticFailure("structured_json_validation_failed") from None
    if parsed != {"ok": True}:
        raise DiagnosticFailure("structured_json_validation_failed")


def _validate_openrouter_config(config: ProviderConfig) -> None:
    if config.name != "openrouter":
        raise DiagnosticFailure("openrouter_not_selected")
    for target in DEFAULT_LOGICAL_MODEL_TARGETS:
        try:
            resolved = config.resolve_model(target.logical_model)
        except Exception:
            raise DiagnosticFailure("logical_model_resolution_failed") from None
        expected = config.model_override or target.resolved_model
        if resolved != expected:
            raise DiagnosticFailure("logical_model_mapping_mismatch")


def _configured_target(target: LogicalModelTarget, config: ProviderConfig) -> LogicalModelTarget:
    """Return a diagnostic target after applying the global model override."""

    resolved_model = config.resolve_model(target.logical_model)
    accepted = set(target.accepted_response_models)
    accepted.add(resolved_model)
    return replace(
        target,
        resolved_model=resolved_model,
        accepted_response_models=frozenset(accepted),
    )


def _is_real_credential(value: str | None) -> bool:
    normalized = str(value or "").strip()
    return bool(normalized) and normalized.lower() not in _PLACEHOLDER_KEYS


def live_smoke_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get(SMOKE_FLAG) == "1"


def web_smoke_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get(WEB_SMOKE_FLAG) == "1"


def openrouter_smoke_ready(env: Mapping[str, str] | None = None) -> bool:
    """Return whether every prerequisite for a live OpenRouter call is present."""

    source = os.environ if env is None else env
    provider = str(source.get("ASO_LLM_PROVIDER") or "").strip().lower()
    return (
        provider == "openrouter"
        and live_smoke_enabled(source)
        and _is_real_credential(source.get(OPENROUTER_CREDENTIAL))
    )


def _single_turn_payload(target: LogicalModelTarget, timeout: float) -> dict[str, Any]:
    return {
        "model": target.logical_model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": _STRICT_JSON_PROMPT}],
            }
        ],
        "text": {"format": _STRICT_JSON_FORMAT},
        "max_output_tokens": 256,
        "parallel_tool_calls": False,
        "timeout": timeout,
    }


def _record_for_response(
    *,
    check: str,
    target: LogicalModelTarget,
    response: Mapping[str, Any],
) -> DiagnosticRecord:
    return DiagnosticRecord(
        check=check,
        provider="openrouter",
        requested_model=target.logical_model,
        resolved_model=target.resolved_model,
        response_model=_safe_response_model(response.get("model"), target),
        status=_safe_status(response.get("status")),
        output_types=_output_types(response),
    )


def run_single_turn_check(
    target: LogicalModelTarget,
    config: ProviderConfig,
    *,
    timeout: float = 90.0,
) -> DiagnosticRecord:
    """Call one logical model and validate its body without returning the body."""

    if not live_smoke_enabled():
        raise DiagnosticFailure("live_smoke_opt_in_missing")
    if not _is_real_credential(config.api_key):
        raise DiagnosticFailure("openrouter_credential_missing")
    _validate_openrouter_config(config)
    target = _configured_target(target, config)

    try:
        response = create_response(_single_turn_payload(target, timeout), config)
    except Exception:
        raise DiagnosticFailure("responses_transport_failed") from None

    record = _record_for_response(check="single_turn_json", target=target, response=response)
    if record.status != "completed":
        raise DiagnosticFailure("response_not_completed", record)
    if record.response_model == "unexpected" or record.response_model is None:
        raise DiagnosticFailure("unexpected_response_model", record)
    try:
        _validate_strict_json_content(response)
    except DiagnosticFailure as exc:
        raise DiagnosticFailure(exc.code, record) from None
    if not record.output_types:
        raise DiagnosticFailure("response_output_types_missing", record)
    return record


def run_function_tool_check() -> DiagnosticRecord:
    """Exercise the public Runner after P1 provider continuation is integrated."""

    if not openrouter_smoke_ready():
        raise DiagnosticFailure("live_smoke_prerequisites_missing")

    from agents import Agent, Runner, function_tool

    calls = 0

    @function_tool
    def provider_canary() -> dict[str, bool]:
        """Return the fixed provider-canary result."""

        nonlocal calls
        calls += 1
        return {"ready": True}

    config = get_provider_config()
    target = _configured_target(DEFAULT_LOGICAL_MODEL_TARGETS[1], config)
    agent = Agent(
        name="OpenRouter function-tool canary",
        instructions=(
            "Call provider_canary exactly once. After it returns ready=true, return exactly "
            "the JSON object {\"tool_round_trip\":true} with no other text."
        ),
        tools=[provider_canary],
        model=target.logical_model,
        log_tool_calls=False,
    )
    try:
        result = asyncio.run(Runner.run(agent, "Run the function-tool canary.", max_steps=3))
    except Exception:
        raise DiagnosticFailure("function_tool_transport_failed") from None

    if calls != 1:
        raise DiagnosticFailure("function_tool_call_count_mismatch")
    final_output = result.final_output
    if not isinstance(final_output, str):
        raise DiagnosticFailure("function_tool_final_output_invalid")
    try:
        parsed = json.loads(final_output)
    except (TypeError, ValueError):
        raise DiagnosticFailure("function_tool_final_output_invalid") from None
    if parsed != {"tool_round_trip": True}:
        raise DiagnosticFailure("function_tool_final_output_invalid")

    response_meta: Mapping[str, Any] = {}
    if result.responses and isinstance(result.responses[-1], Mapping):
        candidate = result.responses[-1].get("response")
        if isinstance(candidate, Mapping):
            response_meta = candidate
    return DiagnosticRecord(
        check="function_tool_round_trip",
        provider="openrouter",
        requested_model=target.logical_model,
        resolved_model=target.resolved_model,
        response_model=_safe_response_model(response_meta.get("model"), target),
        status=_safe_status(response_meta.get("status") or "completed"),
        output_types=tuple(
            _safe_compact_output_type(value)
            for value in (response_meta.get("output_types") or [])
            if isinstance(value, str)
        ),
    )


def run_hosted_web_search_check() -> None:
    """Exercise the migrated hosted-search caller after P2 is integrated."""

    if not openrouter_smoke_ready() or not web_smoke_enabled():
        raise DiagnosticFailure("hosted_web_smoke_prerequisites_missing")

    from aso_agents import pipeline_tools

    try:
        result = pipeline_tools._run_answer_web_search_once(
            query="official Ensembl stable transcript identifier documentation",
            request_text="Find one official source about Ensembl stable transcript identifiers.",
            source_policy="primary_sources_only",
            reason="OpenRouter hosted-search canary",
        )
    except Exception:
        raise DiagnosticFailure("hosted_web_search_transport_failed") from None
    if not isinstance(result, Mapping):
        raise DiagnosticFailure("hosted_web_search_result_invalid")
    if result.get("error") or result.get("status") != "ok":
        raise DiagnosticFailure("hosted_web_search_result_invalid")
    evidence = result.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise DiagnosticFailure("hosted_web_search_evidence_missing")


def _skipped_record(target: LogicalModelTarget, error: str) -> DiagnosticRecord:
    return DiagnosticRecord(
        check="single_turn_json",
        provider="openrouter",
        requested_model=target.logical_model,
        resolved_model=target.resolved_model,
        response_model=None,
        status="skipped",
        output_types=(),
        error=error,
    )


def _failed_record(target: LogicalModelTarget, error: str) -> DiagnosticRecord:
    return DiagnosticRecord(
        check="single_turn_json",
        provider="openrouter",
        requested_model=target.logical_model,
        resolved_model=target.resolved_model,
        response_model=None,
        status="failed",
        output_types=(),
        error=error,
    )


def _print_record(record: DiagnosticRecord) -> None:
    print(json.dumps(record.public_dict(), sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MODAL-ASO's configured OpenRouter model mapping with content-safe, "
            "explicitly opted-in Responses calls. OPENROUTER_MODEL, when set, is used "
            "for every logical model."
        ),
        epilog=(
            f"Live calls require ASO_LLM_PROVIDER=openrouter, {OPENROUTER_CREDENTIAL}, "
            f"and {SMOKE_FLAG}=1. This command never prints prompts, response bodies, "
            "credentials, or request headers."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds (default: 90).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1.0 <= args.timeout <= 600.0:
        _print_record(_failed_record(DEFAULT_LOGICAL_MODEL_TARGETS[0], "invalid_timeout"))
        return 2

    try:
        config = get_provider_config(require_credentials=False)
        _validate_openrouter_config(config)
    except Exception:
        for target in DEFAULT_LOGICAL_MODEL_TARGETS:
            _print_record(_skipped_record(target, "openrouter_not_selected_or_invalid"))
        return 2

    if not live_smoke_enabled():
        for target in DEFAULT_LOGICAL_MODEL_TARGETS:
            _print_record(_skipped_record(target, f"set_{SMOKE_FLAG}_to_1"))
        return 0

    if not _is_real_credential(config.api_key):
        for target in DEFAULT_LOGICAL_MODEL_TARGETS:
            _print_record(_skipped_record(target, f"set_{OPENROUTER_CREDENTIAL}"))
        return 2

    failed = False
    for target in DEFAULT_LOGICAL_MODEL_TARGETS:
        active_target = _configured_target(target, config)
        try:
            record = run_single_turn_check(target, config, timeout=args.timeout)
        except DiagnosticFailure as exc:
            failed = True
            record = exc.record or _failed_record(active_target, exc.code)
            if record.error is None:
                record = replace(record, error=exc.code)
        _print_record(record)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
