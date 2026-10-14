"""Opt-in, evaluation-scoped telemetry for live LLM calls.

The collector intentionally stores a small, JSON-safe projection of each call.
It never stores request prompts, credentials, raw reasoning, or encrypted
reasoning payloads.  Product runs pay only a context-variable lookup when no
evaluation trace is active.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping


_REDACTED = "[REDACTED]"
_TRUNCATED = "... <truncated>"
_MAX_STRING_LENGTH = 8_000
_MAX_COLLECTION_ITEMS = 100
_MAX_DEPTH = 8
_SECRET_VALUE = re.compile(
    r"(?i)(?:bearer\s+[-._~+/=a-z0-9]{8,}|sk-(?:or-)?[a-z0-9_-]{8,})"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|password|secret)"
    r"\s*[:=]\s*[\"']?[^\s\"',;}]+"
)
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "credential",
        "credentials",
        "reasoning",
        "reasoning_content",
        "chain_of_thought",
        "encrypted_content",
        "encrypted_reasoning",
    }
)


@dataclass
class EvaluationTrace:
    """In-memory trace for one evaluation scope."""

    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    calls: list[dict[str, Any]] = field(default_factory=list)
    resource_acquisitions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return the ordered event list consumed by eval manifests."""

        return self.calls

    def snapshot(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable representation."""

        return {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "metadata": copy.deepcopy(self.metadata),
            "calls": copy.deepcopy(self.calls),
            "resource_acquisitions": copy.deepcopy(self.resource_acquisitions),
        }

    def summary(self) -> dict[str, Any]:
        """Aggregate usage, cost, and latency without retaining call outputs."""

        usage_totals: dict[str, int | float] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
        }
        total_latency_ms = 0.0
        call_latencies_ms: list[float] = []
        statuses: dict[str, int] = {}
        providers: set[str] = set()
        upstream_providers: set[str] = set()
        models: set[str] = set()
        requested_reasoning_efforts: set[str] = set()
        for call in self.calls:
            status = str(call.get("status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
            if call.get("provider"):
                providers.add(str(call["provider"]))
            router_metadata = call.get("openrouter_metadata")
            endpoints = (
                router_metadata.get("endpoints")
                if isinstance(router_metadata, Mapping)
                else None
            )
            available = endpoints.get("available") if isinstance(endpoints, Mapping) else None
            for endpoint in available if isinstance(available, list) else []:
                if (
                    isinstance(endpoint, Mapping)
                    and endpoint.get("selected") is True
                    and endpoint.get("provider")
                ):
                    upstream_providers.add(str(endpoint["provider"]))
            if call.get("response_model") or call.get("resolved_model"):
                models.add(str(call.get("response_model") or call.get("resolved_model")))
            if call.get("requested_reasoning_effort"):
                requested_reasoning_efforts.add(
                    str(call["requested_reasoning_effort"])
                )
            latency = call.get("latency_ms")
            if isinstance(latency, (int, float)) and not isinstance(latency, bool):
                numeric_latency = float(latency)
                total_latency_ms += numeric_latency
                call_latencies_ms.append(numeric_latency)
            usage = call.get("usage")
            if not isinstance(usage, Mapping):
                continue
            for key in usage_totals:
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage_totals[key] += value
        resource_wait_ms_by_resource: dict[str, float] = {}
        resource_wait_ms = 0.0
        max_resource_wait_ms = 0.0
        for acquisition in self.resource_acquisitions:
            resource = str(acquisition.get("resource") or "unknown")
            wait_ms = acquisition.get("wait_ms")
            if not isinstance(wait_ms, (int, float)) or isinstance(wait_ms, bool):
                continue
            numeric_wait_ms = max(0.0, float(wait_ms))
            resource_wait_ms += numeric_wait_ms
            max_resource_wait_ms = max(max_resource_wait_ms, numeric_wait_ms)
            resource_wait_ms_by_resource[resource] = (
                resource_wait_ms_by_resource.get(resource, 0.0) + numeric_wait_ms
            )
        return {
            "call_count": len(self.calls),
            "statuses": dict(sorted(statuses.items())),
            "providers": sorted(providers),
            "upstream_providers": sorted(upstream_providers),
            "models": sorted(models),
            "requested_reasoning_efforts": sorted(requested_reasoning_efforts),
            "usage": usage_totals,
            "total_latency_ms": round(total_latency_ms, 3),
            "mean_call_latency_ms": round(
                total_latency_ms / len(call_latencies_ms), 3
            )
            if call_latencies_ms
            else 0.0,
            "max_call_latency_ms": round(max(call_latencies_ms), 3)
            if call_latencies_ms
            else 0.0,
            "resource_acquisition_count": len(self.resource_acquisitions),
            "total_resource_wait_ms": round(resource_wait_ms, 3),
            "max_resource_wait_ms": round(max_resource_wait_ms, 3),
            "resource_wait_ms_by_resource": {
                resource: round(wait_ms, 3)
                for resource, wait_ms in sorted(
                    resource_wait_ms_by_resource.items()
                )
            },
        }


@dataclass(frozen=True)
class PendingLLMCall:
    """Private handle connecting transport completion to one trace call."""

    trace: EvaluationTrace
    call_id: str
    span_id: str
    started_at: float
    attribution: dict[str, str | None]


_CURRENT_TRACE: ContextVar[EvaluationTrace | None] = ContextVar(
    "aso_evaluation_trace",
    default=None,
)


@contextmanager
def evaluation_trace(
    metadata: Mapping[str, Any] | None = None,
    *,
    trace_id: str | None = None,
) -> Iterator[EvaluationTrace]:
    """Capture LLM telemetry within the current context only.

    Nested scopes are independent and restore the surrounding scope on exit.
    Metadata is sanitized at entry so callers cannot accidentally persist a
    credential in the eventual run manifest.
    """

    safe_metadata = _safe_value(dict(metadata or {}))
    if not isinstance(safe_metadata, dict):
        safe_metadata = {}
    trace = EvaluationTrace(
        metadata=safe_metadata,
        trace_id=_safe_identifier(trace_id) or uuid.uuid4().hex,
    )
    token: Token[EvaluationTrace | None] = _CURRENT_TRACE.set(trace)
    try:
        yield trace
    finally:
        _CURRENT_TRACE.reset(token)


# A descriptive alias for callers that prefer an explicit verb.
capture_evaluation_trace = evaluation_trace


@contextmanager
def capture_eval_trace(
    *,
    eval_run_id: str,
    case_id: str,
    output_path: str | Path | None = None,
) -> Iterator[EvaluationTrace]:
    """Stable executor-facing trace API, with optional JSON persistence."""

    with evaluation_trace(
        {"eval_run_id": eval_run_id, "case_id": case_id},
    ) as trace:
        try:
            yield trace
        finally:
            if output_path is not None:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = trace.snapshot()
                payload["summary"] = trace.summary()
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )


def current_evaluation_trace() -> EvaluationTrace | None:
    """Return the active trace, if evaluation capture is enabled."""

    return _CURRENT_TRACE.get()


def current_eval_trace() -> EvaluationTrace | None:
    """Stable executor-facing alias for the active evaluation trace."""

    return current_evaluation_trace()


def record_resource_acquisition(acquisition: Mapping[str, Any]) -> None:
    """Record resource admission delay without retaining caller inputs."""

    trace = _CURRENT_TRACE.get()
    if trace is None:
        return
    resource = _safe_identifier(acquisition.get("resource"))
    wait_seconds = acquisition.get("wait_seconds")
    if not resource or not isinstance(wait_seconds, (int, float)):
        return
    if isinstance(wait_seconds, bool) or not math.isfinite(float(wait_seconds)):
        return
    trace.resource_acquisitions.append(
        {
            "resource": resource,
            "slot": _first_integer(acquisition.get("slot")),
            "limit": _first_integer(acquisition.get("limit")),
            "wait_ms": round(max(0.0, float(wait_seconds)) * 1000, 3),
        }
    )


def begin_llm_call(attribution: Mapping[str, Any]) -> PendingLLMCall | None:
    """Create a pending call only when evaluation capture is active."""

    trace = _CURRENT_TRACE.get()
    if trace is None:
        return None
    normalized_attribution = {
        "provider": _safe_text(attribution.get("provider")),
        "requested_model": _safe_text(attribution.get("requested_model")),
        "resolved_model": _safe_text(attribution.get("resolved_model")),
    }
    pending = PendingLLMCall(
        trace=trace,
        call_id=uuid.uuid4().hex,
        span_id=uuid.uuid4().hex,
        started_at=time.perf_counter(),
        attribution=normalized_attribution,
    )
    _record_live_event(pending, {**_base_call_record(pending), "status": "started"})
    return pending


def _record_live_event(pending: PendingLLMCall, record: Mapping[str, Any]) -> None:
    """Append a compact event immediately, including before a stage completes."""

    configured = os.environ.get("ASO_EVAL_LIVE_EVENTS", "").strip()
    if not configured:
        return
    path = Path(configured)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid duplicating generated output into the supervision journal.
    event = {key: value for key, value in record.items() if key != "parsed_output"}
    event.update({
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "trace_id": pending.trace.trace_id,
        "scope": pending.trace.metadata,
    })
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(_safe_value(event), separators=(",", ":")) + "\n")
        handle.flush()


def instrument_openrouter_request(
    prepared: MutableMapping[str, Any],
    pending: PendingLLMCall | None,
) -> None:
    """Opt in to OpenRouter routing metadata for one evaluation call."""

    if pending is None or pending.attribution.get("provider") != "openrouter":
        return

    extra_headers_value = prepared.get("extra_headers")
    if isinstance(extra_headers_value, Mapping):
        extra_headers = dict(copy.deepcopy(extra_headers_value))
    else:
        extra_headers = {}
    extra_headers.setdefault("X-OpenRouter-Metadata", "enabled")
    prepared["extra_headers"] = extra_headers

    extra_body_value = prepared.get("extra_body")
    if isinstance(extra_body_value, Mapping):
        extra_body = dict(copy.deepcopy(extra_body_value))
    else:
        extra_body = {}
    trace_value = extra_body.get("trace")
    trace_fields = dict(copy.deepcopy(trace_value)) if isinstance(trace_value, Mapping) else {}
    trace_fields.setdefault("trace_id", pending.trace.trace_id)
    trace_fields.setdefault("trace_name", _trace_name(pending.trace))
    trace_fields.setdefault("generation_name", "modal-aso-eval-llm-call")
    trace_fields.setdefault("span_name", pending.call_id)
    parent_span_id = _safe_identifier(pending.trace.metadata.get("parent_span_id"))
    if parent_span_id:
        trace_fields.setdefault("parent_span_id", parent_span_id)
    extra_body["trace"] = trace_fields
    prepared["extra_body"] = extra_body

    reasoning_value = prepared.get("reasoning")
    if not isinstance(reasoning_value, Mapping):
        reasoning_value = extra_body.get("reasoning")
    if isinstance(reasoning_value, Mapping):
        requested_effort = _safe_text(reasoning_value.get("effort"))
        if requested_effort:
            pending.attribution["requested_reasoning_effort"] = requested_effort
        enabled = reasoning_value.get("enabled")
        if isinstance(enabled, bool):
            pending.attribution["requested_reasoning_enabled"] = enabled


def complete_llm_call(
    pending: PendingLLMCall | None,
    response: Mapping[str, Any],
) -> None:
    """Record a successfully parsed transport response."""

    if pending is None:
        return
    body = _response_body(response)
    record = _base_call_record(pending)
    response_status = _safe_text(body.get("status")) or "completed"
    record.update(
        {
            "response_id": _safe_text(body.get("id")),
            "status": response_status,
            "response_model": _safe_text(body.get("model")),
            "output_types": _output_types(body),
            "usage": _normalized_usage(body),
            "latency_ms": _elapsed_ms(pending),
            "parsed_output": _safe_parsed_output(body),
        }
    )
    if response_status == "failed":
        error = body.get("error")
        if isinstance(error, Mapping):
            record["error_type"] = (
                _safe_text(error.get("type"))
                or _safe_text(error.get("code"))
                or "provider_response_failed"
            )
            error_code = _safe_text(error.get("code"))
            if error_code:
                record["error_code"] = error_code
        else:
            record["error_type"] = "provider_response_failed"
    router_metadata = _safe_openrouter_metadata(body.get("openrouter_metadata"))
    if router_metadata:
        record["openrouter_metadata"] = router_metadata
    pending.trace.calls.append(record)
    _record_live_event(pending, record)


def fail_llm_call(
    pending: PendingLLMCall | None,
    *,
    status: str,
    error_type: str,
    transport_details: Mapping[str, Any] | None = None,
    response: Mapping[str, Any] | None = None,
) -> None:
    """Record a failed call and allowlisted metadata, excluding raw messages."""

    if pending is None:
        return
    record = _base_call_record(pending)
    record.update(
        {
            "response_id": None,
            "status": _safe_text(status) or "error",
            "response_model": None,
            "output_types": [],
            "usage": _empty_usage(),
            "latency_ms": _elapsed_ms(pending),
            "parsed_output": {"text": "", "tool_calls": []},
            "error_type": _safe_text(error_type),
        }
    )
    if transport_details:
        record["transport_details"] = _safe_value(dict(transport_details))
    if response is not None:
        body = _response_body(response)
        record.update({
            "response_id": _safe_text(body.get("id")),
            "response_model": _safe_text(body.get("model")),
            "usage": _normalized_usage(body),
        })
        router = _safe_openrouter_metadata(body.get("openrouter_metadata"))
        if router:
            record["openrouter_metadata"] = router
        error = body.get("error")
        if isinstance(error, Mapping) and _safe_text(error.get("code")):
            record["error_code"] = _safe_text(error["code"])
    pending.trace.calls.append(record)
    _record_live_event(pending, record)


def record_tool_execution(
    *,
    agent: str | None,
    tool: str | None,
    call_id: str | None,
    arguments: Any,
    result: Any,
) -> None:
    """Attach one sanitized local tool result to its originating LLM call."""

    trace = _CURRENT_TRACE.get()
    if trace is None:
        return
    safe_call_id = _safe_text(call_id)
    execution = {
        "agent": _safe_text(agent),
        "tool": _safe_text(tool),
        "call_id": safe_call_id,
        "arguments": _safe_value(arguments),
        "result": _safe_value(result),
    }
    for record in reversed(trace.calls):
        parsed = record.get("parsed_output")
        calls = parsed.get("tool_calls") if isinstance(parsed, Mapping) else None
        if safe_call_id and isinstance(calls, list) and any(
            isinstance(item, Mapping) and item.get("call_id") == safe_call_id
            for item in calls
        ):
            record.setdefault("tool_executions", []).append(execution)
            return
    # Preserve diagnostic value if a provider omitted or rewrote its call ID.
    if trace.calls:
        trace.calls[-1].setdefault("tool_executions", []).append(execution)


def _base_call_record(pending: PendingLLMCall) -> dict[str, Any]:
    return {
        "type": "llm_call",
        "call_id": pending.call_id,
        "span_id": pending.span_id,
        **pending.attribution,
    }


def _response_body(response: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = response.get("response")
    if not isinstance(response.get("output"), list) and isinstance(nested, Mapping):
        return nested
    return response


def _output_items(body: Mapping[str, Any]) -> list[Any]:
    output = body.get("output")
    return output if isinstance(output, list) else []


def _output_types(body: Mapping[str, Any]) -> list[str]:
    types: list[str] = []
    for item in _output_items(body):
        if not isinstance(item, Mapping):
            continue
        item_type = _safe_text(item.get("type")) or "unknown"
        if item_type in {"reasoning", "thinking"}:
            continue
        types.append(item_type)
    return types[:24]


def _safe_parsed_output(body: Mapping[str, Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in _output_items(body):
        if not isinstance(item, Mapping):
            continue
        item_type = _safe_text(item.get("type"))
        if item_type == "message":
            content = item.get("content")
            if isinstance(content, list):
                for segment in content:
                    if not isinstance(segment, Mapping):
                        continue
                    segment_type = _safe_text(segment.get("type"))
                    if segment_type in {"output_text", "text"}:
                        text_parts.append(_safe_string(segment.get("text")))
        elif item_type in {"output_text", "text"}:
            text_parts.append(_safe_string(item.get("text")))
        elif item_type in {"function_call", "tool_call"}:
            raw_arguments = item.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    parsed_arguments: Any = json.loads(raw_arguments)
                except (TypeError, ValueError):
                    parsed_arguments = raw_arguments
            else:
                parsed_arguments = raw_arguments
            tool_calls.append(
                {
                    "id": _safe_text(item.get("id")),
                    "call_id": _safe_text(item.get("call_id") or item.get("id")),
                    "name": _safe_text(item.get("name")),
                    "arguments": _safe_value(parsed_arguments),
                }
            )
    return {
        "text": _safe_string("".join(text_parts).strip()),
        "tool_calls": tool_calls,
    }


def _normalized_usage(body: Mapping[str, Any]) -> dict[str, int | float | None]:
    usage_value = body.get("usage")
    usage = usage_value if isinstance(usage_value, Mapping) else {}
    input_details = _first_mapping(
        usage.get("input_tokens_details"), usage.get("prompt_tokens_details")
    )
    output_details = _first_mapping(
        usage.get("output_tokens_details"), usage.get("completion_tokens_details")
    )
    input_tokens = _first_integer(usage.get("input_tokens"), usage.get("prompt_tokens"))
    output_tokens = _first_integer(
        usage.get("output_tokens"), usage.get("completion_tokens")
    )
    total_tokens = _first_integer(
        usage.get("total_tokens"), input_tokens + output_tokens
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": _first_integer(
            input_details.get("cached_tokens"),
            usage.get("cached_tokens"),
            usage.get("cache_read_input_tokens"),
        ),
        "cache_write_tokens": _first_integer(
            input_details.get("cache_write_tokens"),
            usage.get("cache_write_tokens"),
            usage.get("cache_creation_input_tokens"),
        ),
        "reasoning_tokens": _first_integer(
            output_details.get("reasoning_tokens"), usage.get("reasoning_tokens")
        ),
        "cost": _first_number(usage.get("cost"), body.get("cost")),
    }


def _empty_usage() -> dict[str, int | float | None]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "cost": None,
    }


def _safe_openrouter_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("requested", "strategy", "region", "summary", "attempt", "is_byok"):
        scalar = value.get(key)
        if isinstance(scalar, (str, int, float, bool)) or scalar is None:
            result[key] = _safe_value(scalar)

    endpoints = value.get("endpoints")
    if isinstance(endpoints, Mapping):
        safe_endpoints: dict[str, Any] = {}
        total = endpoints.get("total")
        if isinstance(total, int) and not isinstance(total, bool):
            safe_endpoints["total"] = total
        available = endpoints.get("available")
        if isinstance(available, list):
            safe_endpoints["available"] = [
                _selected_fields(item, ("provider", "model", "selected"))
                for item in available[:_MAX_COLLECTION_ITEMS]
                if isinstance(item, Mapping)
            ]
        if safe_endpoints:
            result["endpoints"] = safe_endpoints

    attempts = value.get("attempts")
    if isinstance(attempts, list):
        result["attempts"] = [
            _selected_fields(item, ("provider", "model", "status"))
            for item in attempts[:_MAX_COLLECTION_ITEMS]
            if isinstance(item, Mapping)
        ]
    pipeline = value.get("pipeline")
    if isinstance(pipeline, list):
        result["pipeline"] = [
            _selected_fields(item, ("type", "name", "status"))
            for item in pipeline[:_MAX_COLLECTION_ITEMS]
            if isinstance(item, Mapping)
        ]
    return result


def _selected_fields(value: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: _safe_value(value[key])
        for key in keys
        if key in value and isinstance(value[key], (str, int, float, bool, type(None)))
    }


def _safe_value(value: Any, *, _depth: int = 0) -> Any:
    if _depth >= _MAX_DEPTH:
        return _TRUNCATED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _safe_string(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                result["__truncated__"] = True
                break
            key = str(raw_key)
            normalized_key = key.strip().lower().replace("-", "_")
            if normalized_key in _SENSITIVE_KEYS:
                result[key] = _REDACTED
            else:
                result[key] = _safe_value(item, _depth=_depth + 1)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        safe_items = [
            _safe_value(item, _depth=_depth + 1)
            for item in items[:_MAX_COLLECTION_ITEMS]
        ]
        if len(items) > _MAX_COLLECTION_ITEMS:
            safe_items.append(_TRUNCATED)
        return safe_items
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _safe_value(method(), _depth=_depth + 1)
            except Exception:
                continue
    return _safe_string(str(value))


def _safe_string(value: Any) -> str:
    text = str(value or "")
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}={_REDACTED}", text)
    text = _SECRET_VALUE.sub(_REDACTED, text)
    if len(text) > _MAX_STRING_LENGTH:
        return text[:_MAX_STRING_LENGTH] + _TRUNCATED
    return text


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    return _safe_string(value)


def _safe_identifier(value: Any) -> str | None:
    text = _safe_text(value)
    if not text or text == _REDACTED:
        return None
    return re.sub(r"[^A-Za-z0-9_.:-]", "-", text)[:128] or None


def _trace_name(trace: EvaluationTrace) -> str:
    for key in ("trace_name", "eval_run_id", "eval_id", "run_id", "case_id"):
        name = _safe_identifier(trace.metadata.get(key))
        if name:
            return name
    return "modal-aso-eval"


def _first_mapping(*values: Any) -> Mapping[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return {}


def _first_integer(*values: Any) -> int:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
        if isinstance(value, float) and math.isfinite(value):
            return max(0, int(value))
    return 0


def _first_number(*values: Any) -> int | float | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                return value
    return None


def _elapsed_ms(pending: PendingLLMCall) -> float:
    return round(max(0.0, (time.perf_counter() - pending.started_at) * 1_000), 3)


__all__ = [
    "EvaluationTrace",
    "capture_eval_trace",
    "capture_evaluation_trace",
    "current_eval_trace",
    "current_evaluation_trace",
    "evaluation_trace",
]
