"""Bounded transport retries and diagnostic metadata, without request content."""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import math
import os
import random
import re
import time
from typing import Any, Mapping


def avoid_failed_provider(
    prepared: dict[str, Any], details: Mapping[str, Any],
) -> list[str]:
    """Move a retry past an identified failed endpoint within an explicit allowlist.

    A response can report a server failure after routing succeeds. A new client
    attempt can select that endpoint again unless it is excluded. Never
    widen a caller's provider pool, override disabled fallbacks, or guess an
    endpoint when its display name does not identify one configured slug.
    """

    extra = prepared.get("extra_body")
    preferences = extra.get("provider") if isinstance(extra, dict) else None
    if not isinstance(preferences, dict) or preferences.get("allow_fallbacks") is False:
        return []
    allowed = preferences.get("only")
    if (
        not isinstance(allowed, list) or len(allowed) < 2
        or not all(isinstance(slug, str) for slug in allowed)
    ):
        return []
    router = details.get("openrouter_metadata") or {}
    endpoints = router.get("endpoints") if isinstance(router, Mapping) else None
    available = endpoints.get("available") if isinstance(endpoints, Mapping) else None
    names = [
        item.get("provider") for item in available
        if isinstance(item, Mapping) and item.get("selected") is True
    ] if isinstance(available, list) else []
    attempts = router.get("attempts") if isinstance(router, Mapping) else None
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], Mapping):
        names.append(attempts[-1].get("provider"))
    provider_error = details.get("provider_error")
    if isinstance(provider_error, Mapping):
        names.append(provider_error.get("provider_name"))
    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    excluded: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not normalized(name):
            continue
        matches = [slug for slug in allowed if normalized(slug) == normalized(name)]
        if len(matches) == 1:
            excluded.add(matches[0])
    remaining = [slug for slug in allowed if slug not in excluded]
    if not excluded or not remaining:
        return []
    preferences["only"] = remaining
    order = preferences.get("order")
    if isinstance(order, list):
        preferences["order"] = [slug for slug in order if slug not in excluded]
        if not preferences["order"]:
            preferences.pop("order")
    return sorted(excluded)


@dataclass(frozen=True)
class TransportRetryPolicy:
    retries: int = 0
    base_seconds: float = 15.0
    max_seconds: float = 120.0
    budget_seconds: float = 900.0

    @classmethod
    def from_env(cls) -> "TransportRetryPolicy":
        policy = cls(
            retries=int(os.environ.get("ASO_LLM_TRANSPORT_RETRIES", "0")),
            base_seconds=float(os.environ.get("ASO_LLM_RETRY_BASE_SECONDS", "15")),
            max_seconds=float(os.environ.get("ASO_LLM_RETRY_MAX_SECONDS", "120")),
            budget_seconds=float(os.environ.get("ASO_LLM_RETRY_BUDGET_SECONDS", "900")),
        )
        if not 0 <= policy.retries <= 20:
            raise ValueError("transport retries must be between 0 and 20")
        if not all(math.isfinite(v) and v > 0 for v in (
            policy.base_seconds, policy.max_seconds, policy.budget_seconds
        )) or policy.base_seconds > policy.max_seconds:
            raise ValueError("transport retry delays must be finite, positive, and ordered")
        return policy

    def delay(self, attempt: int, details: Mapping[str, Any]) -> float:
        headers = details.get("headers", {})
        hint = retry_after_seconds(headers)
        backoff = min(self.max_seconds, self.base_seconds * 2 ** min(attempt, 20))
        # Add only positive jitter: never retry earlier than the server asked.
        return max(backoff, hint or 0.0) + random.uniform(0, min(1.0, backoff / 10))


def retry_after_seconds(headers: Mapping[str, Any]) -> float | None:
    """Support delta seconds, HTTP dates, and millisecond hints, including >60s."""

    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            result = float(value) * scale
        except (TypeError, ValueError):
            if name != "retry-after":
                continue
            try:
                result = parsedate_to_datetime(str(value)).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(result) and result >= 0:
            return result
    return None


def transport_error_details(exc: Exception) -> dict[str, Any]:
    """Allowlist diagnostic fields; exclude bodies/messages that can echo prompts."""

    from aso_agents.evals.telemetry import _safe_openrouter_metadata, _safe_value

    details: dict[str, Any] = {}
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        details["status_code"] = status
    body = getattr(exc, "body", None)
    envelope = None
    if response is not None:
        wanted = {
            "retry-after", "retry-after-ms", "x-generation-id", "x-request-id",
            "request-id", "x-should-retry", "date",
        }
        details["headers"] = {
            k.lower(): str(v) for k, v in response.headers.items()
            if k.lower() in wanted or k.lower().startswith("x-ratelimit-")
        }
        try:
            envelope = response.json()
        except Exception:
            pass
    if isinstance(envelope, Mapping):
        body = envelope.get("error", body)
        router = _safe_openrouter_metadata(envelope.get("openrouter_metadata"))
        if router:
            details["openrouter_metadata"] = router
    if isinstance(body, Mapping):
        for key in ("code", "type"):
            if isinstance(body.get(key), (str, int)):
                details["error_" + key] = body[key]
        metadata = body.get("metadata")
        if isinstance(metadata, Mapping):
            details["provider_error"] = {
                key: metadata[key] for key in (
                    "provider_name", "provider_error_code", "provider_code",
                    "limit_source", "error_type", "is_byok"
                ) if isinstance(metadata.get(key), (str, int, bool))
            }
    return _safe_value(details)


def retryable_transport_error(exc: Exception, details: Mapping[str, Any]) -> bool:
    if details.get("headers", {}).get("x-should-retry") == "false":
        return False
    return details.get("status_code") in {408, 409, 429, 500, 502, 503, 504} or (
        type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
    )
