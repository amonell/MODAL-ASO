"""Provider-neutral configuration and transport for LLM Responses calls."""

from __future__ import annotations

import copy
import hmac
import json
import math
import os
import re
import secrets
import threading
import time
from contextlib import nullcontext
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from openai import OpenAI

from aso_agents.resource_limits import acquire_resource, defer_resource, scoped_resource_name
from aso_agents.llm_transport import (
    TransportRetryPolicy,
    avoid_failed_provider,
    retryable_transport_error,
    transport_error_details,
)


_OPENAI_BASE_URL = "https://api.openai.com/v1"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Agentic OpenRouter runs make several large, serial Responses calls and can
# briefly cross an upstream rolling rate window.  The SDK backs retries off to
# eight seconds, so twelve attempts cover a full minute-scale reset window.
_OPENROUTER_MAX_RETRIES = 12
_OPENROUTER_DEFAULT_PROVIDER_SORT = "throughput"
_OPENROUTER_PROVIDER_SORTS = frozenset(
    {"balanced", "price", "throughput", "latency"}
)
_OPENROUTER_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_OPENROUTER_STRUCTURED_OUTPUT_MODES = frozenset({"json_object", "json_schema"})
_OPENROUTER_DEFAULT_STRUCTURED_OUTPUT_MODE = "json_object"
_OPENROUTER_MODEL_ID = re.compile(
    r"[~@]?[A-Za-z0-9][A-Za-z0-9_.:@+~-]*/[A-Za-z0-9][A-Za-z0-9_.:@/+~-]*\Z"
)

_MODEL_NAMES: dict[str, dict[str, str]] = {
    "openai": {
        "gpt-5.2-2025-12-11": "gpt-5.2-2025-12-11",
        "gpt-5-mini-2025-08-07": "gpt-5-mini-2025-08-07",
    },
    "openrouter": {
        "gpt-5.2-2025-12-11": "openai/gpt-5.2",
        "gpt-5-mini-2025-08-07": "openai/gpt-5-mini",
    },
}

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

_CREDENTIAL_FINGERPRINT_KEY = secrets.token_bytes(32)
_ClientCacheKey = tuple[
    str,
    str,
    tuple[tuple[str, str], ...],
    float | None,
    int | None,
    bytes,
]
_CLIENT_CACHE: dict[_ClientCacheKey, OpenAI] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved settings for one supported LLM provider."""

    name: Literal["openai", "openrouter"]
    api_key: str = field(repr=False)
    base_url: str
    default_headers: dict[str, str]
    continuation_mode: Literal["previous_response_id", "stateless"]
    provider_preferences: dict[str, Any]
    model_override: str | None = None
    reasoning_effort_override: str | None = None
    max_output_tokens_cap: int | None = None
    request_timeout_seconds: float | None = None
    max_retries: int | None = None
    structured_output_mode: Literal["json_object", "json_schema"] | None = None

    def resolve_model(self, logical_model: str) -> str:
        """Resolve a logical model name to the selected provider's request ID."""

        if self.name not in _MODEL_NAMES:
            raise ValueError(
                "LLM model resolution failed: unsupported provider "
                f"{self.name!r}; expected 'openai' or 'openrouter'."
            )
        if not isinstance(logical_model, str) or not logical_model.strip():
            raise ValueError(
                f"LLM model resolution failed for provider '{self.name}': "
                "the logical model must be a non-empty string."
            )

        requested_model = logical_model.strip()
        if self.name == "openrouter" and self.model_override is not None:
            return self.model_override

        registered_model = _MODEL_NAMES[self.name].get(requested_model)
        if registered_model is not None:
            return registered_model

        if self.name == "openrouter":
            if "/" in requested_model:
                return requested_model
            raise ValueError(
                "LLM model resolution failed for provider 'openrouter': unknown "
                f"unqualified model {requested_model!r}; use a provider-qualified "
                "slug or a registered logical model ID."
            )

        if "/" in requested_model:
            raise ValueError(
                "LLM model resolution failed for provider 'openai': "
                f"provider-qualified model {requested_model!r} is not supported."
            )
        return requested_model


def selected_provider_name(env: Mapping[str, str] | None = None) -> str:
    """Return the normalized selected provider name."""

    source = os.environ if env is None else env
    raw_name = source.get("ASO_LLM_PROVIDER")
    if raw_name is None:
        return "openai"
    normalized_name = raw_name.strip().lower()
    if normalized_name not in {"openai", "openrouter"}:
        raise ValueError(
            "LLM provider configuration failed: ASO_LLM_PROVIDER must be "
            f"'openai' or 'openrouter', not {raw_name!r}."
        )
    return normalized_name


def provider_credential_help(env: Mapping[str, str] | None = None) -> str:
    """Return focused credential setup help for the selected provider."""

    provider_name = selected_provider_name(env)
    credential_name = _credential_variable(provider_name)
    return f"Set {credential_name} for ASO_LLM_PROVIDER={provider_name}."


def provider_credentials_available(env: Mapping[str, str] | None = None) -> bool:
    """Report whether the selected provider has a non-empty credential."""

    config = get_provider_config(env, require_credentials=False)
    return bool(config.api_key.strip())


def get_provider_config(
    env: Mapping[str, str] | None = None,
    *,
    require_credentials: bool = True,
) -> ProviderConfig:
    """Resolve provider configuration from an environment mapping."""

    source = os.environ if env is None else env
    provider_name = selected_provider_name(source)
    credential_name = _credential_variable(provider_name)
    api_key = (source.get(credential_name) or "").strip()
    if require_credentials and not api_key:
        raise RuntimeError(
            f"LLM provider configuration failed for '{provider_name}': "
            f"{provider_credential_help(source)}"
        )

    if provider_name == "openai":
        return ProviderConfig(
            name="openai",
            api_key=api_key,
            base_url=_env_value_or_default(source, "OPENAI_BASE_URL", _OPENAI_BASE_URL),
            default_headers={},
            continuation_mode="previous_response_id",
            provider_preferences={},
        )

    headers: dict[str, str] = {}
    referer = (source.get("OPENROUTER_HTTP_REFERER") or "").strip()
    if referer:
        headers["HTTP-Referer"] = referer

    app_title = _env_value_or_default(source, "OPENROUTER_APP_TITLE", "MODAL-ASO")
    headers["X-OpenRouter-Title"] = app_title

    provider_preferences: dict[str, Any] = {}
    provider_sort = _env_value_or_default(
        source,
        "OPENROUTER_PROVIDER_SORT",
        _OPENROUTER_DEFAULT_PROVIDER_SORT,
    ).lower()
    if provider_sort not in _OPENROUTER_PROVIDER_SORTS:
        raise ValueError(
            "LLM provider configuration failed for 'openrouter': "
            "OPENROUTER_PROVIDER_SORT must be 'balanced', 'price', "
            "'throughput', or 'latency'."
        )
    if provider_sort != "balanced":
        provider_preferences["sort"] = provider_sort

    zdr = _parse_optional_boolean(source, "OPENROUTER_ZDR")
    if zdr:
        provider_preferences["zdr"] = True

    data_collection = (source.get("OPENROUTER_DATA_COLLECTION") or "").strip().lower()
    if data_collection:
        if data_collection not in {"allow", "deny"}:
            raise ValueError(
                "LLM provider configuration failed for 'openrouter': "
                "OPENROUTER_DATA_COLLECTION must be 'allow' or 'deny'."
            )
        provider_preferences["data_collection"] = data_collection

    only_providers = (source.get("OPENROUTER_ONLY_PROVIDERS") or "").strip()
    if only_providers:
        provider_slugs = [slug.strip() for slug in only_providers.split(",") if slug.strip()]
        if not provider_slugs:
            raise ValueError(
                "LLM provider configuration failed for 'openrouter': "
                "OPENROUTER_ONLY_PROVIDERS must contain at least one provider slug."
            )
        provider_preferences["only"] = provider_slugs

    provider_order = (source.get("OPENROUTER_PROVIDER_ORDER") or "").strip()
    if provider_order:
        ordered_slugs = [
            slug.strip() for slug in provider_order.split(",") if slug.strip()
        ]
        if not ordered_slugs:
            raise ValueError(
                "LLM provider configuration failed for 'openrouter': "
                "OPENROUTER_PROVIDER_ORDER must contain at least one provider slug."
            )
        provider_preferences["order"] = ordered_slugs
        provider_preferences["allow_fallbacks"] = len(ordered_slugs) > 1

    model_override = (source.get("OPENROUTER_MODEL") or "").strip() or None
    if model_override is not None and not _OPENROUTER_MODEL_ID.fullmatch(model_override):
        raise ValueError(
            "LLM provider configuration failed for 'openrouter': "
            "OPENROUTER_MODEL must be an exact provider/model slug or qualified "
            "OpenRouter alias from https://openrouter.ai/models."
        )

    reasoning_effort = (
        source.get("OPENROUTER_REASONING_EFFORT") or ""
    ).strip().lower() or None
    if (
        reasoning_effort is not None
        and reasoning_effort not in _OPENROUTER_REASONING_EFFORTS
    ):
        raise ValueError(
            "LLM provider configuration failed for 'openrouter': "
            "OPENROUTER_REASONING_EFFORT must be 'none', 'minimal', 'low', "
            "'medium', 'high', 'xhigh', or 'max'."
        )

    max_output_tokens_cap = _parse_optional_integer(
        source,
        "OPENROUTER_MAX_OUTPUT_TOKENS",
        minimum=1,
    )
    request_timeout_seconds = _parse_optional_float(
        source,
        "OPENROUTER_TIMEOUT_SECONDS",
        minimum_exclusive=0,
    )
    max_retries = _parse_optional_integer(
        source,
        "OPENROUTER_MAX_RETRIES",
        minimum=0,
    )
    structured_output_mode = _env_value_or_default(
        source,
        "OPENROUTER_STRUCTURED_OUTPUT_MODE",
        _OPENROUTER_DEFAULT_STRUCTURED_OUTPUT_MODE,
    ).lower()
    if structured_output_mode not in _OPENROUTER_STRUCTURED_OUTPUT_MODES:
        raise ValueError(
            "LLM provider configuration failed for 'openrouter': "
            "OPENROUTER_STRUCTURED_OUTPUT_MODE must be 'json_object' or "
            "'json_schema'."
        )

    return ProviderConfig(
        name="openrouter",
        api_key=api_key,
        base_url=_env_value_or_default(source, "OPENROUTER_BASE_URL", _OPENROUTER_BASE_URL),
        default_headers=headers,
        continuation_mode="stateless",
        provider_preferences=provider_preferences,
        model_override=model_override,
        reasoning_effort_override=reasoning_effort,
        max_output_tokens_cap=max_output_tokens_cap,
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
        structured_output_mode=structured_output_mode,
    )


def build_hosted_web_search_tool(
    config: ProviderConfig,
    *,
    max_uses: int = 3,
    max_results: int | None = None,
) -> dict[str, Any]:
    """Build the selected provider's hosted web-search tool definition."""

    _require_positive_integer(max_uses, "max_uses")
    if max_results is not None:
        _require_positive_integer(max_results, "max_results")

    if config.name == "openai":
        return {"type": "web_search"}
    if config.name != "openrouter":
        raise ValueError(
            "Hosted web-search tool construction failed: unsupported provider "
            f"{config.name!r}."
        )

    parameters: dict[str, int] = {"max_uses": max_uses}
    if max_results is not None:
        parameters["max_results"] = max_results
    return {"type": "openrouter:web_search", "parameters": parameters}


def prepare_responses_payload(
    payload: Mapping[str, Any],
    config: ProviderConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy and translate one Responses request for the selected provider."""

    if not isinstance(payload, Mapping):
        raise TypeError("LLM request preparation failed: payload must be a mapping.")

    prepared = dict(copy.deepcopy(payload))
    requested_model = prepared.get("model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        raise ValueError(
            f"LLM request preparation failed for provider '{config.name}': "
            "payload.model must be a non-empty string."
        )

    requested_model = requested_model.strip()
    try:
        resolved_model = config.resolve_model(requested_model)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"LLM request preparation failed for provider '{config.name}' and "
            f"logical model {requested_model!r}: {exc}"
        ) from None
    prepared["model"] = resolved_model
    attribution = {
        "provider": config.name,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
    }

    text_config = prepared.get("text")
    text_format = (
        text_config.get("format")
        if isinstance(text_config, Mapping)
        and isinstance(text_config.get("format"), Mapping)
        else {}
    )
    requested_format_type = str(text_format.get("type") or "").strip().lower()
    canonical_output_schema: dict[str, Any] | None = None
    output_schema_name = str(text_format.get("name") or "response").strip()
    if requested_format_type == "json_schema":
        schema = text_format.get("schema")
        if not isinstance(schema, Mapping):
            raise TypeError(
                "LLM request preparation failed: json_schema format requires a schema mapping."
            )
        canonical_output_schema = dict(copy.deepcopy(schema))
        from aso_agents.structured_outputs import portable_strict_json_schema

        compiled_text = dict(copy.deepcopy(text_config))
        compiled_format = dict(copy.deepcopy(text_format))
        compiled_format["schema"] = portable_strict_json_schema(dict(schema))
        compiled_text["format"] = compiled_format
        prepared["text"] = compiled_text

    if config.name == "openai":
        return prepared, attribution
    if config.name != "openrouter":
        raise ValueError(
            "LLM request preparation failed: unsupported provider "
            f"{config.name!r} for logical model {requested_model!r}."
        )

    if "previous_response_id" in prepared:
        raise ValueError(
            "LLM request preparation failed for provider 'openrouter': "
            "previous_response_id is stateful and must not be sent."
        )

    if (
        requested_format_type == "json_schema"
        and config.structured_output_mode == "json_object"
    ):
        # OpenRouter serving endpoints use several incompatible hosted grammar
        # engines. Keep the canonical schema authoritative in the caller's
        # local validator while asking the provider only for valid JSON syntax.
        prepared_text = dict(prepared.get("text") or {})
        prepared_text["format"] = {"type": "json_object"}
        prepared["text"] = prepared_text
        if canonical_output_schema is None:
            raise TypeError(
                "LLM request preparation failed: canonical output schema is missing."
            )
        schema_instruction = (
            "Return exactly one JSON value satisfying the following canonical JSON "
            "Schema. The schema is authoritative: use only its declared keys, put "
            "each value at its declared location, obey every enum and const, and do "
            "not add aliases or commentary.\n\n"
            f"Schema name: {output_schema_name}\n"
            + json.dumps(
                canonical_output_schema,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        schema_input = {
            "role": "system",
            "content": [{"type": "input_text", "text": schema_instruction}],
        }
        request_input = prepared.get("input")
        if isinstance(request_input, list):
            insertion_index = 0
            while (
                insertion_index < len(request_input)
                and isinstance(request_input[insertion_index], Mapping)
                and request_input[insertion_index].get("role") == "system"
            ):
                insertion_index += 1
            prepared["input"] = [
                *request_input[:insertion_index],
                schema_input,
                *request_input[insertion_index:],
            ]
        elif isinstance(request_input, str):
            prepared["input"] = [
                schema_input,
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request_input}],
                },
            ]
        else:
            raise TypeError(
                "LLM request preparation failed: structured JSON input must be a "
                "string or input-item list."
            )

    prepared.pop("store", None)
    extra_body_value = prepared.get("extra_body")
    if extra_body_value is None:
        extra_body: dict[str, Any] = {}
    elif isinstance(extra_body_value, Mapping):
        extra_body = dict(copy.deepcopy(extra_body_value))
    else:
        raise TypeError(
            "LLM request preparation failed for provider 'openrouter': "
            "extra_body must be a mapping."
        )

    if "previous_response_id" in extra_body:
        raise ValueError(
            "LLM request preparation failed for provider 'openrouter': "
            "previous_response_id is stateful and must not be sent."
        )
    extra_body.pop("store", None)

    top_level_preferences = prepared.pop("provider", None)
    existing_preferences = extra_body.get("provider")
    merged_preferences: dict[str, Any] = {}
    for preference_source in (existing_preferences, top_level_preferences):
        if preference_source is None:
            continue
        if not isinstance(preference_source, Mapping):
            raise TypeError(
                "LLM request preparation failed for provider 'openrouter': "
                "provider preferences must be a mapping."
            )
        merged_preferences.update(copy.deepcopy(dict(preference_source)))
    merged_preferences.update(copy.deepcopy(config.provider_preferences))
    if requested_format_type == "json_schema":
        # Do not route structured JSON calls to an endpoint that may silently
        # ignore the requested response-format parameter.
        merged_preferences["require_parameters"] = True
    if merged_preferences:
        extra_body["provider"] = merged_preferences
    else:
        extra_body.pop("provider", None)

    if config.reasoning_effort_override is not None:
        reasoning_value = prepared.get("reasoning")
        if reasoning_value is None:
            reasoning: dict[str, Any] = {}
        elif isinstance(reasoning_value, Mapping):
            reasoning = dict(copy.deepcopy(reasoning_value))
        else:
            raise TypeError(
                "LLM request preparation failed for provider 'openrouter': "
                "reasoning must be a mapping."
            )
        reasoning["effort"] = config.reasoning_effort_override
        # OpenRouter documents ``none`` as disabling reasoning. Make that
        # inferred state explicit so hybrid-thinking endpoints cannot fall
        # back to a model-level default of enabled.
        if config.reasoning_effort_override == "none":
            reasoning["enabled"] = False
        else:
            reasoning.pop("enabled", None)
        prepared["reasoning"] = reasoning

    if config.max_output_tokens_cap is not None:
        requested_limit = prepared.get("max_output_tokens")
        if requested_limit is None:
            prepared["max_output_tokens"] = config.max_output_tokens_cap
        elif (
            isinstance(requested_limit, bool)
            or not isinstance(requested_limit, int)
            or requested_limit < 1
        ):
            raise TypeError(
                "LLM request preparation failed for provider 'openrouter': "
                "max_output_tokens must be a positive integer."
            )
        else:
            prepared["max_output_tokens"] = min(
                requested_limit,
                config.max_output_tokens_cap,
            )

    if extra_body:
        prepared["extra_body"] = extra_body
    else:
        prepared.pop("extra_body", None)

    return prepared, attribution


def create_response(
    payload: Mapping[str, Any],
    config: ProviderConfig | None = None,
) -> dict[str, Any]:
    """Send one Responses request and return its plain-dictionary response."""

    from aso_agents.evals.telemetry import (
        begin_llm_call,
        complete_llm_call,
        fail_llm_call,
        instrument_openrouter_request,
    )

    resolved_config = get_provider_config() if config is None else config
    retry_policy = (
        TransportRetryPolicy.from_env()
        if resolved_config.name == "openrouter"
        else TransportRetryPolicy()
    )
    if retry_policy.retries:
        # Own retries here so waits release admission slots and publish a
        # provider-wide cooldown instead of multiplying hidden SDK retries.
        resolved_config = replace(resolved_config, max_retries=0)
    prepared, attribution = prepare_responses_payload(payload, resolved_config)

    if not resolved_config.api_key.strip():
        credential_name = _credential_variable(resolved_config.name)
        raise RuntimeError(
            "LLM client configuration failed for provider "
            f"'{attribution['provider']}', logical model "
            f"{attribution['requested_model']!r}, resolved model "
            f"{attribution['resolved_model']!r}. Set {credential_name} for "
            f"ASO_LLM_PROVIDER={resolved_config.name}."
        )

    try:
        client = _get_client(resolved_config)
    except Exception as exc:
        pending_call = begin_llm_call(attribution)
        fail_llm_call(
            pending_call,
            status="client_configuration_error",
            error_type=type(exc).__name__,
        )
        raise RuntimeError(
            "LLM client configuration failed for provider "
            f"'{attribution['provider']}', logical model "
            f"{attribution['requested_model']!r}, resolved model "
            f"{attribution['resolved_model']!r} ({type(exc).__name__})."
        ) from None

    resource_name = str(os.environ.get("ASO_LLM_CALL_RESOURCE") or "").strip()
    if retry_policy.retries and not resource_name:
        route = resolved_config.provider_preferences.get("only") or [resolved_config.name]
        resource_name = scoped_resource_name("llm_transport", "_".join(route))
    # A legitimate first generation may outlast the retry window. Start that
    # window only when the first retryable failure has actually arrived.
    retry_started: float | None = None
    for attempt in range(retry_policy.retries + 1):
        admission = acquire_resource(resource_name) if resource_name else nullcontext()
        with admission:
            pending_call = begin_llm_call(attribution)
            call_prepared = copy.deepcopy(prepared)
            instrument_openrouter_request(call_prepared, pending_call)
            try:
                response = client.responses.create(**call_prepared)
            except Exception as exc:
                details = transport_error_details(exc)
                delay = retry_policy.delay(attempt, details) if retry_policy.retries else 0.0
                eligible = retryable_transport_error(exc, details)
                if eligible and retry_started is None:
                    retry_started = time.monotonic()
                will_retry = (
                    eligible and retry_started is not None and attempt < retry_policy.retries
                    and time.monotonic() - retry_started + delay <= retry_policy.budget_seconds
                )
                if retry_policy.retries and eligible:
                    # Publish before releasing admission so the next waiter
                    # observes the cooldown. Never sleep while holding a slot.
                    defer_resource(resource_name, delay)
                details.update({
                    "attempt": attempt + 1, "will_retry": will_retry,
                    "cooldown_seconds": round(delay, 3) if eligible else 0.0,
                })
                if will_retry:
                    details["excluded_providers_for_retry"] = avoid_failed_provider(prepared, details)
                fail_llm_call(
                    pending_call,
                    status="transport_retry" if will_retry else "transport_error",
                    error_type=type(exc).__name__,
                    transport_details=details,
                )
                if will_retry:
                    continue
                raise RuntimeError(
                    "LLM API transport failed for provider "
                    f"'{attribution['provider']}', logical model "
                    f"{attribution['requested_model']!r}, resolved model "
                    f"{attribution['resolved_model']!r} ({type(exc).__name__}; "
                    f"HTTP {details.get('status_code', 'unknown')})."
                ) from None

            try:
                if isinstance(response, Mapping):
                    response_data = dict(copy.deepcopy(response))
                else:
                    model_dump = getattr(response, "model_dump", None)
                    if not callable(model_dump):
                        raise TypeError("SDK response does not provide model_dump().")
                    if resolved_config.name == "openrouter":
                        dumped_response = model_dump(exclude_none=True)
                    else:
                        dumped_response = model_dump()
                    if not isinstance(dumped_response, Mapping):
                        raise TypeError("SDK response model_dump() did not return a mapping.")
                    response_data = dict(copy.deepcopy(dumped_response))
            except Exception as exc:
                fail_llm_call(
                    pending_call,
                    status="response_parsing_error",
                    error_type=type(exc).__name__,
                )
                raise RuntimeError(
                    "LLM response parsing failed for provider "
                    f"'{attribution['provider']}', logical model "
                    f"{attribution['requested_model']!r}, resolved model "
                    f"{attribution['resolved_model']!r} ({type(exc).__name__})."
                ) from None

            # Responses can report a provider failure inside an HTTP 200
            # envelope. It must follow the same bounded recovery path as a
            # retryable HTTP exception, retaining usage from every attempt.
            raw_error = response_data.get("error")
            error_code = (
                str(raw_error.get("code") or "").strip()
                if isinstance(raw_error, Mapping) else ""
            )
            if (
                retry_policy.retries
                and str(response_data.get("status") or "").lower() == "failed"
                and error_code in {"server_error", "rate_limit_exceeded"}
            ):
                if retry_started is None:
                    retry_started = time.monotonic()
                details = {"response_status": "failed", "error_code": error_code}
                delay = retry_policy.delay(attempt, details)
                will_retry = (
                    attempt < retry_policy.retries
                    and time.monotonic() - retry_started + delay <= retry_policy.budget_seconds
                )
                defer_resource(resource_name, delay)
                details.update({
                    "attempt": attempt + 1, "will_retry": will_retry,
                    "cooldown_seconds": round(delay, 3),
                })
                if will_retry:
                    details["excluded_providers_for_retry"] = avoid_failed_provider(
                        prepared, {"openrouter_metadata": response_data.get("openrouter_metadata")},
                    )
                fail_llm_call(
                    pending_call,
                    status="transport_retry" if will_retry else "transport_error",
                    error_type=error_code, transport_details=details,
                    response=response_data,
                )
                if will_retry:
                    continue
                raise RuntimeError(
                    "LLM API response transport failed for provider "
                    f"'{attribution['provider']}', resolved model "
                    f"{attribution['resolved_model']!r} ({error_code})."
                ) from None

            complete_llm_call(pending_call, response_data)
            response_status = str(response_data.get("status") or "").strip().lower()
            if response_status == "failed":
                raw_error = response_data.get("error")
                if isinstance(raw_error, Mapping):
                    error_code = str(raw_error.get("code") or "").strip()
                    error_type = str(raw_error.get("type") or "").strip()
                    error_message = str(raw_error.get("message") or "").strip()
                    error_parts = [
                        part
                        for part in (
                            f"code={error_code}" if error_code else "",
                            f"type={error_type}" if error_type else "",
                            error_message,
                        )
                        if part
                    ]
                else:
                    error_parts = [str(raw_error).strip()] if str(raw_error or "").strip() else []
                failure_detail = "; ".join(error_parts) or "provider returned status=failed"
                raise RuntimeError(
                    "LLM API response failed for provider "
                    f"'{attribution['provider']}', logical model "
                    f"{attribution['requested_model']!r}, resolved model "
                    f"{attribution['resolved_model']!r}: {failure_detail}."
                )
            return response_data


def reset_provider_runtime_for_tests() -> None:
    """Clear cached SDK clients so tests cannot retain runtime configuration."""

    with _CLIENT_CACHE_LOCK:
        _CLIENT_CACHE.clear()


def _credential_variable(provider_name: str) -> str:
    if provider_name == "openai":
        return "OPENAI_API_KEY"
    if provider_name == "openrouter":
        return "OPENROUTER_API_KEY"
    raise ValueError(f"Unsupported LLM provider {provider_name!r}.")


def _env_value_or_default(source: Mapping[str, str], name: str, default: str) -> str:
    value = source.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _parse_optional_boolean(source: Mapping[str, str], name: str) -> bool | None:
    raw_value = source.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    normalized_value = raw_value.strip().lower()
    if normalized_value in _TRUE_VALUES:
        return True
    if normalized_value in _FALSE_VALUES:
        return False
    raise ValueError(
        "LLM provider configuration failed for 'openrouter': "
        f"{name} must be one of 1, true, yes, on, 0, false, no, or off."
    )


def _parse_optional_integer(
    source: Mapping[str, str],
    name: str,
    *,
    minimum: int,
) -> int | None:
    raw_value = source.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = int(raw_value.strip())
    except ValueError:
        value = minimum - 1
    if value < minimum:
        qualifier = "a positive integer" if minimum == 1 else "a non-negative integer"
        raise ValueError(
            "LLM provider configuration failed for 'openrouter': "
            f"{name} must be {qualifier}."
        )
    return value


def _parse_optional_float(
    source: Mapping[str, str],
    name: str,
    *,
    minimum_exclusive: float,
) -> float | None:
    raw_value = source.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = float(raw_value.strip())
    except ValueError:
        value = minimum_exclusive
    if not math.isfinite(value) or value <= minimum_exclusive:
        raise ValueError(
            "LLM provider configuration failed for 'openrouter': "
            f"{name} must be a positive number."
        )
    return value


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _client_cache_key(config: ProviderConfig) -> _ClientCacheKey:
    return (
        config.name,
        config.base_url,
        tuple(sorted(config.default_headers.items())),
        config.request_timeout_seconds,
        config.max_retries,
        hmac.digest(
            _CREDENTIAL_FINGERPRINT_KEY,
            config.api_key.encode("utf-8"),
            "sha256",
        ),
    )


def _get_client(config: ProviderConfig) -> OpenAI:
    if not config.api_key.strip():
        credential_name = _credential_variable(config.name)
        raise RuntimeError(
            f"LLM provider configuration failed for '{config.name}': "
            f"Set {credential_name} for ASO_LLM_PROVIDER={config.name}."
        )

    cache_key = _client_cache_key(config)
    with _CLIENT_CACHE_LOCK:
        cached_client = _CLIENT_CACHE.get(cache_key)
        if cached_client is not None:
            return cached_client

        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "base_url": config.base_url,
        }
        if config.name == "openrouter":
            client_kwargs["max_retries"] = (
                _OPENROUTER_MAX_RETRIES
                if config.max_retries is None
                else config.max_retries
            )
            if config.request_timeout_seconds is not None:
                client_kwargs["timeout"] = config.request_timeout_seconds
        if config.default_headers:
            client_kwargs["default_headers"] = dict(config.default_headers)
        client = OpenAI(**client_kwargs)
        _CLIENT_CACHE[cache_key] = client
        return client


__all__ = [
    "ProviderConfig",
    "build_hosted_web_search_tool",
    "create_response",
    "get_provider_config",
    "prepare_responses_payload",
    "provider_credential_help",
    "provider_credentials_available",
    "reset_provider_runtime_for_tests",
    "selected_provider_name",
]
