from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import pytest

from aso_agents import llm_provider


@pytest.fixture(autouse=True)
def _reset_provider_runtime() -> None:
    llm_provider.reset_provider_runtime_for_tests()
    yield
    llm_provider.reset_provider_runtime_for_tests()


class _FakeModelResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.model_dump_calls = 0
        self.model_dump_kwargs: list[dict[str, Any]] = []

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        self.model_dump_calls += 1
        self.model_dump_kwargs.append(copy.deepcopy(kwargs))
        return copy.deepcopy(self.data)


class _FakeOpenAI:
    instances: list["_FakeOpenAI"] = []
    queued_responses: list[Any] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = copy.deepcopy(kwargs)
        self.calls: list[dict[str, Any]] = []
        self.responses = self
        type(self).instances.append(self)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(copy.deepcopy(kwargs))
        if type(self).queued_responses:
            return type(self).queued_responses.pop(0)
        return _FakeModelResponse({"id": "resp_test", "output": []})


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeOpenAI]:
    _FakeOpenAI.instances = []
    _FakeOpenAI.queued_responses = []
    monkeypatch.setattr(llm_provider, "OpenAI", _FakeOpenAI)
    return _FakeOpenAI


def test_unset_provider_selects_default_openai() -> None:
    config = llm_provider.get_provider_config({"OPENAI_API_KEY": "openai-test"})

    assert llm_provider.selected_provider_name({}) == "openai"
    assert config.name == "openai"
    assert config.api_key == "openai-test"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.default_headers == {}
    assert config.continuation_mode == "previous_response_id"
    assert config.provider_preferences == {}
    assert config.model_override is None


def test_provider_name_is_case_insensitive_and_trimmed() -> None:
    assert llm_provider.selected_provider_name({"ASO_LLM_PROVIDER": " OpenRouter "}) == "openrouter"
    assert llm_provider.selected_provider_name({"ASO_LLM_PROVIDER": "OPENAI"}) == "openai"


def test_openai_uses_selected_key_and_custom_base_url() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "selected-key",
            "OPENAI_BASE_URL": " https://openai.example/v1 ",
            "OPENROUTER_API_KEY": "other-provider-key",
        }
    )

    assert config.api_key == "selected-key"
    assert config.base_url == "https://openai.example/v1"


def test_openrouter_uses_selected_key_url_headers_and_preferences() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENAI_API_KEY": "other-provider-key",
            "OPENROUTER_API_KEY": "router-test",
            "OPENROUTER_BASE_URL": "https://router.example/api/v1",
            "OPENROUTER_HTTP_REFERER": "https://aso.example",
            "OPENROUTER_APP_TITLE": "ASO Test",
            "OPENROUTER_ZDR": "yes",
            "OPENROUTER_DATA_COLLECTION": "DENY",
            "OPENROUTER_ONLY_PROVIDERS": "OpenAI, Azure ",
            "OPENROUTER_MODEL": " qwen/qwen3.8-27b ",
        }
    )

    assert config.name == "openrouter"
    assert config.api_key == "router-test"
    assert config.base_url == "https://router.example/api/v1"
    assert config.default_headers == {
        "HTTP-Referer": "https://aso.example",
        "X-OpenRouter-Title": "ASO Test",
    }
    assert config.continuation_mode == "stateless"
    assert config.provider_preferences == {
        "sort": "throughput",
        "zdr": True,
        "data_collection": "deny",
        "only": ["OpenAI", "Azure"],
    }
    assert config.model_override == "qwen/qwen3.8-27b"
    assert config.structured_output_mode == "json_object"


def test_openrouter_defaults_url_and_app_title() -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "router-test"}
    )

    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.default_headers == {"X-OpenRouter-Title": "MODAL-ASO"}


def test_openrouter_provider_order_enables_only_ordered_fallbacks() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test",
            "OPENROUTER_ONLY_PROVIDERS": "primary, secondary, tertiary",
            "OPENROUTER_PROVIDER_ORDER": "primary, secondary, tertiary",
            "OPENROUTER_PROVIDER_SORT": "balanced",
        }
    )

    assert config.provider_preferences == {
        "only": ["primary", "secondary", "tertiary"],
        "order": ["primary", "secondary", "tertiary"],
        "allow_fallbacks": True,
    }


def test_openrouter_empty_app_title_uses_default() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test",
            "OPENROUTER_APP_TITLE": "  ",
        }
    )

    assert config.default_headers == {"X-OpenRouter-Title": "MODAL-ASO"}


@pytest.mark.parametrize(
    ("provider", "other_key", "missing_name"),
    [
        ("openai", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        ("openrouter", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    ],
)
def test_missing_selected_provider_credential_does_not_fall_back(
    provider: str,
    other_key: str,
    missing_name: str,
) -> None:
    env = {"ASO_LLM_PROVIDER": provider, other_key: "other-provider-test"}

    with pytest.raises(RuntimeError, match=missing_name) as exc_info:
        llm_provider.get_provider_config(env)

    assert "other-provider-test" not in str(exc_info.value)


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_require_credentials_false_supports_cli_configuration(provider: str) -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": provider},
        require_credentials=False,
    )

    assert config.name == provider
    assert config.api_key == ""
    assert llm_provider.provider_credentials_available({"ASO_LLM_PROVIDER": provider}) is False


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("openai", "OPENAI_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")],
)
def test_credential_availability_and_help_are_provider_specific(
    provider: str,
    key_name: str,
) -> None:
    env = {"ASO_LLM_PROVIDER": provider, key_name: "test-only"}

    assert llm_provider.provider_credentials_available(env) is True
    assert (
        llm_provider.provider_credential_help(env)
        == f"Set {key_name} for ASO_LLM_PROVIDER={provider}."
    )


@pytest.mark.parametrize("value", ["", "anthropic", "open-router"])
def test_invalid_provider_fails_configuration(value: str) -> None:
    with pytest.raises(ValueError, match="ASO_LLM_PROVIDER"):
        llm_provider.get_provider_config(
            {"ASO_LLM_PROVIDER": value},
            require_credentials=False,
        )


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", " On "])
def test_openrouter_truthy_zdr_values(value: str) -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_ZDR": value},
        require_credentials=False,
    )

    assert config.provider_preferences == {"sort": "throughput", "zdr": True}


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", " Off "])
def test_openrouter_false_zdr_values_are_valid_and_omitted(value: str) -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_ZDR": value},
        require_credentials=False,
    )

    assert config.provider_preferences == {"sort": "throughput"}


def test_invalid_openrouter_boolean_fails_configuration() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_ZDR"):
        llm_provider.get_provider_config(
            {"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_ZDR": "sometimes"},
            require_credentials=False,
        )


def test_invalid_openrouter_privacy_value_fails_configuration() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_DATA_COLLECTION"):
        llm_provider.get_provider_config(
            {
                "ASO_LLM_PROVIDER": "openrouter",
                "OPENROUTER_DATA_COLLECTION": "prompt",
            },
            require_credentials=False,
        )


@pytest.mark.parametrize("value", ["price", "throughput", "latency", " Latency "])
def test_openrouter_provider_sort_is_normalized(value: str) -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_PROVIDER_SORT": value,
        },
        require_credentials=False,
    )

    assert config.provider_preferences == {"sort": value.strip().lower()}


def test_openrouter_balanced_sort_restores_router_default() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_PROVIDER_SORT": "balanced",
        },
        require_credentials=False,
    )

    assert config.provider_preferences == {}


def test_invalid_openrouter_provider_sort_fails_configuration() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_PROVIDER_SORT"):
        llm_provider.get_provider_config(
            {
                "ASO_LLM_PROVIDER": "openrouter",
                "OPENROUTER_PROVIDER_SORT": "random",
            },
            require_credentials=False,
        )


def test_openrouter_performance_policy_is_parsed() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_REASONING_EFFORT": " Low ",
            "OPENROUTER_MAX_OUTPUT_TOKENS": "8000",
            "OPENROUTER_TIMEOUT_SECONDS": "60.5",
            "OPENROUTER_MAX_RETRIES": "0",
        },
        require_credentials=False,
    )

    assert config.reasoning_effort_override == "low"
    assert config.max_output_tokens_cap == 8000
    assert config.request_timeout_seconds == 60.5
    assert config.max_retries == 0


def test_openrouter_accepts_maximum_reasoning_effort() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_REASONING_EFFORT": " MAX ",
        },
        require_credentials=False,
    )

    assert config.reasoning_effort_override == "max"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENROUTER_REASONING_EFFORT", "extreme"),
        ("OPENROUTER_MAX_OUTPUT_TOKENS", "0"),
        ("OPENROUTER_MAX_OUTPUT_TOKENS", "many"),
        ("OPENROUTER_TIMEOUT_SECONDS", "0"),
        ("OPENROUTER_TIMEOUT_SECONDS", "later"),
        ("OPENROUTER_TIMEOUT_SECONDS", "nan"),
        ("OPENROUTER_TIMEOUT_SECONDS", "inf"),
        ("OPENROUTER_MAX_RETRIES", "-1"),
        ("OPENROUTER_MAX_RETRIES", "several"),
    ],
)
def test_invalid_openrouter_performance_policy_fails_configuration(
    name: str, value: str
) -> None:
    with pytest.raises(ValueError, match=name):
        llm_provider.get_provider_config(
            {"ASO_LLM_PROVIDER": "openrouter", name: value},
            require_credentials=False,
        )


@pytest.mark.parametrize(
    "value",
    ["gpt-unknown", "openrouter-auto", "model", "/model", "provider/", "provider/model slug"],
)
def test_openrouter_global_model_requires_qualified_slug(value: str) -> None:
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        llm_provider.get_provider_config(
            {
                "ASO_LLM_PROVIDER": "openrouter",
                "OPENROUTER_MODEL": value,
            },
            require_credentials=False,
        )


@pytest.mark.parametrize(
    "value",
    [
        "qwen/qwen3.8-27b",
        "deepseek/deepseek-v4-flash-0731",
        "z-ai/glm-5.2:free",
        "openrouter/auto",
        "~openai/gpt-latest",
        "@preset/general-high",
    ],
)
def test_openrouter_global_model_accepts_qualified_catalog_ids_and_aliases(value: str) -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_MODEL": value,
        },
        require_credentials=False,
    )

    assert config.model_override == value


def test_openai_ignores_openrouter_global_model() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openai",
            "OPENROUTER_MODEL": "qwen/qwen3.8-27b",
        },
        require_credentials=False,
    )

    assert config.model_override is None
    assert config.resolve_model("gpt-5.2-2025-12-11") == "gpt-5.2-2025-12-11"


@pytest.mark.parametrize(
    ("provider", "logical_model", "expected"),
    [
        ("openai", "gpt-5.2-2025-12-11", "gpt-5.2-2025-12-11"),
        ("openai", "gpt-5-mini-2025-08-07", "gpt-5-mini-2025-08-07"),
        ("openrouter", "gpt-5.2-2025-12-11", "openai/gpt-5.2"),
        ("openrouter", "gpt-5-mini-2025-08-07", "openai/gpt-5-mini"),
    ],
)
def test_default_logical_models_resolve_without_moving_aliases(
    provider: str,
    logical_model: str,
    expected: str,
) -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": provider},
        require_credentials=False,
    )

    resolved = config.resolve_model(logical_model)

    assert resolved == expected
    assert "latest" not in resolved
    assert "~" not in resolved


def test_qualified_model_passes_through_only_for_openrouter() -> None:
    openrouter = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter"},
        require_credentials=False,
    )
    openai = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openai"},
        require_credentials=False,
    )

    assert openrouter.resolve_model("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"
    with pytest.raises(ValueError, match="provider-qualified"):
        openai.resolve_model("anthropic/claude-sonnet-4")


@pytest.mark.parametrize(
    "logical_model",
    [
        "gpt-5.2-2025-12-11",
        "gpt-5-mini-2025-08-07",
        "anthropic/claude-sonnet-4",
        "otherwise-unknown-logical-model",
    ],
)
def test_openrouter_global_model_overrides_every_agent_model(logical_model: str) -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_MODEL": "qwen/qwen3.8-27b",
        },
        require_credentials=False,
    )

    assert config.resolve_model(logical_model) == "qwen/qwen3.8-27b"


def test_unknown_unqualified_openrouter_model_fails() -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter"},
        require_credentials=False,
    )

    with pytest.raises(ValueError, match="provider-qualified slug"):
        config.resolve_model("gpt-unknown")


def test_openai_unknown_unqualified_model_is_preserved_for_existing_overrides() -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openai"},
        require_credentials=False,
    )

    assert config.resolve_model("gpt-custom-snapshot") == "gpt-custom-snapshot"


def test_hosted_web_search_tool_is_provider_specific() -> None:
    openai = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openai"},
        require_credentials=False,
    )
    openrouter = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter"},
        require_credentials=False,
    )

    assert llm_provider.build_hosted_web_search_tool(
        openai,
        max_uses=5,
        max_results=8,
    ) == {"type": "web_search"}
    assert llm_provider.build_hosted_web_search_tool(
        openrouter,
        max_uses=5,
        max_results=8,
    ) == {
        "type": "openrouter:web_search",
        "parameters": {"max_uses": 5, "max_results": 8},
    }


@pytest.mark.parametrize(("name", "value"), [("max_uses", 0), ("max_results", False)])
def test_hosted_web_search_rejects_invalid_limits(name: str, value: Any) -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter"},
        require_credentials=False,
    )
    kwargs = {name: value}

    with pytest.raises(ValueError, match=name):
        llm_provider.build_hosted_web_search_tool(config, **kwargs)


def test_prepare_openai_payload_deep_copies_and_attributes_model() -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openai"},
        require_credentials=False,
    )
    payload = {
        "model": "gpt-5-mini-2025-08-07",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        "previous_response_id": "resp_previous",
        "store": True,
    }
    original = copy.deepcopy(payload)

    prepared, metadata = llm_provider.prepare_responses_payload(payload, config)
    prepared["input"][0]["content"][0]["text"] = "changed"

    assert payload == original
    assert prepared["model"] == "gpt-5-mini-2025-08-07"
    assert prepared["previous_response_id"] == "resp_previous"
    assert prepared["store"] is True
    assert metadata == {
        "provider": "openai",
        "requested_model": "gpt-5-mini-2025-08-07",
        "resolved_model": "gpt-5-mini-2025-08-07",
    }


def test_prepare_openrouter_payload_merges_preferences_only_in_extra_body() -> None:
    secret = "router-secret-not-metadata"
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": secret,
            "OPENROUTER_ZDR": "true",
            "OPENROUTER_ONLY_PROVIDERS": "OpenAI",
        }
    )
    payload = {
        "model": "gpt-5.2-2025-12-11",
        "input": [{"role": "user", "content": "hello"}],
        "store": True,
        "provider": {"sort": "throughput"},
        "extra_body": {
            "text": {"format": {"type": "json_object"}},
            "provider": {"data_collection": "allow"},
            "store": True,
        },
    }
    original = copy.deepcopy(payload)

    prepared, metadata = llm_provider.prepare_responses_payload(payload, config)

    assert payload == original
    assert prepared == {
        "model": "openai/gpt-5.2",
        "input": [{"role": "user", "content": "hello"}],
        "extra_body": {
            "text": {"format": {"type": "json_object"}},
            "provider": {
                "data_collection": "allow",
                "sort": "throughput",
                "zdr": True,
                "only": ["OpenAI"],
            },
        },
    }
    assert metadata == {
        "provider": "openrouter",
        "requested_model": "gpt-5.2-2025-12-11",
        "resolved_model": "openai/gpt-5.2",
    }
    assert secret not in repr(metadata)
    assert secret not in repr(config)


def test_prepare_openrouter_schema_uses_portable_json_object_transport() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test",
        }
    )
    payload = {
        "model": "gpt-5.2-2025-12-11",
        "input": [],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "strict_contract",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            }
        },
        "provider": {"require_parameters": False},
    }

    prepared, _ = llm_provider.prepare_responses_payload(payload, config)

    assert prepared["extra_body"]["provider"] == {
        "sort": "throughput",
        "require_parameters": True,
    }
    assert prepared["text"]["format"] == {"type": "json_object"}
    schema_instruction = prepared["input"][0]["content"][0]["text"]
    assert "canonical JSON Schema" in schema_instruction
    assert '"status"' in schema_instruction
    assert '"additionalProperties":false' in schema_instruction


def test_json_object_transport_preserves_system_then_schema_then_user_input() -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter"},
        require_credentials=False,
    )
    payload = {
        "model": "gpt-5.2-2025-12-11",
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "agent instructions"}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "user request"}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "contract",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            }
        },
    }

    prepared, _ = llm_provider.prepare_responses_payload(payload, config)

    assert [item["role"] for item in prepared["input"]] == [
        "system",
        "system",
        "user",
    ]
    assert prepared["input"][0] == payload["input"][0]
    assert prepared["input"][2] == payload["input"][1]


def test_openrouter_compiles_every_model_schema_to_the_same_portable_contract() -> None:
    sparse_schema = {
        "type": "object",
        "properties": {
            "required_value": {"type": "string"},
            "optional_value": {"type": "string"},
        },
        "required": ["required_value"],
        "additionalProperties": False,
    }
    payload = {
        "model": "logical-model",
        "input": [],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sparse_contract",
                "strict": True,
                "schema": sparse_schema,
            }
        },
    }
    deepseek = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_MODEL": "deepseek/deepseek-v4-flash-0731",
            "OPENROUTER_STRUCTURED_OUTPUT_MODE": "json_schema",
        },
        require_credentials=False,
    )
    openai = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_MODEL": "openai/gpt-5-mini",
            "OPENROUTER_STRUCTURED_OUTPUT_MODE": "json_schema",
        },
        require_credentials=False,
    )

    deepseek_payload, _ = llm_provider.prepare_responses_payload(payload, deepseek)
    openai_payload, _ = llm_provider.prepare_responses_payload(payload, openai)

    deepseek_compiled = deepseek_payload["text"]["format"]["schema"]
    openai_compiled = openai_payload["text"]["format"]["schema"]
    assert deepseek_compiled == openai_compiled
    assert set(deepseek_compiled["required"]) == {
        "required_value",
        "optional_value",
    }
    assert deepseek_compiled["properties"]["optional_value"]["type"] == [
        "string",
        "null",
    ]


def test_openrouter_can_explicitly_request_hosted_json_schema() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_STRUCTURED_OUTPUT_MODE": "json_schema",
        },
        require_credentials=False,
    )
    payload = {
        "model": "gpt-5.2-2025-12-11",
        "input": [],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "contract",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            }
        },
    }

    prepared, _ = llm_provider.prepare_responses_payload(payload, config)

    assert prepared["text"]["format"]["type"] == "json_schema"
    assert (
        prepared["text"]["format"]["schema"]
        == payload["text"]["format"]["schema"]
    )


@pytest.mark.parametrize("value", ["schema", "json", "strict"])
def test_openrouter_rejects_unknown_structured_output_mode(value: str) -> None:
    with pytest.raises(ValueError, match="OPENROUTER_STRUCTURED_OUTPUT_MODE"):
        llm_provider.get_provider_config(
            {
                "ASO_LLM_PROVIDER": "openrouter",
                "OPENROUTER_STRUCTURED_OUTPUT_MODE": value,
            },
            require_credentials=False,
        )


def test_prepare_openrouter_payload_applies_global_model_override() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_MODEL": "openrouter/auto",
        },
        require_credentials=False,
    )

    prepared, metadata = llm_provider.prepare_responses_payload(
        {"model": "gpt-5-mini-2025-08-07", "input": []},
        config,
    )

    assert prepared["model"] == "openrouter/auto"
    assert metadata == {
        "provider": "openrouter",
        "requested_model": "gpt-5-mini-2025-08-07",
        "resolved_model": "openrouter/auto",
    }


def test_prepare_openrouter_payload_applies_reasoning_and_output_ceiling() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_REASONING_EFFORT": "low",
            "OPENROUTER_MAX_OUTPUT_TOKENS": "8000",
        },
        require_credentials=False,
    )
    payload = {
        "model": "gpt-5.2-2025-12-11",
        "input": [],
        "reasoning": {"effort": "high", "exclude": True},
        "max_output_tokens": 16000,
    }

    prepared, _ = llm_provider.prepare_responses_payload(payload, config)

    assert payload["reasoning"]["effort"] == "high"
    assert payload["max_output_tokens"] == 16000
    assert prepared["reasoning"] == {"effort": "low", "exclude": True}
    assert prepared["max_output_tokens"] == 8000


def test_prepare_openrouter_payload_explicitly_disables_reasoning_for_none() -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_REASONING_EFFORT": "none",
        },
        require_credentials=False,
    )

    prepared, _ = llm_provider.prepare_responses_payload(
        {
            "model": "gpt-5.2-2025-12-11",
            "input": [],
            "reasoning": {"effort": "high"},
        },
        config,
    )

    assert prepared["reasoning"] == {"effort": "none", "enabled": False}


@pytest.mark.parametrize("location", ["top_level", "extra_body"])
def test_prepare_openrouter_rejects_stateful_previous_response_id(location: str) -> None:
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter"},
        require_credentials=False,
    )
    payload: dict[str, Any] = {"model": "gpt-5.2-2025-12-11", "input": []}
    if location == "top_level":
        payload["previous_response_id"] = "resp_stateful"
    else:
        payload["extra_body"] = {"previous_response_id": "resp_stateful"}

    with pytest.raises(ValueError, match="previous_response_id"):
        llm_provider.prepare_responses_payload(payload, config)


def test_create_response_uses_model_dump_and_sends_plain_prepared_payload(
    fake_openai: type[_FakeOpenAI],
) -> None:
    response = _FakeModelResponse(
        {"id": "resp_router", "model": "openai/gpt-5.2", "output": [{"type": "message"}]}
    )
    fake_openai.queued_responses = [response]
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "router-test"}
    )

    result = llm_provider.create_response(
        {"model": "gpt-5.2-2025-12-11", "input": [], "store": True},
        config,
    )

    assert result == {
        "id": "resp_router",
        "model": "openai/gpt-5.2",
        "output": [{"type": "message"}],
    }
    assert response.model_dump_calls == 1
    assert response.model_dump_kwargs == [{"exclude_none": True}]
    assert fake_openai.instances[0].calls == [
        {
            "model": "openai/gpt-5.2",
            "input": [],
            "extra_body": {"provider": {"sort": "throughput"}},
        }
    ]


def test_create_response_accepts_and_copies_plain_dictionary_response(
    fake_openai: type[_FakeOpenAI],
) -> None:
    raw_response = {"id": "resp_dict", "output": [{"type": "message"}]}
    fake_openai.queued_responses = [raw_response]
    config = llm_provider.get_provider_config(
        {"OPENAI_API_KEY": "openai-test"}
    )

    result = llm_provider.create_response(
        {"model": "gpt-5-mini-2025-08-07", "input": []},
        config,
    )
    result["output"][0]["type"] = "changed"

    assert raw_response["output"][0]["type"] == "message"


def test_create_response_preserves_openai_model_dump_defaults(
    fake_openai: type[_FakeOpenAI],
) -> None:
    response = _FakeModelResponse({"id": "resp_openai", "output": []})
    fake_openai.queued_responses = [response]
    config = llm_provider.get_provider_config({"OPENAI_API_KEY": "openai-test"})

    llm_provider.create_response(
        {"model": "gpt-5-mini-2025-08-07", "input": []},
        config,
    )

    assert response.model_dump_kwargs == [{}]


def test_sdk_clients_are_created_and_cached_for_both_providers(
    fake_openai: type[_FakeOpenAI],
) -> None:
    openai_config = llm_provider.get_provider_config(
        {
            "OPENAI_API_KEY": "openai-test",
            "OPENAI_BASE_URL": "https://openai.example/v1",
        }
    )
    openrouter_config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test",
            "OPENROUTER_HTTP_REFERER": "https://aso.example",
        }
    )
    equivalent_openai_config = llm_provider.get_provider_config(
        {
            "OPENAI_API_KEY": "openai-test",
            "OPENAI_BASE_URL": "https://openai.example/v1",
        }
    )
    equivalent_openrouter_config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test",
            "OPENROUTER_HTTP_REFERER": "https://aso.example",
        }
    )
    payload = {"model": "gpt-5-mini-2025-08-07", "input": []}

    llm_provider.create_response(payload, openai_config)
    llm_provider.create_response(payload, equivalent_openai_config)
    llm_provider.create_response(payload, openrouter_config)
    llm_provider.create_response(payload, equivalent_openrouter_config)

    assert len(fake_openai.instances) == 2
    assert fake_openai.instances[0].kwargs == {
        "api_key": "openai-test",
        "base_url": "https://openai.example/v1",
    }
    assert fake_openai.instances[1].kwargs == {
        "api_key": "router-test",
        "base_url": "https://openrouter.ai/api/v1",
        "max_retries": 12,
        "default_headers": {
            "HTTP-Referer": "https://aso.example",
            "X-OpenRouter-Title": "MODAL-ASO",
        },
    }


def test_openrouter_client_uses_bounded_timeout_and_retry_policy(
    fake_openai: type[_FakeOpenAI],
) -> None:
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test",
            "OPENROUTER_TIMEOUT_SECONDS": "60",
            "OPENROUTER_MAX_RETRIES": "0",
        }
    )

    llm_provider.create_response(
        {"model": "gpt-5-mini-2025-08-07", "input": []}, config
    )

    assert fake_openai.instances[0].kwargs["timeout"] == 60.0
    assert fake_openai.instances[0].kwargs["max_retries"] == 0


def test_llm_call_uses_configured_cross_process_admission_resource(
    fake_openai: type[_FakeOpenAI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired: list[str] = []

    @contextmanager
    def fake_acquire_resource(resource: str):
        acquired.append(resource)
        yield {"resource": resource, "slot": 0, "limit": 1, "wait_seconds": 0}

    monkeypatch.setattr(llm_provider, "acquire_resource", fake_acquire_resource)
    monkeypatch.setenv("ASO_LLM_CALL_RESOURCE", "llm_dynamic_provider")
    config = llm_provider.get_provider_config(
        {
            "ASO_LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-test",
        }
    )

    llm_provider.create_response(
        {"model": "gpt-5-mini-2025-08-07", "input": []}, config
    )

    assert acquired == ["llm_dynamic_provider"]


def test_rotated_credential_creates_a_new_client_without_runtime_reset(
    fake_openai: type[_FakeOpenAI],
) -> None:
    first_key = "first-rotating-test-key"
    second_key = "second-rotating-test-key"
    first_config = llm_provider.get_provider_config({"OPENAI_API_KEY": first_key})
    second_config = llm_provider.get_provider_config({"OPENAI_API_KEY": second_key})
    payload = {"model": "gpt-5-mini-2025-08-07", "input": []}

    llm_provider.create_response(payload, first_config)
    llm_provider.create_response(payload, second_config)

    assert len(fake_openai.instances) == 2
    assert fake_openai.instances[0].kwargs["api_key"] == first_key
    assert fake_openai.instances[1].kwargs["api_key"] == second_key
    cache_keys = repr(list(llm_provider._CLIENT_CACHE))
    assert first_key not in cache_keys
    assert second_key not in cache_keys


def test_runtime_reset_discards_cached_client_without_secret_cache_keys(
    fake_openai: type[_FakeOpenAI],
) -> None:
    secret = "secret-cache-value"
    config = llm_provider.get_provider_config({"OPENAI_API_KEY": secret})
    payload = {"model": "gpt-5-mini-2025-08-07", "input": []}

    llm_provider.create_response(payload, config)
    assert secret not in repr(list(llm_provider._CLIENT_CACHE))

    llm_provider.reset_provider_runtime_for_tests()
    new_config = llm_provider.get_provider_config({"OPENAI_API_KEY": "new-test-key"})
    llm_provider.create_response(payload, new_config)

    assert len(fake_openai.instances) == 2
    assert fake_openai.instances[1].kwargs["api_key"] == "new-test-key"


def test_transport_errors_are_focused_and_redact_underlying_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secret-from-transport"

    class FailingOpenAI(_FakeOpenAI):
        def create(self, **kwargs: Any) -> Any:
            raise RuntimeError(f"upstream included {secret}")

    FailingOpenAI.instances = []
    monkeypatch.setattr(llm_provider, "OpenAI", FailingOpenAI)
    config = llm_provider.get_provider_config(
        {"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "router-test"}
    )

    with pytest.raises(RuntimeError) as exc_info:
        llm_provider.create_response(
            {"model": "gpt-5.2-2025-12-11", "input": []},
            config,
        )

    message = str(exc_info.value)
    assert "API transport" in message
    assert "openrouter" in message
    assert "gpt-5.2-2025-12-11" in message
    assert "openai/gpt-5.2" in message
    assert secret not in message
