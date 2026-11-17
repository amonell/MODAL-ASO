"""Presentation configuration and live checks, separate from scientific logic."""
from dataclasses import dataclass, field
import os
import re

BASES = {"openai": "https://api.openai.com/v1", "openrouter": "https://openrouter.ai/api/v1"}
ALL_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
VALIDATION_TIMEOUT_SECONDS = 40
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash-1017"


class SetupError(ValueError):
    def __init__(self, field, message):
        self.field = field
        super().__init__(message)


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str = field(repr=False)
    model: str = "default"
    effort: str = "default"
    timeout_seconds: float | None = None
    max_retries: int | None = None


def environment_for(settings, base=None):
    """Translate GUI choices into existing CLI variables, without rewriting requests."""
    if settings.provider not in BASES:
        raise SetupError("Provider", "Select OpenAI or OpenRouter.")
    if not settings.api_key.strip():
        raise SetupError("API key", "Paste your API key.")
    env = dict(os.environ if base is None else base)
    env["ASO_LLM_PROVIDER"] = settings.provider
    env["OPENAI_API_KEY" if settings.provider == "openai" else "OPENROUTER_API_KEY"] = settings.api_key.strip()
    if settings.provider == "openai":
        if settings.model != "default" or settings.effort != "default" or settings.timeout_seconds is not None or settings.max_retries is not None:
            raise SetupError("Configuration", "Direct OpenAI uses the existing CLI per-agent settings. Select inherited settings; global GUI overrides are unavailable.")
        return env
    if settings.model != "default":
        env["OPENROUTER_MODEL"] = settings.model
    if settings.effort != "default":
        env["OPENROUTER_REASONING_EFFORT"] = settings.effort
    if settings.timeout_seconds is not None:
        env["OPENROUTER_TIMEOUT_SECONDS"] = str(settings.timeout_seconds)
    if settings.max_retries is not None:
        env["OPENROUTER_MAX_RETRIES"] = str(settings.max_retries)
    return env


def configuration_preview(settings, base=None):
    """Show effective CLI configuration without exposing credentials."""
    from aso_agents.llm_provider import get_provider_config
    env = environment_for(settings, base)
    config = get_provider_config(env)
    return {"Provider": config.name, "API endpoint": config.base_url,
            "Model": config.model_override or "Existing per-agent CLI models",
            "Reasoning effort": config.reasoning_effort_override or "Existing per-agent CLI reasoning settings",
            "Network timeout (seconds)": config.request_timeout_seconds or "Existing SDK default",
            "SDK retries": config.max_retries if config.max_retries is not None else "Existing CLI/SDK default",
            "Provider routing": config.provider_preferences,
            "Role settings from environment": {key: value for key, value in env.items()
                if key.startswith("ASO_") and key.endswith(("_MODEL", "_REASONING_EFFORT"))}}


def efforts(model):
    """Conservative documented presets; unknown models use provider defaults."""
    name = model.removeprefix("openai/")
    if re.fullmatch(r"gpt-5\.2(?:-\d{4}-\d{2}-\d{2})?", name):
        return ["default", "none", "low", "medium", "high", "xhigh"]
    if re.fullmatch(r"gpt-5(?:-mini|-nano)?(?:-\d{4}-\d{2}-\d{2})?", name):
        return ["default", "minimal", "low", "medium", "high"]
    return ["default"]


def catalog_efforts(row):
    reasoning = row.get("reasoning")
    if not isinstance(reasoning, dict) or "supported_efforts" not in reasoning:
        return ["default"]
    supported = reasoning["supported_efforts"]
    allowed = ALL_EFFORTS if supported is None else supported
    return ["default"] + [value for value in ALL_EFFORTS if value in allowed
                           and not (value == "none" and reasoning.get("mandatory"))]


def api_error(exc):
    from openai import APITimeoutError
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return SetupError("Timeout", "The provider did not respond within the 40-second network timeout. No automatic retry was made. Retry validation; if it repeats, check your connection or choose another model. This does not establish that your API key is wrong.")
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None) or {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    param = str(error.get("param", "")) if isinstance(error, dict) else ""
    code = str(error.get("code", "")) if isinstance(error, dict) else ""
    if status == 401:
        return SetupError("API key", "Authentication failed. Check the key for the selected provider.")
    if status == 402 or code == "insufficient_quota":
        return SetupError("Billing", "Insufficient credits or quota. Check your provider account.")
    if status == 403:
        return SetupError("Access", "The provider denied access. Check key permissions and model access.")
    if status == 404 or "model" in param or "model" in code:
        return SetupError("Model", "The selected model is unavailable or inaccessible to this key.")
    if "reasoning" in param:
        return SetupError("Reasoning effort", "This model does not accept the selected reasoning effort.")
    if status == 429:
        return SetupError("Rate limit", "The provider is rate limiting requests. Retry later.")
    if status in (400, 422):
        return SetupError("Model compatibility", "The model rejected the required tools, structured output, or reasoning settings.")
    return SetupError("Connection", "The provider could not complete the check. Check your connection and retry.")


def client_for(provider, key):
    if provider not in BASES:
        raise SetupError("Provider", "Select OpenAI or OpenRouter.")
    if not key.strip():
        raise SetupError("API key", "Paste your API key.")
    from openai import OpenAI
    from aso_agents.llm_provider import get_provider_config
    config = get_provider_config(environment_for(Settings(provider, key)))
    return OpenAI(api_key=key.strip(), base_url=config.base_url,
                  default_headers=config.default_headers,
                  timeout=VALIDATION_TIMEOUT_SECONDS, max_retries=0)


def list_models(provider, key):
    client = client_for(provider, key)
    try:
        if provider == "openrouter":
            import requests
            from aso_agents.llm_provider import get_provider_config
            config = get_provider_config(environment_for(Settings(provider, key)))
            response = requests.get(config.base_url.rstrip("/") + "/models",
                                    headers={**config.default_headers, "Authorization": "Bearer " + key.strip()}, timeout=30)
            if response.status_code != 200:
                error = RuntimeError()
                error.status_code = response.status_code
                raise error
            rows = response.json()["data"]
            return dict(sorted((row["id"], catalog_efforts(row)) for row in rows
                               if {"tools", "structured_outputs"}.issubset(row.get("supported_parameters", []))))
        # The models endpoint has no capability flags: expose only known tool/
        # structured-output families, then verify actual account compatibility.
        return dict(sorted((row.id, efforts(row.id)) for row in client.models.list()
                           if re.fullmatch(r"gpt-(?:5\.2|5(?:-mini|-nano)?)(?:-\d{4}-\d{2}-\d{2})?", row.id)))
    except SetupError:
        raise
    except Exception as exc:
        raise api_error(exc) from None


def validate(settings, models=(), client=None, base=None):
    env = environment_for(settings, base)
    if (settings.model != "default" and settings.model not in models
            and not (settings.provider == "openrouter" and settings.model == DEFAULT_OPENROUTER_MODEL)):
        raise SetupError("Model", "Select a model from this provider's refreshed catalog.")
    allowed_efforts = models[settings.model] if isinstance(models, dict) and settings.model in models else efforts(settings.model)
    if settings.effort != "default" and settings.effort not in allowed_efforts:
        raise SetupError("Reasoning effort", "Select an effort supported by this model.")
    from aso_agents.llm_provider import get_provider_config, prepare_responses_payload
    try:
        config = get_provider_config(env)
    except (ValueError, RuntimeError) as exc:
        raise SetupError("Configuration", str(exc)) from None
    # This is a setup probe, not an analysis. Its short timeout/retry policy
    # never changes the CLI configuration used by the analysis worker.
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=config.api_key, base_url=config.base_url,
                        default_headers=config.default_headers,
                        timeout=VALIDATION_TIMEOUT_SECONDS, max_retries=0)
    payload = {"model": "gpt-5.2-2025-12-11", "input": "Call gui_check with ok=true.",
        "tools": [{"type": "function", "name": "gui_check", "description": "Check configuration",
                   "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                  "required": ["ok"], "additionalProperties": False}, "strict": True}],
        "tool_choice": {"type": "function", "name": "gui_check"},
        "text": {"format": {"type": "json_schema", "name": "check", "strict": True,
                 "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"], "additionalProperties": False}}}}
    try:
        import json
        prepared, _ = prepare_responses_payload(payload, config=config)
        response = client.responses.create(**prepared)
        calls = [item for item in response.output if item.type == "function_call" and item.name == "gui_check"]
        if not calls or json.loads(calls[0].arguments) != {"ok": True}:
            raise SetupError("Model compatibility", "The model did not return the required structured tool call.")
    except SetupError:
        raise
    except Exception as exc:
        raise api_error(exc) from None
