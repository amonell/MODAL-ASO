"""Presentation configuration and live checks, separate from scientific logic."""
from dataclasses import dataclass, field
import os
from datetime import date
from aso_agents.model_defaults import DEFAULT_OPENROUTER_MODEL

BASES = {"openai": "https://api.openai.com/v1", "openrouter": "https://openrouter.ai/api/v1"}
ALL_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
VALIDATION_TIMEOUT_SECONDS = 40


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


def credential_name(provider):
    return f"{provider.upper()}_API_KEY"


def environment_for(settings, base=None):
    """Translate GUI choices into existing CLI variables, without rewriting requests."""
    if settings.provider not in BASES:
        raise SetupError("Provider", "Select OpenAI or OpenRouter.")
    env = dict(os.environ if base is None else base)
    variable = credential_name(settings.provider)
    key = settings.api_key.strip() or env.get(variable, "").strip()
    if not key:
        raise SetupError("API key", f"Paste your API key or set {variable} in the terminal that starts the GUI.")
    env["ASO_LLM_PROVIDER"] = settings.provider
    env[variable] = key
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
            "Transport retries": env.get("ASO_LLM_TRANSPORT_RETRIES", "0") if config.name == "openrouter" else "0",
            "Provider routing": config.provider_preferences,
            "Role settings from environment": {key: value for key, value in env.items()
                if key.startswith("ASO_") and key.endswith(("_MODEL", "_REASONING_EFFORT"))}}


def model_family(model):
    name = model.removeprefix("openai/")
    for family in ("gpt-5.2", "gpt-5-mini", "gpt-5-nano", "gpt-5"):
        if name == family:
            return family
        if name.startswith(family + "-"):
            suffix = name[len(family) + 1:]
            try:
                if len(suffix) == 10 and date.fromisoformat(suffix).isoformat() == suffix:
                    return family
            except ValueError:
                pass
    return None


def efforts(model):
    """Conservative documented presets; unknown models use provider defaults."""
    family = model_family(model)
    if family == "gpt-5.2":
        return ["default", "none", "low", "medium", "high", "xhigh"]
    if family in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
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
        return SetupError("Timeout", "The provider exceeded the 40-second network timeout. Retry validation; if it repeats, check your connection or choose another model.")
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
                           if model_family(row.id)))
    except SetupError:
        raise
    except Exception as exc:
        raise api_error(exc) from None
