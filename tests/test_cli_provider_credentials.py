from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
GOAL_CLI = ROOT / "run_goal_intake.py"
REPORT_REGENERATION = ROOT / "scripts" / "regenerate_report_outputs.py"

_PROVIDER_ENV_NAMES = (
    "ASO_LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_HTTP_REFERER",
    "OPENROUTER_APP_TITLE",
    "OPENROUTER_ZDR",
    "OPENROUTER_DATA_COLLECTION",
    "OPENROUTER_ONLY_PROVIDERS",
    "OPENROUTER_PROVIDER_ORDER",
    "OPENROUTER_PROVIDER_SORT",
    "OPENROUTER_REASONING_EFFORT",
    "OPENROUTER_MAX_OUTPUT_TOKENS",
    "OPENROUTER_TIMEOUT_SECONDS",
    "OPENROUTER_MAX_RETRIES",
)
_CREDENTIAL_NAMES = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _clean_provider_env(provider: str | None) -> dict[str, str]:
    env = dict(os.environ)
    for name in _PROVIDER_ENV_NAMES:
        env.pop(name, None)
    if provider is not None:
        env["ASO_LLM_PROVIDER"] = provider
    return env


def _run(script: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _combined_output(proc: subprocess.CompletedProcess[str]) -> str:
    return f"{proc.stdout}\n{proc.stderr}"


def _load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_cli_help_works_without_provider_credentials(provider: str) -> None:
    proc = _run(GOAL_CLI, "--help", env=_clean_provider_env(provider))

    assert proc.returncode == 0, _combined_output(proc)
    assert "usage" in proc.stdout.lower()
    assert "API_KEY" not in _combined_output(proc)


@pytest.mark.parametrize(
    ("provider", "credential_name"),
    [
        (None, "OPENAI_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_cli_accepts_the_selected_provider_credential(
    provider: str | None,
    credential_name: str,
) -> None:
    env = _clean_provider_env(provider)
    env[credential_name] = "selected-provider-test"

    proc = _run(GOAL_CLI, "--credential-gate-test-only", env=env)
    output = _combined_output(proc)

    assert proc.returncode == 2, output
    assert "unrecognized arguments" in output
    assert "Configure the selected provider" not in output


@pytest.mark.parametrize(
    ("provider", "unselected_credential", "required_credential"),
    [
        (None, "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        ("openai", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        ("openrouter", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    ],
)
def test_cli_rejects_missing_selected_credential_even_when_other_key_exists(
    provider: str | None,
    unselected_credential: str,
    required_credential: str,
) -> None:
    env = _clean_provider_env(provider)
    env[unselected_credential] = "unselected-provider-test"

    proc = _run(GOAL_CLI, "--credential-gate-test-only", env=env)
    output = _combined_output(proc)

    assert proc.returncode == 1, output
    assert f"Set {required_credential}" in output
    assert "unselected-provider-test" not in output




@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("existing_value", [None, ""])
def test_report_regeneration_no_llm_prepares_only_selected_provider_placeholder(
    provider: str,
    existing_value: str | None,
) -> None:
    module = _load_script(REPORT_REGENERATION, f"report_regeneration_{provider}")
    env: dict[str, str] = {"ASO_LLM_PROVIDER": provider}
    if existing_value is not None:
        env[_CREDENTIAL_NAMES[provider]] = existing_value

    module._prepare_provider_credentials(no_llm=True, env=env)

    selected_credential = _CREDENTIAL_NAMES[provider]
    unselected_provider = "openrouter" if provider == "openai" else "openai"
    assert env[selected_credential] == "REPORT_REGEN_REUSE_ONLY"
    assert _CREDENTIAL_NAMES[unselected_provider] not in env


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_report_regeneration_no_llm_runs_without_credentials(
    provider: str,
    tmp_path: Path,
) -> None:
    from aso_agents.run_layout import canonical_run_path

    run_dir = tmp_path / f"goal_report_regeneration_{provider}"
    run_dir.mkdir()
    (run_dir / "request.txt").write_text("Explain this completed test run.", encoding="utf-8")
    structured_goal_path = canonical_run_path(run_dir, "structured_goal.json")
    structured_goal_path.parent.mkdir(parents=True, exist_ok=True)
    structured_goal_path.write_text(
        '{"gene_symbol":"MAPT","species":"human","intent":"knockdown"}',
        encoding="utf-8",
    )
    explainer_path = canonical_run_path(run_dir, "report_explainer.json")
    explainer_path.write_text(
        """{
  "summary": {
    "headline": "Existing report explanation",
    "plain_language_goal": "Explain the completed test run.",
    "what_happened": "Existing outputs were reused.",
    "result": "Report files were regenerated."
  },
  "walkthrough": [
    {
      "title": "Reuse existing explanation",
      "lane": "shared",
      "summary": "The saved explanation was loaded without an LLM call.",
      "why_it_matters": "Report rendering remains provider-independent.",
      "evidence_refs": ["report_explainer.json"]
    }
  ]
}
""",
        encoding="utf-8",
    )

    proc = _run(
        REPORT_REGENERATION,
        str(run_dir),
        "--no-llm",
        env=_clean_provider_env(provider),
    )

    assert proc.returncode == 0, _combined_output(proc)
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "report.html").is_file()
    assert "Explainer mode: reused existing report_explainer.json" in proc.stdout


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_report_regeneration_requires_selected_credential_for_llm(provider: str) -> None:
    module = _load_script(REPORT_REGENERATION, f"report_regeneration_missing_{provider}")
    env = {
        "ASO_LLM_PROVIDER": provider,
        _CREDENTIAL_NAMES["openrouter" if provider == "openai" else "openai"]: "other-key",
    }

    with pytest.raises(SystemExit, match=_CREDENTIAL_NAMES[provider]):
        module._prepare_provider_credentials(no_llm=False, env=env)


def test_model_metadata_uses_provider_resolution_and_response_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_goal_intake

    monkeypatch.setenv("ASO_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")

    assert (
        run_goal_intake._model_attribution_from_trace("gpt-5.2-2025-12-11", [])
        == "openai/gpt-5.2"
    )
    assert (
        run_goal_intake._model_attribution_from_trace(
            "gpt-5.2-2025-12-11",
            [
                {
                    "responses": [
                        {
                            "response": {
                                "provider": "openrouter",
                                "requested_model": "gpt-5.2-2025-12-11",
                                "resolved_model": "openai/gpt-5.2-2025-12-11",
                                "model": "provider-normalized-model",
                            }
                        }
                    ]
                }
            ],
        )
        == "openai/gpt-5.2-2025-12-11"
    )


@pytest.mark.parametrize(
    ("path", "flag"),
    [
        (ROOT / "tests" / "test_request_workbench_api_smoke.py", "ASO_RUN_API_SMOKE"),
        (ROOT / "tests" / "test_full_cli_smoke.py", "ASO_RUN_FULL_SMOKE"),
    ],
)
@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_live_smoke_skip_reason_names_selected_provider_credential(
    path: Path,
    flag: str,
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASO_LLM_PROVIDER", provider)
    for credential_name in _CREDENTIAL_NAMES.values():
        monkeypatch.delenv(credential_name, raising=False)
    monkeypatch.setenv(_CREDENTIAL_NAMES[provider], "test-only")
    monkeypatch.delenv(flag, raising=False)

    module = _load_script(path, f"smoke_skip_{path.stem}_{provider}")
    mark = module.pytestmark.mark
    reason = str(mark.kwargs.get("reason") or "")

    assert _CREDENTIAL_NAMES[provider] in reason
    assert f"ASO_LLM_PROVIDER={provider}" in reason


