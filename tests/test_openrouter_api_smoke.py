from __future__ import annotations

import os

import pytest

from aso_agents.llm_provider import get_provider_config
from scripts import check_llm_provider


pytestmark = pytest.mark.skipif(
    not check_llm_provider.openrouter_smoke_ready(os.environ),
    reason=(
        "Set ASO_LLM_PROVIDER=openrouter, ASO_RUN_OPENROUTER_SMOKE=1, and a real "
        "OPENROUTER_API_KEY to run OpenRouter API smoke tests."
    ),
)


@pytest.mark.parametrize(
    "target",
    check_llm_provider.DEFAULT_LOGICAL_MODEL_TARGETS,
    ids=("strong_logical_model", "fast_logical_model"),
)
def test_openrouter_single_turn_strict_json(target: check_llm_provider.LogicalModelTarget) -> None:
    config = get_provider_config()

    record = check_llm_provider.run_single_turn_check(target, config)

    if record.provider != "openrouter":
        raise AssertionError("OpenRouter smoke used the wrong provider.")
    if record.requested_model != target.logical_model:
        raise AssertionError("OpenRouter smoke used the wrong logical model.")
    if record.resolved_model != config.resolve_model(target.logical_model):
        raise AssertionError("OpenRouter smoke resolved the logical model incorrectly.")
    if record.status != "completed" or not record.output_types:
        raise AssertionError("OpenRouter single-turn smoke did not complete with output.")


def test_openrouter_function_tool_round_trip() -> None:
    record = check_llm_provider.run_function_tool_check()

    if record.status != "completed":
        raise AssertionError("OpenRouter function-tool smoke did not complete.")


@pytest.mark.skipif(
    not check_llm_provider.web_smoke_enabled(os.environ),
    reason="Set ASO_RUN_OPENROUTER_WEB_SMOKE=1 in addition to the base smoke opt-in.",
)
def test_openrouter_hosted_web_search() -> None:
    check_llm_provider.run_hosted_web_search_check()
