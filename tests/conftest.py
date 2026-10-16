from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing a submodule initializes aso_agents.__init__, so seed the selected
# test credential before package imports that may read configuration.
_PROVIDER_HINT = str(os.environ.get("ASO_LLM_PROVIDER") or "openai").strip().lower()
_CREDENTIAL_HINT = (
    "OPENROUTER_API_KEY" if _PROVIDER_HINT == "openrouter" else "OPENAI_API_KEY"
)
os.environ.setdefault(_CREDENTIAL_HINT, "test-only")

from aso_agents.llm_provider import selected_provider_name


_TEST_CREDENTIAL_NAMES = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
os.environ.setdefault(_TEST_CREDENTIAL_NAMES[selected_provider_name()], "test-only")


@pytest.fixture(autouse=True)
def _stub_runtime_binaries(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_bowtie_runtime") or request.node.get_closest_marker("real_runtime"):
        return

    def _ready() -> bool:
        return True

    def _ensure_bowtie(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "installed": False,
            "binaries": {"bowtie": "/usr/bin/bowtie", "bowtie-build": "/usr/bin/bowtie-build"},
        }

    def _ensure_snakemake(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "installed": False,
            "argv": ["/usr/bin/snakemake"],
            "executable": "/usr/bin/snakemake",
        }

    def _ensure_tiling(**_kwargs: object) -> dict[str, object]:
        return {"status": "ok", "installed": False, "missing": [], "tools": {}}

    def _ensure_runtime(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "missing": [],
            "bowtie": _ensure_bowtie(),
            "snakemake": _ensure_snakemake(),
            "tiling_cli": _ensure_tiling(),
        }

    monkeypatch.setattr("aso_agents.runtime_binaries.bowtie_1_ready", _ready)
    monkeypatch.setattr("aso_agents.runtime_binaries.ensure_bowtie_1", _ensure_bowtie)
    monkeypatch.setattr("aso_agents.runtime_binaries.snakemake_ready", _ready)
    monkeypatch.setattr("aso_agents.runtime_binaries.ensure_snakemake", _ensure_snakemake)
    monkeypatch.setattr("aso_agents.runtime_binaries.tiling_cli_tools_ready", _ready)
    monkeypatch.setattr("aso_agents.runtime_binaries.tiling_cli_missing", lambda: [])
    monkeypatch.setattr("aso_agents.runtime_binaries.ensure_tiling_cli_tools", _ensure_tiling)
    monkeypatch.setattr("aso_agents.runtime_binaries.ensure_run_runtime", _ensure_runtime)
    pipeline_tools = sys.modules.get("aso_agents.pipeline_tools")
    if pipeline_tools is not None:
        monkeypatch.setattr(pipeline_tools, "bowtie_1_ready", _ready)
        monkeypatch.setattr(pipeline_tools, "ensure_bowtie_1", _ensure_bowtie)
        monkeypatch.setattr(pipeline_tools, "snakemake_ready", _ready)
        monkeypatch.setattr(pipeline_tools, "ensure_snakemake", _ensure_snakemake)
        monkeypatch.setattr(pipeline_tools, "tiling_cli_tools_ready", _ready)
        monkeypatch.setattr(pipeline_tools, "tiling_cli_missing", lambda: [])
        monkeypatch.setattr(pipeline_tools, "ensure_run_runtime", _ensure_runtime)
        monkeypatch.setattr(pipeline_tools, "bind_snakemake_command", lambda command: str(command or ""))
        # Existing unit tests isolate external model calls. Contract-review
        # integration tests supply their own reviewer or use real_runtime
        # with a mocked provider; live qualification is separate.
        monkeypatch.setattr(pipeline_tools, "_run_analysis_contract_review_sync",
                            lambda payload: {"program_sha256": payload["program_sha256"], "findings": []})
