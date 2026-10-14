from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from aso_agents import pipeline_tools
from aso_agents.llm_provider import (
    get_provider_config,
    provider_credential_help,
    provider_credentials_available,
)
from aso_agents.run_layout import resolve_run_path


ROOT = Path(__file__).resolve().parents[1]

_PROVIDER_CONFIG = get_provider_config(require_credentials=False)
_REAL_PROVIDER_CREDENTIAL = (
    provider_credentials_available()
    and _PROVIDER_CONFIG.api_key.strip().lower() != "test-only"
)
pytestmark = pytest.mark.skipif(
    os.environ.get("ASO_RUN_FULL_SMOKE") != "1"
    or not _REAL_PROVIDER_CREDENTIAL,
    reason=(
        "Set ASO_RUN_FULL_SMOKE=1 with a real selected-provider credential to run full CLI smoke tests. "
        f"{provider_credential_help()}"
    ),
)


def _reference_root() -> str:
    configured = os.environ.get("ASO_FULL_SMOKE_REFERENCE_DATA_ROOT") or os.environ.get("ASO_REFERENCE_DATA_ROOT")
    if configured:
        path = Path(configured).expanduser()
    else:
        pytest.skip("Set ASO_FULL_SMOKE_REFERENCE_DATA_ROOT or ASO_REFERENCE_DATA_ROOT for full smoke tests.")
    if not path.exists():
        pytest.skip(f"Full smoke reference data root not found: {path}")
    return str(path)


def _species_tiling_references_available(species: str) -> bool:
    pipeline_tools.set_reference_data_root(_reference_root())
    return all(Path(path).exists() for path in pipeline_tools._required_tiling_reference_files(species))


def _run_agentic_cli(request: str, *, timeout: int = 1800) -> dict[str, Any]:
    env = dict(os.environ)
    env["ASO_REFERENCE_DATA_ROOT"] = _reference_root()
    env.setdefault("ASO_OFFTARGET_TIMEOUT_SEC", "5")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_goal_intake.py"),
            "--skip-ensembl-check",
            request,
        ],
        cwd=str(ROOT),
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, combined
    for line in reversed(combined.splitlines()):
        if line.startswith("AGENTIC_RUN_RESULT: "):
            parsed = json.loads(line.split(": ", 1)[1])
            assert isinstance(parsed, dict)
            return parsed
    raise AssertionError(f"AGENTIC_RUN_RESULT not found in CLI output:\n{combined}")


def _assert_successful_run(result: dict[str, Any]) -> Path:
    assert result.get("status") == "run_complete", json.dumps(result, indent=2)
    integrity = result.get("integrity") if isinstance(result.get("integrity"), dict) else {}
    assert integrity.get("ok") is True, json.dumps(result, indent=2)
    run_dir_raw = result.get("run_dir")
    assert isinstance(run_dir_raw, str) and run_dir_raw.strip()
    run_dir = Path(run_dir_raw)
    assert (run_dir / "report.md").is_file()
    assert resolve_run_path(run_dir, "report.html").is_file()
    assert resolve_run_path(run_dir, "generated_aso_set.json").is_file()
    root_names = {path.name for path in run_dir.iterdir() if path.name not in {".DS_Store"}}
    assert root_names <= {"request.txt", "report.md", "report.html", "artifacts", "child_runs"}
    return run_dir


def _generated_row_count(run_dir: Path) -> int:
    data = json.loads(resolve_run_path(run_dir, "generated_aso_set.json").read_text(encoding="utf-8"))
    if isinstance(data.get("row_count"), int):
        return int(data["row_count"])
    rows = data.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def test_full_human_mapt_gapmer_cli_run() -> None:
    result = _run_agentic_cli(
        "Design human MAPT gapmer ASOs to reduce MAPT expression in human cortical neurons in a tauopathy model."
    )
    run_dir = _assert_successful_run(result)
    assert _generated_row_count(run_dir) > 0


def test_full_mouse_mapt_gapmer_cli_run() -> None:
    if not _species_tiling_references_available("mouse"):
        pytest.skip("Full mouse MAPT smoke requires local mouse tiling reference assets.")
    result = _run_agentic_cli(
        "Design mouse Mapt gapmer ASOs to reduce Mapt expression in mouse cortical neurons in a tauopathy model."
    )
    run_dir = _assert_successful_run(result)
    assert _generated_row_count(run_dir) > 0


def test_missing_artifact_cli_pauses_with_single_question() -> None:
    missing_path = f"/tmp/agentaso_missing_artifact_{uuid.uuid4().hex}"
    result = _run_agentic_cli(
        "First explain what phyloP means for ASO prioritization, then inspect "
        f"{missing_path} and write one synthesis."
    )

    assert result.get("status") == "needs_user_input", json.dumps(result, indent=2)
    questions = result.get("pending_questions")
    assert isinstance(questions, list)
    non_empty_questions = [str(question).strip() for question in questions if str(question).strip()]
    assert 1 <= len(non_empty_questions) <= 2
    assert any(missing_path in question for question in non_empty_questions)


def test_human_splice_missing_assets_do_not_crash() -> None:
    result = _run_agentic_cli(
        "Design human SMN2 splice-switching ASOs for exon 7 inclusion using the canonical transcript."
    )

    status = str(result.get("status") or "").strip().lower()
    assert status in {"run_complete", "needs_user_input"}, json.dumps(result, indent=2)
    assert status != "fatal"
    if status == "run_complete":
        integrity = result.get("integrity") if isinstance(result.get("integrity"), dict) else {}
        assert integrity.get("ok") is True, json.dumps(result, indent=2)
    else:
        questions = result.get("pending_questions")
        assert isinstance(questions, list) and questions
