from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from aso_agents import pipeline_tools
from aso_agents.llm_provider import (
    get_provider_config,
    provider_credential_help,
    provider_credentials_available,
)


_PROVIDER_CONFIG = get_provider_config(require_credentials=False)
_REAL_PROVIDER_CREDENTIAL = (
    provider_credentials_available()
    and _PROVIDER_CONFIG.api_key.strip().lower() != "test-only"
)
pytestmark = pytest.mark.skipif(
    os.environ.get("ASO_RUN_API_SMOKE") != "1"
    or not _REAL_PROVIDER_CREDENTIAL,
    reason=(
        "Set ASO_RUN_API_SMOKE=1 with a real selected-provider credential to run API smoke tests. "
        f"{provider_credential_help()}"
    ),
)


SMOKE_CASES: list[dict[str, Any]] = [
    {
        "name": "large_design_eval_compare_trace",
        "request": (
            "Design human MAPT gapmer ASOs and mouse Mapt gapmer ASOs, evaluate these two "
            "provided ASOs ACGTACGTACGTACGTACGT and TGCATGCATGCATGCATGCA against the generated "
            "sets, compare the generated human and mouse ASO sets, trace where my provided ASOs "
            "drop out of filtering, and write one synthesis."
        ),
        "expected_task_types": {
            "design_new_asos",
            "evaluate_existing_aso_set",
            "compare_aso_sets",
            "trace_filter_decisions",
            "build_composite_report",
        },
        "expect_composite": True,
    },
    {
        "name": "question_design_artifact_combo",
        "request": (
            "First explain what phyloP means for ASO prioritization, then design human LMNA "
            "knockdown gapmers, and also inspect /tmp/agentaso_missing_run to explain why that "
            "prior run failed if it exists."
        ),
        "expected_task_types": {"answer_question", "design_new_asos", "analyze_artifact", "build_composite_report"},
        "expect_composite": True,
    },
    {
        "name": "cross_species_rank_if_present",
        "request": (
            "Design human SMN2 splice-switching ASOs for exon 7 inclusion, rank conserved mouse "
            "sites if present, but still return viable human candidates even when conservation is absent."
        ),
        "expected_task_types": {"design_new_asos"},
    },
    {
        "name": "missing_existing_aso_sequences",
        "request": "Evaluate my ASOs against newly generated human MAPT candidates and tell me where they fail filtering.",
        "expected_task_types": {"design_new_asos", "evaluate_existing_aso_set"},
    },
    {
        "name": "artifact_analysis_missing_path",
        "request": "Look at /tmp/definitely_missing_agentaso_run and explain why the MAPT outputs differ from the report.",
        "expected_task_types": {"analyze_artifact"},
    },
    {
        "name": "many_parallel_designs_and_comparison",
        "request": (
            "For human MAPT, PCCA, and LMNA, design gapmer knockdown ASOs as separate runs, "
            "then compare the resulting candidate sets and summarize which design has the strongest top candidates."
        ),
        "expected_task_types": {"design_new_asos", "compare_aso_sets", "build_composite_report"},
        "expect_composite": True,
    },
    {
        "name": "custom_analysis_maps_to_supported_task_or_clarifies",
        "request": (
            "Using these ASOs ACGTACGTACGTACGTACGT and TGCATGCATGCATGCATGCA, do a custom "
            "small analysis that groups them by GC content and flags anything outside a typical ASO range."
        ),
        "expected_any_task_types": {"evaluate_existing_aso_set", "sandbox_analysis", "answer_question"},
    },
    {
        "name": "four_designs_compare_and_synthesize",
        "request": (
            "Design separate human gapmer knockdown ASO sets for MAPT, PCCA, LMNA, and CDKN1B, "
            "compare the generated candidate sets, and write one synthesis about which target has "
            "the strongest top-ranked candidates."
        ),
        "expected_task_types": {"design_new_asos", "compare_aso_sets", "build_composite_report"},
        "expect_composite": True,
    },
    {
        "name": "ambiguous_splice_request_survives_planning",
        "request": (
            "Design human splice-switching ASOs for DMD exon skipping, ask me if you need the "
            "exact exon or transcript, and do not silently choose an arbitrary isoform."
        ),
        "expected_task_types": {"design_new_asos"},
    },
    {
        "name": "question_missing_artifact_and_deferred_sequences",
        "request": (
            "Explain why ASO GC range matters, inspect /tmp/agentaso_missing_filter_trace if it exists, "
            "and evaluate my lab ASOs once I provide the sequences."
        ),
        "expected_any_task_types": {"answer_question", "analyze_artifact", "evaluate_existing_aso_set"},
        "expect_composite": True,
    },
]


def _run_request_workbench(request: str) -> dict[str, Any]:
    return asyncio.run(pipeline_tools._run_request_workbench_agent_once(request))


def _dependency_pairs(envelope: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in envelope.get("task_dependencies") or []:
        if not isinstance(item, dict):
            continue
        upstream = str(item.get("upstream_task_id") or "").strip()
        downstream = str(item.get("downstream_task_id") or "").strip()
        if upstream and downstream:
            pairs.add((upstream, downstream))
    return pairs


def _normalized_envelope(parsed: dict[str, Any], request: str, tmp_path: Path) -> dict[str, Any]:
    raw_envelope = parsed.get("request_envelope") if isinstance(parsed.get("request_envelope"), dict) else {}
    if not raw_envelope:
        return {}
    envelope = pipeline_tools._build_request_envelope(
        request_text=request,
        request_id="api_smoke",
        structured_goal={},
        raw_envelope=raw_envelope,
    )
    if str(parsed.get("status") or "").strip().lower() == "ready":
        repaired = pipeline_tools._repair_request_envelope_if_needed(
            request_text=request,
            run_dir=str(tmp_path),
            structured_goal={},
            request_envelope=envelope,
            qa_history=[],
        )
        if repaired.get("status") == "ready" and isinstance(repaired.get("request_envelope"), dict):
            envelope = repaired["request_envelope"]
    return envelope


def _planning_for_envelope(envelope: dict[str, Any], request: str, tmp_path: Path) -> dict[str, Any]:
    state = {
        "request": request,
        "status": "request_parsed_ready",
        "phase": "request_parsed_ready",
        "run_dir": str(tmp_path),
        "request_envelope": envelope,
        "last_error": None,
        "clarification": {"isoform_scope_confirmed": False},
        "pending_questions": [],
        "clarification_qa_history": [],
    }
    pipeline_tools._ensure_v2_request_metadata(state)
    return pipeline_tools._derive_next_actions(state)


@pytest.mark.parametrize("case", SMOKE_CASES, ids=[case["name"] for case in SMOKE_CASES])
def test_request_workbench_api_smoke(case: dict[str, Any], tmp_path: Path) -> None:
    parsed = _run_request_workbench(case["request"])
    assert isinstance(parsed, dict)
    assert str(parsed.get("status") or "").strip().lower() in {"ready", "need_clarification"}

    envelope = _normalized_envelope(parsed, case["request"], tmp_path)
    assert envelope, json.dumps(parsed, indent=2)

    tasks = [task for task in envelope.get("tasks") or [] if isinstance(task, dict)]
    assert tasks
    task_ids = [str(task.get("task_id") or "").strip() for task in tasks]
    task_types = {str(task.get("task_type") or "").strip() for task in tasks}

    assert len(task_ids) == len(set(task_ids))
    assert all(task_id for task_id in task_ids)
    assert all(task_type == task_type.strip().lower().replace("-", "_").replace(" ", "_") for task_type in task_types)

    expected_task_types = case.get("expected_task_types")
    if isinstance(expected_task_types, set):
        assert expected_task_types.issubset(task_types)
    expected_any_task_types = case.get("expected_any_task_types")
    if isinstance(expected_any_task_types, set):
        assert task_types.intersection(expected_any_task_types)
    if case.get("expect_composite"):
        assert envelope.get("request_mode") == "composite"
        assert envelope.get("response_mode") == "composite"

    pairs = _dependency_pairs(envelope)
    assert all(upstream != downstream for upstream, downstream in pairs)

    validation = pipeline_tools._request_validation_summary(envelope)
    assert not validation.get("orphan_blocked_task_ids")

    planning = _planning_for_envelope(envelope, case["request"], tmp_path)
    assert planning.get("terminal_status") != "fatal", json.dumps(
        {"parsed": parsed, "envelope": envelope, "planning": planning},
        indent=2,
    )
