from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools


@pytest.mark.parametrize(
    ("status", "reason", "answer"),
    [
        ("unresolved", "patient_variant_identity_incomplete", "Use the supplied genomic coordinates."),
        ("unresolved", "patient_variant_not_resolved_by_ensembl_vep", "Use the corrected notation from my attachment."),
        ("conflict", "patient_variant_reference_allele_conflicts_with_ensembl", "The reference allele is C, not A."),
        ("unresolved", "patient_variant_transcript_mapping_unresolved", "Use the transcript accession specified in the original request."),
    ],
)
def test_variant_answer_restarts_intake_and_preserves_request(
    tmp_path: Path, monkeypatch, status: str, reason: str, answer: str
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "state.json"
    request = "Design allele-specific ASOs and export the complete library and selected panel."
    state = {
        "status": "goal_schema_ready",
        "phase": "goal_schema_ready",
        "request": request,
        "run_dir": str(run_dir),
        "active_runner": {"tool": "tool_resolve_isoforms_and_sites"},
        "force_rerun_stages": [],
    }
    state_path.write_text(json.dumps(state))
    goal = {
        "gene_symbol": "EXAMPLE",
        "species": "human",
        "target_context": "general human computational baseline",
        "patient_variant": {"kind": "unknown", "notation": "unresolved user input"},
    }
    context = {"payload": {"request": request}, "structured_goal": goal,
               "final_payload": {"structured_goal": goal}, "request_envelope": {}}
    monkeypatch.setattr(pipeline_tools, "provider_credentials_available", lambda: False)
    monkeypatch.setattr(
        pipeline_tools, "_prepare_execution_stage",
        lambda **kwargs: {"status": "ok", "state": state, "run_dir": str(run_dir)},
    )
    monkeypatch.setattr(pipeline_tools, "_load_goal_context_from_run_dir", lambda _: context)
    monkeypatch.setattr(
        pipeline_tools, "_goal_runtime_module",
        lambda: SimpleNamespace(_resolve_patient_variant_context=lambda *args: {"status": status, "reason": reason}),
    )

    result = pipeline_tools.tool_resolve_isoforms_and_sites(str(state_path))
    assert result["needs_user_input"] is True
    assert result["reason"] == reason
    pending = pipeline_tools._load_state(str(state_path))
    # No unresolved identity has been accepted as valid while requesting facts.
    assert "patient_variant_resolution" not in context["final_payload"]
    assert pending["clarification"]["restart_stage"] == "intake"

    recorded = pipeline_tools.submit_clarification_answers(str(state_path), [answer])
    assert recorded["restart_stage"] == "intake"
    applied = pipeline_tools.apply_clarification_to_goal(str(state_path))
    assert applied["restart_stage"] == "intake"
    saved = pipeline_tools._load_state(str(state_path))
    assert saved["request"] == request
    assert saved["clarification_qa_history"][-1]["answer"] == answer
    assert saved["clarification_qa_history"][-1]["request_effect"] == "augment"
    assert saved["pending_clarification_restart_stage"] == "intake"

    restarted = pipeline_tools.restart_from_stage(str(state_path), "intake", "Apply clarified variant identity.")
    assert restarted["status"] == "ok"
    saved = pipeline_tools._load_state(str(state_path))
    assert saved["status"] == "initialized"
    assert saved["phase"] == "restart_intake_pending"
    planning = pipeline_tools._derive_next_actions(saved)
    assert "tool_parse_goal_request" in planning["allowed_next_tools"]
    assert "tool_resolve_isoforms_and_sites" not in planning["allowed_next_tools"]
    prompt = pipeline_tools._build_request_workbench_followup_prompt(
        request, saved["clarification_qa_history"]
    )
    assert answer in prompt and request in prompt
