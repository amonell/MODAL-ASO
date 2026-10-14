from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aso_agents import agentic_runner
from aso_agents import pipeline_tools


DESIGN_REQUEST = "design human MAPT gapmers that reduce 4R"


def test_hi_with_empty_history_does_not_close() -> None:
    assert pipeline_tools._intake_should_close_out_of_scope("hi", []) is False
    assert pipeline_tools._intake_should_close_out_of_scope("hi", None) is False
    assert pipeline_tools._text_looks_like_aso_task_request("hi") is False
    assert pipeline_tools._intake_should_ask_aso_restatement("hi", []) is True


def test_unrelated_non_aso_text_asks_restatement() -> None:
    unrelated = [
        "what's the weather in Paris?",
        "Write a python script to sort files",
        "design a website for my lab",
        "2+2",
    ]
    for text in unrelated:
        assert pipeline_tools._text_looks_like_in_scope_intake_request(text) is False, text
        assert pipeline_tools._intake_should_ask_aso_restatement(text, []) is True, text


def test_aso_work_and_aso_questions_do_not_force_restatement() -> None:
    in_scope = [
        DESIGN_REQUEST,
        "What does phyloP mean?",
        "Generate candidates against RBFOX2 eCLIP sites in K562",
        "Look at /tmp/runs/goal_1/report.md and explain the filters",
        "How do gapmers work?",
    ]
    for text in in_scope:
        assert pipeline_tools._text_looks_like_in_scope_intake_request(text) is True, text
        assert pipeline_tools._intake_should_ask_aso_restatement(text, []) is False, text


def test_unrelated_answer_after_restatement_closes() -> None:
    history = [
        {
            "question": pipeline_tools._ASO_INTAKE_RESTATEMENT_QUESTION,
            "answer": "what's the weather in Paris?",
        }
    ]
    assert pipeline_tools._intake_should_close_out_of_scope("hi", history) is True
    assert pipeline_tools._intake_should_ask_aso_restatement("hi", history) is False


def test_greeting_answer_question_envelope_is_treated_as_placeholder() -> None:
    envelope = {
        "request_summary": "Clarify requested ASO-related task (no actionable ASO request provided yet).",
        "request_classifications": ["intake", "needs_user_goal", "aso_unspecified", "answer_question"],
        "tasks": [
            {
                "task_id": "t1_answer",
                "task_type": "answer_question",
                "label": "Clarify ASO-related request",
            }
        ],
    }
    assert (
        pipeline_tools._envelope_uses_answer_question_as_missing_request_placeholder(envelope, "hi")
        is True
    )


def test_parse_greeting_pauses_for_restatement_instead_of_answering(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(pipeline_tools, "_reject_disallowed_tool_call", lambda **_kwargs: None)

    def fail_workbench(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("request workbench should not classify a greeting")

    monkeypatch.setattr(pipeline_tools, "_run_request_workbench_agent_once", fail_workbench)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {
        "version": "2",
        "status": "initialized",
        "phase": "initialized",
        "request": "hi",
        "run_dir": str(run_dir),
        "flags": {},
        "last_error": None,
        "pending_questions": [],
        "clarification": {"isoform_scope_confirmed": False},
        "clarification_qa_history": [],
        "force_rerun_stages": [],
        "events": [],
        "artifacts": {},
    }
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = pipeline_tools.tool_parse_goal_request(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))
    planning = pipeline_tools._derive_next_actions(saved_state)

    assert result["status"] == "ok"
    assert result.get("intake_partial") is True
    assert saved_state["status"] == "intake_partial"
    assert saved_state["pending_questions"]
    assert "ASO" in saved_state["pending_questions"][0]
    assert planning["terminal_status"] == "needs_user_input"


def test_parse_model_incomplete_is_retryable_failure_not_clarification(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        pipeline_tools,
        "_reject_disallowed_tool_call",
        lambda **_kwargs: None,
    )

    async def incomplete_workbench(_prompt: str) -> dict[str, Any]:
        return {
            "error": "incomplete_response",
            "status": "incomplete",
            "reason": "max_output_tokens",
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_request_workbench_agent_once",
        incomplete_workbench,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {
        "version": "2",
        "status": "initialized",
        "phase": "initialized",
        "request": "Design a steric ASO for a supplied target in reviewed cells.",
        "run_dir": str(run_dir),
        "flags": {},
        "last_error": None,
        "pending_questions": [],
        "clarification": {"isoform_scope_confirmed": False},
        "clarification_qa_history": [],
        "force_rerun_stages": [],
        "events": [],
        "artifacts": {},
    }
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = pipeline_tools.tool_parse_goal_request(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "retryable_error"
    assert result["error"] == "request_workbench_model_failed"
    assert saved_state["status"] == "failed"
    assert saved_state["pending_questions"] == []
    assert saved_state["last_error"]["message"] == (
        "incomplete_response: max_output_tokens"
    )


def test_second_greeting_after_restatement_closes() -> None:
    history = [
        {
            "question": pipeline_tools._ASO_INTAKE_RESTATEMENT_QUESTION,
            "answer": "hello",
        }
    ]
    assert pipeline_tools._intake_should_close_out_of_scope("hi", history) is True
    assert pipeline_tools._intake_should_ask_aso_restatement("hi", history) is False


def test_later_design_answer_becomes_effective_request() -> None:
    history = [
        {"question": "Please restate an ASO-related task.", "answer": "nah no aso request"},
        {
            "question": "Please restate an ASO-related task.",
            "answer": DESIGN_REQUEST,
            "request_effect": "replace",
        },
    ]

    assert pipeline_tools._effective_intake_request_text("hi", history) == DESIGN_REQUEST
    assert pipeline_tools._intake_should_close_out_of_scope("hi", history) is False
    prompt = pipeline_tools._build_request_workbench_followup_prompt("hi", history, None)
    assert DESIGN_REQUEST in prompt
    assert "Original request: hi\n" not in prompt


def test_followup_prompt_carries_schema_bound_validation_feedback() -> None:
    envelope = {"tasks": [{"task_id": "design", "context": {}}]}
    feedback = [
        {
            "path": "/tasks/0/context/structured_goal",
            "reason": "canonical_field_value_invalid",
            "expected_schema": {"type": "object"},
        }
    ]

    prompt = pipeline_tools._build_request_workbench_followup_prompt(
        DESIGN_REQUEST,
        [],
        envelope,
        feedback,
    )

    assert '"path": "/tasks/0/context/structured_goal"' in prompt
    assert '"type": "object"' in prompt
    assert "correct every reported path" in prompt
    assert "resolver-derived labels" in prompt


def test_audit_transport_retry_resumes_preserved_workbench_envelope(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    goal = {
        "gene_symbol": "GENE1",
        "species": "human",
        "target_context": "reviewed cells",
        "intent": "binding",
        "design_programs": [
            {
                "program_id": "primary",
                "label": "primary",
                "modality": "steric",
                "intent": "binding",
                "objective": "bind the requested target",
                "desired_delta": "decrease",
            }
        ],
    }
    candidate = pipeline_tools._build_request_envelope(
        request_text=DESIGN_REQUEST,
        request_id="retry-case",
        structured_goal=goal,
        raw_envelope={
            "schema_version": "v3",
            "request_mode": "single_task",
            "request_summary": "Design the requested ASOs.",
            "request_classifications": ["design_new_asos"],
            "tasks": [
                {
                    "task_id": "design",
                    "task_type": "design_new_asos",
                    "label": "Design the requested ASOs",
                    "native": True,
                    "can_run_in_parallel": True,
                    "context": {
                        "gene_symbol": "GENE1",
                        "species": "human",
                        "target_context": "reviewed cells",
                        "structured_goal": goal,
                    },
                    "depends_on": [],
                }
            ],
            "task_dependencies": [],
            "final_outputs": ["report"],
            "response_mode": "report",
            "requires_execution": True,
            "assumptions": [],
        },
    )

    async def fail_workbench(_prompt: str) -> dict[str, Any]:
        raise AssertionError("a successful workbench envelope must not be regenerated")

    def accept_audit(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ready",
            "request_envelope": kwargs["request_envelope"],
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_verified_request_workbench_agent_once",
        fail_workbench,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_repair_request_envelope_if_needed",
        accept_audit,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": "2",
                "status": "failed",
                "phase": "tool_parse_goal_failed",
                "request": DESIGN_REQUEST,
                "run_dir": str(run_dir),
                "flags": {"skip_ensembl_check": True},
                "last_error": {
                    "kind": "goal_contract_audit_failed",
                    "failed_tool": "tool_parse_goal_request",
                    "stage": "intake",
                    "candidate_request_envelope": candidate,
                },
                "pending_questions": [],
                "clarification": {"isoform_scope_confirmed": False},
                "clarification_qa_history": [],
                "force_rerun_stages": [],
                "events": [],
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )

    result = pipeline_tools.tool_parse_goal_request(
        str(state_path),
        force_rerun=True,
    )
    assert result["status"] == "ok", result
    saved = pipeline_tools._load_state(result["state_path"])

    assert any(
        event.get("message")
        == "Resumed contract audit from the preserved workbench envelope."
        for event in saved["events"]
    )


def test_native_preseeded_design_task_reconciles_quoted_obligations(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    import run_goal_intake

    request = "Design four gapmers whose entire binding sites lie within one coding exon."
    goal = {"gene_symbol": "GENE1", "species": "human", "target_context": "cells",
            "intent": "knockdown", "design_programs": [{"program_id": "p1",
            "modality": "gapmer", "intent": "knockdown", "objective": request}]}
    envelope = pipeline_tools._build_request_envelope(
        request_text=request, request_id="preseeded-design", structured_goal=goal,
    )
    assert envelope["tasks"][0]["context"]["structured_goal"].get("target_region") is None
    monkeypatch.setattr(pipeline_tools, "_request_parse_output_from_envelope_preseed",
                        lambda *_args, **_kwargs: {"status": "ready", "questions": [],
                                                  "request_envelope": envelope})
    monkeypatch.setattr(pipeline_tools, "_repair_request_envelope_if_needed",
                        lambda **kwargs: {"status": "ready", "request_envelope": kwargs["request_envelope"]})
    calls = []

    async def reconcile(payload: dict[str, Any], source: str, history: list) -> dict[str, Any]:
        calls.append((source, history))
        result = json.loads(json.dumps(payload))
        result["structured_goal"]["target_region"] = "cds"
        result["structured_goal"]["design_programs"][0]["target_region"] = "cds"
        result["request_envelope"]["tasks"][0]["context"]["structured_goal"] = result["structured_goal"]
        result["request_obligation_audit"] = {"status": "accepted"}
        return result

    monkeypatch.setattr(run_goal_intake, "_audit_request_obligations", reconcile)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps({"version": "2", "status": "initialized",
        "phase": "initialized", "request": request, "run_dir": str(run_dir),
        "flags": {"skip_ensembl_check": True}, "last_error": None,
        "pending_questions": [], "clarification": {"isoform_scope_confirmed": False},
        "clarification_qa_history": [], "force_rerun_stages": [],
        "events": [], "artifacts": {}}))
    result = pipeline_tools.tool_parse_goal_request(str(state_path))
    assert result["status"] == "ok", result
    assert calls == [(request, [])]
    saved = json.loads((run_dir / "artifacts/intake/structured_goal.json").read_text())
    assert saved["target_region"] == "cds"
    assert saved["design_programs"][0]["target_region"] == "cds"


def test_option_choice_does_not_become_effective_request() -> None:
    history = [{"question": "Which option?", "answer": "option a"}]

    assert pipeline_tools._text_looks_like_short_option_choice("option a") is True
    assert pipeline_tools._text_looks_like_aso_task_request("option a") is False
    assert pipeline_tools._effective_intake_request_text("hi", history) == "hi"
    assert pipeline_tools._intake_should_close_out_of_scope("hi", history) is False


def test_decline_and_math_answers_close_intake() -> None:
    decline_history = [
        {"question": "Please restate an ASO-related task.", "answer": "nah no aso request"}
    ]
    math_history = [{"question": "Please restate an ASO-related task.", "answer": "2+2"}]

    assert pipeline_tools._intake_should_close_out_of_scope("hi", decline_history) is True
    assert pipeline_tools._intake_should_close_out_of_scope("hi", math_history) is True
    assert pipeline_tools._text_looks_like_aso_task_request("2+2") is False


def test_math_does_not_close_an_in_scope_design_request() -> None:
    history = [{"question": "Please provide the biological target context.", "answer": "2+2"}]

    assert pipeline_tools._intake_should_close_out_of_scope(DESIGN_REQUEST, history) is False


def test_planner_stops_on_out_of_scope_after_parse() -> None:
    state = {
        "version": "2",
        "status": "out_of_scope",
        "phase": "out_of_scope",
        "request": "hi",
        "last_tool": "tool_parse_goal_request",
        "last_error": None,
        "pending_questions": [],
        "clarification": {
            "isoform_scope_confirmed": False,
            "kind": "out_of_scope",
            "reason": "This request is outside ASO design, evaluation, or analysis.",
        },
        "clarification_qa_history": [
            {"question": "Please restate an ASO-related task.", "answer": "2+2"}
        ],
    }

    planning = pipeline_tools._derive_next_actions(state)

    assert planning["terminal_status"] == "out_of_scope"
    assert planning["next_actions"] == []
    assert planning["stop_reason"] == "intake_out_of_scope"


def test_infer_status_maps_out_of_scope(tmp_path: Path) -> None:
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": "2",
                "status": "out_of_scope",
                "phase": "out_of_scope",
                "request": "hi",
                "last_tool": "tool_parse_goal_request",
                "last_error": None,
                "pending_questions": [],
                "clarification": {
                    "isoform_scope_confirmed": False,
                    "kind": "out_of_scope",
                    "reason": "This request is outside ASO design, evaluation, or analysis.",
                },
            }
        ),
        encoding="utf-8",
    )

    inferred = agentic_runner._infer_status_from_state(state_path)

    assert inferred["status"] == "out_of_scope"
    assert "outside ASO" in inferred["reason"]


def test_apply_clarification_closes_out_of_scope_on_non_aso_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    question = "Please restate an ASO-related task."
    answer = "2+2"
    state = {
        "status": "needs_user_input",
        "phase": "awaiting_clarification",
        "request": "hi",
        "run_dir": str(run_dir),
        "clarification": {
            "isoform_scope_confirmed": False,
            "pending_apply": True,
            "restart_stage": "intake",
            "last_answers": [{"question": question, "answer": answer}],
        },
        "pending_questions": [question],
        "clarification_qa_history": [],
        "force_rerun_stages": [],
    }
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(pipeline_tools, "provider_credentials_available", lambda: True)
    monkeypatch.setattr(
        pipeline_tools,
        "_run_clarification_request_scope_sync",
        lambda _payload: {
            "decision": "close_out_of_scope",
            "replacement_answer_index": None,
            "reason": "The whole-task restatement was answered with an unrelated request.",
        },
    )

    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert result.get("out_of_scope") is True
    assert saved_state["status"] == "out_of_scope"
    assert saved_state["pending_questions"] == []
    assert saved_state["request"] == "hi"


def test_apply_clarification_pivots_later_design_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    question = "Please restate an ASO-related task."
    state = {
        "status": "needs_user_input",
        "phase": "awaiting_clarification",
        "request": "hi",
        "run_dir": str(run_dir),
        "clarification": {
            "isoform_scope_confirmed": False,
            "pending_apply": True,
            "restart_stage": "intake",
            "last_answers": [{"question": question, "answer": DESIGN_REQUEST}],
        },
        "pending_questions": [question],
        "clarification_qa_history": [
            {"question": question, "answer": "nah no aso request"}
        ],
        "force_rerun_stages": [],
    }
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(pipeline_tools, "provider_credentials_available", lambda: True)
    monkeypatch.setattr(
        pipeline_tools,
        "_run_clarification_request_scope_sync",
        lambda _payload: {
            "decision": "replace_current_request",
            "replacement_answer_index": 0,
            "reason": "The answer supplies a complete replacement to a whole-task restatement.",
        },
    )

    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert result.get("out_of_scope") is not True
    assert saved_state["status"] != "out_of_scope"
    assert saved_state["request"] == DESIGN_REQUEST


def test_target_mapping_clarification_augments_original_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original_request = "Design steric ASOs for the requested transcript feature in reviewed cells."
    question = "What exact intended target feature mapping should be used?"
    answer = "Use transcript feature index 4 at the reviewed interval on the plus strand."
    state = {
        "status": "needs_user_input",
        "phase": "awaiting_clarification",
        "request": original_request,
        "run_dir": str(run_dir),
        "clarification": {
            "isoform_scope_confirmed": False,
            "kind": "target_feature_mapping",
            "pending_apply": True,
            "restart_stage": "intake",
            "last_answers": [{"question": question, "answer": answer}],
        },
        "pending_questions": [question],
        "clarification_qa_history": [],
        "force_rerun_stages": [],
    }
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(pipeline_tools, "provider_credentials_available", lambda: True)
    monkeypatch.setattr(
        pipeline_tools,
        "_run_clarification_request_scope_sync",
        lambda _payload: {
            "decision": "augment_current_request",
            "replacement_answer_index": None,
            "reason": "The answer provides the feature mapping requested by the clarification.",
        },
    )

    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert saved_state["request"] == original_request
    assert saved_state["clarification_qa_history"][-1]["request_effect"] == "augment"
    assert pipeline_tools._effective_intake_request_text(
        original_request,
        saved_state["clarification_qa_history"],
    ) == original_request


def test_fact_answer_cannot_replace_full_request_when_scope_model_misclassifies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original_request = (
        "Design conserved gapmers; export /output/panel.csv and /output/panel.md."
    )
    question = "What target site identifier have you selected?"
    answer = "I have not selected a site. Please find suitable conserved sites."
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps({
        "status": "needs_user_input", "phase": "awaiting_clarification",
        "request": original_request, "run_dir": str(run_dir),
        "clarification": {
            "pending_apply": True, "restart_stage": "intake",
            "last_answers": [{"question": question, "answer": answer}],
        },
        "pending_questions": [question], "clarification_qa_history": [],
        "force_rerun_stages": [],
    }), encoding="utf-8")
    monkeypatch.setattr(pipeline_tools, "provider_credentials_available", lambda: True)
    monkeypatch.setattr(
        pipeline_tools,
        "_run_clarification_request_scope_sync",
        lambda _payload: {
            "decision": "replace_current_request",
            "replacement_answer_index": 0,
            "reason": "Model incorrectly treats this parameter answer as a new request.",
        },
    )

    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    saved = pipeline_tools._load_state(str(state_path))
    assert result["status"] == "ok"
    assert saved["request"] == original_request
    assert saved["clarification_qa_history"][-1]["request_effect"] == "augment"
    assert saved["clarification"]["request_scope_override"]["classifier_decision"]["decision"] == "replace_current_request"
