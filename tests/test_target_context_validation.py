from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aso_agents import pipeline_tools
from aso_agents.run_layout import resolve_run_path


def _seed_goal_state(tmp_path: Path, target_context: Any) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "state" / "run_state.json"
    state_path.parent.mkdir()
    structured_goal: dict[str, Any] = {
        "gene_symbol": "CDKN1B",
        "species": "human",
        "intent": "binding",
        "target_region": "utr3",
        "design_programs": [
            {
                "program_id": "steric_cdkn1b",
                "modality": "steric",
                "intent": "binding",
                "target_region": "utr3",
            }
        ],
    }
    if target_context is not None:
        structured_goal["target_context"] = target_context
    final_payload = dict(structured_goal)
    payload = {"request": "Design CDKN1B 3' UTR target-site blocker ASOs."}
    pipeline_tools._save_goal_payload(str(run_dir), payload, structured_goal, final_payload)
    state = {
        "version": "2",
        "status": "intake_ready",
        "phase": "intake_ready",
        "request": payload["request"],
        "flags": {},
        "run_dir": str(run_dir),
        "artifacts": {},
        "events": [],
        "last_error": None,
        "pending_questions": [],
        "clarification": {"isoform_scope_confirmed": False},
        "clarification_qa_history": [],
        "force_rerun_stages": [],
        "stage_attempts": {},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path, run_dir


def test_goal_schema_accepts_explicit_general_target_context(monkeypatch: Any, tmp_path: Path) -> None:
    state_path, run_dir = _seed_goal_state(
        tmp_path,
        "General human baseline context; no specific tissue, cell type, disease, or model system.",
    )
    monkeypatch.setattr(pipeline_tools, "_reject_disallowed_tool_call", lambda **_kwargs: None)

    result = pipeline_tools.tool_validate_goal_schema(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))
    saved_goal = json.loads(resolve_run_path(run_dir, "structured_goal.json").read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["validation"]["ok"] is True
    assert saved_state["status"] == "goal_schema_ready"
    assert saved_state["phase"] == "goal_schema_ready"
    assert saved_state["pending_questions"] == []
    assert saved_goal["goal_schema_validation"]["ok"] is True
    assert saved_goal["goal_schema_validation"].get("issues", []) == []


def test_target_context_clarification_accepts_bare_general_when_prompt_offers_baseline() -> None:
    question = (
        "Please provide the biological target context for this ASO design (human SMN2): "
        "tissue and/or cell type, plus disease/model/experimental system if relevant. "
        "If you genuinely want a broad baseline context, state that explicitly."
    )

    resolved = pipeline_tools._target_context_from_clarification_answers(
        [{"question": question, "answer": "general"}]
    )

    assert resolved == "general human baseline"


def test_goal_schema_blocks_missing_target_context(monkeypatch: Any, tmp_path: Path) -> None:
    state_path, run_dir = _seed_goal_state(tmp_path, None)
    monkeypatch.setattr(pipeline_tools, "_reject_disallowed_tool_call", lambda **_kwargs: None)

    result = pipeline_tools.tool_validate_goal_schema(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))
    saved_goal = json.loads(resolve_run_path(run_dir, "structured_goal.json").read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["needs_user_input"] is True
    assert result["reason"] == "target_context_missing"
    assert saved_state["status"] == "needs_user_input"
    assert saved_state["phase"] == "awaiting_clarification"
    assert saved_state["pending_clarification_restart_stage"] == "intake"
    assert saved_state["pending_questions"]
    assert "biological target context" in saved_state["pending_questions"][0]
    assert saved_goal["goal_schema_validation"]["ok"] is False
    assert saved_goal["goal_schema_validation"]["issues"] == ["target_context_missing"]


def test_goal_schema_leaves_nonempty_context_semantics_to_llm_review(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    state_path, run_dir = _seed_goal_state(
        tmp_path,
        "SMN2 pre-mRNA splice-switching to increase exon 7 inclusion.",
    )
    monkeypatch.setattr(pipeline_tools, "_reject_disallowed_tool_call", lambda **_kwargs: None)

    result = pipeline_tools.tool_validate_goal_schema(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))
    saved_goal = json.loads(resolve_run_path(run_dir, "structured_goal.json").read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["validation"]["ok"] is True
    assert saved_state["status"] == "goal_schema_ready"
    assert saved_state["pending_questions"] == []
    assert saved_goal["goal_schema_validation"].get("issues", []) == []


def test_goal_schema_accepts_specific_target_context(monkeypatch: Any, tmp_path: Path) -> None:
    state_path, _run_dir = _seed_goal_state(tmp_path, "human dermal fibroblasts in progeria")
    monkeypatch.setattr(pipeline_tools, "_reject_disallowed_tool_call", lambda **_kwargs: None)

    result = pipeline_tools.tool_validate_goal_schema(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert result["validation"]["ok"] is True
    assert saved_state["status"] == "goal_schema_ready"
    assert saved_state["phase"] == "goal_schema_ready"
    assert saved_state["pending_questions"] == []


def test_goal_schema_reconciles_single_task_field_before_required_check(
    monkeypatch: Any, tmp_path: Path,
) -> None:
    state_path, run_dir = _seed_goal_state(tmp_path, "general human baseline")
    goal_path = resolve_run_path(run_dir, "goal_intake.json")
    payload = json.loads(goal_path.read_text())
    task_goal = dict(payload["final_payload"]["structured_goal"])
    envelope = {"tasks": [{"task_id": "design", "task_type": "design_new_asos",
                           "context": {"structured_goal": task_goal}}]}
    payload["request_envelope"] = envelope
    payload["final_payload"]["request_envelope"] = envelope
    payload["final_payload"]["structured_goal"].pop("intent")
    payload["final_payload"].pop("intent", None)
    goal_path.write_text(json.dumps(payload))
    resolve_run_path(run_dir, "structured_goal.json").write_text(
        json.dumps(payload["final_payload"]["structured_goal"])
    )
    monkeypatch.setattr(pipeline_tools, "_reject_disallowed_tool_call", lambda **_kwargs: None)

    result = pipeline_tools.tool_validate_goal_schema(str(state_path))
    saved_goal = json.loads(resolve_run_path(run_dir, "structured_goal.json").read_text())
    assert result["status"] == "ok"
    assert result["validation"]["ok"] is True
    assert saved_goal["intent"] == "binding"
    assert saved_goal["goal_schema_validation"]["ok"] is True


def test_request_validation_allows_explicit_general_design_context() -> None:
    envelope = {
        "request_mode": "composite",
        "tasks": [
            {
                "task_id": "design_cdkn1b",
                "task_type": "design_new_asos",
                "context": {
                    "structured_goal": {
                        "gene_symbol": "CDKN1B",
                        "species": "human",
                        "intent": "binding",
                        "target_context": "general human baseline",
                        "design_programs": [{"program_id": "steric", "modality": "steric"}],
                    }
                },
            }
        ],
    }

    validation = pipeline_tools._request_validation_summary(envelope)

    assert validation["tasks"]["design_cdkn1b"]["status"] == "runnable"
    assert validation["blocked_task_ids"] == []


def test_request_validation_blocks_design_task_without_target_context() -> None:
    envelope = {
        "request_mode": "composite",
        "tasks": [
            {
                "task_id": "design_cdkn1b",
                "task_type": "design_new_asos",
                "context": {
                    "structured_goal": {
                        "gene_symbol": "CDKN1B",
                        "species": "human",
                        "intent": "binding",
                        "design_programs": [{"program_id": "steric", "modality": "steric"}],
                    }
                },
            }
        ],
    }

    validation = pipeline_tools._request_validation_summary(envelope)

    assert validation["tasks"]["design_cdkn1b"]["status"] == "blocked"
    assert "target_context_missing" in validation["tasks"]["design_cdkn1b"]["issues"]
    assert "design_cdkn1b" in validation["blocked_task_ids"]


def test_request_validation_treats_nonempty_context_as_structurally_present() -> None:
    envelope = {
        "request_mode": "composite",
        "tasks": [
            {
                "task_id": "design_smn2",
                "task_type": "design_new_asos",
                "context": {
                    "structured_goal": {
                        "gene_symbol": "SMN2",
                        "species": "human",
                        "intent": "splice_modulation",
                        "target_context": "SMN2 pre-mRNA splice-switching to increase exon 7 inclusion",
                        "design_programs": [{"program_id": "steric", "modality": "steric"}],
                    }
                },
            }
        ],
    }

    validation = pipeline_tools._request_validation_summary(envelope)

    assert validation["tasks"]["design_smn2"]["status"] == "runnable"
    assert "target_context_missing" not in validation["tasks"]["design_smn2"]["issues"]
    assert "design_smn2" not in validation["blocked_task_ids"]


def test_merge_validation_questions_preserves_target_context_issue() -> None:
    merged = pipeline_tools._merge_validation_questions_into_intake(
        [
            "Which sites/regions should be treated as separate groups for 'per site' candidate returns?",
            "If using tiling, what exon 7 flanking window should be included?",
        ],
        {
            "clarification_queue": [
                {
                    "missing_fields": [
                        "target_context_missing",
                        "context.structured_goal.region_constraints (concrete site/region groups for 'per site')",
                    ],
                    "questions": [
                        "Please provide the biological target context for this ASO design (human SMN2): tissue and/or cell type, plus disease/model/experimental system if relevant.",
                        "Which sites/regions should be treated as separate groups for 'per site' candidate returns?",
                    ],
                }
            ]
        },
    )

    assert any("biological target context" in question for question in merged)
    assert any("sites/regions" in question for question in merged)


def test_request_validation_allows_specific_design_context() -> None:
    envelope = {
        "request_mode": "composite",
        "tasks": [
            {
                "task_id": "design_cdkn1b",
                "task_type": "design_new_asos",
                "context": {
                    "structured_goal": {
                        "gene_symbol": "CDKN1B",
                        "species": "human",
                        "intent": "binding",
                        "target_context": "human dermal fibroblasts in progeria",
                        "design_programs": [{"program_id": "steric", "modality": "steric"}],
                    }
                },
            }
        ],
    }

    validation = pipeline_tools._request_validation_summary(envelope)

    assert validation["tasks"]["design_cdkn1b"]["status"] == "runnable"
    assert validation["blocked_task_ids"] == []


def test_apply_clarification_updates_placeholder_design_task_context(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    question = (
        "Please provide the biological target context for this ASO design (human CDKN1B): "
        "tissue and/or cell type, plus disease/model/experimental system if relevant. "
        "If you genuinely want a broad baseline context, state that explicitly."
    )
    answer = "Human HEK293T cells in an in vitro CDKN1B/p27 mRNA and protein upregulation assay."
    envelope = {
        "request_mode": "single_task",
        "default_context": {},
        "blocked_tasks": [
            {
                "task_id": "design_cdkn1b",
                "missing_fields": ["target_context"],
                "questions": [question],
            }
        ],
        "tasks": [
            {
                "task_id": "design_cdkn1b",
                "task_type": "design_new_asos",
                "context": {
                    "gene_symbol": "CDKN1B",
                    "species": "human",
                    "target_context": {
                        "description": "unknown"
                    },
                    "structured_goal": {
                        "gene_symbol": "CDKN1B",
                        "species": "human",
                        "intent": "binding",
                        "target_context": {
                            "description": "unknown"
                        },
                        "off_target_expression": {
                            "target_context": "unknown"
                        },
                            "design_programs": [
                                {
                                    "program_id": "steric",
                                    "label": "steric design",
                                    "modality": "steric",
                                    "intent": "binding",
                                    "objective": "bind the supplied target",
                                    "desired_delta": "decrease",
                                    "target_context": "unknown",
                                }
                        ],
                    },
                },
            }
        ],
    }
    state = {
        "status": "needs_user_input",
        "phase": "awaiting_clarification",
        "request": "Design CDKN1B target-site blocker ASOs.",
        "request_envelope": envelope,
        "clarification": {
            "isoform_scope_confirmed": False,
            "kind": "goal_intake",
            "pending_apply": True,
            "restart_stage": "intake",
            "last_answers": [{"question": question, "answer": answer}],
        },
        "pending_questions": [question],
        "clarification_qa_history": [],
        "task_validation": {
            "design_cdkn1b": {
                "status": "blocked",
                "issues": ["target_context_missing"],
                "questions": [question],
                "context": envelope["tasks"][0]["context"],
            }
        },
        "task_statuses": {"design_cdkn1b": {"status": "blocked"}},
        "task_artifacts": {"design_cdkn1b": {}},
        "blocked_task_ids": ["design_cdkn1b"],
        "clarification_queue": [],
    }
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    async def fake_audit(_prompt: str) -> dict[str, Any]:
        assert answer in _prompt
        return {"status": "valid", "patches": [], "unresolved": []}

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )

    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))
    saved_envelope = saved_state["request_envelope"]
    saved_task = saved_envelope["tasks"][0]
    saved_goal = saved_task["context"]["structured_goal"]

    assert result["status"] == "ok"
    assert saved_state["clarification"]["extracted_target_context"] == answer
    assert saved_task["context"]["target_context"] == answer
    assert saved_goal["target_context"] == answer
    assert saved_goal["off_target_expression"]["target_context"] == answer
    assert saved_goal["design_programs"][0]["target_context"] == answer
    assert "blocked_tasks" not in saved_envelope
    assert pipeline_tools._request_validation_summary(saved_envelope)["blocked_task_ids"] == []


def test_apply_generic_clarification_clears_answered_pending_questions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    question = "What should count as a separate site?"
    answer = "Use acceptor, donor, and ISS-N1 windows as separate target sites."
    state = {
        "status": "needs_user_input",
        "phase": "awaiting_clarification",
        "request": "Design SMN2 ASOs.",
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

    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert saved_state["status"] == "initialized"
    assert saved_state["phase"] == "restart_intake_pending"
    assert saved_state["pending_questions"] == []
    assert saved_state["clarification_qa_history"] == [
        {
            "question": question,
            "answer": answer,
            "request_effect": "augment",
        }
    ]


def test_apply_transcript_scope_clarification_updates_splice_objectives(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    structured_goal = {
        "gene_symbol": "SMN2",
        "species": "human",
        "intent": "splice_modulation",
        "target_context": "human spinal cord motor neurons in SMA model",
        "desired_effect": "inclusion",
        "splice_isoform_objectives": [
            {"desired_delta": "increase_inclusion", "exon_label": "exon 7"}
        ],
    }
    pipeline_tools._save_goal_payload(
        str(run_dir),
        {"request": "Design SMN2 exon 7 inclusion ASOs."},
        dict(structured_goal),
        dict(structured_goal),
    )
    question = "Which transcript or isoform IDs should define the SMN2 exon 7 splice objective?"
    answer = (
        "Use ENST00000626847 as the exon 7 skipped isoform to decrease and "
        "ENST00000380743 as the exon 7 included isoform to increase."
    )
    state = {
        "status": "needs_user_input",
        "phase": "awaiting_clarification",
        "request": "Design SMN2 exon 7 inclusion ASOs.",
        "run_dir": str(run_dir),
        "clarification": {
            "isoform_scope_confirmed": False,
            "kind": "isoform_transcript_scope",
            "pending_apply": True,
            "restart_stage": "resolution",
            "last_answers": [{"question": question, "answer": answer}],
        },
        "pending_questions": [question],
        "clarification_qa_history": [],
        "force_rerun_stages": [],
    }
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))
    saved_goal = json.loads(resolve_run_path(run_dir, "structured_goal.json").read_text(encoding="utf-8"))
    by_tx = {objective["transcript_id"]: objective for objective in saved_goal["splice_isoform_objectives"]}

    assert result["status"] == "ok"
    assert saved_state["status"] == "intake_ready"
    assert saved_state["phase"] == "restart_resolution_pending"
    assert saved_state["pending_questions"] == []
    assert set(by_tx) == {"ENST00000626847", "ENST00000380743"}
    assert by_tx["ENST00000626847"]["desired_abundance"] == "decrease"
    assert by_tx["ENST00000380743"]["desired_abundance"] == "increase"


def test_apply_transcript_scope_clarification_accepts_all_other_isoforms_answer(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    structured_goal = {
        "gene_symbol": "SMN2",
        "species": "human",
        "intent": "splice_modulation",
        "target_context": "human spinal cord motor neurons in SMA model",
        "desired_effect": "inclusion",
        "splice_isoform_objectives": [
            {"desired_delta": "increase_inclusion", "exon_label": "exon 7"}
        ],
    }
    pipeline_tools._save_goal_payload(
        str(run_dir),
        {"request": "Design SMN2 exon 7 inclusion ASOs."},
        dict(structured_goal),
        dict(structured_goal),
    )
    question = "Which transcript or isoform IDs should define the SMN2 exon 7 splice objective?"
    answer = (
        "Ensembl ID ENST00000380743 is the isoform which needs to increase. "
        "All other isoforms of SMN2 can decrease."
    )
    state = {
        "status": "needs_user_input",
        "phase": "awaiting_clarification",
        "request": "Design SMN2 exon 7 inclusion ASOs.",
        "run_dir": str(run_dir),
        "clarification": {
            "isoform_scope_confirmed": False,
            "kind": "isoform_transcript_scope",
            "pending_apply": True,
            "restart_stage": "resolution",
            "last_answers": [{"question": question, "answer": answer}],
        },
        "pending_questions": [question],
        "clarification_qa_history": [],
        "force_rerun_stages": [],
    }
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))
    saved_goal = json.loads(resolve_run_path(run_dir, "structured_goal.json").read_text(encoding="utf-8"))
    by_tx = {objective["transcript_id"]: objective for objective in saved_goal["splice_isoform_objectives"]}

    assert result["status"] == "ok"
    assert saved_state["status"] == "intake_ready"
    assert saved_state["phase"] == "restart_resolution_pending"
    assert saved_state["pending_questions"] == []
    assert set(by_tx) == {"ENST00000380743"}
    assert by_tx["ENST00000380743"]["desired_abundance"] == "increase"
    assert by_tx["ENST00000380743"]["conversion_role"] == "to"


def test_child_target_context_question_resolves_from_parent_history() -> None:
    question = "Please provide a concrete target_context for CDKN1B ASO design."
    answer = "Human HEK293T cells in an in vitro CDKN1B/p27 mRNA and protein upregulation assay."

    resolved = pipeline_tools._resolve_child_clarification_from_inherited_context(
        parent_request="Design CDKN1B ASOs.",
        parent_qa_history=[{"question": question, "answer": answer}],
        task={"task_id": "design_cdkn1b", "task_type": "design_new_asos"},
        task_context={"gene_symbol": "CDKN1B", "species": "human"},
        child_questions=[question],
        resolver=lambda _payload: {"status": "unresolved", "answers": []},
    )

    assert resolved == [{"question": question, "answer": answer}]


def test_child_clarification_resolver_receives_completed_dependency_results() -> None:
    question = "Which source-backed target interval should this design use?"
    dependency_results = [
        {
            "upstream_task_id": "analyze_reference",
            "task_type": "analyze_artifact",
            "status": "completed",
            "output_payload": {
                "answer_summary": "Use chr4:3001-3080 on the plus strand.",
                "evidence_refs": ["inputs/reference.json"],
            },
            "output_artifacts": {},
        }
    ]
    captured: dict[str, object] = {}

    def resolver(payload: dict[str, object]) -> dict[str, object]:
        captured.update(payload)
        return {
            "status": "resolved",
            "answers": [
                {
                    "question": question,
                    "answer": "Use chr4:3001-3080 on the plus strand from inputs/reference.json.",
                }
            ],
            "unresolved_questions": [],
            "reason": "The completed dependency contains the requested fact.",
        }

    resolved = pipeline_tools._resolve_child_clarification_from_inherited_context(
        parent_request="Design ASOs using the analyzed reference.",
        parent_qa_history=[],
        task={"task_id": "design_target", "task_type": "design_new_asos"},
        task_context={"gene_symbol": "GENE1", "species": "human"},
        child_questions=[question],
        direct_artifact_context={
            "entries": [
                {
                    "path": "inputs/reference.json",
                    "exists": True,
                    "excerpt": '{"chrom":"chr4","start":3001,"end":3080}',
                }
            ]
        },
        completed_dependency_results=dependency_results,
        resolver=resolver,
    )

    assert captured["completed_dependency_results"] == dependency_results
    assert captured["direct_artifact_context"] == {
        "entries": [
            {
                "path": "inputs/reference.json",
                "exists": True,
                "excerpt": '{"chrom":"chr4","start":3001,"end":3080}',
            }
        ]
    }
    assert resolved == [
        {
            "question": question,
            "answer": "Use chr4:3001-3080 on the plus strand from inputs/reference.json.",
        }
    ]


_TISSUE_QUESTION = (
    "Please provide the biological target context for this ASO design (human MAPT): "
    "tissue and/or cell type, plus disease/model/experimental system if relevant. "
    "If you genuinely want a broad baseline context, state that explicitly."
)


def test_you_determine_maps_to_baseline_only_for_tissue_question() -> None:
    resolved = pipeline_tools._target_context_from_clarification_answers(
        [{"question": _TISSUE_QUESTION, "answer": "you determine"}]
    )

    assert resolved == "general human baseline"


def test_you_determine_on_exon_question_does_not_set_tissue() -> None:
    resolved = pipeline_tools._target_context_from_clarification_answers(
        [
            {
                "question": "Which exon or genomic window should I tile for 4R-selective gapmers?",
                "answer": "you determine",
            }
        ]
    )

    assert resolved == ""


def test_you_determine_on_target_context_named_exon_question_does_not_set_tissue() -> None:
    resolved = pipeline_tools._target_context_from_clarification_answers(
        [
            {
                "question": "What is the target_context exon/window for this isoform-selective design?",
                "answer": "you determine",
            }
        ]
    )

    assert resolved == ""


def _isoform_class_gapmer_envelope(*, target_context: str | None) -> dict[str, Any]:
    structured_goal: dict[str, Any] = {
        "gene_symbol": "MAPT",
        "species": "human",
        "intent": "knockdown",
        "splice_isoform_objectives": [
            {
                "objective_scope": "specific_isoform",
                "isoform_label": "4R",
                "desired_abundance": "decrease",
                "role": "pathogenic",
            }
        ],
        "design_programs": [
            {
                "program_id": "gapmer_mapt_4r",
                "modality": "gapmer",
                "intent": "knockdown",
            }
        ],
    }
    if target_context is not None:
        structured_goal["target_context"] = target_context
    return {
        "request_mode": "single_task",
        "tasks": [
            {
                "task_id": "design_mapt_4r",
                "task_type": "design_new_asos",
                "context": {
                    "gene_symbol": "MAPT",
                    "species": "human",
                    **({"target_context": target_context} if target_context is not None else {}),
                    "structured_goal": structured_goal,
                },
            }
        ],
    }


def test_request_validation_allows_isoform_class_gapmer_without_exon() -> None:
    envelope = _isoform_class_gapmer_envelope(target_context="general human baseline")

    validation = pipeline_tools._request_validation_summary(envelope)
    issues = validation["tasks"]["design_mapt_4r"]["issues"]
    questions = " ".join(validation["tasks"]["design_mapt_4r"]["questions"]).lower()

    assert validation["tasks"]["design_mapt_4r"]["status"] == "runnable"
    assert validation["blocked_task_ids"] == []
    assert not any("exon" in issue or "window" in issue or "tiling" in issue for issue in issues)
    assert "exon" not in questions
    assert "window" not in questions


def test_request_validation_still_blocks_isoform_class_gapmer_without_tissue() -> None:
    envelope = _isoform_class_gapmer_envelope(target_context=None)

    validation = pipeline_tools._request_validation_summary(envelope)

    assert validation["tasks"]["design_mapt_4r"]["status"] == "blocked"
    assert "target_context_missing" in validation["tasks"]["design_mapt_4r"]["issues"]
    assert "design_mapt_4r" in validation["blocked_task_ids"]
    assert any(
        "biological target context" in question
        for question in validation["tasks"]["design_mapt_4r"]["questions"]
    )
