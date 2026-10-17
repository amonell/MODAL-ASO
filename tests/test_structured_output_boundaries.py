from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aso_agents import pipeline_tools
from aso_agents.async_utils import run_async_from_sync
from aso_agents.goal_intake_agent import goal_intake_clarification_review_agent
from aso_agents.goal_contract import (
    CANONICAL_CLARIFICATION_BOUNDARY_INSTRUCTIONS,
    canonical_goal_patch_value_schema,
    canonical_structured_goal_schema,
)
from aso_agents.structured_outputs import (
    DESIGN_GOAL_SYNTHESIS_OUTPUT_SCHEMA,
    GOAL_CONTRACT_AUDIT_OUTPUT_SCHEMA,
    GOAL_CONTRACT_REPAIR_OUTPUT_SCHEMA,
    REQUEST_GRAPH_REVIEW_OUTPUT_SCHEMA,
    REQUEST_WORKBENCH_OUTPUT_SCHEMA,
    normalize_strict_output,
    openai_strict_json_schema,
)


def _valid_workbench_payload() -> dict[str, Any]:
    return {
        "status": "ready",
        "questions": [],
        "reason": None,
        "request_envelope": {
            "schema_version": "v3",
            "request_mode": "single_task",
            "request_summary": "Evaluate one supplied ASO.",
            "request_classifications": ["evaluate_existing_aso_set"],
            "tasks": [
                {
                    "task_id": "evaluate_supplied_aso",
                    "task_type": "evaluate_existing_aso_set",
                    "label": "Evaluate supplied ASO",
                    "native": True,
                    "can_run_in_parallel": False,
                    "context": {
                        "gene_symbol": "GENE1",
                        "species": "Homo sapiens",
                        "target_context": "user-defined cells",
                        "provided_asos": ["ACGTACGTACGTACGTACGT"],
                    },
                }
            ],
            "task_dependencies": [],
            "final_outputs": ["ASO evaluation"],
            "response_mode": "analysis",
            "requires_execution": True,
            "assumptions": [],
        },
    }


async def _passthrough_completeness_review(
    _request_evidence: str,
    planned: dict[str, Any],
) -> dict[str, Any]:
    return planned


def _design_workbench_payload_with_goal(
    structured_goal: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "ready",
        "questions": [],
        "reason": None,
        "request_envelope": {
            "schema_version": "v3",
            "request_mode": "single_task",
            "request_summary": "Compile one design goal.",
            "request_classifications": ["design_new_asos"],
            "tasks": [
                {
                    "task_id": "design",
                    "task_type": "design_new_asos",
                    "label": "Compile design",
                    "native": True,
                    "can_run_in_parallel": False,
                    "context": {"structured_goal": structured_goal},
                }
            ],
            "task_dependencies": [],
            "final_outputs": ["design result"],
            "response_mode": "report",
            "requires_execution": True,
            "assumptions": [],
        },
    }


def test_design_goal_synthesis_recompiles_nonempty_stale_goal(
    monkeypatch: Any,
) -> None:
    stale_goal = {
        "design_programs": [
            {
                "program_id": "stale",
                "label": "Stale",
                "modality": "steric",
                "intent": "binding",
                "objective": "Incomplete prior objective.",
                "desired_delta": "decrease",
            }
        ]
    }
    replacement_goal = {
        "target_context": "user-supplied context",
        "design_programs": [
            {
                "program_id": "compiled",
                "label": "Compiled",
                "modality": "steric",
                "intent": "binding",
                "objective": "Complete requested objective.",
                "desired_delta": "increase",
            }
        ],
    }
    calls = {"count": 0}

    async def fake_synthesis(_prompt: str) -> dict[str, Any]:
        calls["count"] += 1
        return {
            "status": "ready",
            "questions": [],
            "reason": None,
            "structured_goal": deepcopy(replacement_goal),
        }

    async def no_revision_needed(
        _envelope: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        return {"blocked": False}

    monkeypatch.setattr(
        pipeline_tools,
        "_run_design_goal_synthesis_agent_once",
        fake_synthesis,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_design_goal_revision_feedback",
        no_revision_needed,
    )

    result = asyncio.run(
        pipeline_tools._synthesize_missing_design_goals(
            "request plus clarification evidence",
            _design_workbench_payload_with_goal(stale_goal),
        )
    )

    assert calls["count"] == 1
    assert result["request_envelope"]["tasks"][0]["context"][
        "structured_goal"
    ] == replacement_goal


def test_design_goal_synthesis_retries_from_schema_derived_validation_feedback(
    monkeypatch: Any,
) -> None:
    incomplete_goal = {
        "design_programs": [
            {
                "program_id": "branch",
                "label": "Branch",
                "modality": "steric",
                "intent": "binding",
                "objective": "Requested objective.",
                "desired_delta": "increase",
            }
        ]
    }
    complete_goal = {
        **deepcopy(incomplete_goal),
        "target_context": "clarified context",
    }
    prompts: list[str] = []
    feedback_calls = {"count": 0}

    async def fake_synthesis(prompt: str) -> dict[str, Any]:
        prompts.append(prompt)
        goal = incomplete_goal if len(prompts) == 1 else complete_goal
        return {
            "status": "ready",
            "questions": [],
            "reason": None,
            "structured_goal": deepcopy(goal),
        }

    async def fake_feedback(
        _envelope: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        assert task_id == "design"
        feedback_calls["count"] += 1
        if feedback_calls["count"] == 1:
            return {
                "blocked": True,
                "canonical_contract_issues": [
                    {
                        "json_pointer": (
                            "/tasks/0/context/structured_goal/target_context"
                        ),
                        "expected_schema": {"type": "string"},
                    }
                ],
                "validation_issue_ledger": [],
            }
        return {"blocked": False}

    monkeypatch.setattr(
        pipeline_tools,
        "_run_design_goal_synthesis_agent_once",
        fake_synthesis,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_design_goal_revision_feedback",
        fake_feedback,
    )

    result = asyncio.run(
        pipeline_tools._synthesize_missing_design_goals(
            "request plus clarification evidence",
            _design_workbench_payload_with_goal({}),
        )
    )

    assert len(prompts) == 2
    assert "/tasks/0/context/structured_goal/target_context" in prompts[1]
    assert result["request_envelope"]["tasks"][0]["context"][
        "structured_goal"
    ] == complete_goal


def test_model_output_schemas_compile_to_closed_portable_strict_contracts() -> None:
    def assert_portable(node: Any) -> None:
        if not isinstance(node, dict):
            return
        assert "uniqueItems" not in node
        schema_type = node.get("type")
        enum_values = node.get("enum")
        if (
            isinstance(schema_type, list)
            and "null" in schema_type
            and isinstance(enum_values, list)
        ):
            assert None in enum_values
        if schema_type == "object" or (
            isinstance(schema_type, list) and "object" in schema_type
        ):
            properties = node.get("properties")
            assert isinstance(properties, dict)
            assert node.get("additionalProperties") is False
            assert set(node.get("required") or []) == set(properties)
            for child in properties.values():
                assert_portable(child)
        if schema_type == "array" or (
            isinstance(schema_type, list) and "array" in schema_type
        ):
            assert isinstance(node.get("items"), dict)
            assert_portable(node["items"])
        for union_key in ("anyOf", "oneOf"):
            for branch in node.get(union_key) or []:
                assert_portable(branch)

    for schema in (
        REQUEST_WORKBENCH_OUTPUT_SCHEMA,
        REQUEST_GRAPH_REVIEW_OUTPUT_SCHEMA,
        DESIGN_GOAL_SYNTHESIS_OUTPUT_SCHEMA,
        GOAL_CONTRACT_AUDIT_OUTPUT_SCHEMA,
        GOAL_CONTRACT_REPAIR_OUTPUT_SCHEMA,
        pipeline_tools._REQUEST_COMPLETENESS_REVIEW_SCHEMA,
        pipeline_tools._goal_unresolved_adjudication_schema(2),
    ):
        assert_portable(openai_strict_json_schema(schema))


def test_openai_schema_compiler_preserves_local_uniqueness_validation() -> None:
    canonical = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            }
        },
        "required": ["values"],
        "additionalProperties": False,
    }

    compiled = openai_strict_json_schema(canonical)

    assert canonical["properties"]["values"]["uniqueItems"] is True
    assert "uniqueItems" not in compiled["properties"]["values"]


def test_portable_schema_compiler_keeps_optional_enum_nullable() -> None:
    canonical = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ready", "blocked"]},
        },
        "required": [],
        "additionalProperties": False,
    }

    compiled = openai_strict_json_schema(canonical)

    assert compiled["properties"]["status"] == {
        "type": ["string", "null"],
        "enum": ["ready", "blocked", None],
    }


def test_sync_boundary_does_not_retry_runtime_error_from_operation() -> None:
    calls = 0

    async def fail_once() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        run_async_from_sync(fail_once)

    assert calls == 1


def test_verified_workbench_regenerates_graph_before_goal_synthesis(
    monkeypatch: Any,
) -> None:
    request = "Create candidates from the supplied inline constraints in reviewed cells."
    planner_calls = {"count": 0}
    review_calls = {"count": 0}

    def envelope(*, include_spurious_task: bool) -> dict[str, Any]:
        tasks = []
        dependencies = []
        if include_spurious_task:
            tasks.append(
                {
                    "task_id": "prepare_inline_input",
                    "task_type": "analyze_artifact",
                    "label": "Prepare inline input",
                    "native": True,
                    "can_run_in_parallel": False,
                    "context": {},
                }
            )
            dependencies.append(
                {
                    "upstream_task_id": "prepare_inline_input",
                    "downstream_task_id": "design",
                }
            )
        tasks.append(
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "label": "Create candidates",
                "native": True,
                "can_run_in_parallel": False,
                "context": {"target_context": "reviewed cells"},
            }
        )
        return {
            "status": "ready",
            "questions": [],
            "reason": None,
            "request_envelope": {
                "schema_version": "v3",
                "request_mode": "single_task",
                "request_summary": "Create candidates.",
                "request_classifications": ["design_new_asos"],
                "tasks": tasks,
                "task_dependencies": dependencies,
                "final_outputs": ["candidate set"],
                "response_mode": "report",
                "requires_execution": True,
                "assumptions": [],
            },
        }

    async def fake_planner(_prompt: str) -> dict[str, Any]:
        planner_calls["count"] += 1
        return envelope(include_spurious_task=planner_calls["count"] == 1)

    async def fake_review(_prompt: str) -> dict[str, Any]:
        review_calls["count"] += 1
        if review_calls["count"] == 1:
            return {
                "status": "revise",
                "issues": [
                    {
                        "category": "spurious_task",
                        "task_ids": ["prepare_inline_input"],
                        "path": "/tasks/0",
                        "evidence_refs": ["request"],
                        "evidence_quote": "supplied inline constraints",
                        "reason": "Inline preparation is not a requested operation.",
                    }
                ],
            }
        return {"status": "valid", "issues": []}

    async def fake_synthesis(_prompt: str) -> dict[str, Any]:
        return {
            "status": "ready",
            "questions": [],
            "reason": None,
            "structured_goal": {
                "target_context": "reviewed cells",
                "design_programs": [
                    {
                        "program_id": "primary",
                        "label": "Primary design",
                        "modality": "steric",
                        "intent": "binding",
                        "objective": "Create candidates for the supplied constraints.",
                        "desired_delta": "increase",
                    }
                ],
            },
        }

    async def fake_adjudicator(_prompt: str) -> dict[str, Any]:
        return {
            "status": "revise",
            "issues": [
                {
                    "category": "spurious_task",
                    "task_ids": ["prepare_inline_input"],
                    "path": "/tasks/0",
                    "evidence_refs": ["request"],
                    "evidence_quote": "supplied inline constraints",
                    "reason": "Inline preparation is not a requested operation.",
                }
            ],
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_request_workbench_agent_once",
        fake_planner,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_request_graph_review_agent_once",
        fake_review,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_request_graph_review_adjudicator_once",
        fake_adjudicator,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_design_goal_synthesis_agent_once",
        fake_synthesis,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_review_request_execution_completeness",
        _passthrough_completeness_review,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_review_request_clarifications_if_needed",
        _passthrough_completeness_review,
    )

    result = asyncio.run(
        pipeline_tools._run_verified_request_workbench_agent_once(request)
    )

    tasks = result["request_envelope"]["tasks"]
    assert planner_calls["count"] == 2
    assert review_calls["count"] == 3
    assert [task["task_id"] for task in tasks] == ["design"]
    assert tasks[0]["context"]["structured_goal"]["design_programs"][0][
        "program_id"
    ] == "primary"


def test_goal_audit_status_requires_matching_payload_channel() -> None:
    errors = pipeline_tools._goal_contract_audit_contract_errors(
        {
            "status": "needs_clarification",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [],
        }
    )

    assert errors == [
        "$: needs_clarification audit output requires an unresolved item"
    ]


def test_goal_audit_retries_when_open_validation_obligation_is_omitted(
    monkeypatch: Any,
) -> None:
    request_text = "Compile the requested design using the context supplied later."
    envelope = {
        "schema_version": "v3",
        "request_text": request_text,
        "tasks": [
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "context": {
                    "structured_goal": {
                        "design_programs": [
                            {
                                "program_id": "branch",
                                "label": "Branch",
                                "modality": "steric",
                                "intent": "binding",
                                "objective": "Requested objective.",
                                "desired_delta": "increase",
                            }
                        ]
                    }
                },
            }
        ],
    }
    calls = {"count": 0}

    def fake_localization(payload: dict[str, Any]) -> dict[str, Any]:
        target = next(
            candidate
            for candidate in payload["candidate_destinations"]
            if candidate["canonical_path"]
            == "context.structured_goal.target_context"
        )
        return {
            "verdicts": [
                {
                    "issue_index": 0,
                    "resolution": "localized",
                    "candidate_indexes": [target["candidate_index"]],
                    "reason": "The schema semantics identify the destination.",
                }
            ]
        }

    async def fake_audit(prompt: str) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "status": "valid",
                "patches": [],
                "goal_replacements": [],
                "field_dispositions": [],
                "unresolved": [],
            }
        if calls["count"] == 2:
            assert "prior_rejected_patches" in prompt
            return {
                "status": "repairable",
                "patches": [
                    {
                        "op": "add",
                        "path": (
                            "/tasks/0/context/structured_goal/target_context"
                        ),
                        "value": "user-supplied cells",
                        "evidence_refs": ["clarification.0.answer"],
                        "evidence_quote": "user-supplied cells",
                        "reason": "The clarification supplies the required value.",
                    }
                ],
                "goal_replacements": [],
                "field_dispositions": [],
                "unresolved": [],
            }
        return {
            "status": "valid",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [],
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_validation_issue_localization_sync",
        fake_localization,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )

    result = pipeline_tools._repair_request_envelope_if_needed(
        request_text=request_text,
        run_dir="",
        structured_goal={},
        request_envelope=envelope,
        qa_history=[
            {
                "question": "What execution context should be used?",
                "answer": "user-supplied cells",
            }
        ],
        semantic_audit=True,
    )

    assert result["status"] == "ready", result
    assert calls["count"] == 3
    assert result["request_envelope"]["tasks"][0]["context"][
        "structured_goal"
    ]["target_context"] == "user-supplied cells"


def test_goal_audit_status_is_derived_from_typed_result_channels() -> None:
    empty = pipeline_tools._normalize_goal_contract_audit_status(
        {
            "status": "repairable",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [],
        }
    )
    repairable = pipeline_tools._normalize_goal_contract_audit_status(
        {
            "status": "valid",
            "patches": [{"op": "remove", "path": "/tasks/0/context/structured_goal/platform"}],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [],
        }
    )
    unresolved = pipeline_tools._normalize_goal_contract_audit_status(
        {
            "status": "repairable",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [{"question": "Which user-specific sample should be used?"}],
        }
    )

    assert empty["status"] == "valid"
    assert repairable["status"] == "repairable"
    assert unresolved["status"] == "needs_clarification"


def test_goal_unresolved_adjudication_requires_one_verdict_per_item() -> None:
    schema = pipeline_tools._goal_unresolved_adjudication_schema(2)
    errors = pipeline_tools._goal_unresolved_adjudication_contract_errors(
        {
            "verdicts": [
                {
                    "unresolved_index": 0,
                    "disposition": "repair_from_evidence",
                    "canonical_path": "/tasks/0/context/structured_goal/target_context",
                    "reason": "The supplied clarification already provides the value.",
                },
                {
                    "unresolved_index": 0,
                    "disposition": "user_input",
                    "canonical_path": None,
                    "reason": "A required user-specific input is absent.",
                },
            ]
        },
        item_count=2,
        schema=schema,
    )

    assert errors == [
        "$.verdicts: requires exactly one verdict per unresolved item"
    ]


def test_clarification_reviewer_keeps_only_blocking_questions(
    monkeypatch: Any,
) -> None:
    planned = {
        "status": "need_clarification",
        "questions": [
            "Please provide the missing sample-specific sequence artifact.",
            "Would you like to override any pipeline defaults?",
        ],
        "reason": "Additional inputs requested.",
        "request_envelope": _valid_workbench_payload()["request_envelope"],
    }

    async def fake_run(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["agent"].name == "ASO Request Clarification Necessity Reviewer"
        review_input = json.loads(kwargs["prompt"])
        assert "canonical_requirement_catalog" in review_input
        assert "platform_execution_validation" in review_input
        assert any(
            capability["capability"] == "public_named_isoform_resolution"
            for capability in review_input["declared_platform_capabilities"]
        )
        return {
            "blocking_question_indexes": [0],
            "additional_blocking_questions": [],
            "reason": "Only the sample-specific input is required.",
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_schema_validated_agent_once",
        fake_run,
    )

    result = asyncio.run(
        pipeline_tools._review_request_clarifications_if_needed(
            "Design ASOs using my sample-specific sequence.",
            planned,
        )
    )

    assert result["status"] == "need_clarification"
    assert result["questions"] == [
        "Please provide the missing sample-specific sequence artifact."
    ]
    assert result["reason"] == "Only the sample-specific input is required."


def test_clarification_reviewer_rejects_structurally_invalid_proposals(
    monkeypatch: Any,
) -> None:
    planned = {
        "status": "need_clarification",
        "questions": [",", "..."],
        "reason": "A model asserted that input was missing.",
        "request_envelope": _valid_workbench_payload()["request_envelope"],
    }

    async def completeness_agent_call(**kwargs: Any) -> dict[str, Any]:
        review_input = json.loads(kwargs["prompt"])
        assert review_input["proposed_questions"] == []
        return {
            "blocking_question_indexes": [],
            "additional_blocking_questions": [],
            "reason": "No execution blocker remains.",
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_schema_validated_agent_once",
        completeness_agent_call,
    )

    result = asyncio.run(
        pipeline_tools._review_request_clarifications_if_needed(
            "Create a supported ASO design.",
            planned,
        )
    )

    assert result["status"] == "ready"
    assert result["questions"] == []
    assert result["reason"] is None


def test_clarification_reviewer_adds_omitted_blocker_to_ready_plan(
    monkeypatch: Any,
) -> None:
    planned = _valid_workbench_payload()

    async def fake_run(**kwargs: Any) -> dict[str, Any]:
        review_input = json.loads(kwargs["prompt"])
        assert review_input["proposed_questions"] == []
        return {
            "blocking_question_indexes": [],
            "additional_blocking_questions": [
                {
                    "missing_field": "user_defined_target_identity",
                    "question": "Provide a resolvable identifier or exact coordinates for the user-defined target feature.",
                    "reason": "Its exact identity is necessary and cannot be resolved from public annotation.",
                }
            ],
            "reason": "One user-specific execution input was omitted.",
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_schema_validated_agent_once",
        fake_run,
    )

    result = asyncio.run(
        pipeline_tools._review_request_clarifications_if_needed(
            "Create a design against my custom target feature.",
            planned,
        )
    )

    assert result["status"] == "need_clarification"
    assert result["questions"] == [
        "Provide a resolvable identifier or exact coordinates for the user-defined target feature."
    ]
    assert result["clarification_requirements"] == [
        {
            "missing_field": "user_defined_target_identity",
            "question": "Provide a resolvable identifier or exact coordinates for the user-defined target feature.",
            "reason": "Its exact identity is necessary and cannot be resolved from public annotation.",
        }
    ]


def test_goal_intake_clarification_preserves_typed_requirement_history() -> None:
    clarification = pipeline_tools._update_goal_intake_clarification_from_payload(
        {},
        {
            "reason": "One execution input is absent.",
            "clarification_requirements": [
                {
                    "missing_field": "required_context_missing",
                    "question": "Which context should be used?",
                    "reason": "Execution requires a user-controlled context.",
                }
            ],
        },
        ["Which context should be used?"],
    )

    assert clarification["missing_fields"] == ["required_context_missing"]
    assert clarification["requirement_history"][0]["question"] == (
        "Which context should be used?"
    )


def test_clarification_reviewer_receives_current_execution_blockers(
    monkeypatch: Any,
) -> None:
    envelope = {
        "schema_version": "v3",
        "request_mode": "single_task",
        "request_summary": "Create one ASO design.",
        "request_classifications": ["design_new_asos"],
        "tasks": [
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "label": "Create one ASO design",
                "native": True,
                "can_run_in_parallel": False,
                "context": {
                    "gene_symbol": "GENE1",
                    "species": "human",
                    "structured_goal": {
                        "gene_symbol": "GENE1",
                        "species": "human",
                        "design_programs": [
                            {
                                "program_id": "program_1",
                                "label": "Create candidates",
                                "modality": "gapmer",
                                "intent": "knockdown",
                                "objective": "Decrease the requested RNA output.",
                                "desired_delta": "decrease",
                            }
                        ],
                    },
                },
            }
        ],
        "task_dependencies": [],
        "final_outputs": ["ASO candidates"],
        "response_mode": "analysis",
        "requires_execution": True,
        "assumptions": [],
    }
    planned = {
        "status": "need_clarification",
        "questions": ["Which biological context should this design use?"],
        "reason": "A required execution input is missing.",
        "request_envelope": envelope,
    }

    async def fake_run(**kwargs: Any) -> dict[str, Any]:
        review_input = json.loads(kwargs["prompt"])
        validation = review_input["platform_execution_validation"]
        assert validation["blocked_task_ids"] == ["design"]
        assert validation["tasks"]["design"]["status"] == "blocked"
        assert validation["tasks"]["design"]["issues"] == [
            "target_context_missing"
        ]
        return {
            "blocking_question_indexes": [0],
            "additional_blocking_questions": [],
            "reason": "The question resolves a current execution blocker.",
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_schema_validated_agent_once",
        fake_run,
    )

    result = asyncio.run(
        pipeline_tools._review_request_clarifications_if_needed(
            "Create a design for GENE1.",
            planned,
        )
    )

    assert result["questions"] == [
        "Which biological context should this design use?"
    ]


def test_execution_completeness_reviewer_can_request_absent_user_input(
    monkeypatch: Any,
) -> None:
    planned = {
        "status": "ready",
        "questions": [],
        "reason": None,
        "request_envelope": {
            "schema_version": "v3",
            "request_mode": "single_task",
            "request_summary": "Create candidates using user-controlled input.",
            "request_classifications": ["design_new_asos"],
            "tasks": [
                {
                    "task_id": "design",
                    "task_type": "design_new_asos",
                    "label": "Create candidates",
                    "native": True,
                    "can_run_in_parallel": False,
                    "context": {
                        "gene_symbol": "GENE1",
                        "species": "human",
                        "target_context": "user-defined cells",
                        "structured_goal": {
                            "gene_symbol": "GENE1",
                            "species": "human",
                            "target_context": "user-defined cells",
                            "design_programs": [
                                {
                                    "program_id": "program_1",
                                    "label": "Create candidates",
                                    "modality": "gapmer",
                                    "intent": "knockdown",
                                    "objective": "Decrease the requested RNA output.",
                                    "desired_delta": "decrease",
                                }
                            ],
                        },
                    },
                }
            ],
            "task_dependencies": [],
            "final_outputs": ["ASO candidates"],
            "response_mode": "analysis",
            "requires_execution": True,
            "assumptions": [],
        },
    }

    async def fake_run(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["agent"].name == "ASO Request Execution Completeness Reviewer"
        review_input = json.loads(kwargs["prompt"])
        assert review_input == {
            "request_evidence": "Create candidates using the material I supplied."
        }
        return {
            "status": "needs_clarification",
            "questions": ["Please provide the referenced user-controlled input."],
            "reason": "The actual input is absent.",
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_schema_validated_agent_once",
        fake_run,
    )

    result = asyncio.run(
        pipeline_tools._review_request_execution_completeness(
            "Create candidates using the material I supplied.",
            planned,
        )
    )

    assert result["status"] == "need_clarification"
    assert result["questions"] == [
        "Please provide the referenced user-controlled input."
    ]
    assert result["request_envelope"] == planned["request_envelope"]


def test_execution_completeness_stops_before_goal_synthesis(
    monkeypatch: Any,
) -> None:
    planned = _valid_workbench_payload()

    async def needs_input(
        _request_evidence: str,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        revised = deepcopy(current)
        revised["status"] = "need_clarification"
        revised["questions"] = ["Please provide the required user-controlled input."]
        revised["reason"] = "The input is absent from request evidence."
        return revised

    async def fail_synthesis(
        _request_evidence: str,
        _planned: dict[str, Any],
    ) -> dict[str, Any]:
        raise AssertionError("goal synthesis must wait for required input")

    async def retain_question(
        _request_evidence: str,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        return current

    monkeypatch.setattr(
        pipeline_tools,
        "_review_request_execution_completeness",
        needs_input,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_synthesize_missing_design_goals",
        fail_synthesis,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_review_request_clarifications_if_needed",
        retain_question,
    )

    result = asyncio.run(
        pipeline_tools._finalize_verified_request_plan(
            "Create candidates using material I supplied.",
            planned,
        )
    )

    assert result["status"] == "need_clarification"
    assert result["questions"] == [
        "Please provide the required user-controlled input."
    ]


def test_execution_completeness_questions_are_necessity_reviewed(
    monkeypatch: Any,
) -> None:
    planned = _valid_workbench_payload()
    calls: list[str] = []

    async def proposes_question(
        _request_evidence: str,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append("completeness")
        revised = deepcopy(current)
        revised["status"] = "need_clarification"
        revised["questions"] = ["Please provide a public reference identifier."]
        return revised

    async def drops_question(
        _request_evidence: str,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append("necessity")
        revised = deepcopy(current)
        revised["status"] = "ready"
        revised["questions"] = []
        revised["reason"] = None
        return revised

    async def synthesize(
        _request_evidence: str,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append("synthesis")
        return current

    monkeypatch.setattr(
        pipeline_tools, "_review_request_execution_completeness", proposes_question
    )
    monkeypatch.setattr(
        pipeline_tools, "_review_request_clarifications_if_needed", drops_question
    )
    monkeypatch.setattr(
        pipeline_tools, "_synthesize_missing_design_goals", synthesize
    )

    result = asyncio.run(
        pipeline_tools._finalize_verified_request_plan("request", planned)
    )

    assert result["status"] == "ready"
    assert calls == ["completeness", "necessity", "synthesis", "necessity"]


def test_evidence_ref_schema_is_bound_to_supplied_protocol_ids() -> None:
    bounded = pipeline_tools._schema_with_allowed_evidence_refs(
        GOAL_CONTRACT_AUDIT_OUTPUT_SCHEMA,
        ["request", "clarification.0.answer", "request"],
    )

    patch_ref_items = bounded["properties"]["patches"]["items"][
        "properties"
    ]["evidence_refs"]["items"]
    replacement_ref_items = bounded["properties"]["goal_replacements"][
        "items"
    ]["properties"]["evidence_refs"]["items"]
    assert patch_ref_items["enum"] == ["request", "clarification.0.answer"]
    assert replacement_ref_items["enum"] == [
        "request",
        "clarification.0.answer",
    ]
    assert "enum" not in GOAL_CONTRACT_AUDIT_OUTPUT_SCHEMA["properties"][
        "patches"
    ]["items"]["properties"]["evidence_refs"]["items"]


def test_evidence_span_id_is_schema_bound_then_materialized_exactly() -> None:
    sources = pipeline_tools._evidence_sources_with_quote_spans(
        [
            {
                "evidence_id": "request",
                "text": (
                    "Preserve every supplied literal and keep every requested "
                    "constraint in its canonical destination."
                ),
            }
        ]
    )
    span_map = pipeline_tools._evidence_span_map(sources)
    bounded = pipeline_tools._schema_with_allowed_evidence_refs(
        REQUEST_GRAPH_REVIEW_OUTPUT_SCHEMA,
        ["request"],
        list(span_map),
    )
    issue_schema = bounded["properties"]["issues"]["items"]
    span_id = next(iter(span_map))

    assert issue_schema["properties"]["evidence_refs"]["minItems"] == 1
    assert issue_schema["properties"]["evidence_refs"]["uniqueItems"] is True
    assert "evidence_quote" not in issue_schema["properties"]
    assert issue_schema["properties"]["evidence_span_id"]["enum"] == list(
        span_map
    )
    assert "evidence_span_id" in issue_schema["required"]
    materialized = pipeline_tools._materialize_evidence_span_ids(
        {
            "status": "revise",
            "issues": [
                {
                    "evidence_refs": ["request"],
                    "evidence_span_id": span_id,
                }
            ],
        },
        span_map,
    )
    assert materialized["issues"][0] == {
        "evidence_refs": ["request"],
        "evidence_quote": span_map[span_id]["text"],
    }
    assert span_map[span_id]["text"] in sources[0]["text"]


def test_goal_audit_reads_declared_evidence_ids_from_protocol_payload() -> None:
    prompt = (
        "Audit this envelope.\n\n"
        + json.dumps(
            {
                "evidence_sources": [
                    {"evidence_id": "request", "text": "request text"},
                    {
                        "evidence_id": "clarification.0.answer",
                        "text": "answer text",
                    },
                ]
            }
        )
    )

    assert pipeline_tools._goal_audit_evidence_ids_from_prompt(prompt) == [
        "request",
        "clarification.0.answer",
    ]


def test_graph_review_retries_unverifiable_evidence_once(
    monkeypatch: Any,
) -> None:
    request = "Evaluate the supplied ASO for the requested target."
    review_calls = {"count": 0}

    async def fake_planner(_prompt: str) -> dict[str, Any]:
        return _valid_workbench_payload()

    async def fake_review(prompt: str) -> dict[str, Any]:
        review_calls["count"] += 1
        if review_calls["count"] == 1:
            return {
                "status": "revise",
                "issues": [
                    {
                        "category": "wrong_dependency",
                        "task_ids": ["evaluate_supplied_aso"],
                        "path": None,
                        "evidence_refs": ["invented_source"],
                        "evidence_quote": "supplied ASO",
                        "reason": "The review citation is invalid.",
                    }
                ],
            }
        if review_calls["count"] == 2:
            assert "could not be verified" in prompt
        else:
            assert "operation-coverage" in prompt
        return {"status": "valid", "issues": []}

    monkeypatch.setattr(
        pipeline_tools,
        "_run_request_workbench_agent_once",
        fake_planner,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_request_graph_review_agent_once",
        fake_review,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_review_request_execution_completeness",
        _passthrough_completeness_review,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_review_request_clarifications_if_needed",
        _passthrough_completeness_review,
    )

    result = asyncio.run(
        pipeline_tools._run_verified_request_workbench_agent_once(request)
    )

    assert result == _valid_workbench_payload()
    assert review_calls["count"] == 3


def test_explicit_splice_target_schema_requires_defining_coordinates() -> None:
    goal_schema = DESIGN_GOAL_SYNTHESIS_OUTPUT_SCHEMA["properties"][
        "structured_goal"
    ]["anyOf"][1]
    item_schema = goal_schema["properties"]["explicit_splice_targets"]["items"]

    assert set(item_schema["required"]) == {"chrom", "start", "end"}


def test_allele_specific_variant_has_typed_canonical_destinations() -> None:
    goal_schema = canonical_structured_goal_schema()
    variant_schema = goal_schema["properties"]["patient_variant"]
    allele_schema = goal_schema["properties"]["constraints"]["properties"][
        "allele_specific"
    ]

    assert set(variant_schema["required"]) == {"notation", "kind"}
    assert set(variant_schema["properties"]) >= {
        "notation",
        "kind",
        "chr",
        "pos",
        "ref",
        "alt",
        "reference_accession",
        "target_allele",
        "non_target_allele",
    }
    assert set(allele_schema["required"]) == {
        "target_allele",
        "non_target_allele",
        "require_variant_overlap",
    }
    assert allele_schema["additionalProperties"] is False


def test_preservation_target_has_a_typed_canonical_destination() -> None:
    goal_schema = DESIGN_GOAL_SYNTHESIS_OUTPUT_SCHEMA["properties"][
        "structured_goal"
    ]["anyOf"][1]
    item_schema = goal_schema["properties"]["preservation_targets"]["items"]

    assert set(item_schema["required"]) == {"action", "preservation_target_type"}
    assert item_schema["properties"]["action"]["enum"] == ["preserve"]
    assert item_schema["properties"]["preservation_target_type"]["enum"] == [
        "gene",
        "isoform",
        "transcript",
    ]


def test_graph_review_adjudicator_can_reject_a_no_op_revision(
    monkeypatch: Any,
) -> None:
    planner_calls = {"count": 0}

    async def fake_planner(_prompt: str) -> dict[str, Any]:
        planner_calls["count"] += 1
        return _valid_workbench_payload()

    async def fake_review(_prompt: str) -> dict[str, Any]:
        return {
            "status": "revise",
            "issues": [
                {
                    "category": "wrong_task_type",
                    "task_ids": ["evaluate_supplied_aso"],
                    "path": "/tasks/0/task_type",
                    "evidence_refs": ["request"],
                    "evidence_quote": "Evaluate the supplied ASO",
                    "reason": "The current task type is correct and should remain.",
                }
            ],
        }

    async def fake_adjudicator(prompt: str) -> dict[str, Any]:
        assert "Reject no-op issues" in prompt
        return {"status": "valid", "issues": []}

    monkeypatch.setattr(pipeline_tools, "_run_request_workbench_agent_once", fake_planner)
    monkeypatch.setattr(pipeline_tools, "_run_request_graph_review_agent_once", fake_review)
    monkeypatch.setattr(
        pipeline_tools,
        "_run_request_graph_review_adjudicator_once",
        fake_adjudicator,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_synthesize_missing_design_goals",
        lambda _prompt, payload: asyncio.sleep(0, result=payload),
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_review_request_execution_completeness",
        _passthrough_completeness_review,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_review_request_clarifications_if_needed",
        _passthrough_completeness_review,
    )

    result = asyncio.run(
        pipeline_tools._run_verified_request_workbench_agent_once(
            "Evaluate the supplied ASO."
        )
    )

    assert result == _valid_workbench_payload()
    assert planner_calls["count"] == 1


def test_transport_only_null_placeholders_are_removed_before_validation() -> None:
    payload = _valid_workbench_payload()
    context = payload["request_envelope"]["tasks"][0]["context"]
    context.update(
        {
            field: None
            for field in REQUEST_WORKBENCH_OUTPUT_SCHEMA["properties"]
            ["request_envelope"]["anyOf"][1]["properties"]["tasks"]["items"]
            ["properties"]["context"]["properties"]
            if field not in context
        }
    )

    normalized = normalize_strict_output(payload, REQUEST_WORKBENCH_OUTPUT_SCHEMA)

    assert normalized == _valid_workbench_payload()


def test_union_normalization_selects_the_structurally_compatible_object() -> None:
    payload = {
        "chrom": "chr1",
        "desired_delta": "decrease",
        "desired_effect": "skipping",
        "end": 200,
        "exon_label": "target exon",
        "gene_symbol": "GENE1",
        "isoform_label": None,
        "label": "Target exon route",
        "objective_type": "route_target",
        "position": None,
        "site_kind": "both",
        "start": 100,
        "strand": "+",
        "target_type": "exon",
        "transcript_id": "TX1",
    }

    normalized = normalize_strict_output(
        payload,
        canonical_goal_patch_value_schema(),
    )

    assert normalized == {
        key: value for key, value in payload.items() if value is not None
    }


def test_union_normalization_selects_structurally_compatible_array_items() -> None:
    payload = [
        {
            "chrom": "chr1",
            "start": 100,
            "end": 200,
            "strand": "+",
            "label": "Supplied interval",
            "isoform_label": None,
            "exon_label": "target exon",
            "desired_delta": "decrease",
            "notes": None,
        }
    ]

    normalized = normalize_strict_output(
        payload,
        canonical_goal_patch_value_schema(),
    )

    assert normalized == [
        {
            "chrom": "chr1",
            "start": 100,
            "end": 200,
            "strand": "+",
            "label": "Supplied interval",
            "exon_label": "target exon",
            "desired_delta": "decrease",
        }
    ]


def test_union_normalization_uses_enum_values_as_object_discriminators() -> None:
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["replace"]},
                    "path": {"type": "string", "enum": ["/direction"]},
                    "value": {
                        "type": "string",
                        "enum": ["increase", "decrease"],
                    },
                },
                "required": ["op", "path", "value"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add"]},
                    "path": {"type": "string", "enum": ["/items/-"]},
                    "value": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "note": {"type": "string"},
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                "required": ["op", "path", "value"],
                "additionalProperties": False,
            },
        ]
    }
    payload = {
        "op": "add",
        "path": "/items/-",
        "value": {"name": "requested item", "note": None},
    }

    assert normalize_strict_output(payload, schema) == {
        "op": "add",
        "path": "/items/-",
        "value": {"name": "requested item"},
    }


@pytest.mark.parametrize(
    ("field", "invalid_value", "valid_value", "expected_error"),
    [
        ("candidate_limit_per_site", 0, 8, "less than the minimum"),
        (
            "splice_isoform_objectives",
            [{"objective_scope": "default_exon_context", "desired_abundance": "decrease", "objective_id": "invented"}],
            [{"objective_scope": "default_exon_context", "desired_abundance": "decrease"}],
            "objective_id",
        ),
    ],
)
def test_semantic_repair_retries_only_the_selected_field_schema(
    monkeypatch, field, invalid_value, valid_value, expected_error,
) -> None:
    prompt = pipeline_tools._goal_audit_prompt(
        request_envelope={"tasks": [{
            "task_id": "design", "task_type": "design_new_asos",
            "context": {"structured_goal": {"design_programs": []}},
        }]},
        validation_issues=[], issue_records=[],
        evidence_sources=[{
            "evidence_id": "request", "kind": "user_request",
            "text": "Decrease exon inclusion with eight candidates per route.",
        }],
        semantic_audit=True,
    )
    span_map = pipeline_tools._evidence_span_map(
        pipeline_tools._declared_evidence_sources_from_prompt(prompt)
    )
    audit = {
        "status": "repairable",
        "patches": [{
            "op": "add",
            "path": f"/tasks/0/context/structured_goal/{field}",
            "value": invalid_value,
            "evidence_refs": ["request"],
            "evidence_span_id": next(iter(span_map)),
            "reason": "Preserve the requested outcome and per-route count.",
        }],
        "goal_replacements": [], "field_dispositions": [], "unresolved": [],
    }
    prompts = []

    async def fake_run(_agent, *, input, max_steps):
        prompts.append(input)
        response = deepcopy(audit)
        if len(prompts) == 2:
            errors = input.split("Validation errors:\n", 1)[1].split("\n\n", 1)[0]
            assert expected_error in errors
            assert "$.patches[0].op:" not in errors
            assert "$.patches[0].path:" not in errors
            response["patches"][0]["value"] = valid_value
        return SimpleNamespace(final_output=json.dumps(response))

    monkeypatch.setattr(pipeline_tools.Runner, "run", fake_run)
    result = asyncio.run(pipeline_tools._run_goal_contract_audit_agent_once(prompt))

    assert len(prompts) == 2
    assert result["status"] == "repairable"
    assert result["patches"][0]["value"] == valid_value
    assert result["patches"][0]["evidence_quote"] in prompt


def test_discriminated_diagnostics_still_reject_unknown_paths() -> None:
    schema = {"type": "object", "anyOf": [
        {"properties": {"path": {"const": "/count"}, "value": {"type": "integer", "minimum": 1}}},
        {"properties": {"path": {"const": "/name"}, "value": {"type": "string"}}},
    ]}
    assert pipeline_tools._json_schema_validation_errors(
        {"path": "/invented", "value": 1}, schema
    )


def test_adjudicated_goal_repairs_cannot_be_returned_as_valid() -> None:
    prompt = "Audit.\n\n" + json.dumps(
        {
            "allowed_nonempty_channels": ["patches", "unresolved"],
            "unresolved_adjudication_feedback": [
                {
                    "path": "/tasks/0/context/structured_goal/target_context",
                    "required_action": "repair_from_evidence",
                    "reason": "The clarification supplies the canonical value.",
                }
            ],
        }
    )

    schema = pipeline_tools._goal_contract_audit_schema_for_prompt(prompt)

    assert schema["properties"]["status"]["enum"] == ["repairable"]
    assert schema["properties"]["patches"]["minItems"] == 1
    assert schema["properties"]["unresolved"]["maxItems"] == 0


def test_clarification_review_schema_is_bounded_by_input_cardinality() -> None:
    schema = pipeline_tools._request_clarification_review_schema(2)
    indexes = schema["properties"]["blocking_question_indexes"]

    assert indexes["maxItems"] == 2
    assert indexes["uniqueItems"] is True
    assert pipeline_tools._json_schema_validation_errors(
        {
            "blocking_question_indexes": [0, 1, 1],
            "reason": "Repeated output must be rejected.",
        },
        schema,
    )


def test_graph_review_prompt_supplies_canonical_operation_catalog() -> None:
    prompt = pipeline_tools._request_graph_review_prompt(
        "Perform several supported ASO operations.",
        _valid_workbench_payload()["request_envelope"],
    )
    payload = json.loads(prompt.split("\n\n", 1)[1])

    assert sorted(payload["canonical_task_types"]) == sorted(
        pipeline_tools.CANONICAL_TASK_TYPES
    )
    assert "operation-coverage pass" in str(
        pipeline_tools.request_graph_review_agent.instructions
    )


def test_operation_coverage_review_is_schema_driven() -> None:
    prompt = pipeline_tools._request_graph_operation_coverage_review_prompt(
        "Perform several supported ASO operations.",
        _valid_workbench_payload()["request_envelope"],
    )
    instruction, payload_text = prompt.split("\n\n", 1)
    payload = json.loads(payload_text)

    assert "operation-coverage" in instruction
    assert sorted(payload["canonical_task_types"]) == sorted(
        pipeline_tools.CANONICAL_TASK_TYPES
    )
    assert "mutually exclusive fallback design program" in instruction
    assert "mutually exclusive conditional fallback program" in str(
        pipeline_tools.request_graph_review_agent.instructions
    )
    assert "self-contained computation over supplied literals" in instruction
    assert "self-contained computation over supplied literals" in str(
        pipeline_tools.request_workbench_agent.instructions
    )
    assert "self-contained computation over supplied literals" in str(
        pipeline_tools.request_graph_review_agent.instructions
    )


def test_request_graph_agents_keep_generated_candidate_ranking_inside_design() -> None:
    instructions = (
        str(pipeline_tools.request_workbench_agent.instructions),
        str(pipeline_tools.request_graph_review_agent.instructions),
    )

    for text in instructions:
        assert "Candidate scoring, filtering, ranking, and annotation" in text
        assert "compare_aso_sets" in text
        assert "pre-existing ASO sequences" in text


def test_graph_with_retained_clarification_is_still_topology_reviewed(
    monkeypatch: Any,
) -> None:
    planned = _valid_workbench_payload()
    planned["status"] = "need_clarification"
    planned["questions"] = ["Which user-controlled input should be used?"]
    planned["reason"] = "A required user choice is absent."
    reviews: list[str] = []

    async def fake_planner(_prompt: str) -> dict[str, Any]:
        return deepcopy(planned)

    async def retain(
        _request_evidence: str,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        return current

    async def review(prompt: str) -> dict[str, Any]:
        reviews.append(prompt)
        return {"status": "valid", "issues": []}

    monkeypatch.setattr(
        pipeline_tools, "_run_request_workbench_agent_once", fake_planner
    )
    monkeypatch.setattr(
        pipeline_tools, "_review_request_clarifications_if_needed", retain
    )
    monkeypatch.setattr(
        pipeline_tools, "_run_request_graph_review_agent_once", review
    )

    result = asyncio.run(
        pipeline_tools._run_verified_request_workbench_agent_once("request")
    )

    assert result["status"] == "need_clarification"
    assert result["questions"] == planned["questions"]
    assert len(reviews) == 2
    assert "operation-coverage" in reviews[1]


def test_open_validation_issue_ledger_disables_valid_audit_status() -> None:
    prompt = "Audit.\n\n" + json.dumps(
        {
            "allowed_nonempty_channels": ["patches", "unresolved"],
            "validation_issue_ledger": [
                {
                    "issue_index": 0,
                    "task_id": "design",
                    "validation_issue": "required_scope_absent",
                    "localization": "unlocalized",
                }
            ],
            "canonical_contract_issues": [],
        }
    )

    schema = pipeline_tools._goal_contract_audit_schema_for_prompt(prompt)

    assert schema["properties"]["status"]["enum"] == [
        "repairable",
        "needs_clarification",
    ]


def test_canonical_contract_issue_binds_path_operation_and_status() -> None:
    pointer = "/tasks/0/context/structured_goal/evaluation_targets/0/site_kind"
    prompt = "Audit.\n\n" + json.dumps(
        {
            "allowed_nonempty_channels": ["patches", "unresolved"],
            "canonical_contract_issues": [
                {
                    "json_pointer": pointer,
                    "required_operation": "add",
                    "expected_schema": {
                        "type": "string",
                        "enum": ["acceptor", "donor", "both"],
                    },
                }
            ],
        }
    )

    schema = pipeline_tools._goal_contract_audit_schema_for_prompt(prompt)
    patch_branches = schema["properties"]["patches"]["items"]["anyOf"]
    issue_branch = patch_branches[0]["properties"]
    removal_branch = patch_branches[-1]["properties"]

    assert schema["properties"]["status"]["enum"] == [
        "repairable",
        "needs_clarification",
    ]
    assert schema["properties"]["patches"]["maxItems"] == 1
    assert issue_branch["path"]["enum"] == [pointer]
    assert issue_branch["op"]["enum"] == ["add"]
    assert issue_branch["value"]["enum"] == ["acceptor", "donor", "both"]
    assert removal_branch["op"]["enum"] == ["remove"]
    assert removal_branch["value"] == {"type": "null"}


def test_contract_issue_remove_schema_is_limited_to_existing_optional_fields() -> None:
    prompt = "Audit.\n\n" + json.dumps(
        {
            "allowed_nonempty_channels": ["patches", "unresolved"],
            "canonical_contract_issues": [
                {
                    "json_pointer": "/tasks/0/context/gene_symbol",
                    "required_operation": "replace",
                    "expected_schema": {"type": "string"},
                }
            ],
            "current_request_envelope": {
                "tasks": [
                    {
                        "task_id": "analysis",
                        "task_type": "sandbox_analysis",
                        "context": {
                            "gene_symbol": ".",
                            "provided_asos": ["ACGT"],
                        },
                    }
                ]
            },
        }
    )

    schema = pipeline_tools._goal_contract_audit_schema_for_prompt(prompt)
    branches = schema["properties"]["patches"]["items"]["anyOf"]
    removal = next(
        branch["properties"]
        for branch in branches
        if branch["properties"]["op"]["enum"] == ["remove"]
    )

    assert removal["path"]["enum"] == [
        "/tasks/0/context/gene_symbol",
        "/tasks/0/context/provided_asos",
    ]


def test_region_constraint_schema_requires_an_explicit_overlap_policy() -> None:
    goal = {
        "design_programs": [
            {
                "program_id": "program_1",
                "label": "Constrained design",
                "modality": "steric",
                "intent": "binding",
                "objective": "Bind the supplied target region.",
                "desired_delta": "decrease",
            }
        ],
        "region_constraints": [
            {
                "coordinate_space": "genomic",
                "source_type": "explicit_rows",
                "intervals_bed0_half_open": [
                    {
                        "chrom": "chr1",
                        "start": 100,
                        "end": 200,
                        "strand": "+",
                    }
                ],
            }
        ],
    }

    errors = pipeline_tools._json_schema_validation_errors(
        goal,
        canonical_structured_goal_schema(),
    )

    assert any(
        "aso_overlap_requirement" in error and "required property" in error
        for error in errors
    )


def test_json_object_parser_rejects_malformed_outer_object_with_valid_fragment() -> None:
    malformed = (
        '{"status":"ready","request_envelope":{"tasks":['
        '{"context":{"gene_symbol":"GENE1"}}]}'
    )

    with pytest.raises(RuntimeError, match="did not return a JSON object"):
        pipeline_tools._parse_llm_json_object(malformed, "planner")


def test_schema_validation_reports_nested_enum_paths_from_union() -> None:
    invalid = _valid_workbench_payload()
    invalid["request_envelope"]["request_mode"] = "evaluate_existing_aso_set"
    invalid["request_envelope"]["response_mode"] = "json"

    errors = pipeline_tools._json_schema_validation_errors(
        invalid,
        REQUEST_WORKBENCH_OUTPUT_SCHEMA,
    )

    assert any(
        "$.request_envelope.request_mode" in error
        and "single_task" in error
        and "composite" in error
        for error in errors
    )
    assert any(
        "$.request_envelope.response_mode" in error
        and "analysis" in error
        for error in errors
    )


def test_request_workbench_retries_invalid_structured_output(
    monkeypatch: Any,
) -> None:
    malformed = (
        '{"status":"ready","request_envelope":{"tasks":['
        '{"context":{"gene_symbol":"GENE1"}}]}'
    )
    outputs = iter([malformed, json.dumps(_valid_workbench_payload())])
    prompts: list[str] = []

    async def fake_run(
        _agent: Any,
        input: str,
        max_steps: int,
    ) -> SimpleNamespace:
        assert max_steps == 4
        prompts.append(input)
        return SimpleNamespace(final_output=next(outputs))

    monkeypatch.setattr(pipeline_tools.Runner, "run", fake_run)

    result = asyncio.run(
        pipeline_tools._run_request_workbench_agent_once(
            "Evaluate the supplied ASO for the requested target."
        )
    )

    assert result == _valid_workbench_payload()
    assert len(prompts) == 2
    assert "rejected" in prompts[1]
    assert "did not return a JSON object" in prompts[1]
    assert "Original input" in prompts[1]
    assert malformed not in prompts[1]


def test_request_workbench_retries_typed_max_token_incomplete_response(
    monkeypatch: Any,
) -> None:
    outputs = iter(
        [
            {
                "error": "incomplete_response",
                "status": "incomplete",
                "reason": "max_output_tokens",
            },
            _valid_workbench_payload(),
        ]
    )
    prompts: list[str] = []

    async def fake_run(
        _agent: Any,
        input: str,
        max_steps: int,
    ) -> SimpleNamespace:
        assert max_steps == 4
        prompts.append(input)
        return SimpleNamespace(final_output=next(outputs))

    monkeypatch.setattr(pipeline_tools.Runner, "run", fake_run)

    result = asyncio.run(
        pipeline_tools._run_request_workbench_agent_once(
            "Evaluate the supplied ASO for the requested target."
        )
    )

    assert result == _valid_workbench_payload()
    assert len(prompts) == 2
    assert "incomplete_response: max_output_tokens" in prompts[1]
    assert "Begin with the required JSON object immediately" in prompts[1]


def test_request_workbench_retries_typed_invalid_structured_output(
    monkeypatch: Any,
) -> None:
    outputs = iter(
        [
            {
                "error": "invalid_structured_output",
                "reason": "$.patches[0].value did not satisfy the supplied schema",
            },
            _valid_workbench_payload(),
        ]
    )
    prompts: list[str] = []

    async def fake_run(
        _agent: Any,
        input: str,
        max_steps: int,
    ) -> SimpleNamespace:
        assert max_steps == 4
        prompts.append(input)
        return SimpleNamespace(final_output=next(outputs))

    monkeypatch.setattr(pipeline_tools.Runner, "run", fake_run)

    result = asyncio.run(
        pipeline_tools._run_request_workbench_agent_once(
            "Evaluate the supplied ASO for the requested target."
        )
    )

    assert result == _valid_workbench_payload()
    assert len(prompts) == 2
    assert "did not satisfy the supplied schema" in prompts[1]
    assert "Begin with the required JSON object immediately" in prompts[1]


def test_request_workbench_returns_model_error_after_validation_retries(
    monkeypatch: Any,
) -> None:
    calls = {"count": 0}

    async def fake_run(
        _agent: Any,
        input: str,
        max_steps: int,
    ) -> SimpleNamespace:
        del input, max_steps
        calls["count"] += 1
        return SimpleNamespace(final_output={"gene_symbol": "GENE1"})

    monkeypatch.setattr(pipeline_tools.Runner, "run", fake_run)

    result = asyncio.run(
        pipeline_tools._run_request_workbench_agent_once(
            "Evaluate the supplied ASO for the requested target."
        )
    )

    assert calls["count"] == 4
    assert result["error"] == "invalid_structured_output"
    assert "required property" in result["reason"]


def test_schema_validation_converges_across_multiple_field_repairs(
    monkeypatch: Any,
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["ready"]},
            "questions": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": ["string", "null"]},
        },
        "required": ["decision", "questions", "reason"],
        "additionalProperties": False,
    }
    outputs = iter(
        [
            {"status": "ready", "questions": []},
            {"decision": "ready", "reason": None},
            {"decision": "ready", "questions": [], "reason": None},
        ]
    )
    prompts: list[str] = []

    async def fake_run(
        _agent: Any,
        input: str,
        max_steps: int,
    ) -> SimpleNamespace:
        assert max_steps == 4
        prompts.append(input)
        return SimpleNamespace(final_output=next(outputs))

    monkeypatch.setattr(pipeline_tools.Runner, "run", fake_run)

    result = asyncio.run(
        pipeline_tools._run_schema_validated_agent_once(
            agent=pipeline_tools.request_workbench_agent,
            prompt="Return the typed decision.",
            schema=schema,
            error_prefix="typed_decision",
        )
    )

    assert result == {
        "decision": "ready",
        "questions": [],
        "reason": None,
    }
    assert len(prompts) == 3
    assert '"required_keys"' in prompts[1]
    assert '"decision"' in prompts[1]
    assert '"questions"' in prompts[2]
    assert "include every required key even when" in prompts[1]


def test_schema_validation_can_recover_the_last_contract_invalid_output(
    monkeypatch: Any,
) -> None:
    schema = {
        "type": "object",
        "properties": {"decision": {"type": "string", "enum": ["ready"]}},
        "required": ["decision"],
        "additionalProperties": False,
    }
    calls = {"count": 0}

    async def fake_run(
        _agent: Any,
        input: str,
        max_steps: int,
    ) -> SimpleNamespace:
        calls["count"] += 1
        return SimpleNamespace(final_output={"decision": "unsupported"})

    monkeypatch.setattr(pipeline_tools.Runner, "run", fake_run)

    result = asyncio.run(
        pipeline_tools._run_schema_validated_agent_once(
            agent=pipeline_tools.request_workbench_agent,
            prompt="Return the typed decision.",
            schema=schema,
            error_prefix="typed_decision",
            invalid_output_recovery=lambda candidate, errors: (
                {"decision": "ready"}
                if candidate == {"decision": "unsupported"} and errors
                else None
            ),
            max_attempts=2,
        )
    )

    assert calls["count"] == 2
    assert result == {"decision": "ready"}


def test_clarification_contract_marks_design_target_context_runtime_required() -> None:
    shared = CANONICAL_CLARIFICATION_BOUNDARY_INSTRUCTIONS

    assert "Every design_new_asos task requires target_context" in shared
    assert "sparse construction schema may omit it" in shared
    assert "must not be silently invented" in shared
    assert "user-delegated computational baseline" in shared
    assert shared in str(
        pipeline_tools.request_clarification_review_agent.instructions
    )
    assert shared in str(
        goal_intake_clarification_review_agent.instructions
    )
    assert "target_context is an execution-required user-specific" in str(
        pipeline_tools.goal_unresolved_adjudication_agent.instructions
    )


def test_goal_parse_fails_closed_on_invalid_planner_contract(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def invalid_workbench(_prompt: str) -> dict[str, Any]:
        return {"gene_symbol": "GENE1"}

    monkeypatch.setattr(
        pipeline_tools,
        "_run_request_workbench_agent_once",
        invalid_workbench,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": "2",
                "status": "initialized",
                "phase": "initialized",
                "request": "Evaluate an ASO supplied for a user-defined target.",
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
        ),
        encoding="utf-8",
    )

    result = pipeline_tools.tool_parse_goal_request(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "retryable_error"
    assert result["error"] == "request_workbench_invalid_output"
    assert saved_state["status"] == "failed"
    assert saved_state.get("request_envelope") in (None, {})
    assert not (run_dir / "artifacts" / "intake" / "request_envelope.json").exists()


def _reviewed_clarification_payload() -> dict[str, Any]:
    return {
        "status": "need_clarification",
        "questions": ["Please provide the referenced user-controlled input artifact."],
        "reason": "A required execution input is absent.",
        "request_envelope": {
            "schema_version": "v3",
            "request_mode": "single_task",
            "request_summary": "Design candidates using a referenced input artifact.",
            "request_classifications": ["design_new_asos"],
            "tasks": [],
            "task_dependencies": [],
            "final_outputs": [],
            "response_mode": "answer",
            "requires_execution": False,
            "assumptions": [],
        },
    }


def _initialized_intake_state(request: str, run_dir: Path) -> dict[str, Any]:
    return {
        "version": "2",
        "status": "initialized",
        "phase": "initialized",
        "request": request,
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


def test_goal_parse_preserves_a_reviewed_semantic_clarification(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    request = "Design ASOs using the sample-specific material I supplied."
    planned = _reviewed_clarification_payload()

    async def fake_workbench(_prompt: str, **_kwargs: Any) -> dict[str, Any]:
        return planned

    def fail_audit(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("a reviewed clarification must stop before goal audit")

    monkeypatch.setattr(
        pipeline_tools,
        "_run_verified_request_workbench_agent_once",
        fake_workbench,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_repair_request_envelope_if_needed",
        fail_audit,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(_initialized_intake_state(request, run_dir)),
        encoding="utf-8",
    )

    result = pipeline_tools.tool_parse_goal_request(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok", json.dumps(result, indent=2)
    assert result["intake_partial"] is True
    assert saved_state["status"] == "intake_partial"
    assert saved_state["pending_questions"] == planned["questions"]


def test_cached_goal_parse_preserves_a_reviewed_semantic_clarification(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    request = "Design ASOs using the sample-specific material I supplied."
    planned = _reviewed_clarification_payload()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pipeline_tools._save_goal_payload(
        str(run_dir),
        {
            "request": request,
            "qa_history": [],
            "request_envelope": planned["request_envelope"],
        },
        {},
        planned,
    )

    async def fail_workbench(_prompt: str) -> dict[str, Any]:
        raise AssertionError("cached intake must not rerun the workbench")

    monkeypatch.setattr(
        pipeline_tools,
        "_run_verified_request_workbench_agent_once",
        fail_workbench,
    )
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(_initialized_intake_state(request, run_dir)),
        encoding="utf-8",
    )

    result = pipeline_tools.tool_parse_goal_request(str(state_path))
    saved_state = pipeline_tools._load_state(str(state_path))

    assert result["status"] == "ok"
    assert result["intake_partial"] is True
    assert result["idempotent"] is True
    assert saved_state["status"] == "intake_partial"
    assert saved_state["pending_questions"] == planned["questions"]


def test_semantic_audit_repairs_non_design_task_literals(
    monkeypatch: Any,
) -> None:
    request_text = (
        "Evaluate the supplied ASO ACGTACGTACGTACGTACGT against transcript "
        "TX-USER-001 in a user-defined cell model."
    )
    envelope = {
        "schema_version": "v3",
        "request_text": request_text,
        "tasks": [
            {
                "task_id": "evaluate_supplied_aso",
                "task_type": "evaluate_existing_aso_set",
                "label": "Evaluate supplied ASO",
                "context": {
                    "gene_symbol": "GENE1",
                    "species": "Homo sapiens",
                    "target_context": "user-defined cell model",
                    "transcript_id": "TX-USER-002",
                    "provided_asos": ["ACGTACGTACGTACGTACGA"],
                },
            }
        ],
        "task_dependencies": [],
    }
    calls = {"count": 0}

    async def fake_audit(prompt: str) -> dict[str, Any]:
        calls["count"] += 1
        assert "canonical_task_context_registry" in prompt
        assert "task_context_locations" in prompt
        if calls["count"] == 2:
            assert '"audit_mode": "post_repair_semantic_verification"' in prompt
            return {
                "status": "valid",
                "patches": [],
                "goal_replacements": [],
                "field_dispositions": [],
                "unresolved": [],
            }
        return {
            "status": "repairable",
            "patches": [
                {
                    "op": "replace",
                    "path": "/tasks/0/context/provided_asos",
                    "value": ["ACGTACGTACGTACGTACGT"],
                    "evidence_refs": ["request"],
                    "evidence_quote": "ACGTACGTACGTACGTACGT",
                    "reason": "Restore the exact supplied ASO value.",
                },
                {
                    "op": "replace",
                    "path": "/tasks/0/context/transcript_id",
                    "value": "TX-USER-001",
                    "evidence_refs": ["request"],
                    "evidence_quote": "TX-USER-001",
                    "reason": "Restore the exact supplied transcript identifier.",
                },
            ],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [],
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )

    result = pipeline_tools._repair_request_envelope_if_needed(
        request_text=request_text,
        run_dir="",
        structured_goal={},
        request_envelope=envelope,
        qa_history=[],
        semantic_audit=True,
    )

    context = result["request_envelope"]["tasks"][0]["context"]
    assert result["status"] == "ready"
    assert calls["count"] == 2
    assert context["provided_asos"] == ["ACGTACGTACGTACGTACGT"]
    assert context["transcript_id"] == "TX-USER-001"


def test_goal_audit_adjudicator_routes_model_authored_issue_back_to_repair(
    monkeypatch: Any,
) -> None:
    request_text = "Design a knockdown ASO in user-defined cells."
    path = "/tasks/0/context/structured_goal/evaluation_targets/0/desired_effect"
    envelope = {
        "schema_version": "v3",
        "tasks": [
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "label": "Design a knockdown ASO",
                "context": {
                    "gene_symbol": "GENE1",
                    "species": "human",
                    "target_context": "user-defined cells",
                    "structured_goal": {
                        "gene_symbol": "GENE1",
                        "species": "human",
                        "target_context": "user-defined cells",
                        "design_programs": [
                            {
                                "program_id": "program_1",
                                "label": "Knock down the requested RNA",
                                "modality": "gapmer",
                                "intent": "knockdown",
                                "objective": "Decrease the requested RNA.",
                                "desired_delta": "decrease",
                            }
                        ],
                        "evaluation_targets": [
                            {
                                "label": "Requested RNA",
                                "target_type": "transcript",
                                "objective_type": "primary_outcome",
                                "desired_delta": "decrease",
                                "desired_effect": "skipping",
                            }
                        ],
                    },
                },
            }
        ],
        "task_dependencies": [],
    }
    calls = {"count": 0}

    async def fake_audit(prompt: str) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "status": "needs_clarification",
                "patches": [],
                "goal_replacements": [],
                "field_dispositions": [],
                "unresolved": [
                    {
                        "path": path,
                        "question": "Should the splice-effect field be removed?",
                        "reason": "It conflicts with the requested knockdown operation.",
                        "resolution": "user_input",
                    }
                ],
            }
        if calls["count"] == 2:
            protocol = pipeline_tools._goal_audit_protocol_payload(prompt)
            assert protocol["unresolved_adjudication_feedback"] == [
                {
                    "path": path,
                    "required_action": "remove_model_authored_field",
                    "reason": "Canonical correction is the auditor's responsibility.",
                    "rejected_unresolved": {
                        "path": path,
                        "question": "Should the splice-effect field be removed?",
                        "reason": "It conflicts with the requested knockdown operation.",
                        "resolution": "user_input",
                    },
                }
            ]
            return {
                "status": "repairable",
                "patches": [
                    {
                        "op": "remove",
                        "path": path,
                        "value": None,
                        "evidence_refs": ["request"],
                        "evidence_quote": "Design a knockdown ASO",
                        "reason": "Remove a model-authored splice effect from a knockdown target.",
                    }
                ],
                "goal_replacements": [],
                "field_dispositions": [],
                "unresolved": [],
            }
        return {
            "status": "valid",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [],
        }

    def fake_adjudication(_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "verdicts": [
                {
                    "unresolved_index": 0,
                    "disposition": "remove_model_authored_field",
                    "canonical_path": path,
                    "reason": "Canonical correction is the auditor's responsibility.",
                }
            ]
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_unresolved_adjudication_sync",
        fake_adjudication,
    )

    result = pipeline_tools._repair_request_envelope_if_needed(
        request_text=request_text,
        run_dir="",
        structured_goal={},
        request_envelope=envelope,
        qa_history=[],
        semantic_audit=True,
    )

    goal = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert result["status"] == "ready"
    assert calls["count"] == 3
    assert "desired_effect" not in goal["evaluation_targets"][0]


def test_goal_audit_adjudicator_preserves_genuine_user_input_question(
    monkeypatch: Any,
) -> None:
    question = "Please provide the required private sample sequence."
    envelope = {
        "schema_version": "v3",
        "tasks": [
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "label": "Design from a private sample",
                "context": {
                    "gene_symbol": "GENE1",
                    "species": "human",
                    "target_context": "user-defined cells",
                    "structured_goal": {
                        "gene_symbol": "GENE1",
                        "species": "human",
                        "target_context": "user-defined cells",
                        "design_programs": [
                            {
                                "program_id": "program_1",
                                "label": "Create candidates",
                                "modality": "gapmer",
                                "intent": "knockdown",
                                "objective": "Decrease the requested RNA.",
                                "desired_delta": "decrease",
                            }
                        ],
                    },
                },
            }
        ],
        "task_dependencies": [],
    }

    async def fake_audit(_prompt: str) -> dict[str, Any]:
        return {
            "status": "needs_clarification",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [
                {
                    "path": "/tasks/0/context/structured_goal/artifact_refs",
                    "question": question,
                    "reason": "Execution depends on absent user-controlled material.",
                    "resolution": "user_input",
                }
            ],
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_unresolved_adjudication_sync",
        lambda _payload: {
            "verdicts": [
                {
                    "unresolved_index": 0,
                    "disposition": "user_input",
                    "canonical_path": None,
                    "reason": "The required private material is absent.",
                }
            ]
        },
    )

    result = pipeline_tools._repair_request_envelope_if_needed(
        request_text="Design candidates using my private sample sequence.",
        run_dir="",
        structured_goal={},
        request_envelope=envelope,
        qa_history=[],
        semantic_audit=True,
    )

    assert result["status"] == "need_clarification"
    assert result["questions"] == [question]


def test_goal_audit_adjudicates_unresolved_items_before_applying_other_repairs(
    monkeypatch: Any,
) -> None:
    request_text = "Design a gene-wide knockdown ASO in user-defined cells."
    optional_child_path = (
        "/tasks/0/context/structured_goal/region_constraints/0/"
        "aso_overlap_requirement"
    )
    envelope = {
        "schema_version": "v3",
        "tasks": [
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "label": "Design a knockdown ASO",
                "context": {
                    "gene_symbol": "GENE1",
                    "species": "human",
                    "target_context": "user-defined cells",
                    "structured_goal": {
                        "gene_symbol": "GENE1",
                        "species": "human",
                        "target_context": "user-defined cells",
                        "design_programs": [
                            {
                                "program_id": "program_1",
                                "label": "Knock down the requested RNA",
                                "modality": "gapmer",
                                "intent": "knockdown",
                                "objective": "Decrease the requested RNA.",
                                "desired_delta": "decrease",
                            }
                        ],
                    },
                },
            }
        ],
        "task_dependencies": [],
    }
    calls = {"audit": 0, "adjudication": 0}

    async def fake_audit(_prompt: str) -> dict[str, Any]:
        calls["audit"] += 1
        if calls["audit"] == 1:
            return {
                "status": "needs_clarification",
                "patches": [
                    {
                        "op": "add",
                        "path": "/tasks/0/context/structured_goal/target_scope",
                        "value": "gene-wide",
                        "evidence_refs": ["request"],
                        "evidence_quote": "gene-wide knockdown",
                        "reason": "Preserve the requested target scope.",
                    }
                ],
                "goal_replacements": [],
                "field_dispositions": [],
                "unresolved": [
                    {
                        "path": optional_child_path,
                        "question": "Which policy should populate the optional constraint?",
                        "reason": "A child would be required if the optional constraint were created.",
                        "resolution": "user_input",
                    }
                ],
            }
        return {
            "status": "valid",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [],
            "unresolved": [],
        }

    def fake_adjudication(payload: dict[str, Any]) -> dict[str, Any]:
        calls["adjudication"] += 1
        path_context = payload["proposed_unresolved"][0][
            "canonical_path_context"
        ]
        assert path_context["is_canonical_goal_path"] is True
        assert (
            "/tasks/0/context/structured_goal/region_constraints"
            in path_context["missing_optional_ancestor_paths"]
        )
        return {
            "verdicts": [
                {
                    "unresolved_index": 0,
                    "disposition": "nonblocking",
                    "canonical_path": None,
                    "reason": "Absent optional state cannot create a user obligation.",
                }
            ]
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_unresolved_adjudication_sync",
        fake_adjudication,
    )

    result = pipeline_tools._repair_request_envelope_if_needed(
        request_text=request_text,
        run_dir="",
        structured_goal={},
        request_envelope=envelope,
        qa_history=[],
        semantic_audit=True,
    )

    goal = result["request_envelope"]["tasks"][0]["context"][
        "structured_goal"
    ]
    assert result["status"] == "ready"
    assert calls == {"audit": 2, "adjudication": 1}
    assert goal["target_scope"] == "gene-wide"
    assert "region_constraints" not in goal
