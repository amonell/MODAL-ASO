from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from aso_agents import pipeline_tools
from aso_agents.goal_contract import (
    CANONICAL_DESIGN_GOAL_INSTRUCTIONS,
    canonical_goal_field_registry,
    canonical_structured_goal_schema,
)
from aso_agents.structured_outputs import (
    GOAL_CONTRACT_AUDIT_OUTPUT_SCHEMA,
    GOAL_CONTRACT_REPAIR_OUTPUT_SCHEMA,
)


def _design_envelope(extra_goal: dict[str, Any] | None = None) -> dict[str, Any]:
    goal: dict[str, Any] = {
        "gene_symbol": "GENE",
        "species": "human",
        "target_context": "reviewed cells",
        "intent": "binding",
        "design_programs": [
            {
                "program_id": "primary",
                "label": "primary",
                "modality": "steric",
                "intent": "binding",
                "objective": "block the requested target",
                "desired_delta": "decrease",
            }
        ],
    }
    goal.update(deepcopy(extra_goal or {}))
    return {
        "schema_version": "v3",
        "request_mode": "single_task",
        "request_summary": "Design a steric blocker",
        "request_classifications": ["design_new_asos"],
        "tasks": [
            {
                "task_id": "design",
                "task_type": "design_new_asos",
                "label": "design",
                "native": True,
                "can_run_in_parallel": False,
                "context": {
                    "gene_symbol": "GENE",
                    "species": "human",
                    "target_context": "reviewed cells",
                    "structured_goal": goal,
                },
            }
        ],
        "task_dependencies": [],
        "final_outputs": ["design_report"],
        "response_mode": "report",
        "requires_execution": True,
        "assumptions": [],
    }


def _valid_audit() -> dict[str, Any]:
    return {
        "status": "valid",
        "patches": [],
        "goal_replacements": [],
        "field_dispositions": [],
        "unresolved": [],
    }


def test_canonical_registry_describes_nested_destinations_without_aliases() -> None:
    registry = canonical_goal_field_registry()
    by_path = {item["path_pattern"]: item for item in registry}

    assert "context.structured_goal.constraints.homopolymers" in by_path
    assert (
        "context.structured_goal.design_programs[*].desired_delta"
        in by_path
    )
    assert by_path[
        "context.structured_goal.constraints.homopolymers"
    ]["semantic_role"]
    assert all("aliases" not in item for item in registry)


def test_canonical_contract_separates_source_resolution_from_user_intent() -> None:
    assert "source-resolvable execution facts" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "downstream authoritative resolution" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "transcript_policy='agent_choice'" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "transcript_lock.version" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "splice_relative_window.feature_name" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "species is one anchor organism" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "Never concatenate" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "never store a lookup instruction" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS


def test_canonical_contract_separates_biological_and_bed_coordinate_spaces() -> None:
    assert "evaluation_targets, explicit_splice_targets" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "one-based and inclusive" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "[s-1,e)" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "Never copy a BED0 start back" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS

    item_schema = canonical_structured_goal_schema()["properties"][
        "explicit_splice_targets"
    ]["items"]
    assert item_schema["properties"]["start"]["minimum"] == 1
    assert "One-based inclusive" in item_schema["properties"]["start"]["description"]


def test_canonical_contract_separates_requested_exon_labels_from_resolved_indexes() -> None:
    assert "request-nomenclature fields" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "structural transcript-exon index" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert "must not renumber" in CANONICAL_DESIGN_GOAL_INSTRUCTIONS
    assert (
        canonical_structured_goal_schema()["properties"]["design_programs"]
        ["minItems"]
        == 1
    )


def test_audit_artifact_cannot_renumber_user_exon_label() -> None:
    envelope = _design_envelope(
        {
            "evaluation_targets": [
                {
                    "label": "requested exon",
                    "target_type": "exon_inclusion",
                    "exon_label": 4,
                    "desired_delta": "increase",
                }
            ]
        }
    )
    artifact = {
        "evidence_id": "artifact.0",
        "kind": "declared_task_artifact",
        "text": "The requested feature maps to transcript exon index 5 at chr2:2000-2060.",
    }

    result = pipeline_tools._apply_goal_audit_patches(
        envelope,
        [
            {
                "op": "replace",
                "path": (
                    "/tasks/0/context/structured_goal/"
                    "evaluation_targets/0/exon_label"
                ),
                "value": 5,
                "evidence_refs": ["artifact.0"],
                "evidence_quote": "maps to transcript exon index 5",
                "reason": "Use the resolved structural index.",
            }
        ],
        issue_records=[],
        evidence_sources=[artifact],
    )

    assert result["applied"] == []
    assert result["rejected"] == [
        {
            "path": (
                "/tasks/0/context/structured_goal/"
                "evaluation_targets/0/exon_label"
            ),
            "reason": (
                "request_nomenclature_requires_user_authored_evidence;"
                " preserve_the_user_label_and_store_resolved_mapping_separately"
            ),
        }
    ]
    assert (
        result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
        ["evaluation_targets"][0]["exon_label"]
        == 4
    )


def test_audit_artifact_can_add_coordinates_without_changing_user_exon_label() -> None:
    envelope = _design_envelope(
        {
            "evaluation_targets": [
                {
                    "label": "requested exon",
                    "target_type": "exon_inclusion",
                    "exon_label": 4,
                    "desired_delta": "increase",
                }
            ]
        }
    )
    result = pipeline_tools._apply_goal_audit_patches(
        envelope,
        [
            {
                "op": "add",
                "path": (
                    "/tasks/0/context/structured_goal/evaluation_targets/0/chrom"
                ),
                "value": "chr2",
                "evidence_refs": ["artifact.0"],
                "evidence_quote": "maps to chr2:2000-2060",
                "reason": "Attach a resolved coordinate without changing nomenclature.",
            },
        ],
        issue_records=[],
        evidence_sources=[
            {
                "evidence_id": "artifact.0",
                "kind": "declared_task_artifact",
                "text": "The requested feature maps to chr2:2000-2060.",
            }
        ],
    )

    assert result["rejected"] == []
    assert result["applied"]
    target = result["request_envelope"]["tasks"][0]["context"][
        "structured_goal"
    ]["evaluation_targets"][0]
    assert target["exon_label"] == 4
    assert target["chrom"] == "chr2"


def test_later_user_clarification_can_correct_requested_exon_label() -> None:
    envelope = _design_envelope(
        {
            "evaluation_targets": [
                {
                    "label": "requested exon",
                    "target_type": "exon_inclusion",
                    "exon_label": 4,
                    "desired_delta": "increase",
                }
            ]
        }
    )

    result = pipeline_tools._apply_goal_audit_patches(
        envelope,
        [
            {
                "op": "replace",
                "path": (
                    "/tasks/0/context/structured_goal/"
                    "evaluation_targets/0/exon_label"
                ),
                "value": 6,
                "evidence_refs": ["clarification.0.answer"],
                "evidence_quote": "Use exon 6",
                "reason": "The later user answer replaces the earlier label.",
            }
        ],
        issue_records=[],
        evidence_sources=[
            {
                "evidence_id": "clarification.0.answer",
                "kind": "clarification_answer",
                "text": "Use exon 6 for the requested design.",
            }
        ],
    )

    assert result["rejected"] == []
    assert (
        result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
        ["evaluation_targets"][0]["exon_label"]
        == 6
    )


def test_audit_relinks_exact_user_quote_when_source_id_is_misattributed() -> None:
    envelope = _design_envelope(
        {
            "evaluation_targets": [
                {
                    "label": "requested exon",
                    "target_type": "exon_inclusion",
                    "exon_label": 4,
                    "desired_delta": "increase",
                }
            ]
        }
    )

    result = pipeline_tools._apply_goal_audit_patches(
        envelope,
        [
            {
                "op": "replace",
                "path": (
                    "/tasks/0/context/structured_goal/"
                    "evaluation_targets/0/exon_label"
                ),
                "value": 6,
                "evidence_refs": ["artifact.0"],
                "evidence_quote": "Use exon 6",
                "reason": "Preserve the user's corrected label.",
            }
        ],
        issue_records=[],
        evidence_sources=[
            {
                "evidence_id": "request",
                "kind": "user_request",
                "text": "Use exon 6 for this design.",
            },
            {
                "evidence_id": "artifact.0",
                "kind": "declared_task_artifact",
                "text": "Use exon 6 as the display label.",
            },
        ],
    )

    assert result["rejected"] == []
    assert (
        result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
        ["evaluation_targets"][0]["exon_label"]
        == 6
    )


def test_audit_removes_unsupported_collection_items_without_emptying_required_list() -> None:
    envelope = _design_envelope()
    programs = envelope["tasks"][0]["context"]["structured_goal"][
        "design_programs"
    ]
    programs.extend(
        [
            {
                "program_id": "unsupported-a",
                "label": "unsupported A",
                "modality": "steric",
                "intent": "binding",
                "objective": "unsupported A",
            },
            {
                "program_id": "unsupported-b",
                "label": "unsupported B",
                "modality": "steric",
                "intent": "binding",
                "objective": "unsupported B",
            },
        ]
    )
    evidence_sources = [
        {
            "evidence_id": "request",
            "kind": "user_request",
            "text": "Design only the primary blocker.",
        }
    ]

    result = pipeline_tools._apply_goal_audit_patches(
        envelope,
        [
            {
                "op": "remove",
                "path": "/tasks/0/context/structured_goal/design_programs/1",
                "value": None,
                "evidence_refs": ["request"],
                "evidence_quote": "only the primary blocker",
                "reason": "Remove an unsupported branch.",
            },
            {
                "op": "remove",
                "path": "/tasks/0/context/structured_goal/design_programs/2",
                "value": None,
                "evidence_refs": ["request"],
                "evidence_quote": "only the primary blocker",
                "reason": "Remove an unsupported branch.",
            },
        ],
        issue_records=[],
        evidence_sources=evidence_sources,
    )

    assert result["rejected"] == []
    remaining = result["request_envelope"]["tasks"][0]["context"][
        "structured_goal"
    ]["design_programs"]
    assert [item["program_id"] for item in remaining] == ["primary"]

    cannot_remove_last = pipeline_tools._apply_goal_audit_patches(
        result["request_envelope"],
        [
            {
                "op": "remove",
                "path": "/tasks/0/context/structured_goal/design_programs/0",
                "value": None,
                "evidence_refs": ["request"],
                "evidence_quote": "only the primary blocker",
                "reason": "Attempt to empty a required collection.",
            }
        ],
        issue_records=[],
        evidence_sources=evidence_sources,
    )
    assert cannot_remove_last["applied"] == []
    assert cannot_remove_last["rejected"][0]["reason"] == (
        "remove_target_not_optional_canonical_field"
    )


def test_unknown_field_finder_reports_exact_nested_json_pointers() -> None:
    envelope = _design_envelope(
        {
            "invented_container": {"value": 1},
            "constraints": {
                "gc_content": {"min": 0.3, "invented_bound": 0.4}
            },
        }
    )

    findings = pipeline_tools._noncanonical_goal_field_findings(envelope)

    assert [item["source_path"] for item in findings] == [
        "/tasks/0/context/structured_goal/invented_container",
        "/tasks/0/context/structured_goal/constraints/gc_content/invented_bound",
    ]
    assert [item["finding_index"] for item in findings] == [0, 1]


def test_semantic_checker_relocates_unknown_field_through_repair_agent(
    monkeypatch: Any,
) -> None:
    request_text = (
        "Design a steric blocker in reviewed cells and require a GC fraction "
        "from 0.3 to 0.6."
    )
    invented_key = "model_created_filter_bucket"
    envelope = _design_envelope(
        {invented_key: {"lower": 0.3, "upper": 0.6}}
    )
    audit_calls = {"count": 0}
    repair_calls = {"count": 0}

    async def fake_audit(prompt: str) -> dict[str, Any]:
        audit_calls["count"] += 1
        if audit_calls["count"] == 2:
            assert '"audit_mode": "post_repair_semantic_verification"' in prompt
            protocol = pipeline_tools._goal_audit_protocol_payload(prompt)
            assert invented_key not in json.dumps(protocol["current_request_envelope"])
            removed = next(change for change in protocol["latest_repair_transition"]["changes"]
                           if change["path"].endswith("/" + invented_key))
            assert removed["before"]["present"] is True
            assert removed["after"] == {"present": False}
            return _valid_audit()
        assert '"noncanonical_field_findings"' in prompt
        assert invented_key in prompt
        assert "context.structured_goal.constraints.gc_content" in prompt
        return {
            "status": "repairable",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [
                {
                    "finding_index": 0,
                    "action": "relocate",
                    "source_path": (
                        "/tasks/0/context/structured_goal/" + invented_key
                    ),
                    "destination_patches": [
                        {
                            "path": "/tasks/0/context/structured_goal/constraints",
                            "value": {
                                "gc_content": {"min": 0.3, "max": 0.6}
                            },
                            "evidence_refs": ["request"],
                            "evidence_quote": "GC fraction from 0.3 to 0.6",
                            "reason": "The request supplies canonical GC bounds.",
                        }
                    ],
                    "reason": "Move supported meaning to the schema destination.",
                }
            ],
            "unresolved": [],
        }

    async def fake_repair(prompt: str) -> dict[str, Any]:
        repair_calls["count"] += 1
        assert invented_key in prompt
        return {
            "status": "applied",
            "operations": [
                {
                    "op": "add",
                    "path": "/tasks/0/context/structured_goal/constraints",
                    "value": {"gc_content": {"min": 0.3, "max": 0.6}},
                    "finding_index": 0,
                    "destination_index": 0,
                    "reason": "Apply the supplied destination.",
                },
                {
                    "op": "remove",
                    "path": (
                        "/tasks/0/context/structured_goal/" + invented_key
                    ),
                    "value": None,
                    "finding_index": 0,
                    "destination_index": None,
                    "reason": "Remove the noncanonical source.",
                },
            ],
            "errors": [],
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_repair_agent_once",
        fake_repair,
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
    assert audit_calls["count"] == 2
    assert repair_calls["count"] == 1
    assert invented_key not in goal
    assert goal["constraints"] == {
        "gc_content": {"min": 0.3, "max": 0.6}
    }


def test_unsupported_unknown_field_is_removed_without_clarification(
    monkeypatch: Any,
) -> None:
    request_text = "Design a steric blocker in reviewed cells."
    invented_key = "generated_execution_defaults"
    envelope = _design_envelope({invented_key: {"limit": 4}})
    audit_calls = {"count": 0}

    async def fake_audit(prompt: str) -> dict[str, Any]:
        audit_calls["count"] += 1
        if audit_calls["count"] == 2:
            protocol = pipeline_tools._goal_audit_protocol_payload(prompt)
            assert invented_key not in json.dumps(protocol["current_request_envelope"])
            removed = next(change for change in protocol["latest_repair_transition"]["changes"]
                           if change["path"].endswith("/" + invented_key))
            assert removed["before"]["present"] is True
            assert removed["after"] == {"present": False}
            return _valid_audit()
        return {
            "status": "repairable",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [
                {
                    "finding_index": 0,
                    "action": "remove",
                    "source_path": (
                        "/tasks/0/context/structured_goal/" + invented_key
                    ),
                    "destination_patches": [],
                    "reason": "The generated value has no user evidence.",
                }
            ],
            "unresolved": [],
        }

    async def fake_repair(_prompt: str) -> dict[str, Any]:
        return {
            "status": "applied",
            "operations": [
                {
                    "op": "remove",
                    "path": (
                        "/tasks/0/context/structured_goal/" + invented_key
                    ),
                    "value": None,
                    "finding_index": 0,
                    "destination_index": None,
                    "reason": "Remove the unsupported field.",
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_repair_agent_once",
        fake_repair,
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
    assert audit_calls["count"] == 2
    assert invented_key not in goal


def test_contract_gap_is_actionable_failure_not_a_new_alias(
    monkeypatch: Any,
) -> None:
    invented_key = "unsupported_material_concept"
    envelope = _design_envelope({invented_key: "requested value"})

    async def fake_audit(_prompt: str) -> dict[str, Any]:
        return {
            "status": "repairable",
            "patches": [],
            "goal_replacements": [],
            "field_dispositions": [
                {
                    "finding_index": 0,
                    "action": "contract_gap",
                    "source_path": (
                        "/tasks/0/context/structured_goal/" + invented_key
                    ),
                    "destination_patches": [],
                    "reason": "The registry has no field for this supported concept.",
                }
            ],
            "unresolved": [],
        }

    monkeypatch.setattr(
        pipeline_tools,
        "_run_goal_contract_audit_agent_once",
        fake_audit,
    )

    result = pipeline_tools._repair_request_envelope_if_needed(
        request_text="Design a steric blocker with the requested value in reviewed cells.",
        run_dir="",
        structured_goal={},
        request_envelope=envelope,
        qa_history=[],
        semantic_audit=True,
    )

    assert result["status"] == "model_failure"
    assert result["reason"] == "goal_contract_registry_gap"
    assert all(
        not item["path_pattern"].endswith("." + invented_key)
        for item in canonical_goal_field_registry()
    )


def test_repair_agent_cannot_change_checker_destination() -> None:
    request_text = "Require a GC fraction from 0.3 to 0.6."
    invented_key = "temporary_filter_container"
    envelope = _design_envelope(
        {invented_key: {"lower": 0.3, "upper": 0.6}}
    )
    dispositions = [
        {
            "finding_index": 0,
            "action": "relocate",
            "source_path": (
                "/tasks/0/context/structured_goal/" + invented_key
            ),
            "destination_patches": [
                {
                    "path": "/tasks/0/context/structured_goal/constraints",
                    "value": {"gc_content": {"min": 0.3, "max": 0.6}},
                    "evidence_refs": ["request"],
                    "evidence_quote": "GC fraction from 0.3 to 0.6",
                    "reason": "Use the canonical constraint container.",
                }
            ],
            "reason": "Relocate the supported value.",
        }
    ]
    validated = pipeline_tools._validate_goal_field_dispositions(
        envelope,
        dispositions,
        evidence_sources=[
            {
                "evidence_id": "request",
                "kind": "user_request",
                "text": request_text,
            }
        ],
    )

    result = pipeline_tools._apply_goal_field_repair_operations(
        envelope,
        validated["accepted"],
        {
            "status": "applied",
            "operations": [
                {
                    "op": "add",
                    "path": "/tasks/0/context/structured_goal/target_region",
                    "value": "changed destination",
                    "finding_index": 0,
                    "destination_index": 0,
                    "reason": "Invalid change.",
                },
                {
                    "op": "remove",
                    "path": (
                        "/tasks/0/context/structured_goal/" + invented_key
                    ),
                    "value": None,
                    "finding_index": 0,
                    "destination_index": None,
                    "reason": "Remove source.",
                },
            ],
            "errors": [],
        },
    )

    assert result["applied"] == []
    assert result["rejected"]
    assert result["request_envelope"] == envelope


def test_runtime_projection_is_not_applied_to_staged_audit_envelope(
    monkeypatch: Any,
) -> None:
    calls = {"count": 0}

    def fake_projection(envelope: dict[str, Any], _request: str) -> None:
        calls["count"] += 1
        envelope["runtime_projection_applied"] = True

    monkeypatch.setattr(
        pipeline_tools,
        "_normalize_request_envelope_embedded_parameter_constraints",
        fake_projection,
    )
    raw = _design_envelope()

    staged = pipeline_tools._build_request_envelope(
        request_text="Design a steric blocker in reviewed cells.",
        request_id="staged",
        structured_goal=None,
        raw_envelope=raw,
        finalize_design_contract=False,
    )
    assert calls["count"] == 0
    assert "runtime_projection_applied" not in staged

    finalized = pipeline_tools._build_request_envelope(
        request_text="Design a steric blocker in reviewed cells.",
        request_id="finalized",
        structured_goal=None,
        raw_envelope=raw,
        finalize_design_contract=True,
    )
    assert calls["count"] == 1
    assert finalized["runtime_projection_applied"] is True


def test_structured_output_contracts_require_dispositions_and_exact_operations() -> None:
    assert "field_dispositions" in GOAL_CONTRACT_AUDIT_OUTPUT_SCHEMA[
        "required"
    ]
    disposition_schema = GOAL_CONTRACT_AUDIT_OUTPUT_SCHEMA["properties"][
        "field_dispositions"
    ]["items"]
    assert disposition_schema["additionalProperties"] is False
    assert set(disposition_schema["properties"]["action"]["enum"]) == {
        "relocate",
        "remove",
        "contract_gap",
        "needs_user_input",
    }
    operation_schema = GOAL_CONTRACT_REPAIR_OUTPUT_SCHEMA["properties"][
        "operations"
    ]["items"]
    assert operation_schema["additionalProperties"] is False
    assert set(operation_schema["properties"]["op"]["enum"]) == {
        "add",
        "replace",
        "remove",
    }


def test_goal_audit_noop_replacements_do_not_block_field_repair() -> None:
    envelope = {
        "tasks": [
            {
                "context": {
                    "structured_goal": {
                        "run_splice_router": False,
                    }
                }
            }
        ]
    }
    patches = [
        {
            "op": "replace",
            "path": "/tasks/0/context/structured_goal/run_splice_router",
            "value": False,
        },
        {
            "op": "add",
            "path": "/tasks/0/context/structured_goal/run_splice_eval",
            "value": False,
        },
    ]

    assert pipeline_tools._goal_audit_effective_patches(envelope, patches) == [
        patches[1]
    ]
