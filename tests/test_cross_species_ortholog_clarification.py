from __future__ import annotations

import json
import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import run_goal_intake
from aso_agents import pipeline_tools
from aso_agents.run_layout import resolve_run_path

chooser = importlib.import_module("aso_agents.transcript_chooser_agent")


SOURCE_GENE_ID = "ENSG00000171291"
MOUSE_GENE_ID = "ENSMUSG00000055150"
ALT_MOUSE_GENE_ID = "ENSMUSG00000099999"


def _ambiguous_mouse_candidate() -> dict[str, Any]:
    return {
        "rank": 1,
        "species": "mouse",
        "ensembl_species": "mus_musculus",
        "gene_id": MOUSE_GENE_ID,
        "gene_symbol": "Zfp78",
        "description": "zinc finger protein 78",
        "homology_type": "ortholog_one2many",
        "percent_identity": 41.3,
        "source_percent_identity": 40.8,
        "source_gene_id": SOURCE_GENE_ID,
        "source_gene_symbol": "ZNF439",
        "resolution_source": "ensembl_homology",
    }


def _ambiguous_targets() -> dict[str, dict[str, Any]]:
    candidate = _ambiguous_mouse_candidate()
    return {
        "human": {
            "status": "resolved",
            "species": "human",
            "gene_id": SOURCE_GENE_ID,
            "gene_symbol": "ZNF439",
            "resolution_source": "primary_goal_gene",
        },
        "mouse": {
            "status": "ambiguous",
            "species": "mouse",
            "source_gene_id": SOURCE_GENE_ID,
            "source_gene_symbol": "ZNF439",
            "gene_id": candidate["gene_id"],
            "gene_symbol": candidate["gene_symbol"],
            "homology_type": candidate["homology_type"],
            "percent_identity": candidate["percent_identity"],
            "source_percent_identity": candidate["source_percent_identity"],
            "resolution_source": "ensembl_homology",
            "ortholog_candidates": [
                candidate,
                {
                    "rank": 2,
                    "species": "mouse",
                    "ensembl_species": "mus_musculus",
                    "gene_id": ALT_MOUSE_GENE_ID,
                    "gene_symbol": "Zfp999",
                    "homology_type": "ortholog_one2many",
                    "percent_identity": 35.2,
                    "source_gene_id": SOURCE_GENE_ID,
                    "source_gene_symbol": "ZNF439",
                    "resolution_source": "ensembl_homology",
                },
            ],
            "selected_candidate": candidate,
            "ambiguity_reasons": [
                "homology_type_not_one2one",
                "multiple_ortholog_candidates",
                "percent_identity_below_50",
            ],
            "requires_clarification": True,
        },
    }


def _write_cross_species_goal(run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir()
    structured_goal: dict[str, Any] = {
        "gene_symbol": "ZNF439",
        "gene_id": SOURCE_GENE_ID,
        "species": ["human", "mouse"],
        "intent": "knockdown",
        "target_context": {
            "tissue": "generic cell culture",
            "disease_context": "baseline screening",
        },
        "cross_species_exact_match": {
            "required": True,
            "species": ["human", "mouse"],
        },
        "design_programs": [
            {
                "program_id": "gapmer_znf439",
                "modality": "gapmer",
                "label": "ZNF439 gapmers",
            }
        ],
    }
    request_envelope = {
        "workflow_name": "design_new_asos",
        "tasks": [
            {
                "task_id": "design_znf439",
                "task_type": "design_new_asos",
                "native": True,
                "label": "ZNF439 gapmers",
                "context": {"structured_goal": deepcopy(structured_goal)},
            }
        ],
    }
    payload = {
        "request": "Design ZNF439 gapmers that hit both human and mouse.",
        "request_envelope": request_envelope,
    }
    final_payload = deepcopy(structured_goal)
    final_payload["request_envelope"] = deepcopy(request_envelope)
    pipeline_tools._save_goal_payload(str(run_dir), payload, structured_goal, final_payload)
    return request_envelope


def _seed_ambiguous_ortholog_goal(
    run_dir: Path,
    request_envelope: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    targets = _ambiguous_targets()
    structured_goal = json.loads(resolve_run_path(run_dir, "structured_goal.json").read_text(encoding="utf-8"))
    final_payload = deepcopy(structured_goal)
    structured_goal["cross_species_targets"] = deepcopy(targets)
    final_payload["cross_species_targets"] = deepcopy(targets)
    pending = pipeline_tools._cross_species_ambiguous_ortholog_targets(targets)
    structured_goal["pending_cross_species_ortholog_confirmations"] = deepcopy(pending)
    final_payload["pending_cross_species_ortholog_confirmations"] = deepcopy(pending)
    pipeline_tools._save_goal_payload(
        str(run_dir),
        {
            "request": "Design ZNF439 gapmers that hit both human and mouse.",
            "request_envelope": request_envelope,
        },
        structured_goal,
        final_payload,
    )
    question = pipeline_tools._cross_species_ortholog_clarification_payload(
        structured_goal,
        pending,
    )["questions"][0]
    return structured_goal, pending, question


def _write_blocked_ortholog_state(
    state_path: Path,
    run_dir: Path,
    request_envelope: dict[str, Any],
    question: str,
) -> None:
    state = {
        "request": "Design ZNF439 gapmers that hit both human and mouse.",
        "status": "needs_user_input",
        "phase": "task_waiting_for_clarification",
        "run_dir": str(run_dir),
        "request_envelope": request_envelope,
        "current_task_id": "design_znf439",
        "blocked_task_ids": ["design_znf439"],
        "task_validation": {
            "design_znf439": {
                "task_id": "design_znf439",
                "task_type": "design_new_asos",
                "label": "ZNF439 gapmers",
                "status": "blocked",
                "issues": ["cross_species_ortholog_selection"],
                "missing_fields": ["cross_species_ortholog_selection"],
                "questions": [question],
                "reason": "cross_species_ortholog_selection_required",
                "suggested_answers": ["Use Zfp78"],
            }
        },
        "task_statuses": {
            "design_znf439": {
                "task_type": "design_new_asos",
                "native": True,
                "status": "blocked",
            }
        },
        "clarification_queue": [
            {
                "task_id": "design_znf439",
                "label": "ZNF439 gapmers",
                "missing_fields": ["cross_species_ortholog_selection"],
                "questions": [question],
                "reason": "cross_species_ortholog_selection_required",
                "suggested_answers": ["Use Zfp78"],
            }
        ],
        "pending_questions": [question],
        "clarification": {
            "kind": "cross_species_ortholog_selection",
            "reason": "cross_species_ortholog_selection_required",
            "restart_stage": "region_assembly",
            "pending_apply": False,
        },
        "last_error": None,
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")


def test_resolve_ortholog_gene_marks_one_to_many_low_identity_as_ambiguous(monkeypatch: Any) -> None:
    def fake_lookup_gene_by_id(gene_id: str) -> dict[str, Any]:
        if gene_id == SOURCE_GENE_ID:
            return {"species": "homo_sapiens", "display_name": "ZNF439"}
        if gene_id == MOUSE_GENE_ID:
            return {
                "species": "mus_musculus",
                "display_name": "Zfp78",
                "description": "zinc finger protein 78",
            }
        if gene_id == ALT_MOUSE_GENE_ID:
            return {
                "species": "mus_musculus",
                "display_name": "Zfp999",
                "description": "alternate zinc finger candidate",
            }
        return {"error": "not_found"}

    def fake_get_ensembl_json(path: str, params: dict[str, str]) -> tuple[Any, Any]:
        assert path == f"/homology/id/homo_sapiens/{SOURCE_GENE_ID}"
        assert params["target_species"] == "mus_musculus"
        return (
            SimpleNamespace(status_code=200, url="https://rest.ensembl.org/homology", text=""),
            {
                "data": [
                    {
                        "homologies": [
                            {
                                "type": "ortholog_one2many",
                                "source": {"id": SOURCE_GENE_ID, "perc_id": 40.8},
                                "target": {
                                    "id": MOUSE_GENE_ID,
                                    "display_id": "Zfp78",
                                    "perc_id": 41.3,
                                },
                            },
                            {
                                "type": "ortholog_one2many",
                                "source": {"id": SOURCE_GENE_ID, "perc_id": 34.5},
                                "target": {
                                    "id": ALT_MOUSE_GENE_ID,
                                    "display_id": "Zfp999",
                                    "perc_id": 35.2,
                                },
                            },
                        ]
                    }
                ]
            },
        )

    monkeypatch.setattr(chooser, "lookup_gene_by_id", fake_lookup_gene_by_id)
    monkeypatch.setattr(chooser, "_get_ensembl_json", fake_get_ensembl_json)
    monkeypatch.setenv("ASO_CROSS_SPECIES_ORTHOLOG_AUTO_MIN_IDENTITY", "50")

    resolved = chooser.resolve_ortholog_gene(f"{SOURCE_GENE_ID}.14", "mouse")

    assert resolved["status"] == "resolved"
    assert resolved["target_gene_id"] == MOUSE_GENE_ID
    assert resolved["requires_clarification"] is True
    assert resolved["ambiguous"] is True
    assert resolved["confidence"] == "requires_user_confirmation"
    assert resolved["ortholog_candidates"][0]["gene_symbol"] == "Zfp78"
    assert resolved["ortholog_candidates"][1]["gene_id"] == ALT_MOUSE_GENE_ID
    assert set(resolved["ambiguity_reasons"]) >= {
        "homology_type_not_one2one",
        "multiple_ortholog_candidates",
        "percent_identity_below_50",
    }


def test_intake_cross_species_targets_preserve_ambiguous_ortholog_pending_state(
    monkeypatch: Any,
) -> None:
    candidate = _ambiguous_mouse_candidate()

    monkeypatch.setattr(
        run_goal_intake,
        "lookup_gene_by_id",
        lambda _gene_id: {"display_name": "ZNF439", "species": "homo_sapiens"},
    )
    monkeypatch.setattr(
        run_goal_intake,
        "resolve_ortholog_gene",
        lambda _gene_id, _species: {
            "status": "resolved",
            "source_gene_id": SOURCE_GENE_ID,
            "source_gene_symbol": "ZNF439",
            "target_gene_id": candidate["gene_id"],
            "target_gene_symbol": candidate["gene_symbol"],
            "homology_type": candidate["homology_type"],
            "percent_identity": candidate["percent_identity"],
            "source_percent_identity": candidate["source_percent_identity"],
            "resolution_source": "ensembl_homology",
            "ortholog_candidates": [candidate],
            "selected_candidate": candidate,
            "requires_clarification": True,
            "ambiguous": True,
            "ambiguity_reasons": ["homology_type_not_one2one"],
        },
    )
    structured_goal = {
        "gene_symbol": "ZNF439",
        "gene_id": SOURCE_GENE_ID,
        "species": "human",
    }

    targets = run_goal_intake._resolve_cross_species_gene_targets(
        structured_goal,
        ["human", "mouse"],
    )

    assert targets["human"]["status"] == "resolved"
    assert targets["mouse"]["status"] == "ambiguous"
    assert targets["mouse"]["requires_clarification"] is True
    assert targets["mouse"]["ortholog_candidates"][0]["gene_id"] == MOUSE_GENE_ID
    pending = structured_goal["pending_cross_species_ortholog_confirmations"]
    assert pending[0]["species"] == "mouse"
    assert pending[0]["candidates"][0]["gene_symbol"] == "Zfp78"


def test_deterministic_ortholog_answer_ignores_candidate_ids_in_question() -> None:
    candidates = [_ambiguous_mouse_candidate()]
    question = (
        "Options: Use Zfp78 "
        f"({MOUSE_GENE_ID}; ortholog_one2many; 41.3% identity); or stop this target."
    )

    unresolved = pipeline_tools._deterministic_cross_species_ortholog_answer(
        "What are the options?",
        question,
        candidates,
    )
    selected = pipeline_tools._deterministic_cross_species_ortholog_answer(
        "Use Zfp78",
        question,
        candidates,
    )

    assert unresolved["decision"] == "clarify"
    assert selected["decision"] == "select"
    assert selected["gene_id"] == MOUSE_GENE_ID


def test_tool_build_gapmer_regions_blocks_for_ambiguous_ortholog(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "run"
    request_envelope = _write_cross_species_goal(run_dir)
    state_path = tmp_path / "state.json"
    state = {
        "request": "Design ZNF439 gapmers that hit both human and mouse.",
        "status": "steric_regions_ready",
        "phase": "steric_regions_ready",
        "run_dir": str(run_dir),
        "request_envelope": request_envelope,
        "current_task_id": "design_znf439",
        "clarification": {"isoform_scope_confirmed": True},
        "pending_questions": [],
        "last_error": None,
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    fake_runtime = SimpleNamespace(
        _cross_species_requested_species=lambda _goal: ["human", "mouse"],
        _resolve_cross_species_gene_targets=lambda _goal, _species: _ambiguous_targets(),
    )
    monkeypatch.setattr(pipeline_tools, "_goal_runtime_module", lambda: fake_runtime)

    result = pipeline_tools.tool_build_gapmer_regions(str(state_path))

    assert result["status"] == "ok"
    assert result["needs_user_input"] is True
    assert result["reason"] == "cross_species_ortholog_selection_required"
    assert "Use Zfp78" in result["suggested_answers"][0]
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["task_validation"]["design_znf439"]["status"] == "blocked"
    assert saved_state["task_validation"]["design_znf439"]["missing_fields"] == [
        "cross_species_ortholog_selection"
    ]
    saved_goal = json.loads(resolve_run_path(run_dir, "goal_intake.json").read_text(encoding="utf-8"))
    final_payload = saved_goal["final_payload"]
    assert final_payload["cross_species_preflight"]["status"] == "ortholog_selection_required"
    assert final_payload["pending_cross_species_ortholog_confirmations"][0]["species"] == "mouse"


def test_apply_cross_species_ortholog_clarification_unblocks_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    request_envelope = _write_cross_species_goal(run_dir)
    _structured_goal, _pending, question = _seed_ambiguous_ortholog_goal(run_dir, request_envelope)
    state_path = tmp_path / "state.json"
    _write_blocked_ortholog_state(state_path, run_dir, request_envelope, question)

    submitted = pipeline_tools.submit_clarification_answers(
        str(state_path),
        [{"question": question, "answer": "Use Zfp78"}],
        answer_summary="Use the Zfp78 mouse ortholog.",
    )
    applied = pipeline_tools.apply_clarification_to_goal(str(state_path))

    assert submitted["status"] == "ok"
    assert applied["status"] == "ok"
    assert applied["restart_stage"] == "region_assembly"
    assert applied["applied_cross_species_ortholog_confirmations"] == 1
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["blocked_task_ids"] == []
    assert saved_state["task_validation"]["design_znf439"]["status"] == "runnable"
    assert saved_state["task_statuses"]["design_znf439"]["status"] != "blocked"
    saved_goal = json.loads(resolve_run_path(run_dir, "goal_intake.json").read_text(encoding="utf-8"))
    final_payload = saved_goal["final_payload"]
    assert "pending_cross_species_ortholog_confirmations" not in final_payload
    mouse_target = final_payload["cross_species_targets"]["mouse"]
    assert mouse_target["status"] == "resolved"
    assert mouse_target["gene_id"] == MOUSE_GENE_ID
    assert mouse_target["resolution_source"] == "user_clarification"
    assert mouse_target["user_confirmed"] is True
    assert final_payload["cross_species_preflight"]["status"] == "user_resolved_ortholog_selection"


def test_apply_cross_species_ortholog_clarification_reprompts_wrong_species_id(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "run"
    request_envelope = _write_cross_species_goal(run_dir)
    _structured_goal, _pending, question = _seed_ambiguous_ortholog_goal(run_dir, request_envelope)
    state_path = tmp_path / "state.json"
    _write_blocked_ortholog_state(state_path, run_dir, request_envelope, question)
    monkeypatch.setattr(
        pipeline_tools,
        "lookup_gene_by_id",
        lambda _gene_id: {"species": "homo_sapiens", "display_name": "ZNF439"},
    )

    submitted = pipeline_tools.submit_clarification_answers(
        str(state_path),
        [{"question": question, "answer": f"Use {SOURCE_GENE_ID}"}],
        answer_summary="User supplied a human gene ID for the mouse ortholog.",
    )
    applied = pipeline_tools.apply_clarification_to_goal(str(state_path))

    assert submitted["status"] == "ok"
    assert applied["status"] == "ok"
    assert applied["needs_user_input"] is True
    assert applied["applied_cross_species_ortholog_confirmations"] == 0
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "needs_user_input"
    assert saved_state["pending_questions"]
    assert saved_state["last_error"]["kind"] == "cross_species_ortholog_selection_not_applied"
    saved_goal = json.loads(resolve_run_path(run_dir, "goal_intake.json").read_text(encoding="utf-8"))
    final_payload = saved_goal["final_payload"]
    assert final_payload["cross_species_targets"]["mouse"]["status"] == "ambiguous"
    assert final_payload["pending_cross_species_ortholog_confirmations"][0]["species"] == "mouse"
