"""Regression tests for reference-skip design outcomes and unavailable comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from aso_agents import pipeline_tools, reference_decisions

def typed_interpreter(monkeypatch, action, scope="resources", **fields):
    def interpret(_name, _instructions, payload, _schema, validator):
        answer = payload["answers"][0]["answer"]
        result = dict(action=action, scope=scope, species=[], answer_index=0, evidence=answer,
                      reference_data_root="", download_dir="", model_path="", species_resolutions=[], reason="Unit semantic decision")
        result.update(fields)
        assert not validator(result)
        return result
    monkeypatch.setattr(reference_decisions, "semantic_result", interpret)


def test_reference_skip_classifies_as_skipped(monkeypatch) -> None:
    skip_record = {
        "status": "skipped",
        "reason": "reference_preflight_skip_requested_by_user",
        "preflight": {
            "requested_species": ["rattus_norvegicus"],
            "missing_requirements": [
                "rattus_norvegicus OpenSpliceAI model",
                "rattus_norvegicus genome FASTA",
            ],
        },
    }

    def fake_context(_run_dir: str) -> dict:
        return {
            "final_payload": {"reference_preflight_skip": skip_record},
            "structured_goal": {
                "gene_symbol": "MAPT",
                "species": "rattus_norvegicus",
                "intent": "knockdown",
                "design_programs": [{"modality": "gapmer"}],
            },
        }

    monkeypatch.setattr(pipeline_tools, "_goal_payload_context", fake_context)
    outcome = pipeline_tools._classify_design_outcome(
        child_run_dir="/tmp/agentaso_reference_skip_unit",
        task_context={"species": "rattus_norvegicus", "gene_symbol": "MAPT"},
        child_status="run_complete",
        generated_artifacts={},
    )

    assert outcome["outcome_status"] == "skipped"
    assert outcome["candidate_count"] == 0
    assert "reference_preflight" in outcome["missing_required_evidence"]
    assert "OpenSpliceAI model" in outcome["no_candidate_reason"]


def test_combined_model_and_reference_question() -> None:
    questions, reason, _context = pipeline_tools._reference_preflight_questions(
        {
            "error_message": "reference_data_missing",
            "model_missing_species": ["rattus_norvegicus"],
            "downloadable_species": ["rattus_norvegicus"],
            "download_default_dirs": {
                "rattus_norvegicus": "/refs/rattus_norvegicus",
            },
        }
    )

    assert reason == "reference_data_missing"
    assert len(questions) == 1
    question = questions[0]
    assert "OpenSpliceAI model" in question
    assert "download to default" in question
    assert "skip" in question


def test_reference_provisioning_waits_for_explicit_answer() -> None:
    decision = pipeline_tools._parse_reference_provisioning_decision(
        {
            "clarification": {
                "kind": "reference_preflight",
                "last_answers": [
                    {
                        "question": "Reference assets are missing for rattus_norvegicus. Should I download them now?",
                        "answer": "",
                    }
                ]
            }
        },
        {"downloadable_species": ["rattus_norvegicus"]},
    )

    assert decision["action"] == "ask"


def test_reference_provisioning_does_not_expand_species_skip(monkeypatch) -> None:
    typed_interpreter(monkeypatch, "skip", "species")
    decision = pipeline_tools._parse_reference_provisioning_decision(
        {
            "clarification": {
                "kind": "reference_preflight",
                "last_answers": [
                    {
                        "question": (
                            "These Ensembl organism(s) are missing required OpenSpliceAI model support "
                            "and reference assets: rattus_norvegicus."
                        ),
                        "answer": "Skip the rat species/task; continue only with supported branches.",
                    }
                ]
            }
        },
        {
            "model_missing_species": ["rattus_norvegicus"],
            "downloadable_species": ["rattus_norvegicus"],
        },
    )

    assert decision["action"] == "ask"


def test_reference_provisioning_extracts_explicit_model_path(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "rat_model.pt"
    model.write_text("placeholder", encoding="utf-8")
    typed_interpreter(monkeypatch, "download", model_path=str(model))
    decision = pipeline_tools._parse_reference_provisioning_decision(
        {
            "clarification": {
                "kind": "reference_preflight",
                "last_answers": [
                    {
                        "question": "Provide a local OpenSpliceAI model path for rattus_norvegicus.",
                        "answer": f"Use {model} and download references to default.",
                    }
                ]
            }
        },
        {"model_missing_species": ["rattus_norvegicus"]},
    )

    assert decision["action"] == "download"
    assert decision["model_path"] == str(model)


def test_reference_provisioning_uses_existing_reference_root(monkeypatch, tmp_path: Path) -> None:
    reference_root = tmp_path / "refs"
    reference_root.mkdir()
    typed_interpreter(monkeypatch, "use_existing", reference_data_root=str(reference_root))

    decision = pipeline_tools._parse_reference_provisioning_decision(
        {
            "clarification": {
                "kind": "reference_preflight",
                "last_answers": [
                    {
                        "question": "Reference assets are missing for human. Should I download them now?",
                        "answer": str(reference_root),
                    }
                ]
            }
        },
        {"downloadable_species": ["human"]},
    )

    assert decision["action"] == "use_existing"
    assert decision["reference_data_root"] == str(reference_root)


def test_reference_preflight_skipped_routes_to_partial_report_payload() -> None:
    state = {
        "request": "Design rat MAPT gapmers.",
        "status": "reference_preflight_skipped",
        "phase": "reference_preflight_skipped",
        "run_dir": "/tmp/agentaso_reference_skip_unit",
        "workflow_name": "design_new_asos",
        "request_mode": "single_task",
        "response_mode": "single_task",
    }
    planning = pipeline_tools._derive_next_actions(state)
    assert planning["terminal_status"] == "in_progress"
    assert planning["next_actions"][0]["tool"] == "tool_build_report_payload"


def test_compare_with_one_generated_set_completes_as_unavailable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    human_set = run_dir / "human_generated_aso_set.json"
    human_set.write_text(
        json.dumps({"rows": [{"sequence": "ACGTACGTACGTACGTACGT"}]}),
        encoding="utf-8",
    )
    state_path = run_dir / "run_state.json"
    request_envelope = {
        "request_id": "unit_reference_skip_compare",
        "request_text": "make me rat and human ASOs for gapmer mapt and compare overlap",
        "request_mode": "composite",
        "response_mode": "composite",
        "workflow_name": "composite_workflow",
        "tasks": [
            {
                "task_id": "design_human",
                "task_type": "design_new_asos",
                "label": "Design human MAPT",
                "context": {"gene_symbol": "MAPT", "species": "human"},
            },
            {
                "task_id": "design_rat",
                "task_type": "design_new_asos",
                "label": "Design rat MAPT",
                "context": {"gene_symbol": "MAPT", "species": "rattus_norvegicus"},
            },
            {
                "task_id": "compare_overlap",
                "task_type": "compare_aso_sets",
                "label": "Compare human and rat overlap",
                "depends_on": ["design_human", "design_rat"],
                "context": {},
            },
        ],
        "task_dependencies": [
            {"upstream_task_id": "design_human", "downstream_task_id": "compare_overlap"},
            {"upstream_task_id": "design_rat", "downstream_task_id": "compare_overlap"},
        ],
    }
    state = {
        "request": request_envelope["request_text"],
        "run_dir": str(run_dir),
        "status": "request_parsed_ready",
        "phase": "request_parsed_ready",
        "current_task_id": "compare_overlap",
        "request_envelope": request_envelope,
        "task_graph": {
            "tasks": request_envelope["tasks"],
            "dependencies": request_envelope["task_dependencies"],
        },
        "task_statuses": {
            "design_human": {
                "task_type": "design_new_asos",
                "status": "completed",
                "design_outcome": {
                    "outcome_status": "completed_with_candidates",
                    "candidate_count": 1,
                    "artifacts": {"generated_aso_set_json": str(human_set)},
                },
            },
            "design_rat": {
                "task_type": "design_new_asos",
                "status": "completed",
                "design_outcome": {
                    "outcome_status": "skipped",
                    "candidate_count": 0,
                    "no_candidate_reason": "Design was skipped because reference preflight was skipped by user request.",
                },
            },
            "compare_overlap": {"task_type": "compare_aso_sets", "status": "pending"},
        },
        "task_artifacts": {
            "design_human": {"generated_aso_set_json": str(human_set)},
            "design_rat": {},
        },
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    result = pipeline_tools.tool_compare_aso_sets(str(state_path))
    assert result["status"] == "ok"
    assert result["comparison_mode"] == "comparison_unavailable"
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["task_statuses"]["compare_overlap"]["status"] == "completed"
    payload_path = Path(result["artifacts"]["json"])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["comparison_mode"] == "comparison_unavailable"
    assert payload["loaded_set_count"] == 1
    assert payload["unavailable_sources"][0]["source_task_id"] == "design_rat"


def test_composite_summary_describes_skipped_designs_without_candidate_claim() -> None:
    completed_tasks = [
        {
            "task_id": "design_human",
            "task_type": "design_new_asos",
            "label": "Design human MAPT",
            "payload": {
                "child_run_dir": "/tmp/human",
                "report_excerpt": "Human MAPT gapmer ASO design run completed with saved outputs.",
                "design_outcome": {
                    "outcome_status": "skipped",
                    "candidate_count": 0,
                    "missing_required_evidence": ["reference_preflight"],
                    "no_candidate_reason": "Design was skipped because reference preflight was skipped by user request.",
                },
            },
        },
        {
            "task_id": "design_rat",
            "task_type": "design_new_asos",
            "label": "Design rat MAPT",
            "payload": {
                "child_run_dir": "/tmp/rat",
                "report_excerpt": "Reference preflight was skipped.",
                "design_outcome": {
                    "outcome_status": "skipped",
                    "candidate_count": 0,
                    "missing_required_evidence": ["reference_preflight"],
                    "no_candidate_reason": "Design was skipped because reference preflight was skipped by user request.",
                },
            },
        },
        {
            "task_id": "compare_overlap",
            "task_type": "compare_aso_sets",
            "label": "Compare overlap",
            "payload": {
                "comparison_mode": "comparison_unavailable",
                "reason": "fewer than two sequence-bearing ASO sets were available",
                "loaded_sources": [],
                "unavailable_sources": [
                    {"source_task_id": "design_human", "reason": "skipped"},
                    {"source_task_id": "design_rat", "reason": "skipped"},
                ],
                "loaded_set_count": 0,
                "generated_count": 0,
                "artifact_count": 0,
                "overlap_count": 0,
            },
        },
    ]
    response = pipeline_tools._build_deterministic_composite_response(
        {"request_summary": "Design human and rat MAPT gapmers and compare overlap."},
        {},
        completed_tasks=completed_tasks,
        blocked_tasks=[],
    )

    assert response is not None
    limitations = " ".join(response.get("limitations") or [])
    assert "design run(s) were skipped before candidate generation" in limitations
    assert "produced candidates but are missing required evidence" not in limitations
    assert "completed with saved outputs" not in str(response.get("markdown") or "")
