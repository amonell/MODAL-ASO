"""Sequence design can retain a splice objective while excluding model scores."""
from copy import deepcopy

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline


def goal(evaluate=False):
    return {
        "gene_symbol": "NEUTRAL", "species": "human", "target_context": "reference baseline",
        "intent": "splice_modulation", "desired_effect": "skipping",
        "splice_objective": {"exon_label": 3, "desired_delta": "decrease_inclusion"},
        "design_programs": [{
            "program_id": "sequence_library", "label": "sequence library", "modality": "steric",
            "intent": "splice_modulation", "objective": "Sequence design for later experimental study",
            "desired_effect": "skipping", "desired_delta": "decrease", "length": 19,
            "site_kind": "both", "run_splice_router": True, "run_splice_eval": evaluate,
        }],
    }


@pytest.mark.parametrize("evaluate", [False, True])
def test_intake_and_both_execution_resolvers_honor_explicit_evaluation_scope(evaluate, monkeypatch):
    current = goal(evaluate)
    # An aggregate flag cannot override the more specific branch declaration.
    current["run_splice_eval"] = not evaluate
    before = deepcopy(current)
    assert not pipeline._canonical_design_goal_contract_issues(current)
    assert runtime._resolve_run_splice_eval(current) is evaluate
    assert pipeline._design_request_requires_splice_eval(
        current["design_programs"][0], fallback_goal=current) is evaluate
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: {
        "structured_goal": current, "final_payload": {}, "payload": {},
    })
    assert pipeline._resolve_run_splice_eval("unused") is evaluate
    assert runtime._resolve_run_splice_router(current, "") is True
    # Reference/intent routing remains declared; effect execution requires predictions.
    assert pipeline._resolve_run_splice_router("unused") is evaluate
    assert current == before


@pytest.mark.parametrize("bad_flag", [None, "false", 0])
def test_missing_or_untyped_program_flag_still_requires_intake_repair(bad_flag):
    current = goal(bad_flag)
    assert "context.structured_goal.design_programs[0].run_splice_eval" in (
        pipeline._canonical_design_goal_contract_issues(current))


@pytest.mark.parametrize("enabled_index", [None, 0, 1])
def test_mixed_branches_require_prediction_only_for_enabled_members(enabled_index, monkeypatch):
    current = goal(False)
    second = deepcopy(current["design_programs"][0])
    second["program_id"] = "other_library"
    current["design_programs"].append(second)
    if enabled_index is not None:
        current["design_programs"][enabled_index]["run_splice_eval"] = True
    expected = enabled_index is not None
    assert runtime._resolve_run_splice_eval(current) is expected
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: {
        "structured_goal": current, "final_payload": {}, "payload": {},
    })
    assert pipeline._resolve_run_splice_eval("unused") is expected
    assert [pipeline._design_request_requires_splice_eval(p, fallback_goal=current)
            for p in current["design_programs"]] == [enabled_index == i for i in range(2)]


def test_legacy_missing_flags_preserve_splice_default():
    current = goal()
    current["design_programs"][0].pop("run_splice_eval")
    assert runtime._resolve_run_splice_eval(current) is True
    assert pipeline._design_request_requires_splice_eval(
        current["design_programs"][0], fallback_goal=current) is True


def test_default_exon_context_does_not_restore_excluded_model_scores():
    current = goal(False)
    current["splice_isoform_objectives"] = [
        {"role": "context", "transcript_id": "TX.1", "desired_effect": "skipping"}
    ]
    assert runtime._summarize_steric_program_modes(current) == {
        "has_steric_programs": True, "run_splice_eval": False, "run_splice_router": True,
    }


def test_sequence_only_preflight_retains_reference_work_without_requiring_prediction_assets(monkeypatch, tmp_path):
    current = goal(False)
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: {
        "structured_goal": current, "final_payload": {}, "payload": {},
    })
    monkeypatch.setattr(pipeline, "_required_tiling_reference_files", lambda _: [])
    monkeypatch.setattr(pipeline, "_tiling_reference_requirement_groups", lambda _: [])
    result = pipeline._preflight_reference_requirements(str(tmp_path))
    assert result["ok"] is True
    assert result["run_splice_eval"] is False
    assert result["resource_operations"]["human"] == ["genomic_tiling_and_mapping"]
    assert "splice_eval" not in result["required_components"]
