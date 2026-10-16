"""Isoform-selective RNase-H programs must not create an implicit splice task."""
from copy import deepcopy

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline


def goal():
    return {
        "intent": "splice_modulation",  # Legacy summary conflicts with the executable program.
        "design_programs": [{"program_id": "rnase_h", "modality": "gapmer",
                             "intent": "knockdown", "objective": "Reduce the intended mature transcript"}],
        "splice_isoform_objectives": [{"transcript_id": "intended", "desired_abundance": "decrease"}],
        "preservation_targets": [{"action": "preserve", "transcript_id": "protected"}],
        "evaluation_targets": [{"objective_type": "identity_anchor", "chrom": "chr7",
                                 "start": 150, "end": 220, "transcript_id": "intended"}],
    }


@pytest.mark.parametrize("intent", ["knockdown", "splice_modulation", "binding"])
def test_canonical_rnase_h_program_does_not_infer_splice_work(intent):
    value = goal()
    value["intent"] = intent
    original = deepcopy(value)
    assert runtime._resolve_run_splice_eval(value) is False
    assert runtime._resolve_run_splice_router(value, "Use the supplied mature transcript") is False
    assert runtime._should_skip_splice_strategy(value) is True
    assert runtime._evaluation_targets_are_splice_scoped(value) is False
    assert pipeline._explicit_splice_coordinate_intent(value, "Target chr7:150-220") is False
    assert value == original


@pytest.mark.parametrize("scope", ["global", "program"])
@pytest.mark.parametrize("flag", ["run_splice_eval", "run_splice_router"])
def test_explicit_splice_request_remains_enabled(scope, flag):
    value = goal()
    (value if scope == "global" else value["design_programs"][0])[flag] = True
    assert runtime._evaluation_targets_are_splice_scoped(value) is True
    assert runtime._should_skip_splice_strategy(value) is False
    assert pipeline._explicit_splice_coordinate_intent(value, "Target chr7:150-220") is True


def test_mixed_steric_splice_and_gapmer_programs_keep_splice_work():
    value = goal()
    value["design_programs"].append({"program_id": "splice", "modality": "steric",
                                     "intent": "splice_modulation", "run_splice_eval": True,
                                     "run_splice_router": True})
    assert runtime._resolve_run_splice_eval(value) is True
    assert runtime._resolve_run_splice_router(value, "") is True
    assert runtime._should_skip_splice_strategy(value) is False
    assert runtime._evaluation_targets_are_splice_scoped(value) is True


@pytest.mark.parametrize("programs", [None, [], [{"intent": "knockdown"}],
                                      [{"modality": "gapmer"}], [{"modality": "other", "intent": "knockdown"}]])
def test_incomplete_or_legacy_programs_keep_existing_resolution(programs):
    value = goal()
    value["design_programs"] = programs
    assert runtime._resolve_run_splice_eval(value) is True
    assert runtime._evaluation_targets_are_splice_scoped(value) is True


def test_program_with_exon_skipping_remains_a_splice_request():
    value = goal()
    value["design_programs"][0]["desired_effect"] = "skipping"
    assert runtime._evaluation_targets_are_splice_scoped(value) is True
    assert runtime._should_skip_splice_strategy(value) is False
