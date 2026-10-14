"""Reference coordinates must not manufacture transcript-abundance outcomes."""
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools


def structural_goal(target_type="splice_junction"):
    goal = {
        "intent": "splice_modulation",
        "transcript_lock": {"kind": "explicit_id", "value": "TX-REFERENCE.3"},
        "design_programs": [{
            "program_id": "skip", "label": "Skip the reference exon",
            "modality": "steric", "intent": "splice_modulation",
            "objective": "Decrease inclusion of exon 4 through the donor junction.",
            "transcript_id": "TX-REFERENCE.3", "exon_label": 4,
            "desired_effect": "skipping", "desired_delta": "decrease",
            "site_kind": "donor", "run_splice_router": True,
            "run_splice_eval": True,
        }],
        "evaluation_targets": [{
            "objective_type": "route_target", "target_type": target_type,
            "transcript_id": "TX-REFERENCE.3", "exon_label": 4,
            "desired_effect": "skipping", "desired_delta": "decrease",
            "site_kind": "donor",
        }],
        "splice_isoform_objectives": [{
            "objective_scope": "default_exon_context", "exon_label": 4,
            "desired_effect": "skipping", "desired_abundance": "decrease",
        }],
    }
    if target_type is None:
        goal["evaluation_targets"][0].pop("target_type")
    return goal


@pytest.mark.parametrize("target_type", [
    "exon", "splice_junction", "splice_acceptor", "splice_donor",
    "splice_site", "splice_site_window", None,
])
def test_reference_scoped_structural_route_needs_no_abundance_objective(target_type):
    assert pipeline_tools._canonical_design_goal_contract_issues(
        structural_goal(target_type)
    ) == []


@pytest.mark.parametrize("target_type", [
    "transcript", "transcript_cohort", "isoform", "isoform_cohort",
])
def test_explicit_abundance_or_unknown_type_is_not_reinterpreted(target_type):
    assert "context.structured_goal.splice_isoform_objectives" in (
        pipeline_tools._canonical_design_goal_contract_issues(structural_goal(target_type))
    )


def test_unknown_target_type_is_rejected_at_its_canonical_field():
    assert "context.structured_goal.evaluation_targets[0].target_type" in (
        pipeline_tools._canonical_design_goal_contract_issues(structural_goal("unknown"))
    )


@pytest.mark.parametrize("change", [
    "different_lock", "unversioned_lock", "missing_lock", "different_exon",
    "different_effect", "named_objective", "reference_accession_objective",
])
def test_legacy_reference_interpretation_requires_consistent_structured_evidence(change):
    goal = structural_goal(None)
    if change == "different_lock":
        goal["transcript_lock"]["value"] = "TX-OTHER.3"
    elif change == "unversioned_lock":
        goal["transcript_lock"]["value"] = "TX-REFERENCE"
    elif change == "missing_lock":
        goal.pop("transcript_lock")
    elif change == "different_exon":
        goal["splice_isoform_objectives"][0]["exon_label"] = 5
    elif change == "different_effect":
        goal["splice_isoform_objectives"][0].update(
            desired_effect="inclusion", desired_abundance="increase"
        )
    elif change == "named_objective":
        goal["splice_isoform_objectives"][0]["isoform_label"] = "named isoform"
    elif change == "reference_accession_objective":
        goal["splice_isoform_objectives"][0]["reference_accession"] = "TX-REFERENCE.3"
    assert "context.structured_goal.splice_isoform_objectives" in (
        pipeline_tools._canonical_design_goal_contract_issues(goal)
    )


def test_structural_route_does_not_discharge_separate_transcript_abundance_target():
    goal = structural_goal()
    abundance = deepcopy(goal["evaluation_targets"][0])
    abundance.update(target_type="transcript", label="Named transcript abundance")
    goal["evaluation_targets"].append(abundance)
    assert pipeline_tools._canonical_design_goal_contract_issues(goal) == [
        "context.structured_goal.splice_isoform_objectives"
    ]
    goal["splice_isoform_objectives"].append({
        "objective_scope": "specific_transcript", "transcript_id": "TX-REFERENCE.3",
        "desired_abundance": "decrease",
    })
    assert pipeline_tools._canonical_design_goal_contract_issues(goal) == []


def test_exon_target_direction_is_checked_even_with_reference_transcript():
    goal = structural_goal("exon")
    goal["evaluation_targets"][0]["desired_delta"] = "increase"
    assert pipeline_tools._canonical_design_goal_contract_issues(goal) == [
        "context.structured_goal.evaluation_targets[0].desired_delta"
    ]


def test_transcript_abundance_can_oppose_exon_effect_with_exon_context():
    goal = structural_goal("transcript")
    goal["design_programs"][0]["desired_delta"] = "increase"
    goal["evaluation_targets"][0]["desired_delta"] = "increase"
    goal["splice_isoform_objectives"] = [{
        "objective_scope": "specific_transcript", "transcript_id": "TX-REFERENCE.3",
        "exon_label": 4, "desired_effect": "skipping", "desired_abundance": "increase",
    }]
    assert pipeline_tools._canonical_design_goal_contract_issues(goal) == []
