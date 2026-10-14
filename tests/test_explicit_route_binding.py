from copy import deepcopy

import pytest

from aso_agents import pipeline_tools


def _fixture(strand="+"):
    model = "TX_A::explicit_target_model"
    target = {"chrom": "chr2", "start": 1200, "end": 1280, "strand": strand,
              "requested_exon_label": "feature-x", "transcript_id": "TX_A",
              "isoform_label": model, "base_isoform_label": "TX_A",
              "desired_delta": "decrease"}
    goal = {
        "explicit_splice_targets": [target],
        "isoform_transcripts": {
            "TX_A": {"transcript_id": "TX_A"},
            model: {"transcript_id": model, "metadata": {
                "source": "explicit_splice_target_modified",
                "explicit_target_modified": True, "base_transcript_id": "TX_A",
                "base_isoform_label": "TX_A"}},
        },
        "splice_site_plan": [{
            "chrom": "chr2", "position": position, "site": site, "strand": strand,
            "explicit_start": 1200, "explicit_end": 1280,
            "source": "explicit_splice_target", "present_in": [model],
            "absent_in": ["TX_A"], "delta_per_isoform": {model: "decrease"},
            "delta_direction": "decrease", "covers_all_isoforms": False,
        } for site, position in (("acceptor", 1200 if strand == "+" else 1280),
                                 ("donor", 1280 if strand == "+" else 1200))],
    }
    program = {"program_id": "feature_route", "modality": "steric",
               "intent": "splice_modulation", "transcript_id": "TX_A",
               "exon_label": "feature-x", "desired_delta": "decrease",
               "desired_effect": "skipping", "site_kind": "both"}
    return goal, program


@pytest.mark.parametrize("strand", ["+", "-"])
def test_named_route_uses_attested_hypothetical_feature_without_reference_relabeling(strand):
    goal, program = _fixture(strand)
    original = deepcopy(goal)
    derived, metadata = pipeline_tools._derive_exploratory_steric_programs(
        structured_goal=goal, base_program=program)
    assert len(derived) == 1
    assert {r["position"] for r in derived[0]["anchor_site_records"]} == {1200, 1280}
    assert derived[0]["transcript_id"] == "TX_A"
    assert derived[0]["exon_label"] == "feature-x"
    binding = derived[0]["explicit_target_model_bindings"][0]
    assert binding["model_identity"] == "TX_A::explicit_target_model"
    assert binding["reference_annotation_verified"] is False
    assert metadata["explicit_target_model_bindings"] == [binding]
    assert goal == original


@pytest.mark.parametrize("fault", ["opposite_direction", "other_feature", "other_transcript",
                                  "unattested_model", "wrong_coordinates", "wrong_strand"])
def test_hypothetical_feature_binding_does_not_bypass_route_scope(fault):
    goal, program = _fixture()
    if fault == "opposite_direction":
        program["desired_delta"] = "increase"
    elif fault == "other_feature":
        program["exon_label"] = "feature-y"
    elif fault == "other_transcript":
        program["transcript_id"] = "TX_B"
    elif fault == "unattested_model":
        goal["isoform_transcripts"]["TX_A::explicit_target_model"]["metadata"] = {}
    elif fault == "wrong_coordinates":
        goal["explicit_splice_targets"][0]["start"] = 1000
    elif fault == "wrong_strand":
        goal["explicit_splice_targets"][0]["strand"] = "-"
    derived, metadata = pipeline_tools._derive_exploratory_steric_programs(
        structured_goal=goal, base_program=program)
    assert derived == []
    assert metadata["status"] == "unresolved_named_route_scope"
