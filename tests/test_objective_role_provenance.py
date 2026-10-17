import pytest

import run_goal_intake as native
from aso_agents import pipeline_tools


@pytest.mark.parametrize("direction,effect", [("decrease", "skipping"), ("increase", "inclusion")])
def test_structural_objective_does_not_gain_a_disease_role(direction, effect):
    raw = {"splice_isoform_objectives": [{
        "objective_scope": "default_exon_context", "exon_label": 4,
        "desired_abundance": direction, "desired_effect": effect,
    }]}
    result = pipeline_tools._normalized_design_objective_preseed(raw)
    obj = result["splice_isoform_objectives"][0]
    assert obj["role"] == "context"
    assert obj["desired_abundance"] == direction
    assert obj["desired_effect"] == effect
    assert "role" not in raw["splice_isoform_objectives"][0]


@pytest.mark.parametrize("role", ["beneficial", "pathogenic", "context"])
@pytest.mark.parametrize("direction", ["increase", "decrease"])
def test_explicit_role_survives_independently_of_intervention_direction(role, direction):
    raw = {"splice_isoform_objectives": [{
        "objective_scope": "specific_transcript", "transcript_id": "TX_TEST",
        "desired_abundance": direction, "role": role,
    }]}
    obj = pipeline_tools._normalized_design_objective_preseed(raw)["splice_isoform_objectives"][0]
    assert obj["role"] == role
    assert obj["desired_abundance"] == direction


@pytest.mark.parametrize("effect", ["skipping", "inclusion"])
def test_synthetic_exon_pairs_have_direction_without_pathogenicity(effect):
    pair = native._default_splice_objective_pair("GENE_TEST", 4, effect)
    assert {obj["role"] for obj in pair} == {"context"}
    assert {obj["desired_abundance"] for obj in pair} == {"increase", "decrease"}
