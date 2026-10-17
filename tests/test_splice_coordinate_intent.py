from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as p


@pytest.mark.parametrize("request_text,goal", [
    ("Design steric ASOs for two regulatory sites.", {}),
    ("Design ASOs at the supplied intervals.", {}),
    ("Block a regulatory element using steric ASOs.", {"intent": "binding"}),
    ("Knock down the transcript at its mature splice junction.", {"intent": "knockdown"}),
    ("Evaluate binding in this region.", {"design_programs": [{"intent": "binding"}]}),
])
def test_coordinates_cannot_create_an_unrequested_splice_objective(request_text, goal):
    state = {"request": request_text, "request_envelope": {"default_context": goal}}
    before = deepcopy(state)
    # A proposed splice interpretation in the assistant's question is not a user objective.
    result = p._explicit_splice_target_preseed_from_clarifications(state, [{
        "question": "Should this be a splice-switching target at chr8:500-600?",
        "answer": "Use GRCh38 chr8:700-750 for the requested regulatory site.",
    }])
    assert result == {} and state == before


def test_confirmed_splice_coordinates_retain_original_intent_and_evidence():
    state = {"request": "Design splice-switching ASOs to promote exon skipping.",
             "request_envelope": {"default_context": {"intent": "splice_modulation"}}}
    result = p._explicit_splice_target_preseed_from_clarifications(state, [{
        "question": "The tentative feature was chr8:500-600. What are your coordinates?",
        "answer": "Use chr8:700-750, strand -, for exon 9x; decrease exon inclusion.",
        "evidence_refs": ["patient_annotation.json"],
    }])
    target = result["structured_goal_overlay"]["explicit_splice_targets"]
    assert len(target) == 1
    assert (target[0]["start"], target[0]["end"], target[0]["strand"]) == (700, 750, "-")
    assert target[0]["requested_exon_label"] == "9x"
    assert target[0]["desired_delta"] == "decrease"
    assert target[0]["evidence_refs"] == ["patient_annotation.json"]


def test_question_only_coordinates_are_not_promoted_to_user_input():
    state = {"request": "Design splice-switching ASOs to promote exon skipping."}
    assert p._explicit_splice_target_preseed_from_clarifications(state, [{
        "question": "Is chr8:500-600 your exon?", "answer": "I do not know."
    }]) == {}


@pytest.mark.parametrize("text,expected", [
    ("strand -", "-"), ("strand -, use the stated interval", "-"),
    ("strand +.", "+"), ("strand: reverse", "-"),
    ("strand=forward", "+"), ("chr8:700-750 (-)", "-"),
    ("strand unknown", ""),
])
def test_symbolic_strands_do_not_require_an_impossible_word_boundary(text, expected):
    assert p._strand_from_clarification_text(text) == expected
