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
    # Semantic intake owns the requested biological effect, including explicit
    # changes in an answer. This coordinate helper can only supply geometry.
    assert "desired_delta" not in target[0]
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


@pytest.mark.parametrize("direction", ["increase", "decrease"])
@pytest.mark.parametrize("wording", [
    "retain the original one-based values in the report",
    "include the coordinate table", "skip the long explanation",
    "increase the font size", "decrease the margin", "exclude the brackets",
])
def test_coordinate_answer_cannot_change_typed_biology(direction, wording):
    from aso_agents.explicit_target_identity import reconcile_clarified_targets
    original = {"chrom": "chr8", "start": 700, "end": 750, "transcript_id": "NM_123456.7",
                "desired_delta": direction, "source": "parsed"}
    state = {"request": "Design splice-switching ASOs.", "request_envelope": {
        "default_context": {"intent": "splice_modulation", "explicit_splice_targets": [original],
                            "transcript_lock": {"kind": "reference_accession", "value": "NM_123456.7"}}}}
    result = p._explicit_splice_target_preseed_from_clarifications(state, [{
        "question": "Confirm the coordinates.",
        "answer": f"Use chr8:700-750 for NM_123456.7; {wording}.",
    }])
    targets = result["structured_goal_overlay"]["explicit_splice_targets"]
    assert all("desired_delta" not in t for t in targets)
    reconciled = reconcile_clarified_targets([original, *targets])
    assert len(reconciled) == 1
    assert reconciled[0]["desired_delta"] == direction


@pytest.mark.parametrize("prose", ["retain the coordinates", "promote skipping", "increase inclusion", "lower confidence"])
def test_only_typed_fields_can_define_splice_direction(prose):
    assert p._infer_isoform_objective_desired_abundance({"label": prose, "notes": prose, "objective": prose}) == ""
    assert p._infer_isoform_objective_desired_abundance({"label": prose, "desired_delta": "decrease"}) == "decrease"
    assert p._infer_isoform_objective_desired_abundance({"notes": prose, "desired_effect": "inclusion"}) == "increase"
