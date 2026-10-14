from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as p


TARGET = "ENST00000000111"
OTHER = "ENST00000000222"


def goal():
    return {"intent": "splice_modulation", "splice_isoform_objectives": [
        {"isoform_label": TARGET, "transcript_id": TARGET, "desired_abundance": "increase"}]}


@pytest.mark.parametrize("description", [
    "The included feature is absent from {other}.",
    "The exon is missing from {other}.",
    "It is not present in transcript {other}.",
    "Do not use {other}.",
    "Preserve {other}.",
])
def test_reference_mentions_cannot_replace_the_established_target(description):
    answer = "Use chr8:710-763, strand -, for exon 4. " + description.format(other=OTHER)
    state = {"request": "Design splice-switching ASOs for " + TARGET,
             "request_envelope": {"default_context": goal()}}
    original = deepcopy(state)
    result = p._explicit_splice_target_preseed_from_clarifications(
        state, [{"question": "Is " + OTHER + " the target?", "answer": answer}])
    overlay = result["structured_goal_overlay"]
    assert overlay["transcript_lock"]["value"] == TARGET
    assert overlay["explicit_splice_targets"][0]["transcript_id"] == TARGET
    assert overlay["evaluation_targets"][0]["strand"] == "-"
    assert state == original


@pytest.mark.parametrize("answer", [OTHER, "Use transcript " + OTHER + ".", "Transcript ID: " + OTHER])
def test_explicit_new_choices_are_honored_without_retaining_an_old_identifier_label(answer):
    assert p._clarification_preseed_transcript_id({}, [{"answer": answer}], goal()) == OTHER
    assert p._clarification_preseed_isoform_label(OTHER, goal()) == OTHER


def test_assistant_proposal_alone_cannot_choose_a_transcript():
    assert p._clarification_preseed_transcript_id({}, [{
        "question": "Use " + OTHER + "?", "answer": "The feature is chr8:710-763, strand -."}], goal()) == TARGET


@pytest.mark.parametrize("answer", [
    "Use " + TARGET + "; select " + OTHER + ".",
    "Use " + TARGET + " or " + OTHER + ".",
    "The database also mentions " + OTHER + ".",
])
def test_ambiguous_or_unclassified_prose_remains_with_semantic_intake(answer):
    assert p._clarification_preseed_transcript_id({}, [{"answer": answer}], goal()) == ""


def test_multiple_existing_objectives_do_not_choose_the_first_for_unscoped_coordinates():
    context = goal()
    context["splice_isoform_objectives"].append({"transcript_id": OTHER})
    assert p._clarification_preseed_transcript_id({}, [{"answer": "chr8:710-763, strand -."}], context) == ""


def test_positive_choice_after_a_negative_clause_keeps_the_requested_new_reference():
    answer = "Do not use " + TARGET + "; instead select " + OTHER + "."
    assert p._clarification_preseed_transcript_id({}, [{"answer": answer}], goal()) == OTHER


def test_repeating_a_declared_context_reference_preserves_the_existing_binding():
    assert p._clarification_preseed_transcript_id({}, [{
        "answer": "The feature is chr8:710-763, strand -, in " + TARGET + "."}], goal()) == TARGET


def test_explicit_native_lock_does_not_require_an_abundance_objective():
    context = {"transcript_lock": {"kind": "explicit_id", "value": TARGET}}
    assert p._clarification_preseed_transcript_id({}, [{"answer": "The feature is absent from " + OTHER + "."}], context) == TARGET


def test_changing_an_exact_version_does_not_leave_the_previous_version_as_its_label():
    context = {"splice_isoform_objectives": [{"isoform_label": TARGET + ".3"}]}
    assert p._clarification_preseed_isoform_label(TARGET + ".4", context) == TARGET + ".4"
