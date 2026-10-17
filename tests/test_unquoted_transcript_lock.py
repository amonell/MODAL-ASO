"""A model-selected reference cannot masquerade as a user-locked accession."""

from copy import deepcopy

import run_goal_intake as intake


def _goal(value, source="user_request"):
    return {
        "gene_symbol": "GENE", "intent": "knockdown", "target_region": "cds",
        "transcript_policy": "user_locked",
        "transcript_lock": {"kind": "explicit_id", "value": value, "source": source},
    }


def test_unquoted_user_lock_is_removed_from_goal_and_envelope():
    request = "Design coding-exon gapmers for human GENE; choose a MANE reference."
    goal = _goal("ENST00000557018.5")
    envelope = {"structured_goal": deepcopy(goal), "transcript_lock": deepcopy(goal["transcript_lock"]),
                "transcript_policy": "user_locked", "tasks": [{
                    "task_id": "design", "task_type": "design_new_asos",
                    "context": {"structured_goal": deepcopy(goal),
                                "transcript_lock": deepcopy(goal["transcript_lock"]),
                                "transcript_policy": "user_locked"},
                }]}
    normalized = intake._normalize_intake_output({"structured_goal": goal,
                                                   "request_envelope": envelope}, request)
    assert "transcript_lock" not in normalized["structured_goal"]
    assert "transcript_lock" not in normalized["request_envelope"]
    assert "transcript_lock" not in normalized["request_envelope"]["tasks"][0]["context"]
    assert "transcript_lock" not in normalized["request_envelope"]["tasks"][0]["context"]["structured_goal"]


def test_versioned_user_lock_survives_unversioned_request_identity():
    request = "Design on transcript ENST00000380743 of human SMN2."
    goal = _goal("ENST00000380743.9")
    intake._apply_explicit_transcript_lock_from_request_text(goal, request)
    assert goal["transcript_lock"]["value"] == "ENST00000380743.9"
    assert goal["transcript_policy"] == "user_locked"


def test_reference_selected_transcript_is_not_treated_as_user_lock():
    goal = _goal("ENST00000557018.5", source="reference_selected")
    intake._apply_explicit_transcript_lock_from_request_text(goal, "Choose a MANE transcript.")
    assert goal["transcript_lock"]["value"] == "ENST00000557018.5"
