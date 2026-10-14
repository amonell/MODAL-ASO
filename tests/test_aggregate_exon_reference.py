from copy import deepcopy

import pytest

import run_goal_intake as intake


def _goal():
    return {
        "gene_symbol": "GENE_A", "species": "human", "intent": "splice_modulation",
        "exon_label": 4, "desired_effect": "skipping", "focus_exons_only": True,
        "transcript_policy": "agent_choice",
        "splice_isoform_objectives": [{
            "objective_scope": "aggregate_class",
            "aggregate_class": {"basis": "exon_presence", "exon_label": 4, "presence": presence},
            "desired_abundance": abundance, "desired_effect": "skipping",
        } for presence, abundance in [("included", "decrease"), ("excluded", "increase")]],
    }


def _catalog(monkeypatch, *, annotated=True, exon_count=6):
    transcript_id = "ENST00000000001"
    monkeypatch.setattr(intake, "list_transcripts", lambda **_: {"transcripts": [
        {"id": "ENST00000000002", "length": 999999, "biotype": "protein_coding"},
        {"id": transcript_id, "is_canonical": annotated, "biotype": "protein_coding"},
    ]})
    monkeypatch.setattr(intake, "_fetch_transcript_structure", lambda observed_id: {
        "id": observed_id, "seq_region_name": "7", "strand": 1,
        "exons_tx": [{"index": i, "start": i * 100, "end": i * 100 + 49,
                      "contains_coding": True} for i in range(1, exon_count + 1)],
    })
    monkeypatch.setattr(intake, "_generate_splice_strategy_analysis", lambda *_, **__: None)
    def no_class_matching(*_, **__):
        raise AssertionError("An aggregate biological class must not be matched to a representative transcript.")
    monkeypatch.setattr(intake, "auto_match_isoform_transcript", no_class_matching)
    return transcript_id


@pytest.mark.parametrize("policy", ["agent_choice", "canonical_transcript", None, ""])
def test_exon_reference_preserves_aggregate_objectives(policy, monkeypatch):
    transcript_id = _catalog(monkeypatch)
    goal = _goal(); original_objectives = deepcopy(goal["splice_isoform_objectives"])
    if policy is None:
        goal.pop("transcript_policy")
    else:
        goal["transcript_policy"] = policy
    payload = deepcopy(goal)
    result = intake._resolve_isoform_transcripts(goal, payload, allow_interactive=False)
    assert set(result["isoform_transcripts"]) == {transcript_id}
    assert result["isoform_transcripts"][transcript_id]["match_info"]["method"] == "agent_selected_exon_reference"
    assert goal["splice_isoform_objectives"] == original_objectives
    assert not (set(goal["aggregate_isoform_classes"]) & set(result["isoform_transcripts"]))
    assert payload["isoform_resolution_notes"]["aggregate_exon_reference"]["class_membership_resolved"] is False
    assert result["splice_site_plan"]
    assert {entry["exon_indices"][transcript_id] for entry in result["splice_site_plan"]} == {4}
    assert not goal.get("pending_isoform_transcript_confirmations")


@pytest.mark.parametrize("missing", ["annotation", "exon", "delegation", "exon_class"])
def test_unjustified_reference_selection_stays_blocked(missing, monkeypatch):
    _catalog(monkeypatch, annotated=missing != "annotation", exon_count=2 if missing == "exon" else 6)
    goal = _goal()
    if missing == "delegation":
        goal["transcript_policy"] = "user_locked"
    if missing == "exon_class":
        for objective in goal["splice_isoform_objectives"]:
            objective["aggregate_class"] = {"basis": "role", "role": "pathogenic"}
    with pytest.raises(RuntimeError, match="isoform_resolution_failed"):
        intake._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    assert not goal.get("isoform_transcripts")


@pytest.mark.parametrize("policy", [None, "agent_choice"])
def test_mixed_default_and_aggregate_objectives_share_a_structural_reference(policy, monkeypatch):
    transcript_id = _catalog(monkeypatch)
    goal = _goal()
    goal["splice_isoform_objectives"][0] = {
        "objective_scope": "default_exon_context", "exon_label": 4,
        "desired_effect": "skipping", "desired_abundance": "decrease",
    }
    if policy is None:
        goal.pop("transcript_policy")
    original = deepcopy(goal["splice_isoform_objectives"])
    result = intake._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    assert set(result["isoform_transcripts"]) == {transcript_id}
    assert goal["splice_isoform_objectives"] == original
    assert result["splice_site_plan"]
    assert {entry["exon_indices"][transcript_id] for entry in result["splice_site_plan"]} == {4}
    assert len(goal["aggregate_isoform_classes"]) == 1


@pytest.mark.parametrize("restriction", ["user_locked", "explicit_transcript", "unannotated_reference"])
def test_mixed_objectives_do_not_override_specific_or_unsupported_references(restriction, monkeypatch):
    _catalog(monkeypatch, annotated=restriction != "unannotated_reference")
    goal = _goal(); goal.pop("transcript_policy")
    goal["splice_isoform_objectives"][0] = {
        "objective_scope": "default_exon_context", "exon_label": 4,
        "desired_effect": "skipping", "desired_abundance": "decrease",
    }
    if restriction == "user_locked":
        goal["transcript_policy"] = "user_locked"
    if restriction == "explicit_transcript":
        goal["splice_isoform_objectives"][0]["transcript_id"] = "ENST00000000999"
    with pytest.raises(RuntimeError, match="isoform_resolution_failed"):
        intake._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    assert not goal.get("isoform_transcripts")
