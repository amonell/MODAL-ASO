"""Search intervals must not become reconstructed exon boundaries."""
from copy import deepcopy

import pytest

import run_goal_intake as native


def goal_for(target_type, strand):
    exons = [
        {"start": 100, "end": 199},
        {"start": 400, "end": 499},
        {"start": 800, "end": 899},
    ]
    if strand == "-":
        exons.reverse()
    for index, exon in enumerate(exons, 1):
        exon.update(index=index, seq_region_name="chr8", contains_coding=True)
    target = {
        "target_type": target_type, "objective_type": "identity_anchor",
        "transcript_id": "TX_A.1", "chrom": "chr8", "strand": strand,
        "exon_label": 2, "start": 480, "end": 515,
    }
    return {
        "intent": "splice_modulation", "desired_effect": "skipping",
        "evaluation_targets": [target],
        "isoform_transcripts": {"TX_A.1": {
            "transcript_id": "TX_A.1", "chrom": "chr8", "strand": strand,
            "exons_tx": exons,
            "metadata": {"id": "TX_A.1", "seq_region_name": "chr8"},
        }},
    }


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("target_type", [
    "splice_site_window", "splice_junction", "splice_acceptor", "splice_donor",
    "regulatory_region", "region", "transcript", "sequence",
])
def test_typed_nonexon_interval_preserves_reference_geometry(target_type, strand):
    goal = goal_for(target_type, strand)
    original = deepcopy(goal)

    assert native._ensure_explicit_splice_target_context(goal, {}) == {}

    assert goal == original
    assert "explicit_splice_targets" not in goal


@pytest.mark.parametrize("strand", ["+", "-"])
def test_declared_complete_novel_exon_still_builds_separate_model(strand):
    goal = goal_for("exon", strand)
    goal["evaluation_targets"][0].update(
        start=550, end=589, exon_label="2b", objective_type="primary_outcome",
        desired_delta="decrease",
    )
    original_exons = deepcopy(goal["isoform_transcripts"]["TX_A.1"]["exons_tx"])

    result = native._ensure_explicit_splice_target_context(goal, {})

    assert result["target_count"] == 1
    assert goal["isoform_transcripts"]["TX_A.1"]["exons_tx"] == original_exons
    model = goal["isoform_transcripts"]["TX_A.1::explicit_target_model"]
    assert [(e["start"], e["end"]) for e in model["exons_tx"]
            if e.get("source") == "explicit_splice_target"] == [(550, 589)]
    assert len(model["exons_tx"]) == len(original_exons) + 1


@pytest.mark.parametrize("strand", ["+", "-"])
def test_complete_annotated_exon_uses_original_reference(strand):
    goal = goal_for("exon", strand)
    goal["evaluation_targets"][0].update(start=400, end=499)
    original_exons = deepcopy(goal["isoform_transcripts"]["TX_A.1"]["exons_tx"])

    result = native._ensure_explicit_splice_target_context(goal, {})

    assert result["target_count"] == 1
    assert set(goal["isoform_transcripts"]) == {"TX_A.1"}
    assert goal["isoform_transcripts"]["TX_A.1"]["exons_tx"] == original_exons
    assert goal["explicit_splice_targets"][0]["synthetic_exon_created"] is False


def test_legacy_untyped_explicit_span_remains_supported():
    goal = goal_for("exon", "+")
    target = goal["evaluation_targets"][0]
    target.pop("target_type")
    target.update(start=550, end=589, exon_label="2b")

    assert native._ensure_explicit_splice_target_context(goal, {})["target_count"] == 1
    assert "TX_A.1::explicit_target_model" in goal["isoform_transcripts"]
