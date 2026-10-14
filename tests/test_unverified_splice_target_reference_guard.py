"""Unproven coordinate proposals cannot redirect a locked exon request."""
from __future__ import annotations

from copy import deepcopy

import run_goal_intake as runtime


REFERENCE = {
    "transcript_id": "ENSTTEST",
    "chrom": "chr17",
    "strand": 1,
    "exons_tx": [
        {"index": 1, "start": 1000, "end": 1099, "seq_region_name": "chr17"},
        {"index": 2, "start": 2000, "end": 2099, "seq_region_name": "chr17"},
    ],
}


def test_unverified_off_transcript_span_uses_annotated_requested_exon() -> None:
    target = {"chrom": "chr17", "start": 100, "end": 190,
              "strand": "+", "requested_exon_label": 2}
    assert runtime._repair_unverified_explicit_target_outside_transcript(target, REFERENCE)
    assert (target["chrom"], target["start"], target["end"]) == ("chr17", 2000, 2099)
    assert target["unverified_proposed_interval"]["start"] == 100
    assert target["coordinate_repair"] == "unverified_interval_replaced_by_locked_annotated_exon"


def test_unverified_other_exon_cannot_redirect_locked_exon_request() -> None:
    target = {"chrom": "chr17", "start": 1020, "end": 1050,
              "requested_exon_label": 2}
    assert runtime._repair_unverified_explicit_target_outside_transcript(target, REFERENCE)
    assert (target["chrom"], target["start"], target["end"]) == ("chr17", 2000, 2099)
    assert target["unverified_proposed_interval"]["start"] == 1020


def test_user_supplied_other_exon_coordinate_remains_for_clarification() -> None:
    target = {"chrom": "chr17", "start": 1020, "end": 1050,
              "requested_exon_label": 2, "source": "request_text"}
    original = deepcopy(target)
    assert not runtime._repair_unverified_explicit_target_outside_transcript(target, REFERENCE)
    assert target == original


def test_declared_evaluation_span_remains_authoritative() -> None:
    target = {"chrom": "chr17", "start": 1020, "end": 1050,
              "requested_exon_label": 2, "source": "evaluation_targets"}
    original = deepcopy(target)
    assert not runtime._repair_unverified_explicit_target_outside_transcript(target, REFERENCE)
    assert target == original


def test_trusted_user_coordinate_is_preserved_for_clarification() -> None:
    target = {"chrom": "chr17", "start": 100, "end": 190,
              "source": "request_text", "requested_exon_label": 2}
    original = deepcopy(target)
    assert not runtime._repair_unverified_explicit_target_outside_transcript(target, REFERENCE)
    assert target == original


def test_unverified_nearby_regulatory_span_is_not_rebased() -> None:
    target = {"chrom": "chr17", "start": 1980, "end": 2010,
              "requested_exon_label": 2}
    original = deepcopy(target)
    assert not runtime._repair_unverified_explicit_target_outside_transcript(target, REFERENCE)
    assert target == original


def test_ambiguous_exon_is_not_guessed() -> None:
    target = {"chrom": "chr17", "start": 100, "end": 190,
              "requested_exon_label": 3}
    original = deepcopy(target)
    assert not runtime._repair_unverified_explicit_target_outside_transcript(target, REFERENCE)
    assert target == original
