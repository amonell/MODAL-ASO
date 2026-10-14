from aso_agents.provided_aso_ranking import (
    rank_scored_provided_rows,
    requested_antisense_mismatches,
)


TRANSCRIPT = "ENST00000000001.2"


def record(orientation, mismatches, transcript=TRANSCRIPT):
    return {
        "match_orientation": orientation,
        "mapping_scope": "spliced_cdna",
        "reference_identity": {"transcript_id": transcript},
        "transcript_id": transcript.split(".")[0],
        "mismatch_count": mismatches,
    }


def test_sense_exact_match_cannot_outrank_antisense_binding_evidence():
    originals = [
        {"mapping_hits": [record("input_sequence", 0)],
         "near_match_candidates": [record("input_sequence", 0), record("reverse_complement", 7)]},
        {"mapping_hits": [record("reverse_complement", 0)]},
        {"near_match_candidates": [record("reverse_complement", 2)]},
        {"near_match_candidates": [record("reverse_complement", 1)]},
    ]
    scored = [{"provided_row_index": index, "selection_rank": index + 1}
              for index in range(len(originals))]
    receipt = rank_scored_provided_rows(scored, originals, TRANSCRIPT)
    assert receipt["status"] == "complete"
    assert [row["provided_row_index"] for row in scored] == [1, 3, 2, 0]
    assert [row["selection_rank"] for row in scored] == [1, 2, 3, 4]
    assert [row["selection_score_components"]["requested_transcript_antisense_min_mismatches"]
            for row in scored] == [0, 1, 2, 7]


def test_requested_transcript_version_and_mature_context_are_binding():
    row = {"mapping_hits": [
        record("reverse_complement", 0, "ENST00000000001.1"),
        {**record("reverse_complement", 0), "mapping_scope": "pre_mrna_locus"},
    ], "near_match_candidates": [
        record("reverse_complement", 3, "ENST00000000001.1"),
        record("reverse_complement", 4),
    ]}
    assert requested_antisense_mismatches(row, TRANSCRIPT) == 4
    assert requested_antisense_mismatches(row, "ENST00000000002.2") is None


def test_absent_antisense_evidence_does_not_reorder_generic_proxy():
    scored = [{"provided_row_index": 0, "selection_rank": 2},
              {"provided_row_index": 1, "selection_rank": 1}]
    originals = [{"mapping_hits": [record("input_sequence", 0)]}, {}]
    before = [dict(item) for item in scored]
    result = rank_scored_provided_rows(scored, originals, TRANSCRIPT)
    assert result["status"] == "not_run"
    assert scored == before
