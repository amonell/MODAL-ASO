from copy import deepcopy

from aso_agents.response_context import panel_reference_alignment_summary, evaluation_response_batches


def hit(transcript, organism, assembly, orientation="reverse_complement", context="mature_spliced_cdna", **extra):
    return {"transcript_id": transcript, "reference_identity": {
                "transcript_id": transcript + ".2", "species": organism, "assembly": assembly,
                "version_attested": True},
            "rna_context": context, "match_orientation": orientation, "strand": "+", **extra}


def test_panel_retains_reference_specific_exactness_without_copying_alignment_inventory():
    rows = [
        {"input_row_id": "alpha", "transcript_id": "primary",
         "mapping_hits": [hit("primary", "organism_a", "build_a"), hit("foreign", "organism_b", "build_b")],
         "near_match_candidates": [hit("foreign", "organism_b", "build_b", mismatch_count=0)]},
        {"input_row_id": "beta", "transcript_id": "primary",
         "mapping_hits": [hit("primary", "organism_a", "build_a")],
         "near_match_candidates": [hit("foreign", "organism_b", "build_b", mismatch_count=2)]},
    ]
    payload = {"task_type": "evaluate_existing_aso_set", "evaluation_summary": {"provided_asos": rows}}
    before = deepcopy(payload)
    batches = evaluation_response_batches(payload, target_context_bytes=1)
    assert payload == before
    for batch in batches:
        overview = batch["evaluation_summary"]["panel_comparison_overview"]
        assert all("mapping_hits" not in row and "near_match_candidates" not in row for row in overview)
        groups = [row["reference_alignment_summary"]["groups"] for row in overview]
        assert [g["species"] for g in groups[0]] == ["organism_a", "organism_b"]
        assert [g["exact_mapping_record_count"] for g in groups[0]] == [1, 1]
        assert [g["minimum_retained_mismatches"] for g in groups[0]] == [0, 0]
        assert [g["exact_mapping_record_count"] for g in groups[1]] == [1, 0]
        assert [g["minimum_retained_mismatches"] for g in groups[1]] == [0, 2]
        assert groups[0][1]["transcript_id"] == "foreign.2"
        assert groups[0][1]["assembly"] == "build_b"


def test_orientation_rna_context_build_and_unknown_evidence_stay_separate():
    rows = {"mapping_hits": [
        hit("tx", "organism_a", "build_a"),
        hit("tx", "organism_a", "build_a", orientation="input_sequence"),
        hit("tx", "organism_a", "build_a", context="unspliced_transcript_locus"),
        hit("tx", "organism_a", "build_b")],
        "near_match_candidates": [{"transcript_id": "unresolved", "mismatch_count": None},
            hit("tx", "organism_a", "build_a", mismatch_count=3,
                requested_site_overlaps=[{"site_id": "requested_boundary"}],
                reference_junctions_spanned=[{"reference_boundary": "junction_1"}])]}
    before = deepcopy(rows)
    groups = panel_reference_alignment_summary(rows)["groups"]
    assert rows == before and len(groups) == 5
    assert groups[0]["minimum_retained_mismatches"] == 0
    assert groups[0]["retained_near_match_record_count"] == 1
    assert groups[0]["recorded_requested_site_overlaps"] == [{"site_id": "requested_boundary"}]
    assert groups[0]["has_recorded_reference_junction"] is True
    assert groups[-1]["species"] is None and groups[-1]["assembly"] is None
    assert groups[-1]["minimum_retained_mismatches"] is None
    assert panel_reference_alignment_summary({})["groups"] == []
