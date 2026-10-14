"""The selected design job binds output coordinates to an attested transcript."""
from aso_agents.design_delivery import _resolved_transcript_site


def test_verbal_lock_uses_selected_job_reference_version():
    goal = {
        "transcript_lock": {"kind": "verbal_rule", "value": "canonical transcript"},
        "isoform_transcripts": {"ENST1": {"transcript_id": "ENST1", "chrom": "chr1", "strand": "+",
            "metadata": {"versioned_id": "ENST1.7"},
            "exons_tx": [{"seq_region_name": "chr1", "start": 100, "end": 199,
                          "cdna_start": 1, "index": 1, "coding_start_genomic": 100,
                          "coding_end_genomic": 199}]}},
        "gapmer_tiling_results_by_job": {"job-a": {"transcript_id": "ENST1"}},
    }
    row = {"job_id": "job-a", "binding_intervals_bed0_half_open": [["chr1", 119, 140]]}
    site = _resolved_transcript_site(goal, row)
    assert site == {"transcript_accession": "ENST1.7", "exon_ordinal": 1,
                    "cdna_start": 21, "cdna_end": 41, "complete_coding_exon_site": True}


def test_explicit_transcript_lock_is_not_replaced_by_job_choice():
    goal = {"transcript_lock": {"kind": "transcript_id", "value": "ENST2.4"},
            "isoform_transcripts": {"ENST1": {"transcript_id": "ENST1", "metadata": {"versioned_id": "ENST1.7"}}},
            "gapmer_tiling_results_by_job": {"job-a": {"transcript_id": "ENST1"}}}
    site = _resolved_transcript_site(goal, {"job_id": "job-a", "chrom": "chr1", "start": 100, "end": 119})
    assert site["transcript_accession"] == "ENST2.4"
    assert site["cdna_start"] is None


def test_explicit_version_conflict_does_not_claim_other_reference_coordinates():
    goal = {"transcript_lock": {"kind": "explicit_id", "value": "ENST1.4"},
            "isoform_transcripts": {"ENST1": {"transcript_id": "ENST1",
                                             "metadata": {"versioned_id": "ENST1.7"},
                                             "chrom": "chr1", "strand": "+",
                                             "exons_tx": [{"seq_region_name": "chr1", "start": 100,
                                                           "end": 199, "cdna_start": 1, "index": 1}]}},
            "gapmer_tiling_results_by_job": {"job-a": {"transcript_id": "ENST1"}}}
    site = _resolved_transcript_site(goal, {"job_id": "job-a", "chrom": "chr1", "start": 120, "end": 140})
    assert site["transcript_accession"] == "ENST1.4"
    assert site["cdna_start"] is None
