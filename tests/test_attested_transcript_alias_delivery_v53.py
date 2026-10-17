"""A separately verified RefSeq xref can name the selected Ensembl structure."""

from copy import deepcopy

from aso_agents.design_delivery import _resolved_accession_and_record, _resolved_intronic_site


def _goal():
    lengths = [21, 20]
    return {
        "transcript_lock": {"kind": "explicit_id", "value": "NM_GENERIC"},
        "isoform_transcripts": {"ENST_GENERIC": {
            "transcript_id": "ENST_GENERIC", "chrom": "chr1", "strand": "+",
            "exons_tx": [{"index": 1, "start": 100, "end": 120},
                         {"index": 2, "start": 201, "end": 220}],
            "metadata": {
                "versioned_id": "ENST_GENERIC.4",
                "exon_nomenclature_resolutions": {"biological_exon_2": {
                    "status": "verified_alternate_mapping",
                    "refseq_accession": "NM_GENERIC.3",
                    "ensembl_transcript_id": "ENST_GENERIC",
                    "source_ensembl_transcript_id": "ENST_GENERIC.4",
                    "xref_sha256": "a" * 64, "source_sha256": "b" * 64,
                    "ensembl_exon_count": 2, "refseq_exon_count": 2,
                    "ensembl_exon_lengths": lengths,
                    "refseq_exon_lengths": lengths,
                }},
            },
        }},
    }


def _row():
    return {"binding_intervals_bed0_half_open": [["chr1", 129, 150]]}


def test_verified_refseq_alias_resolves_exact_versioned_structure():
    goal = _goal()
    accession, record = _resolved_accession_and_record(goal)
    assert accession == "ENST_GENERIC.4"
    assert record is goal["isoform_transcripts"]["ENST_GENERIC"]
    assert _resolved_intronic_site(goal, _row()) == {
        "transcript_accession": "ENST_GENERIC.4", "intron_ordinal": 1,
        "donor_distance_start": 10, "donor_distance_end": 30,
        "acceptor_distance_start": 51, "acceptor_distance_end": 71,
    }


def test_alias_does_not_override_explicit_version_or_conflicting_structure():
    for mutation in ("version", "xref", "lengths", "source", "status"):
        goal = _goal()
        resolution = goal["isoform_transcripts"]["ENST_GENERIC"]["metadata"]["exon_nomenclature_resolutions"]["biological_exon_2"]
        if mutation == "version":
            goal["transcript_lock"]["value"] = "NM_GENERIC.2"
        elif mutation == "xref":
            resolution.pop("xref_sha256")
        elif mutation == "lengths":
            resolution["refseq_exon_lengths"] = [20, 20]
        elif mutation == "source":
            resolution["source_ensembl_transcript_id"] = "ENST_OTHER.4"
        else:
            resolution["status"] = "unresolved"
        assert _resolved_accession_and_record(goal)[1] is None
        assert _resolved_intronic_site(goal, _row()) is None


def test_ambiguous_alias_is_not_selected():
    goal = _goal()
    duplicate = deepcopy(goal["isoform_transcripts"]["ENST_GENERIC"])
    duplicate["transcript_id"] = "ENST_OTHER"
    duplicate["metadata"]["versioned_id"] = "ENST_OTHER.1"
    resolution = duplicate["metadata"]["exon_nomenclature_resolutions"]["biological_exon_2"]
    resolution["ensembl_transcript_id"] = "ENST_OTHER"
    resolution["source_ensembl_transcript_id"] = "ENST_OTHER.1"
    goal["isoform_transcripts"]["ENST_OTHER"] = duplicate
    assert _resolved_accession_and_record(goal) == ("NM_GENERIC", None)
