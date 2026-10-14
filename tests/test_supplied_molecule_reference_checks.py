from __future__ import annotations

from aso_agents.mapping_evidence import (
    principal_reference_exon_membership,
    supplied_molecule_sequence_metrics,
)
from aso_agents.pipeline_tools import (
    _normalize_provided_asos,
    _provided_aso_reference_metrics_markdown,
)
from aso_agents.sequence_orientation import interpret_orientation, preferred_binding_hits


def test_oligo_role_does_not_depend_on_an_exact_reference_hit() -> None:
    row = _normalize_provided_asos({"provided_asos": [{
        "name": "candidate", "sequence": "ACATGCAGGATCTTGGTGAA",
        "chemistry": "full PS", "platform": "gapmer",
    }]})[0]
    assert row["declared_sequence_role"] == "antisense_oligo_sequence"
    orientation = interpret_orientation(row)
    assert orientation["provided_sequence_role"] == "antisense_oligo_sequence"
    assert orientation["original_molecule_exact_binding"] is False
    assert orientation["synthesized_antisense_sequence"] == row["sequence"]
    assert "binding uncertainty" in orientation["interpretation_note"]


def test_explicit_target_site_field_preserves_opposite_role() -> None:
    row = _normalize_provided_asos({"provided_asos": [{"target_sequence": "ACTGACTG"}]})[0]
    assert row["declared_sequence_role"] == "target_site_sequence"
    assert interpret_orientation(row)["provided_sequence_role"] == "target_site_sequence"
    hits = [
        {"match_orientation": "input_sequence", "matched_target_sequence": "ACTGACTG"},
        {"match_orientation": "reverse_complement", "matched_target_sequence": "CAGTCAGT"},
    ]
    assert preferred_binding_hits(row, hits) == hits[:1]


def test_sequence_field_takes_precedence_over_auxiliary_target_annotation() -> None:
    row = _normalize_provided_asos({"provided_asos": [{
        "sequence": "ACTGACTG", "target_sequence": "CAGTCAGT", "chemistry": "full PS",
    }]})[0]
    assert row["declared_sequence_role"] == "antisense_oligo_sequence"


def test_reference_exon_geometry_and_molecule_metric_are_reported() -> None:
    row = {
        "name": "candidate", "sequence": "AACCTTGG",
        "mapping_hits": [],
        "near_match_candidates": [{
            "transcript_id": "ENST00000000111", "match_orientation": "reverse_complement",
            "mapping_scope": "spliced_cdna", "mismatch_count": 1, "cdna_start": 121,
            "genomic_segments": [{"chrom": "chr1", "start": 201, "end": 208}],
            "reference_identity": {"transcript_id": "ENST00000000111.2",
                "sequence_evidence": {"exons_tx": [
                    {"start": 100, "end": 150}, {"start": 200, "end": 250},
                ]}},
        }],
    }
    row.update(supplied_molecule_sequence_metrics(row["sequence"]))
    row["principal_reference_exon_membership"] = principal_reference_exon_membership(
        row, "ENST00000000111.2"
    )
    assert row["maximum_homopolymer_run_nt"] == 2
    assert row["principal_reference_exon_membership"]["exons"][0]["transcript_exon_ordinal"] == 2
    assert row["principal_reference_exon_membership"]["fully_exonic"] is True
    markdown = _provided_aso_reference_metrics_markdown({"provided_asos": [row]})
    assert "maximum homopolymer run 2 nt" in markdown
    assert "1-mismatch" in markdown
    assert "exon ordinal(s) 2" in markdown
