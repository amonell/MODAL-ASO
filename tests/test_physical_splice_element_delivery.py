"""Physical target geometry must remain distinct from predicted response sites."""

import csv
from pathlib import Path

from aso_agents.binding_region_reports import physical_splice_element_overlaps
from aso_agents.design_delivery import write_design_panel


def test_splice_boundary_geometry_on_both_transcript_strands():
    plus = {"plus": {"transcript_id": "TX_PLUS", "metadata": {"versioned_id": "TX_PLUS.2"},
                     "strand": "+", "chrom": "chr1",
                     "exons_tx": [{"index": 1, "start": 100, "end": 110},
                                  {"index": 2, "start": 201, "end": 220}]}}
    minus = {"minus": {"transcript_id": "TX_MINUS", "strand": "-", "chrom": "chr1",
                       "exons_tx": [{"index": 1, "start": 201, "end": 220},
                                    {"index": 2, "start": 100, "end": 110}]}}
    plus_hit = physical_splice_element_overlaps([["chr1", 194, 210]], plus)
    assert len(plus_hit) == 1
    assert plus_hit[0]["transcript_id"] == "TX_PLUS.2"
    assert {k: plus_hit[0][k] for k in ("element", "exon_ordinal", "adjacent_intron_ordinal",
                                        "intronic_overlap_bp", "exonic_overlap_bp")} == {
        "element": "acceptor", "exon_ordinal": 2, "adjacent_intron_ordinal": 1,
        "intronic_overlap_bp": 6, "exonic_overlap_bp": 10,
    }
    minus_hit = physical_splice_element_overlaps([["1", 104, 115]], minus)
    assert len(minus_hit) == 1
    assert minus_hit[0]["element"] == "acceptor"
    assert minus_hit[0]["exon_ordinal"] == 2
    assert minus_hit[0]["intronic_overlap_bp"] == 5
    assert minus_hit[0]["exonic_overlap_bp"] == 6
    assert physical_splice_element_overlaps([["chr2", 194, 210]], plus) == []
    assert physical_splice_element_overlaps([["chr1", 201, 210]], plus) == []


def test_panel_exports_per_candidate_physical_element_explanation(tmp_path):
    catalog = {"target": {"transcript_id": "TX", "strand": "+", "chrom": "chr1",
                          "exons_tx": [{"index": 1, "start": 100, "end": 110},
                                       {"index": 2, "start": 201, "end": 220}]}}
    elements = physical_splice_element_overlaps([["chr1", 194, 210]], catalog)
    candidate = {"synthesized_antisense_sequence": "ACGTACGTACGTACGTACGT",
                 "target_binding_sequence": "ACGTACGTACGTACGTACGT",
                 "binding_intervals_bed0_half_open": [["chr1", 194, 210]],
                 "physical_splice_elements": elements,
                 "predicted_response_site": {"position": 220, "role": "scored_splice_response_site"}}
    result = write_design_panel({"rows": [candidate]}, {}, tmp_path)
    with Path(result["design_panel_csv"]).open() as stream:
        row = next(csv.DictReader(stream))
    explanation = row["physical_element_explanation"]
    assert "TX annotated exon 2 acceptor" in explanation
    assert "6 intronic and 10 exonic" in explanation
    assert "220" not in explanation
    markdown = Path(result["design_panel_md"]).read_text()
    assert explanation in markdown
    assert "not motif occupancy or measured activity" in markdown
