import csv
from pathlib import Path

import pytest

from aso_agents.design_delivery import translation_start_panel_evidence


@pytest.mark.parametrize(
    "strand,exon_start,exon_end,genomic_site",
    [(-1, 1000, 1099, "chr1:1078–1080 (- strand)"),
     (1, 1000, 1099, "chr1:1019–1021 (+ strand)")],
)
def test_translation_start_report_uses_attested_strand_and_selected_sites(
    tmp_path: Path, strand: int, exon_start: int, exon_end: int, genomic_site: str,
) -> None:
    panel = tmp_path / "design_panel.csv"
    with panel.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["transcript_accession", "transcript_cdna_start", "transcript_cdna_end"])
        writer.writeheader()
        writer.writerows([
            {"transcript_accession": "ENST000001.2", "transcript_cdna_start": 10, "transcript_cdna_end": 29},
            {"transcript_accession": "ENST000001.2", "transcript_cdna_start": 30, "transcript_cdna_end": 49},
            {"transcript_accession": "ENST000001.2", "transcript_cdna_start": 22, "transcript_cdna_end": 41},
        ])
    goal = {
        "target_region": "translation_start", "transcript_id": "ENST000001.2",
        "flanks": {"up": 10, "down": 25},
        "isoform_transcripts": {"ENST000001": {
            "transcript_id": "ENST000001", "chrom": "chr1", "strand": strand,
            "metadata": {"versioned_id": "ENST000001.2", "reference_annotation": "reference.gtf"},
            "exons_tx": [{"cdna_start": 1, "cdna_end": 100, "coding_start": 20,
                          "start": exon_start, "end": exon_end}],
        }},
    }
    text = translation_start_panel_evidence(goal, panel)
    assert "ENST000001.2" in text
    assert "cDNA 20–22" in text
    assert genomic_site in text
    assert "cDNA 10–47; 2/3" in text
    assert "2/3 windows overlap" in text
    assert "1 of 3 site pairs do not overlap" in text
    assert "not measured target occupancy" in text


def test_translation_start_report_refuses_mismatched_transcript(tmp_path: Path) -> None:
    panel = tmp_path / "design_panel.csv"
    panel.write_text("transcript_accession,transcript_cdna_start,transcript_cdna_end\nOTHER,1,20\n")
    goal = {"target_region": "translation_start", "transcript_id": "ENST000001.2",
            "isoform_transcripts": {"ENST000001": {"transcript_id": "ENST000001",
               "metadata": {"versioned_id": "ENST000001.2"},
               "exons_tx": [{"coding_start": 20, "cdna_start": 1, "cdna_end": 100}]}}}
    assert translation_start_panel_evidence(goal, panel) == ""
