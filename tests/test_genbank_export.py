from __future__ import annotations

import json
from pathlib import Path

from Bio import SeqIO

from aso_agents.genbank_export import export_to_genbank


def test_export_to_genbank_uses_generated_gapmer_candidates(tmp_path: Path) -> None:
    fasta_path = tmp_path / "reference.fa"
    fasta_path.write_text(">chr1\n" + "ACGT" * 300 + "\n", encoding="utf-8")
    (tmp_path / "structured_goal.json").write_text(
        json.dumps(
            {
                "gene_symbol": "PCSK9",
                "species": "human",
                "patient_reference": {"fasta_path": str(fasta_path)},
                "all_transcripts_full": {
                    "ENST_TEST": {
                        "chrom": "1",
                        "strand": 1,
                        "exons_tx": [{"index": 1, "start": 301, "end": 500}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "artifacts" / "candidates"
    generated_dir.mkdir(parents=True)
    (generated_dir / "generated_aso_set.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sequence": "ACGTACGTACGTACGTACGT",
                        "length": 20,
                        "chrom": "chr1",
                        "start": 320,
                        "end": 339,
                        "strand": "+",
                        "site_label": "ENST_TEST_cdna20-39",
                        "source_modality": "gapmer",
                        "selection_rank": 1,
                        "off_target_gene_count": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "snapgene" / "aso_export.gb"

    result = export_to_genbank(tmp_path, output_path)

    assert result == output_path
    record = SeqIO.read(output_path, "genbank")
    aso_features = [
        feature
        for feature in record.features
        if any("rank 1" in label for label in feature.qualifiers.get("label", []))
    ]
    assert len(aso_features) == 1
    assert int(aso_features[0].location.start) == 319
    assert int(aso_features[0].location.end) == 339
    assert any("Off-target gene count: 0" in note for note in aso_features[0].qualifiers["note"])
