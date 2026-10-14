from __future__ import annotations

from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from aso_agents.snapgene_ingestion import load_snapgene_exons


def _write_genbank(path: Path, features: list[SeqFeature]) -> None:
    record = SeqRecord(Seq("ACGT" * 200), id="test", name="test", description="test")
    record.annotations["molecule_type"] = "DNA"
    record.features = features
    SeqIO.write(record, str(path), "genbank")


def test_load_snapgene_exons_accepts_snapgene_labeled_genbank_group(tmp_path: Path) -> None:
    path = tmp_path / "reference.gb"
    _write_genbank(
        path,
        [
            SeqFeature(
                FeatureLocation(10, 40, strand=1),
                type="misc_feature",
                qualifiers={"label": ["ENST000001 exon 1"], "note": ["Isoform: ENST000001; exon 1"]},
            ),
            SeqFeature(
                FeatureLocation(100, 130, strand=1),
                type="misc_feature",
                qualifiers={
                    "label": ["SnapGene isoform (construct.dna) exon 1"],
                    "note": ["Isoform: SnapGene isoform (construct.dna); exon 1"],
                },
            ),
            SeqFeature(
                FeatureLocation(200, 236, strand=1),
                type="misc_feature",
                qualifiers={
                    "label": ["SnapGene isoform (construct.dna) exon 2"],
                    "note": ["Isoform: SnapGene isoform (construct.dna); exon 2"],
                },
            ),
        ],
    )

    exons = load_snapgene_exons(path)

    assert exons["exon_name"].tolist() == [
        "SnapGene isoform (construct.dna) exon 1",
        "SnapGene isoform (construct.dna) exon 2",
    ]
    assert exons["length"].tolist() == [30, 36]
    assert exons["local_start"].tolist() == [100, 200]


def test_load_snapgene_exons_rejects_ambiguous_genbank_groups(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.gb"
    _write_genbank(
        path,
        [
            SeqFeature(
                FeatureLocation(10, 40, strand=1),
                type="misc_feature",
                qualifiers={"label": ["ENST000001 exon 1"], "note": ["Isoform: ENST000001; exon 1"]},
            ),
            SeqFeature(
                FeatureLocation(100, 130, strand=1),
                type="misc_feature",
                qualifiers={"label": ["ENST000002 exon 1"], "note": ["Isoform: ENST000002; exon 1"]},
            ),
        ],
    )

    with pytest.raises(ValueError, match="Multiple exon feature groups"):
        load_snapgene_exons(path)
