"""Transcript-relative windows require their references before materialization."""
from copy import deepcopy

import pytest

import run_goal_intake as runtime
from aso_agents.region_projection import genomic_sources


def reference(strand):
    exons = [{"index": 1, "start": 101, "end": 130},
             {"index": 2, "start": 201, "end": 250}]
    if strand == -1:
        exons.reverse()
        for index, exon in enumerate(exons, 1):
            exon["index"] = index
    return {"id": "TX_WINDOW", "version": 7, "versioned_id": "TX_WINDOW.7",
            "species": "homo_sapiens", "assembly_name": "GRCh38",
            "gene_id": "GENE_WINDOW", "seq_region_name": "chr7",
            "strand": strand, "exons_tx": exons}


def goal(modality="gapmer", region=None):
    constraint = {"constraint_id": "window", "source_type": "explicit_rows",
                  "coordinate_space": "cdna", "operator": "include",
                  "aso_overlap_requirement": "full_containment",
                  "intervals_bed0_half_open": [
                      {"chrom": "TX_WINDOW.7", "start": 20, "end": 60}]}
    program = {"program_id": "design", "modality": modality,
               "transcript_id": "TX_WINDOW.7", "region_constraints": [constraint]}
    if region is not None:
        program["target_region"] = region
    return {"gene_id": "GENE_WINDOW", "species": "human",
            "intent": "knockdown" if modality == "gapmer" else "binding",
            "design_programs": [program]}


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("modality,region", [("gapmer", None), ("steric", None),
                                            ("gapmer", "all")])
def test_declared_window_loads_reference_before_validation_and_projects_both_strands(
        tmp_path, monkeypatch, strand, modality, region):
    g = goal(modality, region)
    declarations = deepcopy(g["design_programs"])
    calls = []

    def fetch(identity):
        calls.append(identity)
        return reference(strand)

    monkeypatch.setattr(runtime, "_fetch_transcript_structure", fetch)
    final = {}
    runtime._materialize_region_constraints(g, final, str(tmp_path))
    result = g["materialized_region_constraints"]["by_program"]["design"][0]
    assert result["materialization_summary"]["status"] == "materialized"
    assert result["materialization_summary"]["rejected_rows"] == []
    assert result["materialized_cdna_rows"][0][:3] == ["TX_WINDOW.7", 20, 60]
    projected = genomic_sources(g, result)[0]["genomic_rows"]
    expected = [["chr7", 120, 130], ["chr7", 200, 230]] if strand == 1 else [
        ["chr7", 200, 230], ["chr7", 120, 130]]
    assert [row[:3] for row in projected] == expected
    assert {row[5] for row in projected} == ({"+"} if strand == 1 else {"-"})
    assert sum(row[2] - row[1] for row in projected) == 40
    assert g["design_programs"] == declarations
    assert final["isoform_transcripts"] == g["isoform_transcripts"]
    runtime._materialize_region_constraints(g, final, str(tmp_path))
    assert calls == ["TX_WINDOW.7"]


@pytest.mark.parametrize("fault", ["version", "gene", "species", "missing_structure"])
def test_invalid_declared_reference_cannot_publish_empty_success_or_use_a_default(
        tmp_path, monkeypatch, fault):
    g = goal()
    before = deepcopy(g)
    record = reference(1)
    if fault == "version":
        record.update(version=8, versioned_id="TX_WINDOW.8")
    elif fault == "gene":
        record["gene_id"] = "OTHER_GENE"
    elif fault == "species":
        record["species"] = "mus_musculus"
    else:
        record["exons_tx"] = []
    monkeypatch.setattr(runtime, "_fetch_transcript_structure", lambda _: record)
    final = {}
    with pytest.raises(ValueError, match="declared_target_reference"):
        runtime._materialize_region_constraints(g, final, str(tmp_path))
    assert "materialized_region_constraints" not in g
    assert "isoform_transcripts" not in g
    assert g["design_programs"] == before["design_programs"]
    assert final == {}


def test_reference_resolution_does_not_relax_out_of_bounds_window(tmp_path, monkeypatch):
    g = goal()
    g["design_programs"][0]["region_constraints"][0]["intervals_bed0_half_open"][0]["end"] = 81
    monkeypatch.setattr(runtime, "_fetch_transcript_structure", lambda _: reference(1))
    runtime._materialize_region_constraints(g, {}, str(tmp_path))
    result = g["materialized_region_constraints"]["by_program"]["design"][0]
    assert result["materialized_rows"] == []
    assert result["materialization_summary"]["status"] == "unresolved"


def test_genomic_only_constraint_does_not_request_a_transcript(tmp_path, monkeypatch):
    g = {"region_constraints": [{"source_type": "explicit_rows", "coordinate_space": "genomic",
          "intervals_bed0_half_open": [{"chrom": "chr7", "start": 20, "end": 60}]}]}
    monkeypatch.setattr(runtime, "_fetch_transcript_structure", lambda _: pytest.fail("Unrequested reference"))
    runtime._materialize_region_constraints(g, {}, str(tmp_path))
    result = g["materialized_region_constraints"]["global"][0]
    assert result["materialized_rows"][0][:3] == ["chr7", 20, 60]
    assert "isoform_transcripts" not in g
