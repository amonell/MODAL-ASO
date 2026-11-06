"""The ordinary catalog and review path must retain verified patient references."""
from copy import deepcopy
import hashlib
import importlib

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pytest

import run_goal_intake as native
from aso_agents import pipeline_tools as pipeline
from aso_agents.declared_reference_catalog import complete_target_references
from aso_agents.supplied_reference_role import artifact_sha256
from aso_agents.supplied_reference_identity import verified_supplied_receipt, canonical_supplied_paths


@pytest.fixture
def supplied_case(tmp_path, monkeypatch):
    def build(strand="+", variant=False):
        accession, physical = "NM_123456.7", "ENST00000000001.2"
        baseline = "ACGT" * 125
        sequence = baseline[:109] + "A" + baseline[110:] if variant else baseline
        path = tmp_path / "patient.gb"
        record = SeqRecord(Seq(sequence), id="patient", annotations={"molecule_type": "DNA"})
        SeqIO.write(record, path, "genbank")
        fasta = tmp_path / "reference.fa"
        fasta.write_text(">chr7\n" + baseline + "\n")
        intervals = [(101, 150), (301, 400)]
        if strand == "-":
            intervals.reverse()
        exons = []
        for index, (start, end) in enumerate(intervals, 1):
            seq = sequence[start - 1:end]
            exons.append({"order": index, "local_start": start - 1, "local_end": end,
                          "feature_strand": 1 if strand == "+" else -1,
                          "sequence": seq if strand == "+" else str(Seq(seq).reverse_complement()),
                          "chrom": "chr7", "strand": strand, "genomic_start": start, "genomic_end": end})
        annotation = {"snapgene_path": str(path), "input_file_sha256": artifact_sha256(path),
                      "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                      "full_sequence": sequence, "reference_fasta_path": str(fasta), "exons": exons,
                      "declared_reference": accession,
                      "patient_variants": [{"chrom": "chr7", "pos": 110, "ref": "C", "alt": "A"}] if variant else []}
        entry = native._build_snapgene_transcript_entry(annotation, accession)
        source = {"id": physical.split(".")[0], "versioned_id": physical, "version": 2,
                  "species": "homo_sapiens", "seq_region_name": "chr7", "strand": 1 if strand == "+" else -1,
                  "reference_source": "local_gtf", "reference_annotation": str(tmp_path / "native.gtf"),
                  "exons_tx": [{"index": i, "start": a, "end": b} for i, (a, b) in enumerate(intervals, 1)]}
        chooser = importlib.import_module("aso_agents.transcript_chooser_agent")
        monkeypatch.setattr(chooser, "resolve_external_transcript_accession", lambda *a, **k: {
            "transcript_id": physical, "requested_accession": accession, "method": "local_gencode_refseq_xref"})
        monkeypatch.setattr(native, "_fetch_transcript_structure", lambda *a: deepcopy(source))
        goal = {"species": "human", "transcript_policy": "user_locked",
                "transcript_lock": {"kind": "reference_accession", "value": accession},
                "isoform_transcripts": {accession: entry}, "snapgene_annotation": annotation}
        return goal, source, path, accession
    return build


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("variant", [False, True])
def test_supplied_reference_is_verified_and_carried_into_semantic_review(supplied_case, strand, variant):
    goal, _, path, accession = supplied_case(strand, variant)
    original = deepcopy(goal["isoform_transcripts"][accession])
    assert complete_target_references(native, goal)
    entry = goal["isoform_transcripts"][accession]
    assert entry["transcript_id"] == original["transcript_id"]
    assert entry["exons_tx"] == original["exons_tx"]
    receipt = verified_supplied_receipt(entry, accession)
    proof = receipt["supplied_reference_identity"]
    assert proof["matched_exon_count"] == 2
    assert len(proof["exonic_substitutions"]) == int(variant)
    exon = entry["exons_tx"][1]
    anchor = exon["start"] if strand == "+" else exon["end"]
    record = {"requested_target": {"reference_transcript_id": accession, "exon_label": 2, "anchor_site": "acceptor"},
              "resolved_target": {"transcript_id": entry["transcript_id"], "requested_exon_label": 2,
                  "resolved_exon_label": 2, "status": "materialized", "anchor_site": "acceptor",
                  "anchor_position": anchor, "materialized_rows": [["chr7", anchor-10, anchor+10, "test", 0, strand]]}}
    evidence = pipeline._resolved_target_reference_identity_evidence(record, goal["isoform_transcripts"])
    assert evidence["source_type"] == "supplied_reference"
    assert evidence["accession_resolution"] == receipt
    compact = pipeline._compact_resolved_target_reference_evidence(goal, [record])
    assert compact["transcript_features"][0]["reference_identity"]["supplied_reference_accessions"] == [receipt]
    before = deepcopy(goal)
    complete_target_references(native, goal)
    assert goal == before
    path.write_text(path.read_text() + "\n")
    assert pipeline._resolved_target_reference_identity_evidence(record, goal["isoform_transcripts"]) is None
    with pytest.raises(ValueError, match="file_changed"):
        complete_target_references(native, goal)


@pytest.mark.parametrize("fault", ["exon", "order", "strand", "chromosome", "version", "sequence", "unrecorded_variant"])
def test_supplied_reference_conflicts_never_publish_a_false_accession(supplied_case, fault):
    goal, source, _, accession = supplied_case(variant=fault == "unrecorded_variant")
    if fault == "exon":
        source["exons_tx"][0]["end"] += 1
    elif fault == "order":
        source["exons_tx"].reverse()
    elif fault == "strand":
        source["strand"] = -1
    elif fault == "chromosome":
        source["seq_region_name"] = "chr8"
    elif fault == "version":
        source.update(version=3, versioned_id="ENST00000000001.3")
    elif fault == "sequence":
        goal["snapgene_annotation"]["exons"][0]["sequence"] = "A" * 50
    else:
        goal["snapgene_annotation"]["patient_variants"] = []
    before, final = deepcopy(goal), {"unchanged": True}
    with pytest.raises(ValueError, match="declared_target_reference"):
        complete_target_references(native, goal, final)
    assert goal == before and final == {"unchanged": True}


def test_identical_copies_are_one_input_but_different_bytes_are_distinct(tmp_path):
    first, second, different = (tmp_path / name for name in ("original.gb", "copy.gb", "different.gb"))
    first.write_text("annotation one"); second.write_bytes(first.read_bytes()); different.write_text("annotation two")
    goal = {"splice_isoform_objectives": [{"artifact_ref_path": str(second)}]}
    assert canonical_supplied_paths([str(first), str(second), str(different)], goal) == [str(first), str(different)]
    assert goal["splice_isoform_objectives"][0]["artifact_ref_path"] == str(first)
    assert goal["supplied_reference_path_aliases"][str(second)] == str(first)


@pytest.mark.parametrize("fault", ["ordinal", "interval", "strand", "identity", "accession"])
def test_receipt_cannot_be_reused_for_a_changed_target(supplied_case, fault):
    goal, _, _, accession = supplied_case()
    complete_target_references(native, goal)
    entry = goal["isoform_transcripts"][accession]
    if fault == "ordinal":
        entry["exons_tx"][0]["index"] = 9
    elif fault == "interval":
        entry["exons_tx"][0]["end"] += 1
    elif fault == "strand":
        entry["strand"] = "-"
    elif fault == "identity":
        entry["transcript_id"] = "SNAPGENE::different"
    else:
        accession = "NM_123456.8"
    assert verified_supplied_receipt(entry, accession) is None


def test_single_file_uses_locked_reference_without_an_isoform_abundance_label(tmp_path, monkeypatch):
    path, copy = tmp_path / "patient.gb", tmp_path / "copy.gb"
    path.write_text("annotation"); copy.write_bytes(path.read_bytes())
    goal = {"species": "human", "transcript_policy": "user_locked",
            "transcript_lock": {"kind": "reference_accession", "value": "NM_123456.7"},
            "artifact_refs": [{"path": str(path)}],
            "splice_isoform_objectives": [{"objective_scope": "default_exon_context", "desired_abundance": "decrease"}]}
    calls = []
    def ingest(p, **kwargs):
        calls.append(p)
        return {"snapgene_path": p, "variant_derivation_attempted": True}
    monkeypatch.setattr(native, "ingest_snapgene_file", ingest)
    monkeypatch.setattr(native, "_build_snapgene_transcript_entry", lambda a, l: {"transcript_id": "SNAPGENE::patient"})
    monkeypatch.setattr(native, "_select_snapgene_isoform_labels_with_llm", lambda *a: pytest.fail("Typed binding needs no guess"))
    result = native._prepare_snapgene_objectives(goal, {}, user_request=f"Use {path}; original copy is {copy}.")
    assert calls == [str(path)]
    assert result["snapgene_labels"] == ["NM_123456.7"]
    assert goal["snapgene_annotation"]["declared_reference"] == "NM_123456.7"
