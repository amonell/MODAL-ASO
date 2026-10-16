"""Named feature coordinates come from the locked annotation, not model offsets."""
from copy import deepcopy

import jsonschema
import pytest
import run_goal_intake as runtime
from aso_agents.goal_contract import _REGION_CONSTRAINT_SCHEMA
from aso_agents.transcript_features import feature_interval, materialize_feature
from aso_agents.transcript_metadata import resolved_transcript_metadata
from aso_agents.transcript_chooser_agent import _transcript_structure_from_payload


def reference(strand="+", split=False):
    exons = [{"index": 1, "start": 101, "end": 110},
             {"index": 2, "start": 201, "end": 240}]
    if strand == "-":
        exons.reverse()
    for i, exon in enumerate(exons, 1):
        exon["index"] = i
    start = (39 if strand == "-" else 9) if split else 5
    offset = 0
    pieces = []
    for exon in exons:
        length = exon["end"]-exon["start"]+1
        a, b = max(start, offset), min(45, offset+length)
        if a < b:
            lo, hi = ((exon["start"]+a-offset, exon["start"]+b-offset-1)
                      if strand == "+" else
                      (exon["end"]-(b-offset)+1, exon["end"]-(a-offset)))
            exon.update(contains_coding=True, coding_start_genomic=lo, coding_end_genomic=hi)
        a, b = max(start, offset), min(start+3, offset+length)
        if a < b:
            pieces.append([exon["start"]+a-offset, exon["start"]+b-offset-1] if strand == "+"
                          else [exon["end"]-(b-offset)+1, exon["end"]-(a-offset)])
        offset += length
    return {"transcript_id": "TXA.3", "chrom": "chr8", "strand": strand,
            "exons_tx": exons, "metadata": {"versioned_id": "TXA.3",
            "start_codon_genomic_intervals": pieces, "annotation_tags": ["basic"]}}


def constraint(**kwargs):
    return {"source_type": "transcript_feature", "feature_name": "start_codon",
            "reference_transcript_id": "TXA.3", "aso_overlap_requirement": "contains_feature",
            "operator": "intersect", **kwargs}


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("split", [False, True])
def test_source_annotation_projects_exact_codon_and_cds_without_mutation(strand, split):
    entry = reference(strand, split)
    before = deepcopy(entry)
    start = (39 if strand == "-" else 9) if split else 5
    assert feature_interval(entry, "start_codon") == (start, start+3)
    assert feature_interval(entry, "cds") == (start, 45)
    result = materialize_feature(constraint(), {"anchor": entry}, "anchor")
    assert result["materialized_rows"][0][0:3] == ["TXA.3", start, start+3]
    assert sum(row[2]-row[1] for row in result["materialized_genomic_rows"]) == 3
    assert result["materialization_summary"]["reference_transcript_id"] == "TXA.3"
    assert entry == before


@pytest.mark.parametrize("fault", [
    "missing_cds", "coding_outside_exon", "short_cds", "discontinuous_cds",
    "wrong_strand", "wrong_order", "incomplete_start", "missing_start_annotation",
    "contradictory_start_annotation", "invalid_start_annotation",
])
def test_missing_or_conflicting_annotation_fails_explicitly(fault):
    entry = reference()
    if fault == "missing_cds":
        for exon in entry["exons_tx"]:
            for key in ("contains_coding", "coding_start_genomic", "coding_end_genomic"):
                exon.pop(key, None)
    elif fault == "coding_outside_exon":
        entry["exons_tx"][0]["coding_start_genomic"] = 99
    elif fault == "short_cds":
        entry["exons_tx"][0].update(coding_start_genomic=106, coding_end_genomic=107)
        for key in ("contains_coding", "coding_start_genomic", "coding_end_genomic"):
            entry["exons_tx"][1].pop(key, None)
    elif fault == "discontinuous_cds":
        entry["exons_tx"][1]["coding_start_genomic"] += 1
    elif fault == "wrong_strand":
        entry["strand"] = "."
    elif fault == "wrong_order":
        entry["exons_tx"].reverse()
    elif fault == "incomplete_start":
        entry["metadata"]["annotation_tags"].append("cds_start_NF")
    elif fault == "missing_start_annotation":
        entry["metadata"]["start_codon_genomic_intervals"] = []
    elif fault == "contradictory_start_annotation":
        entry["metadata"]["start_codon_genomic_intervals"] = [[107, 109]]
    else:
        entry["metadata"]["start_codon_genomic_intervals"] = [[106, 110]]
    with pytest.raises(ValueError, match="transcript_feature"):
        materialize_feature(constraint(), {"anchor": entry}, "anchor")


@pytest.mark.parametrize("change", [
    {"reference_transcript_id": "TXA.4"},
    {"transcript_id": "UNKNOWN.1"},
    {"feature_name": "invented_feature"},
    {"rows": [["TXA.3", 0, 3]]},
])
def test_no_fallback_for_wrong_reference_or_conflicting_contract(change):
    with pytest.raises(ValueError, match="transcript_feature"):
        materialize_feature(constraint(**change), {"anchor": reference()}, "anchor")


def test_contract_normalization_materialization_and_reporting_agree(tmp_path):
    c = constraint()
    jsonschema.validate(c, _REGION_CONSTRAINT_SCHEMA)
    normalized = runtime._normalize_region_constraints([c])[0]
    assert normalized["reference_transcript_id"] == "TXA.3"
    assert normalized["feature_name"] == "start_codon"
    catalog = {"anchor": reference("-", split=True)}
    result = runtime._materialize_region_constraint_entry(
        structured_goal={}, final_payload={}, run_dir=str(tmp_path), constraint=normalized,
        isoform_transcripts=catalog, iso_label_primary="anchor")
    goal = {"isoform_transcripts": catalog, "materialized_region_constraints": {"global": [result]}}
    for space, key in [("cdna", "materialized_cdna_rows"), ("genomic", "materialized_genomic_rows")]:
        rows = runtime._constraint_rows_in_output_space(
            structured_goal=goal, constraint=result, output_space=space, base_rows=[],
            isoform_transcripts=catalog, iso_label_primary="anchor")
        assert rows == result[key]
        reported = runtime._target_feature_interval_index(goal, output_space=space)
        assert sorted(x for v in reported.values() for x in v) == sorted((r[0], r[1], r[2]) for r in rows)


def test_user_supplied_absolute_interval_is_not_reinterpreted_as_an_annotation(tmp_path):
    c = {"source_type": "explicit_rows", "coordinate_space": "cdna",
         "rows": [["TXA.3", 0, 3, "start codon", 0, "+"]],
         "aso_overlap_requirement": "contains_feature"}
    normalized = runtime._normalize_region_constraints([c])[0]
    assert normalized["source_type"] == "explicit_rows"
    assert normalized["rows"][0][1:3] == [0, 3]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(constraint(rows=[["TXA.3", 0, 3]]), _REGION_CONSTRAINT_SCHEMA)


def test_reference_structure_and_metadata_preserve_start_annotation_and_incomplete_flag():
    payload = {"id": "TXA", "version": 3, "strand": 1, "seq_region_name": "8",
               "Exon": [{"start": 101, "end": 140}],
               "Translation": {"start": 106, "end": 135},
               "annotation_tags": ["cds_start_NF"],
               "start_codon_genomic_intervals": []}
    structure = _transcript_structure_from_payload(payload, "TXA.3")
    assert structure["annotation_tags"] == ["cds_start_NF"]
    meta = resolved_transcript_metadata(structure, {"annotation_tags": ["stale"]})
    assert meta["annotation_tags"] == ["cds_start_NF"]
    assert meta["start_codon_genomic_intervals"] == []
    entry = {"transcript_id": "TXA.3", "chrom": "chr8", "strand": "+",
             "exons_tx": structure["exons_tx"], "metadata": meta}
    with pytest.raises(ValueError, match="coding_start_incomplete"):
        feature_interval(entry, "start_codon")



def test_local_gtf_start_codon_segments_and_tags_survive_loading(tmp_path, monkeypatch):
    import importlib
    chooser = importlib.import_module("aso_agents.transcript_chooser_agent")
    from types import SimpleNamespace
    path = tmp_path/"reference.gtf"
    attrs = 'gene_id "GENE.1"; transcript_id "TXG.2"; tag "cds_start_NF";'
    records = [("transcript", 101, 240), ("exon", 101, 110), ("exon", 201, 240),
               ("CDS", 110, 110), ("CDS", 201, 230),
               ("start_codon", 110, 110), ("start_codon", 201, 202)]
    path.write_text("\n".join(f"chr8\ttest\t{feature}\t{a}\t{b}\t.\t+\t.\t{attrs}"
                              for feature, a, b in records)+"\n")
    monkeypatch.setattr(chooser, "_local_gtf_cache_identity", lambda species:(str(path), path.stat().st_mtime))
    for name in ("_LOCAL_GTF_GENE_PAYLOAD_CACHE", "_LOCAL_GTF_ID_PAYLOAD_CACHE", "_ENSEMBL_ID_LOOKUP_CACHE"):
        monkeypatch.setattr(chooser, name, {})
    monkeypatch.setattr(chooser, "get_species_config", lambda species:SimpleNamespace(
        ensembl_species="homo_sapiens", ensembl_assembly="TEST"))
    payload = chooser._local_gtf_gene_payload({"gene_id":"GENE"}, "human")["Transcript"][0]
    assert payload["start_codon_genomic_intervals"] == [[110,110],[201,202]]
    assert payload["annotation_tags"] == ["cds_start_NF"]
    structure = chooser._transcript_structure_from_payload(payload, "TXG.2")
    metadata = resolved_transcript_metadata(structure)
    assert metadata["start_codon_genomic_intervals"] == [[110,110],[201,202]]
    assert metadata["annotation_tags"] == ["cds_start_NF"]


def test_feature_reference_is_loaded_even_without_a_program_transcript_list(monkeypatch):
    from aso_agents.declared_reference_catalog import complete_target_references
    goal = {"species":"human", "gene_id":"GENE", "isoform_transcripts": {},
            "design_programs":[{"program_id":"p", "region_constraints":[constraint()]}]}
    seen = []
    def fetch(reference_id):
        seen.append(reference_id)
        return {"id":"TXA", "version":3, "versioned_id":"TXA.3", "gene_id":"GENE",
                "species":"homo_sapiens", "seq_region_name":"chr8", "strand":1,
                "exons_tx":reference()["exons_tx"]}
    monkeypatch.setattr(runtime, "_fetch_transcript_structure", fetch)
    assert complete_target_references(runtime, goal)
    assert seen == ["TXA.3"] and "TXA.3" in goal["isoform_transcripts"]


def test_cdna_interval_reporting_preserves_namespace_and_version():
    row = {"Chrom":"TXA.3", "Start":7, "End":26, "coordinate_system":"1based_inclusive"}
    assert runtime._tile_row_overlap_intervals(row, output_space="cdna") == [("TXA.3",6,26)]
    other = {"Chrom":"TXA.4", "Start":7, "End":26, "coordinate_system":"1based_inclusive"}
    assert runtime._overlap_interval_bp(
        runtime._tile_row_overlap_intervals(row, output_space="cdna"),
        runtime._tile_row_overlap_intervals(other, output_space="cdna")) == 0


