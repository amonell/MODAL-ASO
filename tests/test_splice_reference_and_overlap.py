from copy import deepcopy
import importlib
import json
from types import SimpleNamespace

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools
chooser = importlib.import_module("aso_agents.transcript_chooser_agent")
from aso_agents.region_overlap import interval_matches


def test_nullable_goal_repair_reports_the_actual_field_error():
    schema = {"type": "object", "properties": {"structured_goal": {"anyOf": [
        {"type": "null"}, {"type": "object", "additionalProperties": False,
         "properties": {"region": {"enum": ["contains_feature"]}}}
    ]}}}
    invalid = {"structured_goal": {"region": "full_containment", "invented": "x" * 10000}}
    errors = pipeline_tools._json_schema_validation_errors(invalid, schema)
    assert errors and any("structured_goal.region" in e for e in errors)
    assert any("invented" in e for e in errors)
    assert not any("null" in e or "x" * 100 in e for e in errors)
    assert not pipeline_tools._json_schema_validation_errors({"structured_goal": None}, schema)
    assert not pipeline_tools._json_schema_validation_errors(
        {"structured_goal": {"region": "contains_feature"}}, schema)


@pytest.fixture
def local_reference(monkeypatch, tmp_path):
    path = tmp_path / "annotation.gtf"
    path.write_text(
        'chr1\tsource\tgene\t1\t1000\t.\t+\t.\tgene_id "ENSG00000000001.3"; gene_name "TEST"; gene_type "protein_coding";\n'
        'chr1\tsource\ttranscript\t1\t1000\t.\t+\t.\tgene_id "ENSG00000000001.3"; gene_name "TEST"; transcript_id "ENST00000000001.2"; transcript_type "protein_coding";\n'
        'chr1\tsource\texon\t100\t300\t.\t+\t.\tgene_id "ENSG00000000001.3"; gene_name "TEST"; transcript_id "ENST00000000001.2"; exon_id "ENSE00000000001.1";\n'
    )
    cfg = SimpleNamespace(gtf_path=path, ensembl_species="homo_sapiens", ensembl_assembly="GRCh38", ucsc_build="hg38")
    monkeypatch.setenv("ASO_DEPENDENCY_MODE", "frozen")
    monkeypatch.setattr(chooser, "get_species_config", lambda _: cfg)
    monkeypatch.setattr(chooser, "list_supported_species", lambda: ["human"])
    monkeypatch.setattr(chooser.requests, "get", lambda *a, **k: pytest.fail("Unexpected network call"))
    for name in ("_ENSEMBL_ID_LOOKUP_CACHE", "_LOCAL_GTF_GENE_SYMBOL_CACHE",
                 "_LOCAL_GTF_STABLE_ID_GENE_CACHE", "_LOCAL_GTF_GENE_PAYLOAD_CACHE",
                 "_LOCAL_GTF_ID_PAYLOAD_CACHE"):
        monkeypatch.setattr(chooser, name, {})
    return path


def test_exact_version_survives_lookup_and_native_metadata(local_reference):
    structure = chooser.get_transcript_structure("ENST00000000001.2")
    assert structure["id"] == "ENST00000000001"
    assert structure["versioned_id"] == "ENST00000000001.2"
    assert structure["version"] == 2
    assert structure["reference_annotation"] == str(local_reference)
    listing = chooser.list_transcripts("TEST", gene_id="ENSG00000000001")
    assert listing["transcripts"][0]["versioned_id"] == "ENST00000000001.2"
    assert runtime._find_transcript_by_id(listing["transcripts"], "ENST00000000001.9") is None


def test_missing_exact_version_is_not_silently_substituted_even_after_cache(local_reference):
    chooser.get_transcript_structure("ENST00000000001.2")
    with pytest.raises(chooser.FrozenDependencyUnavailableError):
        chooser.get_transcript_structure("ENST00000000001.9")
    assert chooser.get_transcript_structure("ENST00000000001")["version"] == 2


@pytest.mark.parametrize('gene_id', [None, 'ENSG00000000001', 'ENSG00000000001.3'])
def test_bulk_structure_uses_same_installed_annotation_as_single_lookup(local_reference, gene_id):
    bulk = chooser.get_gene_transcript_structures('TEST', gene_id=gene_id)
    structure = bulk['structures']['ENST00000000001']
    single = chooser.get_transcript_structure('ENST00000000001.2')
    assert structure == single
    assert structure['reference_source'] == 'local_gtf'
    assert structure['reference_annotation'] == str(local_reference)
    assert bulk['transcripts'][0]['versioned_id'] == 'ENST00000000001.2'


def test_symbol_repeated_in_gene_id_slot_resolves_local_reference(local_reference):
    bulk = chooser.get_gene_transcript_structures('TEST', gene_id='TEST')
    assert bulk['gene_id'] == 'ENSG00000000001'
    assert bulk['transcripts'][0]['versioned_id'] == 'ENST00000000001.2'
    listed = chooser.local_reference_gene_transcripts(
        'human', gene_id='TEST', gene_symbol='TEST')
    assert len(listed) == 1
    assert listed[0]['versioned_id'] == 'ENST00000000001.2'


def test_bulk_local_lookup_does_not_attest_a_wrong_gene_version(local_reference, monkeypatch):
    calls = []
    monkeypatch.setattr(chooser, '_lookup_id', lambda identifier, **kw: calls.append(identifier))
    monkeypatch.setattr(chooser, '_lookup_gene_by_symbol', lambda *a, **kw: None)
    result = chooser.get_gene_transcript_structures('TEST', gene_id='ENSG00000000001.8')
    assert result['status'] == 'not_found'
    assert calls == ['ENSG00000000001.8']


def test_bulk_remote_fallback_keeps_its_actual_provenance(monkeypatch):
    monkeypatch.setattr(chooser, '_lookup_local_gtf_id', lambda *a: None)
    gene = {'id': 'GENE', 'display_name': 'NEUTRAL', 'species': 'homo_sapiens', 'assembly_name': 'GRCh38',
            'Transcript': [{'id': 'TX', 'version': 7, 'strand': 1, 'seq_region_name': '1',
                            'Exon': [{'start': 100, 'end': 129}]}]}
    monkeypatch.setattr(chooser, '_lookup_id', lambda *a, **kw: deepcopy(gene))
    result = chooser.get_gene_transcript_structures('NEUTRAL', gene_id='GENE')
    assert result['structures']['TX']['versioned_id'] == 'TX.7'
    assert 'reference_source' not in result['structures']['TX']


@pytest.mark.parametrize("strand", ["+", "-"])
def test_verified_aliases_and_partial_coding_exon_keep_both_requested_junctions(strand):
    base = "ENST00000000001"
    exons = [{"index": 1, "start": 1, "end": 50},
             {"index": 2, "start": 100, "end": 200, "contains_coding": True},
             {"index": 3, "start": 300, "end": 400}]
    isoforms = {base: {"transcript_id": base, "chrom": "chr1", "strand": strand,
                      "metadata": {"id": base, "versioned_id": base + ".2", "version": 2},
                      "exons_tx": exons,
                      "sites": [{"exon_index": 2, "chrom": "chr1", "strand": strand,
                                 "acceptor": 100 if strand == "+" else 200,
                                 "donor": 200 if strand == "+" else 100,
                                 "coding": True, "acceptor_plan_ok": False, "donor_plan_ok": True}]}}
    goal = {"exon_label": 2, "desired_effect": "skipping", "focus_exons_only": True}
    objectives = [{"isoform_label": base + ".2", "desired_abundance": "decrease"}]
    original = deepcopy(objectives)
    plan, _ = runtime._compute_splice_site_plan(objectives, isoforms, goal)
    assert {s["site"] for s in plan} == {"acceptor", "donor"}
    assert all(s["absent_in"] == [] and s["delta_direction"] == "decrease" for s in plan)
    assert objectives == original
    wrong, _ = runtime._compute_splice_site_plan(
        [{"isoform_label": base + ".9", "desired_abundance": "decrease"}], isoforms, goal)
    assert all(base + ".9" in s["absent_in"] for s in wrong)
    evidence = pipeline_tools._compact_resolved_target_reference_evidence(
        {"isoform_transcripts": isoforms}, [{"resolved_target": {"transcript_id": base}}])
    assert evidence["transcript_features"][0]["reference_identity"]["versioned_id"] == base + ".2"


@pytest.mark.parametrize("strand,site", [("+", "acceptor"), ("+", "donor"), ("-", "acceptor"), ("-", "donor")])
def test_twenty_base_tiles_cover_junction_core_with_five_bases_on_each_side(tmp_path, strand, site):
    boundary = 1000 if (strand, site) in {("+", "acceptor"), ("-", "donor")} else 1100
    isoforms = {"ENSTTEST": {"transcript_id": "ENSTTEST", "chrom": "chr1", "strand": strand,
                            "exons_tx": [{"index": 2, "start": 1001, "end": 1100}]}}
    constraint = {"constraint_id": "junction", "source_type": "splice_relative_window",
                  "aso_overlap_requirement": "contains_feature",
                  "splice_relative_window": {"exon_label": 2, "anchor_site": site,
                     "relative_start_nt": -5 if site == "acceptor" else -4,
                     "relative_end_nt": 4 if site == "acceptor" else 5,
                     "reference_transcript_id": "ENSTTEST"}}
    goal = {"isoform_transcripts": isoforms, "region_constraints": [constraint],
            "chemistry": {"length": 20}, "design_programs": [{"modality": "steric", "length": 20}]}
    runtime._materialize_region_constraints(goal, {}, str(tmp_path))
    materialized = goal["materialized_region_constraints"]["global"][0]["materialized_rows"]
    assert [(r[1], r[2]) for r in materialized] == [(boundary - 5, boundary + 5)]
    regions, meta = runtime._apply_region_constraints_to_rows(
        goal, [["chr1", 900, 1200, "context", 0, strand]], output_space="genomic")
    assert min(r[1] for r in regions) < boundary - 5 and max(r[2] for r in regions) > boundary + 5
    rows = [{"Chrom": "chr1", "Start": str(start), "End": start + 20, "Sequence": "A" * 20,
             "Strand": strand, "test_start0": start} for start in range(boundary - 25, boundary + 6)]
    source, target = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps({"rows": rows}))
    result = pipeline_tools._filter_tiles_json_to_output(
        source_json_path=str(source), output_path=str(target), region_rows=meta["tile_filter_rows"],
        tile_overlap_requirement=meta["tile_overlap_requirement"],
        constraint_filters=meta["tile_constraint_filters"])
    assert result["row_count"] == 11
    selected = json.loads(target.read_text())["rows"]
    assert {r["test_start0"] for r in selected} == set(range(boundary - 15, boundary - 4))
    assert all(boundary-r["test_start0"] >= 5 and r["End"]-boundary >= 5 for r in selected)


def test_containment_direction_and_numeric_overlap_are_enforced():
    assert interval_matches(90, 110, 95, 105, "contains_feature")
    assert not interval_matches(90, 110, 95, 105, "full_containment")
    assert interval_matches(95, 105, 90, 110, "full_containment")
    assert not interval_matches(95, 105, 90, 110, "contains_feature")
    assert not interval_matches(85, 105, 100, 120, "any_overlap", 6)
    assert interval_matches(85, 105, 100, 120, "any_overlap", 5)


def test_native_bed_and_projected_genomic_coordinates_use_their_declared_conventions():
    native = {"Chrom": "chr1", "Start": "100", "End": "120", "Sequence": "ACGT" * 5}
    projected = {"Genomic_chrom": "chr1", "Genomic_start": 101, "Genomic_end": 120}
    for row in (native, projected):
        assert set(pipeline_tools._tile_row_intervals(row)) == {("chr1", 100, 120), ("1", 100, 120)}
        assert runtime._tile_row_overlap_intervals(row) == [("chr1", 100, 120)]
    assert ("chr1", 0, 20) in pipeline_tools._tile_row_intervals({"Chrom": "chr1", "Start": 0, "End": 20})


@pytest.fixture
def reference_lock_goal(tmp_path, local_reference, monkeypatch):
    tx = chooser.get_transcript_structure("ENST00000000001.2")
    monkeypatch.setattr(runtime, "_cached_authoritative_exon_nomenclature_mapping",
                        lambda *args: {"status": "ordinal_default", "source": "transcript_exon_index"})
    goal = {
        "isoform_transcripts": {tx["id"]: {**tx, "transcript_id": tx["id"], "chrom": "chr1"}},
        "region_constraints": [{
            "constraint_id": "junction", "source_type": "splice_relative_window",
            "splice_relative_window": {
                "reference_transcript_id": "ENST00000000001.2", "exon_label": 1,
                "anchor_site": "acceptor", "relative_start_nt": -5, "relative_end_nt": 4,
            },
        }],
    }
    runtime._materialize_region_constraints(goal, {}, str(tmp_path))
    return goal


def test_reference_lock_proof_comes_from_annotation(reference_lock_goal, local_reference):
    record, = pipeline_tools._compact_resolved_target_materialization_records(reference_lock_goal)
    resolved = record["resolved_target"]
    assert "exact_reference_identity" in resolved["evidence_capabilities"]
    assert resolved["reference_identity_evidence"] == {
        "source_type": "local_gtf", "reference_annotation": str(local_reference),
        "source_transcript_id": "ENST00000000001.2", "exon_number": 1,
        "chrom": "chr1", "strand": "+", "exon_start_1based": 100,
        "exon_end_1based_inclusive": 300, "anchor_site": "acceptor",
        "anchor_position_1based": 100,
    }
    finding = {"record_index": 0, "identity_requirement": "Preserve the exact reference transcript and exon.",
               "requirement_kind": "reference_identity_lock", "status": "satisfied",
               "evidence_assessment": "The native annotation verifies the exact version, exon and anchor."}
    payload = {"status": "consistent", "record_findings": [finding],
               "contradiction_record_indexes": [], "question": None, "reason": None}
    assert not pipeline_tools._resolved_target_consistency_contract_errors(
        payload, record_count=1, schema=pipeline_tools._resolved_target_consistency_schema(1), records=[record])
    # The new capability must not certify a separately named feature/exclusion.
    for kind in ("spatial_exclusion", "substitution_exclusion", "nomenclature_distinction",
                 "alternate_feature_exclusion", "isoform_distinction"):
        finding["requirement_kind"] = kind
        assert pipeline_tools._resolved_target_finding_capability_errors([finding], [record], positive_status="satisfied")


def test_verifier_schema_exposes_its_mandatory_reason_before_generation():
    schema = pipeline_tools._resolved_target_consistency_verification_schema(0)
    payload = {"status": "verified_consistent", "record_findings": [],
               "affected_record_indexes": [], "reason": None}
    assert pipeline_tools._json_schema_validation_errors(payload, schema)
    payload["reason"] = ""
    assert pipeline_tools._json_schema_validation_errors(payload, schema)
    payload["reason"] = "No materialized targets require verification."
    assert not pipeline_tools._resolved_target_consistency_verification_contract_errors(
        payload, record_count=0, schema=schema)


@pytest.mark.parametrize("fault", ["requested_version", "resolved_version", "provenance", "exon",
                                   "anchor", "chromosome", "strand", "ambiguous_exon"])
def test_reference_lock_requires_matching_source_proof(reference_lock_goal, fault):
    goal = reference_lock_goal
    entry = next(iter(goal["isoform_transcripts"].values()))
    region = goal["materialized_region_constraints"]["global"][0]
    if fault == "requested_version":
        region["splice_relative_window"]["reference_transcript_id"] = "ENST00000000001.9"
    elif fault == "resolved_version":
        region["materialization_summary"]["transcript_id"] = "ENST00000000001.9"
    elif fault == "provenance":
        entry.pop("reference_source")
    elif fault == "exon":
        region["materialization_summary"]["resolved_exon_label"] = 2
    elif fault == "anchor":
        region["materialization_summary"]["anchor_position"] += 1
    elif fault == "chromosome":
        region["materialized_rows"][0][0] = "chr2"
    elif fault == "strand":
        region["materialized_rows"][0][5] = "-"
    elif fault == "ambiguous_exon":
        entry["exons_tx"].append(deepcopy(entry["exons_tx"][0]))
    record, = pipeline_tools._compact_resolved_target_materialization_records(goal)
    assert "exact_reference_identity" not in record["resolved_target"]["evidence_capabilities"]
    finding = {"record_index": 0, "requirement_kind": "reference_identity_lock", "status": "satisfied"}
    assert pipeline_tools._resolved_target_finding_capability_errors([finding], [record], positive_status="satisfied")
