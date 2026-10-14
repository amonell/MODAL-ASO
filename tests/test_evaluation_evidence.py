from copy import deepcopy
from types import SimpleNamespace
import pytest

from aso_agents.evaluation_evidence import (
    reference_context, annotate_binding_evidence, attach_measurement_references,
    bind_assessment_measurements, exact_gene_count_scope, assessment_measurement_errors,
)


def test_reference_version_must_come_from_loaded_catalog(tmp_path):
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">chr1\nACGT\n")
    cfg = SimpleNamespace(genome_fasta=fasta, ensembl_assembly="test-assembly")
    genome = SimpleNamespace(filename=str(fasta))
    requested = "TX.7"
    result = reference_context(cfg, genome, "TX", {}, requested)
    assert result["reference_build"] == "test-assembly"
    assert result["requested_transcript_id"] == requested
    assert result["resolved_transcript_id"] == "TX"
    assert result["transcript_version_status"] == "unverified"
    assert result["rna_contexts_searched"] == []
    entry = {"metadata": {"versioned_id": "TX.7", "reference_annotation": "test.gtf"}}
    assert reference_context(cfg, genome, "TX", entry, requested)["transcript_version_status"] == "verified"
    assert reference_context(cfg, genome, "TX", entry, "TX.8")["transcript_version_status"] == "mismatch"
    assert reference_context(cfg, None, "TX", entry, requested)["reference_build"] is None


def test_near_match_overlap_keeps_orientation_and_separate_hypotheses():
    row = {"mapping_status": "unmapped", "near_match_candidates": [
        {"transcript_id": "TX", "match_orientation": orientation,
         "mismatch_count": 2, "genomic_span": {"chrom": "chr1", "start": 98, "end": 107}}
        for orientation in ("reverse_complement", "input_sequence")]}
    sites = [{"site_kind": "acceptor", "exon_index": 3,
              "acceptor": {"chrom": "1", "position": 102, "strand": "-"}}]
    annotate_binding_evidence(row, sites, "TX")
    evidence = row["binding_evidence"]
    assert evidence["antisense_near_match_hypotheses"] == 1
    assert len(evidence["antisense_near_matches_overlapping_requested_site"]) == 1
    assert evidence["antisense_near_matches_overlapping_requested_site"][0]["mismatch_count"] == 2
    assert row["mapping_status"] == "unmapped"
    assert "activity" not in evidence
    assert all(m["rna_context"] == "unspliced_transcript_locus_pre_mrna" for m in row["near_match_candidates"])


def test_discontiguous_mapping_does_not_claim_overlap_inside_intron():
    hit = {"transcript_id": "TX", "mapping_scope": "spliced_cdna", "match_orientation": "reverse_complement",
           "genomic_segments": [{"chrom": "1", "start": 1, "end": 4}, {"chrom": "1", "start": 12, "end": 15}],
           "genomic_span": {"chrom": "1", "start": 1, "end": 15}}
    row = {"mapping_hits": [hit]}
    annotate_binding_evidence(row, [{"site_kind": "donor", "donor": {"chrom": "1", "position": 8}}], "TX")
    assert hit["requested_site_overlaps"] == []
    assert hit["rna_context"] == "mature_spliced_cdna"


def test_numeric_binding_preserves_zero_and_uses_the_cited_molecule():
    summary = attach_measurement_references({"provided_asos": [
        {"name": "A", "splice_eval_summary": {"delta_deletion": 0.0}},
        {"name": "B", "splice_eval_summary": {"delta_deletion": -0.8123}}]})
    response = {"markdown": "", "evaluation_assessments": [
        {"name": "A", "mechanistic_plausibility": "delta_deletion={{M001_delta_deletion}}", "ranking_rationale": "Uncertain."},
        {"name": "B", "mechanistic_plausibility": "delta_deletion={{M002_delta_deletion}}", "ranking_rationale": "Uncertain."}]}
    before = deepcopy(response)
    result = bind_assessment_measurements(response, summary)
    assert result["evaluation_assessments"][0]["mechanistic_plausibility"] == "delta_deletion=0.0"
    assert result["evaluation_assessments"][1]["mechanistic_plausibility"] == "delta_deletion=-0.8123"
    assert "delta_deletion=-0.8123" in result["markdown"]
    assert response == before
    bad = deepcopy(response)
    bad["evaluation_assessments"][1]["mechanistic_plausibility"] = "delta_deletion=-0.0000"
    with pytest.raises(ValueError, match="transcription"):
        bind_assessment_measurements(bad, summary)
    bad["evaluation_assessments"][1]["mechanistic_plausibility"] = "{{invented_measurement}}"
    with pytest.raises(ValueError, match="Unknown"):
        bind_assessment_measurements(bad, summary)
    bad["evaluation_assessments"][1]["mechanistic_plausibility"] = "{{M001_delta_deletion}}"
    with pytest.raises(ValueError, match="different molecule"):
        bind_assessment_measurements(bad, summary)


def test_model_scientific_disagreement_is_preserved_not_corrected_to_a_pass():
    response = {"evaluation_assessments": [{"name": "A", "mechanistic_plausibility": "Unsupported model judgment.",
                "ranking_rationale": "Model's original ranking."}], "markdown": "Unexpected contradictory commentary."}
    result = bind_assessment_measurements(response, {"provided_asos": [{"name": "A"}]})
    assert result["evaluation_assessments"] == response["evaluation_assessments"]
    assert "Unsupported model judgment." in result["markdown"]
    assert "Unexpected contradictory commentary." in result["markdown"]


@pytest.mark.parametrize("field", ["assumptions", "limitations", "evidence_refs", "suggested_followups", "markdown"])
def test_report_level_unknown_citations_fail_before_rendering_without_mutation(field):
    from aso_agents.response_contracts import response_contract_errors
    summary = attach_measurement_references({"provided_asos": [{"name": "probe", "input_row_id": "row"}]})
    response = {"evaluation_assessments": [{"name": "probe", "input_row_id": "row",
                 "mechanistic_plausibility": "Activity is unmeasured."}],
                field: "A template example {{unbound}} is not a recorded reference."}
    before = deepcopy(response)
    errors = response_contract_errors(response, {"task_type": "evaluate_existing_aso_set", "evaluation_summary": summary})
    assert any(field in error and "Unknown evaluation measurement reference" in error for error in errors)
    with pytest.raises(ValueError, match="Unknown evaluation measurement reference"):
        bind_assessment_measurements(response, summary)
    assert response == before


def test_merged_original_components_cannot_borrow_another_observations_measurement():
    summary = attach_measurement_references({"provided_asos": [
        {"name": "repeated", "input_row_id": "a", "splice_eval_summary": {"delta_deletion": 0.0}},
        {"name": "repeated", "input_row_id": "b", "splice_eval_summary": {"delta_deletion": -0.2}}]})
    correct = {"name": "repeated", "input_row_id": "a", "uncertainty": "Recorded {{M001_delta_deletion}}; activity remains unknown."}
    response = {"evaluation_assessments": [correct],
                "original_component_responses": [{"evaluation_assessments": [deepcopy(correct)]}]}
    assert assessment_measurement_errors(response, summary) == []
    response["original_component_responses"][0]["evaluation_assessments"][0]["uncertainty"] = "{{M002_delta_deletion}}"
    assert any("original_component_responses" in error and "different molecule" in error
               for error in assessment_measurement_errors(response, summary))


def test_valid_report_level_citations_keep_zero_and_scientific_uncertainty():
    summary = attach_measurement_references({"provided_asos": [
        {"name": "probe", "input_row_id": "a", "splice_eval_summary": {"delta_deletion": 0.0}}]})
    response = {"evaluation_assessments": [{"name": "probe", "input_row_id": "a",
                "mechanistic_plausibility": "Activity remains unknown."}],
                "assumptions": ["Recorded perturbation {{M001_delta_deletion}}; not measured efficacy."]}
    assert assessment_measurement_errors(response, summary) == []
    result = bind_assessment_measurements(response, summary)
    assert result["assumptions"] == ["Recorded perturbation 0.0; not measured efficacy."]
    assert result["evaluation_assessments"] == response["evaluation_assessments"]


def test_runtime_evaluation_catalog_preserves_reference_identity(monkeypatch):
    import run_goal_intake as runtime
    struct = {"seq_region_name": "1", "strand": 1, "exons_tx": [],
              "versioned_id": "TX.7", "version": 7, "reference_source": "local_gtf", "reference_annotation": "test.gtf"}
    monkeypatch.setattr(runtime, "get_gene_transcript_structures", lambda **kwargs:
                        {"transcripts": [{"id": "TX"}], "structures": {"TX": struct}})
    loaded = runtime._load_all_transcripts_for_gene({"gene_symbol": "TEST", "species": "human"})
    for key in ("versioned_id", "version", "reference_source", "reference_annotation"):
        assert loaded["TX"]["metadata"][key] == struct[key]


def test_exact_screening_definition_is_exported_with_its_query_and_zero(tmp_path):
    import json
    from aso_agents import pipeline_tools as pipeline
    gtf = tmp_path / "test.v7.gtf"
    gtf.write_text("# neutral reference identity\n")
    config = SimpleNamespace(gtf_path=gtf, genome_fasta=tmp_path / "test.fa", bowtie_prefix=tmp_path / "test", ensembl_assembly="test")
    scope = exact_gene_count_scope(config, "GENE.4", {"mapper_alignment_cap": 2000, "overmapped_queries": 0, "junction_mapping": {"junctions_scanned": 3}})
    path = tmp_path / "tiles.json"
    path.write_text(json.dumps({"rows": [{"provided_row_index": 0, "Sequence": "AACCGGTA", "off_target_gene_count": 0}]}))
    pipeline._enrich_exact_off_target_bins_for_tiles(str(path), scope)
    scored = json.loads(path.read_text())["rows"]
    rows = [{"name": "A"}]
    pipeline._merge_scored_tiles_into_provided_asos(rows, scored)
    result = pipeline._provided_aso_export_rows({"evaluation_summary": {"provided_asos": rows}})[0]
    assert result["Off_target_exact_count"] == 0
    assert result["off_target_count_status"] == "complete_within_declared_scope"
    assert result["off_target_query_sequence"] == "AACCGGTA"
    assert result["off_target_count_scope"]["excluded_gene_id"] == "GENE"
    assert "both genomic strands" in scope["genomic_strand_policy"]
    assert "entire target gene" in scope["intended_target_exclusion"]
    assert "Intronic-only" in scope["counted_genomic_rna_universe"]
    assert scored[0]["Off_target_gene_ids_by_mismatch"]["1mm"] is None
    summary = pipeline._compact_existing_aso_evaluation_for_response({"provided_asos": rows, "provided_aso_scoring": {"off_target_count_scope": scope}})
    report = bind_assessment_measurements({"evaluation_assessments": []}, summary)["markdown"]
    assert "genomic_strand_policy" in report and "intended_target_exclusion" in report


@pytest.mark.parametrize("fault", ["capped", "junction_error", "skipped_chrom", "missing_count"])
def test_incomplete_exact_search_cannot_be_rendered_as_zero(tmp_path, fault):
    import json
    from aso_agents import pipeline_tools as pipeline
    row = {"provided_row_index": 0, "Sequence": "AACCGGTA", "off_target_gene_count": 0}
    scope = {"junction_search_summary": {}}
    if fault == "capped":
        row.update(off_target_mapping_capped=True, off_target_mapping_count_lower_bound=2001)
    elif fault == "junction_error":
        scope["junction_search_summary"]["error"] = "missing_reference"
    elif fault == "skipped_chrom":
        scope["junction_search_summary"]["skipped_chrom"] = 1
    else:
        row.pop("off_target_gene_count")
    path = tmp_path / "tiles.json"
    path.write_text(json.dumps({"rows": [row]}))
    pipeline._enrich_exact_off_target_bins_for_tiles(str(path), scope)
    scored = json.loads(path.read_text())["rows"]
    assert scored[0]["Off_target_exact_count"] is None
    rows = [{"name": "A", "off_target_gene_count": 0}]
    pipeline._merge_scored_tiles_into_provided_asos(rows, scored)
    assert rows[0]["off_target_gene_count"] is None
    assert rows[0]["Off_target_exact_count"] is None
    assert rows[0]["stage_results"]["off_target_mapping"]["status"] in {"incomplete", "unavailable"}
