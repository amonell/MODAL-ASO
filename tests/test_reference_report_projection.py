from copy import deepcopy
import csv
import json

import pytest

from aso_agents.reference_report_projection import evaluation_projection, metric_evaluation_response
from aso_agents.tabular_reports import render_report_tables
from aso_agents import pipeline_tools as pipeline


def evidence():
    catalog = {"transcript_identities": {
        "ref_a": {"transcript_id": "ref_a.7", "species": "species_a", "assembly": "assembly_a"},
        "ref_b": {"transcript_id": "ref_b.2", "species": "species_b", "assembly": "assembly_b"}}}
    def hit(ref, context, strand, mm=None):
        result = {"transcript_id": ref, "mapping_scope": context, "match_orientation": strand,
                  "cdna_start": 12, "cdna_end": 15, "genomic_segments": [{"start": 27, "end": 30}]}
        if mm is not None:
            result["mismatch_count"] = mm
        return result
    return {"reference_catalog": catalog, "provided_asos": [{
        "aso_id": "arbitrary_id", "sequence": "ACGT", "length": 4, "gc_fraction": .5,
        "mapping_hits": [hit("ref_a", "spliced_cdna", "reverse_complement"),
                         hit("ref_a", "pre_mrna_locus", "reverse_complement")],
        "near_match_candidates": [hit("ref_b", "pre_mrna_locus", "input_sequence", 1),
                                  hit("ref_b", "spliced_cdna", "reverse_complement", 3)],
        "orientation_interpretation": {"provided_sequence_role": "unknown", "requires_confirmation": False},
        "off_target_gene_count": 1, "off_target_count_scope": {"exclusion_applied": False, "excluded_gene_id": None}}]}


@pytest.mark.parametrize("near_order", [False, True])
def test_rna_context_and_orientation_cannot_change_mature_conservation(near_order):
    source = evidence()
    if near_order:
        source["provided_asos"][0]["near_match_candidates"].reverse()
    before = deepcopy(source)
    row = evaluation_projection(source)[0]
    assert row["species_a_ref_a.7_mature_antisense_recorded_exact_hits"] == 1
    assert row["species_b_ref_b.2_mature_antisense_best_recorded_mismatches"] == 3
    assert row["screen_exclusion_applied"] is False and row["recorded_screened_gene_count"] == 1
    assert "isoform census" in row["reference_catalog_scope"]
    assert source == before


def test_unmapped_as_received_identity_is_not_a_mandatory_clarification():
    source = evidence(); row = source["provided_asos"][0]
    row["mapping_hits"] = []
    projected = evaluation_projection(source)[0]
    assert projected["evaluated_molecule_sequence_5to3"] == row["sequence"]
    assert projected["requires_sequence_confirmation"] is False
    assert projected["species_a_ref_a.7_mature_antisense_best_recorded_mismatches"] is None


def test_explicit_target_site_and_orientation_confirmation_are_respected():
    source = evidence(); row = source["provided_asos"][0]
    row["orientation_interpretation"] = {"provided_sequence_role": "target_site_sequence",
        "evaluated_molecule_sequence": "TGCA", "requires_confirmation": True}
    row["mapping_hits"][0]["match_orientation"] = "input_sequence"
    projected = evaluation_projection(source)[0]
    assert projected["input_sequence_5to3"] == "ACGT"
    assert projected["evaluated_molecule_sequence_5to3"] == "TGCA"
    assert projected["requires_sequence_confirmation"] is True
    assert projected["species_a_ref_a.7_mature_antisense_recorded_exact_hits"] == 1


@pytest.mark.parametrize("size", [1, 8, 43])
def test_all_molecules_and_all_reference_groups_survive(size):
    source = evidence()
    source["provided_asos"] = [{**deepcopy(source["provided_asos"][0]), "aso_id": f"different_{i}"} for i in range(size)]
    metric_rows = [{"aso_id": r["aso_id"], "sequence": r["sequence"], "length": 4, "gc_fraction": .5,
                    "additional_metric": i} for i, r in enumerate(source["provided_asos"])]
    tasks = [{"task_type": "evaluate_existing_aso_set", "payload": source},
             {"task_type": "sandbox_analysis", "payload": {"per_aso": metric_rows}}]
    response = pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks)
    assert len(response["reference_projection_rows"]) == size
    assert all(len(r["reference_alignment_groups"]) == 4 for r in response["reference_projection_rows"])
    assert "additional_metric" in response["markdown"]
    assert f"different_{size-1}" in response["markdown"]


@pytest.mark.parametrize("extra", ["analyze_artifact", "trace_filter_decisions", "design_new_asos"])
def test_unrecognized_requested_work_is_not_hidden(extra):
    tasks = [{"task_type": "evaluate_existing_aso_set", "payload": evidence()},
             {"task_type": "sandbox_analysis", "payload": {}}, {"task_type": extra, "payload": {}}]
    assert metric_evaluation_response(tasks) is None


def test_requested_csv_and_markdown_share_projection(tmp_path):
    source = evidence()
    tasks = [{"task_type": "evaluate_existing_aso_set", "payload": source},
             {"task_type": "sandbox_analysis", "payload": {"results": [{"aso_id": "arbitrary_id", "sequence": "ACGT", "length": 4, "gc_fraction": .5}]}}]
    response = metric_evaluation_response(tasks)
    csv_path, md_path = tmp_path / "arbitrary.csv", tmp_path / "arbitrary.md"
    pipeline._write_v2_response_outputs(run_dir=str(tmp_path), markdown_text=response["markdown"], response_payload=response,
        response_basename="composite_report", title="Evaluation", requested_outputs=[{"path": str(csv_path)}, {"path": str(md_path)}])
    row = next(csv.DictReader(csv_path.open()))
    assert row["species_a_ref_a.7_mature_antisense_recorded_exact_hits"] == "1"
    assert row["species_b_ref_b.2_mature_antisense_best_recorded_mismatches"] == "3"
    assert json.loads(row["reference_alignment_groups"])[0]["exact_hit_count"] == 1
    assert md_path.read_text() == (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("identifier", ["renamed_sample", "unseen_98", "control_id"])
def test_audit_notes_cannot_invent_row_discrepancies_or_override_missingness(tmp_path, identifier):
    csv_path, md_path, summary = tmp_path / "audit.csv", tmp_path / "requested.md", tmp_path / "summary.md"
    fields = ["id", "length_match", "claimed_gc_round3_match", "eligibility_status", "eligibility_reasons"]
    rows = [{"id": identifier, "length_match": "True", "claimed_gc_round3_match": "True", "eligibility_status": "ineligible", "eligibility_reasons": "length mismatch"},
            {"id": "another", "length_match": "True", "claimed_gc_round3_match": "True", "eligibility_status": "unknown", "eligibility_reasons": "accessibility not observed"}]
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    for path in (md_path, summary):
        path.write_text("An unsupported illustrative row passes despite wrong claims.\n{{table:arbitrary_audit}}")
    result = {"eligibility": {"criteria": "typed criteria", "accessibility_imputed": False},
              "claim_discrepancies": {"comparison_rule": "round three decimals"},
              "report_tables": [{"id": "arbitrary_audit", "source_path": str(csv_path), "columns": fields}]}
    original = csv_path.read_bytes()
    render_report_tables(result, [csv_path, md_path], summary, required=True)
    assert csv_path.read_bytes() == original
    assert "unsupported illustrative" not in md_path.read_text()
    assert f"ineligible: 1 row(s); {identifier}" in md_path.read_text()
    assert "unknown: 1 row(s); another" in md_path.read_text()
    assert "round three decimals" in md_path.read_text()
    assert "unsupported illustrative" in next(iter(result["historical_model_report_templates"].values()))


def test_custom_analysis_report_is_not_replaced_by_a_metric_audit(tmp_path):
    from aso_agents.audit_reconciliation_report import canonical_audit_texts
    texts = {"custom.md": "A different scientific analysis"}
    assert canonical_audit_texts({"eligibility": {}}, texts, {}, []) == (texts, None)


def test_additional_scientific_schema_keeps_existing_synthesis_path():
    tasks = [{"task_type": "evaluate_existing_aso_set", "payload": evidence()},
             {"task_type": "sandbox_analysis", "payload": {"results": [{"aso_id": "arbitrary_id", "sequence": "ACGT", "length": 4, "gc_fraction": .5}],
                 "additional_binding_analysis": {"observed_result": "must not disappear"}}}]
    assert metric_evaluation_response(tasks) is None
