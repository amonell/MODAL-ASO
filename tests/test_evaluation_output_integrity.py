import csv
import json
from copy import deepcopy
from pathlib import Path

from aso_agents import pipeline_tools as p


def test_artifact_index_accepts_large_evidence_cell(tmp_path):
    table = tmp_path / "evaluation.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "complete_evidence"])
        writer.writerow(["A", "x" * 200_000])
    previous_limit = csv.field_size_limit()
    try:
        summary = p._artifact_delimited_table_summary(table)
    finally:
        csv.field_size_limit(previous_limit)
    assert summary["row_count"] == 1
    assert summary["columns"] == ["candidate", "complete_evidence"]
    assert "error" not in summary


def test_unmapped_input_is_not_relabelled_as_target_or_replacement():
    row = {"sequence": "AACCGGTA", "normalized_sequence": "AACCGGTA",
           "comparison_sequence": "AACCGGTA", "reverse_complement_sequence": "TACCGGTT",
           "mapping_hits": [], "reference_candidate_hits": []}
    result = p._provided_aso_orientation_interpretation(row)
    assert result["input_sequence"] == row["sequence"]
    assert result["provided_sequence_role"] == "unknown"
    assert "matched_target_sequence" not in result
    assert "synthesized_antisense_sequence" not in result


def test_evaluation_exports_retain_measured_rows_and_generated_assessment(tmp_path):
    report = "# Evaluation\n\nMolecule B has mismatched binding; activity remains uncertain."
    source_rows = [
        {"name": "A", "sequence": "AACCGGTA", "mapping_status": "mapped", "off_target_gene_count": 0},
        {"name": "B", "sequence": "ATCCGGTA", "mapping_status": "unmapped", "near_match_candidates": [{"mismatch_count": 1}]},
    ]
    payload = {"answer_summary": "One exact match and one uncertain near match.",
               "markdown": report, "evaluation_summary": {"provided_asos": source_rows, "context_summary": {"reference_build": "test-build"}},
               "evaluation_assessments": [{"name": "B", "sequence": "MUST_NOT_REPLACE_INPUT", "uncertainty": "Activity not measured."}]}
    before = deepcopy(payload)
    table = tmp_path / "exports" / "molecules.csv"
    rationale = tmp_path / "exports" / "rationale.md"
    result = p._write_v2_response_outputs(run_dir=str(tmp_path / "run"), markdown_text=report,
        response_payload=payload, response_basename="existing_aso_evaluation", title="Evaluation",
        requested_outputs=[{"kind": f"Complete table saved as {table}"}, {"kind": "rationale", "path": str(rationale)}])
    rows = list(csv.DictReader(table.open()))
    assert [r["name"] for r in rows] == ["A", "B"]
    assert [r["sequence"] for r in rows] == ["AACCGGTA", "ATCCGGTA"]
    assert rows[0]["off_target_gene_count"] == "0"
    assert json.loads(rows[1]["near_match_candidates"])[0]["mismatch_count"] == 1
    assert json.loads(rows[1]["model_assessment"])["uncertainty"] == "Activity not measured."
    assert rows[0]["model_assessment_status"] == "missing_or_ambiguous"
    assert rationale.read_text() == report == Path(result["report_md"]).read_text()
    stored = json.loads(Path(result["response_json"]).read_text())
    assert stored["answer_summary"] == before["answer_summary"]
    assert stored["evaluation_summary"]["provided_asos"] == source_rows
    assert payload == before


def test_response_composer_receives_chemistry_and_near_match_evidence():
    row = {"name": "B", "sequence": "ATCCGGTA", "chemistry": "steric chemistry",
           "mapping_hits": [{"genomic_blocks": [[100, 103], [200, 203]], "strand": "-"}],
           "near_match_candidates": [{"matched_target_sequence": "TACCGGTT", "mismatch_count": 1}]}
    compact = p._compact_existing_aso_evaluation_for_response({"provided_asos": [row]})
    saved = compact["provided_asos"][0]
    for key in ("sequence", "chemistry", "mapping_hits", "near_match_candidates"):
        assert saved[key] == row[key]


def test_unknown_user_answer_does_not_authorize_skipping_resources(monkeypatch):
    from aso_agents import reference_decisions
    monkeypatch.setattr(reference_decisions, "semantic_result", lambda *args: {"error": "unresolved"})
    for answer in ("I do not have that information.", "Do not download the reference."):
        state = {"clarification": {"kind": "reference_preflight", "last_answers": [{"answer": answer}]}}
        assert p._parse_reference_provisioning_decision(state, {})["action"] == "ask"


def test_explicit_export_path_survives_request_normalization():
    outputs = p._normalize_final_outputs([{"kind": "evaluation", "path": "reports/my table.csv", "format": "csv"}], "task")
    assert outputs[0]["path"] == "reports/my table.csv"
    assert outputs[0]["format"] == "csv"


def test_generated_sequence_roles_and_gene_units_are_evidence_based():
    from aso_agents.reporting_contracts import generated_sequence_fields, sequence_notes, exact_count_unit
    row = {"Sequence": "AACCGGTA", "Synthesis": "TACCGGTT", "Off_target_exact_count": 8, "Off_target_0mm_loci": [1]*8, "off_target_gene_count": 5}
    fields = generated_sequence_fields(row)
    assert fields["sequence_role"] == "transcript_oriented_target_binding_site"
    assert fields["synthesized_antisense_sequence"] == "TACCGGTT"
    assert row["Sequence"] == "AACCGGTA"
    assert "Synthesized antisense ASO" in " ".join(sequence_notes(row))
    assert exact_count_unit(row) == "loci"
    assert p._summary_offtarget_count(row) == 5
    assert p._summary_offtarget_count({"Off_target_exact_count": 8}) is None
    assert p._summary_offtarget_count({"Off_target_exact_count": 0, "off_target_exact_count_unit": "genes"}) == 0
    assert generated_sequence_fields({"Sequence": "AACCGGTA"})["sequence_role"] == "unverified_source_sequence"
    assert generated_sequence_fields({"Sequence": "AACCGGTA", "Synthesis": "AAAA"})["sequence_role"] == "inconsistent_with_synthesis"


def test_near_matches_distinguish_binding_and_same_strand_similarity(monkeypatch):
    monkeypatch.setattr(p, "_fetch_transcript_locus_sequence", lambda **kwargs: {
        "transcript_oriented_sequence": "TACAGGTTNNNAACCGGAA", "chrom": "chrT", "strand": "+", "locus_start": 101, "locus_end": 117,
        "transcript_id": "tx", "isoform_label": "tx"})
    candidates = p._best_locus_near_matches(entry={"sequence": "AACCGGTA", "reverse_complement_sequence": "TACCGGTT"}, isoform_transcripts={"tx": {}}, runtime=None, genome=True, preferred_transcript_id="tx")
    anti = next(x for x in candidates if x["match_orientation"] == "reverse_complement")
    sense = next(x for x in candidates if x["match_orientation"] == "input_sequence")
    assert anti["genomic_span"] == {"chrom": "chrT", "start": 101, "end": 108, "length": 8}
    assert anti["mismatch_positions_in_supplied_sequence_1based"] == [5]
    assert anti["mismatch_count"] == 1
    assert sense["binding_interpretation"].startswith("same_strand_similarity_only")
    assert anti["binding_interpretation"].startswith("antisense_binding_hypothesis")


def test_splice_proxy_ranking_preserves_measurements_without_false_task_rank(tmp_path):
    class Runtime:
        @staticmethod
        def _annotate_gapmer_selection_ranks(path):
            data = json.loads(Path(path).read_text())
            for rank, row in enumerate(data["rows"], 1):
                row["selection_rank"] = rank
                row["selection_score_components"] = {"test_proxy": 0}
            Path(path).write_text(json.dumps(data))
            return {"method": "test_proxy"}
    rows = [{"name": "B", "sequence": "AACCGGTA", "mapping_status": "unmapped"}]
    result = p._score_provided_asos_with_existing_pipeline(rows=rows, eval_goal={}, runtime=Runtime(), run_dir=str(tmp_path), task_id="eval", cdna_map={}, iso_label=None, splice_related=True)
    assert result["status"] == "complete"
    assert rows[0]["knockdown_proxy_rank"] == 1
    assert "selection_rank" not in rows[0]
    stored = json.loads(Path(result["scored_tiles_path"]).read_text())
    assert stored["rows"][0]["knockdown_proxy_score_components"] == {"test_proxy": 0}
    assert "selection_rank" not in stored["rows"][0]
    compact = p._compact_existing_aso_evaluation_for_response({"provided_asos": rows})
    assert compact["provided_asos"][0]["knockdown_proxy_rank"] == 1


def test_compact_response_preserves_zero_exact_and_mismatch_counts():
    result = p._compact_existing_aso_evaluation_for_response({"provided_asos": [{"name": "A", "Off_target_exact_count": 0, "Off_target_1mm_count": 0, "Off_target_weighted_risk_score": 0}]})
    row = result["provided_asos"][0]
    assert row["off_target_exact_count"] == row["off_target_1mm_count"] == row["off_target_weighted_risk_score"] == 0
