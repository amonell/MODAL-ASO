"""Submitted-set identity and requested exports remain bound to native evidence."""
import csv
import json

from aso_agents import pipeline_tools as pipeline
from aso_agents.provided_set_comparison import write_submitted_set_table
from aso_agents.evaluation_response_audit import (panel_comparison_audit_packet,
    panel_comparison_finding_errors)
from aso_agents.response_delivery import normalize_panel_ranking_rationales


def test_submitted_overlap_uses_physical_sequences_not_target_comparison_sequences():
    physical_a = "TAGGTGCCAGGCAACCTCCA"
    target_a = "TGGAGGTTGCCTGGCACCTA"
    set_a = pipeline._build_compare_set(set_id="A", label="A", source_type="provided", rows=[
        {"name": "A1", "sequence": physical_a, "comparison_sequence": target_a}])
    set_b = pipeline._build_compare_set(set_id="B", label="B", source_type="provided", rows=[
        {"name": "B1", "sequence": target_a, "comparison_sequence": physical_a}])
    overlap = pipeline._multi_set_exact_overlap_payload([set_a, set_b])
    assert overlap["pairwise_overlaps"][0]["shared_count"] == 0
    assert set_a["sequence_set"] == {physical_a}
    assert set_b["sequence_set"] == {target_a}


def test_submitted_comparison_exports_one_source_bound_row_per_molecule(tmp_path):
    path = tmp_path / "comparison.csv"
    evidence = [{"set_id": "A", "name": "A1", "sequence": "TAGGTGCCAGGCAACCTCCA",
                 "status": "exact antisense", "site": "ENST00000302118.5 cDNA 504–523",
                 "mismatch": "0", "exploratory_rank": 1, "off_target_exact_count": 0,
                 "off_target_mismatch_budget": 0, "off_target_count_status": "complete_within_declared_scope",
                 "off_target_scope": {"count_unit": "genes"}}]
    write_submitted_set_table(path, evidence)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["physical_aso_5to3"] == evidence[0]["sequence"]
    assert rows[0]["off_target_mismatch_budget"] == "0"
    assert rows[0]["off_target_count_unit"] == "genes"


def test_single_comparison_task_delivers_its_own_report_and_table(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task_id = "comparison"
    task_root = pipeline._run_path(str(run_dir), "v3_tasks", task_id)
    task_root.mkdir(parents=True)
    report = task_root / "aso_set_comparison.md"
    report.write_text("# Native checked comparison\nA vs B: 0 physical overlaps.\n")
    table = task_root / "submitted_set_comparison.csv"
    table.write_text("set_id,physical_aso_5to3\nA,TAGGTGCCAGGCAACCTCCA\n")
    (task_root / "task_outputs_manifest.json").write_text(json.dumps({
        "task_id": task_id, "artifacts": {"markdown": str(report), "comparison_table_csv": str(table)}}))
    output_dir = tmp_path / "output"
    state = {"request_envelope": {"schema_version": "v3", "tasks": [
        {"task_id": task_id, "task_type": "compare_aso_sets"}], "final_outputs": [
        {"task_id": task_id, "kind": "file", "format": "csv", "path": str(output_dir / "comparison.csv")},
        {"task_id": task_id, "kind": "report_md", "source_artifact": "report_md",
         "path": str(output_dir / "comparison.md")}]}}
    catalog = pipeline._native_export_catalog(state, str(run_dir), task_id)
    assert catalog == {"report_md": str(report), "comparison_table_csv": str(table)}
    delivery = pipeline._materialize_requested_final_outputs(state, str(run_dir))
    assert delivery["ok"]
    assert (output_dir / "comparison.md").read_bytes() == report.read_bytes()
    assert (output_dir / "comparison.csv").read_bytes() == table.read_bytes()


def test_mixed_origin_comparison_preserves_duplicate_observations_without_inventing_pool_rank(tmp_path, monkeypatch):
    source = {"input_row_id": "submitted-1", "name": "old", "sequence": "AACC",
              "off_target_exact_count": 0}
    generated = {"Name": "new", "Sequence": "AACC", "Off_target_exact_count": 0,
                 "selection_rank": 1,
                 "sequence_role": "transcript_oriented_target_binding_site",
                 "synthesized_antisense_sequence": "GGTT"}
    sets = [
        pipeline._build_compare_set(set_id="submitted", label="Submitted", source_type="provided", rows=[source]),
        pipeline._build_compare_set(set_id="generated", label="Generated", source_type="generated",
                                    source_task_id="design", rows=[generated]),
    ]
    prepared = {"status": "ok", "state_path": str(tmp_path / "state.json"),
                "state": {"request": "Compare both observations"}, "run_dir": str(tmp_path),
                "task_id": "compare", "task": {"task_type": "compare_aso_sets"},
                "task_request_envelope": {"schema_version": "v3", "request_text": "Compare both observations"},
                "task_context": {}}
    monkeypatch.setattr(pipeline, "_prepare_v2_task_execution", lambda **_: prepared)
    monkeypatch.setattr(pipeline, "_load_task_payload_by_type", lambda *_, **__: {})
    monkeypatch.setattr(pipeline, "_collect_aso_compare_sets", lambda **_: sets)
    monkeypatch.setattr(pipeline, "_build_compare_aso_sets_response",
                        lambda **_: {"markdown": "Two distinct observations share a sequence."})
    monkeypatch.setattr(pipeline, "_complete_v2_task", lambda **kwargs: kwargs)
    result = pipeline.tool_compare_aso_sets(str(tmp_path / "state.json"))
    table = result["artifacts"]["comparison_table_csv"]
    with open(table, newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert [row["source_type"] for row in rows] == ["provided", "generated"]
    assert [row["recorded_sequence_5to3"] for row in rows] == ["AACC", "AACC"]
    assert rows[1]["synthesized_antisense_sequence_5to3"] == "GGTT"
    assert all("pooled_rank" not in row for row in rows)
    state = {"request_envelope": {"schema_version": "v3", "tasks": [
        {"task_id": "compare", "task_type": "compare_aso_sets"}], "final_outputs": [
        {"task_id": "compare", "kind": "comparison_table_csv", "source_artifact": "comparison_table_csv",
         "path": str(tmp_path / "output" / "comparison.csv")}]}}
    delivery = pipeline._materialize_requested_final_outputs(state, str(tmp_path))
    assert delivery["ok"]
    assert (tmp_path / "output" / "comparison.csv").read_bytes() == open(table, "rb").read()


def test_panel_comparison_audit_sees_other_molecules_outside_component():
    source = {"request_text": "Compare all submitted molecules", "evaluation_summary": {
        "provided_asos": [{"input_row_id": "r2", "name": "A2"}],
        "panel_comparison_overview": [
            {"input_row_id": "r1", "name": "A1", "binding_evidence": {"exact_antisense": True}},
            {"input_row_id": "r2", "name": "A2", "binding_evidence": {"exact_antisense": True}},
        ]}}
    response = {"evaluation_assessments": [{"input_row_id": "r2", "name": "A2",
        "ranking_rationale": "A2 alone has an exact antisense match."}]}
    packet = panel_comparison_audit_packet(source, response)
    assert packet["panel_observation_count"] == 2
    assert [row["name"] for row in packet["full_panel_source"]] == ["A1", "A2"]
    assert packet["assessed_subjects"]["/evaluation_assessments/0/ranking_rationale"] == {
        "input_row_id": "r2", "name": "A2",
        "source_pointer": "/evaluation_summary/panel_comparison_overview/1"}
    assert packet["full_panel_source"][0]["source_pointer"] == "/evaluation_summary/panel_comparison_overview/0"
    assert packet["draft_narrative_fields"]["/evaluation_assessments/0/ranking_rationale"] == response["evaluation_assessments"][0]["ranking_rationale"]
    finding = {"response_path": "/evaluation_assessments/0/ranking_rationale",
               "quote": "A2 alone", "kind": "contradicted_fact", "reason": "A1 also matches",
               "evidence_paths": ["/evaluation_summary/panel_comparison_overview/0/binding_evidence/exact_antisense"]}
    assert panel_comparison_finding_errors({"findings": [finding]}, response, source)
    finding["evidence_paths"].append(
        "/evaluation_summary/panel_comparison_overview/1/binding_evidence/exact_antisense")
    assert panel_comparison_finding_errors({"findings": [finding]}, response, source) == []


def test_panel_ranking_delivery_uses_recorded_facts_instead_of_free_form_exclusivity():
    payload = {"task_type": "evaluate_existing_aso_set", "evaluation_summary": {
        "panel_comparison_overview": [
            {"input_row_id": "r1", "name": "A1", "selection_rank": 2, "gc_fraction": 0.45,
             "orientation_interpretation": {"original_molecule_exact_binding": True,
                                            "match_orientation": "reverse_complement"}},
            {"input_row_id": "r2", "name": "A2", "selection_rank": 1, "gc_fraction": 0.5,
             "orientation_interpretation": {"original_molecule_exact_binding": True,
                                            "match_orientation": "reverse_complement"}},
        ]}}
    response = {"evaluation_assessments": [{"input_row_id": "r2", "name": "A2",
        "ranking_rationale": "A2 alone has an exact antisense match."}]}
    delivered = normalize_panel_ranking_rationales(response, payload)
    assert response["evaluation_assessments"][0]["ranking_rationale"].startswith("A2 alone")
    rationale = delivered["evaluation_assessments"][0]["ranking_rationale"]
    assert "rank: 1 of 2" in rationale
    assert "exact antisense sequence-alignment hypothesis" in rationale
    assert "alone" not in rationale
    assert delivered["native_delivery_normalization"]["assessment_text_changed"]
