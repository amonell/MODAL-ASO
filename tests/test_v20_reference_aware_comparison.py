"""A submitted-set comparison must retain set identity and target-reference work."""
import asyncio
import csv
import json

from aso_agents import pipeline_tools as pipeline
from aso_agents.provided_set_comparison import submitted_set_response, write_submitted_set_table
from aso_agents.grounded_mixed_report import grounded_mixed_report


def test_submitted_set_csv_is_two_sets_not_a_self_comparison(tmp_path):
    path = tmp_path / "submitted.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["set_id", "candidate_id", "molecule_5to3", "intended_chemistry"])
        writer.writerows([
            ["first", "a", "ACGTACGTACGTACGTACGT", "full-PS gapmer"],
            ["first", "b", "CACGTACGTACGTACGTACG", "full-PS gapmer"],
            ["second", "c", "TGCATGCATGCATGCATGCA", "full-PS gapmer"],
            ["second", "d", "ATGCATGCATGCATGCATGC", "full-PS gapmer"],
        ])
    context = {
        "provided_aso_files": [{"path": str(path)}],
        "artifact_refs": [{"path": str(path), "artifact_type": "csv"}],
        "provided_asos": [
            {"name": name, "sequence": sequence, "chemistry": "full-PS gapmer",
             "notes": "Assess both orientations"}
            for name, sequence in [
                ("a", "ACGTACGTACGTACGTACGT"), ("b", "CACGTACGTACGTACGTACG"),
                ("c", "TGCATGCATGCATGCATGCA"), ("d", "ATGCATGCATGCATGCATGC")]
        ],
        "request_text": "Compare the two sets in the attached file against a public transcript.",
    }
    sets = pipeline._collect_aso_compare_sets(
        state={}, task={"task_id": "compare", "context": context},
        task_context=context, request_envelope=context, evaluated_payload={})
    assert [(item["set_id"], len(item["rows"])) for item in sets] == [
        ("first", 2), ("second", 2)]
    assert {row["name"] for item in sets for row in item["rows"]} == {"a", "b", "c", "d"}


def test_mandatory_public_reference_cannot_route_to_self_contained_sandbox(monkeypatch):
    base = {"status": "ready", "questions": [], "reason": None,
            "request_envelope": {"tasks": [{"task_id": "compare", "task_type": "sandbox_analysis",
                "native": False, "context": {"gene_symbol": "GENE",
                    "web_retrieval": {"required": True}}}]}}
    corrected = json.loads(json.dumps(base))
    corrected["request_envelope"]["tasks"][0].update(task_type="compare_aso_sets", native=True)
    proposals, reviews = [], []

    async def plan(prompt):
        proposals.append(prompt)
        if len(proposals) == 2:
            feedback = json.loads(prompt.split("\n\n", 1)[1])
            assert feedback["verified_graph_issues"][0]["category"] == "wrong_task_type"
        return base if len(proposals) == 1 else corrected

    async def review(prompt):
        reviews.append(prompt)
        return {"status": "valid", "issues": []}

    async def finalize(_prompt, proposal):
        return proposal

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    async def no_clarification(_prompt, proposal):
        return proposal
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", no_clarification)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_agent_once", review)
    monkeypatch.setattr(pipeline, "_finalize_verified_request_plan", finalize)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once(
        "Compare supplied ASOs against a public transcript."))
    assert result == corrected
    assert len(proposals) == 2
    assert len(reviews) == 2  # Both independent review passes inspect only the capable graph.


def test_target_metadata_does_not_force_native_evaluation():
    envelope = {"tasks": [{"task_id": "compare", "task_type": "sandbox_analysis",
        "native": False, "context": {"provided_asos": [{"name": "A", "sequence": "ACGT"}],
        "target_context": "a supplied molecule against a biological reference"}}]}
    assert pipeline._executor_capability_issues(envelope) == []

    # Supplied-row arithmetic without a biological target remains sandbox work.
    envelope["tasks"][0]["context"].pop("target_context")
    assert pipeline._executor_capability_issues(envelope) == []


def test_gene_label_on_supplied_row_audit_does_not_imply_reference_binding():
    envelope = {"tasks": [{"task_id": "audit", "task_type": "sandbox_analysis",
        "native": False, "context": {"provided_aso_files": [{"path": "/input/pilot.csv"}],
        "provided_asos": [{"name": "row_1", "sequence": "ACGT"}],
        "gene_symbol": "GENE"}}]}
    assert pipeline._executor_capability_issues(envelope) == []

    # Available finite-reference work is assessed by the semantic graph review;
    # an explicit mandatory lookup still excludes the self-contained executor.
    envelope["tasks"][0]["context"]["target_context"] = "align against a reference"
    assert pipeline._executor_capability_issues(envelope) == []
    envelope["tasks"][0]["context"].pop("target_context")
    envelope["tasks"][0]["context"]["web_retrieval"] = {"required": True}
    assert [issue["category"] for issue in pipeline._executor_capability_issues(envelope)] == ["wrong_task_type"]


def test_unrequested_transcript_accession_cannot_become_a_user_lock(tmp_path):
    source = tmp_path / "submitted.csv"
    source.write_text("candidate_id,molecule_5to3\nA,ACGT\n")
    envelope = {"request_text": "Evaluate the supplied molecule against the canonical protein-coding transcript.",
        "tasks": [{"task_id": "evaluate", "task_type": "evaluate_existing_aso_set", "native": True,
                   "context": {"transcript_id": "ENST00000000001.7",
                               "provided_aso_files": [{"path": str(source)}]}}]}
    issues = pipeline._executor_capability_issues(envelope)
    assert len(issues) == 1 and issues[0]["category"] == "unattested_reference_identity"
    envelope["request_text"] += " Use ENST00000000001.7."
    assert pipeline._executor_capability_issues(envelope) == []
    envelope["request_text"] = "Evaluate the supplied molecule against the canonical transcript."
    source.write_text("candidate_id,molecule_5to3,transcript_id\nA,ACGT,ENST00000000001.7\n")
    assert pipeline._executor_capability_issues(envelope) == []


def test_requested_transcript_identity_is_visible_to_intake_guard(tmp_path):
    source = tmp_path / "submitted.csv"
    source.write_text("candidate_id,molecule_5to3\nA,ACGT\n")
    envelope = {"tasks": [{"task_id": "evaluate", "task_type": "evaluate_existing_aso_set",
        "native": True, "context": {"transcript_id": "ENST00000000001.7",
        "evaluation_targets": [{"transcript_id": "ENST00000000001.7"}],
        "provided_aso_files": [{"path": str(source)}]}}]}
    assert [issue["category"] for issue in pipeline._executor_capability_issues(envelope)] == [
        "unattested_reference_identity", "unattested_reference_identity"]
    assert pipeline._executor_capability_issues(
        envelope, request_evidence="Evaluate the panel against ENST00000000001.7."
    ) == []
    assert [issue["category"] for issue in pipeline._executor_capability_issues(
        envelope, request_evidence="Evaluate the panel against the canonical transcript."
    )] == ["unattested_reference_identity", "unattested_reference_identity"]


def test_submitted_sequence_overlap_stays_distinct_from_reverse_complement(tmp_path):
    exact = {"transcript_id": "ENST00000000001.2", "mapping_scope": "spliced_cdna",
             "match_orientation": "reverse_complement", "cdna_start": 10, "cdna_end": 29}
    sense = {**exact, "match_orientation": "input_sequence"}
    a = {"name": "same_target_antisense", "sequence": "ACGTAACGTAACGTAACGTA", "input_metadata": {"set_id": "A"},
         "mapping_hits": [exact], "Off_target_mismatch_budget": 0,
         "principal_reference_exon_membership": {"status": "resolved", "transcript_id": "ENST00000000001.2",
             "alignment_mismatches": 0, "exons": [{"transcript_exon_ordinal": 2}]}}
    b = {"name": "reversed_entry", "sequence": "TACGTTACGTTACGTTACGT", "input_metadata": {"set_id": "B"},
         "mapping_hits": [sense], "Off_target_mismatch_budget": 0}
    sets = [{"set_id": key, "source_type": "provided", "rows": [row]} for key, row in (("A", a), ("B", b))]
    evaluation = {"provided_asos": [a, b], "context_summary": {"transcript_id": "ENST00000000001"},
                  "mapping_strategy": {"searched_spliced_cdna": True}}
    comparison = {"pairwise_overlaps": [{"set_a_label": "A", "set_b_label": "B", "shared_count": 0}]}
    report = submitted_set_response(sets, evaluation, comparison)
    assert "0 identical submitted sequence(s)" in report["markdown"]
    assert report["submitted_set_evidence"][0]["exploratory_rank"] < report["submitted_set_evidence"][1]["exploratory_rank"]
    assert "exon ordinal(s) 2" in report["submitted_set_evidence"][0]["site"]
    assert "same-strand exact site only" in report["submitted_set_evidence"][1]["status"]
    alternative = report["submitted_set_evidence"][1]
    assert alternative["same_strand_site"].startswith("ENST00000000001.2; cDNA 10–29")
    assert alternative["reverse_complement_alternative"] == a["sequence"]
    assert alternative["reverse_complement_submitted_match"] == "A/same_target_antisense"
    assert "B/reversed_entry: written-molecule same-strand site" in report["markdown"]
    export = tmp_path / "comparison.csv"
    write_submitted_set_table(export, report["submitted_set_evidence"])
    with export.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["same_strand_site"].startswith("ENST00000000001.2; cDNA 10–29")
    assert rows[1]["reverse_complement_matches_submitted_candidate"] == "A/same_target_antisense"


def test_submitted_set_composite_uses_complete_physical_set_assessment(tmp_path):
    assessment = {"answer_summary": "Compared two supplied sets and four physical molecules.",
                  "markdown": "All four physical molecules and their target hypotheses.",
                  "assumptions": [], "limitations": [], "evidence_refs": [],
                  "suggested_followups": [], "submitted_set_evidence": [{"name": "first"}]}
    tasks = [{"task_id": "evaluate", "task_type": "evaluate_existing_aso_set", "payload": {}},
             {"task_id": "compare", "task_type": "compare_aso_sets",
              "payload": {"comparison_assessment": assessment}}]
    response = pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks)
    assert response["answer_summary"] == assessment["answer_summary"]
    assert response["markdown"] == assessment["markdown"]
    assert "generated candidate" not in response["markdown"]


def test_mixed_report_uses_selected_panel_scope_not_global_inventory(tmp_path):
    panel = tmp_path / "design_panel.csv"
    with panel.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate", "antisense_5to3", "target_5to3",
            "transcript_accession", "transcript_cdna_start", "transcript_cdna_end", "genomic_chrom",
            "genomic_start_1based", "genomic_end_1based", "target_genomic_strand",
            "chemistry_description", "native_rank", "off_target_exact_count",
            "off_target_exact_count_unit", "off_target_mismatch_budget"])
        writer.writeheader()
        writer.writerow({"candidate": 1, "antisense_5to3": "ACGTACGTACGTACGTACGT",
            "target_5to3": "ACGTACGTACGTACGTACGT", "transcript_accession": "ENST00000000001.2",
            "transcript_cdna_start": 20, "transcript_cdna_end": 39, "genomic_chrom": "chr1",
            "genomic_start_1based": 100, "genomic_end_1based": 119, "target_genomic_strand": "+",
            "chemistry_description": "full-PS gapmer", "native_rank": 1, "off_target_exact_count": 0,
            "off_target_exact_count_unit": "genes", "off_target_mismatch_budget": 0})
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({"rows": [{"synthesized_antisense_sequence": "ACGTACGTACGTACGTACGT",
        "off_target_count_status": "complete_within_declared_scope"}]}))
    tasks = [
        {"task_type": "sandbox_analysis", "label": "Pilot audit", "payload": {
            "original_report_templates": {"audit.md": "# Pilot audit\n3 eligible, 1 indeterminate."}}},
        {"task_type": "design_new_asos", "label": "New designs", "payload": {
            "design_outcome": {"outcome_status": "completed_with_candidates"},
            "generated_artifacts": {"design_panel_csv": str(panel), "generated_aso_set_json": str(generated)}}},
    ]
    report = grounded_mixed_report(tasks, [])
    assert "3 eligible, 1 indeterminate" in report["markdown"]
    assert "| 0 | genes | 0 | complete_within_declared_scope |" in report["markdown"]
    assert "1-mismatch" not in report["markdown"]
