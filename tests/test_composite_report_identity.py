from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline


def design(label, gene, modality, count, objective):
    outcome = {"outcome_status": "completed_with_candidates"}
    if count is not None:
        outcome.update(candidate_count=count, unique_sequence_count=count)
    return {"task_id": label, "task_type": "design_new_asos", "label": label,
            "payload": {"task_id": label, "task_label": label,
                        "child_run_dir": f"/results/{label}",
                        "child_structured_goal": {
                            "gene_symbol": gene, "species": "example_species",
                            "transcript_lock": {"value": f"reference_{gene}"},
                            "design_programs": [{"label": label, "modality": modality,
                                                 "objective": objective}]},
                        "design_outcome": outcome}}


def comparison(label, text):
    return {"task_id": label, "task_type": "compare_aso_sets", "label": label,
            "payload": {"comparison_assessment": {
                "markdown": text, "assumptions": [f"assumption_{label}"],
                "limitations": [f"limitation_{label}"],
                "evidence_refs": [f"/results/{label}.csv"]}}}


@pytest.mark.parametrize("genes", [("GENE_A", "GENE_B"), ("GENE_C", "GENE_C"),
                                  ("Target1", "Target2")])
@pytest.mark.parametrize("modalities", [("gapmer", "steric"), ("steric", "gapmer"),
                                       ("steric", "steric"), ("novel", "gapmer")])
def test_identity_and_objectives_come_from_each_task_not_modality_pair(genes, modalities):
    tasks = [design("first", genes[0], modalities[0], 2, "Decrease exon 3 inclusion"),
             design("second", genes[1], modalities[1], 11, "Block a translated regulatory element"),
             comparison("contrast", "Endpoints differ; no shared potency ordering is established.")]
    original = deepcopy(tasks)
    response = pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks)
    text = response["markdown"]
    for expected in [*genes, "Decrease exon 3 inclusion", "Block a translated regulatory element",
                     "Selected candidate rows: 2", "Selected candidate rows: 11",
                     "Endpoints differ; no shared potency ordering is established."]:
        assert expected in text
    for unsupported in ["MAPT", "exon-10", "three provided", "broader candidate", "Immediate wet-lab"]:
        assert unsupported not in text
    assert tasks == original


def test_many_programs_and_many_comparisons_are_never_coalesced_by_mechanism():
    tasks = [design(f"program_{i}", f"TARGET_{i}", "steric", i + 1, f"objective_{i}")
             for i in range(5)]
    tasks += [comparison(f"comparison_{i}", f"independent_assessment_{i}") for i in range(4)]
    response = pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks)
    for i in range(5):
        assert f"TARGET_{i}" in response["markdown"]
        assert f"objective_{i}" in response["markdown"]
    for i in range(4):
        assert f"independent_assessment_{i}" in response["markdown"]
        assert f"limitation_comparison_{i}" in response["limitations"]
    assert response["deterministic_summary"]["design_task_count"] == 5


def test_absent_counts_are_distinct_from_recorded_zero_and_missing_input_is_not_fabricated():
    tasks = [design("unknown", "UNKNOWN", "gapmer", None, "objective_unknown"),
             design("empty", "EMPTY", "steric", 0, "objective_empty"),
             comparison("contrast", "Uncertainty is retained.")]
    response = pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks)
    first = response["markdown"].split("### unknown", 1)[1].split("### empty", 1)[0]
    assert "Selected candidate rows" not in first
    assert "Selected candidate rows: 0" in response["markdown"]
    assert "provided ASO" not in response["markdown"]


def test_every_evaluation_is_retained_without_claiming_generated_molecules_were_user_supplied():
    tasks = [comparison("contrast", "Distinct results remain distinct.")]
    for i in range(3):
        tasks.append({"task_id": f"evaluation_{i}", "task_type": "evaluate_existing_aso_set",
                      "payload": {"provided_asos": [{"name": f"generated_{i}", "sequence": "ACGT",
                                                     "mapping_status": "mapped"}]}})
    response = pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks)
    for i in range(3):
        assert f"generated_{i}" in response["markdown"]
    assert "user-provided" not in response["markdown"]
    assert response["deterministic_summary"]["evaluation_task_count"] == 3


def test_recorded_comparative_assessment_is_preserved_without_new_potency_preference():
    text = "Steric arm has stronger evidence for its distinct endpoint. Experimental potency is unresolved."
    tasks = [design("gapmer", "GENE_A", "gapmer", 30, "knockdown"),
             design("steric", "GENE_A", "steric", 2, "exon exclusion"),
             comparison("contrast", text)]
    response = pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks)
    assert text in response["markdown"]
    assert "gapmer prioritization" not in response["markdown"]
    assert "retun" not in response["markdown"]


def test_unsupported_mixed_results_fall_through_instead_of_disappearing():
    tasks = [comparison("contrast", "Recorded comparison."),
             {"task_id": "other", "task_type": "analyze_artifact",
              "payload": {"markdown": "Requested additional result."}}]
    assert pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks) is None


def test_missing_trace_evidence_is_not_replaced_by_successful_design_claims():
    tasks = [comparison("contrast", "Recorded comparison."),
             {"task_id": "trace", "task_type": "trace_filter_decisions", "payload": {}}]
    assert pipeline._build_deterministic_composite_response({}, {}, completed_tasks=tasks) is None
