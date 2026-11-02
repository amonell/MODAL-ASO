from copy import deepcopy
import json

import pytest

from aso_agents.screening_delivery import screening_delivery


def screened_row():
    return {"candidate_id": "candidate_1", "sequence": "AACCGG", "synthesized_antisense_sequence": "CCGGTT",
        "off_target_count_status": "complete_within_declared_scope", "off_target_search_status": "complete",
        "Off_target_mismatch_budget": 2, "off_target_exact_count_unit": "genes",
        "Off_target_exact_count": 0, "Off_target_1mm_count": 0, "Off_target_2mm_count": 1,
        "off_target_search_provenance_by_species": {"human": {"assembly": "GRCh38", "genomic_index": "test"}}}


def check(rows, species=("human",), budget=2):
    return screening_delivery(rows, required_species=species, mismatch_budget=budget)


@pytest.mark.parametrize("changes", [
    {"off_target_count_status": "incomplete"}, {"Off_target_2mm_count": None},
    {"off_target_mapping_capped": True}, {"off_target_search_status": "failed"},
    {"Off_target_mismatch_budget": 0}, {"off_target_search_provenance_by_species": {}},
    {"off_target_incomplete_reasons": ["reference_bases_unresolved"]},
    {"off_target_screening_status_by_species": {"human": "mapping_failed"}},
])
def test_each_missing_screening_requirement_blocks_completion(changes):
    row = {**screened_row(), **changes}
    result = check([row])
    assert result["status"] == "incomplete"
    assert result["errors"] == ["selected_off_target_screen_incomplete:candidate_1"]


def test_complete_gene_and_genomic_locus_scopes_can_finish():
    row = screened_row()
    assert check([row])["status"] == "complete"
    row.pop("off_target_count_status")
    row["off_target_exact_count_unit"] = "loci"
    assert check([row])["status"] == "complete"


def test_secondary_species_must_finish_its_own_requested_search():
    row = screened_row()
    row["off_target_search_provenance_by_species"]["mouse"] = {"assembly": "GRCm39"}
    row["off_target_by_species"] = {"human": screened_row(), "mouse": screened_row()}
    assert check([row], ("human", "mouse"))["status"] == "complete"
    row["off_target_by_species"]["mouse"]["off_target_count_status"] = "incomplete"
    assert "species_search_incomplete:mouse" in check([row], ("human", "mouse"))["incomplete_candidates"][0]["reasons"]


def test_candidate_pool_failure_does_not_block_delivered_screened_panel(tmp_path):
    from aso_agents import pipeline_tools as pipeline
    (tmp_path / "goal_intake.json").write_text(json.dumps({"final_payload": {
        "structured_goal": {"intent": "binding", "candidate_minimum": 1}, "tiling_results": {"row_count": 2}}}))
    pool = {**screened_row(), "candidate_id": "unselected", "off_target_mapping_capped": True}
    (tmp_path / "generated_aso_set.json").write_text(json.dumps({"rows": [screened_row()], "pool": [pool]}))
    result = pipeline._extract_pipeline_integrity(str(tmp_path),
        {"exists": {"report_md": True, "report_html": True}}, skip_router=True)
    assert result["ok"]
    assert result["off_target_screening"]["checked_candidates"] == 1
    selected = deepcopy(screened_row())
    selected["Off_target_2mm_count"] = None
    (tmp_path / "generated_aso_set.json").write_text(json.dumps({"rows": [selected]}))
    result = pipeline._extract_pipeline_integrity(str(tmp_path),
        {"exists": {"report_md": True, "report_html": True}}, skip_router=True)
    assert not result["ok"]
    assert "selected_off_target_screen_incomplete:candidate_1" in result["blocking_errors"]


def test_evaluation_only_scope_keeps_explicit_disabled_screening(tmp_path):
    from aso_agents import pipeline_tools as pipeline
    (tmp_path / "goal_intake.json").write_text(json.dumps({"request_envelope": {
        "tasks": [{"task_id": "evaluate", "task_type": "evaluate_existing_asos"}]},
        "final_payload": {"structured_goal": {"off_target_screening": "disabled"}}}))
    result = pipeline._extract_pipeline_integrity(str(tmp_path),
        {"exists": {"report_md": True, "report_html": True}}, skip_router=True)
    assert result["ok"]
