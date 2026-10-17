"""Conservation evidence is independent of whether it excludes candidates."""
import json

import pytest

from aso_agents import report_explainer as reports
from aso_agents import pipeline_tools


@pytest.mark.parametrize("required", [False, True])
def test_eligibility_pass_count_does_not_become_conservation_evidence(required):
    filters = [{"required": required, "requested_species": ["organism_a", "organism_b"],
                "passing_count": 100, "checked_count": 0}]
    rows = [{"cross_species_pass": True}, {"cross_species_pass": False}, {},
            {"cross_species_pass": None}, {"cross_species_pass": "true"}]
    evidence = reports._cross_species_report_evidence(filters, rows)
    assert evidence == {"active": True, "required": required,
                        "requested_species": ["organism_a", "organism_b"],
                        "total": 5, "exact": 1, "nonmatching": 1, "unknown": 3}


@pytest.mark.parametrize("rows,filters,active", [
    ([{}], [{"required": False, "requested_species": ["organism_a"]}], False),
    ([{}], [{"required": False, "requested_species": ["organism_a", "organism_b"]}], True),
    ([{"cross_species_matches": {"organism_a": ["tx1"], "organism_b": ["tx2"]}}], [], True),
    ([], [{"requested_species": None}], False),
])
def test_unavailable_evidence_is_not_a_negative_or_positive_match(rows, filters, active):
    evidence = reports._cross_species_report_evidence(filters, rows)
    assert evidence["active"] is active
    assert evidence["exact"] == evidence["nonmatching"] == 0
    assert evidence["unknown"] == len(rows)


def test_optional_conservation_report_uses_final_panel_measurements(tmp_path):
    run = tmp_path / "neutral_run"
    candidates = run / "artifacts" / "candidates"
    candidates.mkdir(parents=True)
    (run / "request.txt").write_text("Prioritize exact conservation when available; retain eligible primary-species candidates.")
    scored = candidates / "gapmer_scored_uniqueness.json"
    # Intermediate rows must not inflate evidence counts for the delivered panel.
    scored.write_text(json.dumps({"rows": [{"cross_species_pass": True}] * 8}))
    rows = [{"sequence": "ACGTACGTACGTACGTACGT", "cross_species_pass": True},
            {"sequence": "AACGTACGTACGTACGTACG", "cross_species_pass": False},
            {"sequence": "TACGTACGTACGTACGTACG", "cross_species_pass": None}]
    (candidates / "generated_aso_set.json").write_text(json.dumps({
        "rows": rows, "row_count": 3,
        "eligibility_filters": [{"name": "cross_species_exact_match", "required": False,
                                 "requested_species": ["organism_a", "organism_b"],
                                 "passing_count": 3, "checked_count": 0}]}))
    markdown = reports.render_report_markdown({}, {
        "structured_goal": {"gene_symbol": "GENE_A", "species": "organism_a", "intent": "knockdown"},
        "gapmer_tiling": {"scored_tiles_path": str(scored), "row_count": 8}})
    assert "Conservation evidence for the 3 displayed candidate row(s)" in markdown
    assert "1 have recorded exact matches" in markdown
    assert "1 were checked without a full cross-species match" in markdown
    assert "1 have unavailable or incomplete conservation evidence" in markdown
    assert "Conservation is optional" in markdown
    assert "unavailable species reference does not establish absence" in markdown
    assert "single-species decision surface" not in markdown
    assert "Do not expect mouse-conservation" not in markdown


@pytest.mark.parametrize("flags,expected", [
    ([True, False, None], "1 exact; 1 nonmatching; 1 unknown"),
    ([None, None, None], "0 exact; 0 nonmatching; 3 unknown"),
])
def test_optional_observations_reach_parent_without_becoming_required(tmp_path, flags, expected):
    goal = {"gene_symbol": "GENE_A", "species": "human", "intent": "knockdown",
            "design_programs": [{"program_id": "gapmer_panel", "modality": "gapmer"}],
            "cross_species_requirement": {"required": False, "return_nonconserved_candidates": True,
                                          "species": ["human", "mouse"]}}
    (tmp_path / "goal_intake.json").write_text(json.dumps({"final_payload": {"structured_goal": goal}}))
    path = tmp_path / "generated_aso_set.json"
    path.write_text(json.dumps({"rows": [{"sequence": str(index), "cross_species_pass": flag}
                                         for index, flag in enumerate(flags)],
                               "eligibility_filters": [{"name": "cross_species_exact_match",
                                   "required": False, "requested_species": ["human", "mouse"],
                                   "passing_count": 3, "checked_count": 0}]}))
    outcome = pipeline_tools._classify_design_outcome(
        child_run_dir=str(tmp_path), task_context={}, child_status="run_complete",
        generated_artifacts={"generated_aso_set_json": str(path)})
    assert outcome["outcome_status"] == "completed_with_candidates"
    assert outcome["required_evidence"] == outcome["missing_required_evidence"] == []
    response = {"deterministic_summary": {"design_outcomes": [{
        "gene": "GENE_A", "child_run_dir": str(tmp_path), "design_outcome": outcome}]}}
    rows = pipeline_tools._summary_design_outcomes(response, str(tmp_path))
    table = "\n".join(pipeline_tools._summary_design_status_table(rows))
    assert expected in table
    assert "human, mouse" in table
    assert "Unknown or missing species evidence does not establish nonconservation" in table
