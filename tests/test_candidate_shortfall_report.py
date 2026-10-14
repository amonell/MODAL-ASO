from aso_agents.report_explainer import _candidate_shortfall_lines


def test_shortfall_reports_actual_panel_deficit_and_bounded_funnel():
    trace = {"rows": (
        [{"first_drop_stage": "initial_filter", "stage_results": {}}] * 2
        + [{"first_drop_stage": "final_output",
            "stage_results": {"routing_eligibility": {"status": "fail"}}}] * 5
        + [{"first_drop_stage": "",
            "stage_results": {"routing_eligibility": {"status": "pass"}}}] * 3)}
    lines = _candidate_shortfall_lines(
        {"candidate_minimum": 4},
        {"distinct_molecule_count": 3, "unique_sequence_count": 3},
        trace, [{"count": 3}, {"count": 2}])
    text = "\n".join(lines)
    assert "3 distinct delivered" in text
    assert "minimum of 4; deficit 1" in text
    assert "10 candidate sequence" in text
    assert "2 failed initial filters" in text
    assert "5 passed to scoring but were not selected" in text
    assert "5 lacked recorded route eligibility" in text
    assert "3, 2" in text and "must not be summed" in text
    assert "not evidence that no other viable ASOs exist" in text


def test_met_or_unverifiable_minimum_does_not_invent_a_shortfall():
    assert not _candidate_shortfall_lines(
        {"candidate_minimum": 3}, {"distinct_molecule_count": 3}, {}, [])
    assert not _candidate_shortfall_lines(
        {"candidate_minimum": 4}, {"distinct_molecule_count": 4}, {}, [])
    assert not _candidate_shortfall_lines(
        {"candidate_minimum": 4}, {"rows": []}, {}, [])
