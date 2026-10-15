from copy import deepcopy

import pytest

import run_goal_intake as runtime


def relative_window(**identity):
    return {
        "source_type": "splice_relative_window",
        "aso_overlap_requirement": "full_containment",
        "splice_relative_window": {
            "anchor_site": "donor", "relative_start_nt": 0,
            "relative_end_nt": 27, **identity,
        },
    }


def test_relative_window_inherits_only_its_containing_program_exon():
    window = relative_window()
    original = deepcopy(window)
    goal = {"exon_label": 9, "design_programs": [
        {"program_id": "first", "exon_label": 2, "region_constraints": [window]},
        {"program_id": "second", "exon_labels": [6], "region_constraints": [window]},
    ]}
    runtime._normalize_region_constraints_in_goal(goal)
    observed = [program["region_constraints"][0]["splice_relative_window"]
                for program in goal["design_programs"]]
    assert [item["exon_label"] for item in observed] == [2, 6]
    assert all(item["relative_start_nt"] == 0 and item["relative_end_nt"] == 27
               and item["anchor_site"] == "donor" for item in observed)
    assert window == original
    assert not runtime._site_constraint_coordinates_required(goal)


@pytest.mark.parametrize("identity", [
    {"exon_label": 7}, {"junction_exon_label": 7}, {"target_exon": 7},
])
def test_existing_window_identity_is_not_overwritten(identity):
    normalized = runtime._normalize_region_constraints(
        [relative_window(**identity)], context={"exon_label": 2})
    assert normalized[0]["splice_relative_window"]["exon_label"] == 7


@pytest.mark.parametrize("context", [
    {}, {"exon_labels": [2, 6]}, {"exon_label": 2, "exon_labels": [2, 6]},
    {"exon_labels": [2, "unresolved"]},
])
def test_missing_or_ambiguous_exon_is_not_guessed(context):
    assert runtime._normalize_region_constraints([relative_window()], context=context) == []
    goal = {**context, "region_constraints": [relative_window()]}
    assert runtime._site_constraint_coordinates_required(goal)


def test_outcome_identity_alone_never_creates_a_binding_window():
    goal = {"exon_label": 4, "desired_effect": "inclusion",
            "design_programs": [{"exon_label": 4, "targeting_policy": "default_exploratory"}]}
    runtime._normalize_region_constraints_in_goal(goal)
    assert "region_constraints" not in goal
    assert "region_constraints" not in goal["design_programs"][0]


def test_normalized_top_level_window_survives_executable_retrieval():
    goal = {"exon_label": 3, "region_constraints": [relative_window()]}
    runtime._normalize_region_constraints_in_goal(goal)
    assert runtime._goal_region_constraints(goal)[0]["splice_relative_window"]["exon_label"] == 3
    assert runtime._program_region_constraints(goal)[0]["splice_relative_window"]["exon_label"] == 3
