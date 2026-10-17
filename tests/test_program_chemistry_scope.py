"""Chemical choices stay bound to their producing program across stage reentry."""
from copy import deepcopy

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline


@pytest.mark.parametrize("order", [("steric", "gapmer"), ("gapmer", "steric")])
def test_branch_selection_does_not_replace_sibling_chemistry_input(monkeypatch, order):
    programs = [
        {"program_id": "cleavage", "modality": "gapmer", "intent": "knockdown",
         "objective": "Use an RNase H-active architecture.", "length": 20,
         "notes": "Retain the requested chemical partition."},
        {"program_id": "occupancy", "modality": "steric", "intent": "binding",
         "objective": "Use an RNase H-silent architecture.", "length": 20},
    ]
    goal = {"chemistry": "phosphorothioate", "design_programs": deepcopy(programs)}
    final = {modality + "_design_jobs": [{"job_id": program["program_id"],
              "program_id": program["program_id"]}]
             for modality, program in [("gapmer", programs[0]), ("steric", programs[1])]}
    calls = []

    def select(_agent, scoped_goal):
        calls.append(deepcopy(scoped_goal))
        assert scoped_goal["chemistry"] == "phosphorothioate"
        assert "design_programs" not in scoped_goal
        label = "5-10-5" if scoped_goal["modality"] == "gapmer" else "steric-hindrance"
        return pipeline._chemistry_payload_from_label(label, source="chemistry_agent")

    monkeypatch.setattr(pipeline, "_run_chemistry_agent_for_goal", select)
    for _ in range(2):
        for modality in order:
            profiles = pipeline._ensure_program_chemistry_profiles(
                structured_goal=goal, final_payload=final, modality=modality)
    assert len(calls) == 2
    for program in programs:
        received = next(g for g in calls if g["program_id"] == program["program_id"])
        assert received["objective"] == program["objective"]
        assert received.get("notes") == program.get("notes")
        copied = runtime._design_program_goal_copy(goal, program)
        assert copied["chemistry"] == "phosphorothioate"
    assert profiles["by_program"]["cleavage"]["recommendation"]["CHEMISTRY"] == "5-10-5"
    assert profiles["by_program"]["occupancy"]["recommendation"]["CHEMISTRY"] == "steric-hindrance"
    assert final["gapmer_design_jobs"][0]["chemistry_label"] == "5-10-5"
    assert final["steric_design_jobs"][0]["chemistry_label"] == "steric-hindrance"


def test_new_program_of_same_modality_does_not_inherit_generated_choice(monkeypatch):
    goal = {"design_programs": [{"program_id": "first", "modality": "gapmer", "length": 18}]}
    final, calls = {}, []

    def select(_agent, scoped_goal):
        calls.append(deepcopy(scoped_goal))
        assert "chemistry" not in scoped_goal
        return pipeline._chemistry_payload_from_label(
            "4-10-4" if scoped_goal["length"] == 18 else "5-10-5", source="chemistry_agent")

    monkeypatch.setattr(pipeline, "_run_chemistry_agent_for_goal", select)
    pipeline._ensure_program_chemistry_profiles(structured_goal=goal, final_payload=final)
    goal["design_programs"].append({"program_id": "second", "modality": "gapmer", "length": 20})
    profiles = pipeline._ensure_program_chemistry_profiles(structured_goal=goal, final_payload=final)
    assert len(calls) == 2
    assert profiles["by_program"]["first"]["recommendation"]["CHEMISTRY"] == "4-10-4"
    assert profiles["by_program"]["second"]["recommendation"]["CHEMISTRY"] == "5-10-5"
    assert "chemistry" not in runtime._design_program_goal_copy(goal, goal["design_programs"][1])


def test_explicit_global_and_local_inputs_remain_authoritative(monkeypatch):
    monkeypatch.setattr(pipeline, "_run_chemistry_agent_for_goal",
                        lambda *_: pytest.fail("Configured chemistry should not be regenerated"))
    goal = {"chemistry": "4-10-4", "design_programs": [
        {"program_id": "inherited", "modality": "gapmer", "length": 18},
        {"program_id": "local", "modality": "gapmer", "length": 20, "chemistry": "5-10-5"}]}
    final = {}
    for _ in range(2):
        profiles = pipeline._ensure_program_chemistry_profiles(structured_goal=goal, final_payload=final)
        assert profiles["by_program"]["inherited"]["recommendation"]["CHEMISTRY"] == "4-10-4"
        assert profiles["by_program"]["local"]["recommendation"]["CHEMISTRY"] == "5-10-5"
    goal["chemistry"] = "3-10-3"
    goal["design_programs"][0]["length"] = 16
    profiles = pipeline._ensure_program_chemistry_profiles(structured_goal=goal, final_payload=final)
    assert profiles["by_program"]["inherited"]["recommendation"]["CHEMISTRY"] == "3-10-3"
    assert profiles["by_program"]["local"]["recommendation"]["CHEMISTRY"] == "5-10-5"


def test_program_copy_preserves_changed_explicit_input_and_does_not_mutate_source():
    goal = {"chemistry": "3-10-3", "chemistry_profiles": {
        "goal_chemistry_input": "phosphorothioate", "goal_chemistry_summary": "steric-hindrance"}}
    before = deepcopy(goal)
    copied = runtime._design_program_goal_copy(goal, {"program_id": "new", "modality": "gapmer"})
    assert copied["chemistry"] == "3-10-3" and goal == before


@pytest.mark.parametrize("reference, expected_count", [
    ({"transcript_id": "TX.2"}, 0),
    ({"isoform_label": "reduce"}, 0),
    ({"transcript_id": "TX.2", "isoform_label": "reduce"}, 0),
    ({"transcript_id": "TX.3"}, 1),
    ({"transcript_id": "OTHER.2"}, 1),
    ({"transcript_id": "OTHER.2", "isoform_label": "reduce"}, 1),
    ({"target_type": "transcript_cohort", "transcript_ids": ["TX.2", "TY.4"]}, 0),
    ({"target_type": "transcript_cohort", "transcript_ids": ["TX.3", "TY.4"]}, 1),
])
def test_support_program_recognizes_existing_reference_without_collapsing_versions(reference, expected_count):
    goal = {"intent": "splice_modulation", "splice_isoform_objectives": [
        {"isoform_label": "reduce", "desired_abundance": "decrease", "role": "pathogenic"},
        {"isoform_label": "protect", "desired_abundance": "increase", "role": "beneficial"}],
        "site_direction_hints": [{"isoform_label": "reduce", "desired_delta": "decrease"}],
        "isoform_transcripts": {"reduce": {"transcript_id": "TX", "metadata": {"versioned_id": "TX.2"}},
                               "protect": {"transcript_id": "TY.4"}},
        "design_programs": [{"program_id": "requested", "modality": "gapmer", **reference}]}
    before = deepcopy(goal)
    added, plan = pipeline._synthesize_splice_switch_gapmer_programs(goal, goal["design_programs"])
    assert plan["enabled"]
    assert len(added) == expected_count and plan["synthesized_program_count"] == expected_count
    assert goal == before
