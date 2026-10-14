from copy import deepcopy

import run_goal_intake as intake


def _program(number, **changes):
    program = {
        "program_id": f"candidate_{number}",
        "label": f"Candidate {number}",
        "modality": "gapmer",
        "intent": "knockdown",
        "objective": "Find a conserved transcript-targeting panel",
        "length": 20,
        "chemistry": "5-10-5 MOE gapmer",
        "candidate_limit": 5,
    }
    program.update(changes)
    return program


def test_candidate_slots_do_not_repeat_the_same_full_design_search():
    goal = {"candidate_limit": 5, "design_programs": [_program(i) for i in range(1, 6)]}
    intake._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 1
    assert goal["design_programs"][0]["candidate_limit"] == 5
    intake._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 1


def test_equivalent_programs_do_not_split_a_larger_distinct_molecule_panel():
    programs = [_program(1), _program(2)]
    for program in programs:
        program["candidate_limit"] = 6
        program["objective"] = "Design 4-6 distinct mature-transcript gapmers"
    goal = {
        "candidate_minimum": 4,
        "candidate_limit": 6,
        "candidate_limit_per_site": 6,
        "min_asos_per_site": 4,
        "design_programs": programs,
    }
    intake._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 1
    assert goal["design_programs"][0]["candidate_limit"] == 6


def test_candidate_slot_collapse_reaches_scoped_design_task():
    goal = {"candidate_limit": 5, "design_programs": [_program(i) for i in range(1, 6)]}
    intake_output = {
        "structured_goal": deepcopy(goal),
        "request_envelope": {
            "tasks": [{"task_id": "design", "task_type": "design_new_asos",
                       "context": {"structured_goal": deepcopy(goal)}}],
        },
    }
    normalized = intake._normalize_intake_output(intake_output)
    assert len(normalized["structured_goal"]["design_programs"]) == 1
    task_goal = normalized["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert len(task_goal["design_programs"]) == 1


def test_distinct_program_scopes_and_explicit_panels_remain_separate():
    original = [_program(i) for i in range(1, 6)]
    distinct_site = deepcopy(original)
    distinct_site[2]["target_region"] = "utr3"
    distinct_chemistry = deepcopy(original)
    distinct_chemistry[2]["chemistry"] = "fully modified steric chemistry"
    independent = deepcopy(original)
    independent[2]["independent_panel"] = True
    for programs in (distinct_site, distinct_chemistry, independent):
        goal = {"candidate_limit": 5, "design_programs": programs}
        intake._normalize_design_programs_in_goal(goal)
        assert len(goal["design_programs"]) == 5


def test_named_programs_without_explicit_panel_size_are_preserved():
    goal = {"design_programs": [_program(1), _program(2)]}
    intake._normalize_design_programs_in_goal(goal)
    assert [program["program_id"] for program in goal["design_programs"]] == [
        "candidate_1", "candidate_2"
    ]


def test_referenced_program_identity_is_not_dropped():
    goal = {
        "candidate_limit": 5,
        "design_programs": [_program(i) for i in range(1, 6)],
        "evaluation_targets": [{"program_id": "candidate_2", "target_type": "transcript"}],
    }
    intake._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 5


def test_candidate_slot_constraint_ids_do_not_multiply_a_scientific_search():
    programs = []
    for index in range(1, 4):
        programs.append({
            "program_id": f"slot_{index}", "label": f"ASO {index}",
            "modality": "steric", "intent": "splice_modulation",
            "objective": "Increase inclusion at the same annotated exon",
            "region_constraints": [{
                "constraint_id": f"slot_region_{index}", "label": "same exon",
                "source_type": "exon", "exon_label": 9,
                "aso_overlap_requirement": "any_overlap",
            }],
        })
    goal = {"candidate_minimum": 3, "design_programs": programs}
    intake._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 1
    assert goal["design_programs"][0]["region_constraints"][0]["constraint_id"] == "slot_region_1"

    distinct = deepcopy(programs)
    distinct[1]["region_constraints"][0]["exon_label"] = 10
    goal = {"candidate_minimum": 3, "design_programs": distinct}
    intake._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 3

    referenced = {"candidate_minimum": 3, "design_programs": programs,
                  "evaluation_targets": [{"constraint_id": "slot_region_2"}]}
    intake._normalize_design_programs_in_goal(referenced)
    assert len(referenced["design_programs"]) == 3


def test_repeated_panel_notes_collapse_only_when_additive():
    programs = [_program(i) for i in range(1, 4)]
    for program in programs:
        program["candidate_limit"] = 3
        program["notes"] = "Fully phosphorothioate steric chemistry."
        program["region_constraints"] = [{"constraint_id": program["program_id"],
                                          "source_type": "exon", "exon_label": 9}]
    programs[0]["notes"] += " Search the same annotated exon."
    goal = {"candidate_limit": 3, "design_programs": deepcopy(programs)}
    intake._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 1
    assert goal["design_programs"][0]["notes"] == programs[0]["notes"]

    independent = deepcopy(programs)
    independent[1]["notes"] = "Search a distinct donor site."
    goal = {"candidate_limit": 3, "design_programs": independent}
    intake._normalize_design_programs_in_goal(goal)
    assert len(goal["design_programs"]) == 3
