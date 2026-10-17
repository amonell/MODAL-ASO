import run_goal_intake as intake


def _missing_feature_goal():
    return {
        "status": "ready",
        "structured_goal": {
            "gene_symbol": "EXAMPLE",
            "species": "human",
            "intent": "binding",
            "transcript_policy": "agent_choice",
            "design_programs": [{
                "program_id": "single_feature",
                "modality": "steric",
                "intent": "binding",
                "targeting_policy": "explicit_user_region",
                "length": 20,
            }],
        },
    }


def test_explicit_binding_feature_without_executable_region_requires_repair():
    violations = intake._find_goal_schema_key_violations(_missing_feature_goal())
    assert len(violations) == 1
    assert violations[0]["category"] == "binding_feature_region"


def test_region_repair_preserves_exact_numeric_window_in_single_program():
    raw = _missing_feature_goal()
    repaired = intake._merge_binding_constraint_repair(raw, {
        "status": "updated",
        "structured_goal_patch": {
            "target_region": "translation_start",
            "flanks": {"up": 30, "down": 15},
        },
    })
    goal = repaired["structured_goal"]
    assert goal["target_region"] == "translation_start"
    assert goal["flanks"] == {"up": 30, "down": 15}
    assert goal["design_programs"][0]["target_region"] == "translation_start"
    assert goal["design_programs"][0]["flanks"] == {"up": 30, "down": 15}
    assert not intake._find_goal_schema_key_violations(repaired)


def test_region_repair_does_not_copy_scope_into_distinct_programs():
    raw = _missing_feature_goal()
    raw["structured_goal"]["design_programs"].append({
        "program_id": "different_region",
        "modality": "steric",
        "intent": "binding",
        "targeting_policy": "explicit_user_region",
    })
    repaired = intake._merge_binding_constraint_repair(raw, {
        "status": "updated",
        "structured_goal_patch": {"target_region": "translation_start", "flanks": {"up": 30, "down": 15}},
    })
    for program in repaired["structured_goal"]["design_programs"]:
        assert "target_region" not in program
        assert "flanks" not in program


def test_exploratory_and_concrete_regions_are_not_false_positives():
    exploratory = _missing_feature_goal()
    exploratory["structured_goal"]["design_programs"][0]["targeting_policy"] = "default_exploratory"
    assert not intake._find_goal_schema_key_violations(exploratory)
    concrete = _missing_feature_goal()
    concrete["structured_goal"]["design_programs"][0]["target_region"] = "utr3"
    assert not intake._find_goal_schema_key_violations(concrete)


def test_translation_start_transcript_choice_is_respected():
    goal = _missing_feature_goal()["structured_goal"]
    goal["target_region"] = "translation_start"
    assert intake._translation_start_needs_isoform(goal) is False
    goal.pop("transcript_policy")
    assert intake._translation_start_needs_isoform(goal) is True


def test_typed_feature_window_wrapper_is_canonicalized_without_losing_bounds():
    raw = _missing_feature_goal()
    program = raw["structured_goal"]["design_programs"][0]
    program["target_region"] = {
        "feature": "translation_start",
        "upstream_nt": 30,
        "downstream_nt": 15,
        "inclusive": True,
    }
    program["flanks"] = {"upstream_nt": 30, "downstream_nt": 15, "inclusive": True}
    assert not intake._find_goal_schema_key_violations(raw)
    normalized = intake._normalize_intake_output(raw)
    recovered = normalized["structured_goal"]["design_programs"][0]
    assert recovered["target_region"] == "translation_start"
    assert recovered["flanks"]["up"] == 30
    assert recovered["flanks"]["down"] == 15


def test_feature_wrapper_alone_supplies_nonnegative_window_bounds():
    raw = _missing_feature_goal()
    program = raw["structured_goal"]["design_programs"][0]
    program["target_region"] = {"feature": "translation_start", "upstream_nt": 0, "downstream_nt": 15}
    normalized = intake._normalize_intake_output(raw)
    recovered = normalized["structured_goal"]["design_programs"][0]
    assert recovered["flanks"] == {"up": 0, "down": 15}
