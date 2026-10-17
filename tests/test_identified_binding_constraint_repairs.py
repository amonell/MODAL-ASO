"""A correction must replace its identified predicate, not intersect with stale state."""
from copy import deepcopy

import pytest

import run_goal_intake as intake
from aso_agents.region_overlap import matches_constraint_filters
from aso_agents import pipeline_tools as pipeline


def constraint(identity, start, end, *, operator="intersect", minimum=1, relation="any_overlap"):
    return {"constraint_id": identity, "source_type": "explicit_rows",
            "coordinate_space": "genomic", "operator": operator,
            "aso_overlap_requirement": relation, "minimum_overlap_bp": minimum,
            "intervals_bed0_half_open": [{"chrom": "chr8", "start": start, "end": end}]}


def draft(*rules):
    return {"status": "ready", "structured_goal": {"intent": "binding",
            "region_constraints": list(rules)}}


def repair(raw, *rules):
    return intake._merge_binding_constraint_repair(raw, {
        "status": "updated", "structured_goal_patch": {"region_constraints": list(rules)}})


def filters(result):
    return [{**rule, "rows": [[r["chrom"], r["start"], r["end"]]
                              for r in rule["intervals_bed0_half_open"]]}
            for rule in result["structured_goal"]["region_constraints"]]


@pytest.mark.parametrize("maximum", [1, 2, 5])
def test_repaired_exclusion_keeps_permitted_partial_overlaps_and_rejects_the_boundary(maximum):
    raw = draft(constraint("window", 1, 80, relation="full_containment"),
                constraint("upper", 20, 28, operator="exclude"))
    original = deepcopy(raw)
    # Existing canonical execution represents overlap <= M as NOT overlap >= M+1.
    fixed = repair(raw, {"constraint_id": "upper", "aso_overlap_requirement": "any_overlap",
                         "minimum_overlap_bp": maximum + 1})
    rules = fixed["structured_goal"]["region_constraints"]
    assert [r["constraint_id"] for r in rules] == ["window", "upper"]
    assert rules[1]["operator"] == "exclude" and rules[1]["coordinate_space"] == "genomic"
    assert rules[1]["intervals_bed0_half_open"] == original["structured_goal"]["region_constraints"][1]["intervals_bed0_half_open"]
    compiled = filters(fixed)
    assert [matches_constraint_filters([("chr8", 8 + n, 20 + n)], compiled)
            for n in range(maximum + 2)] == [True] * (maximum + 1) + [False]
    assert raw == original
    assert repair(fixed, {"constraint_id": "upper", "aso_overlap_requirement": "any_overlap",
                          "minimum_overlap_bp": maximum + 1}) == fixed


def test_two_predicates_at_the_same_feature_remain_distinct_and_in_order():
    lower = constraint("lower", 20, 28, minimum=2)
    upper = constraint("upper", 20, 28, operator="exclude", minimum=4)
    fixed = repair(draft(lower, upper), {**upper, "minimum_overlap_bp": 6})
    rules = fixed["structured_goal"]["region_constraints"]
    assert [(r["constraint_id"], r["minimum_overlap_bp"]) for r in rules] == [("lower", 2), ("upper", 6)]
    assert [matches_constraint_filters([("chr8", 8 + n, 20 + n)], filters(fixed))
            for n in range(7)] == [False, False, True, True, True, True, False]


def test_new_id_is_appended_without_merging_same_geometry_or_changing_existing_rules():
    first = constraint("lower", 20, 28, minimum=2)
    added = constraint("upper", 20, 28, operator="exclude", minimum=6)
    fixed = repair(draft(first), added)
    assert [r["constraint_id"] for r in fixed["structured_goal"]["region_constraints"]] == ["lower", "upper"]
    assert [matches_constraint_filters([("chr8", 8 + n, 20 + n)], filters(fixed))
            for n in range(7)] == [False, False, True, True, True, True, False]


def test_unidentified_new_rule_does_not_guess_which_existing_predicate_to_replace():
    first = constraint("lower", 20, 28, minimum=2)
    unnamed = constraint("temporary", 20, 28, operator="exclude", minimum=6)
    unnamed.pop("constraint_id")
    fixed = repair(draft(first), unnamed)
    assert len(fixed["structured_goal"]["region_constraints"]) == 2
    assert fixed["structured_goal"]["region_constraints"][0]["constraint_id"] == "lower"


@pytest.mark.parametrize("ambiguous_in", ["existing", "patch"])
def test_ambiguous_target_id_is_rejected_without_mutating_original(ambiguous_in):
    first = constraint("same", 20, 28, operator="exclude")
    changed = {**first, "minimum_overlap_bp": 3}
    raw = draft(first, {**first, "label": "conflicting identity"}) if ambiguous_in == "existing" else draft(first)
    original = deepcopy(raw)
    patches = [changed] if ambiguous_in == "existing" else [changed, {**changed, "minimum_overlap_bp": 5}]
    with pytest.raises(ValueError, match="ambiguous.*constraint_id"):
        repair(raw, *patches)
    assert raw == original


def test_mixed_explicit_predicates_survive_normalization_compilation_and_pool_filter(tmp_path):
    raw = draft(constraint("window", 1, 80, relation="full_containment"),
                constraint("core", 12, 15, relation="contains_feature"),
                constraint("upper", 20, 28, operator="exclude"))
    fixed = repair(raw, {"constraint_id": "upper", "aso_overlap_requirement": "any_overlap",
                         "minimum_overlap_bp": 4})
    goal = fixed["structured_goal"]
    intake._enforce_region_constraint_overlap_requirement(goal)
    assert "aso_overlap_requirement" not in goal
    assert [r["aso_overlap_requirement"] for r in goal["region_constraints"]] == [
        "full_containment", "contains_feature", "any_overlap"]
    goal["materialized_region_constraints"] = {"global": [
        {**r, "materialized_rows": [[v["chrom"], v["start"], v["end"]]
                                    for v in r["intervals_bed0_half_open"]]}
        for r in goal["region_constraints"]]}
    _, compiled = pipeline._apply_shared_region_constraints(
        runtime=intake, structured_goal=goal, program={"length": 12},
        base_rows=[["chr8", 1, 80]], output_space="genomic",
        base_overlap_requirement="full_containment")
    rows = [{"id": str(n), "Chrom": "chr8", "Start": 9 + n, "End": 20 + n,
             "coordinate_system": "1based_inclusive"} for n in range(7)]
    rows += [{"id": "outside_window", "Chrom": "chr8", "Start": 1, "End": 12,
              "coordinate_system": "1based_inclusive"}]
    source, destination = tmp_path / "source.json", tmp_path / "selected.json"
    source.write_text(intake.json.dumps({"rows": rows}))
    outcome = pipeline._filter_tiles_json_to_output(
        source_json_path=str(source), output_path=str(destination),
        region_rows=compiled["tile_filter_rows"],
        tile_overlap_requirement=compiled["tile_overlap_requirement"],
        constraint_filters=compiled["tile_constraint_filters"], allow_empty_subset=True)
    assert "error" not in outcome
    assert [r["id"] for r in intake.json.loads(destination.read_text())["rows"]] == ["0", "1", "2", "3"]


def test_missing_overlap_requirement_still_needs_clarification():
    missing = constraint("unknown", 20, 28)
    missing.pop("aso_overlap_requirement")
    raw = draft(constraint("window", 1, 80, relation="full_containment"), missing)
    assert intake._apply_region_overlap_requirement_guard(deepcopy(raw))["status"] == "need_clarification"
    with pytest.raises(ValueError, match="requires aso_overlap_requirement"):
        intake._enforce_region_constraint_overlap_requirement(raw["structured_goal"])
