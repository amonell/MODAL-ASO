from copy import deepcopy

import jsonschema
import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.goal_contract import canonical_structured_goal_schema, canonical_task_context_schema
from aso_agents.tiling_agent import _resolve_num_mismatches


def design_goal(budget):
    return {"num_mismatches": budget, "design_programs": [{
        "program_id": "neutral", "label": "Neutral panel", "modality": "gapmer",
        "intent": "knockdown", "objective": "Evaluate the requested knockdown panel"}]}


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4])
def test_canonical_budget_survives_normalization_and_native_evaluation_handoff(budget):
    goal = design_goal(budget)
    jsonschema.validate(goal, canonical_structured_goal_schema())
    context = {"num_mismatches": budget}
    jsonschema.validate(context, canonical_task_context_schema())
    normalized = pipeline._normalized_design_objective_preseed(deepcopy(goal))
    assert normalized["structured_goal_overlay"]["num_mismatches"] == budget
    defaults = pipeline._structured_goal_default_context(goal)
    assert defaults["num_mismatches"] == budget
    task = {"task_type": "evaluate_existing_aso_set", "context": context}
    scoped = pipeline._task_context_from_request_envelope({"default_context": {"num_mismatches": 3}}, task)
    assert scoped["num_mismatches"] == budget
    executable = pipeline._overlay_request_context_on_goal({"num_mismatches": 2}, scoped, "")
    assert executable["num_mismatches"] == budget
    if budget <= 3:
        assert _resolve_num_mismatches(executable, default=0) == budget
    else:
        with pytest.raises(ValueError, match="at most 3"):
            _resolve_num_mismatches(executable, default=0)


@pytest.mark.parametrize("value", [-1, True, 1.2, "1", None])
def test_budget_is_a_nonnegative_integer_in_both_public_contracts(value):
    for schema, payload in [(canonical_structured_goal_schema(), design_goal(value)),
                            (canonical_task_context_schema(), {"num_mismatches": value})]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema)


def test_distinct_task_budgets_do_not_bleed_into_each_other():
    envelope = {"default_context": {"num_mismatches": 2}}
    exact = pipeline._task_context_from_request_envelope(envelope, {"context": {"num_mismatches": 0}})
    near = pipeline._task_context_from_request_envelope(envelope, {"context": {"num_mismatches": 1}})
    inherited = pipeline._task_context_from_request_envelope(envelope, {"context": {}})
    assert [c["num_mismatches"] for c in [exact, near, inherited]] == [0, 1, 2]
    assert envelope["default_context"]["num_mismatches"] == 2


def test_design_context_budget_is_inherited_without_changing_hit_count_filter():
    original = {"constraints": {"max_binding_sites": 7}}
    context = {"gene_symbol": "NEUTRAL", "num_mismatches": 1}
    preseed = pipeline._prepare_design_objective_preseed(original, context)
    assert preseed["structured_goal_overlay"]["num_mismatches"] == 1
    assert preseed["structured_goal_overlay"]["constraints"]["max_binding_sites"] == 7
    assert "num_mismatches" not in original
    assert "num_mismatches" not in pipeline._structured_goal_default_context({})
    assert "num_mismatches" not in pipeline._overlay_request_context_on_goal({}, {}, "")

