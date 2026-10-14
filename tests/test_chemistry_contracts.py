import json
from types import SimpleNamespace

import pytest

from aso_agents.chemistry_contracts import chemistry_contract_errors
from aso_agents import pipeline_tools


def recommendation(label="steric-hindrance", length=21, rationale="Fully modified steric blocking preserves the RNA substrate."):
    return {"recommendation": {"CHEMISTRY": label,
            "platform": "steric_blocker" if label == "steric-hindrance" else "gapmer",
            "length": length, "rationale": rationale}}


@pytest.mark.parametrize("label,length", [("5-10-5", 20), ("3-10-3", 16), ("steric-hindrance", 21)])
def test_actual_native_architectures_honor_exact_length(label, length):
    assert not chemistry_contract_errors(recommendation(label, length), length)


@pytest.mark.parametrize("payload,length,fragment", [
    (recommendation("5-10-5", 21), 21, "architecture requires"),
    (recommendation(length=20), 21, "exact oligo length"),
    (recommendation(rationale="A 5-10-5 gapmer is retained as a fallback."), 21, "unselected architecture"),
    (recommendation(rationale="The 21 nt floor permits this design."), 21, "lower bound"),
    (recommendation(rationale="Use at least 21 bp for this program."), 21, "lower bound"),
])
def test_incompatible_architecture_or_scope_is_not_delivered(payload, length, fragment):
    assert any(fragment in error for error in chemistry_contract_errors(payload, length))


def test_chemistry_repair_receives_prior_response_and_exact_constraint(monkeypatch):
    initial = recommendation(rationale="A 3-10-3 fallback may be used later.")
    final = recommendation()
    prompts = []
    async def run(_agent, *, input, **_kwargs):
        if getattr(_agent, "name", None) == "Chemistry Mechanism Consistency Check":
            return SimpleNamespace(final_output=json.dumps({"issues": []}))
        prompts.append(json.loads(input))
        return SimpleNamespace(final_output=json.dumps(initial if len(prompts) == 1 else final))
    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    result = pipeline_tools._run_chemistry_agent_for_goal(object(), {"intent": "binding", "aso_length_nt": 21})
    assert result["recommendation"] == final["recommendation"]
    assert len(prompts) == 2
    feedback = prompts[1]["output_contract_feedback"]
    assert json.loads(feedback["previous_response"]) == initial
    assert prompts[1]["exact_requested_length_nt"] == 21
    assert any("unselected architecture" in error for error in feedback["validation_errors"])


def test_incompatible_gapmer_override_cannot_relabel_a_molecule():
    payload = recommendation("3-10-3", 16)
    with pytest.raises(ValueError, match="exactly 21"):
        pipeline_tools._apply_requested_length_to_chemistry_payload(payload, 21)
    assert payload["recommendation"]["length"] == 16
