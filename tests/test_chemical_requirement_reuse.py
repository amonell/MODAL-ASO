"""Repeated source recording does not repeat or replace molecule selection."""
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from agents import ModelSettings
from aso_agents import chemical_requirements, pipeline_tools
from aso_agents.chemistry_agent import chemistry_agent
from aso_agents.chemistry_contracts import ChemistryResolutionError


TEXT = "Use phosphorothioate backbone; choose the other chemical details for this program."


def goal(length=20, modality="steric", text=TEXT):
    return {"intent": "binding" if modality == "steric" else "knockdown",
            "modality": modality, "aso_length_nt": length, "chemistry": text}


@pytest.fixture
def driver(monkeypatch):
    helper = getattr(pipeline_tools, "_record_chemical_requirements", None)
    if helper is not None:
        helper.cache_clear()
    monkeypatch.setenv("ASO_LLM_PROVIDER", "openai")
    d = SimpleNamespace(calls=Counter(), selections=[], invalid_records=0,
                        raise_record_once=False, wrong_backbone=False)

    async def run(agent, *, input, **kwargs):
        data = json.loads(input)
        if agent.name == "ASO Chemical Requirement Recorder":
            d.calls["record"] += 1
            if d.raise_record_once:
                d.raise_record_once = False
                raise RuntimeError("provider unavailable")
            if d.invalid_records:
                d.invalid_records -= 1
                return SimpleNamespace(final_output='{not_json')
            source = data["sources"][0]
            body = {field: None for field in chemical_requirements.FIELD_TYPES}
            body["backbone"] = {"operator": "equals", "value": "PS",
                                "quote": "phosphorothioate", "source_id": source["source_id"]}
            return SimpleNamespace(final_output=json.dumps({"requirements": body}))
        if agent.name == "Chemistry Mechanism Consistency Check":
            d.calls["audit"] += 1
            return SimpleNamespace(final_output='{"issues": []}')
        d.calls["selection"] += 1
        d.selections.append(deepcopy(data))
        label = ("5-10-5" if data["structured_goal"].get("modality") == "gapmer"
                 else "steric-pmo" if d.wrong_backbone else "steric-hindrance")
        recommendation = {
            "CHEMISTRY": label, "length": data["exact_requested_length_nt"],
            "platform": "gapmer" if label == "5-10-5" else "steric_blocker",
            "rationale": "Match the stated architecture; activity is unmeasured.",
        }
        if label != "steric-pmo":
            recommendation["linkage_specification"] = ["PS"] * (data["exact_requested_length_nt"] - 1)
        return SimpleNamespace(final_output=json.dumps({"recommendation": recommendation,
                                                       "unsupported_reason": None}))

    d.run = run
    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    yield d
    if helper is not None:
        helper.cache_clear()


def select(g):
    return pipeline_tools._run_chemistry_agent_for_goal(chemistry_agent, g)


def test_identical_sources_reuse_only_recording_across_lengths_and_modalities(driver):
    a = select(goal(18))
    b = select(goal(23))
    c = select(goal(20, "gapmer"))

    assert driver.calls == {"record": 1, "selection": 3, "audit": 3}
    assert [a["recommendation"]["length"], b["recommendation"]["length"],
            c["recommendation"]["length"]] == [18, 23, 20]
    assert c["recommendation"]["CHEMISTRY"] == "5-10-5"
    assert [len(x["recommendation"]["linkage_specification"]) for x in (a, b, c)] == [17, 22, 19]
    assert a["chemical_requirements"] == b["chemical_requirements"] == c["chemical_requirements"]


def test_cached_results_are_not_shared_mutable_objects(driver):
    a = select(goal())
    a["chemical_requirements"][0]["value_json"] = '"PO"'
    a["chemical_requirements"].append({"field": "other"})
    b = select(goal())

    assert driver.calls["record"] == 1
    assert len(b["chemical_requirements"]) == 1
    assert json.loads(b["chemical_requirements"][0]["value_json"]) == "PS"


@pytest.mark.parametrize("change", ["text", "source_id", "schema"])
def test_changed_recording_inputs_require_a_new_extraction(driver, monkeypatch, change):
    select(goal())
    other = goal()
    if change == "text":
        other["chemistry"] += " Preserve that backbone."
    elif change == "source_id":
        other["constraints"] = {"chemistry": other.pop("chemistry")}
    else:
        original = chemical_requirements.requirement_schema

        def changed(sources):
            schema = deepcopy(original(sources))
            schema["description"] = "A separately versioned recording contract."
            return schema

        monkeypatch.setattr(chemical_requirements, "requirement_schema", changed)
    select(other)

    assert driver.calls["record"] == 2


@pytest.mark.parametrize("field,value", [
    ("instructions", "Record every stated requirement and its exact source quote."),
    ("model", "example/other-model"),
    ("model_settings", ModelSettings(max_output_tokens=8192)),
])
def test_recorder_changes_invalidate_reuse(driver, monkeypatch, field, value):
    select(goal())
    monkeypatch.setattr(chemical_requirements.chemical_requirement_agent, field, value)
    select(goal())
    assert driver.calls["record"] == 2


@pytest.mark.parametrize("field,first,second", [
    ("OPENROUTER_MODEL", "example/model-a", "example/model-b"),
    ("OPENROUTER_REASONING_EFFORT", "none", "max"),
    ("OPENROUTER_ONLY_PROVIDERS", "route/a", "route/b"),
])
def test_provider_condition_changes_invalidate_reuse(driver, monkeypatch, field, first, second):
    monkeypatch.setenv("ASO_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "example/model-a")
    monkeypatch.setenv(field, first)
    select(goal())
    monkeypatch.setenv(field, second)
    select(goal())
    assert driver.calls["record"] == 2


def test_invalid_recording_is_not_cached_and_retains_its_retry_budget(driver):
    driver.invalid_records = 3
    with pytest.raises(ChemistryResolutionError, match="chemistry_requirements_unresolved"):
        select(goal())
    a = select(goal())
    b = select(goal())

    assert driver.calls == {"record": 4, "selection": 2, "audit": 2}
    assert a["chemical_requirements"] == b["chemical_requirements"]


def test_unavailable_recorder_is_not_cached(driver):
    driver.raise_record_once = True
    with pytest.raises(RuntimeError, match="provider unavailable"):
        select(goal())
    select(goal())
    select(goal())
    assert driver.calls == {"record": 2, "selection": 2, "audit": 2}


def test_capabilities_are_checked_again_even_after_recording_is_reused(driver, monkeypatch):
    select(goal())

    def unavailable(_requirements):
        raise ValueError("required capability unavailable; do not substitute")

    monkeypatch.setattr(chemical_requirements, "require_native_requirement_capabilities", unavailable)
    with pytest.raises(ChemistryResolutionError, match="unsupported_chemistry"):
        select(goal())
    assert driver.calls == {"record": 1, "selection": 1, "audit": 1}


def test_new_molecule_cannot_contradict_cached_source_requirements(driver):
    select(goal())
    driver.wrong_backbone = True
    with pytest.raises(ChemistryResolutionError, match="chemistry_contract_unresolved"):
        select(goal(23))
    assert driver.calls == {"record": 1, "selection": 4, "audit": 1}
    assert "does not fulfill recorded backbone" in str(driver.selections[-1]["output_contract_feedback"])


def test_runner_replacement_does_not_reuse_another_runner_result(driver, monkeypatch):
    select(goal())

    async def other_runner(*args, **kwargs):
        return await driver.run(*args, **kwargs)

    monkeypatch.setattr(pipeline_tools.Runner, "run", other_runner)
    select(goal())
    assert driver.calls["record"] == 2
