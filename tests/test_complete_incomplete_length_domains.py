"""An evidence-bound incomplete domain already specifies the needed operation."""
from copy import deepcopy

import pytest

from test_compile_length_domains import ROOT, fixture, run, apply, facts
from aso_agents import pipeline_tools as tools


@pytest.mark.parametrize("length", [None, 17, 29])
def test_single_template_incomplete_finding_compiles_without_an_extra_command(monkeypatch, length):
    envelope, prompt, answer, evidence = fixture(length)
    answer["program_length_checks"]["program_0"]["verdict"] = "incomplete"
    original = deepcopy(answer)
    result, calls = run(monkeypatch, prompt, answer)
    assert result["status"] == "repairable" and len(calls) == 1
    programs = apply(envelope, result, evidence)
    assert sorted(p["length"] for p in programs if p["program_id"] != "other") == [13, 17, 21]
    assert all(facts(p) == facts(programs[0]) for p in programs if p["program_id"] != "other")
    assert answer == original


def test_noop_scalar_replacements_do_not_hide_the_remaining_domain_repair(monkeypatch):
    envelope, prompt, answer, evidence = fixture(17)
    check = answer["program_length_checks"]["program_0"]
    check["verdict"] = "incomplete"
    answer["patches"] = [{"op": "replace", "path": ROOT + "/0/length", "value": 17,
                          "evidence_refs": check["evidence_refs"],
                          "evidence_span_id": check["evidence_span_id"],
                          "reason": "Preserve the existing scalar while covering the domain."}]
    result, calls = run(monkeypatch, prompt, answer)
    assert result["status"] == "repairable" and len(calls) == 1
    programs = apply(envelope, result, evidence)
    assert sorted(p["length"] for p in programs if p["program_id"] != "other") == [13, 17, 21]


def test_existing_explicit_variant_repairs_are_not_expanded_twice(monkeypatch):
    envelope, prompt, answer, evidence = fixture(13)
    check = answer["program_length_checks"]["program_0"]
    check["verdict"] = "incomplete"
    template = envelope["tasks"][0]["context"]["structured_goal"]["design_programs"][0]
    answer["patches"] = [{"op": "add", "path": ROOT + "/-",
                          "value": {**deepcopy(template), "program_id": f"explicit_{n}", "length": n},
                          "evidence_refs": check["evidence_refs"], "evidence_span_id": check["evidence_span_id"],
                          "reason": "Supply the exact missing scalar variant."} for n in [17, 21]]
    result, calls = run(monkeypatch, prompt, answer)
    assert result["status"] == "repairable" and len(calls) == 1
    assert len(result["patches"]) == 2
    programs = apply(envelope, result, evidence)
    assert sorted(p["length"] for p in programs if p["program_id"] != "other") == [13, 17, 21]


@pytest.mark.parametrize("defect", ["other_group", "missing_evidence", "unknown_length"])
def test_incomplete_does_not_authorize_guessing_or_unbound_repairs(monkeypatch, defect):
    _, prompt, answer, _ = fixture()
    check = answer["program_length_checks"]["program_0"]
    check["verdict"] = "incomplete"
    if defect == "other_group":
        check["represented_program_ids"] = ["branch", "other"]
    elif defect == "missing_evidence":
        check["evidence_refs"] = ["other_source"]
    else:
        check["requested_lengths"] = []
    result, calls = run(monkeypatch, prompt, answer)
    assert result["error"] == "invalid_structured_output" and len(calls) == 4
