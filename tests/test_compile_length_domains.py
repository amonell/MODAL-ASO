"""A finite domain compiles into exact branch copies through the normal executor."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as tools


ROOT = "/tasks/0/context/structured_goal/design_programs"


def fixture(length=None):
    first = {"program_id": "branch", "label": "First interval", "modality": "steric",
             "intent": "binding", "objective": "Compare the requested lengths",
             "target_context": "first reference interval", "chemistry": "steric-hindrance",
             "run_splice_eval": False, "transcript_id": "TXQ.4",
             "region_constraints": [{"source_type": "explicit_rows",
                 "coordinate_space": "genomic", "aso_overlap_requirement": "full_containment",
                 "intervals_bed0_half_open": [{"chrom": "chr7", "start": 301, "end": 359,
                                                "strand": "-"}]}]}
    if length is not None:
        first["length"] = length
    second = {**deepcopy(first), "program_id": "other", "length": 25,
              "target_context": "second reference interval"}
    envelope = {"tasks": [{"task_id": "design", "task_type": "design_new_asos",
                 "context": {"structured_goal": {"design_programs": [first, second]}}}],
                "final_outputs": [{"task_id": "design", "kind": "file"}]}
    evidence = [{"evidence_id": "request", "kind": "user_request",
                 "text": "Compare lengths 13, 17 and 21 in the first interval. Keep its existing constraints. Use 25 in the second interval."},
                {"evidence_id": "other_source", "kind": "user_request", "text": "Keep the requested files."}]
    prompt = tools._goal_audit_prompt(request_envelope=envelope, validation_issues=[],
        issue_records=[], evidence_sources=evidence, semantic_audit=True)
    protocol = tools._goal_audit_protocol_payload(prompt)
    span = next(k for k, v in tools._evidence_span_map(protocol["evidence_sources"]).items()
                if v["evidence_id"] == "request")
    check = {"verdict": "expand", "requested_lengths": [13, 17, 21],
             "represented_program_ids": ["branch"], "evidence_refs": ["request"],
             "evidence_span_id": span, "reason": "One template must execute each explicit length."}
    answer = {"status": "repairable", "patches": [], "goal_replacements": [],
              "field_dispositions": [], "unresolved": [], "program_length_checks": {
                  "program_0": check,
                  "program_1": {**deepcopy(check), "verdict": "complete",
                                "requested_lengths": [25], "represented_program_ids": ["other"]}}}
    return envelope, prompt, answer, evidence


def run(monkeypatch, prompt, answer):
    calls = []

    async def invoke(agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(answer))

    monkeypatch.setattr(tools.Runner, "run", invoke)
    result = asyncio.run(tools._run_goal_contract_audit_agent_once(prompt))
    return result, calls


def apply(envelope, result, evidence):
    # This is the same coalescing boundary used by the full repair loop.
    patches, conflicts = tools._coalesce_goal_audit_patches(result["patches"], evidence_sources=evidence)
    assert not conflicts
    applied = tools._apply_goal_audit_patches(envelope, patches, issue_records=[], evidence_sources=evidence)
    assert not applied["rejected"]
    assert applied["request_envelope"]["final_outputs"] == envelope["final_outputs"]
    return applied["request_envelope"]["tasks"][0]["context"]["structured_goal"]["design_programs"]


def facts(program):
    return {k: v for k, v in program.items() if k not in {"program_id", "label", "objective", "length"}}


@pytest.mark.parametrize("length", [None, 17, 29])
def test_domain_materializes_exact_variants_without_model_copying_or_extra_calls(monkeypatch, length):
    envelope, prompt, answer, evidence = fixture(length)
    before, raw = deepcopy(envelope), deepcopy(answer)
    result, calls = run(monkeypatch, prompt, answer)
    assert result["status"] == "repairable" and len(calls) == 1
    programs = apply(envelope, result, evidence)
    variants = [p for p in programs if p["program_id"] != "other"]
    assert sorted(p["length"] for p in variants) == [13, 17, 21]
    assert len({p["program_id"] for p in programs}) == len(programs)
    template = before["tasks"][0]["context"]["structured_goal"]["design_programs"][0]
    assert all(facts(p) == facts(template) for p in variants)
    assert programs[1] == before["tasks"][0]["context"]["structured_goal"]["design_programs"][1]
    assert programs[0]["program_id"] == "branch"
    assert envelope == before and answer == raw


def test_two_independent_expansions_keep_distinct_branch_domains(monkeypatch):
    envelope, prompt, answer, evidence = fixture()
    answer["program_length_checks"]["program_1"].update(verdict="expand", requested_lengths=[25, 27])
    result, calls = run(monkeypatch, prompt, answer)
    assert result["status"] == "repairable" and len(calls) == 1
    programs = apply(envelope, result, evidence)
    for context, expected in [("first reference interval", [13, 17, 21]),
                              ("second reference interval", [25, 27])]:
        assert sorted(p["length"] for p in programs if p["target_context"] == context) == expected


def test_compilation_preserves_an_independently_audited_leaf_repair(monkeypatch):
    envelope, prompt, answer, evidence = fixture()
    evidence[0]["text"] += " Set the first interval start to 304."
    prompt = tools._goal_audit_prompt(request_envelope=envelope, validation_issues=[],
        issue_records=[], evidence_sources=evidence, semantic_audit=True)
    span = next(iter(tools._evidence_span_map(tools._goal_audit_protocol_payload(prompt)["evidence_sources"])))
    for check in answer["program_length_checks"].values():
        check["evidence_span_id"] = span
    answer["patches"] = [{"op": "replace", "path": ROOT + "/0/region_constraints/0/intervals_bed0_half_open/0/start",
                          "value": 304, "evidence_refs": ["request"], "evidence_span_id": span,
                          "reason": "Apply the explicit interval correction."}]
    result, calls = run(monkeypatch, prompt, answer)
    assert result["status"] == "repairable" and len(calls) == 1
    programs = apply(envelope, result, evidence)
    assert [p["region_constraints"][0]["intervals_bed0_half_open"][0]["start"]
            for p in programs if p["program_id"] != "other"] == [304, 304, 304]


def test_generated_ids_do_not_overwrite_an_unrelated_existing_program(monkeypatch):
    envelope, _, answer, evidence = fixture()
    other = envelope["tasks"][0]["context"]["structured_goal"]["design_programs"][1]
    other["program_id"] = "branch__length_17"
    answer["program_length_checks"]["program_1"]["represented_program_ids"] = [other["program_id"]]
    prompt = tools._goal_audit_prompt(request_envelope=envelope, validation_issues=[],
        issue_records=[], evidence_sources=evidence, semantic_audit=True)
    result, calls = run(monkeypatch, prompt, answer)
    assert result["status"] == "repairable" and len(calls) == 1
    programs = apply(envelope, result, evidence)
    assert programs[1] == other
    assert len({p["program_id"] for p in programs}) == 4
    assert sorted(p["length"] for p in programs if p["target_context"] == "first reference interval") == [13, 17, 21]


def test_existing_equivalent_variant_requires_explicit_group_coverage(monkeypatch):
    envelope, _, answer, evidence = fixture(13)
    programs = envelope["tasks"][0]["context"]["structured_goal"]["design_programs"]
    programs[1] = {**deepcopy(programs[0]), "program_id": "other", "length": 17}
    prompt = tools._goal_audit_prompt(request_envelope=envelope, validation_issues=[],
        issue_records=[], evidence_sources=evidence, semantic_audit=True)
    result, _ = run(monkeypatch, prompt, answer)
    assert result["error"] == "invalid_structured_output"
    assert "equivalent variants already exist" in result["reason"]


def test_uncompiled_expansion_is_not_evidence_of_executed_coverage():
    from aso_agents.program_domains import program_length_audit_obligations, program_length_coverage_errors

    _, prompt, answer, _ = fixture()
    protocol = tools._goal_audit_protocol_payload(prompt)
    errors = program_length_coverage_errors(answer, program_length_audit_obligations(protocol),
        tools._evidence_span_map(protocol["evidence_sources"]))
    assert any("expansion must be compiled" in error for error in errors)


def test_malformed_coverage_shape_is_reported_instead_of_crashing(monkeypatch):
    _, prompt, answer, _ = fixture()
    answer["program_length_checks"] = []
    result, _ = run(monkeypatch, prompt, answer)
    assert result["error"] == "invalid_structured_output"


@pytest.mark.parametrize("defect", ["missing_domain", "duplicate_lengths", "boolean_length",
                                  "unknown_span", "wrong_evidence", "missing_template_id",
                                  "unrelated_group", "removed_template", "manual_length_conflict"])
def test_expansion_does_not_guess_or_bypass_evidence_and_branch_validation(monkeypatch, defect):
    envelope, prompt, answer, _ = fixture()
    check = answer["program_length_checks"]["program_0"]
    if defect == "missing_domain":
        check["requested_lengths"] = []
    elif defect == "duplicate_lengths":
        check["requested_lengths"] = [13, 13]
    elif defect == "boolean_length":
        check["requested_lengths"] = [True, 17]
    elif defect == "unknown_span":
        check["evidence_span_id"] = "invented"
    elif defect == "wrong_evidence":
        check["evidence_refs"] = ["other_source"]
    elif defect == "missing_template_id":
        check["represented_program_ids"] = []
    elif defect == "unrelated_group":
        check["represented_program_ids"] = ["branch", "other"]
    else:
        path = ROOT + "/0" + ("/length" if defect == "manual_length_conflict" else "")
        answer["patches"] = [{"op": "add" if defect == "manual_length_conflict" else "remove",
                              "path": path, "value": 29 if defect == "manual_length_conflict" else None,
                              "evidence_refs": ["request"], "evidence_span_id": check["evidence_span_id"],
                              "reason": "Conflicting expansion instruction."}]
    before = deepcopy(envelope)
    result, calls = run(monkeypatch, prompt, answer)
    assert result["error"] == "invalid_structured_output" and len(calls) == 4
    assert envelope == before
