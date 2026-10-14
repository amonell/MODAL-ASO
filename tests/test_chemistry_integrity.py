"""Chemical identity regressions independent of gene/task or model answers."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from aso_chemistry import (canonical_chemistry_configuration, encode_idt_oligo,
                           require_modality_chemistry, require_native_linkage_identity)
from aso_molecule import molecule_record, parse_ordering_notation, validate_molecule_record
from aso_recipient_report import attest_record, validate_recipient_report, write_recipient_report
from aso_agents.chemical_identity import attest_candidate_rows, write_native_chemistry_exports


def test_native_schema_passes_the_actual_provider_preparation_boundary():
    from aso_agents.chemistry_contracts import chemistry_selection_schema
    from aso_agents.chemistry_agent import chemistry_agent
    from aso_agents.pipeline_tools import _agent_with_strict_output_schema
    from aso_agents.llm_provider import get_provider_config, prepare_responses_payload
    agent = _agent_with_strict_output_schema(chemistry_agent, schema_name="aso_chemical_architecture", schema=chemistry_selection_schema(20))
    payload = {"model": agent.model, "input": []}
    agent.model_settings.apply(payload)
    config = get_provider_config({"ASO_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "test-only", "OPENROUTER_MODEL": "deepseek/deepseek-v4-flash-0731"})
    prepared, _ = prepare_responses_payload(payload, config)
    assert prepared["model"] == config.model_override


def test_model_linkage_count_errors_reach_repair_without_aborting(monkeypatch):
    from types import SimpleNamespace
    from aso_agents import pipeline_tools
    from aso_agents.chemistry_agent import chemistry_agent
    prompts = []
    async def run(agent, *, input, **kwargs):
        if agent.name == "Chemistry Mechanism Consistency Check":
            return SimpleNamespace(final_output=json.dumps({"issues": []}))
        prompts.append(json.loads(input))
        architecture = {"kind": "gapmer", "sugar": "2MOE", "backbone": "PS", "wing_5prime": 5,
                        "dna_gap": 10, "wing_3prime": 5, "linkages": ["PS"] * (20 if len(prompts) == 1 else 19)}
        return SimpleNamespace(final_output=json.dumps({"recommendation": {"architecture": architecture,
            "length": 20, "rationale": "Supported RNase H gapmer chemistry; efficacy is unmeasured."}, "unsupported_reason": None}))
    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    selected = pipeline_tools._run_chemistry_agent_for_goal(chemistry_agent, {"intent": "knockdown", "aso_length_nt": 20})
    assert selected["recommendation"]["linkage_specification"] == ["PS"] * 19
    assert len(prompts) == 2
    assert "length-1" in str(prompts[1]["output_contract_feedback"])


@pytest.mark.parametrize("field,value", [
    ("cytosine_methylation", "none"), ("sugar", "LNA"),
    ("stereochemistry", ["Rp"] * 19), ("conjugates", ["GalNAc"]),
    ("terminal_modifications", ["phosphate"]), ("unregistered_modification", True),
    ("linkage_specification", ["PS"] * 18 + ["PO"]), ("length", 19),
])
@pytest.mark.parametrize("boundary", ["shared", "intake", "pipeline", "compiler"])
def test_native_normalization_never_discards_a_requirement(field, value, boundary):
    from aso_agents import pipeline_tools
    from aso_agents.chemistry_contracts import compile_chemistry_proposal, chemistry_contract_errors
    import run_goal_intake
    rec = {"CHEMISTRY": "5-10-5", "platform": "gapmer", "length": 20, "rationale": "Recorded configuration.", field: value}
    if boundary == "shared":
        with pytest.raises(ValueError): canonical_chemistry_configuration(rec)
    elif boundary == "intake":
        with pytest.raises(ValueError): run_goal_intake._normalize_requested_chemistry(rec)
    elif boundary == "pipeline":
        with pytest.raises(ValueError): pipeline_tools._canonicalize_chemistry_payload({"recommendation": rec})
    else:
        assert chemistry_contract_errors(compile_chemistry_proposal({"recommendation": rec}))


def test_extra_supported_fields_are_preserved_and_goals_are_not_chemical_objects():
    rec = {"CHEMISTRY": " 5-10-5 ", "length": 20, "cytosine_methylation": "all",
           "chemistry_assumption_status": "explicit_request", "conjugates": []}
    result = canonical_chemistry_configuration(rec)
    assert result["CHEMISTRY"] == "5-10-5"
    assert result["cytosine_methylation"] == "all"
    assert result["chemistry_assumption_status"] == "explicit_request"
    require_native_linkage_identity({"kind": "query", "gene": "arbitrary", "chemistry": {"recommendation": result, "mechanism_review": {}}})
    assert rec["CHEMISTRY"] == " 5-10-5 "


def test_an_explicit_model_capability_decline_cannot_be_overridden_by_a_proposal():
    from aso_agents.chemistry_contracts import compile_chemistry_proposal, ChemistryResolutionError
    payload = {"recommendation": {"CHEMISTRY": "5-10-5"}, "unsupported_reason": "Required chemical component unavailable."}
    with pytest.raises(ChemistryResolutionError, match="unsupported_chemistry"):
        compile_chemistry_proposal(payload)
    with pytest.raises(ValueError, match="Unsupported chemistry"):
        canonical_chemistry_configuration(payload)


def test_pmo_profile_default_does_not_assert_unsupported_base_modifications():
    config = canonical_chemistry_configuration({"CHEMISTRY": "steric-pmo", "length": 25, "cytosine_methylation": "profile_default"})
    assert config["cytosine_methylation"] == "profile_default"
    with pytest.raises(ValueError):
        canonical_chemistry_configuration({"CHEMISTRY": "steric-pmo", "cytosine_methylation": "all"})


def test_native_selection_checks_rnase_h_support_separately_from_physical_structure():
    from aso_agents.chemistry_contracts import chemistry_contract_errors
    proposal = {"recommendation": {"CHEMISTRY": "3-1-3", "platform": "gapmer", "length": 7, "rationale": "Physical short-gap architecture."}}
    assert not chemistry_contract_errors(proposal)
    assert any("at least 6" in error for error in chemistry_contract_errors(proposal, native_design=True))


def test_outer_chemistry_requirements_cannot_be_dropped():
    for field in ("sugar", "stereochemistry", "conjugates", "unknown_base_modification"):
        with pytest.raises(ValueError, match="do not discard"):
            canonical_chemistry_configuration({"recommendation": {"CHEMISTRY": "5-10-5"}, field: "requested"})


def requirement(field, value, quote="explicit", operator="equals"):
    return {"field": field, "value_json": json.dumps(value), "source_id": "/chemistry", "quote": quote, "operator": operator}


def recorder_body(*requirements):
    from aso_agents.chemical_requirements import FIELD_TYPES
    result = {field: None for field in FIELD_TYPES}
    for r in requirements:
        field = "wing_chemistry" if r["field"] == "sugar" else r["field"]
        result[field] = {"operator": r["operator"], "value": json.loads(r["value_json"]),
                         "source_id": r["source_id"], "quote": r["quote"]}
    return {"requirements": result}


@pytest.mark.parametrize("field,value", [("conjugates", ["attachment"]), ("sugar", "LNA"), ("stereochemistry", "Rp"), ("linkages", ["PO"]), ("other", "unrepresented chemical group")])
def test_requirement_capability_checks_do_not_need_a_model(field, value):
    from aso_agents.chemical_requirements import require_native_requirement_capabilities
    with pytest.raises(ValueError, match="do not substitute"):
        require_native_requirement_capabilities([requirement(field, value)])


@pytest.mark.parametrize("requirements", [
    [requirement("sugar", "2MOE"), requirement("cytosine_methylation", "none")],
    [requirement("sugar", "2OME"), requirement("cytosine_methylation", "all")],
    [requirement("sugar", "MORPHOLINO"), requirement("platform", "gapmer")],
    [requirement("sugar", "2MOE"), requirement("backbone", "PHOSPHORODIAMIDATE")],
    [requirement("backbone", "PS"), requirement("stereochemistry", ["Rp"], operator="excludes")],
])
def test_impossible_building_block_combinations_are_declined_before_selection(requirements):
    from aso_agents.chemical_requirements import require_native_requirement_capabilities
    with pytest.raises(ValueError, match="combination"):
        require_native_requirement_capabilities(requirements)


def test_supported_chemical_alternatives_are_not_over_rejected():
    from aso_agents.chemical_requirements import require_native_requirement_capabilities
    require_native_requirement_capabilities([requirement("sugar", ["2MOE", "2OME"], operator="one_of"), requirement("cytosine_methylation", "none")])
    require_native_requirement_capabilities([requirement("sugar", "MORPHOLINO"), requirement("gap_chemistry", "none")])


def test_recorder_cannot_hide_an_attachment_by_calling_it_delivery(monkeypatch):
    from types import SimpleNamespace
    from aso_agents import pipeline_tools
    from aso_agents.chemistry_agent import chemistry_agent
    from aso_agents.chemistry_contracts import ChemistryResolutionError
    calls = []
    async def run(agent, *, input, **kwargs):
        calls.append(agent.name)
        assert agent.name == "ASO Chemical Requirement Recorder"
        return SimpleNamespace(final_output=json.dumps(recorder_body(requirement("conjugates", ["attachment"], "delivery attachment"))))
    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    with pytest.raises(ChemistryResolutionError, match="unsupported_chemistry"):
        pipeline_tools._run_chemistry_agent_for_goal(chemistry_agent, {"chemistry": "MOE/DNA with a delivery attachment"})
    assert calls == ["ASO Chemical Requirement Recorder"]


def test_selected_chemistry_is_compared_with_recorded_constraints():
    from aso_agents.chemical_requirements import requirement_configuration_errors
    rec = {"CHEMISTRY": "5-10-5", "length": 20}
    assert requirement_configuration_errors([requirement("cytosine_methylation", "none")], rec)
    assert not requirement_configuration_errors([requirement("sugar", ["2MOE", "LNA"], operator="one_of")], rec)
    assert not requirement_configuration_errors([requirement("sugar", ["2OME"], operator="excludes")], rec)
    assert requirement_configuration_errors([requirement("sugar", "2OME")], rec)
    assert requirement_configuration_errors([requirement("stereochemistry", ["Rp"], operator="excludes")], rec)


@pytest.mark.parametrize("boundary", ["intake", "runtime", "pipeline"])
def test_recorded_requirements_survive_all_native_boundaries(boundary):
    import run_goal_intake
    from aso_agents import pipeline_tools
    payload = {"recommendation": {"CHEMISTRY": "5-10-5", "length": 20},
               "chemical_requirements": [requirement("sugar", "2OME")]}
    with pytest.raises(ValueError, match="does not fulfill"):
        if boundary == "intake": run_goal_intake._normalize_requested_chemistry(payload)
        elif boundary == "runtime": run_goal_intake._resolve_gapmer_chemistry({"chemistry": payload}, tile_len=20)
        else: pipeline_tools._canonicalize_chemistry_payload(payload)


def test_recorded_constraints_need_original_source_quotes():
    from aso_agents.chemical_requirements import validate_requirement_response
    with pytest.raises(ValueError, match="source quote"):
        validate_requirement_response(recorder_body(requirement("sugar", "LNA", "fabricated")), [{"source_id": "/chemistry", "text": "MOE"}])


def test_candidate_views_cannot_disagree_with_the_molecule():
    from aso_molecule import validate_candidate_molecule
    m = molecule_record("ACGT", "steric-hindrance", encode_idt_oligo("ACGT", "steric-hindrance"))
    row = {"molecule_specification": m, "Synthesis": "ACGT", "IDT": m["ordering_notation"]}
    validate_candidate_molecule(row)
    row["Synthesis"] = "AGGT"
    with pytest.raises(ValueError, match="contradicts"):
        validate_candidate_molecule(row)


@pytest.mark.parametrize("profile", ["steric-pmo", " steric-pmo ", "unspecified", " unspecified "])
def test_whitespace_cannot_bypass_non_ps_chemistry(profile):
    with pytest.raises(ValueError):
        canonical_chemistry_configuration({"CHEMISTRY": profile, "length": 4, "linkages": ["PS"] * 3})
    m = molecule_record("ACGT", profile, "")
    assert m["ordering_status"] != "attested"
    validate_molecule_record(m)


@pytest.mark.parametrize("sugar,prefix", [("2MOE", ""), ("2OME", "2ome:")])
@pytest.mark.parametrize("counts", [(1, 6, 2), (4, 8, 3), (5, 10, 5), (3, 10, 3)])
@pytest.mark.parametrize("mixed", [False, True])
def test_arbitrary_supported_layouts_preserve_every_atom_and_bond(sugar, prefix, counts, mixed):
    n = sum(counts); sequence = ("ACTGCC" * n)[:n]
    links = ["PO" if mixed and i % 3 == 0 else "PS" for i in range(n - 1)]
    label = prefix + "-".join(map(str, counts))
    notation = encode_idt_oligo(sequence, label, linkages=links)
    m = molecule_record(sequence, label, notation, linkages=links)
    assert m["nucleotide_sugars_5to3"] == [sugar] * counts[0] + ["DNA"] * counts[1] + [sugar] * counts[2]
    assert m["linkages_5to3"] == links
    assert m["base_modifications_5to3"] == ["5mC" if b == "C" and sugar == "2MOE" else "none" for b in sequence]
    assert notation.count("*") == links.count("PS")
    validate_molecule_record(m)


def test_moe_methylcytosine_is_distinct_from_ome_cytosine():
    moe = parse_ordering_notation(encode_idt_oligo("ACCA", "steric-hindrance"))
    ome = parse_ordering_notation(encode_idt_oligo("ACCA", "steric-2ome-ps"))
    assert moe["base_modifications_5to3"] == ["none", "5mC", "5mC", "none"]
    assert ome["base_modifications_5to3"] == ["none"] * 4


@pytest.mark.parametrize("field", ["nucleotide_sugars_5to3", "base_modifications_5to3", "linkages_5to3", "physical_molecule_sha256", "conjugates"])
def test_record_cannot_be_tampered_independently_of_its_order_string(field):
    m = molecule_record("ACGT", "steric-hindrance", encode_idt_oligo("ACGT", "steric-hindrance"))
    m[field] = "wrong"
    with pytest.raises(ValueError): validate_molecule_record(m)


def test_physical_architecture_does_not_imply_supported_rnase_h_mechanism():
    assert encode_idt_oligo("ACGTACG", "3-1-3")
    with pytest.raises(ValueError, match="at least 6"):
        require_modality_chemistry("gapmer", "3-1-3")


@pytest.mark.parametrize("description", ["full-PS 5-10-5 LNA/DNA gapmer", "full-PS 5-10-5 OMe/DNA gapmer", "full-PS 5-10-5 MOE/DNA gapmer"])
def test_numeric_prose_cannot_imply_moe_wings(description):
    from aso_agents.chemical_position_evidence import chemical_position_evidence
    e = chemical_position_evidence({"sequence": "A" * 20, "chemistry": description})
    assert e["wing_chemistry"] == "unspecified"
    assert e["position_ranges_1based_inclusive"]["dna_gap"] == [6, 15]


def test_multi_program_export_uses_the_producing_chemistry(tmp_path):
    sequence = "ACTG"
    goal = {"design_programs": [{"program_id": "a", "modality": "steric", "chemistry": "steric-hindrance"},
                                {"program_id": "b", "modality": "steric", "chemistry": "steric-2ome-ps"}]}
    rows = [{"program_id": pid, "synthesized_antisense_sequence": sequence, "selection_rank": rank,
             "source_annotations": {"IDT": encode_idt_oligo(sequence, chemistry)}}
            for pid, chemistry, rank in [("a", "steric-hindrance", 7), ("b", "steric-2ome-ps", 3)]]
    original = deepcopy(rows)
    bound = attest_candidate_rows(rows, goal)
    assert [r["selection_rank"] for r in bound] == [7, 3]
    assert [r["molecule_specification"]["architecture"] for r in bound] == ["steric-hindrance", "steric-2ome-ps"]
    assert rows == original
    paths = write_native_chemistry_exports(bound, tmp_path)
    assert paths == write_native_chemistry_exports(bound, tmp_path)
    assert validate_recipient_report(Path(paths["chemical_identity_json"]).parent)["records_checked"] == 2
    bad = deepcopy(rows); bad[0]["source_annotations"]["IDT"] = bad[1]["source_annotations"]["IDT"]
    with pytest.raises(ValueError, match="contradicts"): attest_candidate_rows(bad, goal)


def test_pmo_recipient_views_preserve_nonorderable_identity(tmp_path):
    record = attest_record("pmo", "ACGT", "steric-pmo", "", linkages=None, assumption_status="explicit_request")
    path = tmp_path / "pmo"
    write_recipient_report(path, [record], title="PMO", context="", provenance={})
    assert validate_recipient_report(path)["records_checked"] == 1
    assert record["molecule_specification"]["linkages_5to3"] == ["PHOSPHORODIAMIDATE"] * 3


def test_native_export_catches_missing_and_modified_ordering_notation():
    goal = {"chemistry": "steric-hindrance"}
    with pytest.raises(ValueError):
        attest_candidate_rows([{"synthesized_antisense_sequence": "ACGT"}], goal)
    row = {"synthesized_antisense_sequence": "ACGT", "IDT": encode_idt_oligo("ACGT", "steric-hindrance")}
    bound = attest_candidate_rows([row], goal)
    bound[0]["IDT"] = encode_idt_oligo("AGGT", "steric-hindrance")
    with pytest.raises(ValueError): attest_candidate_rows(bound, goal)
