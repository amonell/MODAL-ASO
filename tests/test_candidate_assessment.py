import json
import math

import pytest

from aso_agents.candidate_assessment import (
    assess_candidates, duplex_reference, motif_screen, off_target_evidence,
    rank_candidates, write_assessment_artifacts,
)
from aso_agents.chemical_requirements import _constraint_matches, require_native_requirement_capabilities
from aso_agents.shortlist_clarification import parse_size_answer, size_question


def row(sequence="ACGTTGCA", *, count=0, budget=2, complete=True, tier=4):
    return {"synthesized_antisense_sequence":sequence, "chemistry_label":"steric-hindrance",
        "recommendation_rank_key":[tier,0.01], "selection_rank":1, "program_id":"p",
        "off_target_count_status":"complete_within_declared_scope" if complete else "incomplete",
        "off_target_exact_count_unit":"genes", "Off_target_mismatch_budget":budget,
        "Off_target_exact_count":count,"Off_target_1mm_count":0,"Off_target_2mm_count":0,
        "off_target_search_provenance_by_species":{"human":{"assembly":"test"}}}


def test_hybrid_reference_uses_target_rna_orientation_and_standard_state():
    value = duplex_reference("ACGTTGCA")
    assert value["complementary_reference_rna_5to3"] == "UGCAACGU"
    # Independently summed published oriented stacks for UGCAACGU/dACGTTGCA.
    assert value["delta_h_kcal_mol"] == pytest.approx(-63.3)
    assert value["delta_s_cal_mol_k"] == pytest.approx(-178.4)
    assert value["delta_g_kcal_mol"] == pytest.approx(-63.3 + 310.15 * 0.1784)
    assert value["tm_c"] == pytest.approx(-63300 / (-178.4 + 1.987 * math.log(250e-9)) - 273.15)
    assert value["conditions"]["sodium_mm"] == 1000
    assert value["status"] == "unmodified_reference_proxy"
    assert value["modified_chemistry_tm_c"] is None
    assert duplex_reference("ACGNTGCA")["status"] == "unavailable"
    assert duplex_reference("ACGTTGCAACGTTGCA")["length_extrapolation"] is True


def test_modified_chemistry_never_acquires_unmodified_affinity_claim():
    molecule = {"nucleotide_sugars_5to3":["DNA"]*8,"linkages_5to3":["PO"]*7,"base_modifications_5to3":["none"]*8}
    assert duplex_reference("ACGTTGCA",molecule)["status"] == "unmodified_duplex_prediction"
    molecule["linkages_5to3"] = ["PS"]*7
    assert duplex_reference("ACGTTGCA",molecule)["status"] == "unmodified_reference_proxy"
    for conditions in ({"temperature_c":float("nan")},{"strand_concentration_nm":0},{"temperature_c":-274}):
        with pytest.raises(ValueError):
            duplex_reference("ACGT", **conditions)


def test_published_flags_are_chemistry_qualified_and_not_exclusions():
    sequence = "ATGCTCCG"
    screen = motif_screen(sequence,{"nucleotide_sugars_5to3":["2MOE"]*8,"base_modifications_5to3":["5mC"]*8})
    assert not screen["excluded"]
    assert {flag["rule_id"] for flag in screen["flags"]} == {"lna_383_tgc","lna_383_tcc","cpg_immune_context"}
    assert screen["flags"][0]["positions_1based"] == [2]
    assert screen["flags"][0]["applicability"] == "outside_studied_architecture"
    assert screen["flags"][2]["cytosines_5mC_at_all_hits"] is True
    studied = {"nucleotide_sugars_5to3":["LNA"]*3+["DNA"]*8+["LNA"]*3,
               "linkages_5to3":["PS"]*13, "base_modifications_5to3":["none"]*14}
    assert motif_screen("ATGC" + "A"*10, studied)["flags"][0]["applicability"] == "matches_studied_chemistry"


def test_custom_iupac_matches_overlap_and_fail_closed_on_invalid_rules():
    rules = [{"rule_id":"custom","motif":"ARA","action":"exclude"}]
    value = motif_screen("AAAAA",rules=rules)
    assert value["flags"][0]["positions_1based"] == [1,2,3]
    assert value["excluded"]
    with pytest.raises(ValueError):
        motif_screen("AAAA",rules=[{"rule_id":"bad","motif":".*","action":"flag"}])
    with pytest.raises(ValueError):
        motif_screen("AAAA",rules=rules*2)
    unverified = assess_candidates([{"Sequence":"AAAA"}],{"sequence_motif_rules":rules})[0]
    assert not unverified["candidate_assessment"]["selection_eligible"]


def test_missing_and_capped_off_target_counts_are_not_clean_screens():
    assert off_target_evidence(row())["status"] == "complete_within_declared_scope"
    assert off_target_evidence({**row(),"off_target_mapping_capped":True})["status"] != "complete_within_declared_scope"
    evidence = off_target_evidence({**row(),"Off_target_2mm_count":None})
    assert evidence["status"] == "incomplete_or_unavailable"
    assert evidence["counts_by_mismatch"][2] is None


def test_triage_preserves_direction_and_never_compares_incompatible_searches():
    original = [row(count=3),row("AACCTTGG",count=0),row("GGAACTCT",complete=False),row("TTCCAAGG",tier=0)]
    assessed = assess_candidates(original,{})
    ranked = rank_candidates(assessed)
    assert [r["synthesized_antisense_sequence"] for r in ranked] == ["AACCTTGG","ACGTTGCA","GGAACTCT","TTCCAAGG"]
    assert "candidate_assessment" not in original[0]
    other_budget = assess_candidates([row(count=4,budget=2),row("AACCTTGG",count=0,budget=0)],{})
    assert rank_candidates(other_budget) == other_budget
    unknown_scope = assess_candidates([{**row(count=4),"off_target_search_provenance_by_species":{}},
        {**row("AACCTTGG",count=0),"off_target_search_provenance_by_species":{}}],{})
    assert rank_candidates(unknown_scope) == unknown_scope
    opposing = assess_candidates([{**row(),"recommendation_status":"opposing_predictions"}],{})
    assert opposing[0]["candidate_assessment"]["recommendation_status"] == "exploratory"


def test_inventory_deduplicates_molecules_but_preserves_full_pool(tmp_path):
    goal = {"candidate_limit":1}
    rows = assess_candidates([row(),row(),row("AACCTTGG")],goal)
    outputs = write_assessment_artifacts(rows,rows[:2],goal,tmp_path)
    full = json.loads(open(outputs["candidate_assessments_json"]).read())
    short = json.loads(open(outputs["recommended_shortlist_json"]).read())
    assert full["pool_row_count"] == 3 and full["pool_molecule_count"] == 2
    assert short["count"] == 1
    assert short["records"][0]["recommendation_status"] == "provisional"
    assert "unmodified" in open(outputs["recommended_shortlist_md"]).read()


def test_count_question_is_late_and_explicit_count_or_all_is_respected():
    assert size_question({},False) is None
    assert size_question({},True)["restart_stage"] == "report"
    for goal in ({"candidate_limit":8},{"candidate_limit_per_site":3},{"shortlist_size_preference":"all_eligible"}):
        assert size_question(goal,True) is None
    assert parse_size_answer([{"answer":"I'd like 7 ASOs"}]) == {"candidate_limit":7,"candidate_minimum":7}
    assert parse_size_answer([{"answer":"all eligible"}]) == {"shortlist_size_preference":"all_eligible"}
    for answer in ("0","-5","5 per site","5 or 10","whatever"):
        with pytest.raises(ValueError):
            parse_size_answer([{"answer":answer}])


def test_empty_search_does_not_wait_for_an_unasked_count(tmp_path):
    outputs = write_assessment_artifacts([],[],{},tmp_path)
    result = json.loads(open(outputs["recommended_shortlist_json"]).read())
    assert result["status"] == "no_eligible_candidates"
    assert result["count"] == 0


def test_uniform_bond_policy_is_not_a_position_vector():
    requirement = {"field":"linkages","operator":"equals","value_json":json.dumps("all_PS"),"quote":"full PS"}
    require_native_requirement_capabilities([requirement])
    assert _constraint_matches(requirement,["PS"]*17)
    assert not _constraint_matches(requirement,["PS"]*16+["PO"])
    assert not _constraint_matches({**requirement,"value_json":json.dumps(["PS"])},["PS"]*17)


@pytest.mark.parametrize("text,expected",[
    ("min gc = 0 and max gc = 1", (0.0,1.0,False,False)),
    ("minimum GC = 40%, maximum GC = 65%", (0.4,0.65,False,False)),
    ("GC_min=0.25 GC_max=0.8", (0.25,0.8,False,False)),
    ("GC = 0", (0.0,0.0,False,False)),
    ("GC > 0 and < 1", (0.0,1.0,True,True)),
])
def test_named_gc_bounds_are_not_exact_gc_equalities(text,expected):
    from aso_agents.parameter_constraints import _extract_gc_from_texts
    assert _extract_gc_from_texts([None,"",text]) == expected
