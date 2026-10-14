from copy import deepcopy

import pytest

import run_goal_intake as intake
from aso_agents.pipeline_tools import _derive_exploratory_steric_programs, _program_isoform_identity_aliases


def _setup(monkeypatch, *, annotated=True):
    canonical = "ENST00000000001"
    alternate = "ENST00000000002"

    def structure(transcript_id):
        positions = [100, 200, 300, 400, 500, 600] if transcript_id == canonical else [100, 200, 300, 500, 600]
        return {"id": transcript_id, "seq_region_name": "5", "strand": 1,
                "exons_tx": [{"index": i, "start": start, "end": start + 49, "contains_coding": True}
                             for i, start in enumerate(positions, 1)]}

    alternate_structure = structure(alternate)
    goal = {
        "gene_symbol": "GENE_A", "species": "human", "intent": "splice_modulation",
        "exon_label": 4, "desired_effect": "inclusion", "focus_exons_only": True,
        "transcript_policy": "agent_choice",
        "splice_isoform_objectives": [
            {"objective_scope": "default_exon_context", "exon_label": 4,
             "desired_effect": "inclusion", "desired_abundance": "increase", "role": "beneficial"},
            {"objective_scope": "specific_isoform", "isoform_label": "alternate",
             "desired_abundance": "decrease", "role": "pathogenic"},
        ],
        "preseeded_isoform_transcripts": {"alternate": {
            "transcript_id": alternate, "metadata": {"id": alternate},
            "chrom": "5", "strand": "+", "exons_tx": alternate_structure["exons_tx"],
            "sites": intake._extract_splice_sites(alternate_structure),
        }},
    }
    monkeypatch.setattr(intake, "list_transcripts", lambda **_: {"transcripts": [
        {"id": alternate, "length": 999999}, {"id": canonical, "is_canonical": annotated},
    ]})
    monkeypatch.setattr(intake, "_fetch_transcript_structure", structure)
    monkeypatch.setattr(intake, "_generate_splice_strategy_analysis", lambda *_, **__: None)
    monkeypatch.setattr(intake, "_cached_authoritative_exon_nomenclature_mapping", lambda *_, **__: {"status": "ordinal_default"})
    return goal, canonical


def test_default_exon_reference_survives_beside_a_named_isoform(monkeypatch):
    goal, canonical = _setup(monkeypatch)
    objectives = deepcopy(goal["splice_isoform_objectives"])
    result = intake._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    assert set(result["isoform_transcripts"]) == {canonical, "alternate"}
    assert goal["splice_isoform_objectives"] == objectives
    acceptors = {(site["position"], site["delta_direction"]) for site in result["splice_site_plan"] if site["site"] == "acceptor"}
    assert (400, "increase") in acceptors
    # This synthetic downstream boundary is physically shared by both isoforms;
    # retain the alternate's requested direction even if the combined direction differs.
    assert any(site["position"] == 500 and site["site"] == "acceptor"
               and site["delta_per_isoform"]["alternate"] == "decrease"
               for site in result["splice_site_plan"])
    assert goal["default_exon_context_reference"]["transcript_id"] == canonical


@pytest.mark.parametrize("restriction", ["unannotated", "user_locked", "explicit_reference"])
def test_default_context_does_not_override_a_restricted_reference(restriction, monkeypatch):
    goal, canonical = _setup(monkeypatch, annotated=restriction != "unannotated")
    if restriction == "user_locked":
        goal["transcript_policy"] = "user_locked"
    if restriction == "explicit_reference":
        goal["splice_isoform_objectives"][0]["transcript_id"] = "ENST00000000999"
    result = intake._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    assert canonical not in result["isoform_transcripts"]
    assert not goal.get("default_exon_context_reference")


def test_existing_canonical_isoform_is_reused_without_duplicate_members(monkeypatch):
    goal, canonical = _setup(monkeypatch)
    structure = intake._fetch_transcript_structure(canonical)
    goal["preseeded_isoform_transcripts"]["full length"] = {
        "transcript_id": canonical, "metadata": {"id": canonical}, "chrom": "5", "strand": "+",
        "exons_tx": structure["exons_tx"], "sites": intake._extract_splice_sites(structure),
    }
    result = intake._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    assert set(result["isoform_transcripts"]) == {"full length", "alternate"}
    assert goal["default_exon_context_reference"]["isoform_label"] == "full length"
    assert any(site["position"] == 400 and site["delta_direction"] == "increase" for site in result["splice_site_plan"])


def test_default_and_named_routes_keep_their_own_reference_sites(monkeypatch):
    goal, canonical = _setup(monkeypatch)
    result = intake._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    goal["splice_site_plan"] = result["splice_site_plan"]
    default = {"program_id": "default", "exon_label": 4, "desired_delta": "increase", "site_kind": "acceptor"}
    named = {"program_id": "named", "exon_label": 4, "desired_delta": "decrease", "site_kind": "acceptor", "isoform_label": "alternate"}
    assert canonical.casefold() in _program_isoform_identity_aliases(goal, default)
    assert canonical.casefold() not in _program_isoform_identity_aliases(goal, named)
    for program, expected in [(default, 400), (named, 500)]:
        derived, _ = _derive_exploratory_steric_programs(structured_goal=goal, base_program=program)
        anchors = {anchor["position"] for item in derived for anchor in item["anchor_site_records"]}
        assert anchors == {expected}


@pytest.mark.parametrize("change", [{"exon_label": 3}, {"desired_delta": "decrease"}, {"exon_labels": [4, 5]}])
def test_unrelated_programs_do_not_inherit_the_default_reference(change, monkeypatch):
    goal, _ = _setup(monkeypatch)
    intake._resolve_isoform_transcripts(goal, deepcopy(goal), allow_interactive=False)
    program = {"exon_label": 4, "desired_delta": "increase", **change}
    assert not _program_isoform_identity_aliases(goal, program)
