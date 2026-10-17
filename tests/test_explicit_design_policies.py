"""Neutral physical-output and policy controls, independent of benchmark cases."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

import run_goal_intake as runtime
from aso_chemistry import encode_idt_oligo
from aso_agents import pipeline_tools as pipeline
from aso_agents.splice_reference_scope import splice_site_reference_scope, scoped_splice_evidence


def test_explicit_nonselection_reaches_native_tiler_without_chemical_encoding(tmp_path):
    goal = {"chemistry_policy": "leave_unspecified", "design_programs": [
        {"program_id": "sequence_only", "modality": "steric", "length": 23}]}
    profiles = pipeline._ensure_program_chemistry_profiles(structured_goal=goal, final_payload={})
    rec = profiles["by_program"]["sequence_only"]["recommendation"]
    assert rec["CHEMISTRY"] == "unspecified" and rec["length"] == 23 and not rec["platform"]
    sequence = "ACGTCGATCGATCGATCGATCGA"
    assert len(sequence) == 23
    assert encode_idt_oligo(sequence, rec["CHEMISTRY"]) == ""
    idt, synthesis, error = runtime._format_idt_and_synthesis(sequence, "unspecified")
    assert idt == "" and synthesis == sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1] and error is None
    path = Path(__file__).parents[1] / "aso-designer-main/aso-designer-main/filter_seqs.py"
    spec = importlib.util.spec_from_file_location("sequence_only_filter", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.apply_configuration(synthesis, "unspecified") == ""


def test_mixed_programs_keep_independent_chemistry_on_reentry():
    goal = {"design_programs": [
        {"program_id": "chemical", "modality": "gapmer", "length": 18, "chemistry": "4-10-4"},
        {"program_id": "deferred", "modality": "steric", "length": 23, "chemistry_policy": "leave_unspecified"}]}
    final = {}
    for _ in range(2):
        profiles = pipeline._ensure_program_chemistry_profiles(structured_goal=goal, final_payload=final)
        assert profiles["by_program"]["chemical"]["recommendation"]["CHEMISTRY"] == "4-10-4"
        assert profiles["by_program"]["deferred"]["recommendation"]["CHEMISTRY"] == "unspecified"
        assert profiles["by_program"]["deferred"]["recommendation"]["length"] == 23


def test_nonselection_is_distinct_from_missing_or_conflicting_chemistry():
    assert pipeline._unselected_chemistry_payload({}) == {}
    assert pipeline._fallback_chemistry_payload("steric", {})["recommendation"]["CHEMISTRY"] == "steric-hindrance"
    with pytest.raises(ValueError, match="conflicts"):
        pipeline._unselected_chemistry_payload({"chemistry_policy": "leave_unspecified", "chemistry": "4-10-4"})
    with pytest.raises(ValueError, match="requires"):
        encode_idt_oligo("ACGT" * 5, "4-10-4")
    with pytest.raises(ValueError, match="unambiguous"):
        encode_idt_oligo("ACGTN", "unspecified")
    idt, _, error = runtime._format_idt_and_synthesis("ACGT" * 5, "4-10-4")
    assert not idt and error.startswith("chemistry_failed:")


def transcript(name, strand="+", shared=True):
    return {"transcript_id": name, "chrom": "chr6", "strand": strand,
            "exons_tx": [{"index": 1, "start": 11, "end": 40},
                         {"index": 2, "start": 101 if shared else 111, "end": 150 if shared else 160},
                         {"index": 3, "start": 211, "end": 240}]}


@pytest.mark.parametrize("strand,kind,position", [("+", "acceptor", 101), ("+", "donor", 150), ("-", "acceptor", 150), ("-", "donor", 101)])
def test_site_sharing_uses_physical_boundary_catalog_not_design_subset(strand, kind, position):
    a, b, c = transcript("TX_A.2", strand), transcript("TX_B.4", strand), transcript("TX_C.1", strand, False)
    goal = {"isoform_transcripts": {"a": a}, "all_transcripts_full": {"a": a, "alias_a": a, "b": b, "c": c}}
    before = deepcopy(goal)
    site = {"chrom": "6", "strand": strand, "position": position, "site": kind}
    scope = splice_site_reference_scope(site, goal)
    assert scope["present_in"] == ["TX_A.2", "TX_B.4"]
    assert scope["status"] == "shared" and scope["absent_in"] == ["TX_C.1"]
    assert not scope["complete_for_gene"] and goal == before
    payload = scoped_splice_evidence({"uniqueness": "unique", "present_in": ["a"]}, site, goal)
    assert "uniqueness" not in payload and payload["selected_transcript_context"]["uniqueness"] == "unique"
    assert payload["reference_scope"] == scope


def test_single_selected_transcript_cannot_establish_global_uniqueness():
    goal = {"isoform_transcripts": {"a": transcript("TX_A.2")}}
    site = {"chrom": "chr6", "strand": "+", "position": 101, "site": "acceptor"}
    assert splice_site_reference_scope(site, goal)["status"] == "specificity_unresolved"
    goal["all_transcripts_full"] = {**goal["isoform_transcripts"], "b": transcript("TX_B.1", shared=False)}
    assert splice_site_reference_scope(site, goal)["status"] == "single_catalog_match"
    goal["all_transcripts_full"]["b"].pop("exons_tx")
    assert splice_site_reference_scope(site, goal)["unresolved_transcript_ids"] == ["TX_B.1"]


@pytest.mark.parametrize("kind,position", [("acceptor", 11), ("donor", 240)])
def test_transcript_terminal_edges_are_not_splice_sites(kind, position):
    scope = splice_site_reference_scope({"chrom": "chr6", "strand": "+", "position": position, "site": kind},
                                        {"isoform_transcripts": {"a": transcript("TX_A.2")}})
    assert scope["present_in"] == []


@pytest.mark.parametrize("required", [False, True])
def test_generated_panel_optional_conservation_keeps_nonmatching_and_unknown_rows(tmp_path, required):
    run = tmp_path / "run"; (run / "tiling").mkdir(parents=True)
    goal = {"species": "human", "cross_species_requirement": {
        "species": ["human", "mouse"], "required": required, "return_nonconserved_candidates": not required}}
    (run / "goal_intake.json").write_text(json.dumps({"final_payload": {"structured_goal": goal}}))
    rows = [{"Sequence": sequence, "cross_species_required": True, "cross_species_pass": passed}
            for sequence, passed in [("AACCGGTTAACCGGTT", True), ("ACGTCGATCGATCGAT", False), ("TGCAACGTACGTACGT", None)]]
    (run / "tiling/steric_scored_uniqueness.json").write_text(json.dumps({"rows": rows}))
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(run))
    result = json.loads(Path(artifacts["generated_aso_set_json"]).read_text())
    assert len(result["rows"]) == (1 if required else 3)
    assert result["eligibility_filters"][0]["required"] is required


@pytest.mark.parametrize("modality", ["gapmer", "steric"])
@pytest.mark.parametrize("required", [True, False])
def test_annotation_does_not_convert_preference_into_requirement(tmp_path, monkeypatch, modality, required):
    path = tmp_path / "tiles.json"; path.write_text(json.dumps({"rows": [{"Sequence": "ACGTCGAT"}]}))
    goal = {"gene_symbol": "NEUTRAL", "species": "human", "cross_species_requirement": {"species": ["human", "mouse"], "required": required}}
    monkeypatch.setattr(runtime, "_resolve_cross_species_gene_targets", lambda *args: {
        s: {"status": "resolved", "gene_id": "NEUTRAL_" + s} for s in ("human", "mouse")})
    monkeypatch.setattr(runtime, "_resolve_cross_species_cdna_sequences", lambda *args, **kwargs: ({"TX": "GGACGTCGATTT"}, None))
    monkeypatch.setattr(runtime, "_resolve_cross_species_steric_sequences", lambda *args, **kwargs: ({"TX": "GGACGTCGATTT"}, None, "neutral_reference"))
    getattr(runtime, "_annotate_" + modality + "_cross_species_matches")(str(path), goal)
    row = json.loads(path.read_text())["rows"][0]
    assert row["cross_species_required"] is required
    assert row["cross_species_pass"] is True
    assert row["cross_species_match_placements"]["mouse"] == [{
        "reference_id": "TX", "start_1based": 3, "end_1based": 10,
        "coordinate_scope": "transcript_cdna" if modality == "gapmer"
        else "searched_steric_reference_sequence",
        **({"genomic_mapping_status": "reference_geometry_unavailable_or_inconsistent"}
           if modality == "gapmer" else {}),
    }]


def test_select_policy_preserves_explicit_inherited_architecture():
    copied = runtime._design_program_goal_copy({"chemistry": "4-10-4"}, {"chemistry_policy": "select"})
    assert copied["chemistry"] == "4-10-4"
