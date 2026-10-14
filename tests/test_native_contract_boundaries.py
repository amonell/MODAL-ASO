"""Contracts at native boundaries, independent of genes or benchmark outcomes."""
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as tools


def test_optional_secondary_species_cannot_replace_the_primary_target():
    context = {"gene_symbol": "GENE_A", "species": "human", "structured_goal": {
        "gene_symbol": "GENE_A", "species": "human", "intent": "binding",
        "target_context": "general baseline",
        "cross_species_requirement": {"required": False, "species": ["mouse"],
                                      "return_nonconserved_candidates": True},
        "design_programs": [{"program_id": "block", "modality": "steric", "intent": "binding"}],
    }}
    original = deepcopy(context)
    task = tools._finalize_design_task_contract(
        {"task_id": "design", "task_type": "design_new_asos", "context": context}, {})
    goal = task["context"]["structured_goal"]
    assert context == original
    assert task["context"]["species"] == goal["species"] == "human"
    assert goal["cross_species_requirement"]["species"] == ["human", "mouse"]
    assert goal["species_targets"][0] == {"gene_symbol": "GENE_A", "species": "human"}
    assert not tools._cross_species_required(None, goal)


@pytest.mark.parametrize("retention", [{"return_nonconserved_candidates": True},
                                     {"retain_only_passing": False}, {"required": False}])
def test_explicit_retention_survives_both_conservation_policy_paths(retention):
    goal = {"species": "human", "cross_species_requirement": {
        "required": True, "species": ["human", "mouse"], **retention}}
    assert tools._request_contract_cross_species_optional(goal)
    assert not tools._cross_species_required(None, goal)


def test_required_conservation_is_not_relaxed():
    goal = {"species": "human", "cross_species_requirement": {
        "required": True, "species": ["human", "mouse"]}}
    assert tools._cross_species_required(None, goal)


@pytest.mark.parametrize("primary", ["ornithorhynchus_anatinus", "unresolved_organism_label"])
def test_primary_identity_is_not_restricted_to_a_local_species_whitelist(primary):
    # Unknown labels must reach the reference resolver/clarification unchanged,
    # rather than silently becoming the first conservation comparator.
    assert tools._request_contract_anchor_species({"species": primary}, {
        "cross_species_requirement": {"species": ["mouse", "human"]}}) == primary


def test_reference_and_metadata_files_are_never_inferred_to_be_aso_panels(tmp_path):
    panel = tmp_path / "panel.csv"
    panel.write_text("name,sequence\none,AACCGGTTAACCGGTT\n")
    reference = tmp_path / "reference.fasta"
    reference.write_text(">transcript\nAACCGGTTAACCGGTT\nTTGGCCAATTGGCCAA\n")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("name,sequence\nreference,AACCGGTTAACCGGTT\n")
    rows = tools._normalize_provided_asos({"provided_aso_files": [str(panel)],
        "artifact_refs": [{"path": str(reference), "artifact_type": "reference_sequence"},
                          {"path": str(metadata), "artifact_type": "reference_manifest"}]})
    assert [(r["name"], r["source_file"]) for r in rows] == [("one", str(panel))]


def test_explicit_panel_role_still_loads_an_artifact(tmp_path):
    path = tmp_path / "observations.tsv"
    path.write_text("name\tsequence\none\tAACCGGTTAACCGGTT\n")
    rows = tools._normalize_provided_asos({"artifact_refs": [
        {"path": str(path), "artifact_type": "provided_aso_set"}]})
    assert [r["name"] for r in rows] == ["one"]


def test_materialized_annotation_regions_can_seed_without_an_unrelated_base():
    import run_goal_intake as cli
    assert cli._constraints_can_seed_without_base([
        {"source_type": "rbp_binding", "materialized_rows": [["chrN", 100, 150, "r", 0, "+"]]},
        {"source_type": "mirna_binding", "materialized_rows": [["chrN", 170, 178, "m", 0, "+"]]},
    ])
    assert not cli._constraints_can_seed_without_base([
        {"source_type": "rbp_binding", "materialized_rows": []}])


def test_annotation_seeding_preserves_explicit_search_bounds():
    import run_goal_intake as cli
    goal = {"materialized_region_constraints": {"global": [
        {"source_type": "rbp_binding", "materialized_rows": [["chrN", 100, 150, "r", 0, "+"]]},
        {"source_type": "mirna_binding", "materialized_rows": [["chrN", 300, 320, "m", 0, "+"]]},
    ], "by_program": {}}}
    seeded, _ = cli._apply_region_constraints_to_rows(goal, [], output_space="genomic",
                                                     expand_for_tile_overlap=False)
    assert [row[1:3] for row in seeded] == [[100, 150], [300, 320]]
    bounded, _ = cli._apply_region_constraints_to_rows(goal,
        [["chrN", 120, 200, "requested", 0, "+"]], output_space="genomic",
        expand_for_tile_overlap=False)
    assert [row[1:3] for row in bounded] == [[120, 150]]
