import json
from pathlib import Path

import pytest

from aso_agents import pipeline_tools as tools
from aso_agents.parameter_constraints import get_binding_count_from_row, evaluate_row_against_constraints


@pytest.mark.parametrize("capped_species", ["human", "mouse"])
def test_mapper_cap_and_unknown_counts_survive_species_aggregation(tmp_path, monkeypatch, capped_species):
    path = tmp_path / "tiles.json"
    path.write_text(json.dumps({"rows": [{"Sequence": "ACGTCATGAGTTCGATCGTA", "source_modality": "gapmer"}]}))
    goal = {"species": "human", "gene_id": "GENE_A", "cross_species_requirement": {
        "required": False, "species": ["human", "mouse"]}}

    def mapper(filename, species_goal, constraints):
        payload = json.loads(Path(filename).read_text())
        row = payload["rows"][0]
        assert "off_target_mapping_capped" not in row
        row.update(off_target_gene_count=0, Times_mapped_off_target_genes=0,
                   off_target_transcript_count=0, Off_target_gene_ids=[], Off_target_genes=[])
        if species_goal["species"] == capped_species:
            row.update(off_target_mapping_capped=True, off_target_mapping_count_lower_bound=2001)
        Path(filename).write_text(json.dumps(payload))
        return {"junction_mapping": {"junctions_scanned": 1}}

    monkeypatch.setattr(tools, "_run_offtarget_expression_with_timeout", lambda **kwargs: {"status": "complete"})
    result = tools._annotate_gapmer_offtargets_by_species(
        initial_tiles_path=str(path), structured_goal=goal, parameter_constraints={}, annotate_offtargets=mapper)
    row = json.loads(path.read_text())["rows"][0]
    assert result["complete"] is True  # search execution completed; per-row evidence can be incomplete
    capped = row["off_target_by_species"][capped_species]
    assert capped["off_target_mapping_capped"] is True
    assert capped["off_target_mapping_count_lower_bound"] == 2001
    assert capped["off_target_count_status"] == "incomplete"
    assert capped["off_target_observed_gene_count"] == 0
    assert capped["off_target_gene_count"] is None and capped["Times_mapped_off_target_genes"] is None
    assert row["off_target_screening_status_by_species"][capped_species] == "incomplete"
    assert row["worst_species_off_target_gene_count"] is None
    assert row["worst_species_off_target_expression_risk"] is None
    assert row["worst_species_off_target_expression_incomplete"] is True
    other = "mouse" if capped_species == "human" else "human"
    assert row["off_target_by_species"][other]["off_target_gene_count"] == 0
    if capped_species == "human":
        assert get_binding_count_from_row(row) is None
        assert evaluate_row_against_constraints(row, {"max_binding_sites": 0})[0] is False
        assert evaluate_row_against_constraints(row, {"max_binding_sites": 5000})[0] is False
        normalized = tools._normalized_generated_candidate_row(row)
        assert normalized["off_target_mapping_capped"] is True
        assert normalized["off_target_count_status"] == "incomplete"


def test_unknown_species_gene_count_never_becomes_zero_or_complete_risk():
    row = {}
    tools._apply_gapmer_species_risk_summary(row, {"human": {}, "mouse": {"off_target_gene_count": 0}}, {}, "human")
    assert row["worst_species_off_target_gene_count"] is None
    assert row["off_target_expression_risk_by_species"]["human"]["value"] is None
    assert row["off_target_expression_risk_by_species"]["mouse"]["value"] == 0
    assert row["worst_species_off_target_expression_incomplete"] is True


def test_completed_zero_gene_screen_still_satisfies_zero_hit_constraint():
    row = {"off_target_count_status": "complete_within_declared_scope", "off_target_gene_count": 0,
           "source_modality": "gapmer", "Sequence": "ACGTCATGAGTTCGATCGTA"}
    assert get_binding_count_from_row(row) == 0
    assert evaluate_row_against_constraints(row, {"max_binding_sites": 0})[0] is True
