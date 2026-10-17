"""An absent species screen cannot establish a complete cross-species risk."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _old_row():
    return {
        "Sequence": "ACGTTACCGAGTACGATGCA", "candidate_id": "candidate-A",
        "off_target_gene_count": 91, "Off_target_exact_count": 91,
        "Off_target_genes": ["OLD"], "Off_target_expression_weighted_risk": 91,
        "off_target_count_scope": {"source": "previous screen"},
    }


@pytest.mark.parametrize("primary", ["human", "mouse"])
@pytest.mark.parametrize("failure", ["mapping_failed", "unavailable"])
def test_missing_species_makes_aggregate_unknown(tmp_path, monkeypatch, primary, failure):
    from aso_agents import pipeline_tools as pipeline
    from aso_agents.mismatch_screening import annotate_mismatch_bins

    path = tmp_path / "tiles.json"
    path.write_text(json.dumps({"rows": [_old_row()]}))
    monkeypatch.setattr(pipeline, "get_species_config", lambda species: SimpleNamespace(
        bowtie_prefix=tmp_path / species / "index", gtf_path=tmp_path / species / "ref.gtf",
        genome_fasta=tmp_path / species / "genome.fa", ensembl_assembly=species))
    monkeypatch.setattr(pipeline, "_run_offtarget_expression_with_timeout", lambda **kwargs: {})
    calls = []

    def mapping(filename, goal, constraints):
        calls.append(goal["species"])
        if goal["species"] == "mouse":
            return {"error": "reference_unavailable"}
        data = json.loads(Path(filename).read_text())
        row = data["rows"][0]
        row.update(Off_target_gene_ids=[], Off_target_genes=[],
                   off_target_observed_gene_ids_by_mismatch={"0mm": []},
                   off_target_observed_genes_by_mismatch={"0mm": []})
        annotate_mismatch_bins(row, 0)
        Path(filename).write_text(json.dumps(data))
        return {"mismatch_budget": 0, "junction_mapping": {}}

    result = pipeline._annotate_gapmer_offtargets_by_species(
        initial_tiles_path=str(path), structured_goal={
            "species": primary, "gene_id": "TARGET",
            "off_target_species": ["human", "mouse"],
            "cross_species_targets": {
                "human": {"status": "resolved", "gene_id": "human_TARGET"},
                "mouse": {"status": "unavailable" if failure == "unavailable" else "resolved",
                          "gene_id": "mouse_TARGET"}}},
        parameter_constraints=None, annotate_offtargets=mapping)
    row = json.loads(path.read_text())["rows"][0]
    assert result["status_by_species"]["mouse"] == failure
    assert not result["complete"]
    assert row["candidate_id"] == "candidate-A"
    assert row["worst_species_off_target_gene_count"] is None
    assert row["worst_species_off_target_expression_risk"] is None
    assert row["worst_species_off_target_expression_incomplete"] is True
    assert row["off_target_expression_risk_by_species"]["mouse"]["incomplete"] is True
    assert row["off_target_by_species"]["human"]["Off_target_exact_count"] == 0
    assert ("mouse" in calls) is (failure == "mapping_failed")
    if primary == "mouse":
        for field in pipeline._SPECIES_OFFTARGET_FIELDS + pipeline._SPECIES_EXPRESSION_FIELDS:
            assert field not in row, field
    else:
        assert row["Off_target_exact_count"] == 0


@pytest.mark.parametrize("primary", ["human", "mouse"])
@pytest.mark.parametrize("status", ["mapping_failed", "unavailable", "incomplete"])
def test_refresh_preserves_unknown_species_and_clears_stale_primary(tmp_path, primary, status):
    from aso_agents import pipeline_tools as pipeline

    row = _old_row()
    row.update(off_target_screening_status_by_species={"human": "complete", "mouse": status},
               off_target_by_species={"human": {"off_target_gene_count": 2}},
               off_target_expression_by_species={"human": {"Off_target_expression_weighted_risk": 0.5}})
    path = tmp_path / "scored.json"
    path.write_text(json.dumps({"rows": [row]}))
    assert pipeline._refresh_gapmer_species_risk_file(str(path), primary)["status"] == "complete"
    actual = json.loads(path.read_text())["rows"][0]
    assert actual["worst_species_off_target_gene_count"] is None
    assert actual["worst_species_off_target_expression_risk"] is None
    assert actual["worst_species_off_target_expression_incomplete"] is True
    if primary == "mouse":
        assert "off_target_gene_count" not in actual
        assert "Off_target_expression_weighted_risk" not in actual
    else:
        assert actual["off_target_gene_count"] == 2


@pytest.mark.parametrize("primary", ["human", "mouse"])
def test_no_ortholog_is_not_a_failed_screen(primary):
    from aso_agents import pipeline_tools as pipeline

    row = _old_row()
    row["off_target_screening_status_by_species"] = {
        "human": "complete", "mouse": "not_applicable_no_ortholog"}
    pipeline._apply_gapmer_species_risk_summary(
        row, {"human": {"off_target_gene_count": 0}}, {"human": {}}, primary)
    assert row["worst_species_off_target_gene_count"] == 0
    assert row["worst_species_off_target_expression_risk"] == 0
    assert row["worst_species_off_target_expression_incomplete"] is False
    assert "mouse" not in row["off_target_expression_risk_by_species"]
    if primary == "mouse":
        assert "off_target_gene_count" not in row
        assert "Off_target_expression_weighted_risk" not in row


def test_all_failed_screens_are_unknown_not_complete():
    from aso_agents import pipeline_tools as pipeline

    row = _old_row()
    row["off_target_screening_status_by_species"] = {
        "human": "mapping_failed", "mouse": "unavailable"}
    pipeline._apply_gapmer_species_risk_summary(row, {}, {}, "human")
    assert row["worst_species_off_target_gene_count"] is None
    assert row["worst_species_off_target_expression_risk"] is None
    assert row["worst_species_off_target_expression_incomplete"] is True
    assert set(row["off_target_expression_risk_by_species"]) == {"human", "mouse"}
    assert "Off_target_exact_count" not in row


def test_absent_legacy_screen_is_unknown():
    from aso_agents import pipeline_tools as pipeline

    row = _old_row()
    pipeline._apply_gapmer_species_risk_summary(row, {}, {}, "human")
    assert row["worst_species_off_target_gene_count"] is None
    assert row["worst_species_off_target_expression_incomplete"] is True
    assert "off_target_gene_count" not in row


@pytest.mark.parametrize("primary", ["human", "mouse"])
def test_complete_species_retain_counts_and_primary_binding(primary):
    from aso_agents import pipeline_tools as pipeline

    row = _old_row()
    row["off_target_screening_status_by_species"] = {"human": "complete", "mouse": "complete"}
    counts = {"human": 2, "mouse": 4}
    risks = {"human": 0.5, "mouse": 1.5}
    pipeline._apply_gapmer_species_risk_summary(
        row, {s: {"off_target_gene_count": n} for s, n in counts.items()},
        {s: {"Off_target_expression_weighted_risk": n} for s, n in risks.items()}, primary)
    assert row["worst_species_off_target_gene_count"] == 4
    assert row["worst_species_off_target_expression_risk"] == 1.5
    assert row["worst_species_off_target_expression_incomplete"] is False
    assert row["off_target_gene_count"] == counts[primary]
    assert row["Off_target_expression_weighted_risk"] == risks[primary]
    assert "Off_target_exact_count" not in row
