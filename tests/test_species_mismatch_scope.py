"""Species aggregation must retain the search actually performed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aso_agents.mismatch_screening import annotate_mismatch_bins


def _run(tmp_path, monkeypatch, budget, primary="human", fault=None, stale=False):
    from aso_agents import pipeline_tools as pipeline

    sequence = "ACGTTACCGAGTACGATGCA"
    path = tmp_path / "tiles.json"
    old = {
        "off_target_unclassified_alignments": True,
        "off_target_observed_gene_ids_by_mismatch": {"0mm": ["STALE"]},
        "off_target_observed_genes_by_mismatch": {"0mm": ["Stale"]},
        "Off_target_1mm_count": 99,
        "Off_target_2mm_gene_ids": ["STALE"],
        "Off_target_3mm_genes": ["Stale"],
        "off_target_query_sequence": "STALE",
        "off_target_query_role": "previous species",
        "off_target_count_scope": {"mismatch_budget": 3, "species": "previous"},
    }
    path.write_text(json.dumps({"rows": [{"Sequence": sequence, **(old if stale else {})}]}))
    configs = {species: SimpleNamespace(
        bowtie_prefix=tmp_path / species / "index", gtf_path=tmp_path / species / "ref.gtf",
        genome_fasta=tmp_path / species / "genome.fa", ensembl_assembly=species + "-assembly")
        for species in ("human", "mouse")}
    monkeypatch.setattr(pipeline, "get_species_config", configs.__getitem__)
    monkeypatch.setattr(pipeline, "_run_offtarget_expression_with_timeout", lambda **kwargs: {})
    expected = {}

    def mapping(filename, goal, constraints):
        data = json.loads(Path(filename).read_text())
        row = data["rows"][0]
        if stale:
            assert not (set(old) & set(row)), "Previous species evidence reached a new search"
        species = goal["species"]
        # A gene can appear in more than one mismatch bin. Union counts cannot
        # be substituted for exact counts or calculated by summing bin counts.
        bins = {f"{mm}mm": ([species + "_shared"] if mm < 2 else []) +
                [species + f"_{mm}_{i}" for i in range(mm + (species == "mouse"))]
                for mm in range(budget + 1)}
        names = {k: [value.title() for value in values] for k, values in bins.items()}
        expected[species] = bins
        row.update(Off_target_gene_ids=sorted(set().union(*map(set, bins.values()))),
                   Off_target_genes=sorted(set().union(*map(set, names.values()))),
                   off_target_observed_gene_ids_by_mismatch=bins,
                   off_target_observed_genes_by_mismatch=names,
                   off_target_unclassified_alignments=fault == "unclassified",
                   off_target_mapping_capped=fault == "capped")
        annotate_mismatch_bins(row, budget, incomplete=bool(fault))
        Path(filename).write_text(json.dumps(data))
        return {"mismatch_budget": budget,
                "junction_mapping": {"skipped_unresolved_strand": 1} if fault == "junction" else {},
                "search_provenance": {"species": species, "assembly": configs[species].ensembl_assembly}}

    result = pipeline._annotate_gapmer_offtargets_by_species(
        initial_tiles_path=str(path),
        structured_goal={"species": primary, "gene_id": "INTENDED",
                         "species_targets": [{"species": "human"}, {"species": "mouse"}],
                         "cross_species_targets": {
                             s: {"status": "resolved", "gene_id": s + "_INTENDED"}
                             for s in configs}},
        parameter_constraints=None, annotate_offtargets=mapping)
    return json.loads(path.read_text())["rows"][0], result, expected


@pytest.mark.parametrize("budget", range(4))
@pytest.mark.parametrize("primary", ["human", "mouse"])
def test_species_counts_preserve_exact_substitution_and_union_scopes(tmp_path, monkeypatch, budget, primary):
    row, result, expected = _run(tmp_path, monkeypatch, budget, primary)
    assert result["status_by_species"] == {"human": "complete", "mouse": "complete"}
    for species, bins in expected.items():
        value = row["off_target_by_species"][species]
        assert value["Off_target_mismatch_budget"] == budget
        assert value["Off_target_exact_count"] == len(bins["0mm"])
        assert value["Off_target_total_upto_mismatch_count"] == len(set().union(*map(set, bins.values())))
        assert value["off_target_observed_gene_ids_by_mismatch"] == bins
        for mm in range(4):
            assert value[f"Off_target_{mm}mm_gene_ids"] == (sorted(bins[f"{mm}mm"]) if mm <= budget else None)
            if mm:
                assert value[f"Off_target_{mm}mm_count"] == (len(bins[f"{mm}mm"]) if mm <= budget else None)
        assert value["off_target_count_scope"]["mismatch_budget"] == budget
        assert value["off_target_count_scope"]["excluded_gene_id"] == species + "_INTENDED"
        assert species in value["off_target_count_scope"]["bowtie_index_prefix"]
        assert value["off_target_query_sequence"] == row["Sequence"]
    assert row["Off_target_exact_count"] == len(expected[primary]["0mm"])
    assert row["off_target_observed_gene_ids_by_mismatch"] == expected[primary]
    assert row["worst_species_off_target_gene_count"] == max(
        len(set().union(*map(set, bins.values()))) for bins in expected.values())


@pytest.mark.parametrize("budget", range(4))
@pytest.mark.parametrize("fault", ["unclassified", "capped", "junction"])
def test_species_partial_search_keeps_observations_and_unknown_counts(tmp_path, monkeypatch, budget, fault):
    row, _, expected = _run(tmp_path, monkeypatch, budget, fault=fault)
    for species, bins in expected.items():
        value = row["off_target_by_species"][species]
        assert value["Off_target_mismatch_budget"] == budget
        assert value["off_target_count_status"] == "incomplete"
        assert value["Off_target_exact_count"] is None
        assert value["Off_target_total_upto_mismatch_count"] is None
        assert value["off_target_observed_gene_ids_by_mismatch"] == bins
        assert value["off_target_observed_gene_count"] == len(set().union(*map(set, bins.values())))
        assert value["off_target_unclassified_alignments"] is (fault == "unclassified")
        for mm in range(1, 4):
            assert value[f"Off_target_{mm}mm_count"] is None
    assert row["worst_species_off_target_gene_count"] is None


def test_previous_species_scope_is_removed_before_each_new_search(tmp_path, monkeypatch):
    row, _, _ = _run(tmp_path, monkeypatch, 1, stale=True)
    assert row["off_target_unclassified_alignments"] is False
    assert row["Off_target_2mm_gene_ids"] is None
