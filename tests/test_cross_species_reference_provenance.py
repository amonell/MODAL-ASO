"""Cross-species evidence reports only the references actually searched."""

import json


def test_species_offtarget_reference_provenance_reaches_candidate_rows(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools

    input_path = tmp_path / "tiles.json"
    input_path.write_text(json.dumps({"rows": [{"Sequence": "ACGTACGTACGT"}]}))
    monkeypatch.setattr(pipeline_tools, "_enrich_exact_off_target_bins_for_tiles",
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline_tools, "_run_offtarget_expression_with_timeout",
                        lambda **kwargs: {})

    def fake_mapping(path, goal, constraints):
        payload = json.loads(open(path).read())
        for row in payload["rows"]:
            row.update(off_target_count_status="complete_within_declared_scope",
                       Off_target_exact_count=0, Off_target_mismatch_budget=0)
        open(path, "w").write(json.dumps(payload))
        species = goal["species"]
        return {
            "junction_mapping": {"junctions_scanned": 10},
            "search_provenance": {"species": species, "assembly": "assembly-A",
                                  "genomic_index": f"/references/{species}/index",
                                  "exon_annotation": f"/references/{species}/annotation.gtf",
                                  "genomic_search": "exact", "junction_search": "complete"},
        }

    result = pipeline_tools._annotate_gapmer_offtargets_by_species(
        initial_tiles_path=str(input_path),
        structured_goal={"species": "human", "gene_symbol": "GENE",
                         "cross_species_targets": {
                             "human": {"status": "resolved", "gene_symbol": "GENE"},
                             "mouse": {"status": "resolved", "gene_symbol": "Gene"},
                         },
                         "species_targets": [{"species": "human"}, {"species": "mouse"}]},
        parameter_constraints=None,
        annotate_offtargets=fake_mapping,
    )

    row = json.loads(input_path.read_text())["rows"][0]
    assert result["status_by_species"] == {"human": "complete", "mouse": "complete"}
    assert set(row["off_target_search_provenance_by_species"]) == {"human", "mouse"}
    assert row["off_target_search_provenance_by_species"]["mouse"]["genomic_index"] == "/references/mouse/index"


def test_missing_junction_search_cannot_be_reported_complete(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools

    input_path = tmp_path / "tiles.json"
    input_path.write_text(json.dumps({"rows": [{"Sequence": "ACGTACGTACGT"}]}))
    monkeypatch.setattr(pipeline_tools, "_enrich_exact_off_target_bins_for_tiles",
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline_tools, "_run_offtarget_expression_with_timeout",
                        lambda **kwargs: {})

    def partial_mapping(path, goal, constraints):
        payload = json.loads(open(path).read())
        payload["rows"][0]["off_target_count_status"] = "complete_within_declared_scope"
        open(path, "w").write(json.dumps(payload))
        return {"junction_mapping": {"error": "junction_reference_unavailable"},
                "search_provenance": {"species": goal["species"], "junction_search": "unavailable"}}

    result = pipeline_tools._annotate_gapmer_offtargets_by_species(
        initial_tiles_path=str(input_path),
        structured_goal={"species": "human", "gene_symbol": "GENE"},
        parameter_constraints=None,
        annotate_offtargets=partial_mapping,
    )

    row = json.loads(input_path.read_text())["rows"][0]
    assert result["status_by_species"] == {"human": "partial_junction_unavailable"}
    assert row["off_target_screening_status_by_species"] == {"human": "incomplete"}
    assert row["off_target_by_species"]["human"]["off_target_count_status"] == "incomplete"
