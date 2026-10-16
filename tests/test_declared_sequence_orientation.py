"""Input molecule identity must survive a missing binding site and file ingestion."""
from copy import deepcopy
import json

import jsonschema
import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.goal_contract import canonical_task_context_schema
from aso_agents.sequence_orientation import interpret_orientation, preferred_binding_hits


ASO = "AACCGGTA"
TARGET = "TACCGGTT"


def test_shared_file_role_survives_context_and_no_exact_binding(tmp_path):
    path = tmp_path / "panel.csv"
    path.write_text(f"name,sequence\nfirst,{ASO}\nsecond,{ASO}\n")
    envelope = {"default_context": {"provided_sequence_role": "antisense_oligo_sequence"}}
    task = {"context": {"provided_aso_files": [{"path": str(path)}]}}
    context = pipeline._task_context_from_request_envelope(envelope, task)
    rows = pipeline._task_context_provided_asos(context)
    assert len(rows) == 2 and len({row["input_row_id"] for row in rows}) == 2
    for row in rows:
        assert row["sequence"] == ASO
        assert row["comparison_sequence"] == TARGET
        assert row["sequence_role_source"] == "task_context"
        orientation = interpret_orientation(row)
        assert orientation["provided_sequence_role"] == "antisense_oligo_sequence"
        assert orientation["evaluated_molecule_sequence"] == ASO
        assert not orientation["requires_confirmation"]
        assert not orientation["original_molecule_exact_binding"]
    tiles, by_index = pipeline._build_provided_aso_scoring_tiles(rows)
    assert [row["Sequence"] for row in tiles] == [TARGET, TARGET]
    assert [row["provided_input_sequence"] for row in tiles] == [ASO, ASO]
    assert list(by_index) == [0, 1]


@pytest.mark.parametrize("role,query,molecule", [
    ("antisense_oligo_sequence", TARGET, ASO),
    ("target_site_sequence", ASO, TARGET),
    ("unknown", ASO, ASO),
])
@pytest.mark.parametrize("file_backed", [False, True])
def test_explicit_row_role_overrides_shared_role_and_chemistry(tmp_path, role, query, molecule, file_backed):
    item = {"name": "sample", "sequence": ASO, "sequence_role": role, "chemistry": "fully modified"}
    request = {"provided_sequence_role": "antisense_oligo_sequence", "provided_asos": [item]}
    if file_backed:
        path = tmp_path / "mixed.csv"
        path.write_text(f"name,sequence,sequence_role,chemistry\nsample,{ASO},{role},fully modified\n")
        request = {"provided_sequence_role": "antisense_oligo_sequence", "provided_aso_files": [str(path)]}
    before = deepcopy(request)
    row = pipeline._normalize_provided_asos(request)[0]
    assert row["declared_sequence_role"] == role
    assert row["sequence_role_source"] == "supplied_row"
    assert pipeline._provided_aso_scoring_sequence(row) == query
    assert interpret_orientation(row)["evaluated_molecule_sequence"] == molecule
    assert request == before


@pytest.mark.parametrize("header,role,query", [
    ("aso_sequence", "antisense_oligo_sequence", TARGET),
    ("oligo_sequence", "antisense_oligo_sequence", TARGET),
    ("target_sequence", "target_site_sequence", ASO),
    ("target_site_sequence", "target_site_sequence", ASO),
])
def test_delimited_semantic_sequence_column_preserves_role(tmp_path, header, role, query):
    path = tmp_path / "panel.tsv"
    path.write_text(f"name\t{header}\nexample\t{ASO}\n")
    row = pipeline._normalize_provided_asos({"provided_aso_files": [str(path)]})[0]
    assert row["declared_sequence_role"] == role
    assert pipeline._provided_aso_scoring_sequence(row) == query


def test_task_declaration_precedes_chemistry_hint_and_task_override_precedes_default():
    raw = {"context": {"provided_sequence_role": "target_site_sequence", "provided_asos": [
        {"sequence": ASO, "chemistry": "fully modified"}]}}
    context = pipeline._normalize_task_context(raw, task_type="evaluate_existing_aso_set",
        default_context={"provided_sequence_role": "antisense_oligo_sequence"}, structured_goal=None)
    row = pipeline._task_context_provided_asos(context)[0]
    assert row["declared_sequence_role"] == "target_site_sequence"
    assert interpret_orientation(row)["evaluated_molecule_sequence"] == TARGET


def test_unstated_role_is_not_inferred_from_task_chemistry_or_free_text():
    row = pipeline._normalize_provided_asos({"provided_asos": [ASO], "chemistry": "fully modified",
        "request_text": "A quoted description of an antisense oligo is not a typed input declaration."})[0]
    assert interpret_orientation(row)["provided_sequence_role"] == "unknown"
    assert pipeline._provided_aso_scoring_sequence(row) == ASO


def test_same_strand_hit_does_not_change_declared_molecule_or_scoring_query():
    row = pipeline._normalize_provided_asos({"provided_sequence_role": "antisense_oligo_sequence",
                                            "provided_asos": [ASO]})[0]
    row.update(comparison_sequence=ASO, normalized_sequence=ASO,
        mapping_hits=[{"match_orientation": "input_sequence", "matched_target_sequence": ASO}])
    assert preferred_binding_hits(row, row["mapping_hits"]) == []
    assert pipeline._provided_aso_scoring_sequence(row) == TARGET
    assert interpret_orientation(row)["evaluated_molecule_sequence"] == ASO
    assert not interpret_orientation(row)["requires_confirmation"]


def test_later_user_confirmation_controls_query_without_mutating_supplied_sequence():
    row = pipeline._normalize_provided_asos({"provided_sequence_role": "antisense_oligo_sequence",
                                            "provided_asos": [ASO]})[0]
    row["sequence_role_confirmation"] = {"role": "target_site_sequence"}
    assert pipeline._provided_aso_scoring_sequence(row) == ASO
    assert interpret_orientation(row)["evaluated_molecule_sequence"] == TARGET
    assert interpret_orientation(row)["sequence_role_source"] == "user_confirmation"
    assert row["sequence"] == ASO


@pytest.mark.parametrize("via_confirmation", [False, True])
def test_exact_match_cannot_resolve_explicitly_unknown_input_role(via_confirmation):
    row = pipeline._normalize_provided_asos({"provided_sequence_role": "unknown", "provided_asos": [ASO]})[0]
    if via_confirmation:
        row["declared_sequence_role"] = "antisense_oligo_sequence"
        row["sequence_role_confirmation"] = {"role": "unknown"}
    row["mapping_hits"] = [{"match_orientation": "reverse_complement", "matched_target_sequence": TARGET}]
    orientation = interpret_orientation(row)
    assert orientation["provided_sequence_role"] == "unknown"
    assert orientation["original_molecule_exact_binding"]
    assert orientation["evaluated_molecule_sequence"] == ASO
    assert "synthesized_antisense_sequence" not in orientation
    assert "conditional" in orientation["interpretation_note"]


def test_query_role_survives_scoring_enrichment_and_export(tmp_path):
    rows = pipeline._normalize_provided_asos({"provided_sequence_role": "antisense_oligo_sequence",
                                             "provided_asos": [ASO]})
    tiles, _ = pipeline._build_provided_aso_scoring_tiles(rows)
    tiles[0]["off_target_gene_count"] = 0
    path = pipeline._write_provided_aso_tiles_json(output_dir=tmp_path, tile_rows=tiles)
    pipeline._enrich_exact_off_target_bins_for_tiles(str(path), {"junction_search_summary": {}})
    scored = json.loads(path.read_text())["rows"]
    pipeline._merge_scored_tiles_into_provided_asos(rows, scored)
    exported = pipeline._provided_aso_export_rows({"evaluation_summary": {"provided_asos": rows}})[0]
    assert exported["off_target_query_sequence"] == TARGET
    assert exported["off_target_query_role"] == tiles[0]["off_target_query_role"]
    assert "reverse complement" in exported["off_target_query_role"]
    assert exported["Off_target_exact_count"] == 0


@pytest.mark.parametrize("strand,genome_sequence", [("+", "TACCNNNNGGTT"), ("-", "AACCNNNNGGTA")])
def test_unmapped_declared_aso_finds_actual_antisense_junction_off_target(monkeypatch, strand, genome_sequence):
    import run_goal_intake as runtime
    monkeypatch.setattr(runtime, "_load_genome_fasta", lambda _: {"synthetic": genome_sequence})
    monkeypatch.setattr(runtime, "_load_gtf_transcript_exons", lambda _: {"transcripts": {
        "synthetic_tx": {"gene_id": "synthetic_offtarget", "chrom": "synthetic",
                         "strand": strand, "exons": [(0, 4), (8, 12)]}}})
    rows = pipeline._normalize_provided_asos({"provided_sequence_role": "antisense_oligo_sequence",
                                             "provided_asos": [ASO]})
    tiles, _ = pipeline._build_provided_aso_scoring_tiles(rows)
    query = tiles[0]["Sequence"]
    hits, summary = runtime._collect_junction_off_targets([query, ASO], "synthetic", "intended_gene")
    assert summary["junctions_scanned"] == 1
    assert hits[query]["gene_ids"] == {"synthetic_offtarget"}
    assert hits[ASO]["gene_ids"] == set()


def test_canonical_role_schema_accepts_typed_roles_and_rejects_guesses():
    schema = canonical_task_context_schema()
    jsonschema.validate({"provided_sequence_role": "antisense_oligo_sequence", "provided_asos": [
        {"sequence": ASO, "sequence_role": "target_site_sequence"}]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"provided_sequence_role": "probably_antisense"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"provided_asos": [{"sequence": ASO, "sequence_role": "binding_hit"}]}, schema)
