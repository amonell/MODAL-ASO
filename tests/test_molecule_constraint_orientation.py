"""Physical ASO sequence constraints across native generation/filter boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aso_agents.parameter_constraints import (
    candidate_constraint_sequence,
    evaluate_row_against_constraints,
)
import run_goal_intake as runtime


CONSTRAINTS = {"gc_min": 0, "gc_max": 1, "homopolymers": {"per_base_max": {"G": 3}}}


@pytest.mark.parametrize("strand", ["+", "-"])
def test_asymmetric_rule_follows_synthesized_molecule_at_both_filters(tmp_path, strand):
    rows = [
        {"Sequence": "CCCCAT", "Synthesis": "ATGGGG", "GC": 2 / 3, "Strand": strand,
         "No_homopolymer": True},
        {"Sequence": "GGGGAT", "Synthesis": "ATCCCC", "GC": 2 / 3, "Strand": strand,
         "No_homopolymer": False},
    ]
    before = deepcopy(rows)
    assert evaluate_row_against_constraints(rows[0], CONSTRAINTS) == (
        False, ["homopolymer_run_exceeds_limit"])
    assert evaluate_row_against_constraints(rows[1], CONSTRAINTS) == (True, [])
    source = tmp_path / "tiles.json"
    source.write_text(json.dumps({"rows": rows}))
    original = source.read_bytes()
    result = runtime._apply_initial_tiling_filters(str(source), CONSTRAINTS)
    assert json.loads(Path(result["passed"]["path"]).read_text())["rows"] == [rows[1]]
    assert json.loads(Path(result["filtered_no_homopolymer"]["path"]).read_text())["rows"] == [rows[0]]
    assert source.read_bytes() == original and rows == before


@pytest.mark.parametrize("field", ["synthesized_antisense_sequence", "Synthesis", "chemistry_synthesis", "antisense_5to3"])
@pytest.mark.parametrize("nested", [False, True])
def test_explicit_physical_sequence_takes_precedence_without_mutation(field, nested):
    row = {"Sequence": "GGGGAT", "GC": 2 / 3}
    container = row.setdefault("source_annotations", {}) if nested else row
    container[field] = "ATCCCC"
    before = deepcopy(row)
    assert candidate_constraint_sequence(row) == "ATCCCC"
    assert evaluate_row_against_constraints(row, CONSTRAINTS) == (True, [])
    assert row == before


@pytest.mark.parametrize("role", ["transcript_oriented_target_binding_site", "target_site_sequence"])
def test_only_explicit_target_sequence_is_reverse_complemented(role):
    row = {"sequence": "CCCCAU", "sequence_role": role, "GC": 2 / 3}
    assert candidate_constraint_sequence(row) == "ATGGGG"
    assert evaluate_row_against_constraints(row, CONSTRAINTS) == (False, ["homopolymer_run_exceeds_limit"])
    assert row["sequence"] == "CCCCAU"


@pytest.mark.parametrize("annotations", [None, [], "untyped"])
def test_legacy_untyped_sequence_retains_existing_interpretation(annotations):
    row = {"Sequence": "GGGGAT", "source_annotations": annotations, "GC": 2 / 3}
    assert candidate_constraint_sequence(row) == "GGGGAT"
    assert evaluate_row_against_constraints(row, CONSTRAINTS) == (False, ["homopolymer_run_exceeds_limit"])


@pytest.mark.parametrize("sequence,synthesis,expected", [
    ("CCCCAT", "ATGGGG", True), ("CCCCCAT", "ATGGGGG", False),
    ("GGGGAT", "ATCCCC", True), ("GGGGGAT", "ATCCCCC", False),
])
def test_symmetric_default_is_orientation_invariant(sequence, synthesis, expected):
    constraints = {"homopolymers": {"max_run": 4}}
    assert evaluate_row_against_constraints({"Sequence": sequence}, constraints)[0] is expected
    assert evaluate_row_against_constraints({"Sequence": sequence, "Synthesis": synthesis}, constraints)[0] is expected


@pytest.mark.parametrize("strand", ["+", "-"])
def test_native_cdna_gapmers_record_asymmetric_screen_on_actual_molecule(tmp_path, monkeypatch, strand):
    cdna = "CCCCATACGTACGTACGTACGGGGATACGTACGTACGTAC"
    transcript = {"transcript_id": "TX", "strand": strand, "chrom": "chr1",
                  "exons_tx": [{"index": 1, "start": 1, "end": len(cdna)}]}
    monkeypatch.setattr(runtime, "_build_isoform_cdna_sequences", lambda *a, **k: {"TX": cdna})
    monkeypatch.setattr(runtime, "_annotate_tile_rows_with_rnafold", lambda rows: None)
    goal = {"species": "human", "chemistry": "5-10-5", "parameter_constraints": CONSTRAINTS,
            "constraints": CONSTRAINTS, "homopolymer_limits": {"G": 3}}
    result = runtime._build_cdna_gapmer_tiles(goal, {"TX": transcript}, "TX", {"TX": transcript}, str(tmp_path))
    assert not result.get("error")
    rows = json.loads(Path(result["json_path"]).read_text())["rows"]
    assert any("GGGG" in row["Synthesis"] for row in rows)
    assert any("GGGG" in row["Sequence"] and "GGGG" not in row["Synthesis"] for row in rows)
    for row in rows:
        expected = "GGGG" not in row["Synthesis"]
        assert str(row["No_homopolymer"]).lower() == str(expected).lower()
        assert evaluate_row_against_constraints(row, CONSTRAINTS)[0] is expected
