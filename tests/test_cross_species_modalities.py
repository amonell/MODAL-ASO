from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import run_goal_intake
from aso_agents import pipeline_tools
from aso_agents.species_config import normalize_requested_species, normalize_species


def _cross_species_goal(*, splice: bool = False) -> dict[str, Any]:
    goal: dict[str, Any] = {
        "gene_symbol": "GENE1",
        "gene_id": "ENSG00000000001",
        "species": ["human", "mouse"],
        "cross_species_exact_match": {
            "required": True,
            "species": ["human", "mouse"],
        },
    }
    if splice:
        goal.update(
            {
                "intent": "splice_mod",
                "exon_label": 3,
                "design_programs": [{"program_id": "steric_exon3", "modality": "steric", "exon_label": 3}],
            }
        )
    return goal


def _resolved_targets() -> dict[str, dict[str, Any]]:
    return {
        "human": {
            "status": "resolved",
            "species": "human",
            "gene_id": "ENSG00000000001",
            "gene_symbol": "GENE1",
        },
        "mouse": {
            "status": "resolved",
            "species": "mouse",
            "gene_id": "ENSMUSG00000000001",
            "gene_symbol": "Gene1",
        },
    }


def _write_goal_context(run_dir: Path, goal: dict[str, Any], **final_updates: Any) -> None:
    final_payload = {"structured_goal": deepcopy(goal), **deepcopy(final_updates)}
    (run_dir / "goal_intake.json").write_text(
        json.dumps({"request": "Design exact-match human/mouse ASOs.", "final_payload": final_payload}),
        encoding="utf-8",
    )


def test_species_normalization_accepts_pan_species_goal_lists() -> None:
    assert normalize_species(["human", "mouse"]) == "human"
    assert normalize_requested_species(["mouse", "human"]) == "mouse"


def test_species_normalization_separates_redundant_assembly_aliases() -> None:
    assert normalize_species("homo_sapiens_hg38") == "human"
    assert normalize_species("Human (GRCh38/hg38)") == "human"
    assert normalize_requested_species("mus_musculus_mm39") == "mouse"


def test_steric_cross_species_matcher_uses_target_genomic_context(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tiles_path = tmp_path / "tiles.json"
    tiles_path.write_text(
        json.dumps(
            {
                "schema": {"columns": {"Sequence": "string"}},
                "rows": [
                    {"Sequence": "AACCGGTT"},
                    {"Sequence": "TTTTCCCC"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_goal_intake,
        "_resolve_cross_species_gene_targets",
        lambda goal, species: deepcopy(_resolved_targets()),
    )

    def fake_target_sequences(
        gene_symbol: str,
        species_key: str,
        *,
        structured_goal: dict[str, Any],
        gene_id: str | None = None,
    ) -> tuple[dict[str, str], None, str]:
        del gene_symbol, structured_goal, gene_id
        sequence = "AAAACCGGTTGGGGTTTTCCCCAAAA" if species_key == "human" else "GGGAACCGGTTCCC"
        return (
            {f"{species_key}_canonical:exon_3_window": sequence},
            None,
            "target_splice_window_exact_sequence_search",
        )

    monkeypatch.setattr(run_goal_intake, "_resolve_cross_species_steric_sequences", fake_target_sequences)

    goal = _cross_species_goal(splice=True)
    summary = run_goal_intake._annotate_steric_cross_species_matches(str(tiles_path), goal)
    payload = json.loads(tiles_path.read_text(encoding="utf-8"))

    assert summary is not None
    assert summary["annotated"] == 2
    assert summary["passing"] == 1
    assert summary["method"] == "target_splice_window_exact_sequence_search"
    assert payload["rows"][0]["cross_species_pass"] is True
    assert set(payload["rows"][0]["cross_species_matches"]) == {"human", "mouse"}
    assert payload["rows"][1]["cross_species_pass"] is False
    assert payload["rows"][1]["cross_species_missing_reason"] == "missing_exact_target_sequence_in:mouse"


def test_steric_cross_species_matcher_marks_unavailable_evidence_unchecked(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tiles_path = tmp_path / "tiles.json"
    tiles_path.write_text(json.dumps({"rows": [{"Sequence": "AACCGGTT"}]}), encoding="utf-8")
    monkeypatch.setattr(
        run_goal_intake,
        "_resolve_cross_species_gene_targets",
        lambda goal, species: deepcopy(_resolved_targets()),
    )

    def fake_target_sequences(
        gene_symbol: str,
        species_key: str,
        *,
        structured_goal: dict[str, Any],
        gene_id: str | None = None,
    ) -> tuple[dict[str, str], str | None, str | None]:
        del gene_symbol, structured_goal, gene_id
        if species_key == "mouse":
            return {}, "genome_reference_missing", None
        return {"human_canonical:gene_locus": "AAAACCGGTTAAAA"}, None, "target_gene_locus_exact_sequence_search"

    monkeypatch.setattr(run_goal_intake, "_resolve_cross_species_steric_sequences", fake_target_sequences)

    summary = run_goal_intake._annotate_steric_cross_species_matches(
        str(tiles_path),
        _cross_species_goal(),
    )
    row = json.loads(tiles_path.read_text(encoding="utf-8"))["rows"][0]

    assert summary is not None
    assert summary["species_errors"] == {"mouse": "genome_reference_missing"}
    assert row["cross_species_pass"] is None
    assert row["cross_species_unavailable_reason"] == "target_evidence_unavailable_in:mouse"


def test_generated_artifacts_fail_closed_for_unchecked_cross_species_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tiling_dir = run_dir / "tiling"
    tiling_dir.mkdir(parents=True)
    _write_goal_context(run_dir, _cross_species_goal())
    (tiling_dir / "steric_scored_uniqueness.json").write_text(
        json.dumps({"rows": [{"Sequence": "AACCGGTTAACCGGTT", "cross_species_required": True}]}),
        encoding="utf-8",
    )

    artifacts = pipeline_tools._emit_v2_generated_design_artifacts(str(run_dir))
    generated = json.loads(Path(artifacts["generated_aso_set_json"]).read_text(encoding="utf-8"))
    trace = json.loads(Path(artifacts["generated_filter_trace_json"]).read_text(encoding="utf-8"))
    eligibility = generated["eligibility_filters"][0]
    outcome = pipeline_tools._classify_design_outcome(
        child_run_dir=str(run_dir),
        task_context={},
        child_status="run_complete",
        generated_artifacts=artifacts,
    )

    assert generated["row_count"] == 0
    assert generated["excluded_row_count"] == 1
    assert eligibility["status"] == "incomplete"
    assert eligibility["unchecked_count"] == 1
    assert trace["rows"][0]["eligible"] is False
    assert trace["rows"][0]["exclusion_reasons"] == ["cross_species_exact_match_unchecked"]
    assert outcome["candidate_count"] == 0
    assert outcome["outcome_status"] == "completed_missing_required_evidence"
    assert "cross_species_exact_match" in outcome["missing_required_evidence"]


def test_generated_artifacts_keep_only_explicit_cross_species_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tiling_dir = run_dir / "tiling"
    tiling_dir.mkdir(parents=True)
    goal = _cross_species_goal()
    goal["intent"] = "binding"
    raw_path = tiling_dir / "initial_filtering_tiles.json"
    raw_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"Sequence": "AACCGGTTAACCGGTT", "cross_species_required": True, "cross_species_pass": True},
                    {"Sequence": "TTTTCCCCAAAAGGGG", "cross_species_required": True, "cross_species_pass": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tiling_dir / "steric_scored_uniqueness.json").write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")
    required_filter = {
        "status": "complete",
        "input_path": str(raw_path),
        "total_count": 2,
        "checked_count": 2,
        "passing_count": 1,
        "failed_count": 1,
        "unchecked_count": 0,
        "species_status": {"human": "resolved", "mouse": "resolved"},
    }
    _write_goal_context(run_dir, goal, tiling_results={"required_evidence_filtering": required_filter})

    artifacts = pipeline_tools._emit_v2_generated_design_artifacts(str(run_dir))
    generated = json.loads(Path(artifacts["generated_aso_set_json"]).read_text(encoding="utf-8"))

    assert generated["row_count"] == 1
    assert generated["rows"][0]["sequence"] == "AACCGGTTAACCGGTT"
    assert generated["rows"][0]["cross_species_pass"] is True
    assert generated["excluded_row_count"] == 1


def test_completed_exact_match_search_with_no_shared_sequence_is_no_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tiling_dir = run_dir / "tiling"
    tiling_dir.mkdir(parents=True)
    goal = _cross_species_goal()
    goal["intent"] = "binding"
    raw_path = tiling_dir / "initial_filtering_tiles.json"
    raw_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"Sequence": "TTTTCCCCAAAAGGGG", "cross_species_required": True, "cross_species_pass": False}
                ]
            }
        ),
        encoding="utf-8",
    )
    (tiling_dir / "steric_scored_uniqueness.json").write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")
    required_filter = {
        "status": "complete",
        "input_path": str(raw_path),
        "total_count": 1,
        "checked_count": 1,
        "passing_count": 0,
        "failed_count": 1,
        "unchecked_count": 0,
        "requested_species": ["human", "mouse"],
        "species_status": {"human": "resolved", "mouse": "resolved"},
    }
    _write_goal_context(run_dir, goal, tiling_results={"required_evidence_filtering": required_filter})

    artifacts = pipeline_tools._emit_v2_generated_design_artifacts(str(run_dir))
    outcome = pipeline_tools._classify_design_outcome(
        child_run_dir=str(run_dir),
        task_context={},
        child_status="run_complete",
        generated_artifacts=artifacts,
    )

    assert outcome["candidate_count"] == 0
    assert outcome["outcome_status"] == "completed_no_candidates"
    assert outcome["missing_required_evidence"] == []
    assert outcome["no_candidate_reason"] == "No candidate had an exact target-sequence match in every requested species."


def test_multispecies_offtarget_annotation_is_modality_independent(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tiles_path = tmp_path / "steric_tiles.json"
    tiles_path.write_text(
        json.dumps({"schema": {"columns": {"Sequence": "string"}}, "rows": [{"Sequence": "AACCGGTT"}]}),
        encoding="utf-8",
    )
    goal = _cross_species_goal()
    goal["cross_species_targets"] = _resolved_targets()

    def fake_mapping(path: str, species_goal: dict[str, Any], constraints: Any) -> dict[str, Any]:
        del constraints
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        row = payload["rows"][0]
        row["off_target_gene_count"] = 1 if species_goal["species"] == "human" else 2
        row["Off_target_gene_ids"] = [f"{species_goal['species']}_offtarget"]
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "complete"}

    def fake_expression(*, initial_filter_json_path: str, structured_goal: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(Path(initial_filter_json_path).read_text(encoding="utf-8"))
        payload["rows"][0]["Off_target_expression_weighted_risk"] = (
            3.0 if structured_goal["species"] == "human" else 7.0
        )
        Path(initial_filter_json_path).write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "complete"}

    monkeypatch.setattr(pipeline_tools, "_run_offtarget_expression_with_timeout", fake_expression)
    summary = pipeline_tools._annotate_gapmer_offtargets_by_species(
        initial_tiles_path=str(tiles_path),
        structured_goal=goal,
        parameter_constraints={},
        annotate_offtargets=fake_mapping,
    )
    row = json.loads(tiles_path.read_text(encoding="utf-8"))["rows"][0]

    assert summary["complete"] is True
    assert summary["status_by_species"] == {"human": "complete", "mouse": "complete"}
    assert set(row["off_target_by_species"]) == {"human", "mouse"}
    assert row["worst_species_off_target_gene_count"] == 2
    assert row["worst_species_off_target_expression_risk"] == 7.0
