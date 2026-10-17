"""The pooled report must preserve physical observations and fail closed."""

import csv
import json
from pathlib import Path

import pytest

from aso_agents.pooled_report import ReportContractError, rank_saved_observations
from aso_agents import pipeline_tools as pipeline


CONTRACT = {"transcript_id": "ENST00000001.2", "exon_ordinal": 3,
            "binding_requirement": "exact_antisense_in_exon", "ranking_metric": "absolute_gc_distance",
            "tie_breaker": "ascending_observation_id",
            "length_nt": 4, "max_homopolymer_nt": 2,
            "gc_min": 0.25, "gc_max": 0.75, "gc_target": 0.5}
DESIGN = {"child_structured_goal": {"isoform_transcripts": {"ENST00000001": {
    "exons_tx": [{"index": 3, "cdna_start": 101, "cdna_end": 150,
                  "start": 1000, "end": 1049}]}}}}


def generated(candidate="2", antisense="AACC", target="GGTT", start=110):
    return {"candidate": candidate, "antisense_5to3": antisense, "target_5to3": target,
            "transcript_accession": "ENST00000001.2", "transcript_cdna_start": start,
            "transcript_cdna_end": start + 3, "exon_ordinal": 3}


def supplied(aso_id="old", sequence="AACC", exact=1, same_strand=0, start=120):
    return {"aso_id": aso_id, "evaluated_molecule_5to3": sequence,
            "physical_exact_antisense_hit_count": exact, "exact_same_strand_hit_count": same_strand,
            "mapping_transcript_id": "ENST00000001.2", "mature_cdna_start_1based": start,
            "mature_cdna_end_1based": start + 3}


def test_ties_use_public_id_and_preserve_repeated_molecules_across_origins():
    rows = rank_saved_observations([generated("2"), generated("1", start=111)],
                                    [supplied("old")], DESIGN, CONTRACT)
    assert len(rows) == 3
    assert [(row["origin"], row["observation_id"], row["pooled_rank"]) for row in rows] == [
        ("generated", "2", 2), ("generated", "1", 1), ("supplied", "old", 3)]


def test_wrong_orientation_near_match_and_wrong_exon_are_retained_unranked():
    rows = rank_saved_observations([generated(start=149)],
                                    [supplied("near", exact=0), supplied("wrong_orientation", exact=0, same_strand=1)],
                                    DESIGN, CONTRACT)
    assert all(row["pooled_rank"] is None for row in rows)
    assert "not wholly inside" in rows[0]["exclusion_reason"]
    assert "no exact antisense" in rows[1]["exclusion_reason"]
    assert "same-strand" in rows[2]["exclusion_reason"]
    assert "locked transcript" not in rows[2]["exclusion_reason"]
    assert "requested exon" not in rows[2]["exclusion_reason"]


def test_bad_complement_or_unsupported_rule_fails_closed():
    with pytest.raises(ReportContractError, match="reverse-complement"):
        rank_saved_observations([generated(target="AAAA")], [], DESIGN, CONTRACT)
    with pytest.raises(ReportContractError, match="unsupported"):
        rank_saved_observations([], [supplied()], DESIGN,
                                {**CONTRACT, "ranking_metric": "imagined_efficacy"})


def test_locked_transcript_version_and_exon_boundary_are_checked():
    wrong = supplied("wrong_transcript")
    wrong["mapping_transcript_id"] = "ENST00000002.1"
    boundary = supplied("boundary", start=99)
    rows = rank_saved_observations([], [wrong, boundary], DESIGN, CONTRACT)
    assert "locked transcript" in rows[0]["exclusion_reason"]
    assert "requested exon" in rows[1]["exclusion_reason"]


def test_numeric_constraints_are_recomputed_from_sequence():
    rows = rank_saved_observations([], [supplied("run", sequence="AAAA"),
                                        supplied("gc", sequence="GGGG")], DESIGN, CONTRACT)
    assert all(not row["eligible"] for row in rows)
    assert all("GC fraction" in row["exclusion_reason"] for row in rows)
    assert all("homopolymer" in row["exclusion_reason"] for row in rows)


def test_native_comparison_uses_one_calculation_for_csv_and_report(tmp_path, monkeypatch):
    design_csv = tmp_path / "design.csv"
    evaluation_csv = tmp_path / "evaluation.csv"
    design_json = tmp_path / "design.json"
    with design_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(generated()))
        writer.writeheader()
        writer.writerow(generated())
    with evaluation_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(supplied()))
        writer.writeheader()
        writer.writerow(supplied())
    design_json.write_text(json.dumps(DESIGN))
    state = {"request": "Rank observations", "request_envelope": {"tasks": [
        {"task_id": "design", "task_type": "design_new_asos"},
        {"task_id": "evaluate", "task_type": "evaluate_existing_aso_set"},
        {"task_id": "compare", "task_type": "compare_aso_sets"}],
        "task_dependencies": [{"upstream_task_id": upstream, "downstream_task_id": "compare"}
                              for upstream in ("design", "evaluate")]},
        "task_artifacts": {
            "design": {"design_panel_csv": str(design_csv), "json": str(design_json)},
            "evaluate": {"comparison_table_csv": str(evaluation_csv)}}}
    sets = [pipeline._build_compare_set(set_id="design", label="design", source_type="generated",
                                        source_task_id="design", rows=[{"Sequence": "GGTT"}]),
            pipeline._build_compare_set(set_id="provided", label="provided", source_type="provided",
                                        rows=[{"sequence": "AACC"}])]
    prepared = {"status": "ok", "state_path": str(tmp_path / "state.json"), "state": state,
                "run_dir": str(tmp_path), "task_id": "compare",
                "task": {"task_type": "compare_aso_sets", "native": True},
                "task_request_envelope": state["request_envelope"],
                "task_context": {"comparison_contract": CONTRACT}}
    monkeypatch.setattr(pipeline, "_prepare_v2_task_execution", lambda **_: prepared)
    captured_dependencies=[]
    def load_evaluation(*_, task_ids=None):
        captured_dependencies.append(task_ids)
        return {"provided_asos": [1]}
    monkeypatch.setattr(pipeline, "_load_task_payload_by_type", load_evaluation)
    monkeypatch.setattr(pipeline, "_collect_aso_compare_sets", lambda **_: sets)
    monkeypatch.setattr(pipeline, "_complete_v2_task", lambda **kwargs: kwargs)
    result = pipeline.tool_compare_aso_sets(str(tmp_path / "state.json"))
    assert captured_dependencies == [{"design", "evaluate"}]
    with open(result["artifacts"]["comparison_table_csv"], newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["pooled_rank"] for row in rows] == ["1", "2"]
    markdown = Path(result["artifacts"]["markdown"])
    assert markdown.is_file()
    assert "old" in markdown.read_text()
    assert "source-set" not in markdown.read_text()
    comparison_payload = json.loads(Path(result["artifacts"]["json"]).read_text())
    composite = pipeline._build_deterministic_composite_response(
        {}, {}, completed_tasks=[{"task_id": "compare", "task_type": "compare_aso_sets",
                                  "payload": comparison_payload}])
    assert "Ranked 2 of 2" in composite["answer_summary"]
    assert "| 2 | supplied | old |" in composite["markdown"]
