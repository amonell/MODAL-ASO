"""Native supplied-set evidence can fulfill a typed comparison-table export."""
import csv

from aso_agents import pipeline_tools as pipeline


def test_composite_evaluation_owns_complete_physical_set_table(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    output = tmp_path / "output" / "comparison.csv"
    envelope = {
        "schema_version": "v3",
        "tasks": [
            {"task_id": "evaluation", "task_type": "evaluate_existing_aso_set"},
            {"task_id": "report", "task_type": "build_composite_report", "depends_on": ["evaluation"]},
        ],
        "final_outputs": [{"kind": "comparison_table_csv", "task_id": "evaluation", "path": str(output)}],
    }
    physical = "GCATAAGCCTATGCCTAGAT"
    payload = {"task_type": "evaluate_existing_aso_set", "context_summary": {}, "provided_asos": [
        {"input_row_id": "a1", "aso_id": "A1", "name": "A1", "sequence": physical,
         "input_metadata": {"set_id": "A"}, "declared_sequence_role": "antisense_oligo_sequence"},
        {"input_row_id": "b2", "aso_id": "B2", "name": "B2", "sequence": physical,
         "input_metadata": {"set_id": "B"}, "declared_sequence_role": "antisense_oligo_sequence"},
    ]}
    artifacts = pipeline._write_v2_task_result_files(
        run_dir=str(run), task_id="evaluation", basename="provided_aso_evaluation",
        payload=payload, markdown_text="# Evaluation", request_envelope=envelope,
    )
    assert "comparison_table_csv" in artifacts
    assert pipeline._native_export_catalog(
        {"request_envelope": envelope}, str(run), "evaluation"
    )["comparison_table_csv"] == artifacts["comparison_table_csv"]
    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync",
                        lambda *_: (_ for _ in ()).throw(AssertionError("native table must be copied")))
    receipt = pipeline._materialize_requested_final_outputs({"request_envelope": envelope}, str(run))
    assert receipt["ok"] and not receipt["execution_attempts"]
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert [row["aso_id"] for row in rows] == ["A1", "B2"]
    assert [row["set_id"] for row in rows] == ["A", "B"]
    assert [row["sequence"] for row in rows] == [physical, physical]
    assert rows[0]["declared_sequence_role"] == "antisense_oligo_sequence"
    assert output.stat().st_size < 10_000


def test_target_site_input_uses_confirmed_complement_as_physical_molecule():
    target = "ATCTAGGCATAGGCTTATGC"
    physical = "GCATAAGCCTATGCCTAGAT"
    rows = pipeline._native_evaluation_table_rows({"provided_asos": [{
        "aso_id": "target-site-1", "sequence": target,
        "reverse_complement_sequence": physical,
        "declared_sequence_role": "target_site_sequence",
        "mapping_hits": [{"match_orientation": "input_sequence",
                          "matched_target_sequence": target,
                          "transcript_id": "ENST1", "cdna_start": 10, "cdna_end": 29}],
    }]})
    assert rows[0]["evaluated_molecule_5to3"] == physical
    assert rows[0]["physical_exact_antisense_hit_count"] == 1
    assert rows[0]["opposite_orientation_alternative_5to3"] == target
    assert rows[0]["reverse_complement_alternative_5to3"] is None
