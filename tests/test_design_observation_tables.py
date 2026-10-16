import copy
import hashlib
import json
from pathlib import Path

import pytest

from aso_agents import pipeline_tools as native
from aso_agents.design_export_evidence import design_export_evidence
from aso_agents.design_observation_tables import write_design_observations


def write(root, name, payload):
    path = native._write_run_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def example(root, strand="+"):
    row = {"Chrom": "chr7", "Start": 146, "End": 163, "Strand": strand,
           "Sequence": "AACCGTAACCGTAACCG", "Synthesis": "CGGTTACGGTTACGGTT",
           "IDT": "recorded-chemistry", "program_id": "p", "Off_target_exact_count": 0,
           "Off_target_2mm_count": None, "binding_site_membership": [{"program_id": "p"}]}
    exons = [{"index": 1, "start": 100, "end": 149},
             {"index": 2, "start": 200, "end": 249}]
    if strand == "-":
        exons = [{"index": 1, "start": 200, "end": 249},
                 {"index": 2, "start": 100, "end": 149}]
        row.update(Start=185, End=202)
    goal = {"species": "human", "genome_build": "assembly",
            "design_programs": [{"program_id": "p", "transcript_id": "TX.2"}],
            "isoform_transcripts": {"TX": {"transcript_id": "TX", "chrom": "chr7",
                "strand": strand, "metadata": {"versioned_id": "TX.2"}, "exons_tx": exons}}}
    raw = write(root, "tiles.json", {"schema": {"table": "aso_tiles"}, "rows": [row]})
    write(root, "structured_goal.json", goal)
    write(root, "generated_filter_trace.json", {"sources": {"raw_generated": [str(raw)]}})
    return row, goal, raw


def project(root, destination=None):
    evidence = design_export_evidence("design", root, native._run_path)
    meta = write_design_observations(evidence, destination or root / "views",
                                     native._goal_runtime_module()._tile_row_overlap_intervals)
    return meta, json.loads(Path(meta["path"]).read_text())


@pytest.mark.parametrize("strand", ["+", "-"])
def test_geometry_counts_boundary_zero_as_exonic_and_preserves_sequences(tmp_path, strand):
    row, goal, raw = example(tmp_path, strand)
    meta, table = project(tmp_path)
    observation = table["rows"][0]
    element, = observation["physical_splice_elements"]
    assert element["transcript_id"] == "TX.2" and element["element"] == "donor"
    assert element["exonic_overlap_bp"] == 3 and element["intronic_overlap_bp"] == 14
    assert element["relative_interval_inclusive"] == [-2, 14]
    assert observation["binding_match_key"]
    assert observation["recorded_fields"]["/Off_target_exact_count"] == 0
    assert observation["recorded_fields"]["/Off_target_2mm_count"] is None
    assert observation["derived_sequence_metrics"] == {
        "length": 17, "gc_bases": 9, "longest_homopolymer_run": 2,
        "basis": "verified antisense sequence"}
    assert table["reference_sources"][0]["sha256"]
    assert hashlib.sha256(Path(meta["path"]).read_bytes()).hexdigest() == meta["sha256"]


@pytest.mark.parametrize("strand,start,end", [("+", 196, 213), ("-", 135, 152)])
def test_acceptor_geometry_and_incomplete_reference_catalog(tmp_path, strand, start, end):
    row, goal, raw = example(tmp_path, strand)
    row.update(Start=start, End=end)
    raw.write_text(json.dumps({"schema": {"table": "aso_tiles"}, "rows": [row]}))
    _, table = project(tmp_path)
    element, = table["rows"][0]["physical_splice_elements"]
    assert element["element"] == "acceptor"
    assert element["exonic_overlap_bp"] == 14 and element["intronic_overlap_bp"] == 3
    assert element["relative_interval_inclusive"] == [-3, 13]
    goal["isoform_transcripts"] = {}
    write(tmp_path, "structured_goal.json", goal)
    _, table = project(tmp_path)
    assert table["rows"][0]["physical_splice_elements"] == []
    assert "does not establish absence" in table["rows"][0]["geometry_scope"]


def test_prediction_schema_locations_and_numeric_zero_are_retained(tmp_path):
    row, goal, raw = example(tmp_path)
    row.update(delta_deletion="0.1, 0.2",
               splice_site_deltas=[
                   {"label": "one", "site_kind": "donor", "delta_donor_deletion": 0.0},
                   {"label": "one", "site_kind": "acceptor", "delta_acceptor_deletion": None}])
    row["source_annotations"] = {"primary_label": "donor@149", "primary_delta_deletion": "-0.53",
                                  "primary_delta_replacement": "NaN", "nested/key": {"~": False}}
    row["splice_eval_summary"] = {"selected_delta_deletion": False, "splice_site_deltas": [
        {"label": "other", "delta_donor_replacement": -0.01}]}
    raw.write_text(json.dumps({"schema": {"table": "aso_tiles"}, "rows": [row]}))
    _, table = project(tmp_path)
    observation = table["rows"][0]
    records = {item["source_pointer"]: item for item in observation["splice_predictions"]}
    assert len(records) == 6
    assert records[""]["numeric_values"] == {}
    assert records["/source_annotations"]["numeric_values"] == {"primary_delta_deletion": -0.53}
    assert records["/source_annotations"]["recorded_values"]["primary_delta_replacement"] == "NaN"
    assert records["/splice_site_deltas/0"]["numeric_values"]["delta_donor_deletion"] == 0.0
    assert records["/splice_site_deltas/1"]["unresolved_fields"] == ["delta_acceptor_deletion"]
    assert records["/splice_eval_summary"]["numeric_values"] == {}
    assert records["/splice_eval_summary/splice_site_deltas/0"]["identity"]["label"] == "other"
    assert observation["recorded_fields"]["/source_annotations/nested~1key/~0"] is False


def test_same_binding_has_distinct_observations_and_preserves_conflicting_scores(tmp_path):
    row, goal, raw = example(tmp_path)
    duplicate = write(tmp_path, "steric_jobs/job/tiles.json",
                      {"schema": {"table": "aso_tiles"}, "rows": [row]})
    scored_row = {**row, "delta_inclusion_deletion": -0.27}
    scored = write(tmp_path, "scored_tiles.json",
                   {"schema": {"table": "aso_tiles"}, "rows": [scored_row, {**scored_row, "delta_inclusion_deletion": 0.1}]})
    write(tmp_path, "generated_filter_trace.json", {"sources": {
        "raw_generated": [str(raw), str(duplicate)], "scored": [str(scored)],
        "expression_annotation": [str(scored)]}})
    _, table = project(tmp_path)
    rows = table["rows"]
    assert len(rows) == 4 and len({r["observation_id"] for r in rows}) == 4
    assert len({r["binding_match_key"] for r in rows}) == 1
    assert [r["splice_predictions"][0]["numeric_values"]["delta_inclusion_deletion"]
            for r in rows[2:]] == [-0.27, 0.1]
    assert rows[2]["stage_roles"] == ["scored", "expression_annotation"]


def test_identity_separates_program_chemistry_site_and_strand_and_unknowns(tmp_path):
    row, goal, raw = example(tmp_path)
    goal["design_programs"].append({"program_id": "other", "transcript_id": "TX.2"})
    write(tmp_path, "structured_goal.json", goal)
    variants = [row, {**row, "Start": 147, "End": 164}, {**row, "Strand": "-"},
                {**row, "IDT": "different-chemistry"},
                {**row, "program_id": "other"},
                {**row, "Synthesis": "A" * 17}, {**row, "IDT": None},
                {**row, "program_id": None}]
    raw.write_text(json.dumps({"schema": {"table": "aso_tiles"}, "rows": variants}))
    _, table = project(tmp_path)
    keys = [r["binding_match_key"] for r in table["rows"]]
    assert len(set(keys[:5])) == 5 and all(keys[:5])
    assert keys[5:] == [None, None, None]


def test_overlapping_program_regions_do_not_replace_recorded_producer(tmp_path):
    row, goal, raw = example(tmp_path)
    row.pop("program_id")
    row.update(chemistry_group_id="group", binding_site_membership=[{"program_id": "p"}, {"program_id": "q"}])
    goal["design_programs"].append({"program_id": "q", "transcript_id": "TX.2"})
    goal["steric_design_jobs"] = [
        {"job_id": "job-p", "program_id": "p", "chemistry_group_id": "group"},
        {"job_id": "job-q", "program_id": "q", "chemistry_group_id": "other-group"}]
    write(tmp_path, "structured_goal.json", goal)
    variants = [row, {**row, "program_id": "p"}, {**row, "job_id": "job-p"},
                {**row, "program_id": "q"}, {**row, "chemistry_group_id": None}]
    raw.write_text(json.dumps({"schema": {"table": "aso_tiles"}, "rows": variants}))
    _, table = project(tmp_path)
    keys = [r["binding_match_key"] for r in table["rows"]]
    assert keys[0] and keys[0] == keys[1] == keys[2]
    assert keys[3:] == [None, None]
    assert table["rows"][0]["binding_identity"]["program_ids"] == ["p"]


def test_typed_generated_rows_join_native_tile_and_keep_nested_values(tmp_path):
    row, goal, raw = example(tmp_path)
    generated = {"program_id": "p", "chrom": row["Chrom"], "start": row["Start"], "end": row["End"],
                 "strand": row["Strand"], "sequence": row["Sequence"],
                 "synthesized_antisense_sequence": row["Synthesis"],
                 "source_annotations": {"chemistry_idt": row["IDT"], "primary_delta_deletion": "0.0"}}
    write(tmp_path, "generated_aso_set.json", {"artifact_type": "generated_aso_set", "rows": [generated]})
    _, table = project(tmp_path)
    assert table["rows"][0]["binding_match_key"] == table["rows"][1]["binding_match_key"]
    assert table["rows"][1]["splice_predictions"][0]["numeric_values"] == {"primary_delta_deletion": 0.0}


def test_unknown_schema_never_guesses_coordinate_origin(tmp_path):
    row, goal, raw = example(tmp_path)
    raw.write_text(json.dumps({"rows": [row, "legacy-row"]}))
    _, table = project(tmp_path)
    assert table["rows"][0]["binding_match_key"] is None
    assert not table["rows"][0]["binding_identity"]["binding_intervals_bed0_half_open"]
    assert not table["rows"][0]["physical_splice_elements"]
    assert table["diagnostics"][0]["reason"] == "non_object_row_retained_in_source"


def test_source_hash_and_destination_guards(tmp_path):
    row, goal, raw = example(tmp_path)
    evidence = design_export_evidence("design", tmp_path, native._run_path)
    raw.write_text(raw.read_text() + " ")
    with pytest.raises(ValueError, match="source_changed"):
        write_design_observations(evidence, tmp_path / "views", lambda *a, **kw: [])
    meta, _ = project(tmp_path)
    view = Path(meta["path"]); view.unlink(); view.symlink_to(raw)
    with pytest.raises(ValueError, match="source_or_symlink"):
        project(tmp_path)


def test_custom_writer_receives_and_protects_observation_table(tmp_path, monkeypatch):
    row, goal, raw = example(tmp_path)
    destination = tmp_path / "library.json"
    state = {"request": "Export all candidates and recorded predictions.", "request_envelope": {
        "schema_version": "v3", "tasks": [{"task_id": "design", "task_type": "design_new_asos"}],
        "final_outputs": [{"kind": "custom", "task_id": "design", "path": str(destination)}]}}
    def generate(payload):
        evidence = payload["declared_inputs"]["completed_run"]["design_evidence"][0]
        path = evidence["observations"]["path"]
        assert path in payload["readonly_output_paths"]
        assert evidence["observations"]["row_count"] == 1
        code = "from pathlib import Path\nimport json\n"
        code += f"source=json.loads(Path({path!r}).read_text())\n"
        code += f"Path({str(destination)!r}).write_text(json.dumps(source))\n"
        code += "Path('sandbox_result.json').write_text('{}')\nPath('sandbox_summary.md').write_text('All observations exported.')\n"
        return {"python_code": code}
    monkeypatch.setattr(native, "_run_sandbox_analysis_sync", generate)
    result = native._materialize_requested_final_outputs(state, str(tmp_path))
    assert result["ok"]
    exported = json.loads(destination.read_text())["rows"][0]
    assert exported["source"]["path"] == str(raw)
    assert exported["physical_splice_elements"][0]["exonic_overlap_bp"] == 3
