"""Cross-species cohort admission must use measured, reference-bound matches."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as tools


def evidence():
    anchor, secondary = "ENST00000000001.2", "ENSMUST00000000002.3"
    row = {
        "Name": "shared", "Sequence": "ACGTACGTACGTACGTACGT",
        "isoform_uniqueness_labels": [anchor.split(".")[0]],
        "cross_species_pass": True,
        "cross_species_method": "target_cdna_exact_sequence_search",
        "cross_species_matches": {"mouse": [secondary]},
        "cross_species_match_placements": {"mouse": [{
            "reference_id": secondary, "coordinate_scope": "transcript_cdna",
            "start_1based": 12, "end_1based": 31,
        }]},
    }
    goal = {
        "design_programs": [{"program_id": "shared", "modality": "gapmer",
            "target_type": "transcript_cohort", "transcript_ids": [anchor, secondary]}],
        "cross_species_targets": {"mouse": {
            "status": "resolved", "resolved_transcript_ids": [secondary],
            "reference_evidence": [{"versioned_id": secondary, "species": "mus_musculus"}],
        }},
    }
    return row, goal


def apply(tmp_path, row, goal):
    path = tmp_path / "scored.json"
    path.write_text(json.dumps({"rows": [row]}))
    result = tools._apply_gapmer_constraints_to_result(
        gapmer_results={"scored_tiles_path": str(path)}, structured_goal=goal,
        runtime=SimpleNamespace(filter_tiles_json=lambda path, constraints: {
            "applied": False, "output_path": path}),
    )
    assert "error" not in result
    return json.loads(Path(result["gapmer_results"]["scored_tiles_path"]).read_text())["rows"]


def test_cross_species_cohort_retains_reference_bound_exact_match(tmp_path):
    row, goal = evidence()
    original = deepcopy(row)
    assert apply(tmp_path, row, goal) == [original]
    assert row == original  # Evidence is neither fabricated nor relabeled.


@pytest.mark.parametrize("fault", [
    "unchecked", "failed", "wrong_method", "no_match", "no_placement",
    "wrong_placement_reference", "wrong_scope", "invalid_position",
    "unresolved", "unsearched", "wrong_version", "wrong_species", "no_reference",
])
def test_cross_species_cohort_rejects_unsubstantiated_identity(tmp_path, fault):
    row, goal = evidence()
    target = goal["cross_species_targets"]["mouse"]
    position = row["cross_species_match_placements"]["mouse"][0]
    if fault == "unchecked": row["cross_species_pass"] = None
    elif fault == "failed": row["cross_species_pass"] = False
    elif fault == "wrong_method": row["cross_species_method"] = "genomic_alignment"
    elif fault == "no_match": row["cross_species_matches"] = {}
    elif fault == "no_placement": row["cross_species_match_placements"] = {}
    elif fault == "wrong_placement_reference": position["reference_id"] = "ENSMUST00000000003.3"
    elif fault == "wrong_scope": position["coordinate_scope"] = "genomic"
    elif fault == "invalid_position": position["start_1based"] = 0
    elif fault == "unresolved": target["status"] = "error"
    elif fault == "unsearched": target["resolved_transcript_ids"] = []
    elif fault == "wrong_version": target["reference_evidence"][0]["versioned_id"] = "ENSMUST00000000002.4"
    elif fault == "wrong_species": target["reference_evidence"][0]["species"] = "human"
    elif fault == "no_reference": target["reference_evidence"] = []
    assert apply(tmp_path, row, goal) == []


def test_same_species_catalog_evidence_is_unchanged(tmp_path):
    row, goal = evidence()
    row = {"Name": "local", "isoform_uniqueness_labels": ["ENST00000000001", "ENST00000000003"]}
    goal["design_programs"][0]["transcript_ids"] = ["ENST00000000001.2", "ENST00000000003.1"]
    assert apply(tmp_path, row, goal) == [row]


@pytest.mark.parametrize('scope,expected', [('cohort',['shared']), ('ordinary',['shared','single'])])
def test_single_job_tables_use_bound_program_without_cross_program_leakage(tmp_path, scope, expected):
    cohort = {'program_id':'cohort','modality':'gapmer','target_type':'transcript_cohort',
              'transcript_ids':['TXA.1','TXB.2']}
    ordinary = {'program_id':'ordinary','modality':'gapmer','target_type':'transcript','transcript_id':'TXA.1'}
    rows = [{'Name':'shared','isoform_uniqueness_labels':['TXA','TXB']},
            {'Name':'single','isoform_uniqueness_labels':['TXA']}]
    path = tmp_path/'scored.json';path.write_text(json.dumps({'rows':rows}))
    result = tools._apply_gapmer_constraints_to_result(
        gapmer_results={'scored_tiles_path':str(path)},
        structured_goal={'design_programs':[cohort,ordinary], 'evaluation_targets':[
            {'target_type':'transcript','objective_type':'primary_outcome',
             'desired_delta':'decrease','transcript_id':'UNRELATED.1'}]},
        runtime=SimpleNamespace(filter_tiles_json=lambda path, constraints: {"applied": False, "output_path": path}), program_scope=cohort if scope == 'cohort' else ordinary)
    assert 'error' not in result
    selected=json.loads(Path(result['gapmer_results']['scored_tiles_path']).read_text())['rows']
    assert [r['Name'] for r in selected] == expected


def test_job_scope_rejects_contradictory_row_identity(tmp_path):
    row, goal = evidence();row['program_id']='different'
    path=tmp_path/'scored.json';path.write_text(json.dumps({'rows':[row]}))
    result=tools._apply_gapmer_constraints_to_result(
        gapmer_results={'scored_tiles_path':str(path)},structured_goal=goal,runtime=SimpleNamespace(filter_tiles_json=lambda path, constraints: {"applied": False, "output_path": path}),
        program_scope=goal['design_programs'][0])
    assert result['error']=='gapmer_transcript_cohort_job_program_mismatch'
