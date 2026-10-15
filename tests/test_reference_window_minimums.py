from copy import deepcopy

import pytest

import run_goal_intake as runtime
from aso_agents.candidate_minimums import requested_region_minimums, belongs_to_region
from aso_agents.splice_site_geometry import verified_reference_window_scope
from test_reference_bound_exon_focus import fixture


def compiled(strand="+", context="site_window"):
    goal, program = fixture(strand)
    program.update(program_id="requested_feature", targeting_policy="explicit_user_region",
                   site_context=context)
    goal["splice_site_plan"], _ = runtime._compute_splice_site_plan(
        goal["splice_isoform_objectives"], goal["isoform_transcripts"], goal)
    rows, metadata = runtime._build_steric_program_site_window_rows(goal, program)
    metadata["region_constraints"] = {"constraint_count": 0}
    job = {**program, "region_rows": rows, "selection_meta": metadata}
    goal.update(min_asos_per_site=1, steric_design_jobs=[job])
    return goal, job


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("context", ["site_window", "upstream_intron", "downstream_intron", "exon_body"])
def test_reference_compiled_windows_count_without_explicit_coordinate_constraints(strand, context):
    goal, job = compiled(strand, context)
    before = deepcopy(goal)
    assert verified_reference_window_scope(goal, job)
    group = requested_region_minimums(goal)[0]
    region = job["region_rows"][0]
    inside = [region[0], region[1], region[1] + 20]
    assert belongs_to_region({}, group, lambda *args, **kwargs: [inside])
    assert not belongs_to_region({}, group, lambda *args, **kwargs: [[region[0], 10000, 10020]])
    assert goal == before


@pytest.mark.parametrize("defect", ["missing_reference", "wrong_boundary", "wrong_chromosome",
    "wrong_strand", "wrong_ordinal", "widened_window", "unresolved_constraint", "full_copy"])
def test_no_scope_resolution_from_unattested_or_broadened_geometry(defect):
    goal, job = compiled()
    target = job["selection_meta"]["resolved_site_targets"][0]
    if defect == "missing_reference":
        goal["isoform_transcripts"].clear()
    elif defect == "wrong_boundary":
        target["position"] += 1
    elif defect == "wrong_chromosome":
        target["chrom"] = "chrDifferent"
    elif defect == "wrong_strand":
        target["strand"] = "-"
    elif defect == "wrong_ordinal":
        target["resolved_exon_ordinals"]["REFERENCE"] = 2
    elif defect == "widened_window":
        job["region_rows"][0][2] += 100
    elif defect == "unresolved_constraint":
        job["region_constraints"] = [{"source_type": "explicit_rows", "rows": []}]
    else:
        job["allow_full_copy"] = True
    assert not verified_reference_window_scope(goal, job)
    assert not requested_region_minimums(goal)[0]["selectors"][0]["explicit_scope_resolved"]
