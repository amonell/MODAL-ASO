"""Physical-site quotas and explicit program quotas use different count scopes."""
import json
from pathlib import Path

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.candidate_minimums import region_minimum_delivery
from aso_chemistry import encode_idt_oligo


@pytest.mark.parametrize("policy", ["hard_nonoverlap", "joint_objective"])
@pytest.mark.parametrize("scope", ["per_region", "explicit_program_bound", "explicit_global_bound", "legacy_program_scope"])
def test_region_quota_does_not_replace_an_explicit_program_limit(tmp_path, monkeypatch, policy, scope):
    rows = []
    for index, (start, suffix) in enumerate([(0, "ACGA"), (10, "ACGG"), (110, "AGGA")]):
        sequence = "ACGT" * 4 + suffix
        rows.append({"Synthesis": sequence,
                     "Sequence": sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1],
                     "IDT": encode_idt_oligo(sequence, "steric-hindrance"),
                     "Chrom": "chrFixture", "Start": start, "End": start + len(sequence), "Strand": "+",
                     "binding_intervals_bed0_half_open": [["chrFixture", start, start + len(sequence)]],
                     "program_id": "program", "source_modality": "steric", "selection_rank": index + 1})
    constraints = [{"constraint_id": label, "label": label, "source_type": "explicit_rows",
                    "operator": "union", "coordinate_space": "genomic",
                    "aso_overlap_requirement": "full_containment",
                    "materialized_rows": [["chrFixture", start, start + 60, label, 0, "+"]]}
                   for label, start in [("first", 0), ("second", 100)]]
    goal = {"species": "human", "candidate_minimum": 2, "candidate_limit": 2,
            "candidate_limit_per_site": 1, "min_asos_per_site": 1,
            "materialized_region_constraints": {"global": constraints},
            "design_programs": [{"program_id": "program", "modality": "steric", "candidate_limit": 2}],
            "steric_design_jobs": [{"program_id": "program", "region_rows": [["chrFixture", 0, 160]]}]}
    if scope == "explicit_program_bound":
        goal["design_programs"][0]["candidate_limit"] = 1
    elif scope == "explicit_global_bound":
        goal["candidate_limit"] = 1
    elif scope == "legacy_program_scope":
        goal.pop("materialized_region_constraints")
        goal.pop("min_asos_per_site")
    if policy == "hard_nonoverlap":
        goal["candidate_selection_policy"] = "nonoverlapping_target_sites"
    else:
        goal["candidate_panel_objectives"] = [{"metric": "gc_fraction", "absolute_distance_from": .5,
                                              "aggregate": "sum", "direction": "minimize"}]
    monkeypatch.setattr(pipeline, "_collect_generated_candidate_rows", lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, "_collect_generated_stage_sets", lambda *a: {})
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda *a: {"structured_goal": goal})
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    panel = json.loads(Path(artifacts["generated_aso_set_json"]).read_text())
    selection = panel["candidate_selection"]
    if scope == "per_region":
        assert selection["status"] == "complete" and len(panel["rows"]) == 2
        delivery = region_minimum_delivery(goal, panel, {}, pipeline._candidate_binding_intervals_for_exon_annotation)
        assert delivery["status"] == "fulfilled"
        assert [group["distinct_molecules"] for group in delivery["groups"]] == [1, 1]
        if policy == "joint_objective":
            assert selection["requested_panel_ranking"]["status"] == "complete"
        else:
            assert selection["hard_nonoverlap"] is True
    else:
        assert selection["status"] == "unfulfilled" and len(panel["rows"]) == 1
        assert selection["solver_status"] == "infeasible"
