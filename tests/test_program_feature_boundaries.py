"""Pooled candidates retain both reference-feature and binding-site predicates."""
from copy import deepcopy
import json

import pytest

import run_goal_intake as native
from aso_agents import pipeline_tools as pipeline


def reference_goal(strand):
    entry = {
        "transcript_id": "TXA.3", "chrom": "chr8", "strand": strand,
        "metadata": {"versioned_id": "TXA.3"},
        "exons_tx": [{
            "index": 1, "start": 1001, "end": 1300,
            "cdna_start": 1, "cdna_end": 300, "contains_coding": True,
            "coding_start": 101, "coding_end": 200,
            "coding_start_genomic": 1101, "coding_end_genomic": 1200,
        }],
    }
    return {"species": "human", "isoform_transcripts": {"primary": entry}}


def region_rule(chrom, start, end, space, **overrides):
    return {
        "source_type": "explicit_rows", "coordinate_space": space,
        "materialized_rows": [[chrom, start, end, "site", 0, "+"]],
        "aso_overlap_requirement": "any_overlap", "minimum_overlap_bp": 8,
        **overrides,
    }


def filter_pool(tmp_path, rows, compiled):
    source = tmp_path / "pool.json"
    destination = tmp_path / "selected.json"
    source.write_text(json.dumps({"rows": rows}))
    result = pipeline._filter_tiles_json_to_output(
        source_json_path=str(source), output_path=str(destination),
        region_rows=compiled["tile_filter_rows"],
        tile_overlap_requirement=compiled["tile_overlap_requirement"],
        constraint_filters=compiled["tile_constraint_filters"],
        allow_empty_subset=True,
    )
    assert "error" not in result
    return [row["id"] for row in json.loads(destination.read_text())["rows"]]


def candidate(name, chrom, start, end):
    return {"id": name, "Chrom": chrom, "Start": start + 1, "End": end,
            "coordinate_system": "1based_inclusive"}


@pytest.mark.parametrize("modality", ["steric", "gapmer"])
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("feature", ["utr3", "utr5", "cds"])
def test_overlapping_a_short_site_cannot_erase_reference_feature_bounds(
    tmp_path, modality, strand, feature,
):
    goal = reference_goal(strand)
    program = {"program_id": "panel", "modality": modality, "length": 20,
               "intent": "binding" if modality == "steric" else "knockdown",
               "target_region": feature, "transcript_id": "TXA.3"}
    if modality == "steric":
        base, meta = native._build_steric_program_region_rows(goal, program)
        space = "genomic"
    else:
        base, meta = native._build_gapmer_program_region_rows(
            goal, goal["isoform_transcripts"], "primary", program)
        space = "cdna"
    assert meta["tile_overlap_requirement"] == "full_containment"
    chrom, start, end = base[0][:3]
    assert end - start == 100
    goal["materialized_region_constraints"] = {"global": [
        region_rule(chrom, start, start + 8, space),
        region_rule(chrom, end - 8, end, space, operator="union"),
    ]}
    before = deepcopy(goal)
    _, compiled = pipeline._apply_shared_region_constraints(
        runtime=native, structured_goal=goal, program=program,
        base_rows=base, output_space=space,
        base_overlap_requirement=meta["tile_overlap_requirement"],
    )
    rows = [candidate("left_spill", chrom, start - 10, start + 10),
            candidate("left_inside", chrom, start, start + 20),
            candidate("no_site", chrom, start + 30, start + 50),
            candidate("right_inside", chrom, end - 20, end),
            candidate("right_spill", chrom, end - 10, end + 10)]
    # A 5' UTR starts at transcript position one, so there is no physical
    # mature-RNA candidate extending beyond its left edge.
    rows = [row for row in rows if row["Start"] > 0]
    # Both alternatives stay available, and a 20-mer may extend beyond its
    # eight-base site provided the complete binding interval fits the feature.
    assert filter_pool(tmp_path, rows, compiled) == ["left_inside", "right_inside"]
    assert goal == before


@pytest.mark.parametrize("space,chrom", [("genomic", "chr8"), ("cdna", "TXA.3")])
def test_base_guard_preserves_alternatives_and_exclusions(tmp_path, space, chrom):
    goal = reference_goal("+")
    goal["materialized_region_constraints"] = {"global": [
        region_rule(chrom, 100, 108, space),
        region_rule(chrom, 192, 200, space, operator="union"),
        region_rule(chrom, 100, 120, space, operator="exclude", minimum_overlap_bp=1),
    ]}
    _, compiled = pipeline._apply_shared_region_constraints(
        runtime=native, structured_goal=goal, base_rows=[[chrom, 100, 200]],
        output_space=space, program={"length": 20},
        base_overlap_requirement="full_containment",
    )
    rows = [candidate("excluded_site", chrom, 100, 120),
            candidate("allowed_site", chrom, 180, 200),
            candidate("outside_base", chrom, 190, 210)]
    assert filter_pool(tmp_path, rows, compiled) == ["allowed_site"]


@pytest.mark.parametrize("base,policy", [([], "full_containment"), ([["chr8", 100, 200]], "")])
def test_no_independent_base_requirement_is_invented(tmp_path, base, policy):
    goal = {"materialized_region_constraints": {"global": [
        region_rule("chr8", 100, 108, "genomic"),
    ]}}
    _, compiled = pipeline._apply_shared_region_constraints(
        runtime=native, structured_goal=goal, base_rows=base,
        output_space="genomic", program={"length": 20},
        base_overlap_requirement=policy,
    )
    rows = [candidate("left_overlap", "chr8", 90, 110),
            candidate("right_overlap", "chr8", 100, 120)]
    assert filter_pool(tmp_path, rows, compiled) == ["left_overlap", "right_overlap"]


@pytest.mark.parametrize("operator", ["include", "exclude"])
def test_canonical_predicates_survive_intake_normalization(operator):
    rule = {"source_type": "explicit_rows", "coordinate_space": "genomic",
            "rows": [["chr8", 100, 108]], "aso_overlap_requirement": "any_overlap",
            "minimum_overlap_bp": 8, "operator": operator}
    normalized = native._normalize_region_constraints([rule])
    assert normalized[0]["operator"] == operator
    assert native._normalize_region_constraints(normalized) == normalized


@pytest.mark.parametrize("requirement", ["any_overlap", "full_containment", "contains_feature"])
def test_leading_exclusion_uses_base_and_keeps_legal_partial_overlaps(tmp_path, requirement):
    goal = {"materialized_region_constraints": {"global": [
        region_rule("chr8", 130, 150, "genomic", operator="exclude",
                    aso_overlap_requirement=requirement),
    ]}}
    _, compiled = pipeline._apply_shared_region_constraints(
        runtime=native, structured_goal=goal, base_rows=[["chr8", 100, 200]],
        output_space="genomic", program={"length": 20},
        base_overlap_requirement="full_containment",
    )
    rows = [candidate("no_overlap", "chr8", 100, 120),
            candidate("seven_base_overlap", "chr8", 117, 137),
            candidate("exact_feature", "chr8", 130, 150),
            candidate("outside_base", "chr8", 190, 210)]
    assert filter_pool(tmp_path, rows, compiled) == ["no_overlap", "seven_base_overlap"]


def test_leading_exclusion_without_reference_domain_reports_unresolved_scope():
    goal = {"materialized_region_constraints": {"global": [
        region_rule("chr8", 130, 150, "genomic", operator="exclude"),
    ]}}
    regions, compiled = pipeline._apply_shared_region_constraints(
        runtime=native, structured_goal=goal, base_rows=[], output_space="genomic",
        program={"length": 20}, base_overlap_requirement="full_containment",
    )
    assert regions == []
    assert compiled["error"] == "region_constraints_missing_base_for_leading_exclusion"


@pytest.mark.parametrize("unresolved", [
    {"source_type": "explicit_rows", "coordinate_space": "genomic", "rows": []},
    {"source_type": "explicit_rows", "coordinate_space": "cdna",
     "rows": [["TXA.9", 20, 40]]},
])
def test_unresolved_exclusion_is_not_treated_as_an_empty_forbidden_region(tmp_path, unresolved):
    goal = reference_goal("+")
    rule = {**unresolved, "operator": "exclude", "aso_overlap_requirement": "any_overlap"}
    materialized = native._materialize_region_constraint_entry(
        structured_goal=goal, final_payload={}, run_dir=str(tmp_path), constraint=rule,
        isoform_transcripts=goal["isoform_transcripts"])
    goal["materialized_region_constraints"] = {"global": [materialized]}
    regions, compiled = pipeline._apply_shared_region_constraints(
        runtime=native, structured_goal=goal, base_rows=[["chr8", 1200, 1300]],
        output_space="genomic", program={"length": 20, "target_region": "utr3"},
        base_overlap_requirement="full_containment",
    )
    assert regions == []
    assert compiled["error"] == "region_exclusion_unresolved"


@pytest.mark.parametrize("strand,bounds", [("+", [2010, 2030]), ("-", [2170, 2190])])
def test_explicit_transcript_rows_project_their_own_reference_not_primary(strand, bounds):
    goal = reference_goal("+")
    goal["isoform_transcripts"]["secondary"] = {
        "transcript_id": "TXB.2", "chrom": "chr9", "strand": strand,
        "metadata": {"versioned_id": "TXB.2"},
        "exons_tx": [{"index": 1, "start": 2001, "end": 2200}],
    }
    rule = region_rule("TXB.2", 10, 30, "cdna")
    projected = native._constraint_rows_in_output_space(
        structured_goal=goal, constraint=rule, output_space="genomic", base_rows=[],
        isoform_transcripts=goal["isoform_transcripts"], iso_label_primary="primary")
    assert len(projected) == 1
    assert projected[0][:3] == ["chr9", *bounds]
    assert projected[0][5] == strand
