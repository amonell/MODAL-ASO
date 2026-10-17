from copy import deepcopy

import pytest

import run_goal_intake as runtime


def fixture(strand="+"):
    ref, other = "REFERENCE", "COMPARATOR"
    physical = [(100, 130), (1000, 1030), (2000, 2030), (3000, 3030), (4000, 4030)]
    if strand == "-":
        physical.reverse()

    def transcript(identifier, coordinates):
        exons = [{"index": i, "start": start, "end": end, "contains_coding": True}
                 for i, (start, end) in enumerate(coordinates, 1)]
        return {"transcript_id": identifier, "chrom": "chr8", "strand": strand,
                "metadata": {"id": identifier, "versioned_id": identifier + ".4",
                    "exon_nomenclature_resolutions": {
                        "2": {"status": "verified_alternate_mapping", "resolved_transcript_ordinal": 3},
                        "3": {"status": "verified_alternate_mapping", "resolved_transcript_ordinal": 4}}},
                "exons_tx": exons,
                "sites": [{"exon_index": exon["index"], "chrom": "chr8", "strand": strand,
                    "acceptor": exon["start"] if strand == "+" else exon["end"],
                    "donor": exon["end"] if strand == "+" else exon["start"], "coding": True}
                    for exon in exons]}

    isoforms = {ref: transcript(ref, physical), other: transcript(other, physical[:2] + physical[3:])}
    goal = {"intent": "splice_modulation", "exon_label": 2, "focus_exons_only": True,
            "desired_effect": "inclusion", "isoform_transcripts": isoforms,
            "transcript_lock": {"kind": "explicit_id", "value": ref + ".4"},
            "evaluation_targets": [{"objective_type": "primary_outcome", "target_type": "exon",
                                   "transcript_id": ref + ".4", "exon_label": 2}],
            "splice_isoform_objectives": [
                {"isoform_label": ref, "transcript_id": ref, "desired_abundance": "increase"},
                {"isoform_label": other, "transcript_id": other, "desired_abundance": "decrease"}]}
    program = {"transcript_id": ref + ".4", "exon_label": 2, "site_kind": "both",
               "modality": "steric", "label": "Requested named exon"}
    return goal, program


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("binding", ["lock", "evaluation_target", "both"])
def test_named_exon_focus_survives_multiple_isoforms_into_actual_windows(strand, binding):
    goal, program = fixture(strand)
    if binding == "lock":
        goal.pop("evaluation_targets")
    if binding == "evaluation_target":
        goal.pop("transcript_lock")
    original = deepcopy(goal)
    plan, _ = runtime._compute_splice_site_plan(
        goal["splice_isoform_objectives"], goal["isoform_transcripts"], goal)
    assert {entry["position"] for entry in plan} == {2000, 2030}
    assert {entry["site"] for entry in plan} == {"acceptor", "donor"}
    assert all(entry["exon_indices"] == {"REFERENCE": 3} for entry in plan)
    goal["splice_site_plan"] = plan
    rows, metadata = runtime._build_steric_program_site_window_rows(goal, program)
    assert {(row[1], row[2]) for row in rows} == {(1799, 2200), (1829, 2230)}
    assert all(entry["requested_exon_label"] == 2 and entry["resolved_exon_ordinals"] == {"REFERENCE": 3}
               for entry in metadata["resolved_site_targets"])
    assert goal["exon_label"] == original["exon_label"] == 2
    assert goal["splice_isoform_objectives"] == original["splice_isoform_objectives"]


@pytest.mark.parametrize("site", ["acceptor", "donor", "both"])
def test_program_does_not_substitute_other_reference_or_unrelated_direct_route(site):
    goal, program = fixture()
    goal["splice_site_plan"] = [
        {"chrom": "chr8", "position": 1000, "strand": "+", "site": "acceptor",
         "delta_direction": "increase", "covers_all_isoforms": False, "exon_indices": {"REFERENCE": 2}},
        {"chrom": "chr8", "position": 4000, "strand": "+", "site": "donor",
         "delta_direction": "increase", "covers_all_isoforms": False, "exon_indices": {"COMPARATOR": 3}}]
    program["site_kind"] = site
    assert runtime._collect_program_splice_plan_entries(goal, program) == []


def test_conflicting_exact_reference_version_cannot_select_a_default():
    goal, program = fixture()
    goal["splice_site_plan"] = [{"site": "acceptor", "exon_indices": {"REFERENCE": 3}}]
    program["transcript_id"] = "REFERENCE.9"
    assert runtime._collect_program_splice_plan_entries(goal, program) == []


def test_single_reference_region_padding_uses_mapped_feature_once():
    goal, program = fixture()
    goal["isoform_transcripts"].pop("COMPARATOR")
    goal["splice_isoform_objectives"] = goal["splice_isoform_objectives"][:1]
    plan, _ = runtime._compute_splice_site_plan(goal["splice_isoform_objectives"], goal["isoform_transcripts"], goal)
    regions = runtime._compute_steric_regions(plan, goal["isoform_transcripts"], goal)
    assert regions is not None
    # Every focused region must cover the resolved feature, never the preceding
    # annotation exon whose ordinal happens to equal the requested name.
    assert regions["global_region"]["start"] <= 2000 and regions["global_region"]["end"] >= 2030
    assert regions["global_region"]["start"] > 1030
