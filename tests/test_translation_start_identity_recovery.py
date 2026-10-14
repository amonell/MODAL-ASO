from copy import deepcopy

import run_goal_intake as intake


def _goal():
    return {
        "intent": "binding",
        "transcript_policy": "user_locked",
        "transcript_lock": {"kind": "explicit_id", "value": "ENST_TEST.1"},
        "design_programs": [{
            "program_id": "p1", "modality": "steric", "intent": "binding",
            "length": 20, "transcript_id": "ENST_TEST.1",
            "target_type": "regulatory_region", "run_splice_eval": False,
            "run_splice_router": False,
        }],
        "evaluation_targets": [{
            "transcript_id": "ENST_TEST.1", "objective_type": "identity_anchor",
            "splice_relative_window": {
                "anchor_site": "acceptor", "reference_transcript_id": "ENST_TEST.1",
                "relative_start_nt": -30, "relative_end_nt": 0,
                "feature_name": "start codon AUG",
            },
        }],
    }


def test_translation_start_identity_recovery_uses_typed_feature_and_full_site_width():
    goal = _goal()
    final = {}
    intake._recover_translation_start_identity_anchor(goal, final)
    assert goal["target_region"] == "translation_start"
    assert goal["design_programs"][0]["target_region"] == "translation_start"
    assert goal["design_programs"][0]["flanks"] == {"up": 30, "down": 21}
    assert "splice_relative_window" not in goal["evaluation_targets"][0]
    assert goal["translation_start_window_repair"]["original_window"]["anchor_site"] == "acceptor"
    assert final["design_programs"] == goal["design_programs"]


def test_translation_start_identity_recovery_preserves_splice_and_conflicting_targets():
    for change in (
        lambda g: g["design_programs"][0].update(intent="splice_modulation"),
        lambda g: g["evaluation_targets"][0].update(transcript_id="ENST_OTHER.1"),
        lambda g: g["design_programs"][0].update(target_region="utr5"),
        lambda g: g["design_programs"][0].update(flanks={"up": 10, "down": 10}),
        lambda g: g.update(region_constraints=[{"source_type": "explicit_rows"}]),
    ):
        goal = deepcopy(_goal())
        change(goal)
        original = deepcopy(goal)
        intake._recover_translation_start_identity_anchor(goal)
        assert goal == original


def test_translation_start_identity_recovery_does_not_reinterpret_other_features():
    goal = _goal()
    goal["evaluation_targets"][0]["splice_relative_window"]["feature_name"] = "splice acceptor"
    original = deepcopy(goal)
    intake._recover_translation_start_identity_anchor(goal)
    assert goal == original


def test_translation_start_window_is_projected_from_mature_transcript_exons():
    goal = {
        "isoform_transcripts": {"chosen": {
            "transcript_id": "ENST_TEST.1", "chrom": "1", "strand": "+",
            "exons_tx": [
                {"index": 1, "start": 1, "end": 100, "contains_coding": False},
                {"index": 2, "start": 201, "end": 300, "contains_coding": True,
                 "coding_start_genomic": 205, "coding_end_genomic": 290},
            ],
        }},
        "transcript_lock": {"kind": "explicit_id", "value": "ENST_TEST.1"},
        "flanks": {"up": 30, "down": 21},
    }
    rows = intake._build_translation_start_bed_rows(goal)
    assert [(row[1], row[2]) for row in rows] == [(74, 100), (200, 226)]
    assert all(row[5] == "+" for row in rows)


def test_translation_start_window_respects_negative_transcript_orientation():
    goal = {
        "isoform_transcripts": {"chosen": {
            "transcript_id": "ENST_NEG.1", "chrom": "2", "strand": "-",
            "exons_tx": [{"index": 1, "start": 300, "end": 399,
                          "contains_coding": True,
                          "coding_start_genomic": 300,
                          "coding_end_genomic": 395}],
        }},
        "transcript_lock": {"kind": "explicit_id", "value": "ENST_NEG.1"},
        "flanks": {"up": 30, "down": 21},
    }
    rows = intake._build_translation_start_bed_rows(goal)
    assert [(row[1], row[2], row[5]) for row in rows] == [(373, 399, "-")]


def test_unattested_one_base_start_anchor_recovers_tileable_default():
    isoforms = {"chosen": {"transcript_id": "ENST_TEST.1", "chrom": "1", "strand": "+",
                           "exons_tx": [{"index": 1, "start": 100, "end": 399,
                                         "contains_coding": True,
                                         "coding_start_genomic": 205,
                                         "coding_end_genomic": 350}]}}
    goal = {"isoform_transcripts": isoforms, "flanks": {"up": 0, "down": 0},
            "target_region": "translation_start", "target_window": None}
    program = {"program_id": "p1", "length": 20, "target_region": "translation_start"}
    rows, metadata = intake._build_program_target_region_rows(
        goal, isoforms, program, output_level="genomic")
    assert [(row[1], row[2]) for row in rows] == [(104, 305)]
    assert metadata["window_recovery"]["reason"] == "unattested_window_shorter_than_oligo"
    assert goal["flanks"] == {"up": 0, "down": 0}

    # An attested user window remains exact even when it is too short to tile.
    explicit = {**goal, "target_window": {"feature": "translation_start", "upstream_nt": 0,
                                           "downstream_nt": 0}}
    rows, metadata = intake._build_program_target_region_rows(
        explicit, isoforms, program, output_level="genomic")
    assert [(row[1], row[2]) for row in rows] == [(204, 205)]
    assert "window_recovery" not in metadata


def test_steric_programs_inherit_panel_target_feature_without_losing_overrides():
    isoforms = {"chosen": {"transcript_id": "ENST_TEST.1", "chrom": "1", "strand": "+",
                           "exons_tx": [{"index": 1, "start": 100, "end": 399,
                                         "contains_coding": True,
                                         "coding_start_genomic": 205,
                                         "coding_end_genomic": 350}]}}
    goal = {"target_region": "translation_start", "isoform_transcripts": isoforms,
            "flanks": {"up": 30, "down": 21}}
    for length in (18, 19, 20):
        program = {"program_id": f"steric_{length}", "modality": "steric",
                   "intent": "binding", "length": length}
        rows, meta = intake._build_steric_program_region_rows(goal, program)
        assert [(row[1], row[2]) for row in rows] == [(174, 226)]
        assert meta["target_region"] == "translation_start"

    overridden = {"program_id": "coding", "modality": "steric", "intent": "binding",
                  "length": 20, "target_region": "cds"}
    rows, meta = intake._build_steric_program_region_rows(goal, overridden)
    assert rows and meta["target_region"] == "cds"
    assert [(row[1], row[2]) for row in rows] != [(174, 226)]

    missing, _ = intake._build_steric_program_region_rows(
        {"isoform_transcripts": isoforms}, {"program_id": "unscoped", "length": 20})
    assert missing == []
