from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools


def _write_zip(path: Path, member_name: str, text: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, text)


def test_binding_site_schema_normalization_converts_semantic_fields() -> None:
    payload: dict[str, Any] = {
        "status": "ready",
        "structured_goal": {
            "gene_symbol": "CDKN1B",
            "species": "human",
            "target_context": "general human baseline",
            "intent": "binding",
            "target_region": "3' UTR",
            "binding_site_tiling": {
                "factor": "PUM1",
                "assay": "eCLIP",
                "overlap_rule": "any_overlap_counts",
            },
            "target_features": [
                {
                    "feature_type": "miRNA_binding_site",
                    "source": "TargetScan",
                    "miRNA_family_or_id": "miR-221/miR-222",
                    "overlap_rule": "any_overlap_counts",
                }
            ],
        },
    }

    out = runtime._normalize_intake_output(payload)
    goal = out["structured_goal"]

    assert out["status"] == "ready"
    assert goal["target_region"] == "utr3"
    assert goal["rbp_binding"]["rbp"] == "PUM1"
    assert goal["rbp_binding"]["aso_overlap_requirement"] == "any_overlap"
    assert goal["mirna_binding"]["source"] == "TargetScan"
    assert goal["mirna_binding"]["target_level"] == "mature_mrna"
    assert set(goal["mirna_binding"]["mirnas"]) >= {"miR-221", "miR-222"}
    assert len(goal["region_constraints"]) == 1

    constraints = runtime._goal_region_constraints(goal)
    assert {entry["source_type"] for entry in constraints} == {"rbp_binding", "mirna_binding"}


@pytest.mark.parametrize("source", ["TargetScan", "TargetScan miRNA predictions", "TargetScanHuman database"])
def test_targetscan_mirna_sites_project_utr_coordinates_to_genome(tmp_path: Path, monkeypatch: Any, source: str) -> None:
    family_zip = tmp_path / "miR_Family_Info.txt.zip"
    predicted_zip = tmp_path / "Predicted_Targets_Info.default_predictions.txt.zip"
    utr_file = tmp_path / 'utr.tsv'
    utr_file.write_text('Refseq ID\tGene ID\tGene Symbol\tSpecies ID\tUTR sequence\nENSTTEST.4\tENSGTEST.1\tTEST1\t9606\tAACCGGTTAACCGGTT\n')
    monkeypatch.setenv('ASO_TARGETSCAN_UTR_SEQUENCES', str(utr_file))
    monkeypatch.setattr(runtime, '_build_isoform_cdna_sequences', lambda *args, **kwargs:{'ENSTTEST':'C'*200+'AACCGGTTAACCGGTT'+'C'*284})
    _write_zip(
        family_zip,
        "miR_Family_Info.txt",
        "\n".join(
            [
                "miR family\tSeed+m8\tSpecies ID\tMiRBase ID\tMature sequence\tFamily Conservation?\tMiRBase Accession",
                "miR-999-3p\tAAAAAAA\t9606\thsa-miR-999-3p\tUAAAAAAA\t2\tMIMATTEST",
            ]
        ),
    )
    _write_zip(
        predicted_zip,
        "Predicted_Targets_Info.default_predictions.txt",
        "\n".join(
            [
                "miR Family\tGene ID\tGene Symbol\tTranscript ID\tSpecies ID\tUTR start\tUTR end\tMSA start\tMSA end\tSeed match\tPCT",
                "miR-999-3p\tENSGTEST.1\tTEST1\tENSTTEST.4\t9606\t5\t12\t5\t12\t8mer\t0.75",
            ]
        ),
    )
    monkeypatch.setenv("ASO_TARGETSCAN_MIR_FAMILY_INFO", str(family_zip))
    monkeypatch.setenv("ASO_TARGETSCAN_PREDICTIONS_INFO", str(predicted_zip))
    runtime._targetscan_family_alias_index.cache_clear()

    def fake_fetch_transcript_structure(transcript_id: str) -> dict[str, Any]:
        assert transcript_id == "ENSTTEST"
        return {
            "seq_region_name": "12",
            "strand": 1,
            "exons_tx": [
                {
                    "index": 1,
                    "start": 100,
                    "end": 599,
                    "cdna_start": 1,
                    "cdna_end": 500,
                    "coding_start": 1,
                    "coding_end": 200,
                    "coding_start_genomic": 100,
                    "coding_end_genomic": 299,
                    "contains_coding": True,
                }
            ],
        }

    monkeypatch.setattr(runtime, "_fetch_transcript_structure", fake_fetch_transcript_structure)

    rows, cdna_rows, summary = runtime._materialize_mirna_binding_rows(
        {
            "mirnas": ["miR-999"],
            "source": source,
            "target_level": "mature_mrna",
            "aso_overlap_requirement": "any_overlap",
        },
        {"gene_symbol": "TEST1", "species": "human"},
        str(tmp_path),
    )

    assert rows == [["chr12", 303, 311, "targetscan_mir_999_3p_8mer", 0.75, "+"]]
    assert cdna_rows == [["ENSTTEST", 204, 212, "targetscan_mir_999_3p_8mer", 0.75, "+"]]
    assert summary["status"] == "materialized"
    assert summary["matched_families"] == ["miR-999-3p"]
    assert summary['site_records'][0]['sequence_projection']['status'] == 'verified'


def test_design_preseed_preserves_binding_site_constraints() -> None:
    preseed = pipeline_tools._prepare_design_objective_preseed(
        {},
        {
            "gene_symbol": "CDKN1B",
            "species": "human",
            "target_context": "general human baseline",
            "intent": "binding",
            "target_region": "utr3",
            "binding_site_tiling": {"factor": "PUM1", "assay": "eCLIP"},
            "mirna_binding": {"mirnas": ["miR-221", "miR-222"], "source": "TargetScan"},
        },
    )

    overlay = preseed["structured_goal_overlay"]
    assert overlay["binding_site_tiling"]["factor"] == "PUM1"
    assert overlay["mirna_binding"]["source"] == "TargetScan"
    assert overlay["target_region"] == "utr3"


def test_advisory_splice_region_preference_does_not_force_coordinate_clarification() -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "LMNA",
        "species": "human",
        "intent": "splice_modulation",
        "region_constraints": [
            {
                "label": "prefer_splice_regulatory_regions",
                "kind": "region_constraints",
                "notes": "Focus candidate binding sites near splice sites and regulatory motifs.",
            }
        ],
        "design_programs": [
            {
                "program_id": "steric_lmna_01",
                "modality": "steric",
                "intent": "splice_modulation",
            }
        ],
    }

    assert runtime._site_constraint_coordinates_required(goal) is False
    assert runtime._site_constraint_coordinate_clarification_payload(goal) is None

    runtime._normalize_region_constraints_in_goal(goal)
    assert "region_constraints" not in goal


def test_broad_utr_motif_tiling_does_not_force_coordinate_clarification() -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "CDKN1B",
        "species": "human",
        "intent": "binding",
        "target_region": "utr3",
        "motif_tiling": {
            "region": "utr3",
            "strategy": "tiling",
        },
        "design_programs": [
            {
                "program_id": "steric_cdkn1b_01",
                "modality": "steric",
                "intent": "binding",
                "target_region": "utr3",
            }
        ],
    }

    assert runtime._extract_motif_tiling_request(goal) is None
    assert runtime._site_constraint_coordinates_required(goal) is False
    assert runtime._site_constraint_coordinate_clarification_payload(goal) is None


def test_exon_region_constraint_materializes_to_cdna_and_genomic_rows(tmp_path: Path) -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "TEST1",
        "species": "human",
        "region_constraints": [
            {
                "constraint_kind": "transcript_region",
                "operator": "include",
                "transcript_id": "ENST000TEST",
                "region": {"exon_label": "exon 4"},
            }
        ],
        "isoform_transcripts": {
            "ENST000TEST": {
                "transcript_id": "ENST000TEST",
                "chrom": "chr1",
                "strand": "+",
                "exons_tx": [
                    {"index": 1, "start": 100, "end": 149, "cdna_start": 1, "cdna_end": 50},
                    {"index": 2, "start": 200, "end": 249, "cdna_start": 51, "cdna_end": 100},
                    {"index": 3, "start": 300, "end": 349, "cdna_start": 101, "cdna_end": 150},
                    {"index": 4, "start": 400, "end": 449, "cdna_start": 151, "cdna_end": 200},
                ],
            }
        },
    }
    final_payload: dict[str, Any] = {}

    runtime._normalize_region_constraints_in_goal(goal)
    assert goal["region_constraints"][0]["aso_overlap_requirement"] == "full_containment"
    summary = runtime._materialize_region_constraints(goal, final_payload, str(tmp_path))
    entry = goal["materialized_region_constraints"]["global"][0]

    assert summary["constraint_count"] == 1
    assert entry["source_type"] == "exon"
    assert entry["exon_label"] == 4
    assert entry["materialized_cdna_rows"] == [["ENST000TEST", 150, 200, "constraint_01", 0, "+"]]
    assert entry["materialized_rows"] == [["chr1", 399, 449, "constraint_01", 0, "+"]]


def test_splice_relative_window_materializes_from_transcript_exon(tmp_path: Path) -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "TEST1",
        "species": "human",
        "design_programs": [
            {
                "program_id": "acceptor_site",
                "modality": "steric",
                "region_constraints": [
                    {
                        "constraint_id": "acceptor_window",
                        "source_type": "splice_relative_window",
                        "aso_overlap_requirement": "full_containment",
                        "splice_relative_window": {
                            "exon_label": 4,
                            "anchor_site": "acceptor",
                            "relative_start_nt": -25,
                            "relative_end_nt": 24,
                            "reference_transcript_id": "ENST000TEST",
                        },
                    }
                ],
            },
            {
                "program_id": "donor_site",
                "modality": "steric",
                "region_constraints": [
                    {
                        "constraint_id": "donor_window",
                        "source_type": "splice_relative_window",
                        "aso_overlap_requirement": "full_containment",
                        "splice_relative_window": {
                            "exon_label": 4,
                            "anchor_site": "donor",
                            "relative_start_nt": -9,
                            "relative_end_nt": 60,
                            "reference_transcript_id": "ENST000TEST",
                        },
                    }
                ],
            },
        ],
        "isoform_transcripts": {
            "ENST000TEST": {
                "transcript_id": "ENST000TEST",
                "chrom": "chr1",
                "strand": "+",
                "exons_tx": [
                    {"index": 4, "start": 400, "end": 449, "cdna_start": 151, "cdna_end": 200},
                ],
            }
        },
    }
    final_payload: dict[str, Any] = {}

    runtime._normalize_region_constraints_in_goal(goal)
    summary = runtime._materialize_region_constraints(goal, final_payload, str(tmp_path))
    by_program = goal["materialized_region_constraints"]["by_program"]

    assert summary["constraint_count"] == 2
    assert by_program["acceptor_site"][0]["materialized_rows"] == [["chr1", 374, 424, "acceptor_window", 0, "+"]]
    assert by_program["donor_site"][0]["materialized_rows"] == [["chr1", 439, 509, "donor_window", 0, "+"]]


def test_canonical_nested_splice_window_does_not_require_redundant_outer_type(
    tmp_path: Path,
) -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "TEST1",
        "species": "human",
        "design_programs": [
            {
                "program_id": "explicit_window",
                "modality": "steric",
                "region_constraints": [
                    {
                        "constraint_id": "window_01",
                        "source": "user_request",
                        "aso_overlap_requirement": "full_containment",
                        "splice_relative_window": {
                            "exon_label": 4,
                            "anchor_site": "acceptor",
                            "relative_start_nt": -12,
                            "relative_end_nt": 8,
                            "reference_transcript_id": "ENST000TEST",
                        },
                    }
                ],
            }
        ],
        "isoform_transcripts": {
            "ENST000TEST": {
                "transcript_id": "ENST000TEST",
                "chrom": "chr1",
                "strand": "+",
                "exons_tx": [
                    {"index": 4, "start": 400, "end": 449, "cdna_start": 151, "cdna_end": 200},
                ],
            }
        },
    }
    final_payload: dict[str, Any] = {}

    runtime._normalize_region_constraints_in_goal(goal)
    constraint = goal["design_programs"][0]["region_constraints"][0]
    assert constraint["source_type"] == "splice_relative_window"
    assert constraint["splice_relative_window"]["anchor_site"] == "acceptor"

    summary = runtime._materialize_region_constraints(goal, final_payload, str(tmp_path))
    materialized = goal["materialized_region_constraints"]["by_program"]["explicit_window"][0]
    assert summary["constraint_count"] == 1
    assert materialized["materialized_rows"] == [["chr1", 387, 408, "window_01", 0, "+"]]


def test_named_motif_with_fallback_does_not_create_fake_sequence_motif() -> None:
    goal = {
        "gene_symbol": "TEST1",
        "species": "human",
        "motif_tiling": {
            "motifs": [
                {
                    "motif_name": "ISS-N1",
                    "region_hint": {
                        "relative_to": {
                            "exon_label": "exon 4",
                            "splice_site": "donor",
                            "exon_numbering_transcript_id": "ENST000TEST",
                        },
                        "intron_downstream_range_nt": [1, 60],
                    },
                    "aso_overlap_requirement": "any_overlap",
                }
            ]
        },
    }

    assert runtime._extract_motif_tiling_request(goal) is None
    constraints = runtime._goal_region_constraints(goal)

    assert constraints[0]["source_type"] == "splice_relative_window"
    assert constraints[0]["splice_relative_window"] == {
        "exon_label": 4,
        "anchor_site": "donor",
        "relative_start_nt": 1,
        "relative_end_nt": 60,
        "reference_transcript_id": "ENST000TEST",
        "feature_name": "ISS-N1",
    }


def test_constraint_notes_with_downstream_feature_range_normalize_to_exact_splice_window() -> None:
    goal = {
        "gene_symbol": "TEST1",
        "species": "human",
        "region_constraints": [
            {
                "constraint_id": "constraint_01",
                "label": "iss_n1_downstream_intronic_silencer",
                "source_type": "exon",
                "exon_label": 7,
                "exon_numbering_transcript_id": "ENST000TEST",
                "notes": (
                    "ISS-N1/downstream silencer window from +10 to +35 nt downstream "
                    "of the exon 7 donor, with enough padding to tile 15-mers fully "
                    "across that interval."
                ),
            }
        ],
    }

    runtime._normalize_region_constraints_in_goal(goal)

    constraint = goal["region_constraints"][0]
    assert constraint["source_type"] == "splice_relative_window"
    assert constraint["aso_overlap_requirement"] == "any_overlap"
    assert constraint["splice_relative_window"] == {
        "exon_label": 7,
        "anchor_site": "donor",
        "relative_start_nt": 10,
        "relative_end_nt": 35,
        "reference_transcript_id": "ENST000TEST",
        "feature_name": "ISS-N1",
    }


def test_user_provided_binding_intervals_are_materialized_verbatim(monkeypatch: Any, tmp_path: Path) -> None:
    def fail_rbp(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("RBP lookup should not run for explicit intervals")

    def fail_mirna(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("miRNA lookup should not run for explicit intervals")

    monkeypatch.setattr(runtime, "_materialize_rbp_binding_rows", fail_rbp)
    monkeypatch.setattr(runtime, "_materialize_mirna_binding_rows", fail_mirna)

    goal: dict[str, Any] = {
        "gene_symbol": "CDKN1B",
        "species": "human",
        "intent": "binding",
        "target_region": "utr3",
        "region_constraints": [
            {
                "constraint_id": "pum1_user",
                "source_type": "rbp_binding",
                "strand": "+",
                "rbp_binding": {"rbp": "PUM1", "aso_overlap_requirement": "any_overlap"},
                "intervals_bed0_half_open": [
                    {"chrom": "chr12", "start": 12721313, "end": 12721416, "name": "PUM1_eCLIP_site_1"},
                    {"chrom": "chr12", "start": 12721400, "end": 12721460, "name": "PUM1_eCLIP_site_2"},
                ],
            },
            {
                "constraint_id": "mir_user",
                "source_type": "mirna_binding",
                "strand": "+",
                "mirna_binding": {
                    "mirnas": ["miR-221", "miR-222"],
                    "source": "user_provided_seed_windows",
                    "aso_overlap_requirement": "any_overlap",
                },
                "intervals_bed0_half_open": [
                    {"chrom": "chr12", "start": 12721208, "end": 12721253, "name": "miR221_222_seed_window_1"},
                    {"chrom": "chr12", "start": 12721281, "end": 12721326, "name": "miR221_222_seed_window_2"},
                ],
            },
        ],
    }
    final_payload: dict[str, Any] = {}

    summary = runtime._materialize_region_constraints(goal, final_payload, str(tmp_path))
    entries = goal["materialized_region_constraints"]["global"]

    assert summary["constraint_count"] == 2
    assert entries[0]["materialized_rows"] == [
        ["chr12", 12721313, 12721416, "PUM1_eCLIP_site_1", 0, "+"],
        ["chr12", 12721400, 12721460, "PUM1_eCLIP_site_2", 0, "+"],
    ]
    assert entries[1]["materialized_rows"] == [
        ["chr12", 12721208, 12721253, "miR221_222_seed_window_1", 0, "+"],
        ["chr12", 12721281, 12721326, "miR221_222_seed_window_2", 0, "+"],
    ]
    assert entries[0]["materialization_summary"]["source"] == "user_provided_intervals"
    assert entries[1]["aso_overlap_requirement"] == "any_overlap"


def test_user_provided_bed_interval_schema_seeds_binding_constraints(tmp_path: Path) -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "CDKN1B",
        "species": "human",
        "intent": "binding",
        "target_region": "utr3",
        "rbp_binding": {
            "rbp": "PUM1",
            "assay": "eCLIP",
            "aso_overlap_requirement": "any_overlap",
        },
        "mirna_binding": {
            "mirnas": ["miR-221", "miR-222"],
            "source": "user_provided_intervals",
            "aso_overlap_requirement": "any_overlap",
        },
        "design_programs": [
            {
                "program_id": "steric_cdkn1b_01",
                "modality": "steric",
                "intent": "binding",
                "oligo_length_nt": 20,
            }
        ],
        "region_constraints": [
            {
                "constraint_id": "pum1_eclip_sites_user_bed",
                "label": "PUM1/eCLIP sites (user-provided GRCh38 intervals)",
                "source": "user_provided_bed_intervals",
                "operator": "union",
                "aso_overlap_requirement": "any_overlap",
                "minimum_overlap_bp": 1,
                "strand": "+",
                "bed_intervals": [
                    {"name": "PUM1_eCLIP_site_1", "chrom": "chr12", "start": 12721313, "end": 12721416, "strand": "+"},
                    {"name": "PUM1_eCLIP_site_2", "chrom": "chr12", "start": 12721400, "end": 12721460, "strand": "+"},
                ],
            },
            {
                "constraint_id": "mir221_222_seed_windows_user_bed",
                "label": "miR-221/miR-222 seed windows (user-provided GRCh38 intervals)",
                "source": "user_provided_bed_intervals",
                "operator": "union",
                "aso_overlap_requirement": "any_overlap",
                "minimum_overlap_bp": 1,
                "strand": "+",
                "bed_intervals": [
                    {"name": "miR_221_222_seed_window_1", "chrom": "chr12", "start": 12721208, "end": 12721253, "strand": "+"},
                    {"name": "miR_221_222_seed_window_2", "chrom": "chr12", "start": 12721281, "end": 12721326, "strand": "+"},
                ],
            },
        ],
    }
    final_payload: dict[str, Any] = {}

    runtime._normalize_region_constraints_in_goal(goal)
    summary = runtime._materialize_region_constraints(goal, final_payload, str(tmp_path))
    entries = goal["materialized_region_constraints"]["global"]
    constrained_rows, meta = runtime._apply_region_constraints_to_rows(
        goal,
        [],
        output_space="genomic",
        program=goal["design_programs"][0],
    )

    assert summary["constraint_count"] == 2
    assert len(entries) == 2
    assert [row[:6] for entry in entries for row in entry["materialized_rows"]] == [
        ["chr12", 12721313, 12721416, "PUM1_eCLIP_site_1", 0, "+"],
        ["chr12", 12721400, 12721460, "PUM1_eCLIP_site_2", 0, "+"],
        ["chr12", 12721208, 12721253, "miR_221_222_seed_window_1", 0, "+"],
        ["chr12", 12721281, 12721326, "miR_221_222_seed_window_2", 0, "+"],
    ]
    assert meta["base_row_count"] == 0
    assert meta["constraint_count"] == 2
    assert sorted(meta["tile_filter_rows"], key=lambda row: (row[0], row[1], row[2], row[3])) == [
        ["chr12", 12721208, 12721253, "miR_221_222_seed_window_1", 0, "+"],
        ["chr12", 12721281, 12721326, "miR_221_222_seed_window_2", 0, "+"],
        ["chr12", 12721313, 12721416, "PUM1_eCLIP_site_1", 0, "+"],
        ["chr12", 12721400, 12721460, "PUM1_eCLIP_site_2", 0, "+"],
    ]
    assert constrained_rows
    assert min(int(row[1]) for row in constrained_rows) < 12721208


def test_genomic_interval_set_schema_preserves_binding_constraints(tmp_path: Path) -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "CDKN1B",
        "species": "human",
        "intent": "binding",
        "target_region": "utr3",
        "aso_overlap_requirement": "any_overlap",
        "design_programs": [
            {
                "program_id": "steric_cdkn1b_01",
                "modality": "steric",
                "intent": "binding",
                "oligo_length_nt": 20,
            }
        ],
        "region_constraints": [
            {
                "constraint_name": "User-specified negative-regulatory elements to block",
                "constraint_type": "genomic_interval_set",
                "genome_build": "GRCh38",
                "strand": "+",
                "operator": "union",
                "aso_overlap_requirement": "any_overlap",
                "minimum_overlap_bp": 1,
                "intervals": [
                    {"chrom": "chr12", "start": 12721313, "end": 12721416, "label": "PUM1/eCLIP site 1"},
                    {"chrom": "chr12", "start": 12721400, "end": 12721460, "label": "PUM1/eCLIP site 2"},
                    {"chrom": "chr12", "start": 12721208, "end": 12721253, "label": "miR-221/miR-222 seed window 1"},
                    {"chrom": "chr12", "start": 12721281, "end": 12721326, "label": "miR-221/miR-222 seed window 2"},
                ],
            }
        ],
    }
    final_payload: dict[str, Any] = {}

    runtime._normalize_region_constraints_in_goal(goal)
    summary = runtime._materialize_region_constraints(goal, final_payload, str(tmp_path))
    constrained_rows, meta = runtime._apply_region_constraints_to_rows(
        goal,
        [],
        output_space="genomic",
        program=goal["design_programs"][0],
    )

    assert summary["constraint_count"] == 1
    assert goal["region_constraints"][0]["source_type"] == "explicit_rows"
    assert goal["region_constraints"][0]["intervals_bed0_half_open"][0]["label"] == "PUM1/eCLIP site 1"
    assert [row[:6] for row in goal["materialized_region_constraints"]["global"][0]["materialized_rows"]] == [
        ["chr12", 12721313, 12721416, "PUM1/eCLIP site 1", 0, "+"],
        ["chr12", 12721400, 12721460, "PUM1/eCLIP site 2", 0, "+"],
        ["chr12", 12721208, 12721253, "miR-221/miR-222 seed window 1", 0, "+"],
        ["chr12", 12721281, 12721326, "miR-221/miR-222 seed window 2", 0, "+"],
    ]
    assert meta["tile_filter_rows"] == [
        ["chr12", 12721313, 12721416, "PUM1/eCLIP site 1", 0, "+"],
        ["chr12", 12721400, 12721460, "PUM1/eCLIP site 2", 0, "+"],
        ["chr12", 12721208, 12721253, "miR-221/miR-222 seed window 1", 0, "+"],
        ["chr12", 12721281, 12721326, "miR-221/miR-222 seed window 2", 0, "+"],
    ]
    assert constrained_rows
    state = {"request": "Block chr12:12721313-12721416 and chr12:12721208-12721253 in the 3' UTR."}
    assert pipeline_tools._request_coordinate_preseed_if_available(state, goal) == {}


def test_binding_site_overlap_block_patches_region_constraints() -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "CDKN1B",
        "species": "human",
        "intent": "binding",
        "target_region": "utr3",
        "binding_site_constraints": {
            "aso_overlap_requirement": "any_overlap",
            "minimum_overlap_bp": 1,
        },
        "region_constraints": [
            {
                "constraint_id": "pum1",
                "source_type": "explicit_rows",
                "coordinate_space": "genomic",
                "intervals_bed0_half_open": [
                    {"chrom": "chr12", "start": 12721313, "end": 12721416, "name": "PUM1/eCLIP site 1"},
                ],
            }
        ],
    }

    runtime._normalize_region_constraints_in_goal(goal)

    assert "binding_site_constraints" not in goal
    assert goal["aso_overlap_requirement"] == "any_overlap"
    assert goal["minimum_overlap_bp"] == 1
    assert goal["region_constraints"][0]["aso_overlap_requirement"] == "any_overlap"
    assert goal["region_constraints"][0]["minimum_overlap_bp"] == 1


def test_region_constraint_preserves_coordinate_convention() -> None:
    constraints = runtime._normalize_region_constraints(
        [
            {
                "source_type": "explicit_rows",
                "coordinate_space": "genomic",
                "coordinate_convention": "BED-style 0-based half-open",
                "intervals_bed0_half_open": [
                    {
                        "chrom": "chr1",
                        "start": 10,
                        "end": 20,
                        "label": "supplied interval",
                        "strand": "+",
                    }
                ],
            }
        ]
    )

    assert constraints[0]["coordinate_convention"] == (
        "BED-style 0-based half-open"
    )


def test_request_text_overlap_hints_patch_explicit_interval_constraints() -> None:
    request_text = (
        "For all constrained binding sites, any overlap counts. Even 1 nt overlap is sufficient. "
        "If a numeric minimum_overlap_bp is required, set it to 1. Do not require full containment."
    )
    payload: dict[str, Any] = {
        "status": "ready",
        "structured_goal": {
            "gene_symbol": "CDKN1B",
            "species": "human",
            "intent": "binding",
            "target_region": "utr3",
            "target_context": "human HEK293T cells in vitro",
            "region_constraints": [
                {
                    "constraint_id": "pum1",
                    "source_type": "explicit_rows",
                    "coordinate_space": "genomic",
                    "intervals_bed0_half_open": [
                        {"chrom": "chr12", "start": 12721313, "end": 12721416, "strand": "+"},
                    ],
                },
                {
                    "constraint_id": "mirna",
                    "source_type": "explicit_rows",
                    "coordinate_space": "genomic",
                    "intervals_bed0_half_open": [
                        {"chrom": "chr12", "start": 12721208, "end": 12721253, "strand": "+"},
                    ],
                },
            ],
        },
    }

    normalized = runtime._normalize_intake_output(payload, request_text=request_text)
    goal = normalized["structured_goal"]

    assert normalized["status"] == "ready"
    assert goal["aso_overlap_requirement"] == "any_overlap"
    assert goal["minimum_overlap_bp"] == 1
    assert {entry["aso_overlap_requirement"] for entry in goal["region_constraints"]} == {"any_overlap"}
    assert {entry["minimum_overlap_bp"] for entry in goal["region_constraints"]} == {1}


def test_enforce_overlap_requirement_uses_request_text_hints() -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "CDKN1B",
        "species": "human",
        "intent": "binding",
        "target_region": "utr3",
        "region_constraints": [
            {
                "constraint_id": "pum1",
                "source_type": "explicit_rows",
                "coordinate_space": "genomic",
                "intervals_bed0_half_open": [
                    {"chrom": "chr12", "start": 12721313, "end": 12721416, "strand": "+"},
                ],
            }
        ],
    }
    final_payload: dict[str, Any] = {}

    runtime._enforce_region_constraint_overlap_requirement(
        goal,
        final_payload,
        request_text="Accept any overlap with each site. Even 1 nt overlap is sufficient.",
    )

    assert goal["region_constraints"][0]["aso_overlap_requirement"] == "any_overlap"
    assert goal["region_constraints"][0]["minimum_overlap_bp"] == 1
    assert final_payload["aso_overlap_requirement"] == "any_overlap"


def test_enforce_overlap_requirement_raises_an_ordinary_validation_error() -> None:
    goal: dict[str, Any] = {
        "region_constraints": [
            {
                "source_type": "explicit_rows",
                "coordinate_space": "genomic",
                "intervals_bed0_half_open": [
                    {"chrom": "chr1", "start": 100, "end": 200, "strand": "+"},
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="aso_overlap_requirement"):
        runtime._enforce_region_constraint_overlap_requirement(goal, {})


def test_explicit_interval_constraints_annotate_target_feature_overlaps(tmp_path: Path) -> None:
    tiles_path = tmp_path / "tiles.json"
    tiles_path.write_text(
        json.dumps(
            {
                "schema": {"columns": {}},
                "rows": [
                    {"Chrom": "chr1", "Start": 99, "End": 119, "Sequence": "A" * 20},
                    {"Chrom": "chr1", "Start": 179, "End": 199, "Sequence": "C" * 20},
                    {"Chrom": "chr1", "Start": 299, "End": 319, "Sequence": "G" * 20},
                ],
            }
        ),
        encoding="utf-8",
    )
    goal = {
        "materialized_region_constraints": {
            "global": [
                {
                    "source_type": "explicit_rows",
                    "coordinate_space": "genomic",
                    "materialized_rows": [
                        ["chr1", 100, 130, "first_user_interval", 0, "+"],
                        ["chr1", 180, 210, "second_user_interval", 0, "+"],
                    ],
                }
            ],
            "by_program": {},
        },
    }

    summary = runtime._annotate_tiles_target_feature_overlaps(
        str(tiles_path),
        goal,
        output_space="genomic",
    )
    rows = json.loads(tiles_path.read_text(encoding="utf-8"))["rows"]

    assert summary["applied"] is True
    assert summary["feature_count"] == 2
    assert summary["annotated_rows"] == 3
    assert summary["overlapping_rows"] == 2
    assert rows[0]["Overlap_feature"] == "first_user_interval"
    assert rows[1]["Overlap_feature"] == "second_user_interval"
    assert rows[2]["Overlap_feature"] == ""


def test_gapmer_program_exon_focus_precedes_broad_target_region() -> None:
    isoforms = {
        "ENST000TEST": {
            "transcript_id": "ENST000TEST",
            "chrom": "chr1",
            "strand": "+",
            "exons_tx": [
                {"index": 1, "start": 100, "end": 149, "cdna_start": 1, "cdna_end": 50},
                {"index": 2, "start": 200, "end": 249, "cdna_start": 51, "cdna_end": 100},
                {"index": 3, "start": 300, "end": 349, "cdna_start": 101, "cdna_end": 150},
                {"index": 4, "start": 400, "end": 449, "cdna_start": 151, "cdna_end": 200},
            ],
        }
    }
    goal = {
        "gene_symbol": "TEST1",
        "species": "human",
        "target_region": "cds",
        "exon_label": 4,
        "isoform_transcripts": isoforms,
    }
    program = {
        "program_id": "gapmer_test_01",
        "label": "TEST1 exon 4 gapmers",
        "modality": "gapmer",
        "target_region": "cds",
    }

    rows, meta = runtime._build_gapmer_program_region_rows(goal, isoforms, "ENST000TEST", program)

    assert meta["sources"] == ["program_exon"]
    assert meta["match_mode"] == "exon_overlap"
    assert meta["tile_overlap_requirement"] == "full_containment"
    assert meta["exon_label"] == 4
    assert rows == [["ENST000TEST", 150, 200, "TEST1 exon 4 gapmers", 0, "+"]]


def test_empty_materialized_exon_constraint_is_resolved_when_transcripts_are_available() -> None:
    isoforms = {
        "ENST000TEST": {
            "transcript_id": "ENST000TEST",
            "chrom": "chr1",
            "strand": "+",
            "exons_tx": [
                {"index": 1, "start": 100, "end": 149, "cdna_start": 1, "cdna_end": 50},
                {"index": 2, "start": 200, "end": 249, "cdna_start": 51, "cdna_end": 100},
                {"index": 3, "start": 300, "end": 349, "cdna_start": 101, "cdna_end": 150},
                {"index": 4, "start": 400, "end": 449, "cdna_start": 151, "cdna_end": 200},
            ],
        }
    }
    goal = {
        "isoform_transcripts": isoforms,
        "materialized_region_constraints": {
            "global": [
                {
                    "constraint_id": "constraint_01",
                    "label": "constraint_01",
                    "source_type": "exon",
                    "exon_label": 4,
                    "transcript_id": "ENST000TEST",
                    "aso_overlap_requirement": "any_overlap",
                    "materialized_rows": [],
                    "materialized_cdna_rows": [],
                }
            ],
            "by_program": {},
        },
    }

    rows, meta = runtime._apply_region_constraints_to_rows(
        goal,
        [["ENST000TEST", 150, 200, "exon4", 0, "+"]],
        output_space="cdna",
    )

    assert rows == [["ENST000TEST", 150, 200, "constraint_01|exon4", 0, "+"]]
    assert meta["applied_constraints"][0]["source_row_count"] == 1
    assert meta["tile_filter_rows"] == [["ENST000TEST", 150, 200, "constraint_01|exon4", 0, "+"]]


def test_empty_exon_materialization_refreshes_for_persisted_provenance(tmp_path: Path) -> None:
    isoforms = {
        "ENST000TEST": {
            "transcript_id": "ENST000TEST",
            "chrom": "chr1",
            "strand": "+",
            "exons_tx": [
                {"index": 1, "start": 100, "end": 149, "cdna_start": 1, "cdna_end": 50},
                {"index": 2, "start": 200, "end": 249, "cdna_start": 51, "cdna_end": 100},
                {"index": 3, "start": 300, "end": 349, "cdna_start": 101, "cdna_end": 150},
                {"index": 4, "start": 400, "end": 449, "cdna_start": 151, "cdna_end": 200},
            ],
        }
    }
    goal = {
        "isoform_transcripts": isoforms,
        "materialized_region_constraints": {
            "global": [
                {
                    "constraint_id": "constraint_01",
                    "label": "constraint_01",
                    "source_type": "exon",
                    "exon_label": 4,
                    "transcript_id": "ENST000TEST",
                    "aso_overlap_requirement": "full_containment",
                    "materialized_rows": [],
                    "materialized_cdna_rows": [],
                    "materialization_summary": {
                        "status": "empty",
                        "reason": "exon_not_found",
                    },
                }
            ],
            "by_program": {},
            "summary": {"status": "materialized", "constraint_count": 1},
        },
    }
    final_payload: dict[str, Any] = {}

    changed = pipeline_tools._refresh_empty_exon_region_constraints(
        runtime=runtime,
        structured_goal=goal,
        final_payload=final_payload,
        run_dir=str(tmp_path),
        isoform_transcripts=isoforms,
        iso_label_primary="ENST000TEST",
    )
    entry = goal["materialized_region_constraints"]["global"][0]

    assert changed is True
    assert entry["materialized_cdna_rows"] == [["ENST000TEST", 150, 200, "constraint_01", 0, "+"]]
    assert entry["materialized_rows"] == [["chr1", 399, 449, "constraint_01", 0, "+"]]
    assert final_payload["materialized_region_constraints"]["global"][0]["row_count"] == 1


def test_malformed_explicit_region_constraint_still_requests_coordinates() -> None:
    goal: dict[str, Any] = {
        "gene_symbol": "TEST1",
        "species": "human",
        "intent": "binding",
        "region_constraints": [
            {
                "source_type": "explicit_rows",
                "rows": [],
            }
        ],
    }

    assert runtime._site_constraint_coordinates_required(goal) is True


def test_binding_constraint_repair_recovers_omitted_sources(monkeypatch: Any) -> None:
    class DummyResult:
        final_output = {
            "status": "updated",
            "structured_goal_patch": {
                "rbp_binding": {
                    "rbp": "PUM1",
                    "assay": "eCLIP",
                    "aso_overlap_requirement": "any_overlap",
                },
                "mirna_binding": {
                    "mirnas": ["miR-221", "miR-222"],
                    "source": "TargetScan",
                    "target_level": "mature_mrna",
                    "operator": "union",
                    "aso_overlap_requirement": "any_overlap",
                },
                "region_constraints": [
                    {
                        "source_type": "rbp_binding",
                        "rbp_binding": {
                            "rbp": "PUM1",
                            "assay": "eCLIP",
                            "aso_overlap_requirement": "any_overlap",
                        },
                        "operator": "union",
                    },
                    {
                        "source_type": "mirna_binding",
                        "mirna_binding": {
                            "mirnas": ["miR-221", "miR-222"],
                            "source": "TargetScan",
                            "target_level": "mature_mrna",
                            "operator": "union",
                            "aso_overlap_requirement": "any_overlap",
                        },
                        "operator": "union",
                    },
                ],
            },
        }

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setattr(runtime.Runner, "run", fake_run)
    intake = {
        "status": "ready",
        "structured_goal": {
            "gene_symbol": "CDKN1B",
            "species": "human",
            "target_context": "general human baseline",
            "intent": "binding",
            "target_region": "utr3",
            "run_splice_router": False,
            "run_splice_eval": False,
        },
    }

    repaired = asyncio.run(
        runtime._repair_binding_site_constraints_if_needed(
            "Design blockers for PUM1 and miR-221/miR-222 sites.",
            intake,
        )
    )
    goal = repaired["structured_goal"]

    assert goal["rbp_binding"]["rbp"] == "PUM1"
    assert goal["mirna_binding"]["mirnas"] == ["miR-221", "miR-222"]
    assert {entry["source_type"] for entry in runtime._goal_region_constraints(goal)} == {
        "rbp_binding",
        "mirna_binding",
    }


def test_quoted_physical_site_scope_becomes_executable_cds_interval(monkeypatch: Any) -> None:
    class DummyResult:
        final_output = {"feature_region": "cds", "require_complete_site_in_one_feature": True,
                        "evidence_quote": "Every complete target site must be within a coding exon."}

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setattr(runtime.Runner, "run", fake_run)
    intake = {"status": "ready", "structured_goal": {
        "gene_symbol": "EXAMPLE", "species": "human", "intent": "knockdown",
        "design_programs": [{"program_id": "coding_panel", "modality": "gapmer",
                             "intent": "knockdown", "objective": "target a coding exon"}],
    }}
    repaired = asyncio.run(runtime._recover_explicit_physical_site_scope(
        "Every complete target site must be within a coding exon.", intake))
    goal = repaired["structured_goal"]
    assert goal["target_region"] == "cds"
    assert goal["design_programs"][0]["target_region"] == "cds"
    assert goal["intent"] == "knockdown"


def test_unquoted_or_conflicting_site_scope_cannot_override_design(monkeypatch: Any) -> None:
    class DummyResult:
        final_output = {"feature_region": "cds", "require_complete_site_in_one_feature": True,
                        "evidence_quote": "inside a coding exon"}

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setattr(runtime.Runner, "run", fake_run)
    intake = {"status": "ready", "structured_goal": {
        "intent": "knockdown", "design_programs": [{"program_id": "panel", "modality": "gapmer"}],
    }}
    assert asyncio.run(runtime._recover_explicit_physical_site_scope(
        "Use a protein-coding transcript and design four gapmers.", intake)) == intake
    intake["structured_goal"]["target_region"] = "utr3"
    assert asyncio.run(runtime._recover_explicit_physical_site_scope(
        "Place each ASO inside a coding exon.", intake)) == intake


def test_quoted_site_scope_applies_to_repeated_panel_programs_only(monkeypatch: Any) -> None:
    class DummyResult:
        final_output = {"feature_region": "cds", "evidence_quote": "Every site lies wholly in one coding exon."}

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setattr(runtime.Runner, "run", fake_run)
    base = {"modality": "gapmer", "intent": "knockdown", "target_type": "exon",
            "objective": "Knock down one target with a coding-exon panel"}
    intake = {"status": "ready", "structured_goal": {"intent": "knockdown", "design_programs": [
        {"program_id": "molecule_a", **base}, {"program_id": "molecule_b", **base},
    ]}}
    prompt = "Every site lies wholly in one coding exon."
    repaired = asyncio.run(runtime._recover_explicit_physical_site_scope(prompt, intake))
    assert repaired["structured_goal"]["target_region"] == "cds"
    assert [p["target_region"] for p in repaired["structured_goal"]["design_programs"]] == ["cds", "cds"]
    intake["structured_goal"]["design_programs"][1]["objective"] = "Target an independently specified UTR site"
    assert asyncio.run(runtime._recover_explicit_physical_site_scope(prompt, intake)) == intake


def test_goal_intake_clarification_review_drops_source_resolvable_question(
    monkeypatch: Any,
) -> None:
    class DummyResult:
        final_output = json.dumps(
            {
                "blocking_question_indexes": [],
                "additional_blocking_questions": [],
                "reason": "Resolvable by a declared capability.",
            }
        )

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setattr(runtime.Runner, "run", fake_run)
    intake = {
        "status": "need_clarification",
        "questions": ["Please provide public reference coordinates."],
        "structured_goal": {
            "gene_symbol": "GENE",
            "species": "human",
            "target_context": "relevant cells",
            "intent": "binding",
            "region_constraints": [
                {
                    "source_type": "rbp_binding",
                    "rbp_binding": {"rbp": "RBP", "assay": "eCLIP"},
                }
            ],
        },
    }

    reviewed = asyncio.run(
        runtime._review_goal_intake_clarifications_if_needed(
            "Design against a supported public binding source.", intake
        )
    )

    assert reviewed["status"] == "ready"
    assert reviewed["questions"] == []


def test_goal_intake_clarification_review_fails_closed_on_malformed_output(
    monkeypatch: Any,
) -> None:
    class DummyResult:
        final_output = json.dumps({"reason": "No index decision."})

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setattr(runtime.Runner, "run", fake_run)
    intake = {
        "status": "need_clarification",
        "questions": ["Supply a private experimental choice."],
        "structured_goal": {"gene_symbol": "GENE"},
    }

    reviewed = asyncio.run(
        runtime._review_goal_intake_clarifications_if_needed("request", intake)
    )

    assert reviewed == intake


def test_goal_intake_clarification_review_adds_omitted_blocker_to_ready_goal(
    monkeypatch: Any,
) -> None:
    class DummyResult:
        final_output = json.dumps(
            {
                "blocking_question_indexes": [],
                "additional_blocking_questions": [
                    {
                        "missing_field": "explicit_target_coordinates",
                        "question": "Provide a resolvable identifier or exact coordinates for the custom target feature.",
                        "reason": "The user-defined feature is required for execution and cannot be resolved publicly.",
                    }
                ],
                "reason": "One user-specific execution input is missing.",
            }
        )

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setattr(runtime.Runner, "run", fake_run)
    intake = {
        "status": "ready",
        "questions": [],
        "structured_goal": {
            "gene_symbol": "GENE",
            "target_context": "user-defined cells",
        },
    }

    reviewed = asyncio.run(
        runtime._review_goal_intake_clarifications_if_needed("request", intake)
    )

    assert reviewed["status"] == "need_clarification"
    assert reviewed["questions"] == [
        "Provide a resolvable identifier or exact coordinates for the custom target feature."
    ]


def test_goal_intake_clarification_review_combines_retained_and_omitted_blockers(
    monkeypatch: Any,
) -> None:
    class DummyResult:
        final_output = json.dumps(
            {
                "blocking_question_indexes": [0],
                "additional_blocking_questions": [
                    {
                        "missing_field": "private_input_artifact",
                        "question": "Provide the private input artifact required by the request.",
                        "reason": "The artifact is user-owned and absent.",
                    }
                ],
                "reason": "Two independent user inputs are required.",
            }
        )

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setattr(runtime.Runner, "run", fake_run)
    intake = {
        "status": "need_clarification",
        "questions": ["Provide the experimental target context."],
        "structured_goal": {"gene_symbol": "GENE"},
    }

    reviewed = asyncio.run(
        runtime._review_goal_intake_clarifications_if_needed("request", intake)
    )

    assert reviewed["questions"] == [
        "Provide the experimental target context.",
        "Provide the private input artifact required by the request.",
    ]


def test_request_envelope_shortcut_uses_binding_constraint_repair(monkeypatch: Any) -> None:
    class DummyResult:
        final_output = json.dumps(
            {
                "status": "updated",
                "structured_goal_patch": {
                    "intent": "binding",
                    "target_region": "utr3",
                    "run_splice_router": False,
                    "run_splice_eval": False,
                    "aso_overlap_requirement": "any_overlap",
                    "rbp_binding": {
                        "rbp": "RBFOX2",
                        "assay": "eCLIP",
                        "aso_overlap_requirement": "any_overlap",
                    },
                    "mirna_binding": {
                        "mirnas": ["miR-132/212"],
                        "source": "TargetScan",
                        "target_level": "mature_mrna",
                        "operator": "union",
                        "aso_overlap_requirement": "any_overlap",
                    },
                    "region_constraints": [
                        {
                            "source_type": "rbp_binding",
                            "operator": "union",
                            "aso_overlap_requirement": "any_overlap",
                            "rbp_binding": {"rbp": "RBFOX2", "assay": "eCLIP"},
                        },
                        {
                            "source_type": "mirna_binding",
                            "operator": "union",
                            "aso_overlap_requirement": "any_overlap",
                            "mirna_binding": {
                                "mirnas": ["miR-132/212"],
                                "source": "TargetScan",
                                "target_level": "mature_mrna",
                            },
                        },
                    ],
                },
            }
        )

    async def fake_run(*_args: Any, **_kwargs: Any) -> DummyResult:
        return DummyResult()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(runtime.Runner, "run", fake_run)

    request_text = (
        "For MAPT in human cortical neurons, design steric blocker ASOs that cover either "
        "RBFOX2 eCLIP binding peaks in the transcript or TargetScan miR-132/212 family sites "
        "in the 3' UTR. Return at least 3 ASOs per site."
    )
    envelope = pipeline_tools._build_request_envelope(
        request_text=request_text,
        request_id="goal_test_MAPT",
        structured_goal={},
        raw_envelope={
            "tasks": [
                {
                    "task_id": "design_mapt_binding",
                    "task_type": "design_new_asos",
                    "context": {
                        "structured_goal": {
                            "gene_symbol": "MAPT",
                            "species": "human",
                            "target_context": "human cortical neurons",
                            "intent": "binding",
                            "target_region": "utr3",
                            "design_programs": [
                                {
                                    "program_id": "mapt_binding",
                                    "modality": "steric",
                                    "intent": "binding",
                                }
                            ],
                        }
                    },
                }
            ]
        },
    )
    goal = envelope["tasks"][0]["context"]["structured_goal"]

    assert goal["candidate_limit_per_site"] == 3
    assert goal["min_asos_per_site"] == 3
    assert goal["rbp_binding"]["rbp"] == "RBFOX2"
    assert goal["mirna_binding"]["source"] == "TargetScan"
    assert {entry["source_type"] for entry in runtime._goal_region_constraints(goal)} == {
        "rbp_binding",
        "mirna_binding",
    }


def test_program_region_constraints_do_not_inherit_global_constraints_by_default() -> None:
    goal = {
        "materialized_region_constraints": {
            "global": [
                {
                    "source_type": "mirna_binding",
                    "constraint_id": "mirna_global",
                    "materialized_rows": [["chr1", 10, 20, "mirna", 0, "+"]],
                }
            ],
            "by_program": {
                "pum1_program": [
                    {
                        "source_type": "rbp_binding",
                        "constraint_id": "pum1_program_constraint",
                        "materialized_rows": [["chr1", 30, 40, "pum1", 0, "+"]],
                    }
                ]
            },
        }
    }

    scoped = runtime._constraint_entries_for_scope(goal, {"program_id": "pum1_program"})
    assert [entry["source_type"] for entry in scoped] == ["rbp_binding"]

    inherited = runtime._constraint_entries_for_scope(
        goal,
        {"program_id": "pum1_program", "inherit_global_region_constraints": True},
    )
    assert [entry["source_type"] for entry in inherited] == ["mirna_binding", "rbp_binding"]


def test_splice_site_route_program_does_not_inherit_global_exon_focus_constraint() -> None:
    goal = {
        "materialized_region_constraints": {
            "global": [
                {
                    "source_type": "exon",
                    "constraint_id": "exon10_focus",
                    "aso_overlap_requirement": "full_containment",
                    "materialized_rows": [["chr17", 100, 200, "exon10", 0, "+"]],
                }
            ],
            "by_program": {},
        }
    }
    base_rows = [["chr17", 0, 401, "donor_window", 0, "+"]]
    program = {
        "program_id": "mapt_donor_route",
        "target_region": "splice",
        "site_kind": "donor",
        "run_splice_eval": True,
        "run_splice_router": True,
    }

    rows, meta = runtime._apply_region_constraints_to_rows(
        goal,
        base_rows,
        output_space="genomic",
        program=program,
    )

    assert rows == base_rows
    assert meta["constraint_count"] == 0

    inherited_rows, inherited_meta = runtime._apply_region_constraints_to_rows(
        goal,
        base_rows,
        output_space="genomic",
        program={**program, "inherit_global_region_constraints": True},
    )

    assert inherited_rows == [["chr17", 100, 200, "exon10|donor_window", 0, "+"]]
    assert inherited_meta["constraint_count"] == 1
