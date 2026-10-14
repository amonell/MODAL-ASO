"""Diverse selection uses the ranked row's actual physical target blocks."""
import json

from aso_agents.target_accessibility import select_panel_rows
from aso_agents.pipeline_tools import _apply_generated_candidate_limit


def test_ranked_genomic_fields_choose_separated_windows_first():
    rows = [
        {"Synthesis": "A" * 20, "IDT": "chem", "Genomic_chrom": "chr1",
         "Genomic_start": 100, "Genomic_end": 119},
        {"Synthesis": "C" * 20, "IDT": "chem", "Genomic_chrom": "chr1",
         "Genomic_start": 105, "Genomic_end": 124},
        {"Synthesis": "G" * 20, "IDT": "chem", "Genomic_chrom": "chr1",
         "Genomic_start": 300, "Genomic_end": 319},
    ]
    selected = select_panel_rows(rows, 2, diversify=True)
    assert selected == [rows[0], rows[2]]


def test_serialized_numeric_coordinates_still_diversify_physical_windows():
    rows = [
        {"Synthesis": "A" * 25, "Chrom": "chr1", "Start": "100", "End": "125"},
        {"Synthesis": "C" * 25, "Chrom": "chr1", "Start": "102", "End": "127"},
        {"Synthesis": "G" * 25, "Chrom": "chr1", "Start": "300", "End": "325"},
        {"Synthesis": "T" * 25, "Chrom": "chr1", "Start": "500", "End": "525"},
    ]
    assert select_panel_rows(rows, 3, diversify=True) == [rows[0], rows[2], rows[3]]


def test_junction_blocks_do_not_claim_intronic_span_as_covered():
    rows = [
        {"Synthesis": "A" * 20, "IDT": "chem", "Genomic_segments": json.dumps([
            {"chrom": "chr1", "start": 100, "end": 109},
            {"chrom": "chr1", "start": 300, "end": 309}])},
        {"Synthesis": "C" * 20, "IDT": "chem", "Genomic_chrom": "chr1",
         "Genomic_start": 200, "Genomic_end": 219},
    ]
    assert select_panel_rows(rows, 2, diversify=True) == rows


def test_generated_panel_keeps_ranked_distinct_physical_windows_when_requested():
    rows = [
        {"Synthesis": "A" * 25, "chemistry_label": "steric-pmo",
         "binding_intervals_bed0_half_open": [["chrX", 100, 125]]},
        {"Synthesis": "C" * 25, "chemistry_label": "steric-pmo",
         "binding_intervals_bed0_half_open": [["chrX", 101, 126]]},
        {"Synthesis": "G" * 25, "chemistry_label": "steric-pmo",
         "binding_intervals_bed0_half_open": [["chrX", 300, 325]]},
        {"Synthesis": "T" * 25, "chemistry_label": "steric-pmo",
         "binding_intervals_bed0_half_open": [["chrX", 500, 525]]},
    ]
    selected, receipt = _apply_generated_candidate_limit(
        rows, candidate_limit=3, per_site=False, diversify=True)
    assert selected == [rows[0], rows[2], rows[3]]
    assert receipt["selection_scope"] == "global"
    assert receipt["selected_count"] == 3
    ranked, _ = _apply_generated_candidate_limit(rows, candidate_limit=3, per_site=False)
    assert ranked == rows[:3]


def test_diverse_range_stops_before_an_avoidable_overlapping_alternative():
    rows = [
        {"Synthesis": base * 20, "IDT": "chem", "Genomic_chrom": "chr1",
         "Genomic_start": start, "Genomic_end": start + 19}
        for base, start in zip("ACGTN", (100, 200, 300, 400, 500))
    ]
    rows.append({"Synthesis": "AG" * 10, "IDT": "chem", "Genomic_chrom": "chr1",
                 "Genomic_start": 103, "Genomic_end": 122})

    selected = select_panel_rows(rows, 6, diversify=True, minimum=4)
    assert selected == rows[:5]
    selected_by_delivery, receipt = _apply_generated_candidate_limit(
        rows, candidate_limit=6, per_site=False, diversify=True, candidate_minimum=4)
    assert selected_by_delivery == rows[:5]
    assert receipt["selected_count"] == 5
    assert receipt["method"] == "ranked_nonoverlapping_targets_then_minimum_fill"
    assert select_panel_rows(rows, 6, diversify=True) == rows


def test_diverse_panel_may_fill_overlap_only_to_meet_explicit_minimum():
    rows = [
        {"Synthesis": base * 20, "Genomic_chrom": "chr1",
         "Genomic_start": start, "Genomic_end": start + 19}
        for base, start in zip("ACGTN", (100, 103, 200, 203, 300))
    ]

    assert select_panel_rows(rows, 5, diversify=True, minimum=4) == [
        rows[0], rows[2], rows[4], rows[1]]
    assert select_panel_rows(rows, 5, diversify=True, minimum=5) == [
        rows[0], rows[2], rows[4], rows[1], rows[3]]
