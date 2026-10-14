"""Physical panel coverage and species requirements at the delivery boundary."""
from types import SimpleNamespace

from aso_agents import pipeline_tools as tools


def test_global_panel_reserves_annotated_exon_and_both_splice_boundaries(monkeypatch):
    monkeypatch.setattr(tools, "_goal_runtime_module", lambda: SimpleNamespace(
        _tile_row_overlap_intervals=lambda row, output_space: [
            (row["chrom"], int(row["start"]), int(row["end"]))
        ]))
    goal = {
        "design_programs": [
            {"target_type": kind, "exon_label": 2} for kind in
            ("exon", "splice_acceptor", "splice_donor")
        ],
        "isoform_transcripts": {"tx": {
            "transcript_id": "tx", "chrom": "chr1", "strand": "+",
            "exons_tx": [
                {"index": 1, "start": 100, "end": 199},
                {"index": 2, "start": 300, "end": 399},
                {"index": 3, "start": 500, "end": 599},
            ],
        }},
    }
    # The first six ranked rows all cross the acceptor. Both other requested
    # physical classes remain available beyond a naive six-row truncation.
    rows = [
        {"chrom": "chr1", "start": str(289 + i), "end": str(314 + i),
         "sequence": "A" * 24 + str(i)}
        for i in range(6)
    ] + [
        {"chrom": "chr1", "start": "330", "end": "355", "sequence": "C" * 25},
        {"chrom": "chr1", "start": "389", "end": "414", "sequence": "G" * 25},
    ]
    selected, summary = tools._select_physical_splice_coverage(rows, goal, 6)
    assert len(selected) == 6
    assert rows[6] in selected and rows[7] in selected
    assert summary["covered_physical_classes"] == [
        "exon:2", "splice_acceptor:2", "splice_donor:2"
    ]
    assert summary["uncovered_physical_classes"] == []


def test_species_targets_are_an_explicit_required_conservation_scope():
    goal = {"species": "human", "species_targets": [
        {"species": "human", "gene_symbol": "GENE"},
        {"species": "mouse", "gene_symbol": "GENE"},
    ]}
    assert tools._requested_species_from_contexts(None, goal) == ["human", "mouse"]
    assert tools._cross_species_required(None, goal)
    goal["cross_species_requirement"] = {"required": False,
                                         "return_nonconserved_candidates": True}
    assert not tools._cross_species_required(None, goal)


def test_exact_reference_placements_report_all_occurrences_without_genomic_inference():
    import run_goal_intake as runtime

    assert runtime._exact_reference_placements("ACGT", {"TX.2": "ACGTTTACGT"}, "transcript_cdna") == [
        {"reference_id": "TX.2", "start_1based": 1, "end_1based": 4,
         "coordinate_scope": "transcript_cdna"},
        {"reference_id": "TX.2", "start_1based": 7, "end_1based": 10,
         "coordinate_scope": "transcript_cdna"},
    ]


def test_panel_exports_reference_positions_with_coordinate_scope(tmp_path):
    import csv
    from aso_agents.design_delivery import write_design_panel

    placement = {"reference_id": "TX.2", "start_1based": 8,
                 "end_1based": 15, "coordinate_scope": "transcript_cdna"}
    row = {"sequence": "AACCGGTT", "target_binding_sequence": "AACCGGTT",
           "synthesized_antisense_sequence": "AACCGGTT", "length": 8,
           "cross_species_matches": {"mouse": ["TX.2"]},
           "cross_species_match_placements": {"mouse": [placement]},
           "off_target_search_provenance_by_species": {"mouse": {
               "assembly": "assembly-A", "genomic_index": "/references/mouse/index",
               "exon_annotation": "/references/mouse/annotation.gtf",
               "junction_search": "complete"}}}
    paths = write_design_panel({"rows": [row]}, {}, tmp_path)
    with open(paths["design_panel_csv"], newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert '"start_1based": 8' in exported[0]["cross_species_match_placements"]
    assert '"genomic_index": "/references/mouse/index"' in exported[0]["off_target_search_provenance_by_species"]
    markdown = (tmp_path / "design_panel.md").read_text()
    assert "| 1 | mouse | TX.2 | 8–15 | transcript_cdna | unresolved / unresolved | unavailable | not_applicable |" in markdown
    assert "| mouse | assembly-A | index | annotation.gtf | complete |" in markdown
