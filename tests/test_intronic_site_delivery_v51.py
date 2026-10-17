"""A resolved physical intron is reported on either transcript strand."""

import csv

from aso_agents.design_delivery import _resolved_intronic_site, write_design_panel


def _goal(strand):
    exons = ([{"index": 1, "start": 100, "end": 120},
              {"index": 2, "start": 201, "end": 220}]
             if strand == "+" else
             [{"index": 1, "start": 201, "end": 220},
              {"index": 2, "start": 100, "end": 120}])
    return {"transcript_id": "TX.2", "isoform_transcripts": {
        "TX": {"transcript_id": "TX", "metadata": {"versioned_id": "TX.2"},
               "chrom": "chr1", "strand": strand, "exons_tx": exons}}}


def _row(start, end, chrom="chr1"):
    return {"source_modality": "steric", "synthesized_antisense_sequence": "ACGTACGTACGTACGTACGT",
            "target_binding_sequence": "ACGTACGTACGTACGTACGT",
            "binding_intervals_bed0_half_open": [[chrom, start - 1, end]]}


def test_intronic_placement_and_rationale_both_strands(tmp_path):
    for strand, start, end, donor_start, donor_end in (("+", 121, 140, 1, 20),
                                                        ("-", 181, 200, 1, 20)):
        goal, row = _goal(strand), _row(start, end)
        assert _resolved_intronic_site(goal, row) == {
            "transcript_accession": "TX.2", "intron_ordinal": 1,
            "donor_distance_start": donor_start, "donor_distance_end": donor_end,
            "acceptor_distance_start": 61, "acceptor_distance_end": 80}
        exported = write_design_panel({"rows": [row]}, goal, tmp_path / strand)
        with open(exported["design_panel_csv"], newline="") as stream:
            panel = next(csv.DictReader(stream))
        assert panel["intron_ordinal"] == "1"
        assert panel["intron_donor_distance_start_1based"] == "1"
        assert panel["intron_donor_distance_end_1based"] == "20"
        assert panel["intron_acceptor_distance_start_1based"] == "61"
        assert panel["intron_acceptor_distance_end_1based"] == "80"
        assert "TX.2 intron 1, 1–20 intronic bases downstream" in panel["physical_element_explanation"]
        assert "remains unmeasured" in panel["recommendation_note"]
        assert panel["physical_element_explanation"] in open(exported["design_panel_md"]).read()


def test_intronic_placement_fails_closed_when_not_attested():
    goal = _goal("-")
    assert _resolved_intronic_site(goal, _row(115, 130)) is None  # boundary overlap
    assert _resolved_intronic_site(goal, _row(181, 200, "chr2")) is None
    assert _resolved_intronic_site(goal, {**_row(181, 200),
        "binding_intervals_bed0_half_open": [["chr1", 180, 190], ["chr1", 190, 200]]}) is None
    goal["isoform_transcripts"]["TX"]["exons_tx"][1]["index"] = 3
    assert _resolved_intronic_site(goal, _row(181, 200)) is None


def test_minus_strand_donor_proximal_interval_uses_transcript_orientation():
    goal = {"transcript_id": "TX", "isoform_transcripts": {"TX": {
        "transcript_id": "TX", "chrom": "chrX", "strand": "-",
        "exons_tx": [{"index": 45, "start": 31968339, "end": 31968514},
                     {"index": 46, "start": 31932080, "end": 31932227}]}}}
    assert _resolved_intronic_site(goal, _row(31968314, 31968338, "chrX")) == {
        "transcript_accession": "TX", "intron_ordinal": 45,
        "donor_distance_start": 1, "donor_distance_end": 25,
        "acceptor_distance_start": 36087, "acceptor_distance_end": 36111}


def test_panel_binds_candidate_specific_route_scores_to_scored_site(tmp_path):
    row = {**_row(121, 140),
           'predicted_response_site': {'chrom': 'chr1', 'position': '120',
                                       'site_kind': 'donor'},
           'source_annotations': {
               'primary_label': 'donor@120', 'primary_direction': 'increase',
               'selected_site_chrom': 'chr1', 'selected_site_kind': 'donor',
               'selected_site_position': '120', 'primary_delta_deletion': '0.0094',
               'primary_delta_replacement': '0.0026'}}
    exported = write_design_panel({'rows': [row]}, _goal('+'), tmp_path / 'bound')
    with open(exported['design_panel_csv'], newline='') as stream:
        panel = next(csv.DictReader(stream))
    explanation = panel['physical_element_explanation']
    assert '0.0094' in explanation and '0.0026' in explanation
    assert 'scored donor at chr1:120' in explanation
    assert 'requested increase' in explanation
    assert 'not model ASO occupancy' in explanation
    assert explanation in open(exported['design_panel_md']).read()

    mismatched = {**row, 'source_annotations': {
        **row['source_annotations'], 'primary_label': 'acceptor@201'}}
    exported = write_design_panel({'rows': [mismatched]}, _goal('+'), tmp_path / 'unbound')
    with open(exported['design_panel_csv'], newline='') as stream:
        panel = next(csv.DictReader(stream))
    assert '0.0094' not in panel['physical_element_explanation']

    tiny = {**row, 'source_annotations': {
        **row['source_annotations'], 'primary_delta_deletion': '0.00000003'}}
    exported = write_design_panel({'rows': [tiny]}, _goal('+'), tmp_path / 'tiny')
    with open(exported['design_panel_csv'], newline='') as stream:
        panel = next(csv.DictReader(stream))
    assert '+3e-08' in panel['physical_element_explanation']
