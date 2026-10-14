import csv

from aso_agents.design_delivery import write_design_panel


def _row(**fields):
    return {'sequence': 'ACGTACGTACGTACGTACGT',
            'target_binding_sequence': 'ACGTACGTACGTACGTACGT',
            'synthesized_antisense_sequence': 'ACGTACGTACGTACGTACGT',
            'binding_intervals_bed0_half_open': [['chr1', 490, 510]],
            **fields}


def test_panel_exports_measured_primary_screen_and_preserves_unknown_bins(tmp_path):
    paths = write_design_panel({'rows': [_row(off_target_search_status='complete',
        Off_target_mismatch_budget=1, Off_target_exact_count=0,
        Off_target_1mm_count=4, off_target_exact_count_unit='loci')]}, {}, tmp_path)
    with open(paths['design_panel_csv'], newline='') as handle:
        record = next(csv.DictReader(handle))
    assert record['off_target_search_status'] == 'complete'
    assert record['off_target_mismatch_budget'] == '1'
    assert record['off_target_exact_count'] == '0'
    assert record['off_target_1mm_count'] == '4'
    assert record['off_target_2mm_count'] == ''
    markdown = (tmp_path / 'design_panel.md').read_text()
    assert 'exact=0 loci' in markdown
    assert '1mm=4' in markdown
    assert 'unsearched or unavailable, not zero' in markdown


def test_panel_does_not_invent_screen_counts_for_unscreened_row(tmp_path):
    paths = write_design_panel({'rows': [_row()]}, {}, tmp_path)
    with open(paths['design_panel_csv'], newline='') as handle:
        record = next(csv.DictReader(handle))
    assert record['off_target_exact_count'] == ''
    assert 'unknown' in (tmp_path / 'design_panel.md').read_text()


def test_panel_derives_physical_metrics_and_versioned_exon_site(tmp_path):
    transcript = {"transcript_id": "ENST_GENERIC.2", "chrom": "chr7", "strand": 1,
                  "exons_tx": [{"index": 3, "start": 100, "end": 149,
                                "seq_region_name": "chr7", "cdna_start": 51,
                                "coding_start_genomic": 110, "coding_end_genomic": 145}]}
    goal = {"transcript_id": "ENST_GENERIC.2", "isoform_transcripts": {"ENST_GENERIC": transcript},
            "chemistry": "5-10-5 MOE/DNA full-PS gapmer"}
    row = _row(chrom="chr7", start=120, end=139, binding_intervals_bed0_half_open=[],
               synthesized_antisense_sequence="AAAACCCCGGGGTTTTACGT")
    path = write_design_panel({"rows": [row]}, goal, tmp_path)["design_panel_csv"]
    with open(path, newline="") as handle:
        record = next(csv.DictReader(handle))
    assert record["transcript_accession"] == "ENST_GENERIC.2"
    assert (record["transcript_cdna_start"], record["transcript_cdna_end"]) == ("71", "90")
    assert record["exon_ordinal"] == "3"
    assert record["complete_coding_exon_site"] == "True"
    assert record["gc_fraction"] == "0.5"
    assert record["max_homopolymer_run_nt"] == "4"
    assert record["chemistry_description"] == "5-10-5 MOE/DNA full-PS gapmer"


def test_panel_maps_minus_strand_cdna_without_genomic_orientation_confusion(tmp_path):
    transcript = {"transcript_id": "ENST_OTHER.1", "chrom": "chr2", "strand": -1,
                  "exons_tx": [{"index": 5, "start": 200, "end": 249,
                                "seq_region_name": "chr2", "cdna_start": 101,
                                "coding_start_genomic": 205, "coding_end_genomic": 245}]}
    goal = {"transcript_id": "ENST_OTHER.1", "isoform_transcripts": {"ENST_OTHER": transcript}}
    row = _row(chrom="chr2", start=220, end=239, strand="-",
               binding_intervals_bed0_half_open=[])
    path = write_design_panel({"rows": [row]}, goal, tmp_path)["design_panel_csv"]
    with open(path, newline="") as handle:
        record = next(csv.DictReader(handle))
    assert (record["transcript_cdna_start"], record["transcript_cdna_end"]) == ("111", "130")
    assert record["exon_ordinal"] == "5"
    assert record["target_genomic_strand"] == "-"


def test_panel_uses_physical_bed0_interval_and_explicit_versioned_transcript_lock(tmp_path):
    goal = {
        "transcript_lock": {"kind": "explicit_id", "value": "ENST_EXAMPLE.9"},
        "isoform_transcripts": {"ENST_EXAMPLE.9": {
            "transcript_id": "ENST_EXAMPLE.9", "chrom": "chr12", "strand": "+",
            "exons_tx": [{"index": 3, "start": 1001, "end": 1200,
                          "seq_region_name": "12", "cdna_start": 201}],
        }},
    }
    row = _row(chrom="chr12", start="1049", end="1069",
               binding_intervals_bed0_half_open=[["chr12", 1049, 1069]])
    path = write_design_panel({"rows": [row]}, goal, tmp_path)["design_panel_csv"]
    with open(path, newline="") as handle:
        record = next(csv.DictReader(handle))
    assert (record["genomic_start_1based"], record["genomic_end_1based"]) == ("1050", "1069")
    assert (record["transcript_accession"], record["transcript_cdna_start"],
            record["transcript_cdna_end"]) == ("ENST_EXAMPLE.9", "250", "269")


def test_panel_does_not_label_a_spliced_multi_block_span_as_one_genomic_interval(tmp_path):
    row = _row(chrom="chr12", start="1049", end="2020",
               binding_intervals_bed0_half_open=[["chr12", 1049, 1060], ["chr12", 2011, 2020]])
    path = write_design_panel({"rows": [row]}, {}, tmp_path)["design_panel_csv"]
    with open(path, newline="") as handle:
        record = next(csv.DictReader(handle))
    assert record["genomic_start_1based"] == record["genomic_end_1based"] == ""
    assert record["physical_binding_intervals_bed0"] == '[["chr12", 1049, 1060], ["chr12", 2011, 2020]]'


def test_panel_keeps_all_structured_chemistry_components(tmp_path):
    goal = {"chemistry": {"recommendation": {"CHEMISTRY": "5-10-5", "wing_chemistry": "2MOE",
                                             "gap_chemistry": "DNA", "backbone": "PS", "platform": "gapmer"}}}
    path = write_design_panel({"rows": [_row()]}, goal, tmp_path)["design_panel_csv"]
    with open(path, newline="") as handle:
        record = next(csv.DictReader(handle))
    assert record["chemistry_description"] == (
        "architecture=5-10-5; wing_sugar=2MOE; gap_sugar=DNA; backbone=PS; platform=gapmer")
