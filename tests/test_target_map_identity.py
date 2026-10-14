import csv
import json

import run_goal_intake as runtime


def candidate(chrom="chr8", start=300, end=320, strand="+"):
    return {"sequence": "ACGT", "chrom": chrom, "start": start, "end": end,
            "strand": strand, "score": 0.5}


def test_equal_region_labels_do_not_merge_distant_locations_or_recommendations():
    goal = {"intent": "binding", "steric_bed": {"rows": [
        ["chr8", 100, 150, "requested_region", 0, "+"],
        ["chr8", 300, 350, "requested_region", 0, "+"]]}}
    targets = runtime._build_targets(goal, {}, [candidate()])
    assert len(targets) == 2 and len({t["label"] for t in targets}) == 2
    assert targets[0]["region_start"] == 100 and targets[0]["recommended_asos"] == []
    assert targets[1]["region_start"] == 300 and targets[1]["recommended_asos"] == [candidate()]
    assert targets[1]["source_label"] == "requested_region"


def test_repeated_identical_geometry_can_still_merge_modalities_without_duplicates():
    bed = {"rows": [["chr8", 300, 350, "requested_region", 0, "+"]]}
    goal = {"intent": "binding", "steric_bed": bed, "gapmer_bed": bed}
    targets = runtime._build_targets(goal, {}, [candidate()], [candidate()])
    assert len(targets) == 1
    assert set(targets[0]["output_types"]) == {"steric_tiling", "gapmer_tiling"}
    assert len(targets[0]["recommended_asos"]) == 1


def test_membership_respects_chromosome_strand_and_half_open_boundaries():
    candidates = [candidate(), candidate(chrom="chr9"), candidate(strand="-"),
                  candidate(start=280, end=300), candidate(start=350, end=370)]
    target = runtime._build_bed_targets_from_rows(
        [["chr8", 300, 350, "region", 0, "+"]], candidates, "binding", "steric_tiling")[0]
    assert target["recommended_asos"] == [candidate()]
    assert target["coordinate_system"] == "bed0_half_open"


def test_spliced_alignment_does_not_gain_membership_from_its_bounding_span():
    aso = candidate(start=100, end=400)
    aso["binding_intervals_bed0_half_open"] = [("chr8", 100, 110), ("chr8", 390, 400)]
    aso["source_intervals_bed0_half_open"] = [("chr8", 100, 400)]
    targets = runtime._build_bed_targets_from_rows(
        [["chr8", 200, 250, "intron", 0, "+"], ["chr8", 390, 410, "exon", 0, "+"]],
        [aso], "binding", "steric_tiling")
    assert not targets[0]["recommended_asos"]
    assert targets[1]["recommended_asos"] == [aso]


def test_csv_loading_preserves_reference_space_and_original_block_geometry(tmp_path):
    path = tmp_path / "ranked.csv"
    row = {"Sequence": "ACGT", "Chrom": "neutral_transcript", "Start": 1, "End": 4,
           "Strand": "-", "source": "junction_gapmer",
           "Genomic_chrom": "chr8", "Genomic_start": 101, "Genomic_end": 392,
           "Genomic_segments": json.dumps([
               {"chrom": "chr8", "start": 391, "end": 392},
               {"chrom": "chr8", "start": 101, "end": 102}])}
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    aso = runtime._load_ranked_asos_csv(path)[0]
    assert aso["chrom"] == "neutral_transcript"
    assert sorted(aso["binding_intervals_bed0_half_open"]) == [("chr8", 100, 102), ("chr8", 390, 392)]
    assert aso["source_intervals_bed0_half_open"] == [("neutral_transcript", 0, 4)]
    assert aso["sequence"] == row["Sequence"]
