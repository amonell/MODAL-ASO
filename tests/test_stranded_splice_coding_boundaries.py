import pytest

import run_goal_intake as intake


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("coding_start,coding_end", [
    (100, 180), (120, 200), (120, 180), (100, 200),
    (None, 180), (120, None), (None, None),
])
def test_splice_eligibility_uses_physical_coding_boundary(strand, coding_start, coding_end):
    sites = intake._extract_splice_sites({
        "seq_region_name": "8", "strand": strand,
        "exons_tx": [{"index": 2, "start": 100, "end": 200, "contains_coding": True,
                      "coding_start_genomic": coding_start, "coding_end_genomic": coding_end}],
    })
    assert len(sites) == 1
    site = sites[0]
    assert site["acceptor"] == (100 if strand == 1 else 200)
    assert site["donor"] == (200 if strand == 1 else 100)
    for kind in ("acceptor", "donor"):
        coordinate = site[kind]
        assert site[kind + "_plan_ok"] == (
            (coding_start is None or coordinate >= coding_start)
            and (coding_end is None or coordinate <= coding_end)
        )


def test_negative_strand_alternative_donors_survive_and_transcript_ends_do_not():
    transcripts = {}
    for label, donor in (("long", 300), ("short", 330)):
        exons = [
            {"index": 1, "start": 500, "end": 550, "contains_coding": False},
            {"index": 2, "start": donor, "end": 400, "contains_coding": True,
             "coding_start_genomic": donor, "coding_end_genomic": 380},
            {"index": 3, "start": 100, "end": 200, "contains_coding": True,
             "coding_start_genomic": 120, "coding_end_genomic": 200},
        ]
        structure = {"seq_region_name": "8", "strand": -1, "exons_tx": exons}
        transcripts[label] = {
            "transcript_id": label, "chrom": "8", "strand": "-", "exons_tx": exons,
            "sites": intake._extract_splice_sites(structure),
        }
    plan, _ = intake._compute_splice_site_plan([
        {"isoform_label": "long", "desired_abundance": "decrease"},
        {"isoform_label": "short", "desired_abundance": "increase"},
    ], transcripts)
    donors = {item["position"]: item for item in plan if item["site"] == "donor"}
    assert set(donors) == {300, 330}
    assert donors[300]["present_in"] == ["long"]
    assert donors[330]["present_in"] == ["short"]
    assert donors[300]["delta_direction"] == "decrease"
    assert donors[330]["delta_direction"] == "increase"
    assert 400 not in {item["position"] for item in plan}
    assert 100 not in {item["position"] for item in plan}


def test_explicit_noncoding_splice_sites_keep_their_force_contract():
    structure = {"strand": -1, "seq_region_name": "8", "exons_tx": [
        {"index": 2, "start": 100, "end": 200, "contains_coding": False},
    ]}
    assert intake._extract_splice_sites(structure) == []
    structure["exons_tx"][0]["force_splice_sites"] = True
    site = intake._extract_splice_sites(structure)[0]
    assert site["acceptor_plan_ok"] and site["donor_plan_ok"]
