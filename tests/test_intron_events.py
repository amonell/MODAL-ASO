from copy import deepcopy

import pytest

from aso_agents.intron_events import resolve_intron_event, bind_intron_events, program_intron_sites


def inputs(strand="+"):
    catalog = {"ref":{"transcript_id":"ENST00000000001.2", "chrom":"chr7", "strand":strand,
        "exons_tx":[{"start":101,"end":150},{"start":301,"end":350},{"start":501,"end":550}]}}
    program = {"program_id":"retention", "modality":"steric", "intent":"splice_modulation",
        "splice_event":{"event_type":"intron_retention","intron_number":1,"reference_transcript_id":"ENST00000000001.2"}}
    return program,catalog


@pytest.mark.parametrize("strand,bounds,sites",[("+",(151,300),[("donor",150),("acceptor",301)]),
                                               ("-",(351,500),[("donor",501),("acceptor",350)])])
def test_transcript_oriented_flanking_sites(strand,bounds,sites):
    program,catalog = inputs(strand)
    event = resolve_intron_event(program,catalog)
    assert (event["intron_start_1based"],event["intron_end_1based"]) == bounds
    assert [(site["site"],site["position"]) for site in event["sites"]] == sites
    assert {site["desired_delta"] for site in event["sites"]} == {"decrease"}


def test_only_requested_flanking_sites_change_and_noncoding_sites_are_materialized():
    program,catalog = inputs()
    plan = [{"chrom":"7","strand":"+","site":kind,"position":pos,"delta_direction":"context"}
            for kind,pos in [("acceptor",101),("donor",150),("acceptor",301),("donor",350)]]
    original = deepcopy(plan)
    bound,_ = bind_intron_events(plan,[],{"design_programs":[program]},catalog)
    assert [row["delta_direction"] for row in bound] == ["context","decrease","decrease","context"]
    assert plan == original
    assert len(program_intron_sites(program,catalog,bound)) == 2
    noncoding,_ = bind_intron_events([],[],{"design_programs":[program]},catalog)
    assert {(row["site"],row["position"]) for row in noncoding} == {("donor",150),("acceptor",301)}
    assert all(row["source"] == "verified_transcript_intron_boundaries" for row in noncoding)


@pytest.mark.parametrize("fault",["reference","ordinal","exon_effect","modality","conflicting_structure"])
def test_invalid_intron_identity_is_never_reinterpreted_as_an_exon(fault):
    program,catalog = inputs()
    if fault == "reference": program["splice_event"]["reference_transcript_id"] = "ENST00000000001.3"
    if fault == "ordinal": program["splice_event"]["intron_number"] = 3
    if fault == "exon_effect": program["desired_effect"] = "inclusion"
    if fault == "modality": program["modality"] = "gapmer"
    if fault == "conflicting_structure":
        catalog["duplicate"] = deepcopy(catalog["ref"])
        catalog["duplicate"]["exons_tx"][0]["end"] = 200
    with pytest.raises(ValueError):
        resolve_intron_event(program,catalog)


def test_noncoding_intron_sites_retain_shared_transcript_membership():
    program,catalog = inputs()
    catalog["other"] = deepcopy(catalog["ref"])
    catalog["other"]["transcript_id"] = "ENST00000000002.1"
    bound,_ = bind_intron_events([],[],{"design_programs":[program]},catalog)
    assert all(set(row["present_in"]) == {"ref","other"} for row in bound)
    assert all(row["uniqueness"] == "shared" and row["covers_all_isoforms"] for row in bound)
