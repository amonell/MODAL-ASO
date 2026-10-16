"""A locked anchor must not erase additional explicitly requested references."""
from copy import deepcopy

import pytest
import run_goal_intake as native
from aso_agents.declared_reference_catalog import complete_target_references


def structure(identity, strand=1):
    base, _, version = identity.rpartition(".")
    return {"id": base, "versioned_id": identity, "version": int(version),
            "gene_id": "ENSG00000000001.4", "species": "homo_sapiens",
            "assembly_name": "GRCh38", "seq_region_name": "chr7", "strand": strand,
            "exons_tx": [{"index": 1, "start": 101, "end": 150},
                         {"index": 2, "start": 301, "end": 400}]}


def entry(identity):
    record = structure(identity)
    return {"transcript_id": identity,
            "metadata": native.resolved_transcript_metadata(record, {}),
            "exons_tx": record["exons_tx"], "chrom": "chr7", "strand": "+"}


def goal():
    return {"gene_symbol": "TEST", "gene_id": "ENSG00000000001",
            "species": "human", "target_region": "translation_start",
            "transcript_policy": "user_locked",
            "transcript_lock": {"kind": "explicit_id", "value": "ENST00000000001.2"},
            "isoform_transcripts": {"anchor": entry("ENST00000000001.2")},
            "design_programs": [{"program_id": "shared", "modality": "steric",
                                "intent": "binding", "target_region": "translation_start",
                                "transcript_ids": ["ENST00000000001.2", "ENST00000000002.3"]}]}


@pytest.mark.parametrize("feature", ["translation_start", "utr3", "utr5", "utrs", "cds"])
def test_native_feature_resolution_completes_cohort_without_replacing_anchor(monkeypatch, feature):
    g = goal()
    g["target_region"] = g["design_programs"][0]["target_region"] = feature
    original = deepcopy(g)
    seen = []
    def fetch(reference):
        seen.append(reference)
        return structure(reference)
    monkeypatch.setattr(native, "_fetch_transcript_structure", fetch)
    final = {}
    native._ensure_isoform_transcripts_for_target_region(g, final)
    assert seen == ["ENST00000000002.3"]
    assert g["isoform_transcripts"]["anchor"] == original["isoform_transcripts"]["anchor"]
    assert g["transcript_lock"] == original["transcript_lock"]
    assert set(g["isoform_transcripts"]) == {"anchor", "ENST00000000002.3"}
    assert final["isoform_transcripts"] == g["isoform_transcripts"]
    assert native._select_design_program_isoform_label(
        g["isoform_transcripts"], g["design_programs"][0], "anchor", g) == "anchor"
    native._ensure_isoform_transcripts_for_target_region(g, final)
    assert seen == ["ENST00000000002.3"]


@pytest.mark.parametrize("fault", ["version", "gene", "species", "chromosome", "exons", "strand"])
def test_failed_declared_reference_preserves_existing_catalog_atomically(monkeypatch, fault):
    g = goal()
    previous = deepcopy(g)
    final = {"retained": True}
    record = structure("ENST00000000002.3")
    if fault == "version":
        record.update(versioned_id="ENST00000000002.4", version=4)
    elif fault == "gene":
        record["gene_id"] = "ENSG00000000009.4"
    elif fault == "species":
        record["species"] = "mus_musculus"
    elif fault == "chromosome":
        record["seq_region_name"] = ""
    elif fault == "exons":
        record["exons_tx"][0]["end"] = 99
    elif fault == "strand":
        record["strand"] = 0
    monkeypatch.setattr(native, "_fetch_transcript_structure", lambda reference: record)
    with pytest.raises(ValueError, match="declared_target_reference"):
        complete_target_references(native, g, final)
    assert g == previous and final == {"retained": True}


def test_declared_foreign_reference_is_left_to_its_species_resolver(monkeypatch):
    g = goal()
    foreign = "ENSMUST00000000003.5"
    g["design_programs"][0]["transcript_ids"] = ["ENST00000000001.2", foreign]
    g["cross_species_targets"] = {"mouse": {"transcript_ids": [foreign]}}
    monkeypatch.setattr(native, "_fetch_transcript_structure",
                        lambda reference: pytest.fail("Foreign reference sent to local resolver"))
    assert complete_target_references(native, g)
    assert set(g["isoform_transcripts"]) == {"anchor"}
    assert g["declared_target_reference_resolution"][1]["status"] == "deferred_to_declared_species"


def test_unrecognized_prefix_is_not_used_as_species_evidence(monkeypatch):
    g = goal()
    foreign = "ENSMUST00000000003.5"
    g["design_programs"][0]["transcript_ids"] = [foreign]
    seen = []
    def fetch(reference):
        seen.append(reference)
        return {**structure(reference), "species": "mus_musculus"}
    monkeypatch.setattr(native, "_fetch_transcript_structure", fetch)
    with pytest.raises(ValueError, match="species_conflict"):
        complete_target_references(native, g)
    assert seen == [foreign]


def test_conflicting_cached_identity_is_not_overwritten(monkeypatch):
    g = goal()
    g["isoform_transcripts"]["ENST00000000002.3"] = entry("ENST00000000002.4")
    previous = deepcopy(g)
    monkeypatch.setattr(native, "_fetch_transcript_structure", structure)
    with pytest.raises(ValueError, match="catalog_conflict"):
        complete_target_references(native, g)
    assert g == previous


def test_programs_keep_distinct_explicit_references_and_deduplicate_fetches(monkeypatch):
    g = goal()
    g["design_programs"] += [
        {"program_id": "another", "transcript_id": "ENST00000000003.1"},
        {"program_id": "repeat", "transcript_id": "ENST00000000002.3"},
    ]
    seen = []
    def fetch(reference):
        seen.append(reference)
        return structure(reference, -1)
    monkeypatch.setattr(native, "_fetch_transcript_structure", fetch)
    assert complete_target_references(native, g)
    assert seen == ["ENST00000000002.3", "ENST00000000003.1"]
    assert g["isoform_transcripts"]["ENST00000000003.1"]["strand"] == "-"
    assert g["declared_target_reference_resolution"][1]["program_ids"] == ["shared", "repeat"]


def test_no_declaration_does_not_change_existing_default_behavior(monkeypatch):
    g = goal()
    g["design_programs"] = [{"program_id": "ordinary", "objective": "mentions a transcript"}]
    previous = deepcopy(g)
    monkeypatch.setattr(native, "_fetch_transcript_structure", lambda reference: pytest.fail(reference))
    assert not complete_target_references(native, g)
    assert g == previous


@pytest.mark.parametrize("strand", ["+", "-", "1", "-1"])
def test_resolved_strand_notation_is_normalized_for_native_splice_sites(monkeypatch, strand):
    g = goal()
    monkeypatch.setattr(native, "_fetch_transcript_structure", lambda reference: structure(reference, strand))
    assert complete_target_references(native, g)
    assert g["isoform_transcripts"]["ENST00000000002.3"]["strand"] == ("-" if strand in {"-", "-1"} else "+")


@pytest.mark.parametrize("field,value", [("species", "mouse"), ("gene_id", "ENSG00000000009")])
def test_existing_reference_cannot_override_declared_species_or_gene(monkeypatch, field, value):
    g = goal()
    g["isoform_transcripts"]["anchor"]["metadata"][field] = value
    previous = deepcopy(g)
    monkeypatch.setattr(native, "_fetch_transcript_structure", lambda reference: pytest.fail(reference))
    with pytest.raises(ValueError, match="conflict"):
        complete_target_references(native, g)
    assert g == previous


def test_malformed_cohort_is_not_iterated_as_individual_characters(monkeypatch):
    g = goal()
    g["design_programs"][0]["transcript_ids"] = "ENST00000000002.3"
    previous = deepcopy(g)
    monkeypatch.setattr(native, "_fetch_transcript_structure", lambda reference: pytest.fail(reference))
    with pytest.raises(ValueError, match="cohort_must_be_list"):
        complete_target_references(native, g)
    assert g == previous


def test_duplicate_catalog_alias_is_not_arbitrarily_selected(monkeypatch):
    g = goal()
    g["isoform_transcripts"]["duplicate"] = deepcopy(g["isoform_transcripts"]["anchor"])
    previous = deepcopy(g)
    monkeypatch.setattr(native, "_fetch_transcript_structure", structure)
    with pytest.raises(ValueError, match="catalog_ambiguous"):
        complete_target_references(native, g)
    assert g == previous


def test_legacy_symbol_in_gene_id_matches_verified_reference_symbol(monkeypatch):
    g = goal()
    g['gene_id'] = 'TEST'
    g['isoform_transcripts']['anchor']['metadata']['gene_symbol'] = 'TEST'
    monkeypatch.setattr(native, '_fetch_transcript_structure',
                        lambda reference: {**structure(reference), 'gene_symbol': 'TEST'})
    assert complete_target_references(native, g)
    assert g['isoform_transcripts']['ENST00000000002.3']['metadata']['gene_id'] == 'ENSG00000000001.4'


@pytest.mark.parametrize('expected,recorded', [('OTHER', 'TEST'), ('TEST', None), ('ENSG00000000009', 'TEST')])
def test_proposed_label_never_resolves_a_conflicting_reference(monkeypatch, expected, recorded):
    g = goal()
    g['gene_id'] = expected
    g['isoform_transcripts']['anchor']['metadata']['gene_symbol'] = recorded
    before = deepcopy(g)
    monkeypatch.setattr(native, '_fetch_transcript_structure',
                        lambda reference: {**structure(reference), 'gene_symbol': recorded})
    with pytest.raises(ValueError, match='gene_conflict'):
        complete_target_references(native, g)
    assert g == before
