from copy import deepcopy

import pytest
import run_goal_intake as native
from aso_agents.junction_products import boundary_position, materialize_junction_products
from aso_agents import pipeline_tools


REFERENCE = "ENST00000000001.2"


@pytest.fixture(autouse=True)
def ordinal_reference(monkeypatch):
    monkeypatch.setattr(native, "resolve_transcript_exon_nomenclature_sync", lambda **kw: {
        "status": "verified_transcript_ordinal", "resolved_transcript_ordinal": kw["requested_exon_label"]})


def example(strand="+"):
    exons = [{"start": start, "end": end, "contains_coding": True,
              "coding_start_genomic": start, "coding_end_genomic": end}
             for start, end in [(101, 150), (301, 399), (501, 550)]]
    if strand == "-":
        exons.reverse()
    for index, exon in enumerate(exons, 1):
        exon["index"] = index
    structure = {"id": REFERENCE.split(".")[0], "versioned_id": REFERENCE,
                 "version": 2, "seq_region_name": "chr8", "strand": 1 if strand == "+" else -1,
                 "exons_tx": exons, "species": "homo_sapiens", "gene_id": "ENSG00000000001"}
    entry = {"transcript_id": REFERENCE, "chrom": "chr8", "strand": strand,
             "metadata": native.resolved_transcript_metadata(structure, {}),
             "exons_tx": exons, "sites": native._extract_splice_sites(structure)}
    entry["metadata"]["exon_nomenclature_resolutions"] = {str(i): {
        "status": "verified_transcript_ordinal", "resolved_transcript_ordinal": i} for i in range(1, 4)}
    change = {"exon_label": 2, "site_kind": "acceptor", "chrom": "chr8", "strand": strand,
              "coordinate_convention": "one_based_inclusive", "coordinate_role": "splice_dinucleotide",
              "start": 302 if strand == "+" else 397, "end": 303 if strand == "+" else 398}
    return {"gene_symbol": "TEST", "gene_id": "ENSG00000000001", "species": "human",
            "intent": "splice_modulation", "exon_label": 2, "focus_exons_only": True,
            "transcript_policy": "user_locked", "transcript_lock": {"kind": "explicit_id", "value": REFERENCE},
            "isoform_transcripts": {REFERENCE: entry},
            "design_programs": [{"program_id": "product", "modality": "steric", "intent": "splice_modulation",
                                 "isoform_label": "shorter_exon_product", "exon_label": 2,
                                 "desired_delta": "increase", "site_kind": "both"}],
            "splice_isoform_objectives": [
                {"objective_scope": "specific_isoform", "isoform_label": "shorter_exon_product",
                 "reference_accession": REFERENCE, "exon_label": 2, "desired_abundance": "increase",
                 "junction_changes": [change]},
                {"objective_scope": "aggregate_class", "transcript_id": REFERENCE,
                 "aggregate_class": {"basis": "exon_presence", "exon_label": 2, "presence": "excluded"},
                 "desired_abundance": "decrease"}]}


@pytest.mark.parametrize("kind,strand,expected", [
    ("acceptor", "+", 304), ("acceptor", "-", 301),
    ("donor", "+", 301), ("donor", "-", 304)])
@pytest.mark.parametrize("convention", ["one_based_inclusive", "bed0_half_open"])
def test_motif_coordinates_become_exonic_boundary_on_both_strands(kind, strand, expected, convention):
    change = {"site_kind": kind, "strand": strand, "coordinate_convention": convention,
              "coordinate_role": "splice_dinucleotide", "start": 301 if convention == "bed0_half_open" else 302,
              "end": 303}
    assert boundary_position(change) == expected


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("supplied_strand", [True, False])
def test_constructed_product_preserves_parent_and_separate_junctions(strand, supplied_strand):
    goal = example(strand)
    if not supplied_strand:
        goal["splice_isoform_objectives"][0]["junction_changes"][0].pop("strand")
    parent = deepcopy(goal["isoform_transcripts"][REFERENCE])
    final = {}
    assert materialize_junction_products(native, goal, final)
    product = goal["isoform_transcripts"]["shorter_exon_product"]
    assert goal["isoform_transcripts"][REFERENCE] == parent
    assert product["transcript_id"] != REFERENCE
    assert product["exons_tx"][1]["end"] - product["exons_tx"][1]["start"] + 1 == 96
    assert len(product["exons_tx"]) == 3
    assert product["metadata"]["source"] == "junction_defined_product"
    assert product["metadata"]["base_reference_snapshot"] == parent["metadata"]
    assert final["isoform_transcripts"] == goal["isoform_transcripts"]
    # Repeat resolution is stable and never relabels the parent as the product.
    before = deepcopy(goal)
    materialize_junction_products(native, goal)
    assert before == goal
    plan, _ = native._compute_splice_site_plan(goal["splice_isoform_objectives"], goal["isoform_transcripts"], goal)
    positions = {(p["site"], p["position"]): p for p in plan}
    expected = {("acceptor", 301), ("acceptor", 304), ("donor", 399)} if strand == "+" else {
        ("acceptor", 399), ("acceptor", 396), ("donor", 301)}
    assert set(positions) == expected
    alternative = positions[("acceptor", 304 if strand == "+" else 396)]
    assert alternative["delta_direction"] == "increase"
    assert alternative["present_in"] == ["shorter_exon_product"]
    assert all(p["delta_direction"] != "decrease" for p in plan)


@pytest.mark.parametrize("fault,error", [
    ({"chrom": "chr9"}, "orientation_conflict"),
    ({"strand": "-"}, "orientation_conflict"),
    ({"end": 304}, "two_bases"),
    ({"coordinate_role": ""}, "coordinate_role_required"),
    ({"coordinate_convention": ""}, "definition_incomplete"),
    ({"exon_label": 1}, "terminal_boundary"),
    ({"start": 400, "end": 401}, "removes_identified_exon"),
    ({"start": 148, "end": 149}, "crosses_another_exon"),
    ({"start": 299, "end": 300}, "leave_reference_unchanged"),
])
def test_invalid_product_geometry_does_not_publish_partial_catalog(fault, error):
    goal = example()
    goal["splice_isoform_objectives"][0]["junction_changes"][0].update(fault)
    original = deepcopy(goal)
    final = {"untouched": True}
    with pytest.raises(ValueError, match=error):
        materialize_junction_products(native, goal, final)
    assert goal == original
    assert final == {"untouched": True}


def test_aggregate_reference_does_not_become_transcript_abundance_objective():
    goal = example()
    normalized = pipeline_tools._normalized_design_objective_preseed(goal)
    aggregate = normalized["splice_isoform_objectives"][1]
    assert aggregate["transcript_id"] == REFERENCE
    assert "isoform_label" not in aggregate
    assert aggregate["desired_abundance"] == "decrease"
    goal["evaluation_targets"] = [{"target_type": "isoform_cohort", "transcript_id": REFERENCE,
                                   "objective_type": "primary_outcome", "desired_delta": "decrease"}]
    assert pipeline_tools._program_isoform_abundance_directions(goal, {"transcript_id": REFERENCE}) == set()


def test_preserving_a_request_accession_does_not_rename_its_derived_product():
    goal = example()
    goal.pop("transcript_lock")
    assert native._apply_explicit_transcript_lock_from_request_text(goal, "Use " + REFERENCE)
    product = goal["splice_isoform_objectives"][0]
    assert product["reference_accession"] == REFERENCE
    assert "transcript_id" not in product


@pytest.mark.parametrize("text", ["decrease skipping", "retain original coordinate values", "increase inclusion"])
def test_execution_does_not_infer_splice_effect_from_prose(text):
    assert native._splice_effect_from_objective({"objective": text, "notes": text}) == ""
    assert native._splice_effect_from_objective({"desired_effect": "skipping", "notes": text}) == "skipping"
    assert native._default_splice_effect_for_placeholder_check({"design_programs": [{"objective": text}]}) == ""


def test_native_resolver_constructs_product_without_matching_parent_as_outcome(monkeypatch):
    goal = example()
    # A legacy normalized class might already carry its reference as a label.
    # Its typed aggregate scope must still prevent a concrete abundance route.
    goal["splice_isoform_objectives"][1]["isoform_label"] = REFERENCE
    goal["preseeded_isoform_transcripts"] = deepcopy(goal["isoform_transcripts"])
    monkeypatch.setattr(native, "list_transcripts", lambda **kw: {"transcripts": [
        {"id": REFERENCE, "versioned_id": REFERENCE, "is_canonical": True}]})
    monkeypatch.setattr(native, "_generate_splice_strategy_analysis", lambda *a, **kw: {})
    record = goal["isoform_transcripts"][REFERENCE]
    monkeypatch.setattr(native, "_fetch_transcript_structure", lambda *_: {
        **record["metadata"], "exons_tx": record["exons_tx"], "strand": 1})
    monkeypatch.setattr(native, "auto_match_isoform_transcript", lambda *a, **kw: pytest.fail("Product sent to transcript matcher"))
    result = native._resolve_isoform_transcripts(goal, {}, allow_interactive=False)
    assert goal["aggregate_isoform_classes"]
    assert goal["isoform_transcripts"]["shorter_exon_product"]["metadata"]["junction_defined_product"]
    assert {(p["site"], p["position"]) for p in result["splice_site_plan"]} == {
        ("acceptor", 301), ("acceptor", 304), ("donor", 399)}
