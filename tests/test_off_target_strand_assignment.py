import io
import json
from types import SimpleNamespace

import pytest

from aso_agents.evaluation_evidence import exact_gene_count_scope


def _screen(tmp_path, monkeypatch, alignments, exons, budget=0, previous=None):
    import run_goal_intake as native
    from aso_agents import pipeline_tools as pipeline
    query = "ACGTTACCGAGTACGATGCA"
    path = tmp_path / "tiles.json"
    path.write_text(json.dumps({"rows": [{"Sequence": query, **(previous or {})}]}))
    config = SimpleNamespace(bowtie_prefix=tmp_path / "index", gtf_path=tmp_path / "ref.gtf",
                             genome_fasta=tmp_path / "genome.fa", ensembl_assembly="neutral")
    monkeypatch.setattr(native, "get_species_config", lambda _: config)
    monkeypatch.setattr(native, "_load_gtf_exon_bins", lambda _: {
        "chrom_bins": {"chrT": {0: exons}}, "bin_size": 1000})
    lines = ["\t".join(["q1", str(flag), "chrT", str(pos), "255", "20M", "*", "0", "0", query, "*", f"NM:i:{mm}"])
             for flag, pos, mm in alignments]
    monkeypatch.setattr(native.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(
        stdout=io.StringIO("\n".join(lines)), stderr=io.StringIO(""), wait=lambda: 0))
    monkeypatch.setattr(native, "_collect_junction_off_targets", lambda *a, **kw: ({}, {}))
    goal = {"species": "human", "gene_id": "INTENDED.8", "num_mismatches": budget}
    summary = native._annotate_gapmer_off_targets(str(path), goal, {})
    scope = exact_gene_count_scope(config, goal["gene_id"], summary)
    pipeline._enrich_exact_off_target_bins_for_tiles(str(path), scope)
    return json.loads(path.read_text())["rows"][0], summary, scope


@pytest.mark.parametrize("budget,distance", [(0, 0), (1, 1)])
@pytest.mark.parametrize("strand,flag", [("+", 0), ("-", 16), ("+", 256), ("-", 272)])
def test_genomic_assignment_matches_actual_mature_rna(tmp_path, monkeypatch, budget, distance, strand, flag):
    # The same locus produces different RNA on opposite strands. Compare the
    # expected count to strings, independently of the mapper's gene assignment.
    from Bio.Seq import Seq
    query = "ACGTTACCGAGTACGATGCA"
    target = ("T" + query[1:]) if distance else query
    genome = target if strand == "+" else str(Seq(target).reverse_complement())
    transcripts = {"PLUS": genome, "MINUS": str(Seq(genome).reverse_complement())}
    expected = sorted(gene for gene, seq in transcripts.items()
                      if sum(a != b for a, b in zip(query, seq)) <= budget)
    exons = [(10, 30, "p", "PLUS", "Plus", "+"), (10, 30, "m", "MINUS", "Minus", "-"),
             (10, 30, "i", "INTENDED", "Intended", strand)]
    row, summary, scope = _screen(tmp_path, monkeypatch, [(flag, 11, distance)], exons, budget)
    assert len(expected) == 1
    assert row["Off_target_gene_ids"] == expected
    assert row["off_target_gene_count"] == 1
    assert row["Off_target_gene_ids_by_mismatch"][f"{distance}mm"] == expected
    assert "SAM alignment strand must match" in scope["genomic_strand_policy"]
    assert summary["search_provenance"]["genomic_strand_policy"] == scope["genomic_strand_policy"]


@pytest.mark.parametrize("budget", [0, 1])
@pytest.mark.parametrize("fault", ["unknown_exon", "invalid_flag", "negative_flag"])
def test_unresolved_strand_evidence_is_incomplete_not_a_zero(tmp_path, monkeypatch, budget, fault):
    flag = "unknown" if fault == "invalid_flag" else -1 if fault == "negative_flag" else 0
    strand = "." if fault == "unknown_exon" else "+"
    row, summary, _ = _screen(tmp_path, monkeypatch, [(flag, 11, 0)],
        [(10, 30, "u", "UNKNOWN", "Unknown", strand)], budget)
    assert summary["unclassified_query_count"] == 1
    assert row["off_target_unclassified_alignments"] is True
    assert row["off_target_count_status"] == "incomplete"
    assert row["Off_target_exact_count"] is None
    assert row["off_target_gene_count"] is None
    assert row["off_target_observed_gene_count"] == 0
    assert row["Off_target_gene_ids"] == []


@pytest.mark.parametrize("budget", [0, 1])
def test_current_search_replaces_stale_identities_including_empty_aliases(tmp_path, monkeypatch, budget):
    old = {name: ["STALE"] for name in ("Off_target_gene_ids", "off_target_gene_ids", "Off_target_genes",
                                      "off_target_genes", "Off_target_transcript_ids", "off_target_transcript_ids")}
    row, _, _ = _screen(tmp_path, monkeypatch, [(0, 11, 0)],
        [(10, 30, "opposite", "OPPOSITE", "Opposite", "-")], budget, old)
    assert row["off_target_count_status"] == "complete_within_declared_scope"
    assert row["off_target_gene_count"] == 0
    assert row["Off_target_exact_count"] == 0
    assert row["Off_target_0mm_gene_ids"] == []
    for field in old:
        assert row[field] == []


@pytest.mark.parametrize("strand", [".", None])
def test_unknown_junction_strand_cannot_supply_a_transcript_match(monkeypatch, strand):
    import run_goal_intake as native
    query = "TACCGGTT"
    monkeypatch.setattr(native, "_load_genome_fasta", lambda _: {"chrT": "TACCNNNNGGTT"})
    monkeypatch.setattr(native, "_load_gtf_transcript_exons", lambda _: {"transcripts": {
        "unknown": {"gene_id": "OTHER", "chrom": "chrT", "strand": strand, "exons": [(0, 4), (8, 12)]},
        "intended": {"gene_id": "INTENDED", "chrom": "chrT", "strand": strand, "exons": [(0, 4), (8, 12)]}}})
    hits, scope = native._collect_junction_off_targets([query], "human", "INTENDED")
    assert hits[query]["gene_ids"] == set()
    assert scope["skipped_unresolved_strand"] == 1  # excluded gene does not add uncertainty


def test_unknown_junction_scope_is_exported_as_incomplete(tmp_path):
    from aso_agents import pipeline_tools as pipeline
    path = tmp_path / "tiles.json"
    path.write_text(json.dumps({"rows": [{"Sequence": "AACCGGTT", "off_target_gene_count": 0}]}))
    pipeline._enrich_exact_off_target_bins_for_tiles(str(path), {
        "junction_search_summary": {"skipped_unresolved_strand": 1}})
    row = json.loads(path.read_text())["rows"][0]
    assert row["off_target_count_status"] == "incomplete"
    assert row["Off_target_exact_count"] is None
