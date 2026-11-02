import gzip
import itertools
import json
from types import SimpleNamespace

import pytest

from aso_agents.mismatch_screening import MismatchQueryIndex, reference_incomplete_reasons
from aso_agents.reference_gap_recovery import recover_exons


@pytest.mark.parametrize("budget", range(4))
def test_ambiguous_reference_matches_independent_expansion_oracle(budget):
    queries = ["".join(bases) for bases in itertools.product("ACGT", repeat=4)]
    index = MismatchQueryIndex(queries, budget)
    alphabet = {**{base: base for base in "ACGT"}, "N": "ACGT", "R": "AG", "Y": "CT"}
    for target in ("ANRT", "NNNN", "GYCA", "ACGT", "RRYY"):
        completions = list(itertools.product(*(alphabet[base] for base in target)))
        distances = {q: min(sum(a != b for a, b in zip(q, t)) for t in completions) for q in queries}
        assert dict(index.possible_matches(target)) == {q: n for q, n in distances.items() if n <= budget}


def test_recovery_requires_lengths_and_all_known_genomic_bases():
    assert recover_exons("ACGTGGTT", ["NNGTGG", "TT"], [6, 2]) == ["ACGTGG", "TT"]
    for sequence in ("ACGTGTT", "ACGTGCTT", "ACGTGGTA"):
        with pytest.raises(ValueError):
            recover_exons(sequence, ["NNGTGG", "TT"], [6, 2])


def reference(monkeypatch, genome, exons, *, strand="+", sequence=None, failure=None):
    import run_goal_intake as native
    from aso_agents import reference_gap_recovery
    monkeypatch.setattr(native, "_load_genome_fasta", lambda _: {"chrT": genome})
    monkeypatch.setattr(native, "_load_gtf_transcript_exons", lambda _: {"transcripts": {
        "tx": {"gene_id": "OTHER", "chrom": "chrT", "strand": strand, "exons": exons}}})
    def recover(species, entries):
        assert set(entries) == {"tx"}
        if failure:
            raise OSError(failure)
        return {"tx": sequence}, {"verified": True, "sha256": "test-reference"}
    monkeypatch.setattr(reference_gap_recovery, "load_transcript_sequences", recover)
    return native


@pytest.mark.parametrize("budget", [0, 2])
def test_recovered_exon_interior_and_junction_are_both_searched(monkeypatch, budget):
    native = reference(monkeypatch, "AAAANNNNAAAAZZZZTTTT", [(0, 12), (16, 20)],
                       sequence="AAAACGTGAAAATTTT")
    hits, summary = native._collect_junction_off_targets(["CGTG", "AATT"], "human", None, budget)
    for query in hits:
        assert hits[query]["gene_ids"] == {"OTHER"}
        assert not reference_incomplete_reasons(summary, query, hits[query]["reference_coverage"])
    assert summary["recovered_exon_windows"] > 0
    assert summary["recovered_transcripts"] == 1
    assert not summary.get("skipped_ambiguous_windows")


def test_recovered_minus_strand_exons_follow_transcript_order(monkeypatch):
    native = reference(monkeypatch, "AACCZZZZNNTA", [(0, 4), (8, 12)], strand="-", sequence="TACCGGTT")
    hits, summary = native._collect_junction_off_targets(["CCGG", "TACC"], "human", None, 0)
    assert all(hit["gene_ids"] == {"OTHER"} for hit in hits.values())
    assert summary["recovered_transcripts"] == 1


@pytest.mark.parametrize("budget", [0, 2])
def test_unresolved_bases_only_block_queries_they_could_match(monkeypatch, budget):
    native = reference(monkeypatch, "AAAN", [(0, 4)], failure="reference download unavailable")
    hits, summary = native._collect_junction_off_targets(["AAAC", "CCCC"], "human", None, budget)
    assert summary["skipped_ambiguous_windows"] == 1
    assert reference_incomplete_reasons(summary, "AAAC", hits["AAAC"]["reference_coverage"])
    assert not reference_incomplete_reasons(summary, "CCCC", hits["CCCC"]["reference_coverage"])
    assert not hits["AAAC"]["gene_ids"]  # A possible completion is never an observed hit.
    assert reference_incomplete_reasons(summary, "AAAC", hits["CCCC"]["reference_coverage"])
    assert reference_incomplete_reasons(summary, "CCCC")  # Old receipt has no per-query proof.


def test_inconsistent_transcript_cannot_cover_reference_gap(monkeypatch):
    native = reference(monkeypatch, "AAAN", [(0, 4)], sequence="CCCT")
    hits, summary = native._collect_junction_off_targets(["AAAC"], "human", None)
    assert "known genomic bases" in summary["transcript_recovery_failures"][0]["reason"]
    assert reference_incomplete_reasons(summary, "AAAC", hits["AAAC"]["reference_coverage"])


def test_same_gene_exclusion_does_not_download_unused_fallback(monkeypatch):
    native = reference(monkeypatch, "NNNN", [(0, 4)], failure="must not download")
    hits, summary = native._collect_junction_off_targets(["AAAA"], "human", "OTHER")
    assert "transcripts_with_reference_gaps" not in summary
    assert hits["AAAA"]["gene_ids"] == set()


def test_fallback_installs_pinned_reference_and_requires_versioned_identity(tmp_path, monkeypatch):
    from aso_agents import reference_gap_recovery as fallback
    monkeypatch.setattr(fallback, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(fallback, "get_species_config", lambda _: SimpleNamespace(gtf_path="gencode.v42.annotation.gtf"))
    monkeypatch.setattr(fallback, "reference_checksums", lambda: {"gencode.v42.transcripts.fa.gz": {"sha256": "pinned"}})
    monkeypatch.setattr(fallback, "read_receipt", lambda _: {"sha256": "pinned"})
    calls = []
    def download(url, destination, **kwargs):
        calls.append((url, kwargs))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, "wt") as stream:
            stream.write(">TX.1|GENE.2|\nACGT\n>TX.2|GENE.2|\nTGCA\n")
    monkeypatch.setattr(fallback, "download", download)
    sequences, receipt = fallback.load_transcript_sequences("human", {
        "TX": {"transcript_id_version": "TX.2", "gene_id_version": "GENE.2"}})
    assert sequences == {"TX": "TGCA"} and receipt["verified"]
    assert calls[0][1]["expected_sha256"] == "pinned"
    assert "release_42/gencode.v42.transcripts.fa.gz" in calls[0][0]
    with pytest.raises(ValueError, match="gene identities"):
        fallback.load_transcript_sequences("human", {"TX": {
            "transcript_id_version": "TX.2", "gene_id_version": "GENE.3"}})


def test_enrichment_preserves_query_specific_coverage(tmp_path):
    from aso_agents.pipeline_tools import _enrich_exact_off_target_bins_for_tiles
    path = tmp_path / "rows.json"
    coverage = {"version": 1, "query": "CCCC", "checked": True, "mismatch_budget": 2,
                "unresolved_possible_matches": 0}
    path.write_text(json.dumps({"rows": [{"Sequence": "CCCC", "off_target_reference_coverage": coverage,
        "off_target_observed_gene_ids_by_mismatch": {f"{n}mm": [] for n in range(3)},
        "off_target_observed_genes_by_mismatch": {f"{n}mm": [] for n in range(3)}}]}))
    _enrich_exact_off_target_bins_for_tiles(str(path), {"mismatch_budget": 2,
        "junction_search_summary": {"mismatch_budget": 2, "skipped_ambiguous_windows": 1}})
    row = json.loads(path.read_text())["rows"][0]
    assert row["off_target_search_status"] == "complete"
    assert row["Off_target_2mm_count"] == 0


def test_missing_summary_does_not_clear_a_failed_screen(tmp_path):
    from aso_agents.pipeline_tools import _enrich_exact_off_target_bins_for_tiles
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": [{"Sequence": "CCCC", "off_target_count_status": "incomplete",
        "off_target_observed_gene_ids_by_mismatch": {"0mm": []},
        "off_target_observed_genes_by_mismatch": {"0mm": []}}]}))
    _enrich_exact_off_target_bins_for_tiles(str(path))
    row = json.loads(path.read_text())["rows"][0]
    assert row["off_target_search_status"] == "incomplete"
    assert row["Off_target_exact_count"] is None


def test_successful_rescreen_replaces_previous_failure_reasons():
    from aso_agents.pipeline_tools import _merge_scored_tiles_into_provided_asos
    rows = [{"off_target_incomplete_reasons": ["reference_bases_unresolved"]}]
    _merge_scored_tiles_into_provided_asos(rows, [{"provided_row_index": 0,
        "off_target_count_status": "complete_within_declared_scope", "off_target_incomplete_reasons": []}])
    assert rows[0]["off_target_incomplete_reasons"] == []
