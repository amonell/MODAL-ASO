import io
import itertools
import json
import random
from types import SimpleNamespace

import pytest

from aso_agents.mismatch_screening import MismatchQueryIndex, empty_hits, add_hit, sam_substitution_count


@pytest.mark.parametrize("budget", range(4))
def test_seeded_search_matches_independent_brute_force_for_all_mismatch_budgets(budget):
    rng = random.Random(920)
    queries = {"".join(rng.choices("ACGT", k=length)) for length in (2, 8, 19, 20, 23) for _ in range(10)}
    index = MismatchQueryIndex(queries, budget)
    targets = set(queries)
    for query in queries:
        for n in range(1, 5):
            mutated = list(query)
            for pos in rng.sample(range(len(query)), min(n, len(query))):
                mutated[pos] = next(base for base in "ACGT" if base != query[pos])
            targets.add("".join(mutated))
    for target in targets:
        expected = {(q, sum(a != b for a, b in zip(q, target))) for q in queries
                    if len(q) == len(target) and sum(a != b for a, b in zip(q, target)) <= budget}
        assert set(index.matches(target)) == expected
    assert index.matches("N" * 20) == []


@pytest.mark.parametrize("budget", [-1, 4, 1.5, True])
def test_index_rejects_unsupported_budgets(budget):
    with pytest.raises(ValueError):
        MismatchQueryIndex(["ACGT"], budget)


@pytest.mark.parametrize("tags", [[], ["NM:i:x"], ["NM:i:-1"], ["NM:i:2"], ["NM:i:0", "NM:i:1"]])
def test_sam_without_unambiguous_measured_distance_is_unknown(tags):
    assert sam_substitution_count([""] * 11 + tags, 1) is None


@pytest.mark.parametrize("strand,genome_sequence", [("+", "TACCNNNNGGTT"), ("-", "AACCNNNNGGTA")])
def test_junction_mismatch_bins_use_actual_transcript_orientation(monkeypatch, strand, genome_sequence):
    import run_goal_intake as native
    query = "TACCGGTA"  # one substitution from transcript-oriented TACC + GGTT
    monkeypatch.setattr(native, "_load_genome_fasta", lambda _: {"chrT": genome_sequence})
    monkeypatch.setattr(native, "_load_gtf_transcript_exons", lambda _: {"transcripts": {
        "off.1": {"gene_id": "OTHER.2", "chrom": "chrT", "strand": strand, "exons": [(0, 4), (8, 12)]},
        "target.1": {"gene_id": "INTENDED.1", "chrom": "chrT", "strand": strand, "exons": [(0, 4), (8, 12)]}}})
    hits, scope = native._collect_junction_off_targets([query], "neutral", "INTENDED", max_mismatches=1)
    assert hits[query]["gene_ids"] == {"OTHER"}
    assert hits[query]["by_mismatch"]["0"]["gene_ids"] == set()
    assert hits[query]["by_mismatch"]["1"]["gene_ids"] == {"OTHER"}
    assert scope["mismatch_budget"] == 1
    exact, _ = native._collect_junction_off_targets([query], "neutral", "INTENDED")
    assert exact[query]["gene_ids"] == set()


def run_screen(tmp_path, monkeypatch, fault=None):
    import run_goal_intake as native
    from aso_agents import pipeline_tools as pipeline
    from aso_agents.evaluation_evidence import exact_gene_count_scope
    query = "ACGTTACCGAGTACGATGCA"
    path = tmp_path / "tiles.json"
    path.write_text(json.dumps({"rows": [{"provided_row_index": 0, "Sequence": query,
        "Off_target_gene_ids": ["STALE_DIFFERENT_SCOPE"]}]}))
    exons = [(10, 30, "A.1", "A", "GeneA", "+"), (40, 60, "A.1", "A", "GeneA", "+"),
             (70, 90, "B.1", "B", "GeneB", "+"), (100, 120, "I.1", "INTENDED", "Intended", "+")]
    monkeypatch.setattr(native, "_load_gtf_exon_bins", lambda _: {"chrom_bins": {"chrT": {0: exons}}, "bin_size": 1000})
    config = SimpleNamespace(bowtie_prefix=tmp_path / "index", gtf_path=tmp_path / "ref.gtf",
                             genome_fasta=tmp_path / "genome.fa", ensembl_assembly="neutral")
    monkeypatch.setattr(native, "get_species_config", lambda _: config)
    observed = []
    def process(command, **kwargs):
        observed.append(command)
        if fault == "capped":
            (tmp_path / "tiles.offtarget_overmapped.fasta").write_text(">q1\n" + query + "\n")
        lines = []
        for pos, mm in [(11, 0), (41, 1), (71, 1), (101, 0)]:
            tags = [] if fault == "missing_nm" and pos == 71 else [f"NM:i:{mm}"]
            lines.append("\t".join(["q1", "0", "chrT", str(pos), "255", "20M", "*", "0", "0", query, "*"] + tags))
        return SimpleNamespace(stdout=io.StringIO("\n".join(lines)), stderr=io.StringIO(""), wait=lambda: 0)
    monkeypatch.setattr(native.subprocess, "Popen", process)
    def junction(sequences, species, excluded, max_mismatches=0):
        assert sequences == [query] and max_mismatches == 1 and excluded == "INTENDED"
        hits = empty_hits(1)
        add_hit(hits, "J.2", "J", "GeneJ", 1)
        return {query: hits}, {"mismatch_budget": 1, **({"error": "missing_reference"} if fault == "junction_error" else {})}
    monkeypatch.setattr(native, "_collect_junction_off_targets", junction)
    goal = {"species": "human", "gene_id": "INTENDED.7",
            "constraints": {"off_target_evaluation": {"max_mismatches": 1}}}
    summary = native._annotate_gapmer_off_targets(str(path), goal, goal["constraints"])
    assert observed[0][observed[0].index("-v") + 1] == "1"
    assert summary["mismatch_budget"] == 1
    scope = exact_gene_count_scope(config, goal["gene_id"], summary)
    pipeline._enrich_exact_off_target_bins_for_tiles(str(path), scope)
    scored = json.loads(path.read_text())["rows"]
    supplied = [{"name": "same_input"}]
    pipeline._merge_scored_tiles_into_provided_asos(supplied, scored)
    exported = pipeline._provided_aso_export_rows({"evaluation_summary": {"provided_asos": supplied}})[0]
    return scored[0], supplied[0], exported


def test_requested_mismatch_search_keeps_bins_union_and_delivered_evidence(tmp_path, monkeypatch):
    scored, merged, exported = run_screen(tmp_path, monkeypatch)
    for row in (scored, merged, exported):
        assert row["Off_target_exact_count"] == 1
        assert row["Off_target_1mm_count"] == 3
        assert row["Off_target_total_upto_mismatch_count"] == 3  # A counted once in union
        assert row["Off_target_mismatch_budget"] == 1
        assert row["Off_target_gene_ids_by_mismatch"]["0mm"] == ["A"]
        assert row["Off_target_gene_ids_by_mismatch"]["1mm"] == ["A", "B", "J"]
        assert row["Off_target_gene_ids_by_mismatch"]["2mm"] is None
        assert row["off_target_count_scope"]["mismatch_budget"] == 1
    assert scored["Off_target_2mm_count"] is None
    assert "STALE_DIFFERENT_SCOPE" not in scored["Off_target_gene_ids"]


@pytest.mark.parametrize("fault", ["capped", "missing_nm", "junction_error"])
def test_partial_mismatch_search_keeps_observations_without_claiming_zero(tmp_path, monkeypatch, fault):
    scored, merged, exported = run_screen(tmp_path, monkeypatch, fault)
    for row in (scored, merged, exported):
        assert row["off_target_count_status"] == "incomplete"
        assert row["Off_target_exact_count"] is None
        assert row["Off_target_1mm_count"] is None
        assert row["Off_target_total_upto_mismatch_count"] is None
    assert scored["off_target_observed_gene_count"] >= 2


def test_default_exact_only_budget_remains_explicit_and_higher_bins_unknown():
    from aso_agents.tiling_agent import _resolve_num_mismatches
    assert _resolve_num_mismatches({}, default=0) == 0
    assert _resolve_num_mismatches({"num_mismatches": 2}, default=0) == 2
    assert _resolve_num_mismatches({"evaluation_targets": {"off_target": {
        "match_policy": {"max_mismatches": 3}}}}, default=0) == 3
    with pytest.raises(ValueError):
        _resolve_num_mismatches({"num_mismatches": 4}, default=0)

