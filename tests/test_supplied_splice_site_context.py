from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.splicing_agent import _nearest_splice_site_for_mid


def reference(strand="+"):
    intervals = [(101, 200), (4001, 4100), (4201, 4300), (4401, 4500), (9001, 9100)]
    if strand == "-":
        intervals.reverse()
    return {"chrom": "chr7", "strand": strand, "transcript_id": "TX_NEUTRAL",
            "metadata": {"versioned_id": "TX_NEUTRAL.3"},
            "exons_tx": [{"index": i, "start": start, "end": end}
                         for i, (start, end) in enumerate(intervals, 1)]}


def binding(strand="+", **changes):
    return {"name": "probe", "transcript_id": "TX_NEUTRAL.3", "strand": strand,
            "genomic_span": {"chrom": "7", "start": 4203, "end": 4222}, **changes}


@pytest.mark.parametrize("strand", ["+", "-"])
def test_requested_remote_site_keeps_all_binding_local_junctions(strand):
    ref = reference(strand)
    target = [{"transcript_id": "TX_NEUTRAL.3", "exon_label": 1, "site_kind": "donor"}]
    planned = pipeline._filter_splice_sites_by_targets(
        pipeline._build_transcript_splice_sites_for_eval(ref), target)
    before = deepcopy((ref, planned, target))
    sites, scope = pipeline._provided_splice_site_context(ref, "TX_NEUTRAL", planned, target, [binding(strand)])
    identities = {(s["exon_index"], s["site_kind"]) for s in sites}
    assert identities == {(1, "donor"), *[(exon, kind) for exon in (2, 3, 4)
                                         for kind in ("acceptor", "donor")]}
    assert len(sites) == 7  # All six local boundaries, not just the two nearest.
    assert scope["annotated_internal_boundary_count"] == 8
    assert scope["transcript_id"] == "TX_NEUTRAL.3"
    assert scope["nearby_radius_from_aso_midpoint_bp"] == 2000
    assert (ref, planned, target) == before


@pytest.mark.parametrize("strand", ["+", "-"])
def test_terminal_transcript_ends_are_not_supplemented_as_junctions(strand):
    sites = pipeline._fallback_splice_sites_for_eval(reference(strand), [])
    assert len(sites) == 8
    assert not any(s["exon_index"] == 1 and s["site_kind"] == "acceptor" for s in sites)
    assert not any(s["exon_index"] == 5 and s["site_kind"] == "donor" for s in sites)
    single = reference(strand)
    single["exons_tx"] = [single["exons_tx"][0]]
    assert pipeline._fallback_splice_sites_for_eval(single, []) == []


def test_plans_cannot_import_other_references_and_duplicate_coordinates_collapse():
    ref = reference()
    candidate = pipeline._build_transcript_splice_sites_for_eval(ref)[4]
    plans = [deepcopy(candidate) for _ in range(4)]
    plans[0].pop("transcript_id")
    plans[0]["label"] = plans[0]["canonical_label"] = "local-alias"
    for kind in ("acceptor", "donor"):
        plans[0][kind]["chrom"] = "7"
        plans[1][kind]["chrom"] = "chr8"
        plans[2][kind]["strand"] = "-"
    plans[3]["transcript_id"] = "TX_NEUTRAL.2"
    before = deepcopy(plans)
    sites, scope = pipeline._provided_splice_site_context(ref, "TX_NEUTRAL.3", plans, [], [binding()])
    assert scope["planned_site_count"] == 1
    assert scope["excluded_incompatible_planned_site_count"] == 3
    assert len(sites) == 6
    assert len({(s["site_kind"], s[s["site_kind"]]["position"]) for s in sites}) == 6
    assert plans == before


@pytest.mark.parametrize("change", [
    {"transcript_id": "TX_FOREIGN.3"}, {"transcript_id": "TX_NEUTRAL.2"},
    {"strand": "-"}, {"genomic_span": {"chrom": "chr8", "start": 4203, "end": 4222}},
    {"genomic_span": {}},
])
def test_unmatched_binding_context_does_not_fall_back_to_every_site(change):
    sites = pipeline._fallback_splice_sites_for_eval(reference(), [])
    assert pipeline._select_relevant_splice_sites(
        splice_sites=sites, evaluation_targets=[], entries=[binding(**change)]) == []


def test_catalog_version_conflict_is_unavailable_not_relabelled():
    sites, scope = pipeline._provided_splice_site_context(reference(), "TX_NEUTRAL.2", [], [], [binding()])
    assert sites == []
    assert scope["coverage"] == "unavailable"
    assert not scope["reference_identity_compatible"]


def test_far_binding_keeps_nearest_distance_observations_without_claiming_scoring():
    far = binding(genomic_span={"chrom": "chr7", "start": 20000, "end": 20019})
    sites, scope = pipeline._provided_splice_site_context(reference(), "TX_NEUTRAL.3", [], [], [far])
    assert {(s["site_kind"], s[s["site_kind"]]["position"]) for s in sites} == {
        ("acceptor", 9001), ("donor", 4500)}
    assert "not proof that every site was scored" in scope["interpretation"]


def test_nearest_observation_respects_declared_boundary_kind():
    site = {"site_kind": "donor", "canonical_label": "donor@100",
            "acceptor": {"position": 20000}, "donor": {"position": 100}}
    nearest = _nearest_splice_site_for_mid(19995, [site])
    assert nearest["site_kind"] == "donor"
    assert nearest["position"] == 100 and nearest["distance_bp"] == 19895


def test_mapping_skip_reason_and_search_scope_reach_response_evidence():
    row = {"name": "unmapped", "stage_results": {"mapping": {"status": "fail"}}}
    pipeline._record_provided_splice_skip(row, "no_exact_binding_mapping")
    assert row["stage_results"]["splice_eval"]["status"] == "skipped"
    assert row["stage_results"]["mapping"]["status"] == "fail"
    _, scope = pipeline._provided_splice_site_context(reference(), "TX_NEUTRAL.3", [], [], [binding()])
    scored = {"name": "mapped", "splice_eval_summary": {"delta_deletion": [0.0], "site_search_scope": scope}}
    compact = pipeline._compact_existing_aso_evaluation_for_response({"provided_asos": [row, scored]})
    assert compact["provided_asos"][0]["splice_eval_summary"]["splice_eval_skip_reason"] == "no_exact_binding_mapping"
    assert compact["provided_asos"][1]["splice_eval_summary"]["site_search_scope"] == scope


def test_evaluation_dispatches_local_sites_and_keeps_unmapped_reason(tmp_path, monkeypatch):
    import json
    import random
    from pathlib import Path
    from types import SimpleNamespace

    import run_goal_intake as runtime
    from aso_agents import transcript_reference

    rng = random.Random(817)
    genome = "".join(rng.choices("ACGT", k=12000))
    fasta = tmp_path / "neutral.fa"
    fasta.write_text(">chr7\n" + genome + "\n")
    config = SimpleNamespace(genome_fasta=fasta, ensembl_assembly="neutral-fixture")
    for module in (pipeline, runtime, transcript_reference):
        monkeypatch.setattr(module, "get_species_config", lambda *_: config)
    ref = reference()
    ref["metadata"].update(species="human", assembly_name="neutral-fixture")
    catalog = {"TX_NEUTRAL": ref}
    monkeypatch.setattr(pipeline, "_goal_runtime_module", lambda: runtime)
    monkeypatch.setattr(pipeline, "_candidate_run_dirs_for_task", lambda *_: [])
    monkeypatch.setattr(pipeline, "_reference_goal_context_for_request", lambda *_: {})
    monkeypatch.setattr(pipeline, "_ensure_eval_transcript_catalog", lambda *_: (catalog, catalog, []))
    monkeypatch.setattr(pipeline, "_choose_eval_transcript", lambda *_: ("TX_NEUTRAL", "TX_NEUTRAL", ref))
    monkeypatch.setattr(pipeline, "_score_provided_asos_with_existing_pipeline", lambda **_: {})
    monkeypatch.setattr(pipeline, "_mapping_conservation_summary", lambda **_: {})
    planned = [s for s in pipeline._build_transcript_splice_sites_for_eval(ref)
               if s["exon_index"] == 1 and s["site_kind"] == "donor"]
    monkeypatch.setattr(runtime, "_build_splice_sites_for_eval", lambda *_args, **_kwargs: planned)
    dispatched = []

    def score(**kwargs):
        sites = json.loads(kwargs["splice_sites_json"])["sites"]
        dispatched.extend(sites)
        tiles = json.loads(Path(kwargs["initial_tiles_path"]).read_text())["rows"]
        output = tmp_path / "scored.json"
        # Deliberately zero deltas: this tests evidence transport, not potency.
        output.write_text(json.dumps({"rows": [{"name": row["name"], "delta_inclusion_deletion": 0.0,
                                                "delta_names": [s["canonical_label"] for s in sites],
                                                "delta_deletion": [0.0] * len(sites)} for row in tiles]}))
        return {"scored_tiles_path": str(output), "count": len(tiles), "skipped_count": 0}

    monkeypatch.setattr(pipeline, "_run_splice_eval_on_tiles_json", score)
    rc = lambda sequence: sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    result = pipeline._evaluate_existing_asos_payload(
        state={"request": "Evaluate the supplied steric ASOs and nearby splice junctions."},
        request_envelope={"request_classifications": ["splice_modification"],
            "evaluation_targets": [{"transcript_id": "TX_NEUTRAL.3", "site_kind": "donor", "position": 200}],
            "provided_asos": [{"name": "mapped", "sequence": rc(genome[4202:4222]), "sequence_role": "aso"},
                              {"name": "unmapped", "sequence": "A" * 20, "sequence_role": "aso"}]},
        run_dir=str(tmp_path / "output"), task_id="neutral_eval")
    assert "error" not in result
    assert {s["exon_index"] for s in dispatched} == {1, 2, 3, 4}
    mapped, unmapped = result["provided_asos"]
    assert mapped["splice_eval_summary"]["delta_inclusion_deletion"] == 0.0
    assert len(mapped["splice_eval_summary"]["site_search_scope"]["selected_sites"]) == 7
    assert unmapped["splice_eval_summary"]["splice_eval_skip_reason"] == "no_exact_binding_mapping"
