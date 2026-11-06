from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import transcript_rules as rules


RULE = "canonical protein-coding transcript"
PLAN = {"filters": {"is_canonical": True, "biotype": "protein_coding"},
        "order_by": None, "unresolved_reason": None}
CANDIDATES = [
    {"id": "ENST00000000111", "versioned_id": "ENST00000000111.3", "is_canonical": True,
     "is_mane_select": False, "biotype": "protein_coding", "cdna_length": 1000, "length": 9000},
    {"id": "ENST00000000222", "is_canonical": False, "is_mane_select": True,
     "biotype": "protein_coding", "cdna_length": 2000, "length": 4000},
]


def runtime(monkeypatch, candidates=None, plan=None, strand=1):
    calls = []
    def interpret(agent, *, input, **kwargs):
        assert agent is rules._rule_agent
        calls.append(json.loads(input))
        return SimpleNamespace(final_output=json.dumps({"selection": PLAN if plan is None else plan,
            "objective_labels": ["EXAMPLE mRNA"], "rule_quote": RULE}))
    monkeypatch.setattr(rules.Runner, "_run_sync", interpret)
    from aso_agents.transcript_metadata import resolved_transcript_metadata
    return SimpleNamespace(
        list_transcripts=lambda **kwargs: {"transcripts": CANDIDATES if candidates is None else candidates},
        _fetch_transcript_structure=lambda tid: {"id": tid, "gene_id": "GENE1", "species": "human",
            "seq_region_name": "1", "strand": strand,
            "exons_tx": [{"start": 100, "end": 200, "index": 1}, {"start": 300, "end": 400, "index": 2}]},
        _source_identifier_matches=lambda structure, tid: structure["id"] == tid,
        normalize_species=lambda value: value,
        _normalize_ensembl_gene_id=lambda value: value,
        resolved_transcript_metadata=resolved_transcript_metadata,
        _extract_splice_sites=lambda structure: [],
        calls=calls,
    )


def goal():
    return {"gene_symbol": "EXAMPLE", "gene_id": "GENE1", "species": "human",
        "transcript_policy": "user_locked", "transcript_lock": {"kind": "verbal_rule", "value": RULE, "source": "user_request"},
        "splice_isoform_objectives": [{"isoform_label": "EXAMPLE mRNA", "desired_abundance": "decrease"},
                                      {"isoform_label": "alternative form", "desired_abundance": "increase"}],
        "design_programs": [{"program_id": "intron", "splice_event": {
            "event_type": "intron_retention", "intron_number": 1, "reference_transcript_id": RULE}},
            {"program_id": "translation", "region_constraints": [{"source_type": "transcript_feature",
                "feature_name": "start_codon", "reference_transcript_id": RULE}]}],
        "rationale": RULE, "cross_species_targets": {"mouse": {"transcript_id": RULE}}}


@pytest.mark.parametrize("strand", [1, -1])
def test_rule_binds_reference_before_feature_and_intron_resolution(monkeypatch, strand):
    rt, source, final = runtime(monkeypatch, strand=strand), goal(), {}
    result = rules.resolve_transcript_rule(rt, source, final, "Use the " + RULE)
    assert result["selected_transcript_id"] == CANDIDATES[0]["id"]
    assert result["annotation"]["is_mane_select"] is False
    assert source["design_programs"][0]["splice_event"]["reference_transcript_id"] == source["transcript_id"]
    assert source["design_programs"][1]["region_constraints"][0]["reference_transcript_id"] == source["transcript_id"]
    assert source["rationale"] == RULE
    assert source["splice_isoform_objectives"][0]["transcript_id"] == source["transcript_id"]
    assert "transcript_id" not in source["splice_isoform_objectives"][1]
    assert source["cross_species_targets"]["mouse"]["transcript_id"] == RULE
    assert source["isoform_transcripts"][source["transcript_id"]]["strand"] == ("-" if strand == -1 else "+")
    assert final["transcript_rule_resolution"]["original_lock"]["value"] == RULE
    assert rules.resolve_transcript_rule(rt, source, final, "Use the " + RULE)["status"] == "not_required"
    assert len(rt.calls) == 1


def test_longest_uses_spliced_length_and_missing_length_prevents_ranking():
    plan = {"filters": {"biotype": "protein_coding"}, "order_by": "cdna_length", "unresolved_reason": None}
    assert rules.select_rule_match(plan, CANDIDATES)[0]["id"] == CANDIDATES[1]["id"]
    assert rules.select_rule_match(plan, [*CANDIDATES, {"id": "UNKNOWN", "biotype": "protein_coding"}]) == []


@pytest.mark.parametrize("candidates", [[], [CANDIDATES[1]], [CANDIDATES[0], {**CANDIDATES[0], "id": "OTHER"}]])
def test_unresolved_or_ambiguous_rule_leaves_goal_unchanged(monkeypatch, candidates):
    rt, source = runtime(monkeypatch, candidates=candidates), goal()
    before = deepcopy(source)
    result = rules.resolve_transcript_rule(rt, source, {}, "Use the " + RULE)
    assert result["status"] == "needs_clarification"
    assert source == before


def test_explicit_reference_bypasses_rule_interpretation(monkeypatch):
    rt, source = runtime(monkeypatch), goal()
    source["transcript_lock"] = {"kind": "reference_accession", "value": "NM_123456.8"}
    assert rules.resolve_transcript_rule(rt, source, {}, "") == {"status": "not_required"}
    assert rt.calls == []


def test_user_rule_also_resolves_when_intake_records_it_as_isoform_scope(monkeypatch):
    rt, source = runtime(monkeypatch), goal()
    source.pop("transcript_lock")
    source.update(transcript_policy="agent_choice", isoform_scope=RULE, isoform_scope_source="user_specified")
    result = rules.resolve_transcript_rule(rt, source, {}, "Use the " + RULE)
    assert result["status"] == "resolved"
    assert source["transcript_id"] == CANDIDATES[0]["id"]
    assert source["transcript_policy"] == "user_locked"


def test_multi_isoform_scope_is_not_narrowed_to_one_reference(monkeypatch):
    rt, source = runtime(monkeypatch), goal()
    source.pop("transcript_lock")
    source.update(isoform_scope="all protein-coding isoforms", isoform_scope_source="user_specified")
    before = deepcopy(source)
    monkeypatch.setattr(rules.Runner, "_run_sync", lambda *_a, **_k: SimpleNamespace(
        final_output=json.dumps({"selection": None, "objective_labels": [], "rule_quote": None})))
    assert rules.resolve_transcript_rule(rt, source, {}, "Target all protein-coding isoforms.")["status"] == "not_required"
    assert source == before


def test_original_request_recovers_rule_omitted_from_intake_lock_and_scope(monkeypatch):
    rt, source = runtime(monkeypatch), goal()
    source.pop("transcript_lock")
    source["transcript_policy"] = "agent_choice"
    assert rules.resolve_transcript_rule(rt, source, {}, "Use the " + RULE)["status"] == "resolved"
    assert source["splice_isoform_objectives"][0]["transcript_id"] == CANDIDATES[0]["id"]


def test_named_isoform_alias_without_rule_continues_to_its_existing_resolver(monkeypatch):
    rt, source = runtime(monkeypatch), goal()
    source.pop("transcript_lock")
    before = deepcopy(source)
    monkeypatch.setattr(rules.Runner, "_run_sync", lambda *_a, **_k: SimpleNamespace(
        final_output=json.dumps({"selection": None, "objective_labels": [], "rule_quote": None})))
    assert rules.resolve_transcript_rule(rt, source, {}, "Use the short form.")["status"] == "not_required"
    assert source == before


@pytest.mark.parametrize("plan", [
    {**PLAN, "filters": {}}, {**PLAN, "filters": {"is_canonical": "true"}},
    {**PLAN, "filters": {"invented_predicate": True}}, {**PLAN, "order_by": "length"},
    {**PLAN, "selected_transcript_id": "INVENTED"},
])
def test_unverified_model_predicates_are_software_errors(plan):
    with pytest.raises(ValueError):
        rules.select_rule_match(plan, CANDIDATES)


def test_failed_structure_is_atomic(monkeypatch):
    rt, source = runtime(monkeypatch), goal()
    before = deepcopy(source)
    rt._source_identifier_matches = lambda *_args: False
    with pytest.raises(ValueError, match="structure_unverified"):
        rules.resolve_transcript_rule(rt, source, {}, "Use the " + RULE)
    assert source == before


def test_public_resolution_binds_rule_before_resolving_features(monkeypatch, tmp_path):
    from aso_agents import pipeline_tools as pipeline
    rt, source = runtime(monkeypatch), goal()
    outcomes = source.pop("splice_isoform_objectives")
    rt._prepare_snapgene_objectives = lambda bound, *_a, **_k: bound.update(splice_isoform_objectives=deepcopy(outcomes)) or {}
    state = {"run_dir": str(tmp_path), "request": "Use the " + RULE, "stage_attempts": {}}
    final = {"structured_goal": source}
    payload = {"request": state["request"], "final_payload": final}
    seen = []
    def resolve(bound, *_args, **_kwargs):
        assert bound["transcript_rule_resolution"]["status"] == "resolved"
        assert bound["design_programs"][0]["splice_event"]["reference_transcript_id"] == CANDIDATES[0]["id"]
        seen.append(bound["transcript_id"])
        return {}
    rt._resolve_gene_id = lambda *_a, **_k: "GENE1"
    rt._resolve_isoform_transcripts = resolve
    monkeypatch.setattr(pipeline, "_prepare_execution_stage", lambda **kwargs: {
        "status": "ok", "state": state, "run_dir": str(tmp_path)})
    monkeypatch.setattr(pipeline, "_load_goal_context_from_run_dir", lambda path: {
        "payload": payload, "final_payload": final, "structured_goal": source})
    monkeypatch.setattr(pipeline, "_goal_runtime_module", lambda: rt)
    monkeypatch.setattr(pipeline, "_save_state", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_save_goal_payload", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_maybe_rename_run_dir_with_gene", lambda **kwargs: (
        str(tmp_path), payload, source, final, False, kwargs["state_path"]))
    monkeypatch.setattr(pipeline, "_promote_atomic_status", lambda **kwargs: kwargs)
    result = pipeline.tool_resolve_isoforms_and_sites(str(tmp_path / "run_state.json"))
    assert result["status"] == "resolution_ready"
    assert seen == [CANDIDATES[0]["id"]]
