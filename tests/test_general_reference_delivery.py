import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as pipeline, reference_decisions as decisions, species_config
from aso_agents.design_delivery import design_delivery
from aso_agents.target_accessibility import select_panel_rows, unpaired_probabilities
from aso_agents.transcript_sequence import reference_cdna
import aso_agents.transcript_sequence as transcript_sequence


def decision(action, answer, scope="resources", **extras):
    return dict(action=action, scope=scope, species=[], answer_index=0, evidence=answer,
                reference_data_root="", download_dir="", model_path="", species_resolutions=[],
                reason="Interpreted complete answer", **extras)


@pytest.mark.parametrize("answer,action,scope", [
    ("Do not skip any requested species.", "continue", "none"),
    ("Never cancel the job; use the references already installed.", "continue", "resources"),
    ("I decline that download; retain the request.", "ask", "none"),
    ("Skip this task.", "skip", "task"),
    ("Cancel the whole request.", "stop", "task"),
])
def test_provisioning_consumes_typed_full_answer_without_reparsing_keywords(monkeypatch, answer, action, scope):
    result = decision(action, answer, scope)
    calls = []
    def interpret(name, instructions, payload, schema, validator):
        calls.append(payload)
        assert not validator(result)
        return result
    monkeypatch.setattr(decisions, "semantic_result", interpret)
    state = {"clarification": {"kind": "reference_preflight", "last_answers": [{"question": "References?", "answer": answer}]}}
    assert pipeline._parse_reference_provisioning_decision(state, {})["action"] == action
    assert pipeline._parse_reference_provisioning_decision(state, {})["action"] == action
    assert len(calls) == 1


def test_question_wording_cannot_authorize_skip_and_partial_scope_cannot_expand(monkeypatch):
    answer = "Retain all work."
    state = {"clarification": {"kind": "reference_preflight", "last_answers": [{"question": "Skip?", "answer": answer}]}}
    monkeypatch.setattr(decisions, "semantic_result", lambda *args: decision("skip", "Skip?", "task"))
    assert decisions.provisioning_decision(state, {})["action"] == "ask"
    state.pop("reference_provisioning_resolution")
    monkeypatch.setattr(decisions, "semantic_result", lambda *args: decision("skip", answer, "species"))
    assert decisions.provisioning_decision(state, {})["action"] == "ask"


def test_unrelated_clarification_does_not_control_reference_provisioning(monkeypatch):
    monkeypatch.setattr(decisions, "semantic_result", lambda *args: pytest.fail("must not interpret unrelated answers"))
    state = {"clarification": {"kind": "target_context", "last_answers": [{"answer": "stop codon assay"}]}}
    assert decisions.provisioning_decision(state, {}) == {"action": "ask"}


@pytest.mark.parametrize("same_identity", [True, False])
def test_composite_gene_label_requires_reference_agreement(monkeypatch, same_identity):
    monkeypatch.setattr(decisions, "semantic_result", lambda *args: {"names": ["Alpha", "AliasA"], "single_gene": True})
    def lookup(name, species):
        return {"gene_id": "gene1" if same_identity or name == "Alpha" else "gene2", "display_name": "Alpha"}
    result = decisions.resolve_gene_label("Alpha (AliasA)", "neutral_species", lookup)
    assert bool(result.get("gene_id")) is same_identity


def test_taxonomic_synonym_resolves_only_matching_catalog_taxon(monkeypatch):
    monkeypatch.setattr(species_config, "list_ensembl_species", lambda: [{"name": "neutral_species", "taxon_id": 123456, "aliases": []}])
    monkeypatch.setattr(species_config.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: [{"id": "123456", "name": "Neutral species"}], url="https://reference/taxonomy"))
    result = species_config.resolve_ensembl_species("neutral common name")
    assert result["species_key"] == "neutral_species"
    assert result["taxonomy_evidence"]["response"][0]["id"] == "123456"


def test_identity_clarification_updates_nested_species_but_preserves_gene_text(monkeypatch):
    monkeypatch.setattr(species_config, "resolve_ensembl_species", lambda name: {"status": "resolved", "species_key": "neutral_species"})
    goal = {"species": "human", "gene_symbol": "old_label", "cross_species_requirement": {"species": ["human", "old_label"]}}
    result = decisions.apply_species_resolutions(goal, [{"requested": "old_label", "reference_name": "Neutral species"}])
    assert result
    assert goal["cross_species_requirement"]["species"] == ["human", "neutral_species"]
    assert goal["gene_symbol"] == "old_label"


@pytest.mark.parametrize("modality,secondary_genome", [("gapmer", False), ("steric", True)])
def test_reference_resources_follow_actual_sequence_operation(monkeypatch, tmp_path, modality, secondary_genome):
    human = tmp_path / "human.fa"; human.write_text("data")
    secondary = tmp_path / "absent-secondary.fa"
    goal = {"species": "human", "design_programs": [{"modality": modality}],
            "cross_species_requirement": {"species": ["human", "neutral_species"], "required": True}}
    monkeypatch.setattr(pipeline, "_goal_payload_context", lambda _: {"structured_goal": goal})
    monkeypatch.setattr(pipeline, "_resolve_run_splice_eval", lambda _: False)
    monkeypatch.setattr(pipeline, "resolve_ensembl_species", lambda _: {"status": "resolved"})
    monkeypatch.setattr(pipeline, "resolve_openspliceai_model", lambda _: pytest.fail("no splice prediction requested"))
    monkeypatch.setattr(pipeline, "_required_tiling_reference_files", lambda key: [str(human if key == "human" else secondary)])
    monkeypatch.setattr(pipeline, "_tiling_reference_requirement_groups", lambda key: [("genome", [str(secondary)])])
    result = pipeline._preflight_reference_requirements(str(tmp_path))
    assert (str(secondary) in result["missing_files"]) is secondary_genome
    assert result["model_missing_species"] == []


def test_reference_sequence_has_identity_length_and_cached_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    record = {"id": "TX123", "seq": "ACGT" * 8}
    calls = []
    def get(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: record, url="https://reference/sequence/TX123")
    monkeypatch.setattr(transcript_sequence.requests, "get", get)
    seq, evidence = reference_cdna("TX123", 32)
    assert seq == "ACGT" * 8 and Path(evidence["record_path"]).is_file()
    assert reference_cdna("TX123", 32)[0] == seq and len(calls) == 1
    with pytest.raises(ValueError, match="length_disagrees"):
        reference_cdna("TX123", 31)
    with pytest.raises(ValueError, match="identity_mismatch"):
        reference_cdna("TX_OTHER", 32)


def test_actual_rnaplfold_distinguishes_unpaired_target_from_basepairing_target(tmp_path):
    unpaired, receipt = unpaired_probabilities("A" * 100, 20, tmp_path / "unpaired")
    paired, _ = unpaired_probabilities("G" * 35 + "A" * 10 + "C" * 35, 20, tmp_path / "paired")
    assert unpaired[(30, 20)] > 0.99
    assert paired[(30, 20)] < unpaired[(30, 20)]
    assert receipt["coordinate_convention"].startswith("1-based inclusive")


def test_panel_selection_counts_unique_molecules_and_covers_distinct_sites():
    rows = [{"Sequence": seq, "Chrom": "TX", "Start": start, "End": start + 19}
            for seq, start in [("A"*20, 1), ("A"*20, 1), ("C"*20, 2), ("G"*20, 100)]]
    assert [row["Start"] for row in select_panel_rows(rows, 2, True)] == [1, 100]
    assert len(select_panel_rows(rows, 10, True)) == 3


def test_empty_report_cannot_fulfil_a_requested_design():
    result = design_delivery({"candidate_minimum": 2}, {"rows": []}, {"gapmer": {"requested": True, "status": "skipped"}})
    assert result["status"] == "unfulfilled"
    assert "requested_design_not_executed:gapmer" in result["errors"]


def test_delivered_sequences_require_correct_orientation_and_requested_evidence():
    row = {"target_binding_sequence": "AACG", "synthesized_antisense_sequence": "CGTT",
           "target_accessibility": {"status": "measured"}}
    goal = {"candidate_minimum": 1, "candidate_limit": 2, "target_accessibility": True}
    branches = {"gapmer": {"requested": True, "status": "outputs_present"}}
    assert design_delivery(goal, {"rows": [row]}, branches)["status"] == "fulfilled"
    bad = deepcopy(row); bad["synthesized_antisense_sequence"] = "AACG"
    assert "candidate_orientation_inconsistent" in design_delivery(goal, {"rows": [bad]}, branches)["errors"]
    bad = deepcopy(row); bad.pop("target_accessibility")
    partial = design_delivery(goal, {"rows": [bad]}, branches)
    assert partial["status"] == "fulfilled_with_limitations"
    assert "requested_target_accessibility_not_measured" in partial["warnings"]
    assert not partial["errors"]


def test_local_cdna_records_actual_reference_and_rejects_missing_exon(monkeypatch, tmp_path):
    import run_goal_intake as runtime
    genome = tmp_path / "reference.fa"
    genome.write_text(">chrA\nAACCGGTTACGTAACCGGTT\n")
    monkeypatch.setattr(runtime, "get_species_config", lambda _: SimpleNamespace(genome_fasta=genome))
    entry = {"transcript_id": "TX_A", "chrom": "chrA", "strand": "-",
             "metadata": {"versioned_id": "TX_A.3", "reference_annotation": "recorded_annotation"},
             "exons_tx": [{"index": 1, "start": 13, "end": 16}, {"index": 2, "start": 1, "end": 4}]}
    result = runtime._build_isoform_cdna_sequences({"TX_A": entry}, "human")
    assert result == {"TX_A": "GGTTGGTT"}
    assert entry["sequence_evidence"]["versioned_id"] == "TX_A.3"
    assert entry["sequence_evidence"]["genome_fasta"] == str(genome)
    incomplete = deepcopy(entry)
    incomplete["exons_tx"][1]["start"] = 100
    incomplete["exons_tx"][1]["end"] = 103
    assert runtime._build_isoform_cdna_sequences({"TX_A": incomplete}, "human") == {}
    assert incomplete["sequence_resolution_error"] == "incomplete_reference_exon_sequence"


def test_report_stage_materializes_species_evidence_and_preserves_panel_history(monkeypatch, tmp_path):
    from aso_agents import report_explainer
    candidates = tmp_path / "artifacts" / "candidates"
    candidates.mkdir(parents=True)
    scored = candidates / "gapmer_scored_uniqueness.json"
    scored.write_text(json.dumps({"rows": [{"Sequence": "AACCGGTT", "Synthesis": "AACCGGTT",
        "cross_species_required": True, "cross_species_pass": True,
        "target_accessibility": {"status": "measured", "measurements": {"human": [{}], "mouse": [{}]}},
        "off_target_screening_status_by_species": {"human": "complete", "mouse": "mapping_failed"}}]}))
    goal = {"gene_symbol": "GENE_A", "species": "human", "intent": "knockdown", "candidate_limit": 1,
            "cross_species_requirement": {"required": True, "species": ["human", "mouse"]}}
    selection = {"method": "ranked_nonoverlapping_targets_then_nearby_alternatives", "eligible_count": 12,
                 "selected_count": 1, "unselected_count": 11, "status": "complete", "candidate_limit": 1}
    final = {"structured_goal": goal, "gapmer_tiling_results": {"scored_tiles_path": str(scored),
             "row_count": 1, "candidate_selection": selection}}
    payload = {"request": "Design a conserved panel.", "final_payload": final}
    pipeline._save_goal_payload(str(tmp_path), payload, goal, final)
    monkeypatch.setattr(pipeline, "_prepare_execution_stage", lambda **kwargs: {
        "status": "ok", "state_path": str(tmp_path / "state.json"), "state": {}, "run_dir": str(tmp_path)})
    monkeypatch.setattr(pipeline, "_promote_atomic_status", lambda **kwargs: kwargs)
    pipeline.tool_build_report_payload(str(tmp_path / "state.json"))
    generated = json.loads((candidates / "generated_aso_set.json").read_text())
    assert generated["candidate_selection"]["method"] == selection["method"]
    assert generated["candidate_selection"]["eligible_count"] == 12
    assert generated["candidate_selection"]["unselected_count"] == 11
    report = pipeline._load_goal_context_from_run_dir(str(tmp_path))["final_payload"]["report_payload"]
    markdown = report_explainer.render_report_markdown({}, report)
    assert "Cross-species exact matching was required" in markdown
    assert "single-species decision surface" not in markdown
    assert "| mouse | 1/1 | 0/1 |" in markdown
