from __future__ import annotations

from copy import deepcopy

import pytest

import run_goal_intake as intake
from aso_agents.pipeline_tools import _clarification_qa_history


def _goal(accession: str) -> dict:
    return {
        "gene_symbol": "GENE_A",
        "isoform_scope_source": "user_specified",
        "splice_isoform_objectives": [
            {"isoform_label": accession, "transcript_id": accession, "exon_label": 4}
        ],
        "design_programs": [
            {"region_constraints": [{"reference_transcript_id": accession}]}
        ],
    }


def _catalog() -> list[dict]:
    return [
        {"id": "ENST00000000111.2", "is_mane_select": True, "is_canonical": True},
        {"id": "ENST00000000222.1", "is_canonical": False},
    ]


def test_inferred_wrong_gene_accession_uses_annotated_reference_everywhere() -> None:
    goal = _goal("ENST00000000999.4")
    payload = {"structured_goal": deepcopy(goal), "transcript_id": "ENST00000000999.4"}
    intake._reconcile_inferred_splice_reference(
        goal, payload, _catalog(), "Use GENCODE MANE Select for gene A exon 4."
    )
    assert goal["splice_isoform_objectives"][0]["transcript_id"] == "ENST00000000111.2"
    assert goal["design_programs"][0]["region_constraints"][0]["reference_transcript_id"] == "ENST00000000111.2"
    assert payload["transcript_id"] == "ENST00000000111.2"
    assert goal["isoform_scope_source"] == "reference_annotation"
    assert payload["isoform_resolution_notes"]["inferred_reference_reconciled"]["discarded_model_accession"] == "ENST00000000999.4"


def test_user_supplied_conflicting_accession_is_not_replaced() -> None:
    goal = _goal("ENST00000000999.4")
    payload = {"structured_goal": deepcopy(goal)}
    with pytest.raises(RuntimeError, match="user-supplied transcript accession conflicts"):
        intake._reconcile_inferred_splice_reference(
            goal, payload, _catalog(), "Use ENST00000000999.4 for gene A."
        )
    assert goal["splice_isoform_objectives"][0]["transcript_id"] == "ENST00000000999.4"


def test_multiple_inferred_refs_do_not_collapse_to_one_transcript() -> None:
    goal = _goal("ENST00000000999.4")
    goal["splice_isoform_objectives"].append(
        {"isoform_label": "ENST00000000888.2", "transcript_id": "ENST00000000888.2"}
    )
    with pytest.raises(RuntimeError, match="multiple model-inferred"):
        intake._reconcile_inferred_splice_reference(goal, {}, _catalog(), "Compare gene A isoforms.")


def test_catalog_valid_accession_is_preserved() -> None:
    goal = _goal("ENST00000000222.1")
    intake._reconcile_inferred_splice_reference(goal, {}, _catalog(), "Use gene A transcript 2.")
    assert goal["splice_isoform_objectives"][0]["transcript_id"] == "ENST00000000222.1"


def test_mane_policy_without_catalog_mane_requires_clarification() -> None:
    goal = _goal("ENST00000000999.4")
    catalog = [{"id": "ENST00000000222.1", "is_canonical": True}]
    with pytest.raises(RuntimeError, match="requested reference policy"):
        intake._reconcile_inferred_splice_reference(goal, {}, catalog, "Use MANE Select for gene A.")


def test_only_host_answer_carries_external_user_provenance() -> None:
    history = _clarification_qa_history({
        "clarification_history": [
            {"answers": [{"question": "Which accession?", "answer": "ENST00000000999.4"}],
             "summary": "Controller-generated answer"},
            {"answers": [{"question": "Which accession?", "answer": "ENST00000000111.2"}],
             "answer_source": "external_user"},
        ],
    })
    assert history[0].get("answer_source") is None
    assert history[1]["answer_source"] == "external_user"


def test_unaccessioned_mane_policy_corrects_design_and_evaluation_references(monkeypatch) -> None:
    monkeypatch.setattr(intake, 'local_reference_gene_transcripts', lambda *a, **kw: [
        {'id': 'ENST00000000111', 'versioned_id': 'ENST00000000111.2',
         'is_mane_select': True, 'reference_annotation': 'pinned.gtf'}])
    goal = {'gene_symbol': 'GENE_A', 'gene_id': 'ENSG00000000111', 'species': 'human',
            'isoform_scope_source': 'user_specified',
            'design_programs': [{'modality': 'steric', 'transcript_id': 'ENST00000000999.4'}],
            'evaluation_targets': [{'transcript_id': 'ENST00000000999.4'}]}
    payload = {'structured_goal': deepcopy(goal)}
    assert intake._reconcile_inferred_mane_reference(
        goal, payload, 'Use the pinned GENCODE MANE Select transcript for gene A.')
    assert goal['design_programs'][0]['transcript_id'] == 'ENST00000000111.2'
    assert goal['evaluation_targets'][0]['transcript_id'] == 'ENST00000000111.2'
    assert payload['structured_goal']['design_programs'][0]['transcript_id'] == 'ENST00000000111.2'
    assert goal['isoform_scope_source'] == 'reference_annotation'


def test_user_accession_is_not_overridden_by_mane_policy(monkeypatch) -> None:
    monkeypatch.setattr(intake, 'local_reference_gene_transcripts', lambda *a, **kw: pytest.fail('lookup should not run'))
    goal = {'gene_symbol': 'GENE_A', 'species': 'human',
            'design_programs': [{'transcript_id': 'ENST00000000999.4'}]}
    assert not intake._reconcile_inferred_mane_reference(
        goal, {}, 'Use ENST00000000999.4; compare with MANE Select.')
    assert goal['design_programs'][0]['transcript_id'] == 'ENST00000000999.4'


def test_mane_policy_replaces_nonaccession_primary_target_placeholder(monkeypatch) -> None:
    monkeypatch.setattr(intake, 'local_reference_gene_transcripts', lambda *a, **kw: [
        {'versioned_id': 'ENST00000000111.2', 'is_mane_select': True}])
    goal = {'gene_symbol': 'GENE_A', 'species': 'human',
            'transcript_lock': {'kind': 'explicit_id', 'value': 'awaiting user retrieval'},
            'evaluation_targets': [{'objective_type': 'primary_outcome',
                                    'transcript_id': 'awaiting user retrieval'}]}
    payload = {'structured_goal': deepcopy(goal)}
    assert intake._reconcile_inferred_mane_reference(goal, payload, 'Use MANE Select for GENE_A.')
    assert goal['evaluation_targets'][0]['transcript_id'] == 'ENST00000000111.2'
    assert payload['structured_goal']['evaluation_targets'][0]['transcript_id'] == 'ENST00000000111.2'
    assert goal['transcript_id'] == 'ENST00000000111.2'
    assert 'transcript_lock' not in goal


def test_multiple_splice_objectives_are_not_collapsed_by_mane_policy(monkeypatch) -> None:
    monkeypatch.setattr(intake, 'local_reference_gene_transcripts', lambda *a, **kw: [
        {'id': 'ENST00000000111', 'versioned_id': 'ENST00000000111.2', 'is_mane_select': True}])
    goal = {'gene_symbol': 'GENE_A', 'species': 'human',
            'splice_isoform_objectives': [{'transcript_id': 'ENST00000000999.4'},
                                          {'transcript_id': 'ENST00000000888.2'}]}
    assert not intake._reconcile_inferred_mane_reference(goal, {}, 'Use MANE Select for the reference.')
    assert goal['splice_isoform_objectives'][0]['transcript_id'] == 'ENST00000000999.4'


def test_distinct_design_program_references_are_not_collapsed_by_mane_policy(monkeypatch) -> None:
    monkeypatch.setattr(intake, 'local_reference_gene_transcripts', lambda *a, **kw: [
        {'versioned_id': 'ENST00000000111.2', 'is_mane_select': True}])
    goal = {'gene_symbol': 'GENE_A', 'species': 'human', 'design_programs': [
        {'transcript_id': 'ENST00000000999.4'}, {'transcript_id': 'ENST00000000888.2'}]}
    assert not intake._reconcile_inferred_mane_reference(goal, {}, 'Use MANE Select as a reference.')
    assert [item['transcript_id'] for item in goal['design_programs']] == [
        'ENST00000000999.4', 'ENST00000000888.2']
