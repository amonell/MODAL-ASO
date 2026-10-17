from copy import deepcopy
import importlib
import json
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools
from aso_agents.clarification_identity import apply_confirmed_transcript_identities
from aso_agents.run_layout import resolve_run_path

chooser = importlib.import_module("aso_agents.transcript_chooser_agent")
OLD = "ENST00000000111.1"
NEW = "ENST00000000222.4"
PRIMARY = "ENST00000000333.2"
ANSWER = f"Use {PRIMARY} for active form and {NEW} for protected form."


def response(**updates):
    item = {"isoform_label": "protected form", "transcript_id": NEW,
            "answer_index": 0, "evidence_quote": ANSWER}
    item.update(updates)
    return {"selections": [item]}


def stub(monkeypatch, payloads):
    calls = []
    def run(agent, *, input, **kwargs):
        assert agent is chooser._additional_transcript_selections_agent
        calls.append(json.loads(input))
        return SimpleNamespace(final_output=json.dumps(payloads[min(len(calls) - 1, len(payloads) - 1)]))
    monkeypatch.setattr(chooser.Runner, "_run_sync", run)
    return calls


def goal():
    return {"gene_symbol": "NEUTRAL", "intent": "splice_modulation", "isoform_resolution_locked": True,
            "isoform_transcripts": {"protected form": {
                "transcript_id": OLD, "exons_tx": [{"index": 1, "start": 100, "end": 150}]}},
            "preservation_targets": [{"isoform_label": "protected form", "action": "preserve",
                "preservation_target_type": "isoform", "gene_symbol": "NEUTRAL"}],
            "splice_isoform_objectives": [{"isoform_label": "active form", "desired_abundance": "decrease"}]}


def test_additional_selection_remains_explicit_and_version_bound(monkeypatch):
    calls = stub(monkeypatch, [response()])
    result = chooser.interpret_additional_transcript_selections([{"answer": ANSWER}], ["protected form"])
    assert len(calls) == 1
    assert result[0]["transcript_id"] == NEW
    assert result[0]["raw_answer"] == ANSWER
    assert result[0]["decision"] == "user_selected"


@pytest.mark.parametrize("bad", [
    {"answer_index": True}, {"answer_index": 9},
    {"evidence_quote": "A fabricated quotation containing protected form and " + NEW},
    {"transcript_id": OLD}, {"isoform_label": "unmentioned form"},
])
def test_unbound_selection_repairs_against_original_evidence(monkeypatch, bad):
    invalid = response(**bad)
    calls = stub(monkeypatch, [invalid, response()])
    result = chooser.interpret_additional_transcript_selections([{"answer": ANSWER}], ["protected form"])
    assert result[0]["transcript_id"] == NEW and len(calls) == 2
    assert calls[1]["answers"] == calls[0]["answers"]
    assert json.loads(calls[1]["previous_response"]) == invalid and calls[1]["repair_errors"]


def test_assistant_question_or_unscoped_yes_cannot_supply_additional_selection(monkeypatch):
    monkeypatch.setattr(chooser.Runner, "_run_sync", lambda *_a, **_k: pytest.fail("No explicit reply reference"))
    assert chooser.interpret_additional_transcript_selections(
        [{"question": ANSWER, "answer": "Yes."}], ["protected form"]) == []


def test_semantically_negative_reply_preserves_no_selection(monkeypatch):
    stub(monkeypatch, [{"selections": []}])
    answer = f"The feature is absent from {NEW} in the protected form; keep the current reference."
    assert chooser.interpret_additional_transcript_selections([{"answer": answer}], ["protected form"]) == []


def test_changed_reference_invalidates_cache_preserves_evidence_and_preservation_role():
    source = goal()
    old = deepcopy(source["isoform_transcripts"]["protected form"])
    selection = [{"isoform_label": "protected form", "transcript_id": NEW, "decision": "user_selected"}]
    changes = apply_confirmed_transcript_identities(source, selection)
    assert changes and "isoform_resolution_locked" not in source
    assert source["preservation_targets"][0]["transcript_id"] == NEW
    assert source["preservation_targets"][0]["action"] == "preserve"
    assert source["isoform_transcripts"] == {}
    assert source["superseded_isoform_transcript_selections"][0]["previous_reference"] == old
    assert len(source["splice_isoform_objectives"]) == 1
    assert not apply_confirmed_transcript_identities(source, selection)


def test_exact_attested_version_can_reuse_structure_but_conflicting_version_cannot():
    source = goal()
    source["isoform_transcripts"]["protected form"].update(
        transcript_id=NEW.split(".")[0], metadata={"versioned_id": NEW})
    selected = [{"isoform_label": "protected form", "transcript_id": NEW, "decision": "accept"}]
    apply_confirmed_transcript_identities(source, selected)
    assert source["isoform_transcripts"]
    selected[0]["transcript_id"] = NEW.split(".")[0] + ".9"
    apply_confirmed_transcript_identities(source, selected)
    assert not source["isoform_transcripts"]


def test_actual_confirmation_boundary_applies_also_to_previously_resolved_isoform(tmp_path, monkeypatch):
    source = goal()
    source["pending_isoform_transcript_confirmations"] = [
        {"isoform_label": "active form", "suggested_transcript_id": PRIMARY}]
    run = tmp_path / "run"
    run.mkdir()
    pipeline_tools._save_goal_payload(str(run), {"request": "Change active form; preserve protected form."},
                                     deepcopy(source), deepcopy(source))
    question = f"Use {PRIMARY} for 'active form'?"
    state = {"status": "clarification_answers_ready", "phase": "clarification_answers_ready",
             "request": "Change active form; preserve protected form.", "run_dir": str(run),
             "clarification": {"kind": "isoform_transcript_confirmation", "pending_apply": True,
                 "restart_stage": "resolution", "last_answers": [{"question": question, "answer": ANSWER}]},
             "pending_questions": [], "clarification_qa_history": [], "force_rerun_stages": []}
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps(state))
    stub(monkeypatch, [response()])
    monkeypatch.setattr(pipeline_tools, "interpret_transcript_confirmation_answer", lambda *_a, **_k:
                        {"decision": "accept", "transcript_id": PRIMARY, "rationale": "Explicit primary choice."})
    monkeypatch.setattr(pipeline_tools, "list_transcripts", lambda **_k: {"transcripts": []})
    result = pipeline_tools.apply_clarification_to_goal(str(state_path))
    assert result["status"] == "ok" and result["applied_transcript_confirmations"] == 2
    saved = json.loads(resolve_run_path(run, "structured_goal.json").read_text())
    assert saved["preservation_targets"][0]["transcript_id"] == NEW
    assert saved["splice_isoform_objectives"][0]["transcript_id"] == PRIMARY
    assert "protected form" not in saved["isoform_transcripts"]
    assert {item["isoform_label"] for item in saved["isoform_transcript_confirmations"]} == {"active form", "protected form"}
