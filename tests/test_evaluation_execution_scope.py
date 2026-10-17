import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents.evaluation_execution_scope import complete_evaluation_scope, scope_input, scope_errors


REQUEST = "Evaluate the supplied panel against its named references. Skip external transcriptome screening."
TASK = {"task_id": "evaluate", "task_type": "evaluate_existing_aso_set", "label": "Evaluate the supplied panel"}


def decision(payload, value="disabled", quote="Skip external transcriptome screening.", source="request"):
    return {"input_sha256": payload["input_sha256"], "decision": value,
            "evidence_ref": source if value != "unspecified" else None,
            "evidence_quote": quote if value != "unspecified" else None,
            "reason": "Apply the user's task-local execution scope."}


def resolve(tmp_path, classify, **overrides):
    return complete_evaluation_scope(**{
        "request": REQUEST, "task": TASK, "context": {"num_mismatches": 1},
        "directory": tmp_path, "classify": classify, **overrides})


@pytest.mark.parametrize("policy", ["enabled", "disabled"])
def test_explicit_context_is_preserved_without_another_model_call(tmp_path, policy):
    context = {"off_target_screening": policy, "num_mismatches": 0}
    result = resolve(tmp_path, lambda _: pytest.fail("Declared policy must not be reinterpreted"), context=context)
    assert result == {"context": context, "status": "declared", "receipt_path": None}
    result["context"]["off_target_screening"] = "changed"
    assert context["off_target_screening"] == policy


def test_missing_policy_is_evidence_bound_and_first_receipt_reused(tmp_path):
    calls = []
    def classify(payload):
        calls.append(deepcopy(payload))
        return decision(payload)
    first = resolve(tmp_path, classify)
    raw = open(first["receipt_path"]).read()
    second = resolve(tmp_path, lambda _: pytest.fail("Resume must reuse the first classification"))
    assert first == second and len(calls) == 1
    assert second["context"] == {"num_mismatches": 1, "off_target_screening": "disabled"}
    assert open(second["receipt_path"]).read() == raw
    assert json.loads(raw)["raw_result"] == decision(calls[0])


def test_unspecified_keeps_runtime_default_and_preserves_zero_budget(tmp_path):
    result = resolve(tmp_path, lambda p: decision(p, "unspecified"), context={"num_mismatches": 0})
    assert result["context"] == {"num_mismatches": 0}
    assert result["status"] == "unspecified"


@pytest.mark.parametrize("corrupt", [
    lambda p: {**decision(p), "input_sha256": "other-request"},
    lambda p: {**decision(p), "evidence_quote": "Invented permission"},
    lambda p: {**decision(p), "evidence_ref": "planner_assumption"},
    lambda p: {**decision(p), "evidence_quote": " "},
    lambda p: {**decision(p), "decision": "unspecified"},
    lambda p: {"error": "provider_unavailable"},
])
def test_unbound_or_failed_classification_cannot_authorize_screening(tmp_path, corrupt):
    first = resolve(tmp_path, corrupt)
    assert first["error"] == "evaluation_screening_scope_unverified" and "context" not in first
    saved = open(first["receipt_path"]).read()
    second = resolve(tmp_path, lambda _: pytest.fail("Failed first classification must remain cached"))
    assert second == first and open(second["receipt_path"]).read() == saved


def test_transport_exception_is_preserved_and_never_defaults_to_enabled(tmp_path):
    def fail(_):
        raise RuntimeError("transport unavailable")
    result = resolve(tmp_path, fail)
    receipt = json.loads(open(result["receipt_path"]).read())
    assert not receipt["accepted"] and receipt["raw_result"] == {"error": "RuntimeError: transport unavailable"}
    assert result["error"] == "evaluation_screening_scope_unverified"


def test_task_and_clarification_changes_are_separately_bound(tmp_path):
    parent = {"num_mismatches": 1}
    first = resolve(tmp_path, decision, context=parent)
    second = resolve(tmp_path, lambda p: decision(p, "unspecified"),
                     context=parent, task={**TASK, "task_id": "other", "label": "Other supplied set"})
    clarification = [{"answer_source": "external_user", "question": "Change screening scope?", "answer": "Run the external screen."},
                     {"answer_source": "model", "answer": "Assume no screen."}]
    third = resolve(tmp_path, lambda p: decision(p, "enabled", "Run the external screen.", "clarification_1"),
                    clarifications=clarification, context=parent)
    assert len({r["receipt_path"] for r in [first, second, third]}) == 3
    receipt = json.loads(open(third["receipt_path"]).read())
    assert [x["text"] for x in receipt["input"]["user_evidence"]] == [REQUEST, "Run the external screen."]
    assert parent == {"num_mismatches": 1} and third["context"]["off_target_screening"] == "enabled"


def test_native_scope_completion_reaches_runtime_without_external_calls(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    monkeypatch.setattr(pipeline, "_run_evaluation_screening_scope_sync", decision)
    state = {"request": REQUEST, "request_envelope": {"tasks": [TASK]}}
    context = {"num_mismatches": 0}
    completed = pipeline._complete_task_evaluation_scope(state, context, str(tmp_path), TASK["task_id"])
    goal = pipeline._overlay_request_context_on_goal({}, completed["context"], REQUEST)
    calls = []
    class Runtime:
        @staticmethod
        def _annotate_gapmer_off_targets(*args):
            pytest.fail("Excluded external search was dispatched")
        @staticmethod
        def _build_gapmer_kmer_index_by_len(cdna, lengths):
            calls.append("local-reference")
            return {8: {"AACCGGTA": {"reference"}}}
        @staticmethod
        def _annotate_gapmer_uniqueness(path, *args, **kwargs):
            return path
    monkeypatch.setattr(pipeline, "annotate_off_target_expression", lambda **_: pytest.fail("Excluded expression lookup"))
    rows = [{"name": "x", "sequence": "AACCGGTA", "alignments": [{"target_sequence": "TACCGGTT", "mismatches": 0}]}]
    before = deepcopy(rows[0]["alignments"])
    result = pipeline._score_provided_asos_with_existing_pipeline(rows=rows, eval_goal=goal,
        cdna_map={"reference": "TTAACCGGTA"}, iso_label="reference", runtime=Runtime(), run_dir=str(tmp_path), task_id="evaluate")
    assert goal["num_mismatches"] == 0 and context == {"num_mismatches": 0}
    assert calls == ["local-reference"] and rows[0]["alignments"] == before
    assert result["off_target_mapping"]["status"] == "not_run"
    assert rows[0]["Off_target_exact_count"] is None


def test_native_evaluation_checks_scope_before_runtime_dispatch(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    monkeypatch.setattr(pipeline, "_run_evaluation_screening_scope_sync", lambda p: {"error": "unavailable"})
    monkeypatch.setattr(pipeline, "_candidate_run_dirs_for_task", lambda *a: pytest.fail("Evaluation proceeded with unresolved scope"))
    result = pipeline._evaluate_existing_asos_payload(state={"request": REQUEST},
        request_envelope={"provided_asos": [{"name": "x", "sequence": "ACGTACGT"}]},
        run_dir=str(tmp_path), task_id="evaluate")
    assert result["error"] == "evaluation_screening_scope_unverified"


def test_actual_agent_validates_bound_first_response(monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    payload = scope_input(request=REQUEST, task=TASK, context={})
    async def run(agent, **kwargs):
        assert json.loads(kwargs["input"]) == payload
        assert agent.model_settings.extra_body["text"]["format"]["schema"]["properties"]["input_sha256"]["enum"] == [payload["input_sha256"]]
        return SimpleNamespace(final_output=json.dumps(decision(payload)))
    monkeypatch.setattr(pipeline.Runner, "run", run)
    result = asyncio.run(pipeline._run_evaluation_screening_scope_agent_once(payload))
    assert result == decision(payload) and scope_errors(result, payload) == []
