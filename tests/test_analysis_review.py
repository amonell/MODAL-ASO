import json
from pathlib import Path

import pytest

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.analysis_review import assess_review, program_digest


REQUEST = "Return the exact sum of the supplied numbers in totals.json."


def finding(quote=REQUEST):
    return {"requirement_quote": quote, "obligation": "Sum every supplied number",
            "status": "violated", "problem": "The total omits an input.",
            "implementation_evidence": "Only the first input is summed.",
            "suggested_fix": "Sum every supplied number."}


def satisfied():
    return {"requirement_quote": REQUEST, "obligation": "Sum every supplied number", "status": "satisfied",
            "implementation_evidence": "sum(values) includes both inputs and the exported total is 5.",
            "problem": None, "suggested_fix": None}


def program(total_expression):
    return (
        "import json\nfrom pathlib import Path\n"
        "values = [2, 3]\n"
        f"result = {{'total': {total_expression}}}\n"
        "Path('totals.json').write_text(json.dumps(result))\n"
        "Path('sandbox_result.json').write_text(json.dumps(result))\n"
        "Path('sandbox_summary.md').write_text('Computed total: ' + str(result['total']))\n"
    )


def execute(tmp_path, generate, review, max_attempts=2, **kwargs):
    return execute_analysis_program(
        directory=tmp_path, payload={"request_text": REQUEST},
        generate=generate, review_program=review,
        required_outputs=[tmp_path / "totals.json"], max_attempts=max_attempts, **kwargs)


def test_actual_output_finding_repairs_program_and_preserves_first_attempt(tmp_path):
    requests, reviews = [], []

    def generate(payload):
        requests.append(payload)
        return {"python_code": program("sum(values[:1])" if len(requests) == 1 else "sum(values)")}

    def review(payload):
        reviews.append(payload)
        actual = json.loads(Path(payload["output_paths"][0]).read_text())
        return {"program_sha256": payload["program_sha256"],
                "checks": [finding()] if actual["total"] != 5 else [satisfied()]}

    result = execute(tmp_path, generate, review)
    assert result["ok"] and result["result"]["total"] == 5
    assert len(requests) == len(reviews) == 2
    assert REQUEST in requests[1]["execution_repair"]["error"]
    assert "sum(values[:1])" in requests[1]["execution_repair"]["previous_code"]
    first = tmp_path / "execution_attempts/01"
    assert "sum(values[:1])" in (first / "generated_code.py").read_text()
    assert json.loads((first / "contract_review.json").read_text())["status"] == "needs_repair"
    archived = result["attempts"][1]["previous_output_archives"]
    output = next(item for item in archived if item["path"].endswith("totals.json"))
    assert json.loads(Path(output["archive"]).read_text())["total"] == 2


@pytest.mark.parametrize("raw", [
    {"program_sha256": "wrong", "checks": [finding()]},
    {"checks": [finding()]},
    {"program_sha256": None, "checks": []},
    None,
])
def test_unbound_feedback_is_unverified(raw):
    result = assess_review(raw, "print(1)", REQUEST)
    assert result["status"] == "unverified" and result["findings"] == []


def test_unrequested_criterion_cannot_trigger_code_rewrite(tmp_path):
    calls = []
    def generate(payload):
        calls.append(payload)
        return {"python_code": program("sum(values)")}
    result = execute(tmp_path, generate, lambda payload: {
        "program_sha256": payload["program_sha256"],
        "checks": [finding("Remove values below a threshold chosen by the reviewer.")]})
    assert result["ok"] and len(calls) == 1
    assert result["result"]["analysis_contract_review"]["status"] == "unverified"
    receipt = json.loads((tmp_path / "execution_attempts/01/contract_review.json").read_text())
    assert receipt["rejected_feedback"] and not receipt["findings"]


def test_review_transport_failure_is_not_a_scientific_finding(tmp_path):
    calls = []
    def generate(payload):
        calls.append(payload)
        return {"python_code": program("sum(values)")}
    def unavailable(payload):
        raise OSError("provider unavailable")
    result = execute(tmp_path, generate, unavailable)
    assert result["ok"] and len(calls) == 1
    assert result["attempts"][0]["contract_review"]["status"] == "unverified"
    receipt = json.loads((tmp_path / "execution_attempts/01/contract_review.json").read_text())
    assert "provider unavailable" in receipt["raw_review"]["error"]


def test_persistent_omission_stops_at_existing_attempt_bound(tmp_path):
    result = execute(tmp_path, lambda payload: {"python_code": program("sum(values[:1])")},
                     lambda payload: {"program_sha256": payload["program_sha256"],
                                      "checks": [finding()]})
    assert not result["ok"] and len(result["attempts"]) == 2
    assert all(a["contract_review"]["status"] == "needs_repair" for a in result["attempts"])


def reviewed_sequence(tmp_path, codes, **kwargs):
    requests, reviews = [], []

    def generate(payload):
        requests.append(payload)
        return {"python_code": codes[len(requests) - 1]}

    def review(payload):
        reviews.append(payload)
        actual = json.loads(Path(payload["output_paths"][0]).read_text())
        return {"program_sha256": payload["program_sha256"],
                "checks": [satisfied()] if actual["total"] == 5 else [finding()]}

    result = execute(tmp_path, generate, review, **kwargs)
    return result, requests, reviews


@pytest.mark.parametrize("repaired", [True, False])
def test_late_first_review_gets_one_bounded_repair(tmp_path, repaired):
    broken = "raise ValueError('execution failed')"
    incomplete = program("sum(values[:1])")
    final = program("sum(values)") if repaired else incomplete
    result, requests, reviews = reviewed_sequence(
        tmp_path, [broken, broken, incomplete, final], max_attempts=3,
        first_review_repair=True)
    assert result["ok"] is repaired
    assert len(requests) == 4 and len(reviews) == 2
    assert [a["attempt"] for a in result["attempts"]] == [1, 2, 3, 4]
    assert result["attempts"][2]["first_review_repair"]["authorized_attempt"] == 4
    assert result["attempts"][3]["review_repair_from_attempt"] == 3
    assert REQUEST in requests[3]["execution_repair"]["error"]
    assert requests[3]["execution_repair"]["previous_code"] == incomplete.strip()
    archived = result["attempts"][3]["previous_output_archives"]
    original = next(item for item in archived if item["path"].endswith("totals.json"))
    assert json.loads(Path(original["archive"]).read_text())["total"] == 2
    assert json.loads((tmp_path / "execution_attempts/03/contract_review.json").read_text())["status"] == "needs_repair"


@pytest.mark.parametrize("early_review", [True, False])
def test_extra_attempt_requires_first_review_at_regular_limit(tmp_path, early_review):
    broken = "raise ValueError('execution failed')"
    codes = [program("sum(values[:1])") if early_review else broken, broken, broken]
    result, requests, reviews = reviewed_sequence(
        tmp_path, codes, max_attempts=3, first_review_repair=True)
    assert not result["ok"] and len(requests) == 3
    assert len(reviews) == int(early_review)
    assert not any(a.get("first_review_repair") for a in result["attempts"])


def test_late_review_does_not_extend_default_budget(tmp_path):
    broken = "raise ValueError('execution failed')"
    result, requests, _ = reviewed_sequence(
        tmp_path, [broken, broken, program("sum(values[:1])")], max_attempts=3)
    assert not result["ok"] and len(requests) == 3


def test_one_attempt_without_earlier_failure_cannot_extend_budget(tmp_path):
    result, requests, _ = reviewed_sequence(
        tmp_path, [program("sum(values[:1])")], max_attempts=1,
        first_review_repair=True)
    assert not result["ok"] and len(requests) == 1


def test_readonly_violation_cannot_authorize_extra_attempt(tmp_path):
    original = tmp_path / "original.csv"
    original.write_text("value\n5\n")
    changed = program("sum(values[:1])") + "Path('original.csv').write_text('changed')\n"
    result, requests, _ = reviewed_sequence(
        tmp_path, ["raise ValueError('execution failed')", changed],
        first_review_repair=True, readonly_outputs=[original])
    assert not result["ok"] and len(requests) == 2
    assert original.read_text() == "value\n5\n"
    assert result["attempts"][-1]["restored_output_paths"] == [str(original)]
    assert "first_review_repair" not in result["attempts"][-1]


def test_persisted_first_review_cannot_grant_another_allowance(tmp_path):
    incomplete = program("sum(values[:1])")
    first, _, _ = reviewed_sequence(tmp_path, [incomplete], max_attempts=1)
    original = (tmp_path / "execution_attempts/01/execution.json").read_bytes()
    assert not first["ok"]
    result, requests, _ = reviewed_sequence(
        tmp_path, ["raise ValueError('execution failed')", incomplete],
        first_review_repair=True)
    assert not result["ok"] and len(requests) == 2
    assert [a["attempt"] for a in result["attempts"]] == [2, 3]
    assert (tmp_path / "execution_attempts/01/execution.json").read_bytes() == original


def test_persisted_execution_failures_allow_late_first_review_repair(tmp_path):
    broken = "raise ValueError('execution failed')"
    reviewed_sequence(tmp_path, [broken, broken])
    result, requests, reviews = reviewed_sequence(
        tmp_path, [program("sum(values[:1])"), program("sum(values)")],
        max_attempts=1, first_review_repair=True)
    assert result["ok"] and len(requests) == len(reviews) == 2
    assert [a["attempt"] for a in result["attempts"]] == [3, 4]


def test_incomplete_persisted_history_cannot_extend_budget(tmp_path):
    prior = tmp_path / "execution_attempts/01"
    prior.mkdir(parents=True)
    result, requests, _ = reviewed_sequence(
        tmp_path, ["raise ValueError('execution failed')", program("sum(values[:1])")],
        first_review_repair=True)
    assert not result["ok"] and len(requests) == 2


@pytest.mark.real_runtime
def test_requested_delivery_uses_late_review_repair_without_bypassing_review(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    intake = tmp_path / "artifacts/intake"
    intake.mkdir(parents=True)
    (intake / "structured_goal.json").write_text("{}")
    output = tmp_path / "totals.json"
    requests, reviews = [], []

    def generate(payload):
        requests.append(payload)
        if len(requests) < 3:
            return {"python_code": "raise ValueError('execution failed')"}
        expression = "sum(values[:1])" if len(requests) == 3 else "sum(values)"
        code = program(expression).replace("Path('totals.json')", f"Path({str(output)!r})")
        return {"python_code": code}

    def review(payload):
        reviews.append(payload)
        total = json.loads(output.read_text())["total"]
        return {"program_sha256": payload["program_sha256"],
                "checks": [satisfied()] if total == 5 else [finding()]}

    monkeypatch.setattr(pipeline, "_run_sandbox_analysis_sync", generate)
    monkeypatch.setattr(pipeline, "_run_analysis_contract_review_sync", review)
    state = {"request": REQUEST, "request_envelope": {
        "schema_version": "v3", "tasks": [{"task_id": "sum", "task_type": "sandbox_analysis"}],
        "final_outputs": [{"path": str(output), "kind": "custom", "task_id": "sum"}]}}
    result = pipeline._materialize_requested_final_outputs(state, str(tmp_path))
    assert result["ok"] and json.loads(output.read_text())["total"] == 5
    assert len(requests) == 4 and len(reviews) == 2
    assert len(result["execution_attempts"]) == 4
    # Rechecking successful delivery neither rewrites the files nor remints a budget.
    assert pipeline._materialize_requested_final_outputs(state, str(tmp_path))["already_present"]
    assert len(requests) == 4


def test_quoted_and_unquoted_findings_are_kept_separate():
    code = "print(1)"
    result = assess_review({"program_sha256": program_digest(code),
                            "checks": [finding(), finding("Invent a new criterion.")]}, code, REQUEST)
    assert result["status"] == "needs_repair"
    assert len(result["findings"]) == len(result["rejected_feedback"]) == 1


def test_empty_audit_cannot_be_recorded_as_a_completed_review():
    code = "print(1)"
    result = assess_review({"program_sha256": program_digest(code), "checks": []}, code, REQUEST)
    assert result["status"] == "unverified" and not result["findings"]


def test_unverified_requirement_is_not_converted_to_a_violation():
    code = "print(1)"
    check = {**satisfied(), "status": "unverified",
             "implementation_evidence": "A bounded preview cannot establish full row coverage."}
    result = assess_review({"program_sha256": program_digest(code), "checks": [check]}, code, REQUEST)
    assert result["status"] == "unverified" and not result["findings"]
    assert result["checks"] == [check]


@pytest.mark.real_runtime
def test_production_review_receives_actual_bounded_output_preview(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    output = tmp_path / "totals.csv"
    output.write_text("total\n5\n")
    captured = []
    async def review(payload):
        captured.append(payload)
        return {"program_sha256": payload["program_sha256"], "checks": [satisfied()]}
    monkeypatch.setattr(pipeline, "_run_analysis_contract_review_agent_once", review)
    code = "print(5)"
    result = pipeline._run_analysis_contract_review_sync({
        "program": code, "program_sha256": program_digest(code),
        "request_text": REQUEST, "output_paths": [str(output)]})
    assert result["checks"][0]["status"] == "satisfied"
    entries = captured[0]["file_previews"]["entries"]
    assert any(e.get("content") == "total\n5\n" and not e["preview_truncated"] for e in entries)
