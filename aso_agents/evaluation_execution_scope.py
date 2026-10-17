"""Bind omitted evaluation-screening intent to explicit, task-scoped user evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def scope_input(*, request, task, context, clarifications=()):
    evidence = [{"id": "request", "text": request}]
    for index, answer in enumerate(clarifications):
        if answer.get("answer_source") == "external_user" and str(answer.get("answer") or "").strip():
            evidence.append({"id": f"clarification_{index + 1}",
                             "question": str(answer.get("question") or ""),
                             "text": str(answer["answer"])})
    payload = {"user_evidence": evidence, "task": deepcopy(task),
               "current_context": {key: deepcopy(context[key]) for key in (
                   "gene_symbol", "gene_id", "species", "transcript_id", "intent",
                   "target_context", "provided_aso_files", "artifact_refs", "source_sets",
                   "request_summary") if key in context}}
    payload["input_sha256"] = hashlib.sha256(_encoded(payload).encode()).hexdigest()
    return payload


def scope_schema(payload):
    return {"type": "object", "additionalProperties": False,
            "properties": {
                "input_sha256": {"type": "string", "enum": [payload["input_sha256"]]},
                "decision": {"type": "string", "enum": ["enabled", "disabled", "unspecified"]},
                "evidence_ref": {"type": ["string", "null"], "enum": [None, *[e["id"] for e in payload["user_evidence"]]]},
                "evidence_quote": {"type": ["string", "null"]},
                "reason": {"type": "string", "minLength": 1}},
            "required": ["input_sha256", "decision", "evidence_ref", "evidence_quote", "reason"]}


def scope_errors(result, payload):
    from jsonschema import Draft202012Validator
    errors = [error.message for error in Draft202012Validator(scope_schema(payload)).iter_errors(result)]
    if errors:
        return errors
    if result["decision"] == "unspecified":
        if result["evidence_ref"] is not None or result["evidence_quote"] is not None:
            errors.append("An unspecified policy must have null evidence_ref and evidence_quote")
    else:
        sources = {e["id"]: e["text"] for e in payload["user_evidence"]}
        quote = result["evidence_quote"]
        source = sources.get(result["evidence_ref"])
        if not isinstance(quote, str) or not quote.strip() or not source or quote not in source:
            errors.append("An explicit decision requires an exact nonempty quotation from the selected user evidence")
    return errors


def complete_evaluation_scope(*, request, task, context, directory, classify, clarifications=()):
    """Fill a missing policy once; never overwrite a declared one or guess on failure.

    Only user evidence can authorize an inferred policy. An unspecified decision
    retains the ordinary runtime default. Request/task-bound receipts avoid
    repeated classification on resume and preserve rejected first results.
    """
    completed = deepcopy(context)
    if "off_target_screening" in completed:
        if completed["off_target_screening"] not in ("enabled", "disabled"):
            return {"error": "invalid_declared_off_target_screening_policy"}
        return {"context": completed, "status": "declared", "receipt_path": None}
    if not str(request or "").strip() and not clarifications:
        return {"context": completed, "status": "unspecified", "receipt_path": None}
    payload = scope_input(request=str(request or ""), task=task, context=context,
                          clarifications=clarifications)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / (payload["input_sha256"] + ".json")
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
            if not isinstance(receipt, dict) or receipt.get("input") != payload:
                raise ValueError("Stored scope receipt does not match this request and task")
            result = receipt["raw_result"]
        except (OSError, ValueError, KeyError) as exc:
            return {"error": "evaluation_screening_scope_unverified", "reason": str(exc),
                    "receipt_path": str(receipt_path)}
    else:
        try:
            result = classify(deepcopy(payload))
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        errors = scope_errors(result, payload)
        receipt = {"input": payload, "raw_result": result,
                   "validation_errors": errors, "accepted": not errors}
        with receipt_path.open("x") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    errors = scope_errors(result, payload)
    if errors:
        return {"error": "evaluation_screening_scope_unverified", "validation_errors": errors,
                "receipt_path": str(receipt_path)}
    decision = result["decision"]
    if decision != "unspecified":
        completed["off_target_screening"] = decision
    return {"context": completed, "status": decision, "receipt_path": str(receipt_path)}
