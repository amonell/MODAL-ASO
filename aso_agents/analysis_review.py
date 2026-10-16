"""Bind analysis-completeness feedback to the actual program and user request."""
from __future__ import annotations

import hashlib


def program_digest(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def review_schema(code: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "program_sha256": {"type": "string", "enum": [program_digest(code)]},
            "checks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement_quote": {"type": "string", "minLength": 1},
                        "obligation": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "enum": ["satisfied", "violated", "unverified"]},
                        "implementation_evidence": {"type": "string", "minLength": 1},
                        "problem": {"type": ["string", "null"]},
                        "suggested_fix": {"type": ["string", "null"]},
                    },
                    "required": ["requirement_quote", "obligation", "status", "implementation_evidence", "problem", "suggested_fix"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["program_sha256", "checks"],
        "additionalProperties": False,
    }


def assess_review(review: dict, code: str, request_text: str) -> dict:
    """Unbound model feedback cannot add or change the user's requirements."""
    from jsonschema import Draft202012Validator

    errors = list(Draft202012Validator(review_schema(code)).iter_errors(review))
    if errors:
        return {"status": "unverified", "findings": [],
                "rejected_feedback": [error.message for error in errors]}
    accepted, rejected, checks = [], [], []
    for check in review["checks"]:
        quote = check["requirement_quote"]
        if not quote.strip() or quote not in request_text:
            rejected.append(check)
            continue
        if check["status"] == "violated":
            if not all(isinstance(check[key], str) and check[key].strip()
                       for key in ("problem", "suggested_fix")):
                rejected.append(check)
                continue
            accepted.append({key: value for key, value in check.items() if key != "status"})
        checks.append(check)
    return {
        "status": ("needs_repair" if accepted else
                   "unverified" if rejected or any(check["status"] == "unverified" for check in checks)
                   else "no_concrete_violation_identified"),
        "checks": checks,
        "findings": accepted,
        "rejected_feedback": rejected,
    }
