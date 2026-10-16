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
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        key: {"type": "string", "minLength": 1}
                        for key in ("requirement_quote", "problem", "implementation_evidence", "suggested_fix")
                    },
                    "required": ["requirement_quote", "problem", "implementation_evidence", "suggested_fix"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["program_sha256", "findings"],
        "additionalProperties": False,
    }


def assess_review(review: dict, code: str, request_text: str) -> dict:
    """Unbound model feedback cannot add or change the user's requirements."""
    from jsonschema import Draft202012Validator

    errors = list(Draft202012Validator(review_schema(code)).iter_errors(review))
    if errors:
        return {"status": "unverified", "findings": [],
                "rejected_feedback": [error.message for error in errors]}
    accepted, rejected = [], []
    for finding in review["findings"]:
        quote = finding["requirement_quote"]
        if not quote.strip() or quote not in request_text:
            rejected.append(finding)
        else:
            accepted.append(finding)
    return {
        "status": ("needs_repair" if accepted else
                   "unverified" if rejected else "no_concrete_violation_identified"),
        "findings": accepted,
        "rejected_feedback": rejected,
    }
