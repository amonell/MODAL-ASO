"""Evidence-boundary policy for interactive evaluation clarifications.

Evaluation answers are inputs to the system under test.  Keeping their provenance
separate from grader-owned fixtures prevents an answer key from becoming hidden
prompt context while still permitting a blinded participant and independently
sourced biological facts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse


BLINDED_USER_PREFERENCE = "blinded_user_preference"
INDEPENDENT_BIOLOGICAL_SOURCE = "independent_biological_source"
RUNTIME_SOURCE = "runtime_source"
DIRECT_EVAL_REGRESSION = "direct_eval_regression"
UNCLEAR_SOURCE = "unclear"

VALID_CLARIFICATION_SOURCE_CLASSES = frozenset(
    {
        BLINDED_USER_PREFERENCE,
        INDEPENDENT_BIOLOGICAL_SOURCE,
        RUNTIME_SOURCE,
        DIRECT_EVAL_REGRESSION,
        UNCLEAR_SOURCE,
    }
)

_EVALUATION_PATH_COMPONENTS = frozenset(
    {
        "eval",
        "evals",
        "evaluation",
        "evaluations",
        "fixture",
        "fixtures",
        "gold",
        "regression",
        "regressions",
        "test",
        "tests",
    }
)
_ANSWER_KEY_TOKENS = frozenset(
    {
        "answerkey",
        "expectedoutput",
        "goldset",
        "goldstandard",
        "regressioncase",
    }
)
_RUNTIME_EVIDENCE_REFS = frozenset({"runtime:reference_data_root"})
_INDEPENDENT_IDENTIFIER_SCHEMES = frozenset(
    {"doi", "ensembl", "ncbi", "pmc", "pmid", "refseq"}
)
_INDEPENDENT_HOST_SUFFIXES = (
    "doi.org",
    "ensembl.org",
    "ncbi.nlm.nih.gov",
)


def _identifier_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for character in value.casefold():
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _reference_path_components(reference: str) -> tuple[str, ...]:
    parsed = urlparse(reference)
    path_text = unquote(parsed.path or reference.split("#", 1)[0])
    components = [
        component.casefold()
        for component in PurePosixPath(path_text.replace("\\", "/")).parts
        if component not in {"", "/", ".", ".."}
    ]
    if parsed.fragment:
        components.extend(_identifier_tokens(unquote(parsed.fragment)))
    return tuple(components)


def _is_evaluation_reference(reference: str) -> bool:
    components = _reference_path_components(reference)
    component_tokens = {
        token for component in components for token in _identifier_tokens(component)
    }
    if component_tokens.intersection(_EVALUATION_PATH_COMPONENTS):
        return True
    compact_tokens = set(component_tokens)
    if compact_tokens.intersection(_ANSWER_KEY_TOKENS):
        return True
    return (
        bool({"benchmark", "benchmarking"}.intersection(compact_tokens))
        and bool({"prompt", "prompts"}.intersection(compact_tokens))
    ) or {"evaluation", "criteria"}.issubset(compact_tokens)


def _is_independent_url(reference: str) -> bool:
    parsed = urlparse(reference)
    scheme = parsed.scheme.casefold()
    if scheme in _INDEPENDENT_IDENTIFIER_SCHEMES:
        return True
    if scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").casefold().rstrip(".")
    return any(
        host == suffix or host.endswith(f".{suffix}")
        for suffix in _INDEPENDENT_HOST_SUFFIXES
    )


def classify_clarification_evidence_ref(
    reference: str,
    *,
    declared_class: str | None = None,
) -> str:
    """Classify one evidence reference without consulting case gold."""

    normalized = str(reference or "").strip()
    declared = str(declared_class or "").strip().casefold()
    if not normalized:
        return UNCLEAR_SOURCE
    if normalized.startswith("simulated_user_ledger:"):
        return BLINDED_USER_PREFERENCE
    if normalized in _RUNTIME_EVIDENCE_REFS:
        return RUNTIME_SOURCE
    if _is_evaluation_reference(normalized):
        return DIRECT_EVAL_REGRESSION
    if _is_independent_url(normalized):
        return INDEPENDENT_BIOLOGICAL_SOURCE
    if declared == INDEPENDENT_BIOLOGICAL_SOURCE:
        # Local papers and curated reference artifacts cannot be identified as
        # independent from their filename alone.  Their declaration is explicit
        # corpus provenance, and evaluation-owned paths were rejected above.
        return INDEPENDENT_BIOLOGICAL_SOURCE
    return UNCLEAR_SOURCE


def classify_clarification_evidence_refs(
    references: Iterable[str],
    *,
    declared_class: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed admission decision for one complete answer."""

    refs = tuple(
        dict.fromkeys(str(item).strip() for item in references if str(item).strip())
    )
    declared = str(declared_class or "").strip().casefold() or None
    if declared is not None and declared not in VALID_CLARIFICATION_SOURCE_CLASSES:
        return {
            "source_class": UNCLEAR_SOURCE,
            "admissible": False,
            "reason": "unsupported_declared_source_class",
            "evidence_refs": list(refs),
        }
    observed = {
        classify_clarification_evidence_ref(item, declared_class=declared)
        for item in refs
    }
    if not refs:
        observed = {UNCLEAR_SOURCE}
    if DIRECT_EVAL_REGRESSION in observed:
        source_class = DIRECT_EVAL_REGRESSION
        reason = "grader_or_regression_evidence_is_not_answer_context"
    elif UNCLEAR_SOURCE in observed or len(observed) != 1:
        source_class = UNCLEAR_SOURCE
        reason = "evidence_source_is_unverified_or_mixed"
    else:
        source_class = next(iter(observed))
        reason = ""
    if declared is not None and source_class != declared:
        return {
            "source_class": source_class,
            "admissible": False,
            "reason": "declared_source_class_does_not_match_evidence",
            "evidence_refs": list(refs),
        }
    admissible = source_class in {
        BLINDED_USER_PREFERENCE,
        INDEPENDENT_BIOLOGICAL_SOURCE,
        RUNTIME_SOURCE,
    }
    return {
        "source_class": source_class,
        "admissible": admissible,
        "reason": reason if not admissible else "",
        "evidence_refs": list(refs),
    }


def clarification_rule_admission(
    rule: Mapping[str, Any],
    *,
    allow_blinded_ledger: bool = False,
) -> dict[str, Any]:
    """Classify a pre-authored rule at the last boundary before model context."""

    evidence_refs = rule.get("evidence_refs")
    refs = (
        [str(item) for item in evidence_refs]
        if isinstance(evidence_refs, Sequence)
        and not isinstance(evidence_refs, (str, bytes, bytearray))
        else []
    )
    admission = classify_clarification_evidence_refs(
        refs,
        declared_class=str(rule.get("source_class") or "") or None,
    )
    if (
        admission["source_class"] == BLINDED_USER_PREFERENCE
        and not allow_blinded_ledger
    ):
        admission = {
            **admission,
            "admissible": False,
            "reason": "authored_rules_cannot_impersonate_the_blinded_participant",
        }
    return admission


def clarification_assistance_summary(
    answers: Iterable[Mapping[str, Any]],
    *,
    simulator_traces: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Summarize assistance provenance for one scored attempt."""

    applied = [dict(item) for item in answers if isinstance(item, Mapping)]
    source_classes: list[str] = []
    inadmissible_count = 0
    for answer in applied:
        admission = clarification_rule_admission(answer, allow_blinded_ledger=True)
        source_class = str(admission["source_class"])
        if source_class not in source_classes:
            source_classes.append(source_class)
        inadmissible_count += int(not admission["admissible"])
    traces = [item for item in simulator_traces if isinstance(item, Mapping)]
    has_simulator_activity = any(
        int(item.get("call_count") or 0) > 0 or bool(item) for item in traces
    )
    if DIRECT_EVAL_REGRESSION in source_classes:
        classification = DIRECT_EVAL_REGRESSION
    elif UNCLEAR_SOURCE in source_classes:
        classification = "unverified_source_assisted"
    elif (
        INDEPENDENT_BIOLOGICAL_SOURCE in source_classes
        or RUNTIME_SOURCE in source_classes
    ):
        classification = "independent_source_assisted"
    elif BLINDED_USER_PREFERENCE in source_classes:
        classification = "blinded_user_preference_assisted"
    elif has_simulator_activity:
        classification = "simulator_trace_no_applied_answer"
    else:
        classification = "unassisted"
    return {
        "applied": bool(applied),
        "answer_count": len(applied),
        "source_classes": sorted(source_classes),
        "classification": classification,
        "fair_admissible": inadmissible_count == 0,
        "inadmissible_answer_count": inadmissible_count,
        "simulator_trace_without_applied_answer": has_simulator_activity
        and not applied,
    }


__all__ = [
    "BLINDED_USER_PREFERENCE",
    "DIRECT_EVAL_REGRESSION",
    "INDEPENDENT_BIOLOGICAL_SOURCE",
    "RUNTIME_SOURCE",
    "UNCLEAR_SOURCE",
    "VALID_CLARIFICATION_SOURCE_CLASSES",
    "clarification_assistance_summary",
    "clarification_rule_admission",
    "classify_clarification_evidence_ref",
    "classify_clarification_evidence_refs",
]
