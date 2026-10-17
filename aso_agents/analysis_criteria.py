"""Keep each required criterion's evidence separate from overall eligibility."""
from collections.abc import Mapping

from numpy import bool_ as numpy_bool


def summarize_required_criteria(criteria: Mapping[str, bool | None]) -> dict:
    """Summarize an all-required conjunction without inventing failure reasons.

    Each named criterion must have its own observed Boolean result, or None
    when unverified. This helper does not evaluate scientific predicates or
    turn alternative conditions into mandatory ones. An empty specification
    is indeterminate, not evidence of eligibility. Input order is preserved.
    """
    if not isinstance(criteria, Mapping):
        raise TypeError("Required criteria must be a mapping of names to Boolean results or None")
    observed = dict(criteria)
    for name, value in observed.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each criterion needs a nonempty string name")
        if value is not None and not isinstance(value, (bool, numpy_bool)):
            raise TypeError(f"Criterion {name!r} must be True, False or None; do not coerce missing data")
        if isinstance(value, numpy_bool):
            observed[name] = bool(value)
    passed = [name for name, value in observed.items() if value is True]
    failed = [name for name, value in observed.items() if value is False]
    unverified = [name for name, value in observed.items() if value is None]
    status = "failed" if failed else "indeterminate" if unverified or not observed else "passed"
    return {
        "status": status,
        "criteria": observed,
        "passed_criteria": passed,
        "failed_criteria": failed,
        "unverified_criteria": unverified,
    }
