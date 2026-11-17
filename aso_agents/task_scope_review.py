"""Bind requested operations to the existing evidence-based graph review.

The model interprets scope; the host requires a judgment for every proposed task
and turns discrepancies into ordinary graph repairs. No request keywords, gene
names or reference-availability heuristics select a route here. Review fields
are transient and do not change the persisted task or scientific goal schemas.
"""
from copy import deepcopy
import json

from .goal_contract import CANONICAL_TASK_TYPES


TASK_SCOPE_CONTRACT = (
    "Preserve the requested operation and its intended result when selecting an executor. "
    "Separate the outcome the user asks you to produce from a method used to obtain it. "
    "Tiling, mapping, filtering, ranking, and reporting can be methods within a larger "
    "operation; completing a method alone does not complete that operation. Determine "
    "scope from original request/source evidence and user clarifications, not from the "
    "proposed task label, summary, assumptions or convenient available files. Available "
    "references resolve inputs; they do not waive requested work. Missing inputs retain "
    "the intended operation and its existing clarification/resolution path; they do not "
    "authorize substitution of a narrower computation.\n"
    "A request to design new ASOs toward a biological effect belongs in design_new_asos, "
    "including when the user specifies the target interval and a tiling method. Preserve "
    "that objective for the native design stages without promising experimental efficacy "
    "or adding unrequested analyses. A request explicitly limited to enumerating or "
    "transforming sequences under supplied rules can remain sandbox_analysis, including "
    "when it produces oligonucleotide strings or mentions biological motivation. The "
    "distinction is the requested result, not particular vocabulary, target metadata, or "
    "whether the reference happens to be installed. A planner-authored 'baseline' assumption "
    "does not establish that the user limited the task to sequence arithmetic. Specifying "
    "coordinates, length, chemistry, GC bounds or tiling alone is not a calculation-only "
    "limitation. Apply this "
    "same scope rule to evaluation, comparison, artifact analysis, questions, recorded "
    "filter traces, reports and mixed requests: retain every requested operation and "
    "deliverable, while preserving legitimate calculations and explicit exclusions.\n"
    "When task_scope_checks is present in the response schema, check every listed task "
    "against original evidence. requested_task_type is the canonical operation required "
    "by that evidence, independently of the proposed type; use null only for an "
    "unsupported task. scope_status is preserved only when the proposed operation and "
    "declared scope cover the requested result, reduced when work is lost, expanded "
    "when extra work is imposed, and unsupported when the task is not supported by the "
    "request. Cite an exact evidence_span_id (or evidence_quote when no spans are offered) "
    "and explain the result/method distinction "
    "when relevant. Do not treat native structured-goal completeness as scope loss; "
    "later stages synthesize those fields. The host converts discrepant scope checks "
    "into graph-repair issues. Still report missing operations and dependencies in "
    "issues: checking proposed tasks does not establish complete request coverage.\n"
)


def declared_task_scopes(prompt):
    """Read protocol fields, never protocol-looking text inside user evidence."""
    seen = set()
    while isinstance(prompt, str) and prompt not in seen:
        seen.add(prompt)
        try:
            payload = json.loads(prompt.partition("\n\n")[2])
        except (ValueError, TypeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        envelope = payload.get("proposed_request_envelope")
        if isinstance(envelope, dict):
            return {
                f"/tasks/{index}": {
                    "task_id": task.get("task_id"),
                    "task_type": task.get("task_type"),
                }
                for index, task in enumerate(envelope.get("tasks") or [])
                if isinstance(task, dict)
            }
        prompt = payload.get("original_review_input")
    return {}


def with_task_scope_schema(schema, tasks):
    if not tasks:
        return schema
    result = deepcopy(schema)
    evidence = result["properties"]["issues"]["items"]["properties"]
    properties = {}
    for pointer, task in tasks.items():
        citation = "evidence_span_id" if "evidence_span_id" in evidence else "evidence_quote"
        fields = {key: deepcopy(evidence[key])
                  for key in ("evidence_refs", citation, "reason")}
        fields.update({
            "requested_task_type": {
                "type": ["string", "null"], "enum": [*sorted(CANONICAL_TASK_TYPES), None],
            },
            "scope_status": {
                "type": "string", "enum": ["preserved", "reduced", "expanded", "unsupported"],
            },
        })
        properties[pointer] = {
            "type": "object", "additionalProperties": False,
            "description": "Review the requested result independently of this proposed operation: "
                           + json.dumps(task, ensure_ascii=False),
            "properties": fields, "required": list(fields),
        }
    result["properties"]["task_scope_checks"] = {
        "type": "object", "additionalProperties": False,
        "properties": properties, "required": list(properties),
    }
    result["required"] = [*result["required"], "task_scope_checks"]
    return result


def has_scope_revision(review, tasks):
    if not isinstance(review, dict):
        return False
    checks = review.get("task_scope_checks") or {}
    if not isinstance(checks, dict):
        return False
    return any(
        isinstance(check, dict) and (
            check.get("scope_status") != "preserved"
            or check.get("requested_task_type") is None
            or (pointer in tasks and check.get("requested_task_type") != tasks[pointer]["task_type"])
        ) for pointer, check in checks.items()
    )


def scope_evidence_errors(review, evidence_sources, span_map):
    """Verify positive judgments too, including short requests without span IDs."""
    sources = {item["evidence_id"]: item.get("text", "") for item in evidence_sources}
    errors = []
    for pointer, check in (review.get("task_scope_checks") or {}).items():
        span = span_map.get(check.get("evidence_span_id"))
        quote = span["text"] if span else check.get("evidence_quote", "")
        refs = check.get("evidence_refs") or []
        if (not isinstance(quote, str) or not quote.strip()
                or not any(quote in sources.get(ref, "") for ref in refs)
                or (span and span["evidence_id"] not in refs)):
            errors.append(f"$.task_scope_checks[{pointer}]: evidence could not be verified")
    return errors


def apply_task_scope_checks(review, tasks):
    """A positive review label cannot override its own cited scope discrepancies."""
    result = deepcopy(review)
    checks = result.pop("task_scope_checks", {})
    if not isinstance(checks, dict) or set(checks) != set(tasks):
        raise ValueError("Task-scope review does not cover every proposed task")
    for pointer, check in checks.items():
        task = tasks[pointer]
        requested = check["requested_task_type"]
        scope = check["scope_status"]
        if requested == task["task_type"] and requested is not None and scope == "preserved":
            continue
        if requested is None or scope == "unsupported":
            category, path = "spurious_task", pointer
        elif requested != task["task_type"]:
            category, path = "wrong_task_type", pointer + "/task_type"
        else:
            # A correctly selected operation with incomplete/extra scope needs
            # a scope repair, not another independently executable task.
            category = "missing_fact" if scope == "reduced" else "unsupported_fact"
            path = pointer
        issue = {
            "category": category,
            "task_ids": [task["task_id"]] if task["task_id"] else [],
            "path": path, "evidence_refs": deepcopy(check["evidence_refs"]),
            "evidence_quote": check["evidence_quote"],
            "reason": (
                f"The source-bound scope check requires {requested!r}; the proposal selects "
                f"{task['task_type']!r} with scope {scope!r}. {check['reason']} "
                "Preserve the requested operation and result through the existing executor; "
                "do not substitute a prerequisite calculation or add unrequested work."
            ),
        }
        # An earlier no-op issue at this path must not suppress the typed
        # discrepancy. Preserve distinct evidence; deduplicate exact repeats.
        if issue not in result["issues"]:
            result["issues"].append(issue)
    if result["issues"]:
        result["status"] = "revise"
    return result
