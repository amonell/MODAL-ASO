"""Assemble completed comparison evidence without inventing task-specific synthesis."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .dependency_references import recorded_trace_markdown


def _text(value: Any) -> str:
    return str(value if value is not None else "not recorded").replace("\n", " ")


def _strings(value: Any) -> List[str]:
    return [item for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def compose_comparison_report(
    designs: List[Dict[str, Any]], comparisons: List[Dict[str, Any]],
    traces: List[Dict[str, Any]], evaluations: List[Dict[str, Any]],
    blocked: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Keep each task's identity, result and assessment separate and source-backed.

    This renderer does not calculate candidate metrics, reinterpret molecule
    orientation, pool ranks, infer efficacy or select a winning mechanism.
    Missing measurements remain unknown; an absent input is not a zero count.
    """
    if not (comparisons or traces):
        return None
    lines: List[str] = []
    refs: List[str] = []
    limitations: List[str] = []
    assumptions: List[str] = []
    outcomes: List[Dict[str, Any]] = []
    comparison_summaries: List[Dict[str, Any]] = []

    def reference(value: Any) -> None:
        if isinstance(value, str) and value.strip() and value not in refs:
            refs.append(value)

    def narrative(response: Any) -> bool:
        if not isinstance(response, dict):
            return False
        markdown = response.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            return False
        lines.extend([markdown.strip(), ""])
        assumptions.extend(_strings(response.get("assumptions")))
        limitations.extend(_strings(response.get("limitations")))
        for item in _strings(response.get("evidence_refs")):
            reference(item)
        return True

    if designs:
        lines.extend(["## Design Runs", ""])
    for payload in designs:
        label = payload.get("task_label") or payload.get("task_id") or "Design task"
        goal = payload.get("child_structured_goal")
        goal = goal if isinstance(goal, dict) else {}
        outcome = payload.get("design_outcome")
        outcome = outcome if isinstance(outcome, dict) else {}
        lines.extend([f"### {_text(label)}", ""])
        identity = [goal[key] for key in ("species", "gene_symbol", "gene_id") if goal.get(key)]
        if identity:
            lines.append("Target: " + " / ".join(_text(value) for value in identity) + ".")
        lock = goal.get("transcript_lock")
        if isinstance(lock, dict) and lock.get("value"):
            lines.append(f"Reference transcript: `{_text(lock['value'])}`.")
        for program in goal.get("design_programs") or []:
            if not isinstance(program, dict):
                continue
            details = [f"{key}: {_text(program[key])}" for key in
                       ("label", "modality", "intent", "objective", "chemistry", "desired_effect")
                       if program.get(key) is not None]
            if details:
                lines.append("- " + "; ".join(details) + ".")
        status = outcome.get("outcome_status")
        lines.append(f"Recorded design outcome: `{_text(status)}`.")
        for key, name in (("candidate_count", "Selected candidate rows"),
                          ("unique_sequence_count", "Distinct selected sequences")):
            value = outcome.get(key)
            if type(value) is int and value >= 0:
                lines.append(f"{name}: {value}.")
        for key in ("no_candidate_reason", "failure_reason"):
            if outcome.get(key):
                lines.append(_text(outcome[key]))
        missing = _strings(outcome.get("missing_required_evidence"))
        if missing:
            lines.append("Missing required evidence: " + ", ".join(missing) + ".")
        if status not in {"completed_with_candidates", "completed_no_candidates"}:
            limitations.append(f"{_text(label)}: {_text(status)}.")
        artifacts = payload.get("generated_artifacts")
        if isinstance(artifacts, dict):
            for key in ("design_panel_csv", "design_panel_md", "generated_aso_set_json"):
                if artifacts.get(key):
                    reference(artifacts[key])
                    lines.append(f"{key}: `{_text(artifacts[key])}`.")
        reference(payload.get("child_run_dir"))
        # Excerpts retain the originating task's interpretation, rather than
        # substituting a template based on the coexistence of two modalities.
        excerpt = payload.get("report_excerpt")
        if status in {"completed_with_candidates", "completed_no_candidates"} and isinstance(excerpt, str) and excerpt.strip():
            lines.extend(["", excerpt.strip()])
        lines.append("")
        outcomes.append({"task_id": payload.get("task_id"), "label": label,
                         "target": identity, "design_outcome": dict(outcome)})

    for evaluation in evaluations:
        rows = evaluation.get("provided_asos")
        if isinstance(rows, list):
            lines.extend(["## Recorded Molecule Evaluation", "",
                          f"Task: {_text(evaluation.get('task_label') or evaluation.get('task_id'))}.",
                          f"Recorded evaluation rows: {len(rows)}.", ""])
            # Evaluated molecules may be generated upstream; do not call them
            # user-provided unless their source actually establishes that.
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = row.get("name") or row.get("aso_id") or "Evaluated molecule"
                sequence = row.get("sequence") or row.get("raw_sequence")
                mapping = row.get("mapping_status")
                lines.append(f"- {_text(name)}: sequence `{_text(sequence)}`; mapping `{_text(mapping)}`.")
        narrative(evaluation.get("llm_response"))
        limitations.extend(_strings(evaluation.get("limitations")))

    if comparisons:
        lines.extend(["## ASO Set Comparisons", ""])
    for payload in comparisons:
        label = payload.get("task_label") or payload.get("task_id") or "Comparison task"
        lines.extend([f"### {_text(label)}", ""])
        assessment = payload.get("comparison_assessment") or payload.get("llm_response")
        if not narrative(assessment):
            mode = payload.get("comparison_mode")
            lines.append(f"Recorded comparison mode: `{_text(mode)}`.")
            if mode == "comparison_unavailable":
                reason = payload.get("reason") or "Required source results are unavailable."
                lines.append(_text(reason))
                limitations.append("Some requested ASO-set comparisons were unavailable because upstream sequence-bearing outputs were missing or skipped.")
            sets = payload.get("sets")
            if not isinstance(sets, list):
                sets = [payload[key] for key in ("set_a", "set_b") if isinstance(payload.get(key), dict)]
            for item in sets:
                if not isinstance(item, dict):
                    continue
                name = item.get("label") or item.get("set_id") or "Recorded set"
                lines.append(f"- {_text(name)}: source `{_text(item.get('source_type'))}`; recorded count `{_text(item.get('count'))}`.")
            pairs = payload.get("pairwise_overlaps")
            if isinstance(pairs, list):
                for pair in pairs:
                    if isinstance(pair, dict):
                        lines.append(f"- Recorded sequence overlap, {_text(pair.get('set_a_label') or pair.get('set_a_id'))} / "
                                     f"{_text(pair.get('set_b_label') or pair.get('set_b_id'))}: `{_text(pair.get('shared_count'))}`.")
            lines.append("Recorded sequence overlap is not comparative efficacy evidence; physical-molecule interpretation depends on the source sequence-role annotations.")
            limitations.append(f"{_text(label)} has no recorded mechanism-aware comparative assessment.")
        summaries = {key: payload[key] for key in ("comparison_mode", "provided_count", "generated_count", "artifact_count", "overlap_count") if key in payload}
        comparison_summaries.append({"task_id": payload.get("task_id"), **summaries})
        sets = payload.get("sets")
        if not isinstance(sets, list):
            sets = [payload[key] for key in ("set_a", "set_b") if isinstance(payload.get(key), dict)]
        for item in sets:
            if isinstance(item, dict):
                reference(item.get("artifact_path"))
        reference(payload.get("generated_reference_run_dir"))
        lines.append("")

    if traces:
        if any(not (payload.get("reference_traces") or payload.get("trace_rows")) for payload in traces):
            return None
        text = recorded_trace_markdown(traces)
        if text:
            lines.extend(["## Recorded Filter Traces", "", text, ""])
        else:
            return None
        for payload in traces:
            reference(payload.get("generated_reference_run_dir"))

    if blocked:
        lines.extend(["## Still Blocked", ""])
        for task in blocked:
            if isinstance(task, dict):
                lines.append(f"- {_text(task.get('label') or task.get('task_id'))}: {_text(task.get('reason'))}.")
        limitations.append("Some requested tasks remain blocked pending clarification.")
    skipped = sum(item["design_outcome"].get("outcome_status") == "skipped" for item in outcomes)
    if skipped:
        limitations.append(f"{skipped} design run(s) were skipped before candidate generation.")
    return {
        "answer_summary": f"Integrated {len(designs)} design result(s), {len(comparisons)} comparison result(s), and {len(traces)} filter-trace result(s) from recorded task evidence.",
        "markdown": "\n".join(lines).strip(),
        "assumptions": list(dict.fromkeys(assumptions)),
        "limitations": list(dict.fromkeys(limitations)),
        "evidence_refs": refs, "suggested_followups": [],
        "source_backed_composite_report": True,
        "deterministic_summary": {"design_task_count": len(designs),
                                  "comparison_task_count": len(comparisons),
                                  "evaluation_task_count": len(evaluations),
                                  "trace_task_count": len(traces),
                                  "design_outcomes": outcomes,
                                  "comparison_summaries": comparison_summaries},
    }
