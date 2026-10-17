"""Compose mixed sandbox/design results from completed, persisted evidence."""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def _cell(value: Any) -> str:
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def _artifact_rows(payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    artifacts = payload.get("generated_artifacts") if isinstance(payload.get("generated_artifacts"), dict) else {}
    panel_path = artifacts.get("design_panel_csv")
    if isinstance(panel_path, str) and panel_path:
        try:
            with Path(panel_path).open(newline="") as stream:
                panel_rows = list(csv.DictReader(stream))
            if panel_rows and all(row.get("antisense_5to3") and row.get("target_5to3") for row in panel_rows):
                generated_path = artifacts.get("generated_aso_set_json")
                try:
                    generated = json.loads(Path(generated_path).read_text()) if isinstance(generated_path, str) else {}
                except (OSError, ValueError):
                    generated = {}
                by_aso = {
                    row.get("synthesized_antisense_sequence"): row
                    for row in generated.get("rows", []) if isinstance(row, dict)
                } if isinstance(generated, dict) else {}
                for row in panel_rows:
                    backing = by_aso.get(row["antisense_5to3"])
                    if isinstance(backing, dict):
                        row.setdefault("off_target_count_status", backing.get("off_target_count_status"))
                return panel_rows
        except (OSError, csv.Error):
            pass
    path = artifacts.get("generated_aso_set_json")
    if not isinstance(path, str) or not path:
        return None
    try:
        artifact = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    rows = artifact.get("rows") if isinstance(artifact, dict) else None
    return rows if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else None


def _sandbox_summary(payload: Dict[str, Any]) -> str:
    templates = payload.get("original_report_templates")
    if isinstance(templates, dict):
        for body in templates.values():
            if not isinstance(body, str) or not body.strip():
                continue
            lines = [line for line in body.splitlines() if not line.strip().startswith("{{table:")]
            return "\n".join(lines).strip()
    counts = payload.get("eligibility_counts") if isinstance(payload.get("eligibility_counts"), dict) else {}
    if counts:
        return "\n".join(f"- {_cell(key)}: {_cell(value)}" for key, value in counts.items())
    return ""


def grounded_mixed_report(completed_tasks: List[Dict[str, Any]],
                          blocked_tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a report only when every mixed task has source-backed content."""
    types = {str(task.get("task_type") or "") for task in completed_tasks if isinstance(task, dict)}
    if types != {"sandbox_analysis", "design_new_asos"}:
        return None
    rendered = []
    evidence_refs = []
    limitations = []
    for task in completed_tasks:
        payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
        label = str(task.get("label") or task.get("task_id") or "Completed task")
        if task.get("task_type") == "sandbox_analysis":
            summary = _sandbox_summary(payload)
            if not summary:
                return None
            rendered.extend([f"## {_cell(label)}", "", summary])
            tables = payload.get("report_tables") if isinstance(payload.get("report_tables"), list) else []
            for table in tables:
                if isinstance(table, dict) and table.get("source_path"):
                    rendered.append(f"\nFull row table: `{table['source_path']}`.")
                    evidence_refs.append(str(table["source_path"]))
        else:
            rows = _artifact_rows(payload)
            if rows is None:
                return None
            outcome = payload.get("design_outcome") if isinstance(payload.get("design_outcome"), dict) else {}
            rendered.extend([f"## {_cell(label)}", "",
                             f"Design outcome: {_cell(outcome.get('outcome_status'))}; "
                             f"{len(rows)} selected candidate row(s).", "",
                             "| Candidate | Physical ASO 5′→3′ | Target binding site 5′→3′ | Site | Genomic interval | Chemistry class | Selection rank | Exact off-target count | Count unit | Mismatch budget | Count status |",
                             "|---|---|---|---|---|---|---:|---:|---|---:|---|"])
            for row in rows:
                physical = row.get("antisense_5to3") or row.get("synthesized_antisense_sequence")
                if not physical and row.get("sequence_role") == "synthesized_antisense_aso":
                    physical = row.get("sequence")
                target = row.get("target_5to3") or row.get("target_binding_sequence")
                if not physical:
                    return None
                chrom = row.get("genomic_chrom") or row.get("chrom")
                start = row.get("genomic_start_1based") or row.get("start")
                end = row.get("genomic_end_1based") or row.get("end")
                strand = row.get("target_genomic_strand") or row.get("strand")
                interval = f"{chrom}:{start}–{end} ({strand})"
                transcript = row.get("transcript_accession") or row.get("site_label")
                cdna_start = row.get("transcript_cdna_start")
                cdna_end = row.get("transcript_cdna_end")
                site = f"{transcript} cDNA {cdna_start}–{cdna_end}" if cdna_start and cdna_end else transcript
                values = (row.get("candidate") or row.get("site_label") or row.get("program_id") or row.get("job_label"),
                          physical, target, site, interval,
                          row.get("chemistry_description") or row.get("source_modality"),
                          row.get("native_rank") or row.get("selection_rank"),
                          row.get("off_target_exact_count") if row.get("off_target_exact_count") is not None else row.get("Off_target_exact_count"),
                          row.get("off_target_exact_count_unit"),
                          row.get("off_target_mismatch_budget") if row.get("off_target_mismatch_budget") is not None else row.get("Off_target_mismatch_budget"),
                          row.get("off_target_count_status"))
                rendered.append("| " + " | ".join(_cell(value) for value in values) + " |")
            rendered.extend(["", "Off-target counts apply only to each candidate's recorded search universe and mismatch budget. "
                             "A complete count within that scope is not a genome-wide safety or efficacy result."])
            artifacts = payload.get("generated_artifacts") if isinstance(payload.get("generated_artifacts"), dict) else {}
            for key in ("design_panel_csv", "generated_aso_set_json"):
                if artifacts.get(key):
                    evidence_refs.append(str(artifacts[key]))
            if outcome.get("outcome_status") != "completed_with_candidates":
                limitations.append(f"{label}: {outcome.get('outcome_status') or 'unknown design outcome'}")
    if blocked_tasks:
        rendered.extend(["", "## Blocked tasks"])
        for task in blocked_tasks:
            if isinstance(task, dict):
                rendered.append(f"- {_cell(task.get('label') or task.get('task_id'))}: {_cell(task.get('reason'))}")
        limitations.append("Some requested tasks remain blocked.")
    return {
        "answer_summary": f"Integrated {len(completed_tasks)} completed task results with source-backed candidate records.",
        "markdown": "\n".join(rendered).strip(), "assumptions": [],
        "limitations": limitations + ["Sequence and screening evidence is exploratory; new candidate efficacy is unmeasured."],
        "evidence_refs": evidence_refs, "suggested_followups": [],
        "source_backed_mixed_report": True,
    }
