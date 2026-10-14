"""Render a typed metric-audit reconciliation directly from its delivered rows."""
import csv
from pathlib import Path


def canonical_audit_texts(result, texts, export_tables, views):
    payload = result.get("sandbox_result") if isinstance(result.get("sandbox_result"), dict) else result
    eligibility = payload.get("eligibility")
    claims = payload.get("claim_discrepancies")
    if not isinstance(eligibility, dict) or not isinstance(claims, dict) or not eligibility.get("criteria"):
        return texts, None
    required = {"id", "eligibility_status", "eligibility_reasons", "length_match", "claimed_gc_round3_match"}
    candidates = []
    for view in views:
        source = Path(view.get("source_path") or "")
        if not source.is_absolute():
            continue
        if str(source.resolve()) not in export_tables:
            continue
        with source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter="\t" if source.suffix == ".tsv" else ",")
            if required.issubset(reader.fieldnames or []):
                candidates.append((view, list(reader)))
    if len(candidates) != 1 or len(views) != 1:
        return texts, None  # Custom/multi-analysis narratives retain their existing path.
    view, rows = candidates[0]
    statuses = {}
    for row in rows:
        statuses.setdefault(row["eligibility_status"], []).append(row["id"])
    lines = ["# Sequence-metric audit reconciliation", "",
             f"Audited {len(rows)} supplied rows. Eligibility uses recomputed metrics and recorded missingness.",
             "", "Criteria: " + str(eligibility["criteria"]), ""]
    for status, identifiers in statuses.items():
        lines.append(f"- {status}: {len(identifiers)} row(s); " + ", ".join(identifiers) + ".")
    lines += ["", "Claim comparison: " + str(claims.get("comparison_rule") or "See the per-row comparison fields."),
              "The complete table contains each row's metrics, discrepancy flags and eligibility reasons. "
              "A claim discrepancy and eligibility are separate decisions; no illustrative row is inferred from prose.",
              "", "{{table:" + view["id"] + "}}", ""]
    for key in ("assumptions", "limitations"):
        if isinstance(payload.get(key), list):
            lines += ["## " + key.title(), ""] + ["- " + str(item) for item in payload[key]] + [""]
    canonical = "\n".join(lines)
    return {path: canonical for path in texts}, {
        "source_path": str(Path(view["source_path"]).resolve()), "row_count": len(rows),
        "status_counts": {key: len(value) for key, value in statuses.items()},
        "source": "delivered typed audit table; no new calculations",
    }
