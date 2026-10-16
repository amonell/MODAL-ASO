"""Expose complete producer-bound design evidence to custom output writers."""
import csv
import hashlib
import json
from pathlib import Path


def design_export_evidence(task_id, producer, artifact_path):
    """Inventory recorded stage sources without filtering or rewriting rows.

    The filter trace is a provenance index, not a replacement for its source
    tables: a trace may group sequences that occur at multiple physical sites.
    """
    root = Path(producer).resolve()
    records, unavailable = {}, []

    def add(path, role):
        candidate = Path(path)
        candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not candidate.is_relative_to(root):
            unavailable.append({"path": str(path), "role": role, "reason": "outside_recorded_producer"})
            return None
        if not candidate.is_file():
            unavailable.append({"path": str(candidate), "role": role, "reason": "source_missing"})
            return None
        key = str(candidate)
        if key in records:
            if role not in records[key]["roles"]:
                records[key]["roles"].append(role)
            return records[key]
        data = candidate.read_bytes()
        entry = {"path": key, "roles": [role], "sha256": hashlib.sha256(data).hexdigest(),
                 "bytes": len(data), "row_count": None, "row_collection": None, "columns": []}
        try:
            if candidate.suffix.lower() == ".json":
                payload = json.loads(data)
                rows = payload if isinstance(payload, list) else None
                collection = "$" if isinstance(payload, list) else None
                if isinstance(payload, dict):
                    collections = [(name, payload[name]) for name in ("rows", "asos", "candidates")
                                   if isinstance(payload.get(name), list)]
                    if len(collections) == 1:
                        collection, rows = collections[0]
                if rows is not None:
                    entry.update(row_count=len(rows), row_collection=collection,
                                 columns=sorted({key for row in rows if isinstance(row, dict) for key in row}))
            elif candidate.suffix.lower() in {".csv", ".tsv"}:
                with candidate.open(newline="", encoding="utf-8-sig") as stream:
                    reader = csv.DictReader(stream, delimiter="\t" if candidate.suffix.lower() == ".tsv" else ",")
                    entry.update(columns=reader.fieldnames or [], row_count=sum(1 for _ in reader), row_collection="table")
        except (ValueError, UnicodeError, csv.Error) as exc:
            entry["table_read_error"] = type(exc).__name__
        records[key] = entry
        return entry

    trace = Path(artifact_path(root, "generated_filter_trace.json"))
    if trace.is_file():
        add(trace, "recorded_filter_decisions")
        try:
            payload = json.loads(trace.read_text())
        except (ValueError, UnicodeError):
            payload = {}
        sources = payload.get("sources") if isinstance(payload, dict) else None
        sources = sources if isinstance(sources, dict) else {}
        for role in ("raw_generated", "initial_filter", "scored", "final_output", "expression_annotation"):
            paths = sources.get(role)
            if isinstance(paths, list):
                for path in paths:
                    if isinstance(path, str) and path:
                        add(path, role)
    for name, role in (("structured_goal.json", "resolved_design_contract"),
                       ("generated_aso_set.json", "final_output"),
                       ("splice_site_comparisons.csv", "recorded_splice_predictions"),
                       ("strategy_router_summary.json", "recorded_selection_evidence")):
        path = Path(artifact_path(root, name))
        if path.is_file():
            add(path, role)
    panel = Path(artifact_path(root, "generated_aso_set.json")).with_name("design_panel.csv")
    if panel.is_file():
        add(panel, "selected_panel")
    return {"task_id": task_id, "producer_run_dir": str(root), "sources": list(records.values()),
            "unavailable_sources": unavailable,
            "scope": "Recorded source tables, not a new analysis or a claim of complete design success.",
            "rules": [
                "raw_generated tables preserve enumerated sites before filtering; initial_filter, scored and final_output are different stages.",
                "Aggregate and per-job raw tables can overlap. Reconcile program/reference/site identities before combining; do not sum counts or deduplicate solely by sequence.",
                "A complete requested library retains screened-out observations; selected-panel membership is a separate field or table.",
                "Use recorded coordinate conventions, strand, chemistry and reference identity from each row and the resolved design contract.",
                "Read complete source files. Column lists and counts are an inventory, not source rows or evidence that missing measurements equal zero.",
            ]}
