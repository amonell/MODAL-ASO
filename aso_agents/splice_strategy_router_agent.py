from __future__ import annotations

import re
import json
import math
import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agents import Agent, ModelSettings, Runner, function_tool
from aso_agents.parameter_constraints import evaluate_row_against_constraints
from aso_agents.splice_recommendations import annotate_route_recommendation, RECOMMENDATION_FIELDS


SHARED_CONTEXT_ABS_LIMIT = 0.2
SHARED_ACCEPTOR_DECREASE_LIMIT = -0.05


def _env_optional_positive_int(name: str) -> Optional[int]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None

PLANNER_MODEL = "gpt-5.2-2025-12-11"
PLANNER_MAX_OUTPUT_TOKENS = _env_optional_positive_int(
    "ASO_SPLICE_STRATEGY_PLANNER_MAX_OUTPUT_TOKENS"
)
PLANNER_MODEL_SETTINGS = ModelSettings(
    reasoning={"effort": "high"},
    max_output_tokens=PLANNER_MAX_OUTPUT_TOKENS,
    extra_body={
        "parallel_tool_calls": True,
        "text": {"format": {"type": "json_object"}},
    },
)

EXECUTOR_MODEL = "gpt-5.2-2025-12-11"
EXECUTOR_MAX_OUTPUT_TOKENS = _env_optional_positive_int(
    "ASO_SPLICE_STRATEGY_EXECUTOR_MAX_OUTPUT_TOKENS"
)
EXECUTOR_MODEL_SETTINGS = ModelSettings(
    max_output_tokens=EXECUTOR_MAX_OUTPUT_TOKENS,
    extra_body={
        "parallel_tool_calls": True,
        "text": {"format": {"type": "json_object"}},
    },
)

ROUTE_RANKER_MODEL = PLANNER_MODEL
ROUTE_RANKER_MAX_OUTPUT_TOKENS = _env_optional_positive_int(
    "ASO_ROUTE_RANKER_MAX_OUTPUT_TOKENS"
)
ROUTE_RANKER_MODEL_SETTINGS = ModelSettings(
    reasoning={"effort": "high"},
    max_output_tokens=ROUTE_RANKER_MAX_OUTPUT_TOKENS,
    extra_body={
        "text": {"format": {"type": "json_object"}},
    },
)


def _call_json_agent(agent: Agent, payload: Dict[str, Any], *, max_steps: int = 4) -> Dict[str, Any]:
    result = Runner._run_sync(
        agent,
        input=json.dumps(payload, ensure_ascii=False),
        max_steps=max_steps,
    )
    output = result.final_output
    if not isinstance(output, str):
        raise RuntimeError(f"{agent.name} returned non-string output.")
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{agent.name} returned non-JSON output: {output}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{agent.name} must return a JSON object.")
    return {
        "parsed": data,
        "responses": result.responses,
        "raw_output": output,
    }


def _shared_context_blocked(
    shared_label: Optional[str],
    shared_delta: Optional[float],
    shared_abs: Optional[float],
    target_labels: Optional[set[str]] = None,
    current_label: Optional[str] = None,
    *,
    shared_kind: Optional[str] = None,
    shared_covers_all: Optional[bool] = None,
) -> bool:
    if not shared_label:
        return False
    label_norm = str(shared_label).strip().lower()
    if not label_norm:
        return False
    if current_label and str(current_label).strip() and label_norm == str(current_label).strip().lower():
        return False
    if target_labels:
        norm_set = {str(lbl).strip().lower() for lbl in target_labels if lbl}
        if label_norm in norm_set:
            return False
    if shared_delta is not None:
        kind_norm = (shared_kind or "").strip().lower()
        if kind_norm == "acceptor" and bool(shared_covers_all) and shared_delta < SHARED_ACCEPTOR_DECREASE_LIMIT:
            return True
    if shared_abs is not None and shared_abs >= SHARED_CONTEXT_ABS_LIMIT:
        return True
    return False


def _load_scored_tiles(json_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("rows") or []
    schema = data.get("schema") or {}
    return rows, schema


def _parse_site_metrics(row: Dict[str, Any]) -> List[Dict[str, Optional[float]]]:
    names_raw = str(row.get("delta_names") or "").split(",")
    deletion_raw = str(row.get("delta_deletion") if row.get("delta_deletion") is not None else "").split(",")
    replacement_raw = str(row.get("delta_replacement") if row.get("delta_replacement") is not None else "").split(",")

    metrics: List[Dict[str, Optional[float]]] = []
    for idx, name in enumerate(names_raw):
        label = name.strip()
        if not label:
            continue
        del_val = _safe_float(deletion_raw, idx)
        rep_val = _safe_float(replacement_raw, idx)
        metrics.append({
            "label": label,
            "delta_deletion": del_val,
            "delta_replacement": rep_val,
        })
    return metrics


def _safe_float(series: List[str], idx: int) -> Optional[float]:
    if idx >= len(series):
        return None
    value = series[idx].strip()
    if not value:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def _site_index(rows: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, Dict[str, Optional[float]]]]]:
    index: Dict[str, List[Tuple[int, Dict[str, Optional[float]]]]] = {}
    for row_idx, row in enumerate(rows):
        metrics = _parse_site_metrics(row)
        for metric in metrics:
            label = metric.get("label")
            if not label:
                continue
            index.setdefault(label, []).append((row_idx, metric))
    return index


def _collect_available_labels(json_path: str) -> Set[str]:
    rows, _ = _load_scored_tiles(json_path)
    labels: Set[str] = set()
    for row in rows:
        for metric in _parse_site_metrics(row):
            label = metric.get("label")
            if not label:
                continue
            label_str = str(label).strip()
            if label_str:
                labels.add(label_str)
    return labels


def _canonicalize_label_value(
    value: Any,
    available_labels: Set[str],
    available_lookup: Dict[str, str],
    alias_lookup: Dict[str, str],
) -> Any:
    if not isinstance(value, str):
        return value
    label = value.strip()
    if not label:
        return label
    if label in available_labels:
        return label
    lower = label.lower()
    if lower in available_lookup:
        return available_lookup[lower]
    mapped = alias_lookup.get(lower)
    if mapped:
        mapped_lower = mapped.lower()
        if mapped in available_labels:
            return mapped
        if mapped_lower in available_lookup:
            return available_lookup[mapped_lower]
    match = re.search(r"(acceptor|donor)[^0-9]*([0-9]{4,})", lower)
    if match:
        candidate = f"{match.group(1)}@{match.group(2)}"
        if candidate in available_labels:
            return candidate
        candidate_lower = candidate.lower()
        if candidate_lower in available_lookup:
            return available_lookup[candidate_lower]
    return label


def _canonicalize_labels_json_string(
    payload: Optional[str],
    available_labels: Set[str],
    available_lookup: Dict[str, str],
    alias_lookup: Dict[str, str],
) -> Optional[str]:
    if not payload or not isinstance(payload, str):
        return payload
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload

    changed = False

    def _canon_label(label_value: Any) -> Any:
        nonlocal changed
        new_value = _canonicalize_label_value(label_value, available_labels, available_lookup, alias_lookup)
        if new_value != label_value:
            changed = True
        return new_value

    if isinstance(parsed, list):
        for idx, item in enumerate(parsed):
            if isinstance(item, dict):
                label_val = item.get("label")
                new_val = _canon_label(label_val)
                if new_val != label_val:
                    item["label"] = new_val
                site_label_val = item.get("site_label")
                new_site_val = _canon_label(site_label_val)
                if new_site_val != site_label_val:
                    item["site_label"] = new_site_val
            elif isinstance(item, str):
                parsed[idx] = _canon_label(item)
    elif isinstance(parsed, dict):
        for key, val in list(parsed.items()):
            if isinstance(val, str) and key in {"label", "site_label"}:
                parsed[key] = _canon_label(val)
    else:
        return payload

    if not changed:
        return payload
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _metric_map_for_row(row: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    mapping: Dict[str, Dict[str, Optional[float]]] = {}
    for metric in _parse_site_metrics(row):
        label = metric.get("label")
        if not label:
            continue
        mapping[label] = {
            "delta_deletion": metric.get("delta_deletion"),
            "delta_replacement": metric.get("delta_replacement"),
        }
    return mapping


def _ensure_output_dir(base_path: str, output_path: Optional[str]) -> str:
    if output_path:
        final_path = output_path if os.path.isabs(output_path) else os.path.join(os.path.dirname(base_path), output_path)
    else:
        final_path = os.path.join(os.path.dirname(base_path), "filtered_tiles.json")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    return final_path


@function_tool
def list_site_labels(json_path: str) -> Dict[str, List[str]]:
    """Return the set of splice-site labels available in delta_names."""
    rows, _ = _load_scored_tiles(json_path)
    labels = sorted(set(metric.get("label") for row in rows for metric in _parse_site_metrics(row) if metric.get("label")))
    return {"labels": labels}


@function_tool
def preview_site_metrics(json_path: str, label: str, limit: int = 10) -> Dict[str, Any]:
    """Preview up to `limit` rows for a given splice-site label with deletion and replacement deltas."""
    rows, _ = _load_scored_tiles(json_path)
    index = _site_index(rows)
    entries = index.get(label, [])
    preview = []
    for row_idx, metric in entries[: max(0, limit)]:
        preview.append({
            "row_index": row_idx,
            "delta_deletion": metric.get("delta_deletion"),
            "delta_replacement": metric.get("delta_replacement"),
        })
    return {
        "label": label,
        "count": len(entries),
        "preview": preview,
    }


@function_tool
def filter_site_metrics(
    json_path: str,
    label: str,
    min_delta_deletion: Optional[float] = None,
    max_delta_deletion: Optional[float] = None,
    min_delta_replacement: Optional[float] = None,
    max_delta_replacement: Optional[float] = None,
    top_n: Optional[int] = None,
    output_path: Optional[str] = None,
    logic: str = "and",
) -> Dict[str, Any]:
    """Filter rows by splice-site label and delta thresholds, writing results to output_path."""
    rows, schema = _load_scored_tiles(json_path)
    index = _site_index(rows)
    entries = index.get(label, [])

    def _check_value(value: Optional[float], min_value: Optional[float], max_value: Optional[float]) -> Optional[bool]:
        if min_value is None and max_value is None:
            return None
        if value is None:
            return False

        # If both bounds provided, normalize ordering and apply inclusive range
        if min_value is not None and max_value is not None:
            lo, hi = sorted([min_value, max_value])
            return lo <= value <= hi

        # Only minimum provided: interpret sign-aware (negative min implies "<=", positive implies ">=")
        if min_value is not None:
            if min_value < 0:
                return value <= min_value
            return value >= min_value

        # Only maximum provided: always treat as upper bound (inclusive)
        if max_value is not None:
            return value <= max_value

        return None

    logic_norm = (logic or "").strip().lower()
    if logic_norm not in {"and", "or"}:
        logic_norm = "and"

    def _passes(metric: Dict[str, Optional[float]]) -> bool:
        checks: List[bool] = []
        del_check = _check_value(metric.get("delta_deletion"), min_delta_deletion, max_delta_deletion)
        if del_check is not None:
            checks.append(del_check)
        rep_check = _check_value(metric.get("delta_replacement"), min_delta_replacement, max_delta_replacement)
        if rep_check is not None:
            checks.append(rep_check)

        if not checks:
            return True
        if logic_norm == "or":
            return any(checks)
        return all(checks)

    filtered_indices: List[int] = []
    filtered_metrics: List[Dict[str, Optional[float]]] = []
    for row_idx, metric in entries:
        if _passes(metric):
            filtered_indices.append(row_idx)
            filtered_metrics.append(metric)

    if top_n is not None and top_n > 0:
        combined = list(zip(filtered_indices, filtered_metrics))
        combined.sort(key=lambda item: (
            -abs(item[1].get("delta_deletion") or 0.0),
            -abs(item[1].get("delta_replacement") or 0.0),
        ))
        combined = combined[:top_n]
        filtered_indices = [idx for idx, _ in combined]
        filtered_metrics = [metric for _, metric in combined]

    selected_rows: List[Dict[str, Any]] = []
    for idx, metric in zip(filtered_indices, filtered_metrics):
        row = rows[idx]
        # Preserve the full tile metadata while annotating selection context.
        row_copy = dict(row)
        for key in (
            "delta_inclusion_deletion",
            "delta_inclusion_replacement",
            "delta_inclusion_deletion_mean",
            "delta_inclusion_replacement_mean",
        ):
            row_copy.pop(key, None)
        row_copy["selected_label"] = label
        row_copy["selected_delta_deletion"] = metric.get("delta_deletion")
        row_copy["selected_delta_replacement"] = metric.get("delta_replacement")
        selected_rows.append(row_copy)

    schema_copy_raw = json.loads(json.dumps(schema)) if isinstance(schema, dict) else {}
    schema_copy = schema_copy_raw if isinstance(schema_copy_raw, dict) else {}
    columns = schema_copy.get("columns")
    if not isinstance(columns, dict):
        columns = {}
    columns.pop("delta_inclusion_deletion", None)
    columns.pop("delta_inclusion_replacement", None)
    columns.pop("delta_inclusion_deletion_mean", None)
    columns.pop("delta_inclusion_replacement_mean", None)
    columns.setdefault("selected_label", "string")
    columns.setdefault("selected_delta_deletion", "number")
    columns.setdefault("selected_delta_replacement", "number")
    schema_copy["columns"] = columns

    output_payload = {
        "schema": schema_copy,
        "rows": selected_rows,
    }
    if constraint_summary:
        output_payload["parameter_constraints"] = constraint_summary
        output_payload["constraint_filtered_count"] = constraint_filtered
        if constraint_failure_counts:
            output_payload["constraint_failure_counts"] = constraint_failure_counts

    final_path = _ensure_output_dir(json_path, output_path)
    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    return {
        "label": label,
        "count": len(selected_rows),
        "total_available": len(entries),
        "output_path": final_path,
        "thresholds": {
            "min_delta_deletion": min_delta_deletion,
            "max_delta_deletion": max_delta_deletion,
            "min_delta_replacement": min_delta_replacement,
            "max_delta_replacement": max_delta_replacement,
            "logic": logic_norm,
        },
    }


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        try:
            result = float(str(value))
            return result if math.isfinite(result) else None
        except Exception:
            return None


@function_tool
def auto_threshold_filter_site_metrics(
    json_path: str,
    label: str,
    direction: str,
    start_threshold: float = 0.5,
    min_threshold: float = 0.02,
    step: float = 0.04,
    min_count: int = 5,
    min_gc: Optional[float] = None,
    max_gc: Optional[float] = None,
    max_binding_sites: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministically filter rows for a splice-site label with an adaptive threshold.

    - direction: 'increase' or 'decrease'.
    - start_threshold: initial absolute delta requirement (default 0.5).
    - min_threshold: smallest absolute delta to consider (default 0.02).
    - step: step size when relaxing the threshold (default 0.04).
    - min_count: ensure at least this many rows when possible (default 5).
    - For each row, the larger absolute effect between deletion and replacement is considered.
    - Rows copy the selected deltas; global delta_inclusion_* columns are removed in the output.
    """
    direction_norm = (direction or "").strip().lower()
    if direction_norm not in {"increase", "decrease"}:
        return {"error": f"direction must be 'increase' or 'decrease', got '{direction}'"}

    rows, schema = _load_scored_tiles(json_path)
    index = _site_index(rows)
    entries = list(index.get(label, []))
    raw_total_available = len(entries)
    constraint_filters, constraint_summary = _normalize_constraint_filters(min_gc, max_gc, max_binding_sites)
    constraint_failure_counts: Dict[str, int] = {}
    constraint_filtered = 0
    if constraint_filters:
        filtered_entries: List[Tuple[int, Dict[str, Optional[float]]]] = []
        for row_idx, metric in entries:
            passes_constraints, reasons = evaluate_row_against_constraints(rows[row_idx], constraint_filters)
            if passes_constraints:
                filtered_entries.append((row_idx, metric))
            else:
                constraint_filtered += 1
                for reason in reasons:
                    constraint_failure_counts[reason] = constraint_failure_counts.get(reason, 0) + 1
        entries = filtered_entries
        if not entries:
            return {
                "label": label,
                "count": 0,
                "total_available": 0,
                "raw_total_available": raw_total_available,
                "output_path": None,
                "thresholds": {},
                "parameter_constraints": constraint_summary or None,
                "constraint_filtered_count": constraint_filtered,
                "constraint_failure_counts": constraint_failure_counts or None,
            }
    if not entries:
        base_result: Dict[str, Any] = {
            "label": label,
            "count": 0,
            "total_available": 0,
            "raw_total_available": raw_total_available,
            "output_path": None,
            "threshold_used": None,
            "threshold_history": [],
            "direction": direction_norm,
        }
        if constraint_summary:
            base_result["parameter_constraints"] = constraint_summary
            base_result["constraint_filtered_count"] = constraint_filtered
            if constraint_failure_counts:
                base_result["constraint_failure_counts"] = constraint_failure_counts
        return base_result

    target_min_count = max(int(min_count), 20)

    def _passes(metric: Dict[str, Optional[float]], threshold: float) -> bool:
        del_val = _parse_float(metric.get("delta_deletion"))
        rep_val = _parse_float(metric.get("delta_replacement"))
        candidate = _select_best_for_direction(direction_norm, del_val, rep_val)
        if candidate is None:
            return False
        if direction_norm == "decrease":
            return candidate <= -threshold
        return candidate >= threshold

    threshold = max(start_threshold, 0.0)
    min_threshold = max(min_threshold, 0.0)
    step = max(step, 0.001)
    threshold_history: List[Dict[str, Any]] = []

    filtered_indices: List[int] = []
    filtered_metrics: List[Dict[str, Optional[float]]] = []

    while threshold >= min_threshold - 1e-9:
        threshold_history.append({"threshold": threshold})
        filtered_indices = []
        filtered_metrics = []
        for row_idx, metric in entries:
            if _passes(metric, threshold):
                filtered_indices.append(row_idx)
                filtered_metrics.append(metric)
        if len(filtered_indices) >= target_min_count or threshold <= min_threshold:
            break
        threshold = max(threshold - step, min_threshold)

    selected_rows: List[Dict[str, Any]] = []
    for idx, metric in zip(filtered_indices, filtered_metrics):
        row = rows[idx]
        row_copy = dict(row)
        for key in (
            "delta_inclusion_deletion",
            "delta_inclusion_replacement",
            "delta_inclusion_deletion_mean",
            "delta_inclusion_replacement_mean",
        ):
            row_copy.pop(key, None)
        shared_abs = _parse_float(row_copy.get("shared_context_abs_delta"))
        shared_delta = _parse_float(row_copy.get("shared_context_delta"))
        shared_label = row_copy.get("shared_context_label")
        shared_kind = row_copy.get("shared_context_site_kind")
        shared_covers = row_copy.get("shared_context_covers_all")
        if _shared_context_blocked(
            shared_label,
            shared_delta,
            shared_abs,
            current_label=label,
            shared_kind=shared_kind,
            shared_covers_all=bool(shared_covers),
        ):
            continue
        row_copy["selected_label"] = label
        row_copy["selected_delta_deletion"] = _parse_float(metric.get("delta_deletion"))
        row_copy["selected_delta_replacement"] = _parse_float(metric.get("delta_replacement"))
        selected_rows.append(row_copy)

    schema_copy_raw = json.loads(json.dumps(schema)) if isinstance(schema, dict) else {}
    schema_copy = schema_copy_raw if isinstance(schema_copy_raw, dict) else {}
    columns = schema_copy.get("columns")
    if not isinstance(columns, dict):
        columns = {}
    columns.pop("delta_inclusion_deletion", None)
    columns.pop("delta_inclusion_replacement", None)
    columns.pop("delta_inclusion_deletion_mean", None)
    columns.pop("delta_inclusion_replacement_mean", None)
    columns.setdefault("selected_label", "string")
    columns.setdefault("selected_delta_deletion", "number")
    columns.setdefault("selected_delta_replacement", "number")
    schema_copy["columns"] = columns

    output_payload = {
        "schema": schema_copy,
        "rows": selected_rows,
    }

    final_path = _ensure_output_dir(json_path, output_path)
    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    result: Dict[str, Any] = {
        "label": label,
        "count": len(selected_rows),
        "total_available": len(entries),
        "raw_total_available": raw_total_available,
        "output_path": final_path,
        "threshold_used": threshold,
        "threshold_history": threshold_history,
        "direction": direction_norm,
    }
    if constraint_summary:
        result["parameter_constraints"] = constraint_summary
        result["constraint_filtered_count"] = constraint_filtered
        if constraint_failure_counts:
            result["constraint_failure_counts"] = constraint_failure_counts
    return result


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        candidate = float(text)
    except (TypeError, ValueError):
        return None
    if not candidate.is_integer():
        return None
    return int(candidate)


def _normalize_constraint_filters(
    min_gc: Optional[Any],
    max_gc: Optional[Any],
    max_binding_sites: Optional[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    constraints: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}

    min_gc_val = _parse_float(min_gc) if min_gc is not None else None
    max_gc_val = _parse_float(max_gc) if max_gc is not None else None
    if min_gc_val is not None:
        min_gc_val = max(0.0, min(1.0, float(min_gc_val)))
    if max_gc_val is not None:
        max_gc_val = max(0.0, min(1.0, float(max_gc_val)))
    if min_gc_val is not None and max_gc_val is not None and min_gc_val > max_gc_val:
        min_gc_val, max_gc_val = max_gc_val, min_gc_val
    gc_entry: Dict[str, float] = {}
    if min_gc_val is not None:
        summary["min_gc"] = min_gc_val
        constraints["gc_min"] = min_gc_val
        gc_entry["min"] = min_gc_val
    if max_gc_val is not None:
        summary["max_gc"] = max_gc_val
        constraints["gc_max"] = max_gc_val
        gc_entry["max"] = max_gc_val
    if gc_entry:
        constraints["gc_content"] = gc_entry

    binding_limit = _parse_int(max_binding_sites)
    if binding_limit is not None:
        summary["max_binding_sites"] = binding_limit
        constraints["max_binding_sites"] = binding_limit

    return constraints, summary


def _select_best_for_direction(direction: str, del_val: Optional[float], rep_val: Optional[float]) -> Optional[float]:
    values = []
    if del_val is not None:
        values.append(del_val)
    if rep_val is not None:
        values.append(rep_val)
    if not values:
        return None
    if direction == "decrease":
        return min(values)
    return max(values)


def _direction_passes(direction: str, del_val: Optional[float], rep_val: Optional[float], threshold: float) -> bool:
    candidate = _select_best_for_direction(direction, del_val, rep_val)
    if candidate is None:
        return False
    if direction == "decrease":
        return candidate <= -threshold
    return candidate >= threshold


@function_tool
def auto_threshold_constrained_site_metrics(
    json_path: str,
    primary_label: str,
    primary_direction: str,
    safeguards_json: str,
    primary_labels_json: Optional[str] = None,
    start_threshold: float = 0.5,
    min_threshold: float = 0.02,
    step: float = 0.04,
    min_count: int = 5,
    default_max_abs_delta: float = 0.1,
    min_gc: Optional[float] = None,
    max_gc: Optional[float] = None,
    max_binding_sites: Optional[int] = None,
    max_results: int = 50,
    allow_zero_delta_fallback: bool = False,
    output_path: Optional[str] = None,
    exon_objectives_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Filter primary splice effects and mandatory guards; report advisory guards.

    - primary_labels_json: optional JSON list of {label, direction} entries. When provided, every listed junction must satisfy the threshold together.
    - primary_direction: 'increase' or 'decrease' (used with primary_label when JSON is omitted; reuse the first JSON entry for compatibility when providing primary_labels_json).
    - safeguards_json: JSON list of constraints, each containing:
        * label (required): splice-site label to limit.
        * max_abs_delta: optional absolute delta cap applied to both deletion/replacement.
        * max_increase: optional positive delta cap.
        * max_decrease: optional magnitude cap for negative deltas.
        * note: optional text retained in the output metadata.
        * required: enforce this guard for explicit preservation constraints; missing measurements cannot pass.
      If numeric caps are omitted, max_abs_delta defaults to `default_max_abs_delta`.
    - Threshold relaxation mirrors auto_threshold_filter_site_metrics but begins at >=0.5 effect size and only relaxes (down to >=0.01) until at least `min_count` tiles remain.
    - The default tolerance is ±0.1; explicitly supplied finite bounds are honored. Use max_abs_delta for two-sided caps, or max_increase / max_decrease when only one direction must be limited.
    - Rows include the evaluated safeguard deltas and pass/fail flags. Guards marked required filter candidates, including those with missing measurements. Other safeguards remain advisory.
    - Optional parameter constraints (min_gc, max_gc, max_binding_sites) filter candidate tiles before thresholding.
    - exon_objectives_json optionally supplies explicit, reference-resolved exon outcomes. Paired evidence is matched by reference and coordinates, then used for signed recommendation order. Missing or conflicting evidence stays unknown; no additional eligibility filter is imposed.
    - max_results caps the base shortlist (default 50), with existing strong-effect extras retained. Signed objective evidence orders the base shortlist, and pre-trim counts remain recorded.
    """
    try:
        exon_objectives = json.loads(exon_objectives_json) if exon_objectives_json else []
    except (TypeError, ValueError) as exc:
        return {"error": f"exon_objectives_json_invalid: {exc}"}
    if not isinstance(exon_objectives, list) or any(
        not isinstance(item, dict) or not item.get("objective_id")
        or item.get("desired_effect") not in ("inclusion", "skipping")
        or item.get("direction") != ("increase" if item.get("desired_effect") == "inclusion" else "decrease")
        or (item.get("status") == "resolved" and any(item.get(key) is None for key in
            ("reference_transcript_id", "chrom", "strand", "acceptor_position", "donor_position")))
        for item in exon_objectives
    ):
        return {"error": "exon_objectives_json_invalid_definitions"}
    primary_specs: List[Dict[str, str]] = []
    direction_norm = (primary_direction or "").strip().lower()

    if primary_labels_json:
        try:
            parsed = json.loads(primary_labels_json)
        except json.JSONDecodeError as exc:
            return {"error": f"primary_labels_json_invalid: {exc}"}
        entries = parsed if isinstance(parsed, list) else [parsed]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = (entry.get("label") or entry.get("site") or entry.get("name") or "").strip()
            direction = (entry.get("direction") or "").strip().lower()
            if not label or direction not in {"increase", "decrease"}:
                continue
            primary_specs.append({"label": label, "direction": direction})

    if not primary_specs:
        label_norm = (primary_label or "").strip()
        if not label_norm:
            return {"error": "primary_label is required when primary_labels_json is not provided"}
        if direction_norm not in {"increase", "decrease"}:
            return {"error": f"primary_direction must be 'increase' or 'decrease', got '{primary_direction}'"}
        primary_specs.append({"label": label_norm, "direction": direction_norm})
    else:
        # When JSON is provided we no longer require primary_direction; reset direction_norm for clarity.
        direction_norm = primary_specs[0]["direction"]

    target_labels: Set[str] = {spec["label"] for spec in primary_specs}
    allow_zero_fallback = bool(allow_zero_delta_fallback) or str(allow_zero_delta_fallback).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    try:
        safeguards_raw = json.loads(safeguards_json or "[]")
    except json.JSONDecodeError as exc:
        return {"error": f"safeguards_json_invalid: {exc}"}

    safeguards_input = safeguards_raw if isinstance(safeguards_raw, list) else [safeguards_raw]
    primary_markers: Set[str] = set()
    for spec in primary_specs:
        primary_markers.update(_extract_isoform_markers(spec["label"]))
    safeguards: List[Dict[str, Any]] = []
    if not math.isfinite(default_max_abs_delta) or default_max_abs_delta < 0:
        return {"error": "default_max_abs_delta must be a finite nonnegative magnitude"}
    for entry in safeguards_input:
        if not isinstance(entry, dict):
            continue
        label = (entry.get("label") or entry.get("site") or entry.get("name") or "").strip()
        if not label:
            continue
        max_abs_raw = _parse_float(entry.get("max_abs_delta"))
        max_inc_raw = _parse_float(entry.get("max_increase"))
        max_dec_raw = _parse_float(entry.get("max_decrease"))
        for key, parsed in (("max_abs_delta", max_abs_raw), ("max_increase", max_inc_raw), ("max_decrease", max_dec_raw)):
            if entry.get(key) is not None and (parsed is None or not math.isfinite(parsed)):
                return {"error": f"safeguard_bound_invalid: {label} {key} must be finite"}

        max_abs = None
        if max_abs_raw is not None:
            max_abs = abs(max_abs_raw)

        max_inc = None
        if max_inc_raw is not None:
            max_inc = abs(max_inc_raw)

        max_dec = None
        if max_dec_raw is not None:
            max_dec = abs(max_dec_raw)

        if max_abs is None and max_inc is None and max_dec is None:
            label_markers = _extract_isoform_markers(label)
            if primary_markers and label_markers & primary_markers:
                max_abs = None  # allow primary isoform junctions to vary unless caps are explicit
            else:
                max_abs = default_max_abs_delta

        safeguard_note = (entry.get("note") or entry.get("description") or entry.get("constraint") or "").strip()
        safeguards.append({
            "label": label,
            "max_abs_delta": max_abs,
            "max_increase": max_inc,
            "max_decrease": max_dec,
            "note": safeguard_note or None,
            "required": entry.get("required") is True,
        })

    rows, schema = _load_scored_tiles(json_path)
    metric_maps: List[Dict[str, Dict[str, Optional[float]]]] = [_metric_map_for_row(row) for row in rows]
    index = _site_index(rows)
    primary_lead_label = primary_specs[0]["label"]
    base_entries = index.get(primary_lead_label, [])
    candidate_row_indices: Set[int] = set(range(len(rows)))
    if base_entries:
        candidate_row_indices = {row_idx for row_idx, _ in base_entries}
    # Ensure candidates include only rows that have all required labels.
    candidate_row_indices = {
        row_idx
        for row_idx in candidate_row_indices
        if target_labels.issubset(metric_maps[row_idx].keys())
    }

    raw_total_available = len(candidate_row_indices)
    constraint_filters, constraint_summary = _normalize_constraint_filters(min_gc, max_gc, max_binding_sites)
    constraint_failure_counts: Dict[str, int] = {}
    constraint_filtered = 0
    if constraint_filters:
        constrained_indices: Set[int] = set()
        for row_idx in candidate_row_indices:
            passes_constraints, reasons = evaluate_row_against_constraints(rows[row_idx], constraint_filters)
            if passes_constraints:
                constrained_indices.add(row_idx)
            else:
                constraint_filtered += 1
                for reason in reasons:
                    constraint_failure_counts[reason] = constraint_failure_counts.get(reason, 0) + 1
        candidate_row_indices = constrained_indices
    total_available = len(candidate_row_indices)

    if not candidate_row_indices:
        empty_result: Dict[str, Any] = {
            "primary_label": primary_specs[0]["label"] if len(primary_specs) == 1 else None,
            "primary_direction": primary_specs[0]["direction"] if len(primary_specs) == 1 else None,
            "primary_targets": primary_specs,
            "count": 0,
            "output_path": None,
            "threshold_used": None,
            "threshold_pass_count": 0,
            "threshold_history": [],
            "safeguards_applied": safeguards,
            "total_available": total_available,
            "raw_total_available": raw_total_available,
        }
        if output_path:
            schema_copy_raw = json.loads(json.dumps(schema)) if isinstance(schema, dict) else {}
            schema_copy = schema_copy_raw if isinstance(schema_copy_raw, dict) else {}
            columns = schema_copy.get("columns")
            if not isinstance(columns, dict):
                columns = {}
            columns.pop("delta_inclusion_deletion", None)
            columns.pop("delta_inclusion_replacement", None)
            columns.pop("delta_inclusion_deletion_mean", None)
            columns.pop("delta_inclusion_replacement_mean", None)
            columns.setdefault("selected_label", "string")
            columns.setdefault("selected_direction", "string")
            columns.setdefault("primary_targets", "array")
            columns.setdefault("safeguard_report", "array")
            columns.setdefault("safeguards_passed", "boolean")
            columns.setdefault("safeguards_failed_labels", "array")
            schema_copy["columns"] = columns
            output_payload = {
                "schema": schema_copy,
                "rows": [],
                "safeguards": safeguards,
                "primary_targets": primary_specs,
                "threshold_pass_count": 0,
                "exon_objectives": exon_objectives,
            }
            if constraint_summary:
                output_payload["parameter_constraints"] = constraint_summary
                output_payload["constraint_filtered_count"] = constraint_filtered
                if constraint_failure_counts:
                    output_payload["constraint_failure_counts"] = constraint_failure_counts
            final_path = _ensure_output_dir(json_path, output_path)
            with open(final_path, "w", encoding="utf-8") as handle:
                json.dump(output_payload, handle, ensure_ascii=False, indent=2)
            empty_result["output_path"] = final_path
        if constraint_summary:
            empty_result["parameter_constraints"] = constraint_summary
            empty_result["constraint_filtered_count"] = constraint_filtered
            if constraint_failure_counts:
                empty_result["constraint_failure_counts"] = constraint_failure_counts
        return empty_result

    def _values_for_metric(metric: Optional[Dict[str, Optional[float]]]) -> List[float]:
        values: List[float] = []
        if not metric:
            return values
        for key in ("delta_deletion", "delta_replacement"):
            parsed = _parse_float(metric.get(key) if metric else None)
            if parsed is not None and math.isfinite(parsed):
                values.append(parsed)
        return values

    def _passes_safeguards(metric_map: Dict[str, Dict[str, Optional[float]]]) -> Tuple[Optional[bool], List[Dict[str, Any]]]:
        reports: List[Dict[str, Any]] = []
        tol = 1e-9
        overall_pass: Optional[bool] = True
        for guard in safeguards:
            label = guard["label"]
            values = _values_for_metric(metric_map.get(label))
            passes: Optional[bool] = True if len(values) == 2 else None
            max_abs = guard.get("max_abs_delta")
            max_inc = guard.get("max_increase")
            max_dec = guard.get("max_decrease")
            if max_abs is not None:
                for value in values:
                    if abs(value) > max_abs + tol:
                        passes = False
                        break
            if passes is not False and max_inc is not None:
                for value in values:
                    if value > max_inc + tol:
                        passes = False
                        break
            if passes is not False and max_dec is not None:
                for value in values:
                    if value < -max_dec - tol:
                        passes = False
                        break
            if passes is False:
                overall_pass = False
            elif passes is None and overall_pass is not False:
                overall_pass = None
            metric = metric_map.get(label) or {}
            reports.append({
                "label": label,
                "values": values,
                "delta_deletion": metric.get("delta_deletion"),
                "delta_replacement": metric.get("delta_replacement"),
                "measurement_count": len(values),
                "max_abs_delta": max_abs,
                "max_increase": max_inc,
                "max_decrease": max_dec,
                "note": guard.get("note"),
                "passes": passes,
                "status": "pass" if passes is True else "fail" if passes is False else "not_evaluated",
                "required": guard.get('required', False),
            })
        return overall_pass, reports

    def _primary_effect_score(row_idx: int) -> float:
        """Return magnitude for ranking candidates across primary targets."""
        metric_map = metric_maps[row_idx]
        per_spec_scores: List[float] = []
        for spec in primary_specs:
            metric = metric_map.get(spec["label"])
            if not metric:
                continue
            del_val = _parse_float(metric.get("delta_deletion"))
            rep_val = _parse_float(metric.get("delta_replacement"))
            candidates = [value for value in (del_val, rep_val) if value is not None]
            if not candidates:
                continue
            if spec["direction"] == "increase":
                directional = max(candidates)
            else:
                directional = min(candidates)
            per_spec_scores.append(abs(directional))
        if not per_spec_scores:
            return 0.0
        if len(primary_specs) == 1:
            return per_spec_scores[0]
        return min(per_spec_scores)

    def _repeat_risk_adjusted_score(row_idx: int) -> float:
        base_score = _primary_effect_score(row_idx)
        row_data = rows[row_idx]
        risk_class = str(row_data.get("repeat_risk_class") or "").strip().lower()
        max_abs_delta = _parse_float(row_data.get("repeat_max_abs_delta")) or 0.0
        penalties = {
            "high": 0.20,
            "borderline": 0.05,
            "low": 0.01,
            "not_evaluated": 0.02 if _parse_float(row_data.get("repeat_hit_count_same_gene")) else 0.0,
        }
        penalty = penalties.get(risk_class, 0.0)
        if max_abs_delta:
            penalty = max(penalty, min(max_abs_delta, 0.20))
        return base_score - penalty

    def _site_position_from_label(label: str) -> Optional[int]:
        match = re.match(r"^(?:acceptor|donor)@(-?\d+)$", str(label or "").strip(), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    primary_positions = [
        position
        for position in (_site_position_from_label(spec["label"]) for spec in primary_specs)
        if position is not None
    ]

    def _row_target_distance(row_idx: int) -> int:
        if not primary_positions:
            return 999999999
        row = rows[row_idx]
        start = _parse_float(row.get("Start") if isinstance(row, dict) else None)
        end = _parse_float(row.get("End") if isinstance(row, dict) else None)
        if start is None:
            start = _parse_float(row.get("start") if isinstance(row, dict) else None)
        if end is None:
            end = _parse_float(row.get("end") if isinstance(row, dict) else None)
        if start is None or end is None:
            return 999999999
        lo = int(min(start, end))
        hi = int(max(start, end))
        distances: List[int] = []
        for position in primary_positions:
            if lo <= position <= hi:
                distances.append(0)
            else:
                distances.append(min(abs(position - lo), abs(position - hi)))
        return min(distances) if distances else 999999999

    # Enforce primary effect strength: begin at >=0.5 and relax downward only if needed.
    threshold = max(start_threshold, 0.5)
    min_threshold = max(min_threshold, 0.01)
    min_threshold = min(min_threshold, threshold)
    step = max(step, 0.005)
    try:
        max_results_int = int(max_results)
    except (TypeError, ValueError):
        max_results_int = 0
    if max_results_int < 0:
        max_results_int = 0
    if max_results_int and max_results_int < min_count:
        max_results_int = min_count
    threshold_history: List[Dict[str, Any]] = []

    selected_indices: List[int] = []
    primary_reports: Dict[int, List[Dict[str, Any]]] = {}
    safeguard_reports: Dict[int, List[Dict[str, Any]]] = {}
    safeguard_status: Dict[int, bool] = {}
    threshold_pass_count = 0
    fallback_selection_used = False
    fallback_positive_count = 0
    zero_delta_fallback_count = 0
    selection_mode = "threshold"
    warning_text: Optional[str] = None

    while threshold >= min_threshold - 1e-9:
        threshold_history.append({"threshold": threshold})
        selected_indices = []
        primary_reports = {}
        safeguard_reports = {}
        safeguard_status = {}
        for row_idx in candidate_row_indices:
            row_data = rows[row_idx]
            metric_map = metric_maps[row_idx]
            shared_abs = _parse_float(row_data.get("shared_context_abs_delta"))
            shared_delta = _parse_float(row_data.get("shared_context_delta"))
            shared_label = row_data.get("shared_context_label")
            shared_kind = row_data.get("shared_context_site_kind")
            shared_covers = row_data.get("shared_context_covers_all")
            if _shared_context_blocked(
                shared_label,
                shared_delta,
                shared_abs,
                target_labels=target_labels,
                shared_kind=shared_kind,
                shared_covers_all=bool(shared_covers),
            ):
                continue

            row_primary_reports: List[Dict[str, Any]] = []
            primary_pass = True
            for spec in primary_specs:
                metric = metric_map.get(spec["label"])
                del_val = _parse_float(metric.get("delta_deletion")) if metric else None
                rep_val = _parse_float(metric.get("delta_replacement")) if metric else None
                passes = _direction_passes(spec["direction"], del_val, rep_val, threshold)
                row_primary_reports.append({
                    "label": spec["label"],
                    "direction": spec["direction"],
                    "delta_deletion": del_val,
                    "delta_replacement": rep_val,
                    "passes_threshold": passes,
                })
                if not passes:
                    primary_pass = False
                    break
            if not primary_pass:
                continue

            passes_safeguards, reports = _passes_safeguards(metric_map)
            if any(r.get('required') and not r['passes'] for r in reports):
                continue
            safeguard_reports[row_idx] = reports
            safeguard_status[row_idx] = passes_safeguards
            selected_indices.append(row_idx)
            primary_reports[row_idx] = row_primary_reports
        if len(selected_indices) >= min_count or threshold <= min_threshold:
            break
        threshold = max(threshold - step, min_threshold)

    threshold_pass_count = len(selected_indices)
    if threshold_pass_count == 0 and primary_specs and all(spec["direction"] == "increase" for spec in primary_specs):
        fallback_primary_reports: Dict[int, List[Dict[str, Any]]] = {}
        fallback_safeguard_reports: Dict[int, List[Dict[str, Any]]] = {}
        fallback_safeguard_status: Dict[int, bool] = {}
        fallback_indices: List[int] = []
        for row_idx in candidate_row_indices:
            row_data = rows[row_idx]
            metric_map = metric_maps[row_idx]
            shared_abs = _parse_float(row_data.get("shared_context_abs_delta"))
            shared_delta = _parse_float(row_data.get("shared_context_delta"))
            shared_label = row_data.get("shared_context_label")
            shared_kind = row_data.get("shared_context_site_kind")
            shared_covers = row_data.get("shared_context_covers_all")
            if _shared_context_blocked(
                shared_label,
                shared_delta,
                shared_abs,
                target_labels=target_labels,
                shared_kind=shared_kind,
                shared_covers_all=bool(shared_covers),
            ):
                continue

            row_primary_reports: List[Dict[str, Any]] = []
            positive_pass = True
            for spec in primary_specs:
                metric = metric_map.get(spec["label"])
                del_val = _parse_float(metric.get("delta_deletion")) if metric else None
                rep_val = _parse_float(metric.get("delta_replacement")) if metric else None
                best_val = _select_best_for_direction(spec["direction"], del_val, rep_val)
                passes = best_val is not None and best_val > 0
                row_primary_reports.append({
                    "label": spec["label"],
                    "direction": spec["direction"],
                    "delta_deletion": del_val,
                    "delta_replacement": rep_val,
                    "passes_threshold": False,
                    "passes_positive_fallback": passes,
                })
                if not passes:
                    positive_pass = False
                    break
            if not positive_pass:
                continue

            passes_safeguards, reports = _passes_safeguards(metric_map)
            if any(r.get('required') and not r['passes'] for r in reports):
                continue
            fallback_safeguard_reports[row_idx] = reports
            fallback_safeguard_status[row_idx] = passes_safeguards
            fallback_indices.append(row_idx)
            fallback_primary_reports[row_idx] = row_primary_reports

        if fallback_indices:
            selected_indices = fallback_indices
            primary_reports = fallback_primary_reports
            safeguard_reports = fallback_safeguard_reports
            safeguard_status = fallback_safeguard_status
            fallback_selection_used = True
            fallback_positive_count = len(fallback_indices)
            selection_mode = "top_positive_fallback"
            warning_text = (
                "No ASOs met the routing threshold for this increase-design site. "
                "The router retained the top candidates with positive predicted splice effect (>0) as a fallback. "
                "Choose these ASOs with caution because the predicted inclusion-supporting effects are weak, "
                "and another splice site may be better to target if available."
            )

    if threshold_pass_count == 0 and allow_zero_fallback:
        fallback_primary_reports = {}
        fallback_safeguard_reports = {}
        fallback_safeguard_status = {}
        fallback_indices = []
        for row_idx in candidate_row_indices:
            row_data = rows[row_idx]
            metric_map = metric_maps[row_idx]
            shared_abs = _parse_float(row_data.get("shared_context_abs_delta"))
            shared_delta = _parse_float(row_data.get("shared_context_delta"))
            shared_label = row_data.get("shared_context_label")
            shared_kind = row_data.get("shared_context_site_kind")
            shared_covers = row_data.get("shared_context_covers_all")
            if _shared_context_blocked(
                shared_label,
                shared_delta,
                shared_abs,
                target_labels=target_labels,
                shared_kind=shared_kind,
                shared_covers_all=bool(shared_covers),
            ):
                continue

            row_primary_reports = []
            for spec in primary_specs:
                metric = metric_map.get(spec["label"])
                del_val = _parse_float(metric.get("delta_deletion")) if metric else None
                rep_val = _parse_float(metric.get("delta_replacement")) if metric else None
                row_primary_reports.append(
                    {
                        "label": spec["label"],
                        "direction": spec["direction"],
                        "delta_deletion": del_val,
                        "delta_replacement": rep_val,
                        "passes_threshold": False,
                        "passes_zero_delta_fallback": True,
                    }
                )
            passes_safeguards, reports = _passes_safeguards(metric_map)
            if any(r.get('required') and not r['passes'] for r in reports):
                continue
            fallback_safeguard_reports[row_idx] = reports
            fallback_safeguard_status[row_idx] = passes_safeguards
            fallback_indices.append(row_idx)
            fallback_primary_reports[row_idx] = row_primary_reports

        if fallback_indices:
            selected_indices = fallback_indices
            primary_reports = fallback_primary_reports
            safeguard_reports = fallback_safeguard_reports
            safeguard_status = fallback_safeguard_status
            fallback_selection_used = True
            fallback_positive_count = 0
            zero_delta_fallback_count = len(fallback_indices)
            selection_mode = "zero_delta_proximity_fallback"
            warning_text = (
                "No ASOs met the requested splice-delta threshold and the evaluated deltas were flat. "
                "The router retained candidates near the requested splice site(s), sorted by proximity plus standard "
                "sequence/off-target quality. Treat these rows as a low-confidence audit set, not model-supported splice hits."
            )

    primary_reports_map = primary_reports
    safeguard_reports_map = safeguard_reports
    safeguard_status_map = safeguard_status
    recommendation_rows = {idx: annotate_route_recommendation(
        dict(rows[idx]), primary_reports_map.get(idx, []), exon_objectives)
        for idx in selected_indices}

    sorted_indices: List[int] = []
    if selected_indices:
        if selection_mode == "zero_delta_proximity_fallback":
            sorted_indices = sorted(
                list(selected_indices),
                key=lambda idx: (
                    _row_target_distance(idx),
                    -(_parse_float(rows[idx].get("Score")) or 0.0),
                    _parse_float(rows[idx].get("Off_target_exact_count")) or 0.0,
                    _parse_float(rows[idx].get("Off_target_total_upto_mismatch_count")) or 0.0,
                    -(_parse_float(rows[idx].get("Phastcon")) or 0.0),
                    -(_parse_float(rows[idx].get("PhyloP")) or 0.0),
                    idx,
                ),
            )
        else:
            sorted_indices = sorted(
                list(selected_indices),
                key=lambda idx: (
                    *recommendation_rows[idx]["recommendation_rank_key"],
                    _repeat_risk_adjusted_score(idx),
                    _primary_effect_score(idx),
                    _parse_float(rows[idx].get("Score")) or 0.0,
                    -idx,
                ),
                reverse=True,
            )
    pre_cap_count = len(sorted_indices)
    max_results_applied: Optional[int] = None
    if max_results_int and pre_cap_count > max_results_int:
        # Keep top-N, but allow strong-effect outliers beyond the cap.
        base_indices = sorted_indices[:max_results_int]
        if primary_specs and len(primary_specs) == 1:
            direction_norm = primary_specs[0]["direction"]
            primary_label_for_effect = primary_specs[0]["label"]
        else:
            direction_norm = "decrease"
            primary_label_for_effect = primary_specs[0]["label"] if primary_specs else None
        strong_extra: List[int] = []
        for idx in sorted_indices[max_results_int:]:
            metric_map = metric_maps[idx]
            metric_single = metric_map.get(primary_label_for_effect) if primary_label_for_effect else None
            del_val = _parse_float(metric_single.get("delta_deletion")) if metric_single else None
            rep_val = _parse_float(metric_single.get("delta_replacement")) if metric_single else None
            best_val = _select_best_for_direction(direction_norm, del_val, rep_val)
            if best_val is None:
                continue
            if direction_norm == "decrease":
                if best_val <= -0.2:
                    strong_extra.append(idx)
            else:
                if best_val >= 0.02:
                    strong_extra.append(idx)
        selected_indices = base_indices + strong_extra
        max_results_applied = max_results_int if strong_extra else max_results_int
    else:
        selected_indices = sorted_indices

    primary_reports = {idx: primary_reports_map.get(idx, []) for idx in selected_indices}
    safeguard_reports = {idx: safeguard_reports_map.get(idx) for idx in selected_indices}
    safeguard_status = {idx: safeguard_status_map.get(idx) for idx in selected_indices}

    selected_rows: List[Dict[str, Any]] = []
    for idx in selected_indices:
        row = rows[idx]
        metric_map = metric_maps[idx]
        row_copy = dict(recommendation_rows[idx])
        for key in (
            "delta_inclusion_deletion",
            "delta_inclusion_replacement",
            "delta_inclusion_deletion_mean",
            "delta_inclusion_replacement_mean",
        ):
            row_copy.pop(key, None)
        if len(primary_specs) == 1:
            row_copy["selected_label"] = primary_specs[0]["label"]
            row_copy["selected_direction"] = primary_specs[0]["direction"]
            metric_single = metric_map.get(primary_specs[0]["label"])
            row_copy["selected_delta_deletion"] = _parse_float(metric_single.get("delta_deletion")) if metric_single else None
            row_copy["selected_delta_replacement"] = _parse_float(metric_single.get("delta_replacement")) if metric_single else None
            row_copy["primary_targets"] = primary_reports.get(idx, [])
        else:
            row_copy["selected_label"] = " & ".join(spec["label"] for spec in primary_specs)
            row_copy["selected_direction"] = "multi"
            row_copy["primary_targets"] = primary_reports.get(idx, [])
        reports = safeguard_reports.get(idx)
        if reports is not None:
            row_copy["safeguard_report"] = reports
            failures = [entry["label"] for entry in reports if entry.get("passes") is False]
            row_copy["safeguards_passed"] = (False if failures else
                None if any(entry.get("passes") is None for entry in reports) else True)
            row_copy.pop("safeguards_failed_labels", None)
            if failures:
                row_copy["safeguards_failed_labels"] = failures
        elif len(primary_specs) > 1:
            row_copy.setdefault("primary_targets", primary_reports.get(idx, []))
        if idx in safeguard_status and "safeguards_passed" not in row_copy:
            row_copy["safeguards_passed"] = safeguard_status[idx]
        if not safeguards or not reports:
            row_copy["safeguards_passed"] = None
            row_copy["safeguards_status"] = "not_configured" if not safeguards else "not_evaluated"
        else:
            status = row_copy.get("safeguards_passed")
            row_copy["safeguards_status"] = "pass" if status is True else "fail" if status is False else "not_evaluated"
        selected_rows.append(row_copy)

    schema_copy_raw = json.loads(json.dumps(schema)) if isinstance(schema, dict) else {}
    schema_copy = schema_copy_raw if isinstance(schema_copy_raw, dict) else {}
    columns = schema_copy.get("columns")
    if not isinstance(columns, dict):
        columns = {}
    columns.pop("delta_inclusion_deletion", None)
    columns.pop("delta_inclusion_replacement", None)
    columns.pop("delta_inclusion_deletion_mean", None)
    columns.pop("delta_inclusion_replacement_mean", None)
    columns.setdefault("selected_label", "string")
    columns.setdefault("selected_direction", "string")
    columns.setdefault("primary_targets", "array")
    if len(primary_specs) == 1:
        columns.setdefault("selected_delta_deletion", "number")
        columns.setdefault("selected_delta_replacement", "number")
    columns.setdefault("safeguard_report", "array")
    columns.setdefault("safeguards_passed", "boolean")
    columns.setdefault("safeguards_status", "string")
    columns.setdefault("safeguards_failed_labels", "array")
    for field in RECOMMENDATION_FIELDS:
        columns[field] = "array" if field in {
            "selected_effect_evidence", "exon_objective_evidence", "recommendation_rank_key", "safeguard_report"
        } else "string"
    schema_copy["columns"] = columns

    output_payload = {
        "schema": schema_copy,
        "rows": selected_rows,
        "safeguards": safeguards,
        "primary_targets": primary_specs,
        "threshold_pass_count": threshold_pass_count,
        "selection_mode": selection_mode,
        "exon_objectives": exon_objectives,
        "recommendation_status_counts": {status: sum(row.get("recommendation_status") == status for row in selected_rows)
            for status in sorted({row.get("recommendation_status") for row in selected_rows})},
    }
    if fallback_selection_used:
        output_payload["fallback_selection_used"] = True
        output_payload["fallback_positive_count"] = fallback_positive_count
    if zero_delta_fallback_count:
        output_payload["zero_delta_fallback_count"] = zero_delta_fallback_count
    if warning_text:
        output_payload["warning"] = warning_text
    if max_results_applied:
        output_payload["max_results_applied"] = max_results_applied
        output_payload["pre_cap_count"] = pre_cap_count
    if constraint_summary:
        output_payload["parameter_constraints"] = constraint_summary
        output_payload["constraint_filtered_count"] = constraint_filtered
        if constraint_failure_counts:
            output_payload["constraint_failure_counts"] = constraint_failure_counts

    final_path = _ensure_output_dir(json_path, output_path)
    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    safeguard_pass_count = sum(row.get("safeguards_passed") is True for row in selected_rows)
    safeguard_fail_count = sum(row.get("safeguards_passed") is False for row in selected_rows)
    failed_labels_summary: Set[str] = set()
    for reports in safeguard_reports.values():
        for entry in reports or []:
            if entry.get("passes") is False:
                label = entry.get("label")
                if label:
                    failed_labels_summary.add(label)

    result: Dict[str, Any] = {
        "primary_label": primary_specs[0]["label"] if len(primary_specs) == 1 else None,
        "primary_direction": primary_specs[0]["direction"] if len(primary_specs) == 1 else None,
        "primary_targets": primary_specs,
        "count": len(selected_rows),
        "total_available": total_available,
        "raw_total_available": raw_total_available,
        "output_path": final_path,
        "threshold_used": threshold,
        "threshold_pass_count": threshold_pass_count,
        "threshold_history": threshold_history,
        "safeguards_applied": safeguards,
        "safeguards_pass_count": safeguard_pass_count,
        "safeguards_fail_count": safeguard_fail_count,
        "safeguards_failed_labels": sorted(failed_labels_summary),
        "selection_mode": selection_mode,
    }
    if fallback_selection_used:
        result["fallback_selection_used"] = True
        result["fallback_positive_count"] = fallback_positive_count
    if zero_delta_fallback_count:
        result["zero_delta_fallback_count"] = zero_delta_fallback_count
    if warning_text:
        result["warning"] = warning_text
    if max_results_applied:
        result["max_results_applied"] = max_results_applied
        result["pre_cap_count"] = pre_cap_count
    if constraint_summary:
        result["parameter_constraints"] = constraint_summary
        result["constraint_filtered_count"] = constraint_filtered
        if constraint_failure_counts:
            result["constraint_failure_counts"] = constraint_failure_counts
    return result




router_planner_agent = Agent(
    name="ASO Strategy Routing Planner",
    instructions=(
        "You design structured execution plans for the splice strategy routing executor (a deterministic Python runner with a limited command vocabulary). "
        "Input provides the routing context (strategy analysis, safeguards, scored tile path). "
        "Your output MUST be valid JSON with the following top-level keys:\n"
        "{\n"
        "  \"plan_summary\": string <=320 chars,\n"
        "  \"steps\": [ ... ],\n"
        "  \"notes\": string?  // optional\n"
        "}\n"
        "If you omit the steps array or return anything other than strict JSON, the router fails. Do NOT include retired fields such as executor_prompt.\n\n"
        "Each element of steps MUST match one of the command schemas below (no other commands are allowed):\n"
        "1) list_site_labels - {\"tool\":\"list_site_labels\", \"args\":{\"json_path\": <scored_tiles_json>}}\n"
        "2) preview_site_metrics - {\"tool\":\"preview_site_metrics\", \"args\":{\"json_path\": <scored_tiles_json>, \"label\": <junction label>, \"limit\": int}}\n"
        "3) auto_threshold_constrained_site_metrics - args MUST include:\n"
        "   {\n"
        "     \"json_path\": <scored_tiles_json>,\n"
        "     \"primary_label\": <junction label>,\n"
        "     \"primary_direction\": \"increase\"|\"decrease\",\n"
        "     \"safeguards_json\": <JSON string>,\n"
        "     \"start_threshold\": float,\n"
        "     \"min_threshold\": float,\n"
        "     \"step\": float,\n"
        "     \"min_count\": int,\n"
        "     \"default_max_abs_delta\": float,\n"
        "     \"output_path\": <absolute path under strategy_routes>\n"
        "   }\n"
        "   Optional: include \"min_gc\", \"max_gc\", and/or \"max_binding_sites\" to enforce parameter constraints.\n"
        "   • When multiple primary junctions must move together, include \"primary_labels_json\": JSON string.\n"
        "4) auto_threshold_filter_site_metrics - args MUST include json_path, label, direction, start_threshold, min_threshold, step, min_count, output_path.\n"
        "   Optional: add \"min_gc\", \"max_gc\", and/or \"max_binding_sites\".\n"
        "5) filter_site_metrics - args MUST include json_path, label, output_path (plus any threshold keys you need).\n"
        "For each filtering step (3-5), ensure output_path is unique and lives under the supplied strategy_routes directory.\n\n"
        "Optional: attach a \"goal\" object to any filtering step using the schema {\"rank\": int, \"goal_label\": str, \"primary_effect\": str, \"primary_targets\": [str], \"off_target_safeguards\": [str], \"notes\": str?}.\n\n"
        "Safeguard guidance:\n"
        "- Safeguards are advisory; log any breaches so downstream spreadsheets flag them, but do not remove tiles solely because a safeguard is exceeded.\n"
        "- When the strategy requires keeping a junction neutral, encode it via safeguards_json. Example: '[{\"label\":\"acceptor@...\",\"max_abs_delta\":0.10}]'.\n"
        "- Only reference labels exactly as they appear in the context.\n"
        "- For every safeguard entry, include a short \"note\" rationale (<=120 chars) explaining why it is needed.\n"
        "- Reasonable defaults (adjust as needed): start_threshold=0.6, min_threshold=0.01, step=0.04, min_count=5, default_max_abs_delta=0.10.\n\n"
        "Example (for illustration only; adapt to target labels):\n"
        "{\n"
        "  \"plan_summary\": \"Suppress donor X, boost donor Y with safeguards.\",\n"
        "  \"steps\": [\n"
        "    {\"tool\":\"list_site_labels\", \"args\":{\"json_path\":\"/abs/path/scored_tiles.json\"}},\n"
        "    {\"tool\":\"auto_threshold_constrained_site_metrics\", \"args\":{\n"
        "       \"json_path\":\"/abs/path/scored_tiles.json\",\n"
        "       \"primary_label\":\"LABELED_DONOR_A\",\n"
        "       \"primary_direction\":\"decrease\",\n"
        "       \"safeguards_json\":\"[{\\\"label\\\":\\\"acceptor_shared\\\",\\\"max_abs_delta\\\":0.1,\\\"note\\\":\\\"shared with beneficial isoform\\\"}]\",\n"
        "       \"start_threshold\":0.6,\n"
        "       \"min_threshold\":0.01,\n"
        "       \"step\":0.04,\n"
        "       \"min_count\":5,\n"
        "       \"default_max_abs_delta\":0.1,\n"
        "       \"output_path\":\"/abs/path/strategy_routes/rank01_donorA.route.json\"\n"
        "    }, \"goal\":{\n"
        "       \"rank\":1,\n"
        "       \"goal_label\":\"Decrease LABELED_DONOR_A\",\n"
        "       \"primary_effect\":\"decrease donor usage\",\n"
        "       \"primary_targets\":[\"LABELED_DONOR_A\"],\n"
        "       \"off_target_safeguards\":[\"Keep acceptor_shared within ±0.1\"],\n"
        "       \"notes\":\"Main suppression bundle\"\n"
        "    }}\n"
        "  ],\n"
        "  \"notes\":\"Lower thresholds if <5 tiles survive while tracking any safeguard breaches.\"\n"
        "}\n\n"
        "Do NOT emit commentary or executor_prompt text-only the JSON object described above. "
        "If no actions are required, emit \"steps\": [] (an empty array), never omit the key. "
        "FAILURE MODES TO AVOID:\n"
        "- Do NOT invent coordinate-style labels (e.g., '5N_E1_acceptor_chr12_51688758'); use the exact strings returned by list_site_labels (e.g., 'exon 1 5N acceptor'). The executor only recognizes those literals.\n"
        "- Pass numeric values as numbers (not strings). Keep safeguards JSON strictly formatted.\n"
        "- Never omit the steps array; when no work is needed, emit \"steps\": [].\n"
        "Failure to comply with this schema will cause the router to halt."
    ),
    tools=[],
    model=PLANNER_MODEL,
    log_tool_calls=False,
    model_settings=PLANNER_MODEL_SETTINGS,
)


splice_strategy_router_planner_agent = router_planner_agent
splice_strategy_router_agent = router_planner_agent


route_ranker_agent = Agent(
    name="ASO Strategy Route Ranker",
    instructions=(
        "You rank splice-modulating routing candidates into the execution order that should become rank01, rank02, and so on.\n"
        "Input JSON: {\"gene_symbol\": string?, \"gene_id\": string?, \"user_request\": string?, \"intent\": string?, \"job\": {...}, \"isoform_objectives\": {...}, \"splice_site_plan\": [...], \"strategy_analysis_excerpt\": string?, \"candidates\": [...]}.\n"
        "Output JSON: {\"ranking_summary\": string?, \"ranked_routes\": [{\"rank\": int, \"label\": string, \"direction\": \"increase\"|\"decrease\", \"rationale\": string, \"confidence\": \"high\"|\"medium\"|\"low\"}], \"excluded_routes\": [{\"label\": string, \"direction\": \"increase\"|\"decrease\", \"reason\": string}]?}.\n"
        "Return only a JSON object. Do not include markdown fences or prose.\n"
        "Rules:\n"
        "- candidates[].direction is the canonical site-action direction. Every ranked or excluded route must copy the direction paired with that exact candidate label, even when job.desired_delta describes a different high-level outcome.\n"
        "- job.desired_delta describes the requested program-level outcome and is ranking context only; it must never replace or reinterpret a candidate's site-action direction.\n"
        "- Do not invent labels, directions, measurements, identifiers, or evidence. Copy labels and supported measurements only from the input JSON.\n"
        "- reference_scope is physical splice-site sharing among its listed compared transcripts. selected_transcript_context describes the design subset only. A single selected transcript cannot establish gene-wide uniqueness. Scope every selectivity claim to the compared catalog; outside-catalog specificity remains unknown. Earlier strategy prose is a hypothesis, not additional reference evidence.\n"
        "- Rank only candidates that are biologically targetable for the request and have plausible requested-direction support; omit off-objective or effectively untargetable candidates and explain them in excluded_routes when useful.\n"
        "- Copy each candidate label and direction exactly as provided.\n"
        "- Optimize for the user's splice objective, not for alphabetical order or anchor status.\n"
        "- Prefer candidates with stronger requested-direction leverage (observed_directional_max_delta and observed_directional_support). Opposite-direction dominance lowers rank or confidence but does not by itself erase an explicitly requested job that has measured requested-direction support.\n"
        "- Demote shared-risk or fallback candidates when metadata shows conflict_between_objectives, shared_with_beneficial, covers_all_isoforms, or terminal_donor_isoforms risk, unless their observed leverage clearly justifies keeping them high.\n"
        "- Use routing_selection_source as context only; anchor_candidate is not automatically best.\n"
        "- Rationales must be specific, concise, and decision-oriented (<=180 characters). Mention the main trade-off or strength driving the rank.\n"
        "- ranking_summary, if provided, should briefly describe the overall ordering logic in <=220 characters."
    ),
    tools=[],
    model=ROUTE_RANKER_MODEL,
    model_settings=ROUTE_RANKER_MODEL_SETTINGS,
)


route_ranker_repair_agent = Agent(
    name="ASO Strategy Route Ranking Contract Repair",
    instructions=(
        "Repair one route-ranking JSON object so it satisfies the supplied canonical candidate catalog and contract errors. "
        "Return exactly the same output shape as the route ranker: ranking_summary, ranked_routes, and optional excluded_routes. "
        "Each ranked_routes item must contain only rank, label, direction, rationale, and confidence. Each excluded_routes item "
        "must contain only label, direction, and reason. Preserve supported ordering and rationales where possible. Every label "
        "and direction must be copied as an exact pair from canonical_candidate_pairs; candidate direction is the site action and "
        "must not be replaced by a different job-level outcome. Drop an item if its label cannot be matched exactly. Do not add "
        "candidates, facts, fields, aliases, or explanations outside the JSON object. Return JSON only."
    ),
    tools=[],
    model=ROUTE_RANKER_MODEL,
    model_settings=ROUTE_RANKER_MODEL_SETTINGS,
)


def _route_ranking_contract_errors(
    source_payload: Dict[str, Any],
    ranking: Dict[str, Any],
) -> List[str]:
    candidate_pairs = {
        (
            str(candidate.get("label") or "").strip(),
            str(candidate.get("direction") or "").strip().lower(),
        )
        for candidate in source_payload.get("candidates") or []
        if isinstance(candidate, dict)
        and str(candidate.get("label") or "").strip()
        and str(candidate.get("direction") or "").strip().lower()
        in {"increase", "decrease"}
    }
    errors: List[str] = []
    ranked_routes = ranking.get("ranked_routes")
    if not isinstance(ranked_routes, list):
        errors.append("ranked_routes must be an array")
        ranked_routes = []
    seen_ranks: Set[int] = set()
    seen_pairs: Set[Tuple[str, str]] = set()
    for index, item in enumerate(ranked_routes):
        if not isinstance(item, dict):
            errors.append(f"ranked_routes[{index}] must be an object")
            continue
        label = str(item.get("label") or "").strip()
        direction = str(item.get("direction") or "").strip().lower()
        pair = (label, direction)
        if pair not in candidate_pairs:
            errors.append(
                f"ranked_routes[{index}] label/direction must be an exact canonical candidate pair"
            )
        if pair in seen_pairs:
            errors.append(f"ranked_routes[{index}] duplicates a candidate pair")
        seen_pairs.add(pair)
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            errors.append(f"ranked_routes[{index}].rank must be a positive integer")
        elif rank in seen_ranks:
            errors.append(f"ranked_routes[{index}].rank must be unique")
        else:
            seen_ranks.add(rank)
        if not str(item.get("rationale") or "").strip():
            errors.append(f"ranked_routes[{index}].rationale must be non-empty")
        if str(item.get("confidence") or "").strip().lower() not in {
            "high",
            "medium",
            "low",
        }:
            errors.append(
                f"ranked_routes[{index}].confidence must be high, medium, or low"
            )
    excluded_routes = ranking.get("excluded_routes")
    if excluded_routes is not None and not isinstance(excluded_routes, list):
        errors.append("excluded_routes must be an array when present")
        excluded_routes = []
    for index, item in enumerate(excluded_routes or []):
        if not isinstance(item, dict):
            errors.append(f"excluded_routes[{index}] must be an object")
            continue
        pair = (
            str(item.get("label") or "").strip(),
            str(item.get("direction") or "").strip().lower(),
        )
        if pair not in candidate_pairs:
            errors.append(
                f"excluded_routes[{index}] label/direction must be an exact canonical candidate pair"
            )
        if not str(item.get("reason") or "").strip():
            errors.append(f"excluded_routes[{index}].reason must be non-empty")
    return errors


def rank_route_candidates_with_llm(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _call_json_agent(route_ranker_agent, payload, max_steps=4)
    parsed = result.get("parsed") if isinstance(result, dict) else {}
    if not isinstance(parsed, dict):
        raise RuntimeError("Route ranker returned an invalid payload.")
    responses = list(result.get("responses") or []) if isinstance(result, dict) else []
    raw_outputs = [str(result.get("raw_output") or "")] if isinstance(result, dict) else []
    contract_errors = _route_ranking_contract_errors(payload, parsed)
    if contract_errors:
        canonical_pairs = [
            {
                "label": str(candidate.get("label") or "").strip(),
                "direction": str(candidate.get("direction") or "").strip().lower(),
            }
            for candidate in payload.get("candidates") or []
            if isinstance(candidate, dict)
            and str(candidate.get("label") or "").strip()
            and str(candidate.get("direction") or "").strip().lower()
            in {"increase", "decrease"}
        ]
        repair_result = _call_json_agent(
            route_ranker_repair_agent,
            {
                "job": deepcopy(payload.get("job") or {}),
                "canonical_candidate_pairs": canonical_pairs,
                "invalid_ranking": deepcopy(parsed),
                "contract_errors": contract_errors,
            },
            max_steps=4,
        )
        repaired = repair_result.get("parsed") if isinstance(repair_result, dict) else {}
        if not isinstance(repaired, dict):
            raise RuntimeError("Route ranking repair returned an invalid payload.")
        repaired_errors = _route_ranking_contract_errors(payload, repaired)
        if repaired_errors:
            raise RuntimeError(
                "Route ranking repair failed contract validation: "
                + "; ".join(repaired_errors)
            )
        parsed = repaired
        responses.extend(repair_result.get("responses") or [])
        raw_outputs.append(str(repair_result.get("raw_output") or ""))
        parsed["_repair_applied"] = True
        parsed["_initial_contract_errors"] = contract_errors
    parsed["_responses"] = responses
    parsed["_raw_output"] = "\n".join(output for output in raw_outputs if output)
    return parsed


EXECUTOR_TOOL_MAP: Dict[str, Callable[..., Dict[str, Any]]] = {
    "list_site_labels": list_site_labels,
    "preview_site_metrics": preview_site_metrics,
    "auto_threshold_constrained_site_metrics": auto_threshold_constrained_site_metrics,
    "auto_threshold_filter_site_metrics": auto_threshold_filter_site_metrics,
    "filter_site_metrics": filter_site_metrics,
}


def execute_router_plan(
    planner_plan: Dict[str, Any],
    scored_tiles_path: str,
    output_dir: str,
    *,
    label_aliases: Optional[Dict[str, str]] = None,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    steps = planner_plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return {
            "error": "executor_no_steps",
            "planner_plan": planner_plan,
        }

    available_labels = _collect_available_labels(scored_tiles_path)
    available_lookup: Dict[str, str] = {label.lower(): label for label in available_labels}
    alias_lookup: Dict[str, str] = {}
    if label_aliases:
        for alias_raw, canonical_raw in label_aliases.items():
            if not isinstance(alias_raw, str) or not isinstance(canonical_raw, str):
                continue
            alias = alias_raw.strip().lower()
            canonical = canonical_raw.strip()
            if not alias or not canonical:
                continue
            canonical_resolved = canonical if canonical in available_labels else available_lookup.get(canonical.lower())
            if not canonical_resolved:
                continue
            alias_lookup.setdefault(alias, canonical_resolved)
    # Ensure canonical labels map to themselves for case-insensitive matching.
    for label in available_labels:
        alias_lookup.setdefault(label.lower(), label)

    step_results: List[Dict[str, Any]] = []
    goals: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, raw_step in enumerate(steps, 1):
        if not isinstance(raw_step, dict):
            errors.append(f"step_{idx}: step must be object")
            continue
        tool_name = raw_step.get("tool")
        if not isinstance(tool_name, str):
            errors.append(f"step_{idx}: missing tool name")
            continue
        args = raw_step.get("args") or {}
        if not isinstance(args, dict):
            errors.append(f"step_{idx}: args must be object")
            continue
        args = dict(args)
        json_path = args.get("json_path") or args.get("tiles_json") or scored_tiles_path

        if tool_name in {"preview_site_metrics", "filter_site_metrics", "auto_threshold_filter_site_metrics"}:
            if "label" in args:
                args["label"] = _canonicalize_label_value(args["label"], available_labels, available_lookup, alias_lookup)
        if tool_name in {"auto_threshold_constrained_site_metrics", "auto_threshold_filter_site_metrics"}:
            if "primary_label" in args:
                args["primary_label"] = _canonicalize_label_value(args["primary_label"], available_labels, available_lookup, alias_lookup)
            if "primary_labels_json" in args:
                args["primary_labels_json"] = _canonicalize_labels_json_string(
                    args.get("primary_labels_json"),
                    available_labels,
                    available_lookup,
                    alias_lookup,
                )
            if "safeguards_json" in args:
                args["safeguards_json"] = _canonicalize_labels_json_string(
                    args.get("safeguards_json"),
                    available_labels,
                    available_lookup,
                    alias_lookup,
                )
        if tool_name == "auto_threshold_filter_site_metrics" and "primary_targets" in args:
            primary_targets_input = args.get("primary_targets") or []
            canonical_primary_targets: List[Dict[str, Any]] = []
            for entry in primary_targets_input:
                if not isinstance(entry, dict):
                    continue
                entry_copy = dict(entry)
                entry_copy["site_label"] = _canonicalize_label_value(
                    entry_copy.get("site_label"),
                    available_labels,
                    available_lookup,
                    alias_lookup,
                )
                canonical_primary_targets.append(entry_copy)
            args["primary_targets"] = canonical_primary_targets
            safeguards_input = args.get("safeguards") or []
            canonical_safeguards: List[Dict[str, Any]] = []
            for entry in safeguards_input:
                if not isinstance(entry, dict):
                    continue
                entry_copy = dict(entry)
                entry_copy["site_label"] = _canonicalize_label_value(
                    entry_copy.get("site_label"),
                    available_labels,
                    available_lookup,
                    alias_lookup,
                )
                canonical_safeguards.append(entry_copy)
            args["safeguards"] = canonical_safeguards

        func = EXECUTOR_TOOL_MAP.get(tool_name)
        call_args: Dict[str, Any] = {}

        if tool_name == "auto_threshold_filter_site_metrics" and "primary_targets" in args:
            primary_targets = args.get("primary_targets") or []
            if not primary_targets:
                errors.append(f"step_{idx}: primary_targets required for {tool_name}")
                continue
            primary = primary_targets[0]
            primary_label = primary.get("site_label")
            primary_label = _canonicalize_label_value(primary_label, available_labels, available_lookup, alias_lookup)
            primary_direction = (primary.get("desired_delta") or "increase").lower()
            if primary_direction not in {"increase", "decrease"}:
                errors.append(f"step_{idx}: invalid desired_delta '{primary_direction}'")
                continue
            primary_targets[0]["site_label"] = primary_label
            safeguards = args.get("safeguards") or []
            safeguard_entries: List[Dict[str, Any]] = []
            for entry in safeguards:
                label = entry.get("site_label")
                if not label:
                    continue
                disallow = [str(val).lower() for val in entry.get("disallow", [])]
                guard: Dict[str, Any] = {"label": label, "note": "planner safeguard"}
                if "increase" in disallow:
                    guard["max_increase"] = 0.05
                if "decrease" in disallow:
                    guard["max_decrease"] = 0.05
                safeguard_entries.append(guard)
            safeguards_json = json.dumps(safeguard_entries)
            base = raw_step.get("output_basename") or raw_step.get("goal", {}).get("goal_label") or f"route_step_{idx}"
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base)) or f"route_step_{idx}"
            output_path = args.get("output_path") or os.path.join(output_dir, f"{safe}.json")
            call_args = {
                "json_path": json_path,
                "primary_label": primary_label,
                "primary_direction": primary_direction,
                "safeguards_json": safeguards_json,
                "primary_labels_json": json.dumps([{"label": primary_label, "direction": primary_direction}]),
                "start_threshold": float(args.get("start_threshold", 0.6)),
                "min_threshold": float(args.get("min_threshold", 0.01)),
                "step": float(args.get("step", 0.04)),
                "min_count": int(args.get("min_count", 5)),
                "default_max_abs_delta": float(args.get("default_max_abs_delta", 0.1)),
                "output_path": output_path,
            }
            func = EXECUTOR_TOOL_MAP["auto_threshold_constrained_site_metrics"]
        else:
            func = EXECUTOR_TOOL_MAP.get(tool_name)
            if not func:
                errors.append(f"step_{idx}: unsupported tool '{tool_name}'")
                continue
            call_args = dict(args)
            call_args.setdefault("json_path", json_path)
            if tool_name in {"auto_threshold_constrained_site_metrics", "auto_threshold_filter_site_metrics", "filter_site_metrics"}:
                if not call_args.get("output_path"):
                    base = raw_step.get("output_basename") or raw_step.get("goal", {}).get("goal_label") or f"route_step_{idx}"
                    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base)) or f"route_step_{idx}"
                    call_args["output_path"] = os.path.join(output_dir, f"{safe}.json")

        try:
            result = func(**call_args)
        except Exception as exc:
            errors.append(f"step_{idx}: tool '{tool_name}' failed ({exc})")
            continue

        step_entry: Dict[str, Any] = {
            "step_index": idx,
            "tool": tool_name,
            "args": args,
            "result": result,
        }

        goal_meta = raw_step.get("goal")
        if isinstance(goal_meta, dict):
            goal_entry = dict(goal_meta)
            goal_entry.setdefault("step_index", idx)
            goal_entry.setdefault("tool", tool_name)
            if "primary_label" in goal_entry:
                goal_entry["primary_label"] = _canonicalize_label_value(
                    goal_entry.get("primary_label"),
                    available_labels,
                    available_lookup,
                    alias_lookup,
                )
            primary_targets_meta = goal_entry.get("primary_targets")
            if isinstance(primary_targets_meta, list):
                canonical_targets_meta: List[Any] = []
                for target in primary_targets_meta:
                    if isinstance(target, str):
                        canonical_targets_meta.append(
                            _canonicalize_label_value(target, available_labels, available_lookup, alias_lookup)
                        )
                    elif isinstance(target, dict):
                        target_copy = dict(target)
                        target_copy["label"] = _canonicalize_label_value(
                            target_copy.get("label"),
                            available_labels,
                            available_lookup,
                            alias_lookup,
                        )
                        target_copy["site_label"] = _canonicalize_label_value(
                            target_copy.get("site_label"),
                            available_labels,
                            available_lookup,
                            alias_lookup,
                        )
                        canonical_targets_meta.append(target_copy)
                    else:
                        canonical_targets_meta.append(target)
                goal_entry["primary_targets"] = canonical_targets_meta
            if isinstance(result, dict):
                output_path = result.get("output_path")
                if output_path:
                    goal_entry["output_path"] = output_path
                if "count" in result:
                    goal_entry["count"] = result.get("count")
                if "threshold_used" in result:
                    goal_entry["threshold_used"] = result.get("threshold_used")
                if "safeguards_applied" in result:
                    goal_entry["safeguards_applied"] = result.get("safeguards_applied")
                for extra_key in (
                    "threshold_pass_count",
                    "selection_mode",
                    "fallback_selection_used",
                    "fallback_positive_count",
                    "zero_delta_fallback_count",
                    "warning",
                ):
                    if extra_key in result:
                        goal_entry[extra_key] = result.get(extra_key)
            goals.append(goal_entry)
            step_entry["goal_meta"] = goal_entry

        step_results.append(step_entry)

    summary: Dict[str, Any] = {
        "steps": step_results,
        "goals": goals,
        "planner_plan": planner_plan,
        "output_dir": output_dir,
        "scored_tiles_path": scored_tiles_path,
    }
    if planner_plan.get("notes"):
        summary["notes"] = planner_plan["notes"]
    if errors:
        summary["executor_errors"] = errors
        summary["error"] = "executor_step_errors"

    if debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception:
            pass
        else:
            exec_log_path = os.path.join(debug_dir, "executor_steps.json")
            with open(exec_log_path, "w", encoding="utf-8") as handle:
                json.dump(step_results, handle, ensure_ascii=False, indent=2)
            summary["executor_steps_path"] = exec_log_path

    return summary

def _extract_isoform_markers(label: str) -> Set[str]:
    """Extract tokens that look like isoform markers (contain both letters and digits)."""
    markers: Set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]+", label or ""):
        if any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
            markers.add(token.lower())
    return markers
