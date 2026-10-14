from agents import Agent, function_tool
import os
import json
from typing import Any, Dict, List, Optional, Tuple

# Allowed columns the agent may inspect/filter
ALLOWED_BASE_COLUMNS = {
    "Chrom", "Start", "End", "Strand",
    "No_homopolymer", "GC", "Phastcon", "PhyloP",
    "No_secondary_structure", "isoform_hit_count", "Times_mapped",
    "Score",
    "exon_index", "exon_conservation_fraction",
    "delta_names", "delta_deletion", "delta_replacement",
    # Splicing evaluation metrics (added downstream)
    "delta_inclusion_deletion", "delta_inclusion_replacement",
}


# Deterministic column definitions and recommended starting guidance (derived from pipeline README)
COLUMN_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "Chrom": {
        "definition": "Chromosome name where the tile maps (e.g., chr17).",
        "recommendation": "Do not filter by Chrom by default."
    },
    "Start": {
        "definition": "0-based start coordinate of the tile in the reference genome.",
        "recommendation": "Do not filter by Start by default."
    },
    "End": {
        "definition": "1-based end coordinate of the tile in the reference genome.",
        "recommendation": "Do not filter by End by default."
    },
    "Strand": {
        "definition": "+ or - strand for the tile in the reference genome.",
        "recommendation": "No default filter; keep both strands unless biology dictates."
    },
    "No_homopolymer": {
        "definition": "Boolean indicating the sequence does not contain homopolymer runs (e.g., CCCCC/ GGGGG).",
        "recommendation": "Prefer True."
    },
    "GC": {
        "definition": "GC fraction of the tile sequence (0-1).",
        "recommendation": "Start with 0.4 <= GC <= 0.7; adjust to reach 5-10 candidates."
    },
    "Phastcon": {
        "definition": "Average phastCons conservation score over the tile (higher = more conserved).",
        "recommendation": "Consider a minimum threshold; begin around 0.9 if distribution supports it."
    },
    "PhyloP": {
        "definition": "Average phyloP conservation score over the tile (higher = more conserved).",
        "recommendation": "Consider a minimum threshold; begin around 0.9 if distribution supports it."
    },
    "No_secondary_structure": {
        "definition": "Boolean indicating predicted lack of stable secondary structure (from RNAfold).",
        "recommendation": "Prefer True."
    },
    "isoform_hit_count": {
        "definition": "Number of annotated target-gene isoforms containing this oligo site. Same meaning as isoform_uniqueness_count on scored gapmer tiles.",
        "recommendation": "For pan-isoform knockdown, higher is better. Prefer 1 only when the goal is isoform-specific targeting.",
    },
    "Times_mapped": {
        "definition": "On gapmer cDNA tiles, a legacy alias for isoform_hit_count (target-gene isoforms containing the site), not a count of off-target genes. Classic ASO-designer genomic uniqueness still uses this name for genomic mapping count.",
        "recommendation": "Do not treat a value above 1 as off-target genes. For pan-isoform knockdown, higher isoform coverage is better.",
    },
    "Score": {
        "definition": "Aggregate score (max 1000). Penalties for failed filters and additional mappings.",
        "recommendation": "Favor higher scores; start with a lower bound (e.g., >=600) and tune."
    },
    "delta_inclusion_deletion": {
        "definition": "Predicted delta in exon inclusion proxy (ALT-REF) if the ASO sequence is deleted at its locus.",
        "recommendation": "Direction depends on intent: splice_modulation(skipping) -> prefer more negative; splice_modulation(inclusion) -> prefer more positive; knockdown -> prefer values near 0 (minimal change)."
    },
    "delta_inclusion_replacement": {
        "definition": "Predicted delta in exon inclusion proxy (ALT-REF) if the ASO sequence is replaced by a neutral motif (ALT has equal length).",
        "recommendation": "Same directional rule as deletion: skipping -> negative; inclusion -> positive; knockdown -> near zero. Use as a corroborating signal."
    },
    "exon_index": {
        "definition": "1-based exon index within the selected transcript for this ASO tile.",
        "recommendation": "Distribute candidates across biologically relevant exons and align with conserved exon priorities supplied in the structured goal."
    },
    "exon_conservation_fraction": {
        "definition": "Fraction of gene isoforms (0-1) that include this exon, derived from conserved_exons_for_transcript.",
        "recommendation": "For knockdown (especially gene-wide), favor higher fractions (>=0.5) to ensure coverage of shared exons."
    },
}


def _load_rows(json_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows") or []
    # Determine available columns from schema when present, else from first row
    schema_cols = list((data.get("schema", {}).get("columns") or {}).keys())
    if not schema_cols and rows:
        schema_cols = list(rows[0].keys())
    # Restrict to allowed base columns that exist in data
    allowed = [c for c in schema_cols if c in ALLOWED_BASE_COLUMNS]
    if "Score" in schema_cols and "Score" not in allowed:
        allowed.append("Score")
    # Fallback: if schema was missing or different case, intersect by case-insensitive match
    if not allowed and rows:
        row_cols = set(rows[0].keys())
        for ac in ALLOWED_BASE_COLUMNS:
            for rc in row_cols:
                if rc.lower() == ac.lower():
                    allowed.append(rc)
                    break
    return rows, allowed


def _infer_primitive_type(values: List[Any]) -> str:
    # Inspect first meaningful values
    sample = [v for v in values if v is not None and str(v) != ""][:10]
    # Try boolean
    def _to_bool(s: Any) -> Optional[bool]:
        if isinstance(s, bool):
            return s
        s2 = str(s).strip().lower()
        if s2 in ("true", "t", "1", "yes"):
            return True
        if s2 in ("false", "f", "0", "no"):
            return False
        return None
    if sample and all(_to_bool(v) is not None for v in sample):
        return "boolean"
    # Try integer/number
    is_int_like = True
    is_num_like = True
    for v in sample:
        try:
            int(str(v))
        except Exception:
            is_int_like = False
            try:
                float(str(v))
            except Exception:
                is_num_like = False
        if not is_num_like:
            break
    if is_int_like:
        return "integer"
    if is_num_like:
        return "number"
    return "string"


def _to_bool_series(values: List[Any]) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    for v in values:
        if isinstance(v, bool):
            out.append(v)
            continue
        s2 = str(v).strip().lower()
        if s2 in ("true", "t", "1", "yes"):
            out.append(True)
        elif s2 in ("false", "f", "0", "no"):
            out.append(False)
        else:
            out.append(None)
    return out


def _parse_numeric_like(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").rstrip("+")
    while text and text[0] in (">", "<", "=", "~", ">=", "<="):
        text = text[1:]
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_float_series(values: List[Any]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in values:
        out.append(_parse_numeric_like(v))
    return out


def _apply_constraints(rows: List[Dict[str, Any]], constraints: List[Dict[str, Any]], allowed_cols: List[str]) -> List[Dict[str, Any]]:
    def normalize_col(col: str) -> Optional[str]:
        for a in allowed_cols:
            if a == col or a.lower() == col.lower():
                return a
        return None

    def check(row: Dict[str, Any]) -> bool:
        for c in constraints:
            col = normalize_col(str(c.get("column", "")))
            if not col:
                return False
            op = str(c.get("op", "")).strip()
            if op not in (">=", ">", "<=", "<", "==", "!=", "in", "between"):
                return False
            rv = row.get(col)
            if op == "in":
                vals = c.get("value") if isinstance(c.get("value"), list) else c.get("values")
                if not isinstance(vals, list):
                    return False
                if rv not in vals:
                    return False
            elif op == "between":
                lo = c.get("min", None)
                hi = c.get("max", None)
                rvf = _parse_numeric_like(rv)
                if rvf is None:
                    return False
                if lo is not None and rvf < float(lo):
                    return False
                if hi is not None and rvf > float(hi):
                    return False
            else:
                lhs = rv
                rhs = c.get("value")
                # Try numeric compare when both look numeric
                lhs_f = _parse_numeric_like(lhs)
                rhs_f = _parse_numeric_like(rhs)
                num_compare = lhs_f is not None and rhs_f is not None
                if num_compare:
                    if op == ">=":
                        if not (lhs_f >= rhs_f):
                            return False
                    elif op == ">":
                        if not (lhs_f > rhs_f):
                            return False
                    elif op == "<=":
                        if not (lhs_f <= rhs_f):
                            return False
                    elif op == "<":
                        if not (lhs_f < rhs_f):
                            return False
                    elif op == "==":
                        if not (lhs_f == rhs_f):
                            return False
                    elif op == "!=":
                        if not (lhs_f != rhs_f):
                            return False
                else:
                    # String/boolean compare
                    if op == "==":
                        if not (str(lhs) == str(rhs)):
                            return False
                    elif op == "!=":
                        if not (str(lhs) != str(rhs)):
                            return False
                    else:
                        return False
        return True

    return [r for r in rows if check(r)]


@function_tool
def list_columns(json_path: str) -> Dict[str, Any]:
    """Return allowed columns available in the dataset at json_path."""
    rows, allowed = _load_rows(json_path)
    return {"columns": allowed}


@function_tool
def column_definitions(json_path: str = "") -> Dict[str, Any]:
    """Return deterministic definitions and starting recommendations for allowed columns (no file read required)."""
    return {"columns": COLUMN_DEFINITIONS}


@function_tool
def starting_constraints(json_path: str, intent: Optional[str] = None, desired_effect: Optional[str] = None) -> Dict[str, Any]:
    """Return a recommended initial set of constraints to reach 5-10 candidates, to be tuned by trial_filter.

    Direction of delta constraints depends on intent/desire:
      - intent == 'knockdown' -> minimize |delta| (near zero)
      - intent == 'splice_modulation' & desired_effect in {'skipping','skip'} -> delta <= -0.02
      - intent == 'splice_modulation' & desired_effect in {'inclusion','include'} -> delta >= +0.02

    The agent should adjust thresholds based on observed distributions.
    """
    # Seed with core constraints; add delta preferences when present
    constraints = [
        {"column": "No_homopolymer", "op": "==", "value": True},
        {"column": "No_secondary_structure", "op": "==", "value": True},
        {"column": "Times_mapped", "op": "==", "value": 1},
        {"column": "GC", "op": "between", "min": 0.4, "max": 0.7},
        {"column": "Score", "op": ">=", "value": 600},
    ]
    try:
        rows, allowed = _load_rows(json_path)
        has_delta_del = "delta_inclusion_deletion" in allowed
        has_delta_rep = "delta_inclusion_replacement" in allowed

        # Determine direction from intent/effect
        intent_norm = (intent or "").strip().lower()
        effect_norm = (desired_effect or "").strip().lower()
        # Defaults if not specified: assume splice_modulation with skipping
        mode = "skip"  # skip | include | knockdown
        if intent_norm == "knockdown":
            mode = "knockdown"
        elif intent_norm == "splice_modulation":
            if effect_norm in ("inclusion", "include", "+inclusion", "+include"):
                mode = "include"
            else:
                mode = "skip"

        has_exon_fraction = any(c.lower() == "exon_conservation_fraction" for c in allowed)

        if has_delta_del:
            if mode == "knockdown":
                constraints.append({"column": "delta_inclusion_deletion", "op": "between", "min": -0.01, "max": 0.01})
            elif mode == "include":
                constraints.append({"column": "delta_inclusion_deletion", "op": ">=", "value": 0.02})
            else:
                constraints.append({"column": "delta_inclusion_deletion", "op": "<=", "value": -0.02})
        if has_delta_rep:
            if mode == "knockdown":
                constraints.append({"column": "delta_inclusion_replacement", "op": "between", "min": -0.01, "max": 0.01})
            elif mode == "include":
                constraints.append({"column": "delta_inclusion_replacement", "op": ">=", "value": 0.02})
            else:
                constraints.append({"column": "delta_inclusion_replacement", "op": "<=", "value": -0.02})
        if mode == "knockdown" and has_exon_fraction:
            constraints.append({"column": "exon_conservation_fraction", "op": ">=", "value": 0.5})
    except Exception:
        pass
    return {"constraints": constraints}


@function_tool
def column_type_and_preview(json_path: str, column: str, n: int = 10) -> Dict[str, Any]:
    """Return inferred type and first n values of a column (restricted to allowed columns)."""
    rows, allowed = _load_rows(json_path)
    if column not in allowed and column.lower() not in [c.lower() for c in allowed]:
        return {"error": f"Column '{column}' is not allowed", "allowed": allowed}
    # Normalize column name
    for c in allowed:
        if c == column or c.lower() == column.lower():
            column = c
            break
    values = [r.get(column) for r in rows]
    inferred = _infer_primitive_type(values)
    preview = values[: max(0, min(n, 10))]
    return {"column": column, "inferred_type": inferred, "preview": preview}


@function_tool
def numeric_distribution(json_path: str, column: str) -> Dict[str, Any]:
    """For a numeric-like column, return count, nan_count, min, max, mean, std, and percentiles (5,25,50,75,95)."""
    rows, allowed = _load_rows(json_path)
    if column not in allowed and column.lower() not in [c.lower() for c in allowed]:
        return {"error": f"Column '{column}' is not allowed", "allowed": allowed}
    # Normalize column name
    for c in allowed:
        if c == column or c.lower() == column.lower():
            column = c
            break
    values = [r.get(column) for r in rows]
    inferred = _infer_primitive_type(values)
    floats = _to_float_series(values)
    nums = [v for v in floats if v is not None]
    if inferred not in ("integer", "number") and len(nums) < 1:
        return {"error": f"Column '{column}' does not appear numeric"}
    if not nums:
        return {"column": column, "count": 0, "nan_count": len(values), "percentiles": {}}
    nums_sorted = sorted(nums)
    import math
    def pct(p: float) -> float:
        if not nums_sorted:
            return float("nan")
        k = (len(nums_sorted) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return nums_sorted[int(k)]
        d0 = nums_sorted[f] * (c - k)
        d1 = nums_sorted[c] * (k - f)
        return d0 + d1
    percentiles = {str(p): pct(p) for p in (5, 25, 50, 75, 95)}
    mean = sum(nums) / len(nums)
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    std = var ** 0.5
    return {
        "column": column,
        "count": len(nums),
        "nan_count": len(values) - len(nums),
        "min": nums_sorted[0],
        "max": nums_sorted[-1],
        "mean": mean,
        "std": std,
        "percentiles": percentiles,
    }


@function_tool
def boolean_value_counts(json_path: str, column: str) -> Dict[str, Any]:
    """For a boolean-like column, return value counts of True and False (others binned as other/nan)."""
    rows, allowed = _load_rows(json_path)
    if column not in allowed and column.lower() not in [c.lower() for c in allowed]:
        return {"error": f"Column '{column}' is not allowed", "allowed": allowed}
    # Normalize column name
    for c in allowed:
        if c == column or c.lower() == column.lower():
            column = c
            break
    values = [r.get(column) for r in rows]
    bs = _to_bool_series(values)
    true_c = sum(1 for v in bs if v is True)
    false_c = sum(1 for v in bs if v is False)
    other_c = sum(1 for v in bs if v is None)
    return {"column": column, "true": true_c, "false": false_c, "other": other_c}


@function_tool
def trial_filter(json_path: str, constraints_json: str) -> Dict[str, Any]:
    """Return row count matching the given constraints without writing output.

    constraints_json should be a JSON list of rules like:
    [{"column":"Score","op":">=","value":600}, {"column":"GC","op":"between","min":0.4,"max":0.7}]
    """
    try:
        constraints = json.loads(constraints_json)
        if not isinstance(constraints, list):
            raise ValueError("constraints_json must be a JSON list")
    except Exception as e:
        return {"error": f"Invalid constraints_json: {e}"}
    rows, allowed = _load_rows(json_path)
    filtered = _apply_constraints(rows, constraints, allowed)
    return {"count": len(filtered)}


@function_tool
def apply_filter(json_path: str, constraints_json: str, output_path: Optional[str] = None, min_count: int = 5, max_count: int = 10, enforce_range: bool = True) -> Dict[str, Any]:
    """Apply constraints and write filtered JSON to output_path (defaults to filtered_tiles.json in same dir).

    Returns {filtered_json_path, count, constraints}.
    If enforce_range, ensures min_count <= count <= max_count.
    """
    try:
        constraints = json.loads(constraints_json)
        if not isinstance(constraints, list):
            raise ValueError("constraints_json must be a JSON list")
    except Exception as e:
        return {"error": f"Invalid constraints_json: {e}"}
    rows, allowed = _load_rows(json_path)
    filtered_rows = _apply_constraints(rows, constraints, allowed)
    count = len(filtered_rows)
    if enforce_range and not (min_count <= count <= max_count):
        return {"ok": False, "count": count, "error": f"Filtered count {count} outside [{min_count}, {max_count}]"}
    # Load original to preserve schema, then replace rows
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_out = {"schema": data.get("schema", {}), "rows": filtered_rows}
    base_dir = os.path.dirname(json_path)
    if not output_path:
        output_path = os.path.join(base_dir, "filtered_tiles.json")
    else:
        if not os.path.isabs(output_path):
            output_path = os.path.join(base_dir, output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)
    return {"ok": True, "filtered_json_path": output_path, "count": count, "constraints": constraints}


aso_choice_agent = Agent(
    name="ASO Choice Agent",
    instructions=(
        "You select 10-30 ASO candidates by inspecting columns and distributions using tools with a json file path. "
        "Never load JSON contents into your messages; only call tools with json_path. "
        "You may only inspect and filter allowed columns. If delta_inclusion_* columns are present, prioritize them first: "
        "- For splice_modulation with exon skipping: prefer more negative deltas. "
        "- For splice_modulation with inclusion: prefer more positive deltas. "
        "- For knockdown: minimize absolute delta (|delta| as small as possible), using both deletion and replacement as corroborating signals. "
        "When exon_index and exon_conservation_fraction columns are available (knockdown), emphasize exons listed in conserved_exons and maintain diversity across them. "
        "Then apply standard quality filters. "
        "First, call 'column_definitions' and 'starting_constraints' (pass intent and desired_effect if known) to review guidance based on the pipeline README and the delta metrics. "
        "Workflow: 1) list_columns; 2) column_type_and_preview; 3) numeric_distribution/boolean_value_counts; 4) trial_filter to tune thresholds; 5) when the candidate count is within 10-30, immediately call apply_filter exactly once to write filtered_tiles.json. "
        "Your FINAL message must be EXACTLY a JSON object with keys: filtered_json_path, constraints, count, rationale (string explaining why these constraints were chosen). No extra text."
    ),
    tools=[
        list_columns,
        column_definitions,
        starting_constraints,
        column_type_and_preview,
        numeric_distribution,
        boolean_value_counts,
        trial_filter,
        apply_filter,
    ],
    model="gpt-5-mini-2025-08-07",
    log_tool_calls=False,
) 
