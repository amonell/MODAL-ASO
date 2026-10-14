from agents import Agent, function_tool
import os
import json
from typing import Any, Dict, List, Optional, Tuple


def _load_filtered_rows(json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows") or []
    return data, rows


def _summarize_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        out.append({
            "index": i,
            "Chrom": r.get("Chrom") or r.get("chrom"),
            "Start": r.get("Start") or r.get("start"),
            "End": r.get("End") or r.get("end"),
            "Strand": r.get("Strand") or r.get("strand"),
            "Sequence": r.get("Sequence") or r.get("sequence"),
            "Score": r.get("Score"),
            "GC": r.get("GC"),
            "Phastcon": r.get("Phastcon"),
            "PhyloP": r.get("PhyloP"),
            "No_secondary_structure": r.get("No_secondary_structure"),
            "Times_mapped": r.get("Times_mapped"),
            "Off_target_genes": r.get("Off_target_genes"),
            "Off_target_gene_ids": r.get("Off_target_gene_ids"),
            "delta_inclusion_deletion": r.get("delta_inclusion_deletion"),
            "delta_inclusion_replacement": r.get("delta_inclusion_replacement"),
            "exon_index": r.get("exon_index"),
            "exon_conservation_fraction": r.get("exon_conservation_fraction"),
        })
    return out


@function_tool
def list_candidates(json_path: str) -> Dict[str, Any]:
    """Return a compact summary of the filtered ASO candidates (no JSON writing)."""
    data, rows = _load_filtered_rows(json_path)
    return {"count": len(rows), "candidates": _summarize_candidates(rows)}


@function_tool
def apply_ranking(json_path: str, ranking_indices_json: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Apply a ranking (list of row indices, 0-based) to filtered tiles and write rank + rationale slots.

    ranking_indices_json should be a JSON list of integers indicating candidate order from best to worst.
    Writes 'rank' (1..N) on each row; preserves all other fields.
    """
    try:
        indices = json.loads(ranking_indices_json)
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            raise ValueError("ranking_indices_json must be a JSON list of integers")
    except Exception as e:
        return {"error": f"Invalid ranking_indices_json: {e}"}
    data, rows = _load_filtered_rows(json_path)
    n = len(rows)
    # Clip/unique in order, ignore out-of-range
    seen = set()
    order: List[int] = []
    for i in indices:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    # Append any missing indices at the end in original order
    for i in range(n):
        if i not in seen:
            order.append(i)
    # Apply rank
    rank_of: Dict[int, int] = {idx: (rank + 1) for rank, idx in enumerate(order)}
    for i, row in enumerate(rows):
        row["rank"] = rank_of.get(i)
    # Write output
    if not output_path:
        output_path = os.path.join(os.path.dirname(json_path), "ranked_filtered_tiles.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"schema": data.get("schema", {}), "rows": rows}, f, ensure_ascii=False, indent=2)
    return {"ok": True, "ranked_json_path": output_path, "count": n}


aso_ranker_agent = Agent(
    name="ASO Ranker Agent",
    instructions=(
        "You will rank the final ASO candidates.\n"
        "First, call list_candidates to see a compact summary (with indices).\n"
        "Use exon_index and exon_conservation_fraction (when provided) to prioritize conserved exons for knockdown contexts.\n"
        "Then, respond with ONLY a JSON object: {\"ranking\": [indices in best-to-worst order], \"general_rationale\": string}.\n"
        "Do NOT write or edit any files yourself. After you provide the ranking, a tool will be called to apply it."
    ),
    tools=[list_candidates, apply_ranking],
    model="gpt-5.2-2025-12-11",
    log_tool_calls=False,
) 
