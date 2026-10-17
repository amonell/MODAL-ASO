#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RowRecord:
    idx: int
    chrom: str
    start: int
    end: int
    strand: str
    row: Dict[str, Any]

    @property
    def length(self) -> int:
        return max(0, self.end - self.start + 1)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_listish(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # fallback split
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _get_off_target_gene_count(row: Dict[str, Any]) -> Optional[int]:
    for key in ("off_target_gene_count", "Off_target_gene_count"):
        val = _parse_int(row.get(key))
        if val is not None:
            return val
    for key in ("Off_target_gene_ids", "off_target_gene_ids"):
        vals = _parse_listish(row.get(key))
        if vals:
            return len(vals)
    return None


def _get_score(row: Dict[str, Any]) -> Optional[float]:
    for key in ("Score", "score", "ASO_score", "score_total"):
        val = _parse_float(row.get(key))
        if val is not None:
            return val
    return None


def _parse_float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, list):
        vals = []
        for item in value:
            num = _parse_float(item)
            if num is not None:
                vals.append(num)
        return vals if vals else None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            vals = []
            for item in parsed:
                num = _parse_float(item)
                if num is not None:
                    vals.append(num)
            return vals if vals else None
    if "," in text:
        vals = []
        for chunk in text.split(","):
            num = _parse_float(chunk)
            if num is not None:
                vals.append(num)
        return vals if vals else None
    num = _parse_float(text)
    return [num] if num is not None else None


def _get_delta_value(row: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        raw = row.get(key)
        if raw is None:
            continue
        # Prefer single float if parseable
        val = _parse_float(raw)
        if val is not None:
            return val
        # Otherwise parse list-like and average
        vals = _parse_float_list(raw)
        if vals:
            return sum(vals) / float(len(vals))
    return None


def _desired_direction_from_primary_targets(primary_targets: Any) -> Optional[str]:
    if not isinstance(primary_targets, list) or not primary_targets:
        return None
    directions = []
    for entry in primary_targets:
        if not isinstance(entry, dict):
            continue
        direction = str(entry.get("direction") or "").strip().lower()
        if direction in {"increase", "decrease"}:
            directions.append(direction)
    if not directions:
        return None
    if all(d == directions[0] for d in directions):
        return directions[0]
    return None


def _directional_delta_score(row: Dict[str, Any], desired_direction: Optional[str]) -> Optional[float]:
    if desired_direction not in {"increase", "decrease"}:
        return None
    delta_del = _get_delta_value(
        row,
        "selected_delta_deletion",
        "delta_deletion",
        "delta_inclusion_deletion",
        "delta_deletion_value",
    )
    delta_rep = _get_delta_value(
        row,
        "selected_delta_replacement",
        "delta_replacement",
        "delta_inclusion_replacement",
        "delta_replacement_value",
    )
    if delta_del is None and delta_rep is None:
        return None
    if delta_del is None:
        mean_delta = delta_rep
    elif delta_rep is None:
        mean_delta = delta_del
    else:
        mean_delta = (delta_del + delta_rep) / 2.0
    if mean_delta is None:
        return None
    return mean_delta if desired_direction == "increase" else -mean_delta


def _representative_key(
    row: Dict[str, Any],
    desired_direction: Optional[str],
    idx: int,
) -> Tuple[float, float, float, float, float, int]:
    dir_score = _directional_delta_score(row, desired_direction)
    has_dir = 1.0 if dir_score is not None else 0.0
    dir_val = dir_score if dir_score is not None else -1e18
    score = _get_score(row)
    score_val = score if score is not None else -1e18
    off_target = _get_off_target_gene_count(row)
    off_val = off_target if off_target is not None else 1e9
    gc_val = _parse_float(row.get("GC") or row.get("gc"))
    gc_penalty = abs(gc_val - 0.5) if gc_val is not None else 1e6
    # Prefer directional delta magnitude in desired direction, then score/off-target/GC.
    return (has_dir, dir_val, score_val, -off_val, -gc_penalty, -idx)


def _rank_representatives(
    indices: List[int],
    records: List[RowRecord],
    desired_direction: Optional[str],
    top_n: int = 2,
) -> List[int]:
    if not indices:
        return []
    ranked = sorted(
        indices,
        key=lambda idx: _representative_key(records[idx].row, desired_direction, idx),
        reverse=True,
    )
    return ranked[: max(1, top_n)]


def _overlap_len(a: RowRecord, b: RowRecord) -> int:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    if hi < lo:
        return 0
    return hi - lo + 1


def _reciprocal_overlap(a: RowRecord, b: RowRecord, threshold: float) -> bool:
    if a.length <= 0 or b.length <= 0:
        return False
    ov = _overlap_len(a, b)
    if ov <= 0:
        return False
    return (ov / a.length) >= threshold and (ov / b.length) >= threshold


def _cluster_records(records: List[RowRecord], threshold: float) -> List[List[int]]:
    if not records:
        return []
    n = len(records)
    uf = UnionFind(n)
    # group by chrom+strand
    groups: Dict[Tuple[str, str], List[int]] = {}
    for idx, rec in enumerate(records):
        key = (rec.chrom, rec.strand)
        groups.setdefault(key, []).append(idx)

    for _key, idxs in groups.items():
        idxs.sort(key=lambda i: (records[i].start, records[i].end))
        for i_pos, i in enumerate(idxs):
            a = records[i]
            for j in idxs[i_pos + 1:]:
                b = records[j]
                if b.start > a.end:
                    break
                if _reciprocal_overlap(a, b, threshold):
                    uf.union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())


def _row_value(row: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def cluster_route_file(path: Path, overlap: float, min_size: int = 1) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    desired_direction = _desired_direction_from_primary_targets(payload.get("primary_targets"))
    records: List[RowRecord] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        chrom = _row_value(row, "Chrom", "chrom")
        start = _parse_int(_row_value(row, "Start", "start"))
        end = _parse_int(_row_value(row, "End", "end"))
        strand = _row_value(row, "Strand", "strand") or ""
        if not chrom or start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        records.append(RowRecord(idx=idx, chrom=str(chrom), start=start, end=end, strand=str(strand), row=row))

    clusters = _cluster_records(records, overlap)
    cluster_entries = []
    for cluster_id, member_indices in enumerate(clusters, start=1):
        if len(member_indices) < min_size:
            continue
        members = [records[i] for i in member_indices]
        chrom = members[0].chrom if members else ""
        strand = members[0].strand if members else ""
        span_start = min(m.start for m in members)
        span_end = max(m.end for m in members)
        rep_indices = _rank_representatives(member_indices, records, desired_direction, top_n=2)
        rep_idx = rep_indices[0] if rep_indices else member_indices[0]
        rep_rec = records[rep_idx]
        cluster_entries.append(
            {
                "cluster_id": cluster_id,
                "chrom": chrom,
                "strand": strand,
                "start": span_start,
                "end": span_end,
                "size": len(member_indices),
                "member_row_indices": [records[i].idx for i in member_indices],
                "representative_row_index": rep_rec.idx,
                "representative_row_indices": [records[i].idx for i in rep_indices],
                "desired_direction": desired_direction,
                "representative": rep_rec.row,
                "representatives": [records[i].row for i in rep_indices],
            }
        )

    return {
        "route_file": path.name,
        "overlap_threshold": overlap,
        "row_count": len(records),
        "desired_direction": desired_direction,
        "clusters": cluster_entries,
    }


def write_representatives_csv(output_path: Path, summaries: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "route_file",
        "cluster_id",
        "rep_rank",
        "size",
        "chrom",
        "strand",
        "cluster_start",
        "cluster_end",
        "desired_direction",
        "rep_row_index",
        "rep_start",
        "rep_end",
        "rep_sequence",
        "rep_delta_avg",
        "rep_delta_direction_score",
        "rep_score",
        "rep_off_target_gene_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            route = summary.get("route_file")
            for cluster in summary.get("clusters") or []:
                desired_direction = cluster.get("desired_direction") or summary.get("desired_direction")
                reps = cluster.get("representatives") or []
                rep_indices = cluster.get("representative_row_indices") or []
                if not reps:
                    reps = [cluster.get("representative") or {}]
                    rep_indices = [cluster.get("representative_row_index")]
                for rank_idx, rep in enumerate(reps, start=1):
                    delta_del = _get_delta_value(
                        rep,
                        "selected_delta_deletion",
                        "delta_deletion",
                        "delta_inclusion_deletion",
                        "delta_deletion_value",
                    )
                    delta_rep = _get_delta_value(
                        rep,
                        "selected_delta_replacement",
                        "delta_replacement",
                        "delta_inclusion_replacement",
                        "delta_replacement_value",
                    )
                    if delta_del is None and delta_rep is None:
                        delta_avg = None
                    elif delta_del is None:
                        delta_avg = delta_rep
                    elif delta_rep is None:
                        delta_avg = delta_del
                    else:
                        delta_avg = (delta_del + delta_rep) / 2.0
                    dir_score = _directional_delta_score(rep, desired_direction)
                    rep_row_index = rep_indices[rank_idx - 1] if rank_idx - 1 < len(rep_indices) else None
                    writer.writerow(
                        {
                            "route_file": route,
                            "cluster_id": cluster.get("cluster_id"),
                            "rep_rank": rank_idx,
                            "size": cluster.get("size"),
                            "chrom": cluster.get("chrom"),
                            "strand": cluster.get("strand"),
                            "cluster_start": cluster.get("start"),
                            "cluster_end": cluster.get("end"),
                            "desired_direction": desired_direction,
                            "rep_row_index": rep_row_index,
                            "rep_start": _parse_int(_row_value(rep, "Start", "start")),
                            "rep_end": _parse_int(_row_value(rep, "End", "end")),
                            "rep_sequence": _row_value(rep, "Sequence", "sequence"),
                            "rep_delta_avg": delta_avg,
                            "rep_delta_direction_score": dir_score,
                            "rep_score": _get_score(rep),
                            "rep_off_target_gene_count": _get_off_target_gene_count(rep),
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster strategy route ASOs by 80% reciprocal overlap.")
    parser.add_argument("--run-dir", required=True, help="Path to a completed run directory.")
    parser.add_argument("--overlap", type=float, default=0.80, help="Reciprocal overlap threshold (default 0.80).")
    parser.add_argument("--min-cluster-size", type=int, default=1, help="Minimum cluster size to report.")
    parser.add_argument("--output-dir", default=None, help="Output directory (default <run_dir>/strategy_route_clusters).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    route_dir = run_dir / "strategy_routes"
    if not route_dir.exists():
        raise SystemExit(f"strategy_routes not found under {run_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "strategy_route_clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for path in sorted(route_dir.glob("*.route.json")):
        summary = cluster_route_file(path, overlap=args.overlap, min_size=args.min_cluster_size)
        summaries.append(summary)
        out_path = output_dir / f"{path.stem}.clusters.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    reps_csv = output_dir / "cluster_representatives.csv"
    write_representatives_csv(reps_csv, summaries)

    index_path = output_dir / "clusters_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_dir": str(run_dir),
                "overlap_threshold": args.overlap,
                "min_cluster_size": args.min_cluster_size,
                "route_count": len(summaries),
                "routes": [s.get("route_file") for s in summaries],
                "outputs": {
                    "cluster_representatives_csv": str(reps_csv),
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(summaries)} cluster summaries to {output_dir}")
    print(f"Representative CSV: {reps_csv}")


if __name__ == "__main__":
    main()
