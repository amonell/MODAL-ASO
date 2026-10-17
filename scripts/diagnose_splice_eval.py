#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _parse_site_deltas(row: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    names_raw = str(row.get("delta_names") or "").split(",")
    deletion_raw = str(row.get("delta_deletion") or "").split(",")
    replacement_raw = str(row.get("delta_replacement") or "").split(",")
    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for idx, name in enumerate(names_raw):
        label = name.strip()
        if not label:
            continue
        del_val = _safe_float(deletion_raw, idx)
        rep_val = _safe_float(replacement_raw, idx)
        metrics[label] = {
            "delta_deletion": del_val,
            "delta_replacement": rep_val,
        }
    return metrics


def _safe_float(series: List[str], idx: int) -> Optional[float]:
    if idx >= len(series):
        return None
    value = series[idx].strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _summarize_scored(scored_path: Path) -> Dict[str, Any]:
    data = _load_json(scored_path)
    rows = data.get("rows") or []
    nan_inclusion_del = 0
    nan_inclusion_rep = 0
    nan_site_tokens = 0
    rows_with_names = 0
    labels_seen: Dict[str, int] = {}
    for row in rows:
        if row.get("delta_inclusion_deletion") is not None and _is_nan(row.get("delta_inclusion_deletion")):
            nan_inclusion_del += 1
        if row.get("delta_inclusion_replacement") is not None and _is_nan(row.get("delta_inclusion_replacement")):
            nan_inclusion_rep += 1
        if row.get("delta_names"):
            rows_with_names += 1
        site_metrics = _parse_site_deltas(row)
        for label, metric in site_metrics.items():
            labels_seen[label] = labels_seen.get(label, 0) + 1
            for key in ("delta_deletion", "delta_replacement"):
                value = metric.get(key)
                if _is_nan(value):
                    nan_site_tokens += 1
    return {
        "rows": len(rows),
        "rows_with_delta_names": rows_with_names,
        "nan_inclusion_deletion": nan_inclusion_del,
        "nan_inclusion_replacement": nan_inclusion_rep,
        "nan_site_token_count": nan_site_tokens,
        "labels": labels_seen,
    }


def _build_subset_tiles(tiles_path: Path, indices: List[int], out_path: Path) -> None:
    data = _load_json(tiles_path)
    rows = data.get("rows") or []
    subset_rows = [rows[i] for i in indices]
    data["rows"] = subset_rows
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "site"


def _pick_isoform_label_from_plan(structured_goal: Dict[str, Any]) -> Optional[str]:
    plan = structured_goal.get("splice_site_plan") or []
    if not isinstance(plan, list):
        return None
    for entry in plan:
        if not isinstance(entry, dict):
            continue
        direction = (entry.get("delta_direction") or "").lower()
        present_in = entry.get("present_in") or []
        if direction in {"increase", "decrease"} and len(present_in) == 1:
            return present_in[0]
    for entry in plan:
        if not isinstance(entry, dict):
            continue
        present_in = entry.get("present_in") or []
        if present_in:
            return present_in[0]
    return None


def _pick_transcript_id(structured_goal: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    direct_id = structured_goal.get("transcript_id")
    if direct_id:
        return direct_id, structured_goal.get("isoform_label")

    isoform_label = _pick_isoform_label_from_plan(structured_goal)
    isoform_transcripts = structured_goal.get("isoform_transcripts") or {}
    if isinstance(isoform_transcripts, dict):
        if isoform_label and isoform_label in isoform_transcripts:
            details = isoform_transcripts.get(isoform_label) or {}
            tx_id = details.get("transcript_id") or details.get("id")
            if not tx_id and isinstance(details.get("metadata"), dict):
                tx_id = details["metadata"].get("id")
            if tx_id:
                return tx_id, isoform_label
        for iso, details in isoform_transcripts.items():
            if not isinstance(details, dict):
                continue
            tx_id = details.get("transcript_id") or details.get("id")
            if not tx_id and isinstance(details.get("metadata"), dict):
                tx_id = details["metadata"].get("id")
            if tx_id:
                return tx_id, iso
    return None, None


def _build_splice_sites_for_eval(
    structured_goal: Dict[str, Any],
    isoform_label: Optional[str],
    window_bp: int,
) -> List[Dict[str, Any]]:
    plan = structured_goal.get("splice_site_plan") or []
    if not isinstance(plan, list) or not plan:
        return []

    targeted_label = isoform_label
    sites: List[Dict[str, Any]] = []

    for entry in plan:
        if not isinstance(entry, dict):
            continue
        present_in = entry.get("present_in") or []
        selected_iso = None
        if targeted_label:
            if targeted_label in present_in:
                selected_iso = targeted_label
            else:
                continue
        else:
            selected_iso = present_in[0] if present_in else None

        if not selected_iso:
            continue

        isoform_transcripts = structured_goal.get("isoform_transcripts") if isinstance(structured_goal, dict) else None
        isoform_details = isoform_transcripts.get(selected_iso) if isinstance(isoform_transcripts, dict) else None

        chrom_hint = entry.get("chrom")
        strand_hint = entry.get("strand") or "+"
        exon_lookup: Dict[int, Dict[str, Any]] = {}
        if isinstance(isoform_details, dict):
            chrom_hint = isoform_details.get("chrom") or chrom_hint
            strand_hint = isoform_details.get("strand") or strand_hint
            for exon in isoform_details.get("exons_tx") or []:
                if not isinstance(exon, dict):
                    continue
                try:
                    idx = int(exon.get("index"))
                except Exception:
                    continue
                exon_lookup[idx] = exon

        def _junctions_for_exon(exon_idx: int) -> Tuple[Optional[int], Optional[int]]:
            exon = exon_lookup.get(exon_idx)
            if not exon:
                return None, None
            try:
                start = int(exon.get("start"))
                end = int(exon.get("end"))
            except Exception:
                return None, None
            strand_local = (exon.get("strand") or strand_hint or "+")
            strand_local = str(strand_local).strip()
            if strand_local in ("-", "-1"):
                acceptor = end
                donor = start
            else:
                acceptor = start
                donor = end
            return acceptor, donor

        exon_indices = entry.get("exon_indices") or {}
        exon_idx = exon_indices.get(selected_iso)
        try:
            exon_idx = int(exon_idx)
        except Exception:
            exon_idx = None
        if exon_idx is None:
            continue

        acceptor_pos, donor_pos = _junctions_for_exon(exon_idx)
        if acceptor_pos is None or donor_pos is None:
            try:
                fallback_pos = int(entry.get("position"))
            except Exception:
                fallback_pos = None
            if fallback_pos is None:
                continue
            if (entry.get("site") or "").lower() == "donor":
                donor_pos = fallback_pos
                acceptor_pos = fallback_pos
            else:
                acceptor_pos = fallback_pos
                donor_pos = fallback_pos

        site_kind = (entry.get("site") or "").strip().lower()
        chrom_entry = entry.get("chrom") or chrom_hint
        strand_entry = entry.get("strand") or strand_hint or "+"
        covers_all = bool(entry.get("covers_all_isoforms"))
        if covers_all:
            fallback_display = f"exon {exon_idx} shared {site_kind}".strip()
        else:
            fallback_display = f"exon {exon_idx} {selected_iso} {site_kind}".strip()

        canonical_label = None
        if site_kind in {"acceptor", "donor"}:
            pos_for_label = acceptor_pos if site_kind == "acceptor" else donor_pos
            if pos_for_label is not None:
                canonical_label = f"{site_kind}@{pos_for_label}"

        label_display = canonical_label or fallback_display
        label_key = _sanitize_label(fallback_display)

        acceptor_window = {
            "chrom": chrom_entry,
            "position": acceptor_pos,
            "start": max(1, acceptor_pos - window_bp),
            "end": acceptor_pos + window_bp,
            "window_bp": window_bp,
            "strand": strand_entry,
        }
        donor_window = {
            "chrom": chrom_entry,
            "position": donor_pos,
            "start": max(1, donor_pos - window_bp),
            "end": donor_pos + window_bp,
            "window_bp": window_bp,
            "strand": strand_entry,
        }

        sites.append(
            {
                "label": label_key,
                "display_label": label_display,
                "site_kind": site_kind,
                "exon_index": exon_idx,
                "isoform_label": selected_iso,
                "delta_direction": entry.get("delta_direction"),
                "window_bp": window_bp,
                "uniqueness": entry.get("uniqueness"),
                "covers_all_isoforms": covers_all,
                "present_in": entry.get("present_in"),
                "absent_in": entry.get("absent_in"),
                "canonical_label": canonical_label,
                "fallback_display_label": fallback_display if fallback_display != label_display else None,
                "acceptor": acceptor_window,
                "donor": donor_window,
            }
        )

    return sites


def _pick_tiles_path(run_dir: Path) -> Path:
    filtered_path = run_dir / "tiling" / "initial_filtering_tiles.json"
    if filtered_path.exists():
        return filtered_path
    return run_dir / "tiling" / "tiles.json"


def _subset_eval(run_dir: Path, sample_size: int, seed: int) -> Dict[str, Any]:
    try:
        from aso_agents.splicing_agent import evaluate_splice_shifts_on_tiles
    except Exception as exc:
        return {"error": f"import_failed: {exc}"}

    structured_goal = _load_json(run_dir / "structured_goal.json")
    transcript_id, iso_label = _pick_transcript_id(structured_goal)
    if not transcript_id:
        return {"error": "transcript_id_unresolved"}

    splice_sites = _build_splice_sites_for_eval(structured_goal, None, window_bp=10)
    if not splice_sites and iso_label:
        splice_sites = _build_splice_sites_for_eval(structured_goal, iso_label, window_bp=10)
    splice_sites_json = json.dumps({"sites": splice_sites}, ensure_ascii=False) if splice_sites else None

    tiles_path = _pick_tiles_path(run_dir)
    scored_path = run_dir / "tiling" / "scored_tiles.json"
    tiles = _load_json(tiles_path)
    rows = tiles.get("rows") or []
    if not rows:
        return {"error": "tiles_empty"}

    rand = random.Random(seed)
    indices = sorted(rand.sample(range(len(rows)), min(sample_size, len(rows))))
    subset_path = run_dir / "tiling" / "tiles_subset_eval.json"
    _build_subset_tiles(tiles_path, indices, subset_path)

    patient_reference_fasta = None
    patient_ref_payload = structured_goal.get("patient_reference") if isinstance(structured_goal, dict) else None
    if isinstance(patient_ref_payload, dict):
        patient_reference_fasta = patient_ref_payload.get("fasta_path")

    try:
        result = evaluate_splice_shifts_on_tiles(
            tiles_json_path=str(subset_path),
            transcript_id=transcript_id,
            structured_goal_json=json.dumps(structured_goal, ensure_ascii=False),
            pad=10,
            ref_fasta_path=patient_reference_fasta,
            splice_sites_json=splice_sites_json,
            site_window_bp=10,
        )
    except Exception as exc:
        return {"error": f"evaluation_failed: {exc}"}
    if isinstance(result, dict) and result.get("error"):
        return {"error": result.get("error")}

    scored = _load_json(scored_path)
    scored_rows = scored.get("rows") or []
    results = result.get("results") or []
    nan_del = 0
    nan_rep = 0
    mismatch = 0
    for local_idx, row_idx in enumerate(indices):
        result_entry = results[local_idx] if local_idx < len(results) else {}
        del_val = result_entry.get("delta_inclusion_deletion")
        rep_val = result_entry.get("delta_inclusion_replacement")
        if _is_nan(del_val):
            nan_del += 1
        if _is_nan(rep_val):
            nan_rep += 1
        if row_idx < len(scored_rows):
            scored_entry = scored_rows[row_idx]
            scored_del = scored_entry.get("delta_inclusion_deletion")
            scored_rep = scored_entry.get("delta_inclusion_replacement")
            if scored_del != del_val or scored_rep != rep_val:
                mismatch += 1

    return {
        "subset_size": len(indices),
        "nan_inclusion_deletion": nan_del,
        "nan_inclusion_replacement": nan_rep,
        "mismatched_rows": mismatch,
        "subset_path": str(subset_path),
        "indices": indices,
        "transcript_id": transcript_id,
        "isoform_label": iso_label,
        "tiles_source": str(tiles_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SpliceTransformer NaNs between runs.")
    parser.add_argument("--run-good", required=True, help="Path to known-good run directory.")
    parser.add_argument("--run-bad", required=True, help="Path to recent run directory.")
    parser.add_argument("--sample-size", type=int, default=25, help="Subset size for re-eval.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for subset selection.")
    args = parser.parse_args()

    run_good = Path(args.run_good).expanduser().resolve()
    run_bad = Path(args.run_bad).expanduser().resolve()

    tiles_good = run_good / "tiling" / "tiles.json"
    tiles_bad = run_bad / "tiling" / "tiles.json"
    tiles_good_filtered = run_good / "tiling" / "initial_filtering_tiles.json"
    tiles_bad_filtered = run_bad / "tiling" / "initial_filtering_tiles.json"
    tiles_good_eval = tiles_good_filtered if tiles_good_filtered.exists() else tiles_good
    tiles_bad_eval = tiles_bad_filtered if tiles_bad_filtered.exists() else tiles_bad
    scored_good = run_good / "tiling" / "scored_tiles.json"
    scored_bad = run_bad / "tiling" / "scored_tiles.json"

    print("== Tiles checksum ==")
    print(f"{tiles_good} sha256={_sha256(tiles_good)[:16]}")
    print(f"{tiles_bad} sha256={_sha256(tiles_bad)[:16]}")
    if tiles_good_filtered.exists():
        print(f"{tiles_good_filtered} sha256={_sha256(tiles_good_filtered)[:16]}")
    if tiles_bad_filtered.exists():
        print(f"{tiles_bad_filtered} sha256={_sha256(tiles_bad_filtered)[:16]}")
    print(f"eval_tiles_good={tiles_good_eval}")
    print(f"eval_tiles_bad={tiles_bad_eval}")
    print()

    print("== Scored summary (good) ==")
    summary_good = _summarize_scored(scored_good)
    print(json.dumps(summary_good, indent=2, sort_keys=True))
    print()

    print("== Scored summary (bad) ==")
    summary_bad = _summarize_scored(scored_bad)
    print(json.dumps(summary_bad, indent=2, sort_keys=True))
    print()

    print("== Subset re-eval (bad run) ==")
    subset = _subset_eval(run_bad, args.sample_size, args.seed)
    print(json.dumps(subset, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
