#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aso_agents.splicing_agent import evaluate_splice_shifts_on_tiles


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _format_delta(val: Optional[float]) -> str:
    if isinstance(val, (int, float)):
        return f"{float(val):.4f}"
    return "NA"


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _select_tiles_path(run_dir: Path) -> Path:
    filtered_path = run_dir / "tiling" / "initial_filtering_tiles.json"
    if filtered_path.exists():
        return filtered_path
    return run_dir / "tiling" / "tiles.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score existing tiles with SpliceTransformer.")
    ap.add_argument("--run-dir", required=True, help="Run directory containing structured_goal.json and tiling outputs.")
    ap.add_argument("--tiles", default=None, help="Override tiles JSON path (default: initial_filtering_tiles.json).")
    ap.add_argument("--output", default=None, help="Output scored JSON path (default: tiling/scored_tiles_rescored.json).")
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--site-window-bp", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    structured_goal_path = run_dir / "structured_goal.json"
    if not structured_goal_path.exists():
        raise SystemExit(f"structured_goal.json not found at {structured_goal_path}")

    structured_goal = _load_json(structured_goal_path)
    transcript_id, iso_label = _pick_transcript_id(structured_goal)
    if not transcript_id:
        raise SystemExit("Unable to resolve transcript_id from structured_goal.json")

    tiles_path = Path(args.tiles).expanduser().resolve() if args.tiles else _select_tiles_path(run_dir)
    if not tiles_path.exists():
        raise SystemExit(f"tiles JSON not found at {tiles_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else run_dir / "tiling" / "scored_tiles_rescored.json"

    splice_sites = _build_splice_sites_for_eval(structured_goal, None, window_bp=args.site_window_bp)
    if not splice_sites and iso_label:
        splice_sites = _build_splice_sites_for_eval(structured_goal, iso_label, window_bp=args.site_window_bp)
    splice_sites_json = json.dumps({"sites": splice_sites}, ensure_ascii=False) if splice_sites else None

    patient_reference_fasta = None
    patient_ref_payload = structured_goal.get("patient_reference") if isinstance(structured_goal, dict) else None
    if isinstance(patient_ref_payload, dict):
        patient_reference_fasta = patient_ref_payload.get("fasta_path")

    splicing_eval = evaluate_splice_shifts_on_tiles(
        tiles_json_path=str(tiles_path),
        transcript_id=transcript_id,
        structured_goal_json=json.dumps(structured_goal, ensure_ascii=False),
        pad=args.pad,
        ref_fasta_path=patient_reference_fasta,
        splice_sites_json=splice_sites_json,
        site_window_bp=args.site_window_bp,
    )
    if isinstance(splicing_eval, dict) and splicing_eval.get("error"):
        raise SystemExit(f"Splice eval failed: {splicing_eval.get('error')}")

    tiles_payload = _load_json(tiles_path)
    rows = tiles_payload.get("rows") or []
    results = splicing_eval.get("results") if isinstance(splicing_eval, dict) else None
    if not isinstance(results, list) or not rows:
        raise SystemExit("Splice eval returned no results.")

    splice_sites_meta = splicing_eval.get("splice_sites") if isinstance(splicing_eval, dict) else []
    ordered_sites: List[Dict[str, Any]] = []
    if isinstance(splice_sites_meta, list):
        for site in splice_sites_meta:
            if not isinstance(site, dict):
                continue
            label = site.get("label")
            display_label = site.get("display_label") or label
            site_kind = site.get("site_kind")
            ordered_sites.append({
                "label": label,
                "display_label": display_label,
                "canonical_label": site.get("canonical_label") or display_label,
                "fallback_display_label": site.get("fallback_display_label"),
                "site_kind": site_kind,
                "uniqueness": site.get("uniqueness"),
                "covers_all_isoforms": site.get("covers_all_isoforms"),
            })

    min_len = min(len(rows), len(results))
    site_label_set: set[str] = set()
    for idx in range(min_len):
        row = rows[idx]
        result_entry = results[idx] if isinstance(results[idx], dict) else {}
        delta_del = result_entry.get("delta_inclusion_deletion")
        delta_rep = result_entry.get("delta_inclusion_replacement")
        if delta_del is not None:
            row["delta_inclusion_deletion"] = delta_del
        if delta_rep is not None:
            row["delta_inclusion_replacement"] = delta_rep

        site_deltas = result_entry.get("splice_site_deltas") if isinstance(result_entry, dict) else None
        site_delta_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(site_deltas, list):
            for site_delta in site_deltas:
                if not isinstance(site_delta, dict):
                    continue
                label = site_delta.get("label")
                if label:
                    site_delta_map[str(label)] = site_delta

        delta_names_list: List[str] = []
        delta_del_list: List[str] = []
        delta_rep_list: List[str] = []
        shared_context_label: Optional[str] = None
        shared_context_delta: Optional[float] = None
        shared_context_abs: float = 0.0
        shared_context_site_kind: Optional[str] = None
        shared_context_covers_all: Optional[bool] = None

        for meta in ordered_sites:
            label = meta.get("label")
            display_label = meta.get("display_label") or label or "site"
            canonical_label = meta.get("canonical_label") or display_label
            site_kind = (meta.get("site_kind") or "").lower()
            delta_entry = site_delta_map.get(label or "") or {}
            report_label = canonical_label or display_label
            if report_label:
                site_label_set.add(report_label)
            if site_kind == "acceptor":
                del_val = delta_entry.get("delta_acceptor_deletion")
                rep_val = delta_entry.get("delta_acceptor_replacement")
            elif site_kind == "donor":
                del_val = delta_entry.get("delta_donor_deletion")
                rep_val = delta_entry.get("delta_donor_replacement")
            else:
                del_val = delta_entry.get("delta_inclusion_deletion")
                rep_val = delta_entry.get("delta_inclusion_replacement")
            del_num = _coerce_float(del_val)
            rep_num = _coerce_float(rep_val)
            delta_names_list.append(report_label)
            delta_del_list.append(_format_delta(del_num))
            delta_rep_list.append(_format_delta(rep_num))
            if bool(meta.get("covers_all_isoforms")):
                for candidate in (del_num, rep_num):
                    if candidate is None:
                        continue
                    abs_val = abs(candidate)
                    if abs_val > shared_context_abs:
                        shared_context_abs = abs_val
                        shared_context_delta = candidate
                        shared_context_label = report_label
                        shared_context_site_kind = site_kind if site_kind else None
                        shared_context_covers_all = bool(meta.get("covers_all_isoforms"))

        if delta_names_list:
            row["delta_names"] = ", ".join(delta_names_list)
            row["delta_deletion"] = ", ".join(delta_del_list)
            row["delta_replacement"] = ", ".join(delta_rep_list)
        row["shared_context_label"] = shared_context_label
        row["shared_context_delta"] = shared_context_delta
        row["shared_context_abs_delta"] = shared_context_abs if shared_context_abs > 0 else None
        row["shared_context_site_kind"] = shared_context_site_kind
        row["shared_context_covers_all"] = shared_context_covers_all

    schema = tiles_payload.get("schema") or {}
    columns_map = schema.get("columns") or {}
    columns_map["delta_inclusion_deletion"] = "number"
    columns_map["delta_inclusion_replacement"] = "number"
    columns_map["delta_names"] = "string"
    columns_map["delta_deletion"] = "string"
    columns_map["delta_replacement"] = "string"
    columns_map["shared_context_label"] = "string"
    columns_map["shared_context_delta"] = "number"
    columns_map["shared_context_abs_delta"] = "number"
    columns_map["shared_context_site_kind"] = "string"
    columns_map["shared_context_covers_all"] = "boolean"
    schema["columns"] = columns_map
    tiles_payload["schema"] = schema

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(tiles_payload, handle, ensure_ascii=False, indent=2)

    print(f"[info] SpliceTransformer re-score complete for {min_len} ASOs.")
    print(f"[info] Wrote: {output_path}")
    if site_label_set:
        print(f"[info] Sites: {', '.join(sorted(site_label_set))}")


if __name__ == "__main__":
    main()
