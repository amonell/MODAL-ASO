from __future__ import annotations

import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .splicing_agent import evaluate_splice_shifts_on_tiles, resolve_splice_eval_backend
from .transcript_chooser_agent import get_transcript_structure
from .repeat_inventory import mapping_inventory, observed_placements


REPEAT_HIGH_RISK_DELTA = 0.10
REPEAT_BORDERLINE_DELTA = 0.02
REPEAT_NEARBY_SPLICE_RADIUS_BP = 2000

REPEAT_RISK_COLUMNS: Dict[str, str] = {
    "Mapped_loci": "array",
    "repeat_hit_count_exact": "number",
    "repeat_hit_count_same_transcript": "number",
    "repeat_hit_count_same_gene": "number",
    "repeat_hit_loci": "string",
    "repeat_hit_transcripts": "string",
    "repeat_hit_exons": "string",
    "repeat_hit_details": "array",
    "repeat_splice_eval_status": "string",
    "repeat_max_abs_delta": "number",
    "repeat_high_risk_sites": "string",
    "repeat_risk_class": "string",
    "repeat_risk_reason": "string",
    "repeat_inventory_status": "string",
    "repeat_inventory_evidence": "object",
    "repeat_observed_hit_count_exact": "number",
    "repeat_observed_hit_count_same_transcript": "number",
    "repeat_observed_hit_count_same_gene": "number",
}


def parse_mapped_loci(mapped_value: Any, *, mismatch_label: str = "exact") -> List[Dict[str, Any]]:
    """Parse ASO-designer mappedto/off-target position strings into structured loci."""
    text = str(mapped_value or "").strip()
    if not text or text.lower() in {"none", "nan", "null", "overflow"}:
        return []
    if "3+matches" in text:
        return []
    if text.lower().startswith("mappedto="):
        text = text.split("=", 1)[1]
    loci: List[Dict[str, Any]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.match(r"^(chr[0-9A-Za-z_.-]+):(\d+)-(\d+)$", token)
        if not match:
            continue
        chrom, start_s, end_s = match.groups()
        try:
            start = int(start_s)
            end = int(end_s)
        except ValueError:
            continue
        if start > end:
            start, end = end, start
        loci.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "mismatch_bin": mismatch_label,
                "locus": f"{chrom}:{start}-{end}",
            }
        )
    return loci


def add_locus_columns_to_row(row: Dict[str, Any]) -> None:
    mapped_value = row.get("Mapped_to") or row.get("mapped_to") or row.get("mappedto")
    exact_loci = parse_mapped_loci(mapped_value, mismatch_label="exact")
    if exact_loci:
        row["Mapped_loci"] = exact_loci
    for label in ("0mm", "1mm", "2mm", "3mm"):
        raw = (
            row.get(f"Off_target_{label}_positions")
            or row.get(f"off_target_{label}_positions")
        )
        loci = parse_mapped_loci(raw, mismatch_label=label)
        if loci:
            row[f"Off_target_{label}_loci"] = loci


def ensure_locus_schema(columns: Dict[str, str]) -> None:
    columns.setdefault("Mapped_loci", "array")
    for label in ("0mm", "1mm", "2mm", "3mm"):
        columns.setdefault(f"Off_target_{label}_loci", "array")


def ensure_repeat_schema(columns: Dict[str, str]) -> None:
    for key, value in REPEAT_RISK_COLUMNS.items():
        columns.setdefault(key, value)


def _norm_chrom(value: Any) -> str:
    chrom = str(value or "").strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom.lower()


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _row_locus(row: Dict[str, Any]) -> Optional[Tuple[str, int, int]]:
    chrom = row.get("Chrom") if row.get("Chrom") is not None else row.get("chrom")
    start = row.get("Start") if row.get("Start") is not None else row.get("start")
    end = row.get("End") if row.get("End") is not None else row.get("end")
    start_i = _safe_int(start)
    end_i = _safe_int(end)
    if chrom is None or start_i is None or end_i is None:
        return None
    return str(chrom), start_i + 1, end_i


def _same_locus(row_loc: Optional[Tuple[str, int, int]], hit: Dict[str, Any]) -> bool:
    if not row_loc:
        return False
    row_chrom, row_start, row_end = row_loc
    hit_start = _safe_int(hit.get("start"))
    hit_end = _safe_int(hit.get("end"))
    if hit_start is None or hit_end is None:
        return False
    if _norm_chrom(row_chrom) != _norm_chrom(hit.get("chrom")):
        return False
    return row_start == hit_start and row_end == hit_end


def _strand_to_int(value: Any, default: int = 1) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if text in {"-", "-1"}:
        return -1
    if text in {"+", "+1", "1"}:
        return 1
    try:
        return 1 if int(text) >= 1 else -1
    except Exception:
        return default


def _extract_exons(entry: Dict[str, Any], fallback_chrom: Optional[str]) -> List[Dict[str, Any]]:
    raw_exons = entry.get("exons_tx") or entry.get("exons_genomic") or entry.get("exons") or []
    if not isinstance(raw_exons, list):
        return []
    exons: List[Dict[str, Any]] = []
    for idx, exon in enumerate(raw_exons, start=1):
        if not isinstance(exon, dict):
            continue
        start = _safe_int(exon.get("start"))
        end = _safe_int(exon.get("end"))
        if start is None or end is None:
            continue
        if start > end:
            start, end = end, start
        chrom = (
            exon.get("seq_region_name")
            or exon.get("chrom")
            or exon.get("chr")
            or fallback_chrom
        )
        exons.append(
            {
                "index": _safe_int(exon.get("index")) or idx,
                "start": start,
                "end": end,
                "chrom": str(chrom) if chrom is not None else "",
            }
        )
    return exons


def build_transcript_catalog(
    structured_goal: Dict[str, Any],
    *,
    primary_transcript_id: Optional[str] = None,
    ref_fasta_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    seen: set[str] = set()

    isoform_transcripts = structured_goal.get("isoform_transcripts") if isinstance(structured_goal, dict) else None
    if isinstance(isoform_transcripts, dict):
        for iso_label, entry in isoform_transcripts.items():
            if not isinstance(entry, dict):
                continue
            transcript_id = str(entry.get("transcript_id") or entry.get("id") or iso_label or "").strip()
            if not transcript_id or transcript_id in seen:
                continue
            chrom = entry.get("seq_region_name") or entry.get("chrom") or (entry.get("metadata") or {}).get("chrom")
            exons = _extract_exons(entry, str(chrom) if chrom is not None else None)
            if not exons:
                continue
            seen.add(transcript_id)
            catalog.append(
                {
                    "transcript_id": transcript_id,
                    "isoform_label": str(iso_label),
                    "strand": _strand_to_int(entry.get("strand") or (entry.get("metadata") or {}).get("strand")),
                    "chrom": str(chrom) if chrom is not None else exons[0].get("chrom"),
                    "exons": exons,
                }
            )

    if primary_transcript_id and primary_transcript_id not in seen:
        try:
            tx = get_transcript_structure(primary_transcript_id, use_reference=ref_fasta_path)
        except Exception:
            tx = {}
        if isinstance(tx, dict) and not tx.get("error"):
            chrom = tx.get("seq_region_name")
            exons = _extract_exons(tx, str(chrom) if chrom is not None else None)
            if exons:
                seen.add(primary_transcript_id)
                catalog.append(
                    {
                        "transcript_id": primary_transcript_id,
                        "isoform_label": "primary",
                        "strand": _strand_to_int(tx.get("strand")),
                        "chrom": str(chrom) if chrom is not None else exons[0].get("chrom"),
                        "exons": exons,
                    }
                )
    return catalog


def _hit_overlaps_exons(hit: Dict[str, Any], transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    strand = str(hit.get("strand") or "")
    if strand in {"+", "-"} and strand != ("+" if _strand_to_int(transcript.get("strand")) == 1 else "-"):
        return []
    if hit.get("genomic_segments"):
        overlaps = []
        for block in hit["genomic_segments"]:
            for exon in _hit_overlaps_exons({**block, "strand":strand}, transcript):
                if exon not in overlaps:
                    overlaps.append(exon)
        return overlaps
    hit_start = _safe_int(hit.get("start"))
    hit_end = _safe_int(hit.get("end"))
    if hit_start is None or hit_end is None:
        return []
    overlaps: List[Dict[str, Any]] = []
    for exon in transcript.get("exons") or []:
        if _norm_chrom(hit.get("chrom")) != _norm_chrom(exon.get("chrom") or transcript.get("chrom")):
            continue
        exon_start = _safe_int(exon.get("start"))
        exon_end = _safe_int(exon.get("end"))
        if exon_start is None or exon_end is None:
            continue
        if hit_end < exon_start or hit_start > exon_end:
            continue
        overlaps.append(exon)
    return overlaps


def _hit_within_gene_span(hit: Dict[str, Any], catalog: Sequence[Dict[str, Any]]) -> bool:
    hit_start = _safe_int(hit.get("start"))
    hit_end = _safe_int(hit.get("end"))
    if hit_start is None or hit_end is None:
        return False
    for transcript in catalog:
        exons = transcript.get("exons") or []
        chroms = {_norm_chrom(exon.get("chrom") or transcript.get("chrom")) for exon in exons}
        if _norm_chrom(hit.get("chrom")) not in chroms:
            continue
        starts = [_safe_int(exon.get("start")) for exon in exons]
        ends = [_safe_int(exon.get("end")) for exon in exons]
        starts_i = [value for value in starts if value is not None]
        ends_i = [value for value in ends if value is not None]
        if starts_i and ends_i and hit_end >= min(starts_i) and hit_start <= max(ends_i):
            return True
    return False


def _exon_junctions(transcript: Dict[str, Any], exon: Dict[str, Any]) -> Tuple[int, int]:
    strand = _strand_to_int(transcript.get("strand"))
    start = int(exon["start"])
    end = int(exon["end"])
    if strand == 1:
        return start, end
    return end, start


def build_splice_sites_for_transcript(transcript: Dict[str, Any], *, window_bp: int = 10) -> List[Dict[str, Any]]:
    sites: List[Dict[str, Any]] = []
    for exon in transcript.get("exons") or []:
        try:
            acceptor, donor = _exon_junctions(transcript, exon)
        except Exception:
            continue
        chrom = exon.get("chrom") or transcript.get("chrom")
        exon_index = exon.get("index")
        strand_char = "+" if _strand_to_int(transcript.get("strand")) == 1 else "-"
        label = f"{transcript.get('transcript_id')}:exon_{exon_index}"
        sites.append(
            {
                "label": label,
                "display_label": label,
                "canonical_label": label,
                "exon_index": exon_index,
                "transcript_id": transcript.get("transcript_id"),
                "window_bp": window_bp,
                "acceptor": {
                    "chrom": chrom,
                    "position": acceptor,
                    "start": max(1, acceptor - window_bp),
                    "end": acceptor + window_bp,
                    "window_bp": window_bp,
                    "strand": strand_char,
                },
                "donor": {
                    "chrom": chrom,
                    "position": donor,
                    "start": max(1, donor - window_bp),
                    "end": donor + window_bp,
                    "window_bp": window_bp,
                    "strand": strand_char,
                },
            }
        )
    return sites


def _hit_has_nearby_splice_site(hit: Dict[str, Any], sites: Sequence[Dict[str, Any]]) -> bool:
    start = _safe_int(hit.get("start"))
    end = _safe_int(hit.get("end"))
    if start is None or end is None:
        return False
    mid = int(round((start + end) / 2))
    for site in sites:
        try:
            acc = int(site["acceptor"]["position"])
            don = int(site["donor"]["position"])
        except Exception:
            continue
        if min(abs(mid - acc), abs(mid - don)) <= REPEAT_NEARBY_SPLICE_RADIUS_BP:
            return True
    return False


def _format_hit(hit: Dict[str, Any]) -> str:
    return f"{hit.get('chrom')}:{hit.get('start')}-{hit.get('end')}"


def _row_sequence(row: Dict[str, Any]) -> str:
    return str(row.get("Sequence") or row.get("sequence") or row.get("seq") or "")


def classify_repeat_hits_for_rows(
    rows: List[Dict[str, Any]],
    structured_goal: Dict[str, Any],
    *,
    primary_transcript_id: Optional[str] = None,
    ref_fasta_path: Optional[str] = None,
    modality: str = "steric",
) -> Dict[str, Any]:
    catalog = build_transcript_catalog(
        structured_goal,
        primary_transcript_id=primary_transcript_id,
        ref_fasta_path=ref_fasta_path,
    )
    primary_id = str(primary_transcript_id or "").strip()
    transcript_by_id = {str(tx.get("transcript_id")): tx for tx in catalog if tx.get("transcript_id")}
    repeat_rows = 0
    same_transcript_hits = 0
    same_gene_hits = 0
    incomplete_rows = 0
    observed = observed_placements(rows)

    for row in rows:
        if not isinstance(row, dict):
            continue
        exact_loci, inventory = mapping_inventory(row, observed, parse_mapped_loci)
        add_locus_columns_to_row(row)
        complete = inventory["complete"]
        incomplete_rows += int(not complete)
        row["repeat_inventory_status"] = inventory["status"]
        row["repeat_inventory_evidence"] = inventory
        details: List[Dict[str, Any]] = []
        for hit in exact_loci:
            if not isinstance(hit, dict):
                continue
            primary_overlaps: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            other_overlaps: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for transcript in catalog:
                overlaps = _hit_overlaps_exons(hit, transcript)
                if not overlaps:
                    continue
                for exon in overlaps:
                    if primary_id and transcript.get("transcript_id") == primary_id:
                        primary_overlaps.append((transcript, exon))
                    else:
                        other_overlaps.append((transcript, exon))

            if primary_overlaps:
                hit_class = "same_transcript_repeat"
                overlaps_for_detail = primary_overlaps
            elif other_overlaps:
                hit_class = "same_gene_other_transcript"
                overlaps_for_detail = other_overlaps
            elif _hit_within_gene_span(hit, catalog):
                hit_class = "same_gene_other_transcript"
                overlaps_for_detail = []
            else:
                off_target_genes = str(row.get("Off_target_genes") or row.get("off_target_genes") or "").strip()
                hit_class = "other_gene" if off_target_genes else "intergenic/unknown"
                overlaps_for_detail = []

            transcript_ids = []
            exon_labels = []
            for transcript, exon in overlaps_for_detail:
                tid = str(transcript.get("transcript_id") or "").strip()
                if tid and tid not in transcript_ids:
                    transcript_ids.append(tid)
                exon_label = f"{tid}:exon_{exon.get('index')}" if tid else f"exon_{exon.get('index')}"
                if exon_label not in exon_labels:
                    exon_labels.append(exon_label)

            detail = dict(hit)
            detail.update(
                {
                    "classification": hit_class,
                    "transcript_ids": transcript_ids,
                    "exon_labels": exon_labels,
                    "is_functional_repeat_candidate": hit_class in {"same_transcript_repeat", "same_gene_other_transcript"},
                }
            )
            details.append(detail)

        same_tx = [d for d in details if d.get("classification") == "same_transcript_repeat"]
        same_gene = [
            d for d in details
            if d.get("classification") in {"same_transcript_repeat", "same_gene_other_transcript"}
        ]
        row["repeat_hit_count_exact"] = len(details) if complete else None
        row["repeat_hit_count_same_transcript"] = len(same_tx) if complete else None
        row["repeat_hit_count_same_gene"] = len(same_gene) if complete else None
        row["repeat_observed_hit_count_exact"] = len(details)
        row["repeat_observed_hit_count_same_transcript"] = len(same_tx)
        row["repeat_observed_hit_count_same_gene"] = len(same_gene)
        row["repeat_hit_loci"] = ";".join(f"{d.get('classification')}:{_format_hit(d)}" for d in details)
        row["repeat_hit_transcripts"] = ";".join(
            sorted({tid for d in details for tid in (d.get("transcript_ids") or []) if tid})
        )
        row["repeat_hit_exons"] = ";".join(
            sorted({label for d in details for label in (d.get("exon_labels") or []) if label})
        )
        row["repeat_hit_details"] = details
        row["repeat_max_abs_delta"] = None
        row["repeat_high_risk_sites"] = ""
        if not details and not complete:
            row["repeat_splice_eval_status"] = "mapping_inventory_unavailable"
            row["repeat_risk_class"] = "not_evaluated"
            row["repeat_risk_reason"] = "A complete exact-locus inventory is unavailable; alternate repeats and their risk are unresolved."
        elif not details:
            row["repeat_splice_eval_status"] = "no_repeat_hits"
            row["repeat_risk_class"] = "none"
            row["repeat_risk_reason"] = "No alternate exact repeat hits were detected."
        elif modality == "gapmer":
            row["repeat_splice_eval_status"] = "gapmer_repeat_splice_eval_skipped"
            row["repeat_risk_class"] = "not_evaluated"
            row["repeat_risk_reason"] = "Repeat hits annotated; splice-modulation repeat scoring is skipped for gapmers by default."
        else:
            row["repeat_splice_eval_status"] = "not_evaluated"
            row["repeat_risk_class"] = "not_evaluated"
            row["repeat_risk_reason"] = "Repeat hits annotated but functional repeat scoring has not run."
        if details and not complete:
            row["repeat_risk_reason"] += " Observed hits are a lower bound; the complete repeat count remains unknown."

        if details:
            repeat_rows += 1
            same_transcript_hits += len(same_tx)
            same_gene_hits += len(same_gene)

    return {
        "repeat_row_count": repeat_rows,
        "same_transcript_hit_count": same_transcript_hits,
        "same_gene_hit_count": same_gene_hits,
        "catalog_transcript_count": len(catalog),
        "catalog_transcript_ids": sorted(transcript_by_id.keys()),
        "incomplete_inventory_row_count": incomplete_rows,
        "count_scope": "Observed annotated hits, not complete totals when inventories are incomplete.",
    }


def _numeric_values(obj: Any) -> Iterable[Tuple[str, float]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (int, float)):
                yield key, float(value)
            elif isinstance(value, (dict, list)):
                for subkey, subvalue in _numeric_values(value):
                    yield f"{key}.{subkey}", subvalue
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            for subkey, subvalue in _numeric_values(item):
                yield f"{idx}.{subkey}", subvalue


def _repeat_risk_class(max_abs_delta: Optional[float], evaluated_count: int, no_nearby_count: int) -> Tuple[str, str, str]:
    if evaluated_count <= 0:
        if no_nearby_count > 0:
            return (
                "low",
                "repeat_hit_no_nearby_splice_site",
                "Alternate repeat hits were far from known splice sites, so splice-modulation risk is lower.",
            )
        return (
            "not_evaluated",
            "repeat_splice_eval_not_run",
            "Repeat hits were found but functional repeat scoring did not run.",
        )
    if max_abs_delta is not None and max_abs_delta >= REPEAT_HIGH_RISK_DELTA:
        return (
            "high",
            "repeat_splice_eval_high_delta",
            f"At least one unintended repeat-hit splice delta was >= {REPEAT_HIGH_RISK_DELTA:.2f}.",
        )
    if max_abs_delta is not None and max_abs_delta >= REPEAT_BORDERLINE_DELTA:
        return (
            "borderline",
            "repeat_splice_eval_borderline_delta",
            f"At least one unintended repeat-hit splice delta was between {REPEAT_BORDERLINE_DELTA:.2f} and {REPEAT_HIGH_RISK_DELTA:.2f}.",
        )
    return (
        "low",
        "repeat_splice_eval_low_delta",
        "Repeat hits were evaluated and did not exceed the borderline splice-delta threshold.",
    )


def _mark_repeat_eval_unsupported_species(
    rows: List[Dict[str, Any]],
    species_value: Any,
) -> Dict[str, Any]:
    affected = 0
    species_text = str(species_value or "").strip() or "unknown"
    for row in rows:
        details = row.get("repeat_hit_details") if isinstance(row.get("repeat_hit_details"), list) else []
        if not details:
            continue
        affected += 1
        row["repeat_splice_eval_status"] = "repeat_splice_eval_unsupported_species"
        row["repeat_risk_class"] = "not_evaluated"
        row["repeat_risk_reason"] = (
            f"Repeat hits were annotated, but no configured splice-evaluation backend exists for species {species_text}."
        )
        row["repeat_max_abs_delta"] = None
        row["repeat_high_risk_sites"] = ""
    return {
        "repeat_eval_task_count": 0,
        "repeat_eval_evaluated_rows": 0,
        "repeat_eval_no_nearby_count": 0,
        "repeat_eval_unsupported_species": species_text,
        "repeat_eval_unsupported_row_count": affected,
    }


def annotate_repeat_risk_for_tiles_json(
    tiles_json_path: str,
    structured_goal: Dict[str, Any],
    *,
    output_path: Optional[str] = None,
    primary_transcript_id: Optional[str] = None,
    ref_fasta_path: Optional[str] = None,
    splice_eval: bool = True,
    modality: str = "steric",
) -> Dict[str, Any]:
    path = Path(str(tiles_json_path or "")).expanduser()
    if not path.exists():
        return {"error": f"tiles_json_missing: {path}"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"tiles_json_read_failed: {exc}"}
    if not isinstance(payload, dict):
        return {"error": "tiles_json_invalid"}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]

    summary = classify_repeat_hits_for_rows(
        rows,
        structured_goal if isinstance(structured_goal, dict) else {},
        primary_transcript_id=primary_transcript_id,
        ref_fasta_path=ref_fasta_path,
        modality=modality,
    )

    goal_obj = structured_goal if isinstance(structured_goal, dict) else {}
    if splice_eval and modality != "gapmer":
        backend_info = resolve_splice_eval_backend(goal_obj.get("species"))
        if backend_info.get("error"):
            eval_summary = _mark_repeat_eval_unsupported_species(rows, goal_obj.get("species"))
        else:
            eval_summary = _evaluate_functional_repeat_risk(
                rows,
                goal_obj,
                primary_transcript_id=primary_transcript_id,
                ref_fasta_path=ref_fasta_path,
            )
            eval_summary["splice_eval_backend"] = backend_info.get("backend")
            eval_summary["splice_eval_species"] = backend_info.get("species")
            eval_summary["splice_eval_model"] = backend_info.get("model")
        summary.update(eval_summary)

    schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
    columns = schema.get("columns") if isinstance(schema.get("columns"), dict) else {}
    ensure_locus_schema(columns)
    ensure_repeat_schema(columns)
    schema["columns"] = columns
    payload["schema"] = schema
    payload["rows"] = rows
    out_path = Path(output_path).expanduser() if output_path else path
    try:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        return {"error": f"tiles_json_write_failed: {exc}"}
    summary["path"] = str(out_path)
    summary["row_count"] = len(rows)
    return summary


def summarize_repeat_risk_for_tiles_json(tiles_json_path: str) -> Dict[str, Any]:
    path = Path(str(tiles_json_path or "")).expanduser()
    if not path.exists():
        return {"error": f"tiles_json_missing: {path}"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"tiles_json_read_failed: {exc}"}
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    class_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    repeat_rows = 0
    same_transcript_hits = 0
    same_gene_hits = 0
    high_risk_rows = 0
    incomplete_rows = 0
    for row in rows:
        incomplete_rows += int(row.get("repeat_inventory_status") != "complete")
        try:
            exact_count = int(row.get("repeat_observed_hit_count_exact", row.get("repeat_hit_count_exact")) or 0)
        except Exception:
            exact_count = 0
        if exact_count > 0:
            repeat_rows += 1
        try:
            same_transcript_hits += int(row.get("repeat_observed_hit_count_same_transcript", row.get("repeat_hit_count_same_transcript")) or 0)
        except Exception:
            pass
        try:
            same_gene_hits += int(row.get("repeat_observed_hit_count_same_gene", row.get("repeat_hit_count_same_gene")) or 0)
        except Exception:
            pass
        risk_class = str(row.get("repeat_risk_class") or "not_evaluated").strip() or "not_evaluated"
        status = str(row.get("repeat_splice_eval_status") or "not_evaluated").strip() or "not_evaluated"
        class_counts[risk_class] = class_counts.get(risk_class, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        if risk_class == "high":
            high_risk_rows += 1
    return {
        "path": str(path),
        "row_count": len(rows),
        "repeat_row_count": repeat_rows,
        "same_transcript_hit_count": same_transcript_hits,
        "same_gene_hit_count": same_gene_hits,
        "high_risk_row_count": high_risk_rows,
        "risk_class_counts": class_counts,
        "status_counts": status_counts,
        "incomplete_inventory_row_count": incomplete_rows,
        "count_scope": "Observed annotated hits; incomplete inventories do not establish zero repeats.",
    }


def _evaluate_functional_repeat_risk(
    rows: List[Dict[str, Any]],
    structured_goal: Dict[str, Any],
    *,
    primary_transcript_id: Optional[str],
    ref_fasta_path: Optional[str],
) -> Dict[str, Any]:
    catalog = build_transcript_catalog(
        structured_goal,
        primary_transcript_id=primary_transcript_id,
        ref_fasta_path=ref_fasta_path,
    )
    by_id = {str(tx.get("transcript_id")): tx for tx in catalog if tx.get("transcript_id")}
    primary_id = str(primary_transcript_id or "").strip()
    if primary_id and primary_id not in by_id and catalog:
        by_id[primary_id] = catalog[0]

    tasks_by_transcript: Dict[str, List[Dict[str, Any]]] = {}
    no_nearby_by_row: Dict[int, int] = {}
    for row_idx, row in enumerate(rows):
        details = row.get("repeat_hit_details") if isinstance(row.get("repeat_hit_details"), list) else []
        for hit_idx, detail in enumerate(details):
            if not isinstance(detail, dict) or not detail.get("is_functional_repeat_candidate"):
                continue
            if len(detail.get("genomic_segments") or []) > 1:
                detail["splice_eval_status"] = "not_evaluated_discontiguous_rna_placement"
                continue  # Never substitute a gap-spanning genomic interval.
            transcript_ids = [str(tid) for tid in (detail.get("transcript_ids") or []) if str(tid).strip()]
            if not transcript_ids and primary_id:
                transcript_ids = [primary_id]
            transcript_id = next((tid for tid in transcript_ids if tid in by_id), "")
            if not transcript_id:
                no_nearby_by_row[row_idx] = no_nearby_by_row.get(row_idx, 0) + 1
                continue
            transcript = by_id[transcript_id]
            sites = build_splice_sites_for_transcript(transcript, window_bp=10)
            if not sites or not _hit_has_nearby_splice_site(detail, sites):
                no_nearby_by_row[row_idx] = no_nearby_by_row.get(row_idx, 0) + 1
                continue
            tasks_by_transcript.setdefault(transcript_id, []).append(
                {
                    "row_index": row_idx,
                    "hit_index": hit_idx,
                    "hit": detail,
                    "transcript": transcript,
                }
            )

    evaluated_by_row: Dict[int, int] = {}
    max_abs_by_row: Dict[int, float] = {}
    high_sites_by_row: Dict[int, List[str]] = {}
    failures: List[str] = []

    for transcript_id, tasks in tasks_by_transcript.items():
        transcript = by_id.get(transcript_id)
        if not transcript:
            continue
        sites = build_splice_sites_for_transcript(transcript, window_bp=10)
        eval_rows: List[Dict[str, Any]] = []
        eval_tasks: List[Dict[str, Any]] = []
        for task_idx, task in enumerate(tasks):
            source_row = rows[task["row_index"]]
            hit = task["hit"]
            seq = _row_sequence(source_row)
            if not seq:
                continue
            start = _safe_int(hit.get("start"))
            end = _safe_int(hit.get("end"))
            if start is None or end is None:
                continue
            synthetic = dict(source_row)
            synthetic["Chrom"] = hit.get("chrom")
            synthetic["Start"] = start - 1
            synthetic["End"] = end
            synthetic["Sequence"] = seq
            synthetic["Strand"] = source_row.get("Strand") or ("+" if _strand_to_int(transcript.get("strand")) == 1 else "-")
            synthetic["_repeat_task_index"] = task_idx
            eval_rows.append(synthetic)
            eval_tasks.append(task)
        if not eval_rows:
            continue

        eval_payload = {
            "schema": {"table": "repeat_hit_synthetic_tiles", "columns": {}},
            "rows": eval_rows,
        }
        with tempfile.TemporaryDirectory(prefix="aso_repeat_eval_") as tmpdir:
            tmp_path = Path(tmpdir) / "repeat_tiles.json"
            tmp_path.write_text(json.dumps(eval_payload, ensure_ascii=False), encoding="utf-8")
            goal_for_eval = deepcopy(structured_goal) if isinstance(structured_goal, dict) else {}
            goal_for_eval.pop("exon_label", None)
            result = evaluate_splice_shifts_on_tiles(
                tiles_json_path=str(tmp_path),
                transcript_id=transcript_id,
                structured_goal_json=json.dumps(goal_for_eval, ensure_ascii=False),
                ref_fasta_path=ref_fasta_path,
                pad=10,
                splice_sites_json=json.dumps({"sites": sites}, ensure_ascii=False),
                site_window_bp=10,
            )
        if not isinstance(result, dict) or result.get("error"):
            failures.append(f"{transcript_id}:{(result or {}).get('error') or 'repeat_eval_failed'}")
            continue
        results = result.get("results") if isinstance(result.get("results"), list) else []
        for task, result_entry in zip(eval_tasks, results):
            if not isinstance(result_entry, dict):
                continue
            row_idx = int(task["row_index"])
            evaluated_by_row[row_idx] = evaluated_by_row.get(row_idx, 0) + 1
            row_max = max_abs_by_row.get(row_idx, 0.0)
            site_names: List[str] = []
            for key, value in _numeric_values(result_entry.get("splice_site_deltas") or []):
                if "delta" not in key:
                    continue
                abs_value = abs(float(value))
                row_max = max(row_max, abs_value)
            for site_delta in result_entry.get("splice_site_deltas") or []:
                if not isinstance(site_delta, dict):
                    continue
                label = str(site_delta.get("canonical_label") or site_delta.get("display_label") or site_delta.get("label") or "").strip()
                site_max = 0.0
                for key, value in _numeric_values(site_delta):
                    if "delta" in key:
                        site_max = max(site_max, abs(float(value)))
                if label and site_max >= REPEAT_HIGH_RISK_DELTA:
                    site_names.append(label)
            if row_max:
                max_abs_by_row[row_idx] = row_max
            if site_names:
                high_sites_by_row.setdefault(row_idx, [])
                for label in site_names:
                    if label not in high_sites_by_row[row_idx]:
                        high_sites_by_row[row_idx].append(label)

    for row_idx, row in enumerate(rows):
        details = row.get("repeat_hit_details") if isinstance(row.get("repeat_hit_details"), list) else []
        if not details or row.get("repeat_risk_class") == "none":
            continue
        evaluated_count = evaluated_by_row.get(row_idx, 0)
        no_nearby_count = no_nearby_by_row.get(row_idx, 0)
        max_abs = max_abs_by_row.get(row_idx)
        risk_class, status, reason = _repeat_risk_class(max_abs, evaluated_count, no_nearby_count)
        row["repeat_splice_eval_status"] = status
        row["repeat_risk_class"] = risk_class
        row["repeat_risk_reason"] = reason
        row["repeat_max_abs_delta"] = max_abs if max_abs is not None else None
        row["repeat_high_risk_sites"] = ";".join(high_sites_by_row.get(row_idx, []))
        if row.get("repeat_inventory_status") != "complete":
            row["repeat_risk_reason"] += " This result covers observed placements only; the complete exact-locus inventory is unavailable."

    if failures:
        for row in rows:
            if row.get("repeat_risk_class") == "not_evaluated" and row.get("repeat_hit_count_exact"):
                row["repeat_splice_eval_status"] = "repeat_splice_eval_failed"
                row["repeat_risk_reason"] = "Functional repeat scoring failed; repeat hits remain annotated."

    return {
        "repeat_eval_task_count": sum(len(tasks) for tasks in tasks_by_transcript.values()),
        "repeat_eval_evaluated_rows": len(evaluated_by_row),
        "repeat_eval_no_nearby_count": sum(no_nearby_by_row.values()),
        "repeat_eval_failures": failures[:10],
    }
