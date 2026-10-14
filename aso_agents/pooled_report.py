"""Evidence-bound, report-only ranking of generated and submitted ASO observations.

The caller supplies a typed comparison contract.  This module never infers a
ranking rule from prose or treats a source-set rank as a pooled rank.  Missing
physical evidence excludes an observation with an explicit reason.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


class ReportContractError(ValueError):
    """The requested report cannot be supported by the supplied evidence."""


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportContractError(f"{label} must be numeric") from exc
    if not 0 <= result <= 1:
        raise ReportContractError(f"{label} must be a fraction from 0 to 1")
    return result


def _int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ReportContractError(f"{label} must be an integer") from exc
    return result


def _sequence(value: Any) -> str:
    sequence = str(value or "").strip().upper().replace("U", "T")
    if not sequence or any(base not in "ACGT" for base in sequence):
        raise ReportContractError("an observation has a missing or invalid DNA sequence")
    return sequence


def _max_run(sequence: str) -> int:
    longest = current = 0
    previous = ""
    for base in sequence:
        current = current + 1 if base == previous else 1
        longest = max(longest, current)
        previous = base
    return longest


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _reference_exon(design_result: Mapping[str, Any], contract: Mapping[str, Any]) -> Dict[str, Any]:
    goal = design_result.get("child_structured_goal") or {}
    transcripts = goal.get("isoform_transcripts") or {}
    requested_id = str(contract["transcript_id"])
    matched = [record for key, record in transcripts.items()
               if key.split(".")[0] == requested_id.split(".")[0] and isinstance(record, dict)]
    if len(matched) != 1:
        raise ReportContractError("locked transcript is missing or ambiguous in saved reference evidence")
    exons = [item for item in matched[0].get("exons_tx") or []
             if isinstance(item, dict) and item.get("index") == _int(contract["exon_ordinal"], "exon_ordinal")]
    if len(exons) != 1 or not all(exons[0].get(key) is not None for key in ("cdna_start", "cdna_end", "start", "end")):
        raise ReportContractError("locked transcript exon has no unambiguous saved coordinates")
    return exons[0]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ReportContractError(f"saved source table is empty: {path.name}")
    return rows


def _correct_generated_overlap(rows: List[Dict[str, str]], exon: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Recompute a displayed overlap from attested BED0 binding intervals."""
    corrections = []
    for row in rows:
        if "requested_exon_overlap_bp" not in row:
            continue
        raw = row.get("physical_binding_intervals_bed0")
        try:
            intervals = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError) as exc:
            raise ReportContractError("invalid saved physical binding intervals") from exc
        if not isinstance(intervals, list) or not intervals:
            raise ReportContractError("missing saved physical binding intervals")
        overlap = 0
        for interval in intervals:
            if not isinstance(interval, list) or len(interval) < 3:
                raise ReportContractError("malformed saved physical binding interval")
            chrom, start, end = str(interval[0]), _int(interval[1], "BED0 start"), _int(interval[2], "BED0 end")
            if chrom.removeprefix("chr") != str(exon.get("seq_region_name") or "").removeprefix("chr"):
                continue
            overlap += max(0, min(end, _int(exon["end"], "exon end"))
                           - max(start, _int(exon["start"], "exon start") - 1))
        old = str(row.get("requested_exon_overlap_bp") or "")
        if old != str(overlap):
            corrections.append({"candidate": row.get("candidate"), "old_overlap_bp": old,
                                "corrected_overlap_bp": overlap})
        row["requested_exon_overlap_bp"] = str(overlap)
    return corrections


def _id_key(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def rank_saved_observations(
    generated_rows: Sequence[Mapping[str, Any]],
    supplied_rows: Sequence[Mapping[str, Any]],
    design_result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Return one row per input observation with traceable eligibility and rank.

    Supported contract: exact antisense binding wholly within a locked exon,
    sequence constraints, then absolute distance of GC fraction from a declared
    target and ascending observation ID.  The contract is explicit and reusable;
    an unsupported metric is rejected rather than silently approximated.
    """
    if contract.get("ranking_metric") != "absolute_gc_distance":
        raise ReportContractError("unsupported pooled ranking metric")
    if contract.get('ranking_order', 'ascending') != 'ascending':
        raise ReportContractError('unsupported pooled ranking order')
    if contract.get('observation_policy', 'preserve_all_origins') != 'preserve_all_origins':
        raise ReportContractError('unsupported pooled observation policy')
    if contract.get("tie_breaker") != "ascending_observation_id":
        raise ReportContractError("unsupported pooled ranking tie-breaker")
    if contract.get("binding_requirement") != "exact_antisense_in_exon":
        raise ReportContractError("unsupported binding requirement")
    exon = _reference_exon(design_result, contract)
    expected_transcript = str(contract["transcript_id"]).split(".")[0]
    length = _int(contract["length_nt"], "length_nt")
    max_run = _int(contract["max_homopolymer_nt"], "max_homopolymer_nt")
    gc_min = _number(contract["gc_min"], "gc_min")
    gc_max = _number(contract["gc_max"], "gc_max")
    gc_target = _number(contract["gc_target"], "gc_target")
    if gc_min > gc_max or length <= 0 or max_run <= 0:
        raise ReportContractError("invalid sequence constraints")

    observations: List[Dict[str, Any]] = []
    for origin, source_rows in (("generated", generated_rows), ("supplied", supplied_rows)):
        for index, source in enumerate(source_rows, 1):
            if origin == "generated":
                observation_id = str(source.get("candidate") or "").strip()
                molecule = _sequence(source.get("antisense_5to3"))
                target = _sequence(source.get("target_5to3"))
                if molecule != _reverse_complement(target):
                    raise ReportContractError("generated ASO does not reverse-complement its target tile")
                exact = True  # A generated native target tile, checked against its mapped reference below.
                transcript = str(source.get("transcript_accession") or "")
                start = _int(source.get("transcript_cdna_start"), "generated cDNA start")
                end = _int(source.get("transcript_cdna_end"), "generated cDNA end")
                mapped_exon = _int(source.get("exon_ordinal"), "generated exon ordinal")
                exact = exact and mapped_exon == _int(contract["exon_ordinal"], "exon_ordinal")
                same_strand = False
                source_row = index
            else:
                observation_id = str(source.get("aso_id") or "").strip()
                molecule = _sequence(source.get("evaluated_molecule_5to3") or source.get("sequence"))
                target = ""
                exact = _int(source.get("physical_exact_antisense_hit_count"), "exact antisense hits") > 0
                same_strand = _int(source.get("exact_same_strand_hit_count"), "same-strand hits") > 0
                transcript = str(source.get("mapping_transcript_id") or "")
                start = _int(source.get("mature_cdna_start_1based"), "supplied cDNA start")
                end = _int(source.get("mature_cdna_end_1based"), "supplied cDNA end")
                source_row = source.get("source_row") or index
            if not observation_id:
                raise ReportContractError("an observation has no stable public ID")
            if end - start + 1 != len(molecule):
                raise ReportContractError(f"cDNA interval length contradicts molecule {observation_id}")
            gc = (molecule.count("G") + molecule.count("C")) / len(molecule)
            run = _max_run(molecule)
            reasons: List[str] = []
            if not exact:
                reasons.append("no exact antisense target" + ("; exact match is same-strand" if same_strand else ""))
            else:
                # A nearest near-match or alternate-orientation placement is
                # not the unchanged molecule's exact antisense target.  Its
                # transcript/coordinates cannot establish exon eligibility
                # and must not be described as the molecule's only mapping.
                if transcript.split(".")[0] != expected_transcript:
                    reasons.append("exact antisense target is not on locked transcript")
                if start < _int(exon["cdna_start"], "exon cDNA start") or end > _int(exon["cdna_end"], "exon cDNA end"):
                    reasons.append("exact antisense target is not wholly inside requested exon")
            if len(molecule) != length:
                reasons.append("sequence length outside requested constraint")
            if not gc_min <= gc <= gc_max:
                reasons.append("GC fraction outside requested range")
            if run > max_run:
                reasons.append("homopolymer exceeds requested maximum")
            observations.append({
                "origin": origin,
                "observation_id": observation_id,
                "source_row": source_row,
                "antisense_5to3": molecule,
                "target_5to3": target,
                "mapping_transcript_id": transcript,
                "mature_cdna_start_1based": start,
                "mature_cdna_end_1based": end,
                "requested_exon_ordinal": contract["exon_ordinal"],
                "exact_antisense_target": exact,
                "exact_same_strand_hit": same_strand,
                "gc_fraction": round(gc, 6),
                "max_homopolymer_run_nt": run,
                "gc_distance_from_target": round(abs(gc - gc_target), 6),
                "eligible": not reasons,
                "exclusion_reason": "; ".join(reasons),
                "pooled_rank": None,
            })
    identities = [(row["origin"], row["observation_id"], row["source_row"]) for row in observations]
    if len(set(identities)) != len(identities):
        raise ReportContractError("duplicate observation identity in saved tables")
    eligible = sorted((row for row in observations if row["eligible"]),
                      key=lambda row: (row["gc_distance_from_target"], _id_key(row["observation_id"]), row["origin"], str(row["source_row"])))
    for rank, row in enumerate(eligible, 1):
        row["pooled_rank"] = rank
    return observations


def render_pooled_report(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> str:
    ordered = sorted(rows, key=lambda row: (row["pooled_rank"] is None,
                                            row["pooled_rank"] or 0, row["origin"], str(row["observation_id"])))
    lines = ["# Pooled ASO observation report", "",
             f"Locked transcript: `{contract['transcript_id']}`; exon: {contract['exon_ordinal']}.",
             f"Eligibility: exact antisense target wholly inside the exon; {contract['length_nt']}-nt sequence; "
             f"GC {contract['gc_min']}–{contract['gc_max']} inclusive; maximum homopolymer {contract['max_homopolymer_nt']} nt.",
             f"Rank eligible observations by absolute GC distance from {contract['gc_target']}, then ascending observation ID. "
             "This is a sequence-metric ordering, not predicted efficacy.", "",
             "| Pooled rank | Origin | Observation ID | ASO 5′→3′ | GC | Distance | Status |",
             "|---:|---|---|---|---:|---:|---|"]
    for row in ordered:
        status = "eligible" if row["eligible"] else row["exclusion_reason"]
        lines.append(f"| {row['pooled_rank'] or '—'} | {row['origin']} | {row['observation_id']} | "
                     f"`{row['antisense_5to3']}` | {row['gc_fraction']:.2f} | "
                     f"{row['gc_distance_from_target']:.2f} | {status} |")
    lines += ["", "Each input row remains a separate observation, including repeated molecules from different origins.",
              "Near matches and opposite-orientation matches are retained as unranked observations.", ""]
    return "\n".join(lines)


def build_saved_report(*, generated_csv: Path, supplied_csv: Path, design_result_json: Path,
                       contract_json: Path, output_dir: Path) -> Dict[str, Any]:
    """Build a versioned report from completed native artifacts; never mutate them."""
    contract = json.loads(contract_json.read_text(encoding="utf-8"))
    design_result = json.loads(design_result_json.read_text(encoding="utf-8"))
    generated_rows = _read_csv(generated_csv)
    supplied_rows = _read_csv(supplied_csv)
    rows = rank_saved_observations(generated_rows, supplied_rows, design_result, contract)
    corrections = _correct_generated_overlap(generated_rows, _reference_exon(design_result, contract))
    output_dir.mkdir(parents=True, exist_ok=False)
    if "requested_exon_overlap_bp" in generated_rows[0]:
        corrected_panel = output_dir / "generated_panel_corrected.csv"
        with corrected_panel.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(generated_rows[0]))
            writer.writeheader()
            writer.writerows(generated_rows)
    table = output_dir / "pooled_observations.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = output_dir / "pooled_report.md"
    report.write_text(render_pooled_report(rows, contract), encoding="utf-8")
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (generated_csv, supplied_csv, design_result_json, contract_json)}
    receipt = {"input_sha256": inputs, "total_observations": len(rows),
               "eligible_count": sum(bool(row["eligible"]) for row in rows),
               "unranked_count": sum(not row["eligible"] for row in rows),
               "ordered_ids": [row["observation_id"] for row in sorted(rows, key=lambda row: row["pooled_rank"] or 10**9)
                               if row["pooled_rank"] is not None],
               "display_overlap_corrections": corrections,
               "original_native_artifacts_unchanged": True,
               "report_kind": "offline_report_only_reconstruction"}
    (output_dir / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
