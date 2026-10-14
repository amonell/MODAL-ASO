#!/usr/bin/env python3
"""
Export selected ASO features and transcript exons into a GenBank file for SnapGene.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from aso_agents.reporting_contracts import sequence_notes

from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Bio.Seq import Seq  # type: ignore
from Bio.SeqFeature import CompoundLocation, SeqFeature, FeatureLocation  # type: ignore
from Bio.SeqRecord import SeqRecord  # type: ignore
from Bio import SeqIO  # type: ignore
from pyfaidx import Fasta  # type: ignore
from aso_agents.species_config import get_species_config, normalize_species
from aso_agents.run_layout import canonical_run_path, resolve_run_path


RUN_DEFAULT = Path("runs")
DEFAULT_SPECIES = "human"
GENOME_FASTA = get_species_config(DEFAULT_SPECIES).genome_fasta

ISOFORM_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

STRATEGY_COLORS = {
    "primary": "#E41A1C",
    "secondary": "#377EB8",
    "conditional": "#4DAF4A",
    "backup": "#984EA3",
    "tertiary": "#FF7F00",
}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitise_isoform_label(label: str) -> str:
    if not label:
        return "isoform"
    name = Path(label).name
    return name or label


def resolve_isoform_labels(structured_goal: Dict) -> List[str]:
    labels: List[str] = []
    objectives = structured_goal.get("splice_isoform_objectives") or []
    for obj in objectives:
        iso_label = obj.get("isoform_label")
        if iso_label and iso_label not in labels:
            labels.append(iso_label)
    if not labels:
        labels = list((structured_goal.get("isoform_transcripts") or {}).keys())
    if not labels:
        labels = list((structured_goal.get("all_transcripts_full") or {}).keys())
    return labels


def compute_gene_region(isoform_transcripts: Dict[str, Dict], isoform_labels: Iterable[str]) -> Tuple[str, int, int, int]:
    chrom_name: Optional[str] = None
    strand_val: Optional[int] = None
    start_min: Optional[int] = None
    end_max: Optional[int] = None

    for label in isoform_labels:
        info = isoform_transcripts.get(label)
        if not isinstance(info, dict):
            continue
        chrom = info.get("chrom") or info.get("seq_region_name")
        if chrom:
            chrom_str = str(chrom)
            if chrom_str.lower().startswith("chr"):
                chrom_norm = chrom_str
            else:
                chrom_norm = f"chr{chrom_str}"
            chrom_name = chrom_name or chrom_norm
        strand_raw = info.get("strand")
        if strand_raw in ("+", "+1", 1):
            strand = 1
        elif strand_raw in ("-", "-1", -1):
            strand = -1
        else:
            strand = 1
        strand_val = strand_val or strand
        for exon in info.get("exons_tx") or []:
            try:
                ex_start = int(exon.get("start"))
                ex_end = int(exon.get("end"))
            except Exception:
                continue
            if start_min is None or ex_start < start_min:
                start_min = ex_start
            if end_max is None or ex_end > end_max:
                end_max = ex_end

    if chrom_name is None or start_min is None or end_max is None:
        raise ValueError("Unable to determine gene region from isoform transcripts.")
    # extend padding
    padding = 300
    padded_start = max(1, start_min - padding)
    padded_end = end_max + padding
    return chrom_name, strand_val or 1, padded_start, padded_end


def fetch_genomic_sequence(chrom: str, start: int, end: int, strand: int, fasta_path: Path) -> Seq:
    if not fasta_path.exists():
        raise FileNotFoundError(f"Reference FASTA not found at {fasta_path}")
    fasta = Fasta(str(fasta_path))
    if chrom not in fasta:
        raise KeyError(f"Chromosome {chrom} not present in FASTA.")
    # pyfaidx is 0-based, end exclusive when slicing
    seq = fasta[chrom][start - 1 : end].seq.upper()
    return Seq(seq)


def color_for_index(index: int) -> str:
    return ISOFORM_COLORS[index % len(ISOFORM_COLORS)]


def build_isoform_features(
    isoform_transcripts: Dict[str, Dict],
    isoform_labels: List[str],
    region_start: int,
) -> List[SeqFeature]:
    features: List[SeqFeature] = []
    for idx, label in enumerate(isoform_labels):
        info = isoform_transcripts.get(label)
        if not isinstance(info, dict):
            continue
        color = color_for_index(idx)
        clean_label = sanitise_isoform_label(label)
        strand_raw = info.get("strand")
        if strand_raw in ("-", "-1", -1):
            strand = -1
        else:
            strand = 1
        for exon in info.get("exons_tx") or []:
            try:
                exon_index = exon.get("index")
                ex_start = int(exon.get("start"))
                ex_end = int(exon.get("end"))
            except Exception:
                continue
            start_rel = ex_start - region_start
            end_rel = ex_end - region_start + 1
            if start_rel < 0 or end_rel <= start_rel:
                continue
            location = FeatureLocation(start_rel, end_rel, strand=strand)
            qualifiers = {
                "label": [f"{clean_label} exon {exon_index}"],
                "note": [f"Isoform: {clean_label}; exon {exon_index}"],
                "color": [color],
                "ApEinfo_fwdcolor": [color],
                "ApEinfo_revcolor": [color],
            }
            features.append(SeqFeature(location=location, type="misc_feature", qualifiers=qualifiers))
    return features


def parse_strategy_summary(summary_path: Path) -> List[Dict[str, Any]]:
    data = load_json(summary_path)
    return data.get("goals") or []


def strategy_prefix(goal_label: str) -> str:
    return (goal_label or "").split("_", 1)[0].lower()


def color_for_strategy(prefix: str) -> str:
    return STRATEGY_COLORS.get(prefix, "#555555")


def read_route_csv(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_aso_features(
    strategy_goals: List[Dict],
    region_start: int,
    region_end: int,
) -> List[SeqFeature]:
    features: List[SeqFeature] = []
    seen_locs: Dict[Tuple[int, int, int], int] = defaultdict(int)

    def _get_first(row: Dict[str, Any], *keys: str) -> Optional[str]:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return value
        return None

    def _get_int(row: Dict[str, Any], *keys: str) -> Optional[int]:
        value = _get_first(row, *keys)
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    for goal in strategy_goals:
        count = goal.get("count") or 0
        if count <= 0:
            continue
        csv_path = goal.get("csv_path")
        goal_label = goal.get("goal_label") or "strategy"
        description = goal.get("description") or ""
        prefix = strategy_prefix(goal_label)
        color = color_for_strategy(prefix)
        if not csv_path:
            continue
        rows = read_route_csv(Path(csv_path))
        for row in rows:
            start = _get_int(row, "Start", "start_bp")
            end = _get_int(row, "End", "end_bp")
            if start is None or end is None:
                continue
            strand_char = _get_first(row, "Strand", "strand") or "+"
            strand = -1 if str(strand_char).strip() == "-" else 1
            seq_len = max(0, end - start)
            # Native steric route tables are BED0 half-open; region_start is
            # the one-based inclusive origin of the genomic sequence record.
            start_rel = start - (region_start - 1)
            end_rel = start_rel + seq_len
            if start_rel < 0 or end_rel > (region_end - region_start + 1):
                continue
            key = (start_rel, end_rel, strand)
            seen_locs[key] += 1
            suffix = f" #{seen_locs[key]}" if seen_locs[key] > 1 else ""
            location = FeatureLocation(start_rel, end_rel, strand=strand)
            label = f"{goal_label}{suffix}"
            seq = _get_first(row, "Sequence", "sequence") or ""
            score = _get_first(row, "Score", "tile_score")
            qualifiers = {
                "label": [label],
                "note": [
                    f"Strategy: {goal_label}",
                    f"Description: {description}",
                    *sequence_notes(row),
                ],
                "color": [color],
                "ApEinfo_fwdcolor": [color],
                "ApEinfo_revcolor": [color],
            }
            extra_fields = [
                ("selected_delta_replacement", "primary_delta_replacement", "delta_peak_replacement"),
                ("selected_delta_deletion", "primary_delta_deletion", "delta_peak_deletion"),
                ("Off_target_expression_sum", "off_target_expression_sum"),
                ("Off_target_expression_ratio", "off_target_expression_ratio"),
                ("Target_gene_expression", "target_gene_expression"),
            ]
            for aliases in extra_fields:
                canonical = aliases[0]
                value = _get_first(row, *aliases)
                if value not in (None, "", "None"):
                    qualifiers.setdefault("note", []).append(f"{canonical}={value}")
            if score not in (None, "", "None"):
                qualifiers.setdefault("note", []).append(f"Score={score}")
            features.append(SeqFeature(location=location, type="misc_feature", qualifiers=qualifiers))
    return features


def _generated_row_segments(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_segments = row.get("Genomic_segments") or row.get("genomic_segments")
    if isinstance(raw_segments, str):
        try:
            raw_segments = json.loads(raw_segments)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_segments = None
    if isinstance(raw_segments, list):
        segments = [item for item in raw_segments if isinstance(item, dict)]
        if segments:
            return segments
    return [row]


def build_generated_aso_features(
    generated_rows: List[Dict[str, Any]],
    chrom: str,
    region_start: int,
    region_end: int,
    default_strand: int,
) -> List[SeqFeature]:
    """Build target-site features from the canonical generated candidate set.

    Gapmer genomic coordinates are persisted as 1-based inclusive spans. Rows
    that expose a different convention are detected from their sequence length
    so steric candidate sets can use this fallback as well.
    """

    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalized_chrom(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text if text.lower().startswith("chr") else f"chr{text}"

    def _sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int, int]:
        index, row = item
        rank = _as_int(row.get("selection_rank"))
        start = _as_int(row.get("start") or row.get("Genomic_span_start"))
        return (rank if rank is not None else 10**12, start if start is not None else 10**12, index)

    features: List[SeqFeature] = []
    expected_chrom = _normalized_chrom(chrom)
    region_length = region_end - region_start + 1
    for index, row in sorted(enumerate(generated_rows, start=1), key=_sort_key):
        if not isinstance(row, dict):
            continue
        strand_text = str(row.get("strand") or row.get("Genomic_strand") or "").strip()
        strand = -1 if strand_text in {"-", "-1"} else 1 if strand_text in {"+", "+1", "1"} else default_strand
        locations: List[FeatureLocation] = []
        segments = _generated_row_segments(row)
        for segment in segments:
            segment_chrom = _normalized_chrom(
                segment.get("chrom")
                or segment.get("Chrom")
                or segment.get("Genomic_chrom")
                or row.get("chrom")
                or row.get("Genomic_span_chrom")
            )
            if segment_chrom and segment_chrom != expected_chrom:
                locations = []
                break
            start = _as_int(
                segment.get("start")
                or segment.get("Start")
                or segment.get("Genomic_start")
                or row.get("start")
                or row.get("Genomic_span_start")
            )
            end = _as_int(
                segment.get("end")
                or segment.get("End")
                or segment.get("Genomic_end")
                or row.get("end")
                or row.get("Genomic_span_end")
            )
            if start is None or end is None or end < start:
                locations = []
                break
            sequence_length = _as_int(row.get("length"))
            inclusive = bool(
                row.get("Genomic_segments") is not None
                or row.get("Genomic_span_start") is not None
                or str(row.get("source_modality") or "").strip().lower() == "gapmer"
                or (sequence_length is not None and sequence_length == end - start + 1)
            )
            start_rel = start - region_start if inclusive else start - region_start + 1
            end_rel = end - region_start + 1
            if start_rel < 0 or end_rel > region_length or end_rel <= start_rel:
                locations = []
                break
            locations.append(FeatureLocation(start_rel, end_rel, strand=strand))
        if not locations:
            continue
        location = locations[0] if len(locations) == 1 else CompoundLocation(locations, operator="join")
        rank = _as_int(row.get("selection_rank"))
        label = str(row.get("site_label") or row.get("route_label") or f"ASO {index}").strip()
        if rank is not None:
            label = f"rank {rank}: {label}"
        sequence = str(row.get("sequence") or row.get("Sequence") or "").strip()
        modality = str(row.get("source_modality") or "ASO").strip() or "ASO"
        notes = [f"Modality: {modality}", *sequence_notes(row)]
        if row.get("molecule_specification"):
            from aso_molecule import validate_candidate_molecule
            molecule = validate_candidate_molecule(row)
            notes.extend(["Physical chemistry: " + json.dumps(molecule, sort_keys=True)])
        for field, display in (
            ("selection_rank", "Selection rank"),
            ("gc_fraction", "GC fraction"),
            ("off_target_gene_count", "Off-target gene count"),
            ("Times_mapped_off_target_genes", "Off-target mapped genes"),
        ):
            value = row.get(field)
            if value not in (None, ""):
                notes.append(f"{display}: {value}")
        color = color_for_strategy(modality.lower())
        qualifiers = {
            "label": [label],
            "note": notes,
            "color": [color],
            "ApEinfo_fwdcolor": [color],
            "ApEinfo_revcolor": [color],
        }
        features.append(SeqFeature(location=location, type="misc_feature", qualifiers=qualifiers))
    return features


def build_record(
    gene_symbol: str,
    chrom: str,
    start: int,
    end: int,
    strand: int,
    sequence: Seq,
    features: List[SeqFeature],
) -> SeqRecord:
    record_id = f"{gene_symbol}_{chrom}:{start}-{end}"
    description = f"{gene_symbol} genomic segment ({chrom}:{start}-{end}, strand {'+' if strand == 1 else '-'})"
    record = SeqRecord(
        sequence,
        id=record_id[:16],
        name=gene_symbol[:10],
        description=description,
    )
    record.annotations["molecule_type"] = "DNA"
    record.features.extend(features)
    return record


def export_to_genbank(
    run_dir: Path,
    output_path: Path,
) -> Path:
    structured_goal = load_json(resolve_run_path(run_dir, "structured_goal.json"))
    summary_path = resolve_run_path(run_dir, "strategy_router_summary.json")
    generated_path = resolve_run_path(run_dir, "generated_aso_set.json")
    if not summary_path.exists() and not generated_path.exists():
        raise FileNotFoundError(
            f"Neither strategy router summary nor generated ASO set found under {run_dir}"
        )

    species_key = normalize_species(structured_goal.get("species"))
    structured_goal["species"] = species_key
    species_cfg = get_species_config(species_key)

    isoform_transcripts = (
        structured_goal.get("isoform_transcripts")
        or structured_goal.get("all_transcripts_full")
        or {}
    )
    if not isoform_transcripts:
        raise ValueError("structured_goal transcript catalog missing.")

    isoform_labels = resolve_isoform_labels(structured_goal)
    patient_reference = structured_goal.get("patient_reference") if isinstance(structured_goal, dict) else None
    genome_path = (
        Path(patient_reference.get("fasta_path"))
        if isinstance(patient_reference, dict) and patient_reference.get("fasta_path")
        else species_cfg.genome_fasta
    )
    chrom, strand, region_start, region_end = compute_gene_region(isoform_transcripts, isoform_labels)
    seq = fetch_genomic_sequence(chrom, region_start, region_end, strand, genome_path)

    isoform_features = build_isoform_features(isoform_transcripts, isoform_labels, region_start)
    if summary_path.exists():
        strategy_goals = parse_strategy_summary(summary_path)
        aso_features = build_aso_features(strategy_goals, region_start, region_end)
    else:
        generated_payload = load_json(generated_path)
        generated_rows = generated_payload.get("rows") if isinstance(generated_payload.get("rows"), list) else []
        aso_features = build_generated_aso_features(
            generated_rows,
            chrom,
            region_start,
            region_end,
            strand,
        )
    features = isoform_features + aso_features

    record = build_record(
        structured_goal.get("gene_symbol") or "GENE",
        chrom,
        region_start,
        region_end,
        strand,
        seq,
        features,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(record, str(output_path), "genbank")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export routed ASO features and isoform exons to GenBank.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing structured_goal.json and strategy outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination GenBank file. Defaults to <run-dir>/snapgene/aso_export.gb",
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_path = args.output
    if output_path is None:
        output_path = canonical_run_path(run_dir, "snapgene", "aso_export.gb")

    result_path = export_to_genbank(run_dir, output_path)
    print(f"Wrote GenBank export to {result_path}")


if __name__ == "__main__":
    main()
