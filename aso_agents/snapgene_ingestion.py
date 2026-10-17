from __future__ import annotations

import difflib
import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from pyfaidx import Fasta

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from snapgene_reader import snapgene_file_to_dict, snapgene_file_to_seqrecord
from .species_config import DATA_ROOT, get_species_config, normalize_species


ASO_DESIGNER_DATA_ROOT = DATA_ROOT
DEFAULT_SPECIES = "human"
DEFAULT_BOWTIE_PREFIX = get_species_config(DEFAULT_SPECIES).bowtie_prefix
REFERENCE_FASTA = get_species_config(DEFAULT_SPECIES).genome_fasta
FULL_SEQUENCE_MAX_MISMATCHES = 10
GENBANK_SUFFIXES = {".gb", ".gbk", ".genbank"}
EXON_NAME_RE = re.compile(r"\bexon[\s_-]*(\d+)\b", re.IGNORECASE)
ISOFORM_QUALIFIER_RE = re.compile(r"\bisoform\s*:\s*([^;]+)", re.IGNORECASE)


@dataclass
class SnapExonAlignment:
    chrom: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive
    strand: str
    mapq: int
    cigar: str


def _normalize_chrom(chrom: Optional[str]) -> Optional[str]:
    if not chrom:
        return None
    chrom = str(chrom)
    if chrom.lower().startswith("chr"):
        return chrom
    return f"chr{chrom}"


def _ensure_bowtie_index(prefix: Path) -> None:
    candidates = [
        prefix.with_suffix(".1.ebwt"),
        prefix.with_suffix(".1.ebwtl"),
        prefix.with_suffix(".1.bt2"),
    ]
    if not any(path.exists() for path in candidates):
        raise FileNotFoundError(
            f"Bowtie index not found for prefix '{prefix}'. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )


def _is_genbank_path(path: Path) -> bool:
    return path.suffix.lower() in GENBANK_SUFFIXES


def _read_sequence_record(path: Path):
    if _is_genbank_path(path):
        return SeqIO.read(str(path), "genbank")
    return snapgene_file_to_seqrecord(str(path))


def _feature_qualifier_values(feature: Any, keys: Sequence[str]) -> List[str]:
    values: List[str] = []
    qualifiers = getattr(feature, "qualifiers", {}) or {}
    for key in keys:
        raw = qualifiers.get(key)
        if raw is None:
            continue
        if isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
        else:
            text = str(raw).strip()
            if text:
                values.append(text)
    return values


def _exon_group_from_texts(texts: Sequence[str]) -> str:
    for text in texts:
        match = ISOFORM_QUALIFIER_RE.search(text)
        if match:
            return match.group(1).strip()
    for text in texts:
        match = EXON_NAME_RE.search(text)
        if match:
            prefix = text[: match.start()].strip(" \t\r\n-_:;()[]")
            if prefix:
                return prefix
    return ""


def _select_genbank_exon_group(rows: List[Dict[str, object]], path: Path) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        group = str(row.get("isoform_group") or "")
        groups.setdefault(group, []).append(row)

    if len(groups) <= 1:
        return rows

    snapgene_groups = [group for group in groups if "snapgene" in group.lower()]
    if len(snapgene_groups) == 1:
        return groups[snapgene_groups[0]]

    nonempty_groups = [group for group in groups if group]
    if len(nonempty_groups) == 1:
        return groups[nonempty_groups[0]]

    group_preview = ", ".join(sorted(group or "<ungrouped>" for group in groups)[:8])
    raise ValueError(
        f"Multiple exon feature groups found in GenBank file {path}: {group_preview}. "
        "Provide a file containing one isoform or include a SnapGene-labeled isoform group."
    )


def _load_genbank_exons(path: Path) -> pd.DataFrame:
    record = SeqIO.read(str(path), "genbank")
    exon_rows: List[Dict[str, object]] = []
    for feature in record.features:
        texts = _feature_qualifier_values(feature, ("label", "note", "gene", "product"))
        type_text = str(getattr(feature, "type", "") or "")
        joined_text = " ".join(texts)
        exon_match = EXON_NAME_RE.search(joined_text)
        if type_text.lower() != "exon" and not exon_match:
            continue
        if exon_match is None and type_text.lower() == "exon":
            exon_match = EXON_NAME_RE.search(str(texts[0] if texts else ""))

        start = int(feature.location.start)
        end = int(feature.location.end)
        if end <= start:
            continue
        strand = int(feature.location.strand or 1)
        seq_slice = record.seq[start:end]
        oriented_seq = seq_slice if strand >= 0 else seq_slice.reverse_complement()
        label = texts[0] if texts else type_text or f"exon {len(exon_rows) + 1}"

        exon_rows.append(
            {
                "exon_name": label,
                "order_hint": int(exon_match.group(1)) if exon_match else None,
                "local_start": start,
                "local_end": end,
                "length": len(oriented_seq),
                "feature_strand": strand,
                "sequence": str(oriented_seq).upper(),
                "read_id_base": label.replace(" ", "_"),
                "isoform_group": _exon_group_from_texts(texts),
            }
        )

    exon_rows = _select_genbank_exon_group(exon_rows, path) if exon_rows else []
    if not exon_rows:
        raise ValueError(f"No exon features were found in {path}")

    def _sort_key(row: Dict[str, Any]) -> Tuple[int, int]:
        if row.get("order_hint") is not None:
            return (int(row["order_hint"]), int(row["local_start"]))
        return (0, int(row["local_start"]))

    exon_rows.sort(key=_sort_key)
    for idx, row in enumerate(exon_rows, start=1):
        row["order"] = idx
        row["read_id"] = f"{row['read_id_base']}_{idx:02d}"
        row.pop("order_hint", None)
        row.pop("read_id_base", None)
        row.pop("isoform_group", None)

    return pd.DataFrame(exon_rows)


def load_snapgene_exons(snap_path: Path) -> pd.DataFrame:
    if not snap_path.exists():
        raise FileNotFoundError(f"SnapGene file not found: {snap_path}")

    if _is_genbank_path(snap_path):
        return _load_genbank_exons(snap_path)

    snap_dict = snapgene_file_to_dict(str(snap_path))
    seqrecord = snapgene_file_to_seqrecord(str(snap_path))
    features = snap_dict.get("features") or []

    exon_rows: List[Dict[str, object]] = []
    for feature in features:
        name = str(feature.get("name") or "").strip()
        if not name or not name.lower().startswith("exon"):
            continue
        start = int(feature.get("start", 0))
        end = int(feature.get("end", 0))
        match = None
        for token in name.replace("_", " ").split():
            if token.isdigit():
                match = int(token)
                break
        raw_strand = feature.get("strand", 1)
        if isinstance(raw_strand, str):
            strand = -1 if raw_strand.strip() in {"-", "-1"} else 1
        else:
            strand = int(raw_strand or 1)

        seq_slice = Seq(str(seqrecord.seq[start:end])).upper()
        oriented_seq = seq_slice if strand >= 0 else seq_slice.reverse_complement()

        exon_rows.append(
            {
                "exon_name": name,
                "order_hint": match if match is not None else None,
                "local_start": start,
                "local_end": end,
                "length": len(oriented_seq),
                "feature_strand": strand,
                "sequence": str(oriented_seq),
                "read_id_base": name.replace(" ", "_"),
            }
        )

    if not exon_rows:
        raise ValueError(f"No exon features were found in {snap_path}")

    def _sort_key(row: Dict[str, Any]) -> Tuple[int, int]:
        if row.get("order_hint") is not None:
            return (int(row["order_hint"]), int(row["local_start"]))
        return (0, int(row["local_start"]))

    exon_rows.sort(key=_sort_key)
    for idx, row in enumerate(exon_rows, start=1):
        row["order"] = idx
        row["read_id"] = f"{row['read_id_base']}_{idx:02d}"
        row.pop("order_hint", None)
        row.pop("read_id_base", None)

    return pd.DataFrame(exon_rows)


def _extract_snapgene_utr_features(snap_path: Path) -> Dict[str, Any]:
    try:
        if _is_genbank_path(snap_path):
            seqrecord = SeqIO.read(str(snap_path), "genbank")
            raw_features = []
            for feature in seqrecord.features:
                texts = _feature_qualifier_values(feature, ("label", "note", "gene", "product"))
                name = texts[0] if texts else str(getattr(feature, "type", "") or "")
                raw_features.append(
                    {
                        "name": name,
                        "start": int(feature.location.start),
                        "end": int(feature.location.end),
                        "strand": int(feature.location.strand or 1),
                    }
                )
        else:
            snap_dict = snapgene_file_to_dict(str(snap_path))
            seqrecord = snapgene_file_to_seqrecord(str(snap_path))
            raw_features = snap_dict.get("features") or []
    except Exception:
        return {"raw": []}

    utr_features: List[Dict[str, Any]] = []
    for feature in raw_features:
        name_raw = feature.get("name")
        name = str(name_raw or "").strip()
        if not name:
            continue
        if "utr" not in name.lower():
            continue
        start = int(feature.get("start", 0))
        end = int(feature.get("end", 0))
        raw_strand = feature.get("strand", 1)
        strand = -1 if (isinstance(raw_strand, str) and raw_strand.strip() in {"-", "-1"}) or (isinstance(raw_strand, (int, float)) and int(raw_strand) < 0) else 1
        seq_slice = seqrecord.seq[start:end]
        oriented_seq = seq_slice if strand == 1 else seq_slice.reverse_complement()
        utr_features.append({
            "name": name,
            "start": start,
            "end": end,
            "strand": strand,
            "sequence": str(oriented_seq).upper(),
        })
    return {
        "raw": utr_features,
    }


def _touch_score(exon_rec: Dict[str, Any], utr_rec: Dict[str, Any]) -> Optional[int]:
    try:
        exon_start = int(exon_rec.get("local_start"))
        exon_end = int(exon_rec.get("local_end"))
        utr_start = int(utr_rec.get("start"))
        utr_end = int(utr_rec.get("end"))
    except Exception:
        return None
    # ensure ordering
    if exon_start > exon_end:
        exon_start, exon_end = exon_end, exon_start
    if utr_start > utr_end:
        utr_start, utr_end = utr_end, utr_start
    overlap_start = max(exon_start, utr_start)
    overlap_end = min(exon_end, utr_end)
    if overlap_start <= overlap_end:
        # positive score favours larger overlap
        return (overlap_end - overlap_start) + 1
    # Adjacent within 1 base is acceptable (score 0 or -1)
    gap = min(abs(exon_start - utr_end), abs(utr_start - exon_end))
    if gap <= 1:
        return 0
    return None


def _classify_snapgene_utrs(
    ordered_exons: List[Dict[str, Any]],
    raw_utrs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    classification: Dict[str, Any] = {
        "raw": raw_utrs,
        "utr5": None,
        "utr3": None,
        "has_utrs": False,
    }
    if not ordered_exons or not raw_utrs:
        return classification

    first_exon = ordered_exons[0]
    last_exon = ordered_exons[-1]
    candidates5: List[Tuple[int, Dict[str, Any]]] = []
    candidates3: List[Tuple[int, Dict[str, Any]]] = []

    for feature in raw_utrs:
        score5 = _touch_score(first_exon, feature)
        if score5 is not None:
            candidates5.append((score5, feature))
        score3 = _touch_score(last_exon, feature)
        if score3 is not None:
            candidates3.append((score3, feature))

    if candidates5:
        # prefer greater overlap (higher score). If equal, choose closest upstream/downstream feature.
        chosen5 = max(candidates5, key=lambda item: (item[0], -abs(item[1]["end"] - first_exon.get("local_start", 0))))
        classification["utr5"] = {
            "feature": chosen5[1],
            "exon_order": int(first_exon.get("order") or 1),
        }
    if candidates3:
        chosen3 = max(candidates3, key=lambda item: (item[0], -abs(item[1]["start"] - last_exon.get("local_end", 0))))
        classification["utr3"] = {
            "feature": chosen3[1],
            "exon_order": int(last_exon.get("order") or len(ordered_exons)),
        }

    classification["has_utrs"] = bool(classification["utr5"] and classification["utr3"])
    return classification


def _apply_utr_annotations(exons: List[Dict[str, Any]], utr_features: Dict[str, Any]) -> Dict[str, Any]:
    if not exons:
        utr_summary = {
            "raw": utr_features.get("raw", []) if isinstance(utr_features, dict) else [],
            "utr5": None,
            "utr3": None,
            "has_utrs": False,
        }
        return utr_summary
    ordered = sorted(
        [ex for ex in exons if isinstance(ex, dict)],
        key=lambda rec: int(rec.get("order") or 0),
    )
    if not ordered:
        utr_summary = {
            "raw": utr_features.get("raw", []) if isinstance(utr_features, dict) else [],
            "utr5": None,
            "utr3": None,
            "has_utrs": False,
        }
        return utr_summary

    raw_utrs = utr_features.get("raw", []) if isinstance(utr_features, dict) else []
    utr_summary = _classify_snapgene_utrs(ordered, raw_utrs)

    strand_char = str((ordered[0].get("strand") or "+")).strip()

    for record in ordered:
        try:
            g_start = int(record.get("genomic_start"))
            g_end = int(record.get("genomic_end"))
        except Exception:
            continue
        record["coding_start_genomic"] = g_start
        record["coding_end_genomic"] = g_end
        record["contains_coding"] = True

    def _overlap_length(exon_rec: Dict[str, Any], utr_rec: Optional[Dict[str, Any]]) -> int:
        if not utr_rec:
            return 0
        try:
            exon_start = int(exon_rec.get("local_start"))
            exon_end = int(exon_rec.get("local_end"))
            utr_start = int(utr_rec.get("start"))
            utr_end = int(utr_rec.get("end"))
        except Exception:
            return 0
        overlap_start = max(exon_start, utr_start)
        overlap_end = min(exon_end, utr_end)
        return max(0, overlap_end - overlap_start)

    first_exon = ordered[0]
    last_exon = ordered[-1]

    utr5_entry = utr_summary.get("utr5")
    utr3_entry = utr_summary.get("utr3")
    utr5_feature = utr5_entry.get("feature") if isinstance(utr5_entry, dict) else None
    utr3_feature = utr3_entry.get("feature") if isinstance(utr3_entry, dict) else None

    offset5 = _overlap_length(first_exon, utr5_feature)

    if offset5:
        try:
            g_start = int(first_exon.get("genomic_start"))
            g_end = int(first_exon.get("genomic_end"))
        except Exception:
            g_start = g_end = None
        if g_start is not None and g_end is not None:
            exon_len = max(0, g_end - g_start + 1)
            offset5 = min(offset5, exon_len)
            if strand_char == "-":
                new_start = max(g_start, g_end - offset5)
            else:
                new_start = min(g_end, g_start + offset5)
            first_exon["coding_start_genomic"] = new_start

    offset3 = _overlap_length(last_exon, utr3_feature)

    if offset3:
        try:
            g_start = int(last_exon.get("genomic_start"))
            g_end = int(last_exon.get("genomic_end"))
        except Exception:
            g_start = g_end = None
        if g_start is not None and g_end is not None:
            exon_len = max(0, g_end - g_start + 1)
            offset3 = min(offset3, exon_len)
            if strand_char == "-":
                new_end = min(g_end, g_start + offset3)
            else:
                new_end = max(g_start, g_end - offset3)
            last_exon["coding_end_genomic"] = new_end

    for record in ordered:
        cs = record.get("coding_start_genomic")
        ce = record.get("coding_end_genomic")
        try:
            cs_val = int(cs) if cs is not None else None
            ce_val = int(ce) if ce is not None else None
        except Exception:
            cs_val = ce_val = None
        if cs_val is None or ce_val is None or cs_val > ce_val:
            record["coding_start_genomic"] = None
            record["coding_end_genomic"] = None
            record["contains_coding"] = False
        else:
            record["contains_coding"] = True

    utr5_order = utr5_entry.get("exon_order") if isinstance(utr5_entry, dict) else None
    utr3_order = utr3_entry.get("exon_order") if isinstance(utr3_entry, dict) else None
    for record in ordered:
        record["utr5_adjacent"] = bool(utr5_order is not None and int(record.get("order") or 0) == utr5_order)
        record["utr3_adjacent"] = bool(utr3_order is not None and int(record.get("order") or 0) == utr3_order)

    if isinstance(utr_features, dict):
        utr_features.clear()
        utr_features.update(utr_summary)

    return utr_summary


def _write_temp_fasta(exon_df: pd.DataFrame) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False)
    try:
        for _, row in exon_df.iterrows():
            sequence = row["sequence"]
            if not sequence:
                continue
            tmp.write(f">{row['read_id']}\n{sequence}\n")
    finally:
        tmp.close()
    return Path(tmp.name)


def _write_sequence_fasta(sequence: str, read_id: str = "FULL_SEQ") -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False)
    try:
        tmp.write(f">{read_id}\n{sequence}\n")
    finally:
        tmp.close()
    return Path(tmp.name)


def _parse_bowtie_sam(stdout: str) -> Dict[str, List[SnapExonAlignment]]:
    alignments: Dict[str, List[SnapExonAlignment]] = {}
    for line in stdout.splitlines():
        if not line or line.startswith("@"):
            continue
        fields = line.split("\t")
        if len(fields) < 11:
            continue
        qname, flag, rname, pos, mapq, cigar, _, _, _, seq = fields[:10]
        if rname == "*" or pos == "0":
            continue
        flag_val = int(flag)
        strand = "-" if flag_val & 0x10 else "+"
        start = int(pos)
        end = start + len(seq) - 1
        alignments.setdefault(qname, []).append(
            SnapExonAlignment(
                chrom=rname,
                start=start,
                end=end,
                strand=strand,
                mapq=int(mapq),
                cigar=cigar,
            )
        )
    return alignments


def _resolve_global_context(
    alignments: Dict[str, List[SnapExonAlignment]],
    expected_chroms: Iterable[str],
    expected_strand: Optional[str],
) -> Optional[Tuple[str, str]]:
    expected_set = {c for chrom in expected_chroms for c in {chrom, _normalize_chrom(chrom)} if c}
    strand_options = {"-", "+"}
    if expected_strand in strand_options:
        strand_options = {expected_strand}

    score: Dict[Tuple[str, str], int] = {}
    for matches in alignments.values():
        for match in matches:
            key = (_normalize_chrom(match.chrom) or match.chrom, match.strand)
            weight = max(match.mapq, 0) + 1
            score[key] = score.get(key, 0) + weight

    if not score:
        return None

    def _sort_key(item: Tuple[Tuple[str, str], int]) -> Tuple[int, int]:
        (chrom, strand), total = item
        chrom_pref = 1 if chrom in expected_set else 0
        strand_pref = 1 if strand == expected_strand else 0
        return (chrom_pref + strand_pref, total)

    preferred = max(score.items(), key=_sort_key)[0]
    return preferred


def _choose_alignment(
    matches: List[SnapExonAlignment],
    preferred: Optional[Tuple[str, str]],
) -> Optional[SnapExonAlignment]:
    if not matches:
        return None
    if preferred:
        chrom_pref, strand_pref = preferred
        filtered = [
            m for m in matches if (_normalize_chrom(m.chrom) == chrom_pref and m.strand == strand_pref)
        ]
        if filtered:
            matches = filtered
    return max(matches, key=lambda m: (m.mapq, m.start * -1))


def _align_full_sequence(
    sequence: str,
    index_prefix: Path,
    max_mismatches: int,
    preferred_context: Optional[Tuple[str, str]],
) -> Optional[SnapExonAlignment]:
    if not sequence:
        return None
    fasta_path = _write_sequence_fasta(sequence, read_id="FULL_SEQ")
    try:
        mismatch_budget = max(int(max_mismatches), FULL_SEQUENCE_MAX_MISMATCHES)
        cmd = [
            "bowtie",
            "-f",
            "-S",
            "-v",
            str(mismatch_budget),
            "-a",
            "--best",
            str(index_prefix),
            str(fasta_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        fasta_path.unlink(missing_ok=True)

    if result.returncode not in {0, 1}:
        return None

    alignments_map = _parse_bowtie_sam(result.stdout)
    matches = alignments_map.get("FULL_SEQ") or []
    if not matches:
        return None
    chosen = _choose_alignment(matches, preferred_context)
    return chosen


def _fallback_full_sequence_alignment(
    sequence: str,
    per_exon_records: List[Dict[str, object]],
    chrom: Optional[str],
    strand: Optional[str],
    fasta: Fasta,
    *,
    window_padding: int = 500,
) -> Optional[SnapExonAlignment]:
    if not sequence or not chrom or not per_exon_records:
        return None
    strand_val = strand if strand in {"+", "-"} else "+"
    chrom_norm = _normalize_chrom(chrom) or chrom
    if chrom_norm not in fasta:
        return None
    starts: List[int] = []
    ends: List[int] = []
    for record in per_exon_records:
        try:
            start = int(record.get("genomic_start"))
            end = int(record.get("genomic_end"))
        except Exception:
            continue
        starts.append(min(start, end))
        ends.append(max(start, end))
    if not starts or not ends:
        return None
    min_start = min(starts)
    max_end = max(ends)
    seq_len = len(sequence)
    window_start = max(1, min_start - window_padding)
    window_end = max_end + window_padding + seq_len
    try:
        reference_slice = str(fasta[chrom_norm][window_start - 1 : window_end]).upper()
    except Exception:
        return None
    if not reference_slice:
        return None
    query_seq = sequence.upper()
    if strand_val == "-":
        query_seq = str(Seq(query_seq).reverse_complement()).upper()
    ref_len = len(reference_slice)
    if ref_len < seq_len:
        return None
    best_offset: Optional[int] = None
    best_mismatches: Optional[int] = None
    for offset in range(ref_len - seq_len + 1):
        mismatches = 0
        for idx in range(seq_len):
            if reference_slice[offset + idx] != query_seq[idx]:
                mismatches += 1
        if best_mismatches is None or mismatches < best_mismatches:
            best_mismatches = mismatches
            best_offset = offset
            if mismatches == 0:
                break
    if best_offset is None:
        return None
    align_start = window_start + best_offset
    align_end = align_start + seq_len - 1
    return SnapExonAlignment(
        chrom=chrom_norm,
        start=align_start,
        end=align_end,
        strand=strand_val,
        mapq=255,
        cigar=f"{seq_len}M",
    )


def _variants_from_aligned_sequence(
    alignment: SnapExonAlignment,
    sequence: str,
    fasta: Fasta,
) -> List[Dict[str, object]]:
    if not alignment or not sequence:
        return []
    chrom = _normalize_chrom(alignment.chrom) or alignment.chrom
    if not chrom:
        return []
    start = int(alignment.start)
    end = start + len(sequence) - 1
    try:
        ref_seq = str(fasta[chrom][start - 1 : end]).upper()
    except Exception:
        return []
    alt_seq = sequence.upper()
    if alignment.strand == "-":
        alt_seq = str(Seq(alt_seq).reverse_complement()).upper()
    return _call_variants_for_sequence(
        chrom,
        start,
        ref_seq,
        alt_seq,
        fasta,
    )


def ingest_snapgene_file(
    snapgene_path: str | Path,
    *,
    gene_symbol: Optional[str] = None,
    expected_chroms: Optional[Sequence[str]] = None,
    expected_strand: Optional[str] = None,
    bowtie_prefix: Optional[str | Path] = None,
    max_mismatches: int = 2,
    reference_fasta_path: Optional[str | Path] = None,
    patient_output_dir: Optional[str | Path] = None,
    species: str = DEFAULT_SPECIES,
    derive_patient_variants: bool = True,
) -> Dict[str, object]:
    snap_path = Path(snapgene_path).expanduser().resolve()
    exon_df = load_snapgene_exons(snap_path)
    utr_features = _extract_snapgene_utr_features(snap_path)
    try:
        full_sequence = str(_read_sequence_record(snap_path).seq).upper()
        sequence_sha256 = hashlib.sha256(full_sequence.encode("utf-8")).hexdigest()
    except Exception:
        full_sequence = ""
        sequence_sha256 = None

    species_key = normalize_species(species)
    species_cfg = get_species_config(species_key)
    index_prefix = Path(bowtie_prefix).expanduser().resolve() if bowtie_prefix else species_cfg.bowtie_prefix
    _ensure_bowtie_index(index_prefix)

    fasta_path = _write_temp_fasta(exon_df)
    try:
        cmd = [
            "bowtie",
            "-f",
            "-S",
            "-v",
            str(int(max_mismatches)),
            "-a",
            "--best",
            str(index_prefix),
            str(fasta_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        fasta_path.unlink(missing_ok=True)

    if result.returncode not in {0, 1}:
        raise RuntimeError(
            f"Bowtie exited with code {result.returncode}:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    raw_alignments = _parse_bowtie_sam(result.stdout)
    preferred_context = _resolve_global_context(
        raw_alignments,
        expected_chroms or [],
        expected_strand,
    )
    full_alignment = None
    if derive_patient_variants:
        full_alignment = _align_full_sequence(
            full_sequence,
            index_prefix,
            max_mismatches,
            preferred_context,
        )

    annotated = exon_df.copy()
    annotated["alignment_count"] = 0
    annotated["chrom"] = pd.NA
    annotated["genomic_start"] = pd.NA
    annotated["genomic_end"] = pd.NA
    annotated["strand"] = pd.NA
    annotated["mapq"] = pd.NA

    per_exon_records: List[Dict[str, object]] = []
    warnings: List[str] = []

    for _, row in annotated.iterrows():
        read_id = row["read_id"]
        matches = raw_alignments.get(read_id, [])
        annotated_matches = [
            {
                "chrom": _normalize_chrom(m.chrom) or m.chrom,
                "start": m.start,
                "end": m.end,
                "strand": m.strand,
                "mapq": m.mapq,
                "cigar": m.cigar,
            }
            for m in matches
        ]
        best = _choose_alignment(matches, preferred_context)
        alignment_count = len(matches)
        if best:
            chrom_val = _normalize_chrom(best.chrom) or best.chrom
            annotated.loc[annotated["read_id"] == read_id, ["chrom", "genomic_start", "genomic_end", "strand", "mapq"]] = [
                chrom_val,
                best.start,
                best.end,
                best.strand,
                best.mapq,
            ]
        else:
            chrom_val = None
            warnings.append(f"No genomic alignment for {row['exon_name']}.")

        annotated.loc[annotated["read_id"] == read_id, "alignment_count"] = alignment_count

        donor_site: Optional[int] = None
        acceptor_site: Optional[int] = None
        strand = best.strand if best else "+"

        if best:
            if strand == "+":
                donor_site = best.end
                acceptor_site = best.start
            else:
                donor_site = best.start
                acceptor_site = best.end

        record = {
            "exon_name": row["exon_name"],
            "order": int(row["order"]),
            "local_start": int(row["local_start"]),
            "local_end": int(row["local_end"]),
            "length": int(row["length"]),
            "feature_strand": int(row["feature_strand"]),
            "sequence": row["sequence"],
            "read_id": read_id,
            "alignment_count": alignment_count,
            "alignments": annotated_matches,
            "chosen_alignment": annotated_matches[0] if best else None,
            "chrom": chrom_val,
            "genomic_start": best.start if best else None,
            "genomic_end": best.end if best else None,
            "strand": strand if best else None,
            "mapq": best.mapq if best else None,
            "donor_site": donor_site,
            "acceptor_site": acceptor_site,
        }
        per_exon_records.append(record)

    utr_summary = _apply_utr_annotations(per_exon_records, utr_features)
    if not utr_summary.get("has_utrs"):
        warnings.append(
            "SnapGene UTR annotations missing or disconnected; treating all junctions as non-terminal."
        )

    summary = {
        "snapgene_path": str(snap_path),
        "bowtie_index": str(index_prefix),
        "gene_symbol": gene_symbol,
        "expected_chroms": list(expected_chroms or []),
        "expected_strand": expected_strand,
        "sequence_length": len(full_sequence) if full_sequence else None,
        "sequence_sha256": sequence_sha256,
        "full_sequence": full_sequence,
        "preferred_context": {
            "chrom": preferred_context[0] if preferred_context else None,
            "strand": preferred_context[1] if preferred_context else None,
        },
        "species": species_key,
        "full_sequence_alignment": {
            "chrom": _normalize_chrom(full_alignment.chrom) if full_alignment else None,
            "start": full_alignment.start if full_alignment else None,
            "end": full_alignment.end if full_alignment else None,
            "strand": full_alignment.strand if full_alignment else None,
            "mapq": full_alignment.mapq if full_alignment else None,
        } if full_alignment else None,
        "exons": per_exon_records,
        "warnings": warnings,
        "utr_features": utr_summary,
    }
    reference_path = Path(reference_fasta_path).expanduser().resolve() if reference_fasta_path else species_cfg.genome_fasta
    if reference_path.exists() and derive_patient_variants:
        try:
            variants = _collect_patient_variants(per_exon_records, reference_path)
        except Exception as exc:
            warnings.append(f"Patient variant derivation failed: {exc}")
            variants = []
        try:
            fasta_obj = Fasta(str(reference_path), sequence_always_upper=True)
        except Exception as exc:
            warnings.append(f"Failed to open reference FASTA for full-sequence diff: {exc}")
            fasta_obj = None
        full_sequence_variants: List[Dict[str, object]] = []
        active_alignment = full_alignment
        if fasta_obj and not active_alignment:
            chrom_ctx = preferred_context[0] if preferred_context else None
            if not chrom_ctx:
                for record in per_exon_records:
                    chrom_val = record.get("chrom")
                    if chrom_val:
                        chrom_ctx = chrom_val
                        break
            strand_ctx = preferred_context[1] if (preferred_context and len(preferred_context) > 1) else None
            active_alignment = _fallback_full_sequence_alignment(
                full_sequence,
                per_exon_records,
                chrom_ctx,
                strand_ctx,
                fasta_obj,
            )
            if active_alignment:
                summary["full_sequence_alignment"] = {
                    "chrom": active_alignment.chrom,
                    "start": active_alignment.start,
                    "end": active_alignment.end,
                    "strand": active_alignment.strand,
                    "mapq": active_alignment.mapq,
                }
        if fasta_obj and active_alignment:
            full_sequence_variants = _variants_from_aligned_sequence(active_alignment, full_sequence, fasta_obj)
        variant_map: Dict[Tuple[str, int, str, str], Dict[str, object]] = {}
        for var in variants:
            key = (var["chrom"], var["pos"], var["ref"], var["alt"])
            variant_map[key] = var
        for var in full_sequence_variants:
            key = (var["chrom"], var["pos"], var["ref"], var["alt"])
            variant_map[key] = var
        variants = sorted(variant_map.values(), key=lambda item: (item["chrom"], item["pos"]))
        summary["patient_variants"] = variants
        summary["reference_fasta_path"] = str(reference_path)
        summary["patient_reference"] = None
    else:
        if derive_patient_variants:
            warnings.append(f"Reference FASTA not found at {reference_path}")
        else:
            warnings.append(
                "Patient variant derivation skipped; SnapGene file is being used as an isoform reference."
            )
        summary["patient_variants"] = []
        summary["reference_fasta_path"] = str(reference_path)
        summary["patient_reference"] = None

    return summary


def _collect_patient_variants(
    exon_records: List[Dict[str, object]],
    reference_fasta_path: Path,
) -> List[Dict[str, object]]:
    fasta = Fasta(str(reference_fasta_path), sequence_always_upper=True)
    variants: Dict[Tuple[str, int, str, str], Dict[str, object]] = {}

    for record in exon_records:
        chrom = record.get("chrom")
        start = record.get("genomic_start")
        end = record.get("genomic_end")
        strand = record.get("strand") or "+"
        sequence = record.get("sequence")
        chosen_alignment = record.get("chosen_alignment") if isinstance(record.get("chosen_alignment"), dict) else {}
        chrom = _normalize_chrom(chosen_alignment.get("chrom")) or chrom
        start = chosen_alignment.get("start") if chosen_alignment.get("start") is not None else start
        end = chosen_alignment.get("end") if chosen_alignment.get("end") is not None else end
        strand = chosen_alignment.get("strand") or strand
        if not chrom or start is None or end is None or not sequence:
            continue
        chrom_str = str(chrom)
        try:
            ref_seq = str(fasta[chrom_str][int(start) - 1 : int(end)]).upper()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load reference bases for {chrom_str}:{start}-{end}"
            ) from exc
        seq_alt = str(sequence).upper()
        if strand == "-":
            seq_alt = str(Seq(seq_alt).reverse_complement()).upper()
        seq_ref = ref_seq.upper()
        if len(seq_ref) != len(seq_alt):
            # Allow for indels; continue to diff
            pass
        seq_variants = _call_variants_for_sequence(
            chrom_str,
            int(start),
            seq_ref,
            seq_alt,
            fasta,
        )
        for var in seq_variants:
            key = (var["chrom"], var["pos"], var["ref"], var["alt"])
            variants[key] = var

    ordered = sorted(variants.values(), key=lambda item: (item["chrom"], item["pos"]))
    return ordered


def _call_variants_for_sequence(
    chrom: str,
    start_pos: int,
    ref_seq: str,
    alt_seq: str,
    fasta: Fasta,
) -> List[Dict[str, object]]:
    matcher = difflib.SequenceMatcher(None, ref_seq, alt_seq, autojunk=False)
    variants: List[Dict[str, object]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        ref_sub = ref_seq[i1:i2]
        alt_sub = alt_seq[j1:j2]
        emitted = _emit_variant_records(
            chrom,
            start_pos,
            i1,
            ref_sub,
            alt_sub,
            fasta,
            ref_seq,
        )
        variants.extend(emitted)
    return variants


def _emit_variant_records(
    chrom: str,
    region_start: int,
    ref_index: int,
    ref_sub: str,
    alt_sub: str,
    fasta: Fasta,
    ref_seq_full: str,
) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []
    ref_len = len(ref_sub)
    alt_len = len(alt_sub)

    if ref_len == 0 and alt_len == 0:
        return variants

    if ref_len == 0 or alt_len == 0:
        # Skip insertions or deletions; only emit mismatch substitutions.
        return variants

    # substitution or complex replace
    ref = ref_sub.upper()
    alt = alt_sub.upper()
    pos = region_start + ref_index

    if len(ref) != len(alt):
        # Skip complex replacements that imply indels.
        return variants
    else:
        while ref and alt and ref[0] == alt[0]:
            ref = ref[1:]
            alt = alt[1:]
            pos += 1
        while ref and alt and ref[-1] == alt[-1]:
            ref = ref[:-1]
            alt = alt[:-1]
    if not ref:
        return variants

    variants.append({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt})
    return variants


def _fetch_base(fasta: Fasta, chrom: str, pos: int) -> Optional[str]:
    try:
        return fasta[chrom][pos - 1 : pos].seq.upper()
    except Exception:
        return None


def _write_patient_vcf(
    variants: List[Dict[str, object]],
    reference_fasta_path: Path,
    output_path: Path,
    sample_name: str = "patient",
) -> None:
    header_lines = [
        "##fileformat=VCFv4.2",
        f"##reference={reference_fasta_path}",
        "##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of Samples With Data\">",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(line + "\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}\n".format(sample_name))
        for var in variants:
            chrom = var["chrom"]
            pos = var["pos"]
            ref = var["ref"]
            alt = var["alt"]
            handle.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\tNS=1\tGT\t1/1\n")


def _write_patient_reference(
    reference_fasta_path: Path,
    variants: List[Dict[str, object]],
    output_path: Path,
    *,
    sequence: Optional[str] = None,
    sample_name: str = "patient",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sequence:
        seq_clean = sequence.strip().upper()
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f">{sample_name}\n")
            _write_fasta_sequence(handle, seq_clean)
        try:
            Fasta(str(output_path), build_index=True)
        except Exception:
            pass
        return

    variants_by_chrom: Dict[str, List[Dict[str, object]]] = {}
    for var in variants:
        variants_by_chrom.setdefault(var["chrom"], []).append(var)
    for chrom_vars in variants_by_chrom.values():
        chrom_vars.sort(key=lambda item: item["pos"])

    with open(reference_fasta_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        current_chrom: Optional[str] = None
        buffered_seq: List[str] = []

        def _flush_buffer(chrom_name: str, seq_lines: List[str]) -> None:
            seq_plain = "".join(line.strip() for line in seq_lines)
            mutated_seq = _apply_variants_to_sequence(seq_plain, variants_by_chrom.get(chrom_name, []))
            _write_fasta_sequence(dst, mutated_seq)

        for line in src:
            if line.startswith(">"):
                if current_chrom is not None and current_chrom in variants_by_chrom:
                    _flush_buffer(current_chrom, buffered_seq)
                elif current_chrom is not None:
                    for seq_line in buffered_seq:
                        dst.write(seq_line)
                current_chrom = line[1:].split()[0]
                dst.write(line)
                buffered_seq = [] if current_chrom in variants_by_chrom else []
            else:
                if current_chrom in variants_by_chrom:
                    buffered_seq.append(line)
                else:
                    dst.write(line)

        if current_chrom is not None and current_chrom in variants_by_chrom:
            _flush_buffer(current_chrom, buffered_seq)
        elif current_chrom is not None:
            for seq_line in buffered_seq:
                dst.write(seq_line)

    # build FASTA index
    Fasta(str(output_path), build_index=True)


def _apply_variants_to_sequence(sequence: str, variants: Optional[List[Dict[str, object]]]) -> str:
    if not variants:
        return sequence
    segments: List[str] = []
    cursor = 0
    for var in variants:
        pos = var["pos"]
        ref = var["ref"].upper()
        alt = var["alt"].upper()
        idx = pos - 1
        if sequence[idx : idx + len(ref)].upper() != ref.upper():
            # Reference mismatch; skip variant
            continue
        segments.append(sequence[cursor:idx])
        segments.append(alt)
        cursor = idx + len(ref)
    segments.append(sequence[cursor:])
    mutated = "".join(segments)
    return mutated


def _write_fasta_sequence(handle, sequence: str, line_length: int = 60) -> None:
    for idx in range(0, len(sequence), line_length):
        handle.write(sequence[idx : idx + line_length] + "\n")
