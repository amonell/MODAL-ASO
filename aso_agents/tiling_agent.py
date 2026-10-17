from agents import Agent, function_tool
import os
import json
import shlex
import sys
from typing import List, Dict, Any, Optional, Tuple
import re
import bisect

from .species_config import get_species_config, normalize_species
from .parameter_constraints import (
    DEFAULT_HOMOPOLYMER_LIMIT,
    _coerce_fraction as _pc_coerce_fraction,  # reuse existing helper
    get_homopolymer_sequences,
)
from .repeat_risk import add_locus_columns_to_row, ensure_locus_schema

# Resolve pipeline root
PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aso-designer-main", "aso-designer-main"))
INPUTS_DIR = os.path.join(PIPELINE_ROOT, "inputs")

os.makedirs(INPUTS_DIR, exist_ok=True)


def _safe_float(val, default):
    try:
        return float(val)
    except Exception:
        return default


def _safe_int(val, default):
    try:
        return int(val)
    except Exception:
        return default


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except Exception:
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"\d+", text)
        if not match:
            return None
        number = int(match.group(0))
    return number if number >= 0 else None


def _resolve_num_mismatches(structured_goal: Dict[str, Any], default: int = 1) -> int:
    """Resolve the requested off-target mismatch budget from structured fields."""
    candidates: List[Any] = []
    if isinstance(structured_goal, dict):
        candidates.append(structured_goal.get("num_mismatches"))
        constraints = structured_goal.get("constraints")
        if isinstance(constraints, dict):
            off_target = constraints.get("off_target_evaluation")
            if isinstance(off_target, dict):
                candidates.append(off_target.get("max_mismatches"))
            match_policy = constraints.get("match_policy")
            if isinstance(match_policy, dict):
                candidates.append(match_policy.get("max_mismatches"))
        evaluation_targets = structured_goal.get("evaluation_targets")
        if isinstance(evaluation_targets, dict):
            off_target = evaluation_targets.get("off_target")
            if isinstance(off_target, dict):
                match_policy = off_target.get("match_policy")
                if isinstance(match_policy, dict):
                    candidates.append(match_policy.get("max_mismatches"))
                candidates.append(off_target.get("max_mismatches"))
    for candidate in candidates:
        parsed = _coerce_non_negative_int(candidate)
        if parsed is not None:
            if parsed > 3:
                raise ValueError(
                    "Bowtie -v mismatch mode supports at most 3 mismatches; "
                    f"received {parsed}."
                )
            return parsed
    return default


def _coerce_tile_length(value: Any) -> Optional[int]:
    """Convert assorted user or agent hints into a positive integer tile length."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            num = int(round(float(value)))
        except Exception:
            return None
        return num if num > 0 else None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    candidate = int(match.group(1))
    return candidate if candidate > 0 else None


def _resolve_tile_length(structured_goal: Dict[str, Any], chemistry: Any, default: int = 20) -> int:
    """Determine tile length preference from structured goal constraints or chemistry guidance."""

    def _gather(obj: Optional[Dict[str, Any]], keys: Tuple[str, ...]) -> List[Any]:
        if not isinstance(obj, dict):
            return []
        return [obj.get(key) for key in keys if key in obj]

    candidates: List[Any] = []
    if isinstance(structured_goal, dict):
        candidates.extend(_gather(structured_goal, ("tile_len", "tile_length")))
        candidates.extend(_gather(structured_goal.get("constraints"), ("tile_len", "tile_length", "length_nt", "length", "aso_length")))
        candidates.extend(_gather(structured_goal.get("parameter_constraints"), ("tile_len", "tile_length", "length_nt", "length")))
    if isinstance(chemistry, dict):
        candidates.append(chemistry.get("length"))
        recommendation = chemistry.get("recommendation") if isinstance(chemistry, dict) else None
        if isinstance(recommendation, dict):
            candidates.extend(
                _gather(
                    recommendation,
                    ("tile_len", "tile_length", "length", "length_nt", "recommended_length"),
                )
            )
    for candidate in candidates:
        length = _coerce_tile_length(candidate)
        if length is not None:
            return length
    return default


def _normalize_gapmer_tile_length(chemistry_mode: str, tile_len: int) -> Tuple[int, Optional[str]]:
    from aso_chemistry import require_chemistry_length
    require_chemistry_length(chemistry_mode, tile_len)
    return tile_len, None


def _extract_chemistry_mode(structured_goal: Dict[str, Any], chemistry: Any) -> str:
    """Pull a string chemistry label from nested dicts, defaulting to steric-hindrance."""

    def _from_obj(obj: Any) -> Optional[str]:
        if isinstance(obj, str) and obj.strip():
            return obj.strip()
        if isinstance(obj, dict):
            for key in ("chemistry", "CHEMISTRY", "mode", "name"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            for nested_key in ("recommendation", "recommendations", "profile"):
                nested = obj.get(nested_key)
                if nested:
                    nested_val = _from_obj(nested)
                    if nested_val:
                        return nested_val
        return None

    candidates = [chemistry, structured_goal.get("chemistry"), structured_goal.get("chemistry_mode"), structured_goal.get("chemistry_profile")]
    for candidate in candidates:
        val = _from_obj(candidate)
        if val:
            return val
    return "steric-hindrance"


def _normalize_gene_set(
    primary_candidates: Any,
    target_gene_symbol: Optional[str],
    additional_genes: Optional[List[str]] = None,
) -> List[str]:
    """Collect unique off-target gene symbols, excluding the primary gene."""
    tokens: List[str] = []
    if primary_candidates is not None:
        if isinstance(primary_candidates, (list, tuple, set)):
            for item in primary_candidates:
                if item is None:
                    continue
                tokens.extend(re.split(r"[;,|]", str(item)))
        else:
            text = str(primary_candidates).strip()
            if text and text.lower() not in {"none", "nan", "null"}:
                tokens.extend(re.split(r"[;,|]", text))

    target_norm = (target_gene_symbol or "").strip().lower()
    seen: set[str] = set()
    off_targets: List[str] = []
    for raw in tokens:
        gene = raw.strip()
        if not gene or gene.lower() in {"none", "nan", "null"}:
            continue
        if ":" in gene and gene.upper().startswith("MAPPEDTO="):
            continue
        if target_norm and gene.lower() == target_norm:
            continue
        key = gene.lower()
        if key not in seen:
            seen.add(key)
            off_targets.append(gene)

    if additional_genes:
        for gene in additional_genes:
            if not gene:
                continue
            gene_str = str(gene).strip()
            if not gene_str:
                continue
            key = gene_str.lower()
            if key in {"none", "nan", "null"}:
                continue
            if target_norm and key == target_norm:
                continue
            if key not in seen:
                seen.add(key)
                off_targets.append(gene_str)

    return off_targets


def _normalize_id_set(
    canonical_ids: List[str],
    supplemental_ids: Optional[List[str]] = None,
) -> List[str]:
    """Return unique Ensembl IDs without duplicates."""
    seen: set[str] = set()
    out: List[str] = []
    for entry in canonical_ids or []:
        if not entry:
            continue
        cleaned = str(entry).strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    for entry in supplemental_ids or []:
        if not entry:
            continue
        cleaned = str(entry).strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _parse_bed_metadata(name_field: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    for token in name_field.split("|"):
        if "=" in token:
            key, value = token.split("=", 1)
            info[key] = value
    return info


def _locate_scored_bed(directory: str) -> Optional[str]:
    try:
        for entry in os.listdir(directory):
            if entry.endswith(".scored.bed"):
                path = os.path.join(directory, entry)
                if os.path.isfile(path):
                    return path
    except Exception:
        return None
    return None


def _load_sequence_mapping(bed_path: str) -> Dict[str, str]:
    """Return mapping of sequence -> mappedto string from the scored BED."""
    mapping: Dict[str, str] = {}
    try:
        with open(bed_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                metadata = _parse_bed_metadata(parts[3])
                seq = metadata.get("seq")
                mapped = metadata.get("mappedto")
                if seq and mapped:
                    mapping[seq.upper()] = mapped
    except Exception:
        return {}
    return mapping


def _load_sequence_metadata(bed_path: str) -> Dict[str, Dict[str, str]]:
    """Return mapping of sequence -> parsed scored BED metadata."""
    mapping: Dict[str, Dict[str, str]] = {}
    try:
        with open(bed_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                metadata = _parse_bed_metadata(parts[3])
                seq = metadata.get("seq")
                if seq:
                    mapping[seq.upper()] = metadata
    except Exception:
        return {}
    return mapping


def _parse_mapped_locations(mapped_value: str) -> List[Tuple[str, int, int]]:
    if not mapped_value or mapped_value.lower() in {"none", "nan", "null"}:
        return []
    if "3+matches" in mapped_value:
        return []
    regions: List[Tuple[str, int, int]] = []
    for token in mapped_value.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.match(r"^(chr[0-9A-Za-z_.]+):(\d+)-(\d+)$", token)
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
        regions.append((chrom, start, end))
    return regions


def _first_present_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _as_int_or_zero(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def _measurement_int_or_unknown(value: Any) -> Optional[int]:
    try:
        number = float(str(value).strip())
        return int(number) if number >= 0 and number.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def _measurement_float_or_unknown(value: Any) -> Optional[float]:
    import math
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


_GENE_INDEX_CACHE: Dict[str, Dict[str, Dict[str, List]]] = {}


def _load_gene_index(annotation_path: str) -> Dict[str, Dict[str, List]]:
    cached = _GENE_INDEX_CACHE.get(annotation_path)
    if cached is not None:
        return cached

    gene_index: Dict[str, Dict[str, List]] = {}
    try:
        with open(annotation_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                chrom, feature = parts[0], parts[2]
                if feature != "gene":
                    continue
                try:
                    start = int(parts[3])
                    end = int(parts[4])
                except ValueError:
                    continue
                attrs = parts[8]
                gene_name = None
                gene_id = None
                for attr in attrs.split(";"):
                    attr = attr.strip()
                    if attr.startswith("gene_name"):
                        _, value = attr.split(" ", 1)
                        gene_name = value.strip().strip('"')
                    elif attr.startswith("gene_id"):
                        _, value = attr.split(" ", 1)
                        gene_id = value.strip().strip('"')
                if not gene_name and not gene_id:
                    continue
                chrom_index = gene_index.setdefault(chrom, {"starts": [], "intervals": []})
                chrom_index["starts"].append(start)
                chrom_index["intervals"].append((start, end, gene_name, gene_id))
    except Exception:
        _GENE_INDEX_CACHE[annotation_path] = {}
        return {}

    for chrom, info in gene_index.items():
        paired = sorted(zip(info["starts"], info["intervals"]), key=lambda x: x[0])
        info["starts"] = [p[0] for p in paired]
        info["intervals"] = [p[1] for p in paired]

    _GENE_INDEX_CACHE[annotation_path] = gene_index
    return gene_index


def _lookup_genes(chrom: str, start: int, end: int, gene_index: Dict[str, Dict[str, List]]) -> List[Tuple[Optional[str], Optional[str]]]:
    info = gene_index.get(chrom)
    if not info:
        return []
    starts = info["starts"]
    intervals = info["intervals"]
    hi = bisect.bisect_right(starts, end)
    genes: List[Tuple[Optional[str], Optional[str]]] = []
    seen: set[Tuple[Optional[str], Optional[str]]] = set()
    for idx in range(hi):
        iv_start, iv_end, gene_name, gene_id = intervals[idx]
        if iv_end < start:
            continue
        key = (gene_name or "", gene_id or "")
        if key in seen:
            continue
        seen.add(key)
        genes.append((gene_name, gene_id))
    return genes


@function_tool
def emit_pipeline_config(structured_goal_json: str, bed_json: str, chemistry_json: str, sample_name: str = "QUERY1", output_dir: str = "") -> Dict[str, str]:
    """Create BED + config.yaml for the tiling pipeline and return run command.

    Inputs are JSON strings. Expected shapes:
    - structured_goal_json: { "gene_symbol": str, ... }
    - bed_json: { "rows": [[chr, start, end, name, score, strand], ...] }
    - chemistry_json: { "chemistry": "5-10-5" | "steric-hindrance" | "3-10-3", ... }

    Returns JSON with keys: config_path, bed_path, output_dir, snakemake_cmd, tile_len, notes
    """
    try:
        structured_goal = json.loads(structured_goal_json) if structured_goal_json else {}
    except Exception:
        structured_goal = {}
    try:
        bed_info = json.loads(bed_json) if bed_json else {}
    except Exception:
        bed_info = {}
    try:
        chemistry = json.loads(chemistry_json) if chemistry_json else {}
    except Exception:
        chemistry = {}

    gene_symbol = (structured_goal or {}).get("gene_symbol") or sample_name or "QUERY1"
    sample_base = gene_symbol.replace(" ", "_")

    # Each emitted command must retain the exact BED and configuration that
    # produced it.  A process-global config or input path lets a later chemistry
    # group (or concurrent run) silently change an earlier command's meaning.
    # Keep execution inputs beside, but outside, the result directory so result
    # discovery cannot mistake the source BED for a pipeline output.
    resolved_output_dir = (
        output_dir
        if output_dir
        else os.path.join(os.path.dirname(PIPELINE_ROOT), "runs", f"{sample_base}", "tiling")
    )
    OUTPUT_DIR = os.path.abspath(os.path.expanduser(resolved_output_dir))
    runtime_dir = f"{OUTPUT_DIR}.runtime"
    runtime_inputs_dir = os.path.join(runtime_dir, "inputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(runtime_inputs_dir, exist_ok=True)
    bed_path = os.path.join(runtime_inputs_dir, f"{sample_base}.bed")

    # Write BED6 file
    rows: List[List] = bed_info.get("rows") or bed_info.get("bed6") or []
    if rows and isinstance(rows, list):
        with open(bed_path, "w", encoding="utf-8") as f:
            for r in rows:
                try:
                    chrom, start0, end, name, score, strand = r
                    f.write(f"{chrom}\t{int(start0)}\t{int(end)}\t{name}\t{int(score)}\t{strand}\n")
                except Exception:
                    continue

    # Defaults and references
    species_key = normalize_species((structured_goal or {}).get("species"))
    species_cfg = get_species_config(species_key)
    patient_reference = (structured_goal.get("patient_reference") or {}) if isinstance(structured_goal, dict) else {}
    patient_fasta = patient_reference.get("fasta_path") if isinstance(patient_reference, dict) else None
    genome_fasta = os.path.abspath(patient_fasta) if patient_fasta else os.fspath(species_cfg.genome_fasta)
    phastcons_bw = os.fspath(species_cfg.phastcons_path)
    phylop_bw = os.fspath(species_cfg.phylop_path)
    chrom_sizes = os.fspath(species_cfg.chrom_sizes_path)
    bowtie_index = os.fspath(species_cfg.bowtie_prefix)
    conservation_mode = str(getattr(species_cfg, "conservation_mode", "available") or "available")
    if conservation_mode != "disabled" and (not os.path.exists(phastcons_bw) or not os.path.exists(phylop_bw)):
        conservation_mode = "disabled"

    # Parameters from goal/chemistry or sensible defaults
    chemistry_mode = _extract_chemistry_mode(structured_goal, chemistry)
    tile_len = _resolve_tile_length(structured_goal if isinstance(structured_goal, dict) else {}, chemistry, default=20)
    tile_len, length_note = _normalize_gapmer_tile_length(chemistry_mode, tile_len)
    num_mismatches = _resolve_num_mismatches(structured_goal if isinstance(structured_goal, dict) else {}, default=1)
    if isinstance(structured_goal, dict):
        structured_goal["num_mismatches"] = num_mismatches

    def _extract_gc_bound(goal: Dict[str, Any], key: str, default: float) -> float:
        """Prefer explicit gc_min/gc_max but fall back to constraint blocks."""
        candidates: List[Any] = []
        if isinstance(goal, dict):
            candidates.append(goal.get(key))
            param_constraints = goal.get("parameter_constraints")
            if isinstance(param_constraints, dict):
                candidates.append(param_constraints.get(key))
                gc_entry = param_constraints.get("gc_content")
                if isinstance(gc_entry, dict):
                    if key == "gc_min":
                        candidates.append(gc_entry.get("min"))
                    else:
                        candidates.append(gc_entry.get("max"))
            constraint_block = goal.get("constraints")
            if isinstance(constraint_block, dict):
                candidates.append(constraint_block.get(key))
                gc_entry = constraint_block.get("gc_content")
                if isinstance(gc_entry, dict):
                    if key == "gc_min":
                        candidates.append(gc_entry.get("min"))
                    else:
                        candidates.append(gc_entry.get("max"))
        for candidate in candidates:
            val = _pc_coerce_fraction(candidate)
            if val is not None:
                return val
        return default

    gc_min = _extract_gc_bound(structured_goal, "gc_min", 0.4)
    gc_max = _extract_gc_bound(structured_goal, "gc_max", 0.7)
    phastcons_threshold = _safe_float(structured_goal.get("phastcons_threshold", 0.9), 0.9)
    phylop_threshold = _safe_float(structured_goal.get("phylop_threshold", 0.9), 0.9)

    # Secondary mapping resources
    secondary_index = bowtie_index  # fallback to same index
    secondary_annotation = os.fspath(species_cfg.gtf_path)

    config_path = os.path.join(runtime_dir, "config.generated.yaml")

    # Build YAML manually to avoid requiring PyYAML
    yaml_lines: List[str] = []
    yaml_lines.append(f"INPUT_FOLDER: {runtime_inputs_dir}")
    yaml_lines.append("REGIONS:")
    yaml_lines.append(f"  - {os.path.basename(bed_path)}")
    yaml_lines.append(f"OUTPUT_FOLDER: {OUTPUT_DIR}")
    yaml_lines.append("")
    yaml_lines.append(f"TILE_LEN: {tile_len}")
    yaml_lines.append(f"CHEMISTRY: {json.dumps(chemistry_mode)}")
    yaml_lines.append("")
    yaml_lines.append(f"GENOME: {genome_fasta}")
    yaml_lines.append(f"PHASTCONS: {phastcons_bw}")
    yaml_lines.append(f"PHYLOP: {phylop_bw}")
    yaml_lines.append(f"BOWTIE_INDEX: {bowtie_index}")
    yaml_lines.append(f"SECONDARY_INDEX: {secondary_index}")
    yaml_lines.append(f"SECONDARY_ANNOTATION: {secondary_annotation}")
    yaml_lines.append(f"CHROM_SIZES: {chrom_sizes}")
    yaml_lines.append(f"UCSC_SPECIES: {species_cfg.ucsc_build}")
    yaml_lines.append(f"CONSERVATION_MODE: {conservation_mode}")
    yaml_lines.append("")
    yaml_lines.append(f"PHASTCONS_THRESHOLD: {phastcons_threshold}")
    yaml_lines.append(f"PHYLOP_THRESHOLD: {phylop_threshold}")
    yaml_lines.append(f"NUM_MISMATCHES: {num_mismatches}")
    yaml_lines.append(f"GC_MIN: {gc_min}")
    yaml_lines.append(f"GC_MAX: {gc_max}")

    homopolymer_constraints = structured_goal.get("parameter_constraints") if isinstance(structured_goal, dict) else {}
    if not isinstance(homopolymer_constraints, dict):
        homopolymer_constraints = {}
    if "homopolymer_max_run" not in homopolymer_constraints and structured_goal.get("homopolymer_max_run") is not None:
        homopolymer_constraints = dict(homopolymer_constraints)
        homopolymer_constraints["homopolymer_max_run"] = structured_goal.get("homopolymer_max_run")

    avoid_sequences: List[str] = []

    def _append_sequence(seq: Any) -> None:
        if not seq:
            return
        seq_norm = str(seq).upper()
        if seq_norm and seq_norm not in avoid_sequences:
            avoid_sequences.append(seq_norm)

    allow_homopolymers = False
    if isinstance(homopolymer_constraints, dict):
        allow_homopolymers = str(homopolymer_constraints.get("allow_homopolymer_runs") or "").lower() in {"true", "1", "yes"}
        homo_entry = homopolymer_constraints.get("homopolymers")
        has_homopolymer_settings = (
            homopolymer_constraints.get("homopolymer_max_run") is not None
            or (
                isinstance(homo_entry, dict)
                and (
                    homo_entry.get("max_run") is not None
                    or bool(homo_entry.get("per_base_max"))
                    or bool(homo_entry.get("avoid_sequences"))
                )
            )
        )
        if not allow_homopolymers and not has_homopolymer_settings:
            homopolymer_constraints = dict(homopolymer_constraints)
            homopolymer_constraints["homopolymer_max_run"] = DEFAULT_HOMOPOLYMER_LIMIT
    if not allow_homopolymers:
        for seq in get_homopolymer_sequences(homopolymer_constraints):
            _append_sequence(seq)

    if avoid_sequences:
        yaml_lines.append("AVOID:")
        for seq in avoid_sequences:
            yaml_lines.append(f"  - {seq}")
    else:
        yaml_lines.append("AVOID: []")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    # Construct snakemake command (user can run; may require dependencies).
    # Snakefile shell rules call `python`; force them to use the same environment
    # that generated the config instead of whichever interpreter is first in the
    # user's login shell.
    from aso_agents.runtime_binaries import ensure_snakemake, ensure_tiling_cli_tools, refresh_runtime_path, snakemake_argv

    ensure_snakemake()
    ensure_tiling_cli_tools()
    runtime_path = refresh_runtime_path()
    snakemake_prefix = " ".join(shlex.quote(part) for part in snakemake_argv())
    snakemake_cmd = (
        f"cd {shlex.quote(PIPELINE_ROOT)} && "
        f"PATH={shlex.quote(runtime_path)} "
        f"{snakemake_prefix} -s Snakefile --configfile {shlex.quote(config_path)} "
        "-j 4 --scheduler greedy"
    )

    notes = []
    if length_note:
        notes.append(length_note)
    if patient_fasta:
        notes.append(f"patient_reference={patient_fasta}")
    if not os.path.exists(phylop_bw):
        if conservation_mode == "disabled":
            notes.append("conservation_mode=disabled; neutral conservation scores will be used")
        else:
            notes.append(f"phyloP bigWig not found; expected at {phylop_bw}")
    bowtie_missing = not any(
        os.path.exists(f"{bowtie_index}.{suffix}") for suffix in ("1.ebwt", "1.ebwtl")
    )
    if bowtie_missing:
        notes.append(f"Bowtie index not found; ensure index files exist at prefix {bowtie_index}")

    return {
        "config_path": config_path,
        "bed_path": bed_path,
        "output_dir": OUTPUT_DIR,
        "snakemake_cmd": snakemake_cmd,
        "tile_len": tile_len,
        "notes": "; ".join(notes) if notes else ""
    }


# New: extract scored results as JSON matching the requested schema
@function_tool
def collect_results_json(
    output_dir: str,
    table_name: str = "aso_tiles",
    primary_key: str = "",
    target_gene_symbol: Optional[str] = None,
    species: Optional[str] = None,
) -> Dict:
    """Scan OUTPUT_DIR for a scored BED/TSV and return JSON with schema + rows.

    Output format:
    {
      "schema": {
        "table": "",
        "primary_key": "",
        "columns": {"": ""},
        "notes": ""
      },
      "rows": [ { ... } ]
    }

    - Columns are derived from the file header when present; otherwise BED6 names
      plus generic fieldN for extra columns.
    - Types are inferred from values: integer | number | string.
    - Primary key defaults to 'name' when present, else 'id', else 'chrom,start,end'.
    - Pass `species` when available so gene lookups use the matching annotation set.
    """
    def _is_comment(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if stripped.startswith("#") or stripped.startswith("track") or stripped.startswith("browser"):
            return True
        # BED name fields can have metadata separated by '|'. Treat those as data lines.
        fields = stripped.split("\t")
        if len(fields) >= 4 and "|" in fields[3]:
            return False
        return False

    def _infer_type(v: str) -> str:
        v = v.strip()
        if v == "":
            return "string"
        try:
            iv = int(v)
            return "integer"
        except Exception:
            try:
                float(v)
                return "number"
            except Exception:
                return "string"

    def _widen_type(prev: str, new: str) -> str:
        order = {"integer": 0, "number": 1, "string": 2}
        if prev not in order:
            return new
        if new not in order:
            return prev
        return prev if order[prev] >= order[new] else new

    # Find candidate result files
    candidates: List[tuple] = []
    if not os.path.isdir(output_dir):
        return {"error": f"Output directory not found: {output_dir}"}
    for root, _, files in os.walk(output_dir):
        for fn in files:
            lower = fn.lower()
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (".bed", ".tsv", ".txt"):
                continue
            priority = 2
            if "scored.idt" in lower or lower.endswith(".idt.tsv"):
                priority = -1
            elif ("score" in lower or "scored" in lower or "rank" in lower):
                priority = 0
            elif ("tile" in lower or "tiles" in lower):
                priority = 1
            full = os.path.join(root, fn)
            try:
                size = os.path.getsize(full)
            except Exception:
                size = 0
            candidates.append((priority, -size, full))
    candidates.sort()
    if not candidates:
        return {"error": f"No BED/TSV results found in {output_dir}"}
    result_path = candidates[0][2]

    # Parse file
    header: List[str] = []
    data_start = 0
    lines: List[str] = []
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as e:
        return {"error": f"Failed reading results: {e}"}

    # Try to recover header from comment lines, otherwise default to BED6 mapping
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if _is_comment(line):
            stripped = line[1:].strip() if line.startswith("#") else ""
            if "\t" in stripped:
                header = [c.strip() for c in stripped.split("\t")]
            elif ":" in stripped and any(k in stripped.lower() for k in ("columns", "header")):
                maybe = stripped.split(":", 1)[1].strip()
                header = [c.strip() for c in re.split(r"[\t,]", maybe) if c.strip()]
            continue
        data_start = i
        break

    # Determine columns
    default_bed6 = ["chrom", "start", "end", "name", "score", "strand"]
    if data_start < len(lines):
        first_fields = lines[data_start].split("\t")
    else:
        first_fields = []

    if not header:
        # Detect explicit header row (e.g., IDT TSV header) on the first non-comment line
        idt_header = [
            "Chrom", "Start", "End", "Strand", "No_homopolymer", "GC", "Phastcon", "PhyloP",
            "No_secondary_structure", "Times_mapped", "Secondary_hit", "IDT", "Synthesis", "Sequence", "Score"
        ]
        if first_fields and len(first_fields) == len(idt_header) and (
            first_fields == idt_header or [c.strip().lower() for c in first_fields] == [c.lower() for c in idt_header]
        ):
            header = first_fields
            data_start += 1  # skip the header row from data
        elif first_fields:
            # Fallback header detection if the first row looks like a real header
            lower_fields = [c.strip().lower() for c in first_fields]
            if (
                len(lower_fields) >= 3 and
                lower_fields[0] in ("chrom", "chromosome") and
                lower_fields[1] in ("start", "chromstart") and
                lower_fields[2] in ("end", "chromend")
            ):
                header = first_fields
                data_start += 1
            else:
                if len(first_fields) <= 6:
                    header = default_bed6[:len(first_fields)]
                else:
                    header = default_bed6 + [f"field{i}" for i in range(7, len(first_fields) + 1)]
        else:
            header = default_bed6

    # Infer types
    col_types: Dict[str, str] = {c: "string" for c in header}
    for line in lines[data_start: data_start + 2000]:  # sample up to 2000 lines for typing
        if not line.strip() or _is_comment(line):
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        for c, v in zip(header, parts[:len(header)]):
            new_t = _infer_type(v)
            col_types[c] = _widen_type(col_types.get(c, new_t), new_t)

    sequence_to_mapped: Dict[str, str] = {}
    sequence_to_metadata: Dict[str, Dict[str, str]] = {}
    gene_index: Dict[str, Dict[str, List]] = {}
    bed_companion = _locate_scored_bed(os.path.dirname(result_path))
    if bed_companion:
        sequence_to_metadata = _load_sequence_metadata(bed_companion)
        sequence_to_mapped = _load_sequence_mapping(bed_companion)
        if sequence_to_mapped or sequence_to_metadata:
            species_key = normalize_species(species)
            species_cfg = get_species_config(species_key)
            annotation_path = os.fspath(species_cfg.gtf_path)
            if os.path.exists(annotation_path):
                gene_index = _load_gene_index(annotation_path)

    target_norm = (target_gene_symbol or "").strip().lower()

    # Rows
    rows: List[Dict] = []
    for line in lines[data_start:]:
        if not line.strip() or _is_comment(line):
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        elif len(parts) > len(header):
            parts = parts[:len(header)]
        obj: Dict = {}
        for c, v in zip(header, parts):
            t = col_types.get(c, "string")
            if t == "integer":
                try:
                    obj[c] = int(v)
                except Exception:
                    obj[c] = v
            elif t == "number":
                try:
                    obj[c] = float(v)
                except Exception:
                    obj[c] = v
            else:
                obj[c] = v

        # Expand BED name metadata into explicit columns when present
        name_value = obj.get("name")
        if isinstance(name_value, str) and "|" in name_value:
            parsed_meta: Dict[str, str] = {}
            for token in name_value.split("|"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                key_lower = key.lower()
                if key_lower == "gc":
                    meta_key = "GC"
                elif key_lower == "pc":
                    meta_key = "Phastcon"
                elif key_lower == "ph":
                    meta_key = "PhyloP"
                elif key_lower in {"nostruct", "no_struct"}:
                    meta_key = "No_secondary_structure"
                    value = "True" if value.upper() in {"YES", "TRUE", "1"} else "False"
                elif key_lower == "mappednum":
                    meta_key = "Times_mapped"
                elif key_lower == "mappedto":
                    meta_key = "Mapped_to"
                elif key_lower == "secondary_hit":
                    meta_key = "Secondary_hit"
                elif key_lower == "seq":
                    meta_key = "Sequence"
                elif key_lower == "overflow":
                    meta_key = "Overflow"
                    value = True if value.upper() in {"YES", "TRUE", "1"} else False
                elif key_lower in {"orig_id", "id"}:
                    meta_key = "Orig_id"
                else:
                    meta_key = key
                parsed_meta[meta_key] = value
            for meta_k, meta_v in parsed_meta.items():
                obj.setdefault(meta_k, meta_v)
                if meta_k not in header:
                    header.append(meta_k)
                if meta_k == "Overflow":
                    col_types[meta_k] = "boolean"
                elif meta_k not in col_types:
                    col_types[meta_k] = "string"

        seq_key = str(obj.get("Sequence") or obj.get("sequence") or "").upper()
        seq_metadata = sequence_to_metadata.get(seq_key, {})
        for field in ('Phastcon', 'PhyloP'):
            if field in obj:
                obj[field] = _measurement_float_or_unknown(obj[field])
                col_types[field] = 'number'
        for field in ('pc_covered_bases', 'ph_covered_bases', 'conservation_interval_bases'):
            value = _first_present_value(obj.get(field), seq_metadata.get(field))
            if value is not None:
                obj[field] = _measurement_int_or_unknown(value)
                col_types[field] = 'integer'
                if field not in header:
                    header.append(field)
        off_target_budget = _as_int_or_zero(
            _first_present_value(
                obj.get("Off_target_mismatch_budget"),
                obj.get("off_target_mismatch_budget"),
                seq_metadata.get("off_target_mismatch_budget"),
            )
        )
        if off_target_budget <= 0:
            off_target_budget = 0
        off_target_budget = min(off_target_budget, 3)

        for mm, label in ((0, "exact"), (1, "1mm"), (2, "2mm"), (3, "3mm")):
            source_key = f"off_target_{label}_count"
            if label == "exact":
                pascal_key = "Off_target_exact_count"
            else:
                pascal_key = f"Off_target_{label}_count"
            value = _first_present_value(obj.get(pascal_key), obj.get(source_key), seq_metadata.get(source_key))
            if value is not None:
                obj[pascal_key] = _measurement_int_or_unknown(value)

        total_value = _first_present_value(
            obj.get("Off_target_total_upto_mismatch_count"),
            obj.get(f"off_target_total_upto_{off_target_budget}mm_count"),
            seq_metadata.get(f"off_target_total_upto_{off_target_budget}mm_count"),
        )
        if total_value is not None:
            obj["Off_target_total_upto_mismatch_count"] = _measurement_int_or_unknown(total_value)
        obj["Off_target_mismatch_budget"] = off_target_budget
        risk_value = _first_present_value(
            obj.get("Off_target_weighted_risk_score"),
            obj.get("off_target_weighted_risk_score"),
            seq_metadata.get("off_target_weighted_risk_score"),
        )
        if risk_value is not None:
            obj["Off_target_weighted_risk_score"] = _measurement_int_or_unknown(risk_value)
        for key in ('off_target_search_status', 'off_target_exact_count_unit', 'off_target_position_convention'):
            if key in seq_metadata:
                obj[key] = seq_metadata[key]

        mapped_value = sequence_to_mapped.get(seq_key)
        mapped_gene_names: List[str] = []
        mapped_gene_ids: List[str] = []
        if mapped_value and gene_index:
            regions = _parse_mapped_locations(mapped_value)
            for chrom_v, start_v, end_v in regions:
                for gene_name, gene_id in _lookup_genes(chrom_v, start_v, end_v, gene_index):
                    name_clean = (gene_name or "").strip()
                    if target_norm and name_clean.lower() == target_norm:
                        continue
                    if gene_name:
                        mapped_gene_names.append(gene_name)
                    if gene_id:
                        mapped_gene_ids.append(gene_id)

        mismatch_gene_names: Dict[str, List[str]] = {}
        mismatch_gene_ids: Dict[str, List[str]] = {}
        if gene_index:
            for mm in range(0, 4):
                label = f"{mm}mm"
                raw_positions = _first_present_value(
                    obj.get(f"Off_target_{label}_positions"),
                    obj.get(f"off_target_{label}_positions"),
                    seq_metadata.get(f"off_target_{label}_positions"),
                )
                names_for_bin: List[str] = []
                ids_for_bin: List[str] = []
                for chrom_v, start_v, end_v in _parse_mapped_locations(str(raw_positions or "")):
                    for gene_name, gene_id in _lookup_genes(chrom_v, start_v, end_v, gene_index):
                        name_clean = (gene_name or "").strip()
                        if target_norm and name_clean.lower() == target_norm:
                            continue
                        if gene_name:
                            names_for_bin.append(gene_name)
                        if gene_id:
                            ids_for_bin.append(gene_id)
                names_for_bin = _normalize_gene_set(names_for_bin, target_gene_symbol)
                ids_for_bin = _normalize_id_set([], ids_for_bin)
                mismatch_gene_names[label] = names_for_bin
                mismatch_gene_ids[label] = ids_for_bin
                obj[f"Off_target_{label}_genes"] = ",".join(names_for_bin)
                obj[f"Off_target_{label}_gene_ids"] = ",".join(ids_for_bin)

        off_target_genes = _normalize_gene_set(
            obj.get("Secondary_hit") or obj.get("secondary_hit"),
            target_gene_symbol,
            additional_genes=mapped_gene_names + [
                gene for genes in mismatch_gene_names.values() for gene in genes
            ],
        )
        off_target_ids = _normalize_id_set(
            [],
            supplemental_ids=mapped_gene_ids + [
                gene_id for ids in mismatch_gene_ids.values() for gene_id in ids
            ],
        )
        obj["Off_target_genes"] = ",".join(off_target_genes)
        obj["Off_target_gene_ids"] = ",".join(off_target_ids)
        obj["Off_target_genes_by_mismatch"] = mismatch_gene_names
        obj["Off_target_gene_ids_by_mismatch"] = mismatch_gene_ids
        add_locus_columns_to_row(obj)
        rows.append(obj)

    # Primary key selection
    pk = primary_key.strip() if primary_key else ("name" if "name" in header else ("id" if "id" in header else "chrom,start,end"))

    # Build payload and write to JSON file inside the output directory
    payload = {
        "schema": {
            "table": table_name,
            "primary_key": pk,
            "columns": {c: col_types.get(c, "string") for c in header},
            "notes": f"Parsed from {result_path}"
        },
        "rows": rows
    }
    payload["schema"]["columns"]["Off_target_genes"] = "string"
    payload["schema"]["columns"]["Off_target_gene_ids"] = "string"
    payload["schema"]["columns"]["Off_target_genes_by_mismatch"] = "object"
    payload["schema"]["columns"]["Off_target_gene_ids_by_mismatch"] = "object"
    payload["schema"]["columns"]["Off_target_mismatch_budget"] = "number"
    payload["schema"]["columns"]["Off_target_weighted_risk_score"] = "number"
    payload["schema"]["columns"]["Off_target_total_upto_mismatch_count"] = "number"
    ensure_locus_schema(payload["schema"]["columns"])
    for label in ("exact", "1mm", "2mm", "3mm"):
        prefix = "exact" if label == "exact" else label
        payload["schema"]["columns"][f"Off_target_{prefix}_count"] = "number"
    for label in ("0mm", "1mm", "2mm", "3mm"):
        payload["schema"]["columns"][f"Off_target_{label}_genes"] = "string"
        payload["schema"]["columns"][f"Off_target_{label}_gene_ids"] = "string"
    if "Times_mapped" in payload["schema"]["columns"]:
        payload["schema"]["columns"]["Times_mapped"] = "string"
    if "Overflow" in payload["schema"]["columns"]:
        payload["schema"]["columns"]["Overflow"] = "boolean"
    try:
        json_path = os.path.join(output_dir, "tiles.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        return {"error": f"Failed to write results JSON: {e}"}

    # Return only the path to the JSON file
    return {"json_path": json_path}


tiling_agent = Agent(
    name="Tiling Pipeline Agent",
    instructions=(
        "You generate a valid config and BED for the tiling pipeline. "
        "Always call the tool 'emit_pipeline_config' exactly once with JSON strings for structured_goal, bed, and chemistry. "
        "Return ONLY the tool's JSON output as your final answer."
    ),
    tools=[emit_pipeline_config],
    model="gpt-5-mini-2025-08-07",
) 
