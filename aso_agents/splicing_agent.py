from typing import Any, Dict, List, Optional, Callable, Mapping, Sequence, Tuple
import json
from agents import Agent, function_tool
from .benchmark.client import splice_resource_limited
from .transcript_chooser_agent import get_transcript_structure
import os
import re
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path
from Bio.Seq import Seq
from pyfaidx import Fasta  # type: ignore
from .species_config import get_species_config, normalize_species, try_normalize_species

# Optional progress callback (set by UI/orchestrator)
PROGRESS_CALLBACK: Optional[Callable[[str], None]] = None

BASE_DIR = Path(__file__).resolve().parents[1]
OPENSPLICEAI_SCRIPT = BASE_DIR / "scripts" / "run_openspliceai_eval.py"
OPENSPLICEAI_MODEL_DIR = Path(
    os.environ.get("OPENSPLICEAI_MODEL_DIR", BASE_DIR / "spliceAI_integration" / "openSpliceAI")
)
OPENSPLICEAI_MODEL_PATH = OPENSPLICEAI_MODEL_DIR / "OSAI_mouse_10000nt_model1.pt"
OPENSPLICEAI_MODEL_REGISTRY = (
    os.environ.get("OPENSPLICEAI_MODEL_REGISTRY", "").strip()
    or str(OPENSPLICEAI_MODEL_DIR / "model_registry.json")
)
OPENSPLICEAI_CONDA_ENV = os.environ.get("OPENSPLICEAI_CONDA_ENV", "").strip()
OPENSPLICEAI_FLANKING_SIZE = int(os.environ.get("OPENSPLICEAI_FLANKING_SIZE", "10000"))
OPENSPLICEAI_MODELS_URL = "https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models"
OPENSPLICEAI_MOUSE_10000NT_URL = (
    "https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mouse/10000nt"
)
SPLICETRANSFORMER_WEIGHTS_URL = (
    "https://drive.google.com/file/d/1d8n4vHDSbXqpPc_JFEswLomSUDBgHvno/view?usp=drive_link"
)
SPLICETRANSFORMER_HG38_FASTA_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
SPLICETRANSFORMER_GENCODE_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"
)
SPLICE_EVAL_NEARBY_SITE_RADIUS_BP = 2000


def _nearest_splice_site_for_mid(
    mid: int,
    normalized_sites: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    nearest: Optional[Dict[str, Any]] = None
    for site in normalized_sites:
        if not isinstance(site, Mapping):
            continue
        label = (
            str(site.get("canonical_label") or "").strip()
            or str(site.get("display_label") or "").strip()
            or str(site.get("label") or "").strip()
        )
        for site_kind in ("acceptor", "donor"):
            payload = site.get(site_kind)
            if not isinstance(payload, Mapping):
                continue
            try:
                position = int(payload.get("position"))
            except Exception:
                continue
            distance = abs(int(mid) - position)
            if nearest is None or distance < int(nearest.get("distance_bp") or 0):
                nearest = {
                    "distance_bp": distance,
                    "site_kind": site_kind,
                    "position": position,
                    "label": label or None,
                    "exon_index": site.get("exon_index"),
                    "isoform_label": site.get("isoform_label"),
                }
    return nearest


def _requested_splice_site_for_mid(
    mid: int,
    site: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(site, Mapping):
        return None
    label = (
        str(site.get("canonical_label") or "").strip()
        or str(site.get("display_label") or "").strip()
        or str(site.get("label") or "").strip()
    )
    site_kind = str(site.get("site_kind") or "").strip().lower()
    kinds: Sequence[str]
    if site_kind in {"acceptor", "donor"}:
        kinds = (site_kind,)
    else:
        kinds = ("acceptor", "donor")
    nearest: Optional[Dict[str, Any]] = None
    for kind in kinds:
        payload = site.get(kind)
        if not isinstance(payload, Mapping):
            continue
        try:
            position = int(payload.get("position"))
        except Exception:
            continue
        distance = abs(int(mid) - position)
        if nearest is None or distance < int(nearest.get("distance_bp") or 0):
            nearest = {
                "distance_bp": distance,
                "site_kind": kind,
                "position": position,
                "label": label or None,
                "exon_index": site.get("exon_index"),
                "isoform_label": site.get("isoform_label"),
            }
    return nearest


def _load_openspliceai_model_registry() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {
        "mouse": {
            "model_path": str(OPENSPLICEAI_MODEL_PATH),
            "source": "repo_default",
            "download_url": OPENSPLICEAI_MOUSE_10000NT_URL,
        },
        "mus_musculus": {
            "model_path": str(OPENSPLICEAI_MODEL_PATH),
            "source": "repo_default",
            "download_url": OPENSPLICEAI_MOUSE_10000NT_URL,
        },
    }
    inline = os.environ.get("OPENSPLICEAI_MODEL_REGISTRY_JSON", "").strip()
    payload: Any = None
    if inline:
        try:
            payload = json.loads(inline)
        except Exception:
            payload = None
    elif OPENSPLICEAI_MODEL_REGISTRY:
        try:
            registry_path = Path(OPENSPLICEAI_MODEL_REGISTRY).expanduser()
            if registry_path.exists():
                payload = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
    if isinstance(payload, dict):
        for raw_key, raw_value in payload.items():
            species_key = normalize_species(str(raw_key))
            if isinstance(raw_value, str):
                registry[species_key] = {"model_path": raw_value, "source": "registry"}
            elif isinstance(raw_value, dict):
                entry = dict(raw_value)
                if entry.get("model_path") or entry.get("path"):
                    entry["model_path"] = str(entry.get("model_path") or entry.get("path"))
                    entry.setdefault("source", "registry")
                    registry[species_key] = entry
    for key, value in os.environ.items():
        if not key.startswith("OPENSPLICEAI_MODEL_") or key in {
            "OPENSPLICEAI_MODEL_DIR",
            "OPENSPLICEAI_MODEL_PATH",
            "OPENSPLICEAI_MODEL_REGISTRY",
            "OPENSPLICEAI_MODEL_REGISTRY_JSON",
        }:
            continue
        species_key = normalize_species(key.removeprefix("OPENSPLICEAI_MODEL_").lower())
        if value.strip():
            registry[species_key] = {"model_path": value.strip(), "source": "environment"}
    return registry


def resolve_openspliceai_model(species_value: Optional[str]) -> Dict[str, Optional[str]]:
    species_key = normalize_species(species_value)
    registry = _load_openspliceai_model_registry()
    entry = registry.get(species_key)
    if not entry:
        return {
            "error": "openspliceai_model_unconfigured",
            "species": species_key,
            "model": None,
            "source": None,
        }
    model_path = str(entry.get("model_path") or "").strip()
    if not model_path:
        return {
            "error": "openspliceai_model_path_missing",
            "species": species_key,
            "model": None,
            "source": str(entry.get("source") or ""),
        }
    return {
        "species": species_key,
        "model": model_path,
        "source": str(entry.get("source") or ""),
        "download_url": str(entry.get("download_url") or "") or None,
    }


def resolve_splice_eval_backend(species_value: Optional[str]) -> Dict[str, Optional[str]]:
    """Return the splice-evaluation backend for a supported species."""
    species_key = try_normalize_species(species_value)
    if species_key is None:
        return {
            "error": "unsupported_splice_eval_species",
            "species": str(species_value or "").strip() or None,
            "backend": None,
            "model": None,
        }
    if species_key == "mouse":
        model_info = resolve_openspliceai_model(species_key)
        if model_info.get("error"):
            return {
                "error": model_info.get("error"),
                "species": species_key,
                "backend": "openspliceai",
                "model": model_info.get("model"),
                "download_url": model_info.get("download_url"),
            }
        return {
            "species": species_key,
            "backend": "openspliceai",
            "model": model_info.get("model"),
            "download_url": model_info.get("download_url"),
        }
    if species_key != "human":
        model_info = resolve_openspliceai_model(species_key)
        if model_info.get("error"):
            return {
                "error": "unsupported_splice_eval_species",
                "species": species_key,
                "backend": None,
                "model": None,
                "details": model_info.get("error"),
            }
        return {
            "species": species_key,
            "backend": "openspliceai",
            "model": model_info.get("model"),
            "download_url": model_info.get("download_url"),
        }
    return {
        "species": species_key,
        "backend": "splicetransformer",
        "model": "SpTransformer_pytorch.ckpt",
    }


def _call_openspliceai(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENSPLICEAI_SCRIPT.exists():
        return {
            "error": f"run_openspliceai_eval.py not found at {OPENSPLICEAI_SCRIPT}",
        }
    if OPENSPLICEAI_CONDA_ENV and not shutil.which("conda"):
        return {
            "error": "conda executable not found in PATH; required when OPENSPLICEAI_CONDA_ENV is set",
            "details": f"OPENSPLICEAI_CONDA_ENV={OPENSPLICEAI_CONDA_ENV}",
        }

    with tempfile.TemporaryDirectory(prefix="openspliceai_eval_") as tmpdir:
        tmp_dir_path = Path(tmpdir)
        input_path = tmp_dir_path / "payload.json"
        output_path = tmp_dir_path / "result.json"
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        if OPENSPLICEAI_CONDA_ENV:
            cmd = [
                "conda",
                "run",
                "-n",
                OPENSPLICEAI_CONDA_ENV,
                "python",
                str(OPENSPLICEAI_SCRIPT),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        else:
            cmd = [
                sys.executable,
                str(OPENSPLICEAI_SCRIPT),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        # A separately installed interpreter must use its own compiled Python
        # packages. Inheriting the caller's PYTHONPATH can load extensions built
        # for a different Python version (e.g. the shared benchmark NumPy).
        helper_env = None
        if OPENSPLICEAI_CONDA_ENV:
            helper_env = os.environ.copy()
            helper_env.pop("PYTHONPATH", None)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=helper_env)
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            stdout = proc.stdout.strip()
            message = stderr or stdout or "unknown error"
            return {
                "error": "openspliceai_inference_failed",
                "details": message,
                "returncode": proc.returncode,
                "command": cmd[:5],
            }

        if not output_path.exists():
            return {
                "error": "openspliceai_output_missing",
                "details": "Expected result.json was not created by helper script.",
            }
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            return {"error": f"openspliceai_output_parse_error: {exc}"}


def _evaluate_with_openspliceai(
    rows: List[Dict[str, Any]],
    tx: Dict[str, Any],
    normalized_sites: List[Dict[str, Any]],
    site_positions: List[int],
    site_window_max: int,
    focus_site_global: Optional[Dict[str, Any]],
    default_site_window: int,
    fixed_acc: Optional[int],
    fixed_don: Optional[int],
    motif: str,
    strand_val: int,
    strand_char_tx: str,
    ex_g: List[Dict[str, Any]],
    ref_fasta_path: Optional[str],
    exon_index: Optional[int],
    species_key: str,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    species_cfg = get_species_config(species_key)
    fasta_path = ref_fasta_path or os.fspath(species_cfg.genome_fasta)
    if not fasta_path or not os.path.exists(fasta_path):
        return {"error": f"Reference FASTA not found: {fasta_path}"}

    try:
        fasta = Fasta(fasta_path)
    except Exception as exc:
        return {"error": f"Failed to open FASTA {fasta_path}: {exc}"}

    def map_col(row_keys: List[str], candidates: List[str]) -> Optional[str]:
        lower = {k.lower(): k for k in row_keys}
        for name in candidates:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    def roi_rel_index(genomic_pos: int, left: int, right: int, strand_char: str) -> int:
        return int(genomic_pos - left) if strand_char == '+' else int(right - genomic_pos)

    def fetch_sequence(chrom: str, start: int, end: int) -> Optional[str]:
        try:
            return fasta.get_seq(chrom, start, end).seq.upper()
        except Exception:
            if chrom.startswith("chr"):
                alt = chrom[3:]
            else:
                alt = f"chr{chrom}"
            try:
                return fasta.get_seq(alt, start, end).seq.upper()
            except Exception:
                return None

    def compute_exon_junctions(exon_dict: Dict[str, Any]) -> Tuple[int, int]:
        if strand_val == 1:
            return int(exon_dict["start"]), int(exon_dict["end"])
        return int(exon_dict["end"]), int(exon_dict["start"])

    requests_payload: List[Dict[str, Any]] = []
    contexts: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            _log_skip("row is not a dict; skipping")
            continue

        keys = list(row.keys())
        col_chrom = map_col(keys, ["Chrom", "chrom", "chromosome", "seq_region_name"])
        col_start = map_col(keys, ["Start", "start", "chromstart"])
        col_end = map_col(keys, ["End", "end", "chromend"])
        col_strand = map_col(keys, ["Strand", "strand"])
        col_seq = map_col(keys, ["Sequence", "sequence", "seq", "Seq", "Oligo", "ASO"])
        col_name = map_col(keys, ["name", "Name", "ID", "id", "label"])

        if not (col_chrom and col_start and col_end and col_strand and col_seq):
            _log_skip("required columns missing; skipping row")
            continue

        chrom = str(row.get(col_chrom))
        try:
            start0 = int(row.get(col_start))
            end_bp = int(row.get(col_end))
        except Exception:
            _log_skip("start/end not integers; skipping row")
            continue

        seq_value = str(row.get(col_seq) or "")
        if not seq_value:
            _log_skip("sequence column empty; skipping row")
            continue

        length = len(seq_value)
        offset = start0 + 1
        strand_char = _normalize_strand_char(row.get(col_strand), strand_val)
        mid = int((start0 + end_bp) // 2) + 1

        acc_j = None
        don_j = None
        target_positions: List[int] = []

        if normalized_sites:
            # Restrict to splice sites within ±2000 bp of the ASO span
            near_sites = []
            left_bp = min(offset, offset + length - 1)
            right_bp = max(offset, offset + length - 1)
            for pos in site_positions:
                if abs(pos - left_bp) <= 2000 or abs(pos - right_bp) <= 2000:
                    near_sites.append(pos)
            target_positions.extend(near_sites)
        else:
            if fixed_acc is not None and fixed_don is not None:
                acc_j, don_j = fixed_acc, fixed_don
            else:
                if not ex_g:
                    _log_skip("no exons_genomic data; skipping row")
                    continue
                chosen_exon = None
                min_dist = None
                for ex in ex_g:
                    ex_start = int(ex["start"])
                    ex_end = int(ex["end"])
                    if ex_start <= mid <= ex_end:
                        chosen_exon = ex
                        break
                    dist = abs(mid - ex_start) if mid < ex_start else abs(mid - ex_end)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        chosen_exon = ex
                if not chosen_exon:
                    _log_skip("no exon overlap for row; skipping")
                    continue
                acc_j, don_j = compute_exon_junctions(chosen_exon)
            if acc_j is None or don_j is None:
                _log_skip("unable to determine splice junctions; skipping row")
                continue
            target_positions.extend([acc_j, don_j])

        target_positions.extend([offset, offset + length - 1])

        if not target_positions:
            _log_skip("no nearby splice sites for ASO; skipping")
            continue

        roi_left_row = max(1, min(target_positions) - site_window_max)
        roi_right_row = max(roi_left_row + 1, max(target_positions) + site_window_max)

        roi_seq_genomic = fetch_sequence(chrom, roi_left_row, roi_right_row)
        if not roi_seq_genomic:
            _log_skip(f"failed to fetch reference sequence for {chrom}:{roi_left_row}-{roi_right_row}")
            continue

        aso_offset = offset - roi_left_row
        aso_end_offset = aso_offset + length - 1
        if aso_offset < 0 or aso_end_offset >= len(roi_seq_genomic):
            _log_skip("ASO bounds fall outside ROI; skipping row")
            continue

        alt_repeat = (motif * ((length + len(motif) - 1) // len(motif)))[:length]
        placeholder = "N" * length

        del_seq_genomic = (
            roi_seq_genomic[:aso_offset] + placeholder + roi_seq_genomic[aso_end_offset + 1 :]
        )
        rep_seq_genomic = (
            roi_seq_genomic[:aso_offset] + alt_repeat + roi_seq_genomic[aso_end_offset + 1 :]
        )

        def orient(seq: str) -> str:
            if strand_char == "-":
                return str(Seq(seq).reverse_complement())
            return seq

        roi_seq_oriented = orient(roi_seq_genomic)
        del_seq_oriented = orient(del_seq_genomic)
        rep_seq_oriented = orient(rep_seq_genomic)

        def build_site_entries() -> Tuple[List[Dict[str, Any]], Optional[Dict[str, int]]]:
            site_entries: List[Dict[str, Any]] = []
            for idx_site, site in enumerate(normalized_sites, start=1):
                try:
                    acc_pos = int(site["acceptor"]["position"])
                    don_pos = int(site["donor"]["position"])
                except Exception:
                    continue
                acc_idx = roi_rel_index(acc_pos, roi_left_row, roi_right_row, strand_char)
                don_idx = roi_rel_index(don_pos, roi_left_row, roi_right_row, strand_char)
                window_radius = int(site.get("window_bp") or default_site_window)
                site_entries.append(
                    {
                        "key": f"{idx}_{idx_site}",
                        "acceptor_index": acc_idx,
                        "donor_index": don_idx,
                        "window": window_radius,
                        "metadata": site,
                    }
                )
            primary_entry = None
            if site_entries:
                chosen = None
                if focus_site_global:
                    for entry in site_entries:
                        meta = entry["metadata"]
                        if (
                            focus_site_global.get("label") == meta.get("label")
                            or focus_site_global.get("exon_index") == meta.get("exon_index")
                        ):
                            chosen = entry
                            break
                if chosen is None:
                    chosen = site_entries[0]
                primary_entry = {
                    "acceptor_index": chosen["acceptor_index"],
                    "donor_index": chosen["donor_index"],
                    "window": chosen["window"],
                }
            elif acc_j is not None and don_j is not None:
                primary_entry = {
                    "acceptor_index": roi_rel_index(acc_j, roi_left_row, roi_right_row, strand_char),
                    "donor_index": roi_rel_index(don_j, roi_left_row, roi_right_row, strand_char),
                    "window": default_site_window,
                }
            return site_entries, primary_entry

        site_entries, primary_entry = build_site_entries()
        tile_id = f"tile_{idx}"
        requests_payload.append(
            {
                "id": tile_id,
                "reference_sequence": roi_seq_oriented,
                "deletion_sequence": del_seq_oriented,
                "replacement_sequence": rep_seq_oriented,
                "sites": [
                    {
                        "key": entry["key"],
                        "acceptor_index": entry["acceptor_index"],
                        "donor_index": entry["donor_index"],
                        "window": entry["window"],
                    }
                    for entry in site_entries
                ],
                "primary": primary_entry,
            }
        )

        contexts.append(
            {
                "id": tile_id,
                "name": str(row.get(col_name)) if col_name else None,
                "chrom": chrom,
                "start": start0,
                "end": end_bp,
                "strand": strand_char,
                "length": length,
                "site_entries": site_entries,
                "primary_sites": primary_entry,
                "normalized_sites": normalized_sites,
                "default_site_window": default_site_window,
                "acc_j": acc_j,
                "don_j": don_j,
                "roi_left": roi_left_row,
                "roi_right": roi_right_row,
            }
        )

    if not requests_payload:
        fasta.close()
        return {"error": "No valid tiles available for OpenSpliceAI evaluation."}

    payload = {
        "model_path": str(model_path or OPENSPLICEAI_MODEL_PATH),
        "flanking_size": OPENSPLICEAI_FLANKING_SIZE,
        "requests": requests_payload,
    }
    response = _call_openspliceai(payload)
    fasta.close()

    if response.get("error"):
        return response

    result_entries = response.get("results")
    if not isinstance(result_entries, list):
        return {"error": "openspliceai_result_invalid"}

    result_map = {entry.get("id"): entry for entry in result_entries if isinstance(entry, dict)}

    results: List[Dict[str, Any]] = []
    for ctx in contexts:
        tile_result = result_map.get(ctx["id"])
        if not tile_result:
            continue
        primary_metrics = tile_result.get("metrics") or {}
        deletion_metrics = primary_metrics.get("deletion") or {}
        replacement_metrics = primary_metrics.get("replacement") or {}

        def _extract(metric_map: Dict[str, Any], key: str) -> Optional[float]:
            value = metric_map.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            return None

        site_metric_map = {}
        for site_entry in tile_result.get("sites") or []:
            if isinstance(site_entry, dict) and site_entry.get("key"):
                site_metric_map[site_entry["key"]] = site_entry

        site_deltas: List[Dict[str, Any]] = []
        for entry in ctx["site_entries"]:
            meta = entry["metadata"]
            metrics_entry = site_metric_map.get(entry["key"], {})
            del_metrics = metrics_entry.get("deletion") or {}
            rep_metrics = metrics_entry.get("replacement") or {}
            site_deltas.append(
                {
                    **meta,
                    "evaluation_status": "evaluated" if metrics_entry else "unavailable",
                    "delta_inclusion_deletion": _extract(del_metrics, "pair"),
                    "delta_inclusion_replacement": _extract(rep_metrics, "pair"),
                    "delta_acceptor_deletion": _extract(del_metrics, "acceptor"),
                    "delta_acceptor_replacement": _extract(rep_metrics, "acceptor"),
                    "delta_donor_deletion": _extract(del_metrics, "donor"),
                    "delta_donor_replacement": _extract(rep_metrics, "donor"),
                }
            )

        def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
            vals = [float(v) for v in values if isinstance(v, (int, float))]
            if not vals:
                return None
            return float(sum(vals) / len(vals))

        delta_del = _extract(deletion_metrics, "pair")
        delta_rep = _extract(replacement_metrics, "pair")

        result_entry = {
            "name": ctx["name"],
            "chrom": ctx["chrom"],
            "start": ctx["start"],
            "end": ctx["end"],
            "strand": ctx["strand"],
            "length": ctx["length"],
            "delta_inclusion_deletion": delta_del,
            "delta_inclusion_replacement": delta_rep,
        }
        if site_deltas:
            result_entry["splice_site_deltas"] = site_deltas
            result_entry["delta_inclusion_deletion_mean"] = _mean(
                [sd.get("delta_inclusion_deletion") for sd in site_deltas]
            )
            result_entry["delta_inclusion_replacement_mean"] = _mean(
                [sd.get("delta_inclusion_replacement") for sd in site_deltas]
            )
            if focus_site_global:
                for sd in site_deltas:
                    if sd.get("label") == focus_site_global.get("label"):
                        result_entry["primary_splice_site_label"] = sd.get("label")
                        break
            else:
                result_entry["primary_splice_site_label"] = site_deltas[0].get("label")

        results.append(result_entry)

    primary_acceptor = focus_site_global.get("acceptor", {}).get("position") if focus_site_global else None
    primary_donor = focus_site_global.get("donor", {}).get("position") if focus_site_global else None
    if primary_acceptor is None and fixed_acc is not None:
        primary_acceptor = fixed_acc
    if primary_donor is None and fixed_don is not None:
        primary_donor = fixed_don

    return {
        "transcript_id": tx.get("id"),
        "exon_index": exon_index,
        "acceptor_junction": primary_acceptor,
        "donor_junction": primary_donor,
        "focus_splice_site_label": focus_site_global.get("label") if focus_site_global else None,
        "pad": default_site_window,
        "site_window_bp": default_site_window,
        "splice_sites": normalized_sites,
        "ref_fasta": fasta_path,
        "count": len(results),
        "results": results,
    }



def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        if isinstance(value, float):
            return int(round(value))
        text = str(value).strip()
        if not text:
            return None
        if "." in text or "e" in text.lower():
            return int(round(float(text)))
        return int(text)
    except Exception:
        return None


def _prune_isoform_transcript(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a minimal isoform transcript record sufficient for splice strategy decisions."""
    kept: Dict[str, Any] = {}
    for key in ("transcript_id", "id", "strand", "chrom", "seq_region_name"):
        if key in entry:
            value = entry[key]
            if value is not None:
                kept[key] = value

    exons_src = entry.get("exons_tx")
    if isinstance(exons_src, list):
        pruned_exons: List[Dict[str, Any]] = []
        exon_keys = (
            "index",
            "start",
            "end",
            "seq_region_name",
            "phase",
            "end_phase",
            "cdna_start",
            "cdna_end",
            "coding_start",
            "coding_end",
            "coding_start_genomic",
            "coding_end_genomic",
            "contains_coding",
        )
        for exon in exons_src:
            if not isinstance(exon, Mapping):
                continue
            pruned: Dict[str, Any] = {}
            for key in exon_keys:
                if key in exon:
                    value = exon[key]
                    if value is not None:
                        pruned[key] = value
            if pruned:
                pruned_exons.append(pruned)
        if pruned_exons:
            kept["exons_tx"] = pruned_exons

    return kept


def build_splice_strategy_goal_payload(
    structured_goal: Any, *, transcript_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract the minimal structured goal fields needed by decide_splice_strategy.

    Only intent, patient_variant, exon_label, and the isoform entry for the selected transcript
    are retained to keep the payload compact for tool calls.
    """
    if not isinstance(structured_goal, Mapping):
        return {}

    payload: Dict[str, Any] = {}
    for key in ("intent", "patient_variant", "exon_label"):
        if key in structured_goal:
            value = structured_goal[key]
            if value is not None:
                payload[key] = value

    goal_isoforms = structured_goal.get("isoform_transcripts")
    if isinstance(goal_isoforms, Mapping):
        chosen_label: Optional[str] = None
        chosen_entry: Optional[Mapping[str, Any]] = None

        if transcript_id:
            for label, entry in goal_isoforms.items():
                if not isinstance(entry, Mapping):
                    continue
                tid = entry.get("transcript_id") or entry.get("id")
                if tid and tid == transcript_id:
                    chosen_label = str(label)
                    chosen_entry = entry
                    break
            if chosen_entry is None:
                maybe_entry = goal_isoforms.get(transcript_id)
                if isinstance(maybe_entry, Mapping):
                    chosen_label = str(transcript_id)
                    chosen_entry = maybe_entry

        if chosen_entry is None:
            for label, entry in goal_isoforms.items():
                if isinstance(entry, Mapping):
                    chosen_label = str(label)
                    chosen_entry = entry
                    break

        if chosen_entry is not None and chosen_label is not None:
            payload["isoform_transcripts"] = {
                chosen_label: _prune_isoform_transcript(chosen_entry)
            }

    return payload


def _log_progress(msg: str, *, stdout: bool = True) -> None:
    if stdout:
        try:
            print(f"[SpliceEval] {msg}")
        except Exception:
            pass
    if PROGRESS_CALLBACK:
        try:
            PROGRESS_CALLBACK(msg)
        except Exception:
            pass

def _log_skip(msg: str) -> None:
    try:
        print(f"[SpliceEval][skip] {msg}")
    except Exception:
        pass
    if PROGRESS_CALLBACK:
        try:
            PROGRESS_CALLBACK(f"SpliceEval skip: {msg}")
        except Exception:
            pass

def _normalize_strand_char(value: Any, transcript_strand_int: int) -> str:
    """Return '+' or '-' given a variety of strand encodings, defaulting to transcript strand."""
    default_char = '+' if int(transcript_strand_int) == 1 else '-'
    if value is None:
        return default_char
    # numeric case
    if isinstance(value, (int, float)):
        try:
            return '+' if int(value) >= 1 else '-'
        except Exception:
            return default_char
    # string case
    s = str(value).strip()
    if not s:
        return default_char
    if s in ('+', 'plus', '+1'):
        return '+'
    if s in ('-', 'minus', '-1'):
        return '-'
    # numeric string like '1' or '0' or '-1'
    try:
        v = int(s)
        return '+' if v >= 1 else '-'
    except Exception:
        # sign prefix fallback
        if s.startswith('+'):
            return '+'
        if s.startswith('-'):
            return '-'
        return default_char

def set_progress_callback(cb: Optional[Callable[[str], None]]) -> None:
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = cb


@function_tool
@splice_resource_limited
def evaluate_splice_shifts_on_tiles(
    tiles_json_path: str,
    transcript_id: str,
    structured_goal_json: str,
    ref_fasta_path: Optional[str] = None,
    pad: int = 50,
    motif: str = "AAA",
    splice_sites_json: Optional[str] = None,
    site_window_bp: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute delta-inclusion shifts (deletion and replacement) for ASO candidates in tiles.json
    using SpTransformerDriver, centred on user-supplied splice acceptor and donor sites.

    Inputs:
      - tiles_json_path: absolute path to tiles.json produced by tiling agent (collect_results_json)
      - transcript_id: Ensembl transcript stable ID
      - structured_goal_json: JSON string with at least { intent: 'splice_modulation', exon_label: int }
      - ref_fasta_path: absolute path to hg38 fasta (defaults to repo copy)
      - pad: default window radius (bp) when splice_sites_json is absent (default 50)
      - motif: replacement motif used for equal-length substitution (default 'AAA')
      - splice_sites_json: JSON payload describing splice acceptor/donor sites to score. Accepts either
        a list of site dicts or an object with a "sites" list. Each site should include "acceptor"
        and "donor" objects with genomic coordinates (position/start/end) and may optionally specify
        per-site window via "window_bp".
      - site_window_bp: override default window radius (bp) when per-site window metadata is missing

    Output JSON:
      {
        transcript_id,
        exon_index,
        acceptor_junction,
        donor_junction,
        focus_splice_site_label?,
        pad,
        site_window_bp,
        splice_sites,
        ref_fasta,
        count,
        results: [
          {
            name?, chrom, start, end, strand, length,
            delta_inclusion_deletion,
            delta_inclusion_replacement,
            primary_splice_site_label?,
            delta_inclusion_deletion_mean?,
            delta_inclusion_replacement_mean?,
            splice_site_deltas?: [
              {
                label, exon_index?, window_bp,
                acceptor: {chrom, start, end, position, window_bp, strand},
                donor:    {chrom, start, end, position, window_bp, strand},
                delta_inclusion_deletion?,
                delta_inclusion_replacement?,
              }
            ]
          }
        ]
      }
    """

    try:
        goal = json.loads(structured_goal_json) if structured_goal_json else {}
    except Exception as e:
        return {"error": f"structured_goal_json is not valid JSON: {e}"}

    backend_info = resolve_splice_eval_backend(goal.get("species"))
    if backend_info.get("error"):
        return {
            "error": "unsupported_splice_eval_species",
            "species": backend_info.get("species"),
            "message": "No configured splice-evaluation backend exists for this species.",
        }
    species_key = str(backend_info.get("species") or normalize_species(goal.get("species")))

    species_cfg = get_species_config(species_key)
    if not ref_fasta_path:
        ref_fasta_path = os.fspath(species_cfg.genome_fasta)

    exon_label = goal.get("exon_label")
    try:
        exon_index = int(str(exon_label)) if exon_label is not None else None
    except Exception:
        exon_index = None

    # Load transcript structure to get junctions (supports synthetic SnapGene transcripts)
    isoform_transcripts = goal.get("isoform_transcripts")
    snapgene_entry: Optional[Dict[str, Any]] = None
    if isinstance(isoform_transcripts, dict):
        for iso_label, details in isoform_transcripts.items():
            if not isinstance(details, dict):
                continue
            tid = details.get("transcript_id") or details.get("id")
            if tid == transcript_id:
                snapgene_entry = details
                break

    if snapgene_entry and (
        transcript_id.startswith("SNAPGENE::")
        or "snapgene" in str((snapgene_entry.get("metadata") or {}).get("source") or "").lower()
    ):
        exons_tx_entry = snapgene_entry.get("exons_tx") or []
        if not exons_tx_entry:
            return {"error": f"snapgene_transcript_missing_exons: {transcript_id}"}
        strand_raw = snapgene_entry.get("strand") or (snapgene_entry.get("metadata") or {}).get("strand") or "+"
        strand_char = str(strand_raw).strip()
        strand_val = 1 if strand_char in ("+", "+1", "1") else -1
        chrom_val = snapgene_entry.get("chrom") or (snapgene_entry.get("metadata") or {}).get("chrom")
        chrom_str = str(chrom_val) if chrom_val is not None else None
        seq_region_name = None
        if chrom_str:
            seq_region_name = chrom_str[3:] if chrom_str.lower().startswith("chr") else chrom_str
        cleaned_tx: List[Dict[str, Any]] = []
        cleaned_genomic: List[Dict[str, Any]] = []
        last_cdna_end = 0
        for idx_exon, exon in enumerate(exons_tx_entry, start=1):
            if not isinstance(exon, dict):
                continue
            start = _safe_int(exon.get("start"))
            end = _safe_int(exon.get("end"))
            if start is None or end is None:
                continue
            index = _safe_int(exon.get("index")) or idx_exon
            exon_seq_region = exon.get("seq_region_name") or seq_region_name
            contains_coding = exon.get("contains_coding")
            if contains_coding is None:
                contains_coding = True
            length = abs(end - start) + 1
            cdna_start = _safe_int(exon.get("cdna_start"))
            cdna_end = _safe_int(exon.get("cdna_end"))
            if cdna_start is None or cdna_start <= 0:
                cdna_start = last_cdna_end + 1 if last_cdna_end else 1
            if cdna_end is None or cdna_end < cdna_start:
                cdna_end = cdna_start + length - 1
            last_cdna_end = cdna_end
            cleaned_tx.append({
                "index": index,
                "start": start,
                "end": end,
                "seq_region_name": exon_seq_region,
                "phase": exon.get("phase"),
                "end_phase": exon.get("end_phase"),
                "cdna_start": cdna_start,
                "cdna_end": cdna_end,
                "coding_start": _safe_int(exon.get("coding_start")) or cdna_start,
                "coding_end": _safe_int(exon.get("coding_end")) or cdna_end,
                "coding_start_genomic": _safe_int(exon.get("coding_start_genomic")) or start,
                "coding_end_genomic": _safe_int(exon.get("coding_end_genomic")) or end,
                "contains_coding": bool(contains_coding),
            })
            cleaned_genomic.append({
                "start": start,
                "end": end,
                "seq_region_name": exon_seq_region,
            })
        if not cleaned_tx:
            return {"error": f"snapgene_transcript_missing_exons: {transcript_id}"}
        tx = {
            "id": transcript_id,
            "seq_region_name": seq_region_name or chrom_str,
            "strand": strand_val,
            "exons_tx": cleaned_tx,
            "exons_genomic": cleaned_genomic,
        }
    else:
        tx = get_transcript_structure(transcript_id, use_reference=ref_fasta_path)
        if tx.get("error"):
            return tx

    strand_val = int(tx.get("strand") or 1)
    ex_tx: List[Dict[str, int]] = tx.get("exons_tx") or []
    if ex_tx and exon_index is not None and not (1 <= exon_index <= len(ex_tx)):
        return {"error": f"exon_label {exon_index} out of range (1..{len(ex_tx)})"}

    # Helper: compute exon junctions for a given exon dict
    def exon_junctions_for(exon_dict: Dict[str, Any]) -> tuple:
        if strand_val == 1:
            return int(exon_dict["start"]), int(exon_dict["end"])  # (acceptor, donor)
        else:
            return int(exon_dict["end"]), int(exon_dict["start"])  # (acceptor, donor)

    # If a fixed exon is specified, precompute its junctions; else will pick per-row
    fixed_acc = None
    fixed_don = None
    if exon_index is not None and 1 <= exon_index <= len(ex_tx):
        _ex = ex_tx[exon_index - 1]
        fixed_acc, fixed_don = exon_junctions_for(_ex)

    default_site_window = max(1, int(site_window_bp if site_window_bp is not None else pad or 50))

    parsed_sites: List[Dict[str, Any]] = []
    if splice_sites_json:
        try:
            sites_payload = json.loads(splice_sites_json)
        except Exception as e:
            return {"error": f"splice_sites_json is not valid JSON: {e}"}

        def _as_site_list(payload: Any) -> List[Dict[str, Any]]:
            if isinstance(payload, list):
                return [p for p in payload if isinstance(p, dict)]
            if isinstance(payload, dict):
                if isinstance(payload.get("sites"), list):
                    return [p for p in payload["sites"] if isinstance(p, dict)]
                return [payload]
            return []

        parsed_sites = _as_site_list(sites_payload)

    tx_chrom = tx.get("seq_region_name")
    strand_char_tx = '+' if strand_val == 1 else '-'

    def _extract_window(info: Dict[str, Any]) -> Optional[int]:
        for key in ("window_bp", "window", "half_window_bp", "pad", "padding"):
            if key in info:
                win = _safe_int(info.get(key))
                if win and win > 0:
                    return win
        return None

    def _extract_chrom(info: Dict[str, Any]) -> Optional[str]:
        if not isinstance(info, dict):
            return None
        for key in ("chrom", "chr", "chromosome", "seq_region_name"):
            val = info.get(key)
            if val:
                return str(val)
        return None

    def _extract_position(info: Dict[str, Any]) -> Optional[int]:
        if not isinstance(info, dict):
            return None
        for key in ("position", "pos", "coordinate", "bp"):
            if key in info:
                c = _safe_int(info.get(key))
                if c is not None:
                    return c
        start_v = _safe_int(info.get("start"))
        end_v = _safe_int(info.get("end"))
        if start_v is not None and end_v is not None:
            return int(round((start_v + end_v) / 2))
        if start_v is not None:
            return start_v
        if end_v is not None:
            return end_v
        return None

    normalized_sites: List[Dict[str, Any]] = []
    for idx_site, site in enumerate(parsed_sites, start=1):
        if not isinstance(site, dict):
            continue
        acceptor_raw = site.get("acceptor") or site.get("acceptor_site")
        donor_raw = site.get("donor") or site.get("donor_site")
        if not (isinstance(acceptor_raw, dict) and isinstance(donor_raw, dict)):
            continue
        acc_pos = _extract_position(acceptor_raw)
        don_pos = _extract_position(donor_raw)
        if acc_pos is None or don_pos is None:
            continue
        win_acc = _extract_window(acceptor_raw)
        win_don = _extract_window(donor_raw)
        combined_window = max(
            [w for w in (win_acc, win_don, default_site_window) if w is not None]
        ) if any(x is not None for x in (win_acc, win_don)) else default_site_window
        chrom_acc = _extract_chrom(acceptor_raw) or _extract_chrom(donor_raw) or tx_chrom
        chrom_don = _extract_chrom(donor_raw) or _extract_chrom(acceptor_raw) or tx_chrom

        from .explicit_target_identity import strand_symbol, chromosome
        if any(strand_symbol(point.get("strand")) not in (None, strand_char_tx)
               for point in (acceptor_raw, donor_raw)):
            raise ValueError("Explicit splice-site strand conflicts with the evaluated transcript; resolve the target before computing predictions.")
        if ((strand_char_tx == "+" and acc_pos > don_pos)
                or (strand_char_tx == "-" and acc_pos < don_pos)):
            raise ValueError("Explicit splice acceptor/donor order conflicts with the evaluated transcript strand.")
        if tx_chrom and any(chromosome(value) != chromosome(tx_chrom) for value in (chrom_acc, chrom_don)):
            raise ValueError("Explicit splice-site chromosome conflicts with the evaluated transcript.")

        def _window_obj(position: int, win_override: Optional[int], chrom_val: Optional[str]) -> Dict[str, Any]:
            win = win_override if (win_override and win_override > 0) else combined_window
            left = max(1, position - win)
            right = position + win
            return {
                "chrom": chrom_val or tx_chrom,
                "position": position,
                "start": left,
                "end": right,
                "window_bp": win,
                "strand": strand_char_tx,
            }

        exon_idx_norm = _safe_int(site.get("exon_index"))
        label_norm = site.get("label")
        if not label_norm and exon_idx_norm is not None:
            label_norm = f"exon_{exon_idx_norm}"
        if not label_norm:
            label_norm = f"site_{idx_site}"

        canonical_label = site.get("canonical_label")
        fallback_display = site.get("fallback_display_label")
        display_label_value = site.get("display_label") or canonical_label or label_norm

        normalized_sites.append({
            "exon_index": exon_idx_norm,
            "label": label_norm,
            "display_label": display_label_value,
            "canonical_label": canonical_label,
            "fallback_display_label": fallback_display,
            "site_kind": site.get("site_kind"),
            "isoform_label": site.get("isoform_label"),
            "delta_direction": site.get("delta_direction"),
            "window_bp": combined_window,
            "uniqueness": site.get("uniqueness"),
            "covers_all_isoforms": bool(site.get("covers_all_isoforms")),
            "present_in": site.get("present_in"),
            "absent_in": site.get("absent_in"),
            "acceptor": _window_obj(acc_pos, win_acc, chrom_acc),
            "donor": _window_obj(don_pos, win_don, chrom_don),
        })

    if not normalized_sites and fixed_acc is not None and fixed_don is not None:
        # Fallback to the focused exon when no explicit site list supplied
        chrom_fallback = tx_chrom
        fallback_window = default_site_window

        def _fallback_window_obj(position: int) -> Dict[str, Any]:
            left = max(1, position - fallback_window)
            right = position + fallback_window
            return {
                "chrom": chrom_fallback,
                "position": position,
                "start": left,
                "end": right,
                "window_bp": fallback_window,
                "strand": strand_char_tx,
            }

        normalized_sites.append({
            "exon_index": exon_index,
            "label": f"exon_{exon_index}" if exon_index is not None else "site_1",
            "display_label": f"exon_{exon_index}" if exon_index is not None else "site_1",
            "site_kind": None,
            "isoform_label": None,
            "delta_direction": None,
            "window_bp": fallback_window,
            "uniqueness": None,
            "covers_all_isoforms": False,
            "present_in": None,
            "absent_in": None,
            "acceptor": _fallback_window_obj(fixed_acc),
            "donor": _fallback_window_obj(fixed_don),
        })

    site_positions: List[int] = []
    site_window_max = default_site_window
    for site in normalized_sites:
        try:
            acc_pos_val = int(site["acceptor"]["position"])
            site_positions.append(acc_pos_val)
        except Exception:
            pass
        try:
            don_pos_val = int(site["donor"]["position"])
            site_positions.append(don_pos_val)
        except Exception:
            pass
        try:
            site_window_max = max(site_window_max, int(site.get("window_bp") or default_site_window))
        except Exception:
            pass

    if normalized_sites and not site_positions:
        return {"error": "No valid splice site coordinates parsed from splice_sites_json."}

    focus_site_global: Optional[Dict[str, Any]] = None
    if normalized_sites:
        if exon_index is not None:
            for site in normalized_sites:
                if site.get("exon_index") == exon_index:
                    focus_site_global = site
                    break
        if focus_site_global is None:
            focus_site_global = normalized_sites[0]

    if not tiles_json_path or not os.path.exists(tiles_json_path):
        return {"error": f"tiles_json_path not found: {tiles_json_path}"}

    try:
        with open(tiles_json_path, "r", encoding="utf-8") as f:
            tiles = json.load(f)
    except Exception as e:
        return {"error": f"Failed to read tiles JSON: {e}"}

    rows = tiles.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return {"error": "tiles.json contains no rows"}

    if backend_info.get("backend") == "openspliceai":
        openspliceai_result = _evaluate_with_openspliceai(
            rows=rows,
            tx=tx,
            normalized_sites=normalized_sites,
            site_positions=site_positions,
            site_window_max=site_window_max,
            focus_site_global=focus_site_global,
            default_site_window=default_site_window,
            fixed_acc=fixed_acc,
            fixed_don=fixed_don,
            motif=motif,
            strand_val=strand_val,
            strand_char_tx=strand_char_tx,
            ex_g=tx.get("exons_genomic") or [],
            ref_fasta_path=ref_fasta_path,
            exon_index=exon_index,
            species_key=species_key,
            model_path=str(backend_info.get("model") or ""),
        )
        if isinstance(openspliceai_result, dict) and not openspliceai_result.get("error"):
            openspliceai_result["splice_eval_backend"] = backend_info.get("backend")
            openspliceai_result["splice_eval_species"] = species_key
            openspliceai_result["splice_eval_model"] = backend_info.get("model")
        return openspliceai_result

    if backend_info.get("backend") != "splicetransformer":
        return {
            "error": "unsupported_splice_eval_backend",
            "species": species_key,
            "backend": backend_info.get("backend"),
        }

    total_rows = len(rows)
    _log_progress(f"SpliceTransformer: evaluating {total_rows} ASOs...")

    # Column mapping helper (case-insensitive, allow synonyms)
    def map_col(row_keys: List[str], candidates: List[str]) -> Optional[str]:
        lower = {k.lower(): k for k in row_keys}
        for name in candidates:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    if os.environ.get("ASO_BENCHMARK_SERVICE_URL"):
        # Benchmark workers have no GPU devices. Both platforms use the same
        # host FIFO service; native preprocessing and score calculations stay here.
        from .benchmark.client import RemoteSpliceDriver, SNPIntervalMutated
        sp = RemoteSpliceDriver("hg38", load_db=False)
    else:
        # Ensure SpliceTransformer can be imported and its relative weight paths resolve.
        st_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spliceAI_integration", "SpliceTransformer"))
        tasks_path = os.path.join(st_root, "tasks_annotate_mutations.py")
        if not os.path.exists(tasks_path):
            return {
                "error": (
                    "SpliceTransformer submodule is not initialized. Run: "
                    "git submodule update --init --recursive spliceAI_integration/SpliceTransformer"
                )
            }
        checkpoint_path = os.path.join(st_root, "model", "weights", "SpTransformer_pytorch.ckpt")
        if not os.path.exists(checkpoint_path):
            return {
                "error": (
                    f"SpliceTransformer checkpoint not found: {checkpoint_path}. "
                    f"Download the weights from {SPLICETRANSFORMER_WEIGHTS_URL} and place them at that path."
                )
            }

        import sys as _sys
        if st_root not in _sys.path:
            _sys.path.insert(0, st_root)
        old_cwd = os.getcwd()
        sp = None
        try:
            os.chdir(st_root)
            from tasks_annotate_mutations import SpTransformerDriver, SNPIntervalMutated  # type: ignore

            if not ref_fasta_path:
                ref_fasta_path = os.path.join(st_root, "data", "data_package", "hg38.fa")
            if not os.path.exists(ref_fasta_path):
                return {
                    "error": (
                        f"Reference FASTA not found: {ref_fasta_path}. "
                        f"Download hg38.fa.gz from {SPLICETRANSFORMER_HG38_FASTA_URL}, decompress it, "
                        "and place it at that path."
                    )
                }

            import torch as _torch

            original_torch_load = _torch.load

            def _load_splicetransformer_checkpoint(*args: Any, **kwargs: Any) -> Any:
                payload = original_torch_load(*args, **kwargs)
                if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
                    return payload
                state_dict = payload["state_dict"]
                normalized_state = {}
                changed = False
                for key, value in state_dict.items():
                    normalized_key = re.sub(r"\.pos_emb\.weights_(\d+)", r".pos_emb.weights.\1", str(key))
                    if normalized_key != key:
                        changed = True
                    normalized_state[normalized_key] = value
                if changed:
                    payload = dict(payload)
                    payload["state_dict"] = normalized_state
                return payload

            _torch.load = _load_splicetransformer_checkpoint
            try:
                sp = SpTransformerDriver(ref_fasta="hg38", load_db=False)
            finally:
                _torch.load = original_torch_load
        except Exception as exc:
            return {"error": f"SpliceTransformer initialization failed: {exc}"}
        finally:
            os.chdir(old_cwd)

        if sp is None:
            return {"error": "SpliceTransformer initialization failed: unknown_error"}
    try:
        sp.ref_fasta = Fasta(ref_fasta_path)
    except Exception as exc:
        return {"error": f"Reference FASTA load failed: {exc}"}

    def roi_rel_index(genomic_pos: int, left: int, right: int, strand_char: str) -> int:
        return int(genomic_pos - left) if strand_char == '+' else int(right - genomic_pos)

    def _array_window_max(arr, idx: int, w: int = default_site_window, col: int = 1) -> float:
        lo = max(0, idx - w)
        hi = min(arr.shape[0], idx + w + 1)
        if lo >= hi:
            return 0.0
        try:
            return float(arr[lo:hi, col].max())
        except Exception:
            return 0.0

    # Progress bar (fallback to plain iterator if tqdm unavailable)
    try:
        from tqdm import tqdm as _tqdm  # type: ignore
    except Exception:
        def _tqdm(x, **kwargs):
            return x

    # Build fast structures for per-row exon lookup
    # exons_genomic order and intervals
    ex_g: List[Dict[str, int]] = tx.get("exons_genomic") or []
    results: List[Dict[str, Any]] = []
    evaluated_count = 0
    skipped_count = 0
    for idx, row in enumerate(_tqdm(rows, desc="SpliceTransformer: evaluating ASOs", unit="ASO"), start=1):
        if idx == 1 or idx % 10 == 0 or idx == total_rows:
            _log_progress(f"SpliceTransformer progress: {idx}/{total_rows} ASOs", stdout=False)

        if not isinstance(row, dict):
            _log_skip("row is not a dict; skipping")
            continue

        keys = list(row.keys())
        col_chrom = map_col(keys, ["Chrom", "chrom", "chromosome", "seq_region_name"])
        col_start = map_col(keys, ["Start", "start", "chromstart"])  # likely 0-based
        col_end = map_col(keys, ["End", "end", "chromend"])          # likely 1-based
        col_strand = map_col(keys, ["Strand", "strand"])
        col_seq = map_col(keys, ["Sequence", "sequence", "seq", "Seq", "Oligo", "ASO"])  # try common names
        col_name = map_col(keys, ["name", "Name", "ID", "id", "label"])  # optional

        if not (col_chrom and col_start and col_end and col_strand and col_seq):
            _log_skip("required columns missing; skipping row")
            continue

        chrom = str(row.get(col_chrom))
        start_val = row.get(col_start)
        end_val = row.get(col_end)
        strand_val_aso = row.get(col_strand)
        seq = str(row.get(col_seq) or "")

        if not seq:
            _log_skip("sequence column empty; skipping row")
            continue

        try:
            start0 = int(start_val)
            end_bp = int(end_val)
        except Exception:
            _log_skip("start/end not integers; skipping row")
            continue

        length = len(seq)
        offset = start0 + 1
        strand_char = _normalize_strand_char(strand_val_aso, strand_val)
        mid = int((start0 + end_bp) // 2) + 1

        acc_j = None
        don_j = None
        target_positions: List[int] = []
        site_window_max_row = default_site_window

        local_sites: List[Dict[str, Any]] = []
        nearest_splice_site: Optional[Dict[str, Any]] = None
        if normalized_sites:
            for site in normalized_sites:
                if not isinstance(site, dict):
                    continue
                site_nearest = _nearest_splice_site_for_mid(mid, [site])
                if site_nearest is None:
                    continue
                if nearest_splice_site is None or int(site_nearest.get("distance_bp") or 0) < int(nearest_splice_site.get("distance_bp") or 0):
                    nearest_splice_site = site_nearest
                requested_nearest = _requested_splice_site_for_mid(mid, site)
                if (
                    requested_nearest is not None
                    and int(requested_nearest.get("distance_bp") or 0) <= SPLICE_EVAL_NEARBY_SITE_RADIUS_BP
                ):
                    local_sites.append(site)

        if normalized_sites:
            if not local_sites:
                skip_entry: Dict[str, Any] = {
                    "name": str(row.get(col_name)) if col_name else None,
                    "chrom": chrom,
                    "start": start0,
                    "end": end_bp,
                    "strand": strand_char,
                    "length": length,
                    "delta_inclusion_deletion": None,
                    "delta_inclusion_replacement": None,
                    "splice_eval_status": "skipped_no_nearby_splice_site",
                    "splice_eval_skip_reason": f"no_splice_sites_within_{SPLICE_EVAL_NEARBY_SITE_RADIUS_BP}bp",
                    "row_index": idx,
                    "aso_mid": mid,
                    "aso_start": offset,
                    "aso_end": offset + length - 1,
                }
                if nearest_splice_site:
                    skip_entry["nearest_splice_site_distance_bp"] = nearest_splice_site.get("distance_bp")
                    skip_entry["nearest_splice_site_kind"] = nearest_splice_site.get("site_kind")
                    skip_entry["nearest_splice_site_position"] = nearest_splice_site.get("position")
                    skip_entry["nearest_splice_site_label"] = nearest_splice_site.get("label")
                    skip_entry["nearest_splice_site_exon_index"] = nearest_splice_site.get("exon_index")
                    skip_entry["nearest_splice_site_isoform_label"] = nearest_splice_site.get("isoform_label")
                results.append(skip_entry)
                skipped_count += 1
                continue
            for site in local_sites:
                try:
                    acc_pos_site = int(site["acceptor"]["position"])
                    don_pos_site = int(site["donor"]["position"])
                except Exception:
                    continue
                site_kind = str(site.get("site_kind") or "").strip().lower()
                if site_kind == "acceptor":
                    target_positions.append(acc_pos_site)
                elif site_kind == "donor":
                    target_positions.append(don_pos_site)
                else:
                    target_positions.extend([acc_pos_site, don_pos_site])
                try:
                    site_window_max_row = max(site_window_max_row, int(site.get("window_bp") or default_site_window))
                except Exception:
                    pass
        else:
            if fixed_acc is not None and fixed_don is not None:
                acc_j, don_j = fixed_acc, fixed_don
            else:
                if not ex_g:
                    _log_skip("no exons_genomic data; skipping row")
                    continue
                chosen_exon = None
                min_dist = None
                for ex in ex_g:
                    ex_start = int(ex["start"])
                    ex_end = int(ex["end"])
                    if ex_start <= mid <= ex_end:
                        chosen_exon = ex
                        break
                    dist = abs(mid - ex_start) if mid < ex_start else abs(mid - ex_end)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        chosen_exon = ex
                if not chosen_exon:
                    _log_skip("no exon overlap for row; skipping")
                    continue
                acc_j, don_j = exon_junctions_for(chosen_exon)
            if acc_j is None or don_j is None:
                _log_skip("unable to determine splice junctions; skipping row")
                continue
            target_positions.extend([acc_j, don_j])

        if not target_positions:
            _log_skip("no splice site positions available; skipping row")
            continue

        target_positions.extend([offset, offset + length - 1])

        # Reference sequence for the ASO span
        try:
            ref_seq = sp.ref_fasta.get_seq(chrom, offset, offset + length - 1).seq.upper()
        except Exception:
            alt_chrom = chrom if chrom.startswith('chr') else f"chr{chrom}"
            try:
                ref_seq = sp.ref_fasta.get_seq(alt_chrom, offset, offset + length - 1).seq.upper()
                chrom = alt_chrom
            except Exception:
                _log_skip(f"failed to pull reference sequence for {chrom}:{offset}-{offset+length-1}")
                continue

        roi_left_row = max(1, min(target_positions) - site_window_max_row)
        roi_right_row = max(roi_left_row + 1, max(target_positions) + site_window_max_row)

        # Evaluate deletion (remove ASO span) and replacement (motif repeat)
        alt_repeat = (motif * ((length + len(motif) - 1) // len(motif)))[:length]

        try:
            snp_del = SNPIntervalMutated(chrom, strand_char_tx, offset, ref_seq, '', roi_left_row, roi_right_row)
            pred_ref_del, pred_alt_del = sp.calc_snp_misaligned(snp_del, context_len=4000)
        except Exception as exc:
            raise RuntimeError(
                f"splice_transformer_calc_failed(deletion) row={idx} chrom={chrom} start={offset} end={offset + length - 1}: {exc}"
            ) from exc

        try:
            snp_rep = SNPIntervalMutated(chrom, strand_char_tx, offset, ref_seq, alt_repeat, roi_left_row, roi_right_row)
            pred_ref_rep, pred_alt_rep = sp.calc_snp_misaligned(snp_rep, context_len=4000)
        except Exception as exc:
            raise RuntimeError(
                f"splice_transformer_calc_failed(replacement) row={idx} chrom={chrom} start={offset} end={offset + length - 1}: {exc}"
            ) from exc

        def compute_delta_metrics(pred_ref, pred_alt, acc_pos: int, don_pos: int, window_radius: int) -> Dict[str, Optional[float]]:
            metrics: Dict[str, Optional[float]] = {
                "pair": None,
                "acceptor": None,
                "donor": None,
            }
            if pred_ref is None or pred_alt is None:
                return metrics
            try:
                acc_idx = roi_rel_index(acc_pos, roi_left_row, roi_right_row, strand_char_tx)
                don_idx = roi_rel_index(don_pos, roi_left_row, roi_right_row, strand_char_tx)
                acc_in_bounds = 0 <= acc_idx < pred_ref.shape[0]
                don_in_bounds = 0 <= don_idx < pred_ref.shape[0]
                if not (acc_in_bounds or don_in_bounds):
                    return metrics
                window_val = max(1, int(window_radius))
                acc_ref = acc_alt = don_ref = don_alt = None
                if acc_in_bounds:
                    acc_ref = _array_window_max(pred_ref, acc_idx, w=window_val, col=1)
                    acc_alt = _array_window_max(pred_alt, acc_idx, w=window_val, col=1)
                    metrics["acceptor"] = float(acc_alt - acc_ref)
                if don_in_bounds:
                    don_ref = _array_window_max(pred_ref, don_idx, w=window_val, col=2)
                    don_alt = _array_window_max(pred_alt, don_idx, w=window_val, col=2)
                    metrics["donor"] = float(don_alt - don_ref)
                if acc_ref is not None and acc_alt is not None and don_ref is not None and don_alt is not None:
                    I_ref = (acc_ref * don_ref) ** 0.5
                    I_alt = (acc_alt * don_alt) ** 0.5
                    metrics["pair"] = float(I_alt - I_ref)
            except Exception:
                return metrics
            return metrics

        def compute_delta_pair(pred_ref, pred_alt, acc_pos: int, don_pos: int, window_radius: int) -> Optional[float]:
            """Return the combined acceptor/donor delta (pair) for a splice site window."""
            metrics = compute_delta_metrics(pred_ref, pred_alt, acc_pos, don_pos, window_radius)
            return metrics.get("pair")

        site_deltas: List[Dict[str, Any]] = []
        evaluated_labels: set[str] = set()
        if local_sites:
            for site in local_sites:
                try:
                    acc_pos_site = int(site["acceptor"]["position"])
                    don_pos_site = int(site["donor"]["position"])
                except Exception:
                    continue
                window_radius = int(site.get("window_bp") or default_site_window)
                del_metrics = compute_delta_metrics(pred_ref_del, pred_alt_del, acc_pos_site, don_pos_site, window_radius)
                rep_metrics = compute_delta_metrics(pred_ref_rep, pred_alt_rep, acc_pos_site, don_pos_site, window_radius)
                label_val = site.get("label")
                evaluated_labels.add(label_val)
                site_deltas.append({
                    "label": label_val,
                    "display_label": site.get("display_label"),
                    "canonical_label": site.get("canonical_label"),
                    "fallback_display_label": site.get("fallback_display_label"),
                    "site_kind": site.get("site_kind"),
                    "exon_index": site.get("exon_index"),
                    "isoform_label": site.get("isoform_label"),
                    "delta_direction": site.get("delta_direction"),
                    "window_bp": window_radius,
                    "uniqueness": site.get("uniqueness"),
                    "covers_all_isoforms": site.get("covers_all_isoforms"),
                    "present_in": site.get("present_in"),
                    "absent_in": site.get("absent_in"),
                    "acceptor": dict(site.get("acceptor", {})),
                    "donor": dict(site.get("donor", {})),
                    "evaluation_status": "evaluated",
                    "delta_inclusion_deletion": del_metrics.get("pair"),
                    "delta_inclusion_replacement": rep_metrics.get("pair"),
                    "delta_acceptor_deletion": del_metrics.get("acceptor"),
                    "delta_acceptor_replacement": rep_metrics.get("acceptor"),
                    "delta_donor_deletion": del_metrics.get("donor"),
                    "delta_donor_replacement": rep_metrics.get("donor"),
                })

        if normalized_sites:
            for site in normalized_sites:
                label_val = site.get("label")
                if label_val in evaluated_labels:
                    continue
                site_deltas.append({
                    "label": label_val,
                    "display_label": site.get("display_label"),
                    "canonical_label": site.get("canonical_label"),
                    "fallback_display_label": site.get("fallback_display_label"),
                    "site_kind": site.get("site_kind"),
                    "exon_index": site.get("exon_index"),
                    "isoform_label": site.get("isoform_label"),
                    "delta_direction": site.get("delta_direction"),
                    "window_bp": site.get("window_bp"),
                    "uniqueness": site.get("uniqueness"),
                    "covers_all_isoforms": site.get("covers_all_isoforms"),
                    "present_in": site.get("present_in"),
                    "absent_in": site.get("absent_in"),
                    "acceptor": dict(site.get("acceptor", {})),
                    "donor": dict(site.get("donor", {})),
                    "evaluation_status": "outside_evaluated_window",
                    "delta_inclusion_deletion": None,
                    "delta_inclusion_replacement": None,
                    "delta_acceptor_deletion": None,
                    "delta_acceptor_replacement": None,
                    "delta_donor_deletion": None,
                    "delta_donor_replacement": None,
                })

        summary_site = None
        if site_deltas:
            if exon_index is not None:
                for sd in site_deltas:
                    if sd.get("exon_index") == exon_index:
                        summary_site = sd
                        break
            if summary_site is None:
                summary_site = site_deltas[0]

        if summary_site:
            delta_del = summary_site.get("delta_inclusion_deletion")
            delta_rep = summary_site.get("delta_inclusion_replacement")
        else:
            delta_del = compute_delta_pair(pred_ref_del, pred_alt_del, acc_j, don_j, default_site_window) if acc_j is not None and don_j is not None else None
            delta_rep = compute_delta_pair(pred_ref_rep, pred_alt_rep, acc_j, don_j, default_site_window) if acc_j is not None and don_j is not None else None

        def _mean(values) -> Optional[float]:
            vals = [float(v) for v in values if isinstance(v, (int, float))]
            if not vals:
                return None
            return float(sum(vals) / len(vals))

        result_entry = {
            "name": str(row.get(col_name)) if col_name else None,
            "chrom": chrom,
            "start": start0,
            "end": end_bp,
            "strand": strand_char,
            "length": length,
            "delta_inclusion_deletion": delta_del,
            "delta_inclusion_replacement": delta_rep,
        }
        if site_deltas:
            result_entry["splice_site_deltas"] = site_deltas
            result_entry["delta_inclusion_deletion_mean"] = _mean([sd.get("delta_inclusion_deletion") for sd in site_deltas])
            result_entry["delta_inclusion_replacement_mean"] = _mean([sd.get("delta_inclusion_replacement") for sd in site_deltas])
            result_entry["primary_splice_site_label"] = summary_site.get("label") if summary_site else None

        results.append(result_entry)
        evaluated_count += 1

    _log_progress("SpliceTransformer evaluation complete.")

    primary_acceptor = None
    primary_donor = None
    if focus_site_global:
        try:
            primary_acceptor = int(focus_site_global["acceptor"]["position"])
        except Exception:
            primary_acceptor = fixed_acc
        try:
            primary_donor = int(focus_site_global["donor"]["position"])
        except Exception:
            primary_donor = fixed_don
    else:
        primary_acceptor = fixed_acc
        primary_donor = fixed_don

    return {
        "transcript_id": transcript_id,
        "exon_index": exon_index,
        "acceptor_junction": primary_acceptor,
        "donor_junction": primary_donor,
        "focus_splice_site_label": focus_site_global.get("label") if focus_site_global else None,
        "pad": default_site_window,
        "site_window_bp": default_site_window,
        "splice_sites": normalized_sites,
        "ref_fasta": ref_fasta_path,
        "count": len(results),
        "evaluated_count": evaluated_count,
        "skipped_count": skipped_count,
        "splice_eval_backend": backend_info.get("backend"),
        "splice_eval_species": species_key,
        "splice_eval_model": backend_info.get("model"),
        "results": results,
    }


@function_tool
def decide_splice_strategy(structured_goal_json: str, transcript_id: str) -> Dict[str, Any]:
    """
    Decide whether to use exon junctions from the supplied transcript or run SpliceAI (deferred) if a patient mutation is provided.

    Inputs:
      - structured_goal_json: JSON string containing at least {
          intent: 'splice_modulation' | 'knockdown',
          exon_label?: '10',
          patient_variant?: {
            notation: string,   # original user text like 'chr17:7674235 G>A' or 'NM_000546.6:c.743G>A'
            kind: 'genomic' | 'cdna' | 'protein' | 'hgvs' | 'unknown',
            gene_symbol?: string,
            chr?: string,
            pos?: int,
            ref?: string,
            alt?: string
          }
        }
      - transcript_id: Ensembl transcript stable ID (e.g., ENST00000...)

    Behavior:
      - If intent != splice_modulation: return { strategy: 'exon_junctions', reason: 'knockdown uses exon tiling' }
      - If patient_variant is present: return { strategy: 'run_spliceai', reason: 'patient mutation provided' }
      - Else: return { strategy: 'exon_junctions' } and compute likely donor/acceptor sites
        from the provided transcript exon boundaries. If exon_label provided, focus on that exon
        and include donor/acceptor at its boundaries; otherwise include all exon junctions.

    Output JSON:
      {
        strategy: 'exon_junctions' | 'run_spliceai',
        reason: str,
        transcript_id: str,
        sites?: [ { exon_index: int, donor: {chrom,start,end}, acceptor: {chrom,start,end} } ],
        focus_exon?: int
      }
    """
    try:
        goal = json.loads(structured_goal_json)
    except Exception as e:
        return {"error": f"structured_goal_json is not valid JSON: {e}"}

    intent = (goal.get("intent") or "").lower()
    if intent != "splice_modulation":
        return {
            "strategy": "exon_junctions",
            "reason": "knockdown intent uses exon-only tiling; junction inference not needed",
            "transcript_id": transcript_id,
        }

    patient_variant = goal.get("patient_variant")
    if isinstance(patient_variant, dict) and patient_variant.get("notation"):
        return {
            "strategy": "run_spliceai",
            "reason": "patient mutation provided; delegate to SpliceAI (to be implemented)",
            "transcript_id": transcript_id,
            "patient_variant": patient_variant,
        }

    custom_tx_entry: Optional[Dict[str, Any]] = None
    isoform_transcripts = goal.get("isoform_transcripts") if isinstance(goal, dict) else None
    if isinstance(isoform_transcripts, dict):
        for details in isoform_transcripts.values():
            if not isinstance(details, dict):
                continue
            tid = details.get("transcript_id") or details.get("id")
            if tid and tid == transcript_id:
                custom_tx_entry = details
                break

    def _tx_from_isoform(details: Dict[str, Any]) -> Dict[str, Any]:
        strand_raw = details.get("strand")
        if isinstance(strand_raw, str):
            strand = -1 if strand_raw.strip() in {"-", "-1"} else 1
        elif isinstance(strand_raw, (int, float)):
            strand = -1 if int(strand_raw) < 0 else 1
        else:
            strand = 1

        seq_region_name = details.get("chrom") or details.get("seq_region_name")
        if seq_region_name is None:
            ex_src = details.get("exons_tx") or []
            if ex_src:
                seq_region_name = ex_src[0].get("seq_region_name")
        seq_region_name = str(seq_region_name) if seq_region_name is not None else None

        exons_tx_src = details.get("exons_tx") or []
        exons_tx: List[Dict[str, Any]] = []
        for exon in exons_tx_src:
            if not isinstance(exon, dict):
                continue
            try:
                idx = int(exon.get("index")) if exon.get("index") is not None else len(exons_tx) + 1
                start = int(exon.get("start"))
                end = int(exon.get("end"))
            except Exception as exc:
                raise RuntimeError(f"Invalid exon coordinates in patient transcript: {exon}") from exc
            exons_tx.append(
                {
                    "index": idx,
                    "start": start,
                    "end": end,
                    "seq_region_name": exon.get("seq_region_name") or seq_region_name,
                    "phase": exon.get("phase"),
                    "end_phase": exon.get("end_phase"),
                    "cdna_start": _safe_int(exon.get("cdna_start")),
                    "cdna_end": _safe_int(exon.get("cdna_end")),
                    "coding_start": _safe_int(exon.get("coding_start")),
                    "coding_end": _safe_int(exon.get("coding_end")),
                    "coding_start_genomic": _safe_int(exon.get("coding_start_genomic")),
                    "coding_end_genomic": _safe_int(exon.get("coding_end_genomic")),
                    "contains_coding": bool(exon.get("contains_coding")),
                }
            )

        exons_tx.sort(key=lambda ex: ex["index"])
        exons_genomic = [
            {
                "start": ex["start"],
                "end": ex["end"],
                "seq_region_name": ex.get("seq_region_name") or seq_region_name,
            }
            for ex in sorted(exons_tx, key=lambda ex: ex["start"])
        ]

        if not exons_tx:
            raise RuntimeError("Patient transcript missing exons")

        tx_start = min(ex["start"] for ex in exons_tx)
        tx_end = max(ex["end"] for ex in exons_tx)
        exon_count = len(exons_tx)
        if strand == 1:
            tx_to_genomic = {i: (i - 1) for i in range(1, exon_count + 1)}
        else:
            tx_to_genomic = {i: (exon_count - i) for i in range(1, exon_count + 1)}

        return {
            "id": transcript_id,
            "seq_region_name": seq_region_name,
            "strand": strand,
            "start": tx_start,
            "end": tx_end,
            "exon_count": exon_count,
            "exons_tx": exons_tx,
            "exons_genomic": exons_genomic,
            "tx_to_genomic": tx_to_genomic,
        }

    if custom_tx_entry:
        try:
            tx = _tx_from_isoform(custom_tx_entry)
        except Exception as exc:
            return {"error": f"patient_transcript_invalid: {exc}"}
    else:
        tx = get_transcript_structure(transcript_id)
    if tx.get("error"):
        return tx

    chrom = tx.get("seq_region_name")
    strand = int(tx.get("strand") or 1)
    ex_tx: List[Dict[str, int]] = tx.get("exons_tx") or []

    focus_exon = None
    exon_label = goal.get("exon_label")
    if exon_label is not None:
        try:
            ex_idx = int(str(exon_label))
            if 1 <= ex_idx <= len(ex_tx):
                focus_exon = ex_idx
        except Exception:
            pass

    sites: List[Dict[str, Any]] = []

    window_bp = 50

    def site_row(exon_index: int) -> Dict[str, Any]:
        ex = ex_tx[exon_index - 1]
        if strand == 1:
            donor_pos = int(ex["end"])
            acceptor_pos = int(ex["start"])
        else:
            donor_pos = int(ex["start"])
            acceptor_pos = int(ex["end"])

        def _window_obj(position: int) -> Dict[str, Any]:
            left = max(1, position - window_bp)
            right = position + window_bp
            return {
                "chrom": chrom,
                "position": position,
                "start": left,
                "end": right,
                "window_bp": window_bp,
                "strand": "+" if strand == 1 else "-",
            }

        return {
            "exon_index": exon_index,
            "label": f"exon_{exon_index}",
            "donor": _window_obj(donor_pos),
            "acceptor": _window_obj(acceptor_pos),
        }

    if focus_exon is not None:
        sites.append(site_row(focus_exon))
    else:
        for i in range(1, len(ex_tx) + 1):
            sites.append(site_row(i))

    return {
        "strategy": "exon_junctions",
        "reason": "no patient mutation provided; using transcript exon junctions as splice sites",
        "transcript_id": transcript_id,
        "sites": sites,
        "focus_exon": focus_exon,
    }


splicing_agent = Agent(
    name="ASO Splicing Agent",
    instructions=(
        "You can either (a) decide splice site coordinates from the transcript, or (b) evaluate ASO candidates using SpliceTransformer.\n"
        "When given tiles_json_path, transcript_id, and structured_goal_json for splice_modulation, call 'evaluate_splice_shifts_on_tiles' exactly once and return its JSON.\n"
        "Otherwise, call 'decide_splice_strategy' to return likely donor/acceptor coordinates.\n"
        "Return compact JSON only."
    ),
    tools=[evaluate_splice_shifts_on_tiles, decide_splice_strategy],
    model="gpt-5-mini-2025-08-07",
) 
