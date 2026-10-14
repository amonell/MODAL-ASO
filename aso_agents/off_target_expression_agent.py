import json
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from agents import Agent, Runner, function_tool
from .species_config import normalize_species


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _env_optional_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _env_optional_float(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    candidate = _coerce_int(value)
    if candidate is None:
        return None
    return candidate if candidate > 0 else None


def _coerce_positive_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except Exception:
            return None
        return number if number > 0 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except Exception:
        return None
    return number if number > 0 else None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


CELLXGENE_API_BASE = os.environ.get("CELLXGENE_API_BASE", "https://api.cellxgene.cziscience.com")
CELLXGENE_PAGE_SIZE = _env_int("CELLXGENE_PAGE_SIZE", 50)
CELLXGENE_TIMEOUT = _env_int("CELLXGENE_TIMEOUT", 60)
CELLXGENE_MAX_DATASET_CANDIDATES = _env_int("CELLXGENE_MAX_DATASET_CANDIDATES", 120)
from .paths import cache_root
CELLXGENE_CACHE_DIR = os.environ.get("CELLXGENE_CACHE_DIR", str(cache_root() / "cellxgene"))
CELLXGENE_MAX_CELLS = _env_optional_int("CELLXGENE_MAX_CELLS") or 1000
CELLXGENE_GENE_BATCH_SIZE = _env_optional_int("CELLXGENE_GENE_BATCH_SIZE") or 5000
CELLXGENE_SAMPLE_MODE = (os.environ.get("CELLXGENE_SAMPLE_MODE") or "first").strip().lower() or "first"
CELLXGENE_SAMPLE_SEED = _env_optional_int("CELLXGENE_SAMPLE_SEED")
CELLXGENE_MAX_H5AD_BYTES = _env_optional_int("CELLXGENE_MAX_H5AD_BYTES")
CELLXGENE_MAX_H5AD_GB = _env_optional_float("CELLXGENE_MAX_H5AD_GB")
CELLXGENE_TISSUE_CANDIDATE_LIMIT = _env_int("CELLXGENE_TISSUE_CANDIDATE_LIMIT", 80)
CELLXGENE_CELL_TYPE_CANDIDATE_LIMIT = _env_int("CELLXGENE_CELL_TYPE_CANDIDATE_LIMIT", 80)
DEFAULT_MAX_H5AD_BYTES = 500 * 1024 * 1024  # 500MB max for h5ad downloads


def _cellxgene_cache_path(cache_dir: str, context: Dict[str, Any]) -> Optional[Path]:
    cache_root = str(cache_dir or "").strip()
    if not cache_root:
        return None
    canonical = json.dumps(context, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Path(cache_root).expanduser() / "census_expression" / f"{digest}.json"


def _load_cellxgene_expression_cache(
    cache_dir: str,
    context: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    path = _cellxgene_cache_path(cache_dir, context)
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict) or payload.get("context") != context:
        return {}
    genes = payload.get("genes")
    if not isinstance(genes, dict):
        return {}
    return {
        str(gene): dict(stats)
        for gene, stats in genes.items()
        if isinstance(stats, dict)
    }


def _save_cellxgene_expression_cache(
    cache_dir: str,
    context: Dict[str, Any],
    genes: Dict[str, Dict[str, float]],
) -> Optional[str]:
    path = _cellxgene_cache_path(cache_dir, context)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_cellxgene_expression_cache(cache_dir, context)
    existing.update({str(gene): dict(stats) for gene, stats in genes.items() if isinstance(stats, dict)})
    payload = {"version": 1, "context": context, "genes": existing}
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return str(path)


def _resolve_cellxgene_census_version(cellxgene_census: Any) -> str:
    requested = str(os.environ.get("CELLXGENE_CENSUS_VERSION") or "stable").strip() or "stable"
    try:
        description = cellxgene_census.get_census_version_description(requested)
    except Exception:
        return requested
    if isinstance(description, dict):
        release_build = str(description.get("release_build") or "").strip()
        if release_build:
            return release_build
    return requested


def _cellxgene_selection_cache_path(cache_dir: str, context: Dict[str, Any]) -> Optional[Path]:
    cache_root = str(cache_dir or "").strip()
    if not cache_root:
        return None
    canonical = json.dumps(context, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Path(cache_root).expanduser() / "census_selection" / f"{digest}.json"


def _load_cellxgene_selection_cache(
    cache_dir: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    path = _cellxgene_selection_cache_path(cache_dir, context)
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict) or payload.get("context") != context:
        return {}
    selection = payload.get("selection")
    return dict(selection) if isinstance(selection, dict) else {}


def _save_cellxgene_selection_cache(
    cache_dir: str,
    context: Dict[str, Any],
    selection: Dict[str, Any],
) -> Optional[str]:
    path = _cellxgene_selection_cache_path(cache_dir, context)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "context": context, "selection": selection}
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return str(path)


# =============================================================================
# SOMA-based CellXGene Census functions (fast, no full AnnData loading)
# =============================================================================

def _soma_list_species(census) -> List[Dict[str, Any]]:
    """List available species with cell counts."""
    organisms = list(census["census_data"].keys())
    results = []
    for org in organisms:
        results.append({"organism": org})
    return results


def _soma_list_tissues(census, organism: str, limit: int = 30) -> List[Dict[str, int]]:
    """List tissues for an organism with cell counts (fast - just counts)."""
    exp = census["census_data"][organism]
    df = exp.obs.read(column_names=["tissue_general"]).concat().to_pandas()
    counts = df["tissue_general"].value_counts().head(limit)
    return [{"tissue": t, "cell_count": int(c)} for t, c in counts.items()]


def _soma_list_cell_types(census, organism: str, tissue: str, limit: int = 30) -> List[Dict[str, int]]:
    """List cell types within a tissue (fast - just counts)."""
    exp = census["census_data"][organism]
    df = exp.obs.read(
        value_filter=f"tissue_general == '{tissue}'",
        column_names=["cell_type"]
    ).concat().to_pandas()
    counts = df["cell_type"].value_counts().head(limit)
    return [{"cell_type": ct, "cell_count": int(c)} for ct, c in counts.items()]


def _soma_find_genes(census, organism: str, gene_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Find genes by Ensembl gene ID (feature_id), return soma_joinid for each."""
    exp = census["census_data"][organism]
    var = exp.ms["RNA"].var
    results: Dict[str, Dict[str, Any]] = {}

    tokens: List[str] = []
    token_candidates: Dict[str, List[str]] = {}
    seen_candidates: set[str] = set()
    for gene in gene_names:
        gene_clean = gene.strip()
        if not gene_clean:
            continue
        base_id = gene_clean.split(".", 1)[0]
        candidates = [gene_clean]
        if base_id != gene_clean:
            candidates.append(base_id)
        token_candidates[gene_clean] = candidates
        for cand in candidates:
            if cand and cand not in seen_candidates:
                seen_candidates.add(cand)
                tokens.append(cand)

    if not tokens:
        return {}

    def _escape_filter_value(value: str) -> str:
        return value.replace("'", "\\'")

    id_to_row: Dict[str, Any] = {}
    batch_size = CELLXGENE_GENE_BATCH_SIZE
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start:start + batch_size]
        values = ", ".join(f"'{_escape_filter_value(val)}'" for val in batch)
        filter_expr = f"feature_id in [{values}]"
        df = var.read(
            value_filter=filter_expr,
            column_names=["soma_joinid", "feature_id", "feature_name"],
        ).concat().to_pandas()
        if df.empty:
            continue
        for _, row in df.iterrows():
            feature_id = row.get("feature_id")
            if feature_id and feature_id not in id_to_row:
                id_to_row[str(feature_id)] = row

    for token, candidates in token_candidates.items():
        match_row = None
        for cand in candidates:
            match_row = id_to_row.get(cand)
            if match_row is not None:
                break
        if match_row is None:
            continue
        results[token] = {
            "soma_joinid": int(match_row["soma_joinid"]),
            "feature_id": match_row["feature_id"],
            "feature_name": match_row["feature_name"],
        }

    print(
        f"[cellxgene] gene lookup complete: {len(results)}/{len(token_candidates)} matched",
        flush=True,
    )
    return results


def _soma_get_expression(
    census,
    organism: str,
    tissue: str,
    cell_types: List[str],
    gene_info: Dict[str, Dict[str, Any]],
    max_cells: int = 5000,
    sample_seed: Optional[int] = None,
    sample_mode: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Fetch expression for genes using direct SOMA queries (no AnnData).
    Returns {gene_name: {mean_expression, fraction_expressing, n_cells}}.
    """
    import numpy as np

    if not gene_info:
        return {}

    exp = census["census_data"][organism]

    # Build cell type filter
    broad_tissue = str(tissue or "").strip().lower() in {"__all_tissues__", "all_tissues", "all tissues"}
    if broad_tissue and not cell_types:
        value_filter = ""
    elif cell_types:
        ct_clauses = " or ".join([f"cell_type == '{ct}'" for ct in cell_types])
        value_filter = f"tissue_general == '{tissue}' and ({ct_clauses})"
    else:
        value_filter = f"tissue_general == '{tissue}'"

    # Get cell IDs matching filter (just IDs, fast)
    if broad_tissue and not value_filter:
        n_cells = max_cells
        total_cells = max_cells
        cell_ids = list(range(max_cells))
        print(
            f"[cellxgene] querying broad baseline sample across {organism}: n_cells={max_cells}",
            flush=True,
        )
    else:
        print(f"[cellxgene] querying cells: {value_filter[:80]}...", flush=True)
        obs_start = time.time()
        read_kwargs: Dict[str, Any] = {"column_names": ["soma_joinid"]}
        if value_filter:
            read_kwargs["value_filter"] = value_filter
        obs_df = exp.obs.read(**read_kwargs).concat().to_pandas()
        obs_elapsed = time.time() - obs_start

        total_cells = len(obs_df)
        print(
            f"[cellxgene] found {total_cells} cells matching filter "
            f"(obs read {obs_elapsed:.1f}s)",
            flush=True,
        )

        if total_cells == 0:
            return {}

        # Sample cell IDs if needed
        if total_cells <= max_cells:
            cell_ids = obs_df["soma_joinid"].tolist()
            n_cells = total_cells
        else:
            mode = (sample_mode or "").strip().lower()
            use_first = mode in {"first", "head", "top"}
            if use_first:
                cell_ids = obs_df["soma_joinid"].to_numpy()[:max_cells].tolist()
                print(f"[cellxgene] sampling mode=first n_cells={max_cells}", flush=True)
            else:
                seed = sample_seed if sample_seed is not None else 42
                rng = np.random.default_rng(seed)
                cell_ids = sorted(
                    int(item)
                    for item in rng.choice(obs_df["soma_joinid"].to_numpy(), size=max_cells, replace=False).tolist()
                )
                print(f"[cellxgene] sampling mode=random seed={seed} n_cells={max_cells}", flush=True)
            n_cells = max_cells

    # Map soma_joinid to tokens and de-duplicate IDs for batch reads
    gid_to_tokens: Dict[int, List[str]] = {}
    for gene_name, info in gene_info.items():
        gid = int(info["soma_joinid"])
        gid_to_tokens.setdefault(gid, []).append(gene_name)
    gene_ids = sorted(gid_to_tokens.keys())

    # Fetch expression via SOMA (sparse COO format) in gene batches
    batch_size = CELLXGENE_GENE_BATCH_SIZE
    total_batches = (len(gene_ids) + batch_size - 1) // batch_size
    print(
        f"[cellxgene] fetching expression for {len(gene_ids)} genes in {total_batches} batches, {n_cells} cells...",
        flush=True,
    )
    X = exp.ms["RNA"].X["raw"]

    expr_sums: Dict[int, float] = {gid: 0.0 for gid in gene_ids}
    expr_counts: Dict[int, int] = {gid: 0 for gid in gene_ids}

    import pyarrow as pa

    for batch_idx, start in enumerate(range(0, len(gene_ids), batch_size), 1):
        batch = gene_ids[start:start + batch_size]
        batch_start = time.time()
        print(f"[cellxgene] reading batch {batch_idx}/{total_batches} ({len(batch)} genes)", flush=True)
        expr_iter = X.read(coords=(cell_ids, batch)).tables()
        chunks = list(expr_iter)
        batch_elapsed = time.time() - batch_start
        print(
            f"[cellxgene] batch {batch_idx}/{total_batches} read in {batch_elapsed:.1f}s "
            f"(chunks={len(chunks)})",
            flush=True,
        )
        if not chunks:
            continue
        expr_table = pa.concat_tables(chunks).to_pandas()
        if expr_table.empty:
            continue
        grouped = expr_table.groupby("soma_dim_1")["soma_data"].agg(["sum", "count"])
        for gid, row in grouped.iterrows():
            gid_int = int(gid)
            expr_sums[gid_int] += float(row["sum"])
            expr_counts[gid_int] += int(row["count"])

    # Calculate stats per gene token (sparse format: only non-zero values stored)
    results = {}
    for gene_name, info in gene_info.items():
        gid = int(info["soma_joinid"])
        expr_sum = expr_sums.get(gid, 0.0)
        n_expressing = expr_counts.get(gid, 0)
        mean_expr = expr_sum / n_cells if n_cells > 0 else 0
        frac_expr = n_expressing / n_cells if n_cells > 0 else 0

        results[gene_name] = {
            "mean_expression": float(mean_expr),
            "fraction_expressing": float(frac_expr),
            "n_cells": n_cells,
            "total_cells": total_cells,
            "feature_id": info["feature_id"],
            "feature_name": info["feature_name"],
        }

    print(f"[cellxgene] expression retrieved for {len(results)} genes", flush=True)
    return results


# Agent for selecting species/tissue/cell types
_census_selector_agent = Agent(
    name="CellXGene Census Selector",
    instructions=(
        "You select the best species, tissue, and cell types from CellXGene Census for expression analysis.\n"
        "Input JSON: {target_context, species_hint, available_species, available_tissues?, available_cell_types?}\n"
        "Rules:\n"
        "- First select the organism matching species_hint (e.g., 'human' -> 'homo_sapiens', 'mouse' -> 'mus_musculus')\n"
        "- Select the tissue most relevant to target_context (e.g., 'neurons' -> 'brain', 'liver cells' -> 'liver')\n"
        "- Select up to 3 cell types most relevant to target_context (e.g., 'motor neurons' -> ['neuron', 'glutamatergic neuron'])\n"
        "- If target_context mentions a specific cell type, prioritize that\n"
        "- Choose only from the provided available_tissues and available_cell_types lists; do not invent labels\n"
        "- Always return a tissue and at least one cell type, even if the match is imperfect\n"
        "- If preferred_tissue or preferred_cell_type is provided and present in the available lists, favor it\n"
        "- If error_hint is provided, correct the output accordingly\n"
        "Output JSON: {\"organism\": \"...\", \"tissue\": \"...\", \"cell_types\": [\"...\", ...]}"
    ),
    tools=[],
    model="gpt-5-mini-2025-08-07",
)


_dataset_selector_agent = Agent(
    name="Cellxgene Dataset Selector",
    instructions=(
        "You choose the single most relevant Cellxgene dataset for the user's context.\n"
        "Input JSON: {target_context, gene_symbol, species, datasets:[{dataset_id, dataset_title, collection_name, organisms, description, cell_count?, asset_filesize_bytes?}]}.\n"
        "Rules:\n"
        "- Select exactly one dataset_id from the provided list.\n"
        "- Prefer datasets whose title/collection/description best match the target_context and gene_symbol.\n"
        "- The dataset list already includes downloadable h5ad assets.\n"
        "- When multiple datasets are comparably relevant, prefer smaller asset_filesize_bytes.\n"
        "- If multiple are close, pick the most specific or disease-relevant dataset.\n"
        "Output JSON only: {\"dataset_id\": \"<ID FROM LIST>\"}."
    ),
    tools=[],
    model="gpt-5-mini-2025-08-07",
)


def _cxg_get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = CELLXGENE_TIMEOUT,
    retries: int = 3,
    backoff_s: float = 1.0,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    requester = session or requests
    for attempt in range(1, retries + 1):
        try:
            response = requester.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            return {"data": payload}
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_s * attempt)
                continue
            raise
    if last_error:
        raise last_error
    return {}


def _payload_to_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("collections", "data", "datasets"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _fetch_collections(
    api_base: str,
    *,
    page_size: int = CELLXGENE_PAGE_SIZE,
    timeout: int = CELLXGENE_TIMEOUT,
    session: Optional[requests.Session] = None,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    collections: List[Dict[str, Any]] = []
    seen: set[str] = set()
    page = 0
    while True:
        if max_pages is not None and page >= max_pages:
            break
        params = {"limit": page_size, "page": page}
        payload = _cxg_get_json(
            f"{api_base}/curation/v1/collections",
            params=params,
            timeout=timeout,
            session=session,
        )
        batch = _payload_to_list(payload)
        if not batch:
            break
        new_items = 0
        for entry in batch:
            collection_id = entry.get("collection_id") or entry.get("id")
            if collection_id and str(collection_id) in seen:
                continue
            if collection_id:
                seen.add(str(collection_id))
            collections.append(entry)
            new_items += 1
        if new_items == 0:
            break
        page += 1
    return collections


def _fetch_collection_detail(
    api_base: str,
    collection_id: str,
    *,
    timeout: int = CELLXGENE_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    try:
        return _cxg_get_json(
            f"{api_base}/curation/v1/collections/{collection_id}",
            timeout=timeout,
            session=session,
        )
    except requests.RequestException:
        return {}


def _extract_organism_labels(dataset_entry: Dict[str, Any]) -> List[str]:
    organisms = dataset_entry.get("organism") or dataset_entry.get("organisms") or []
    labels: List[str] = []
    for org in organisms:
        if isinstance(org, dict):
            label = org.get("label")
            if label:
                labels.append(str(label))
        elif isinstance(org, str):
            labels.append(org)
    return labels


def _extract_dataset_id(dataset_entry: Dict[str, Any]) -> Optional[str]:
    dataset_id = dataset_entry.get("dataset_id") or dataset_entry.get("id") or dataset_entry.get("datasetId")
    return str(dataset_id) if dataset_id else None


def _extract_dataset_title(dataset_entry: Dict[str, Any]) -> str:
    return str(dataset_entry.get("title") or dataset_entry.get("name") or "")


def _stringify_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value if str(item).strip()]
        return "; ".join(items)
    if isinstance(value, dict):
        return "; ".join(f"{k}:{v}" for k, v in value.items() if str(v).strip())
    return str(value)


def _build_dataset_description(
    dataset_entry: Dict[str, Any],
    collection_entry: Optional[Dict[str, Any]] = None,
    *,
    fallback: str = "",
) -> str:
    parts: List[str] = []
    for key in ("tissue", "disease", "cell_type", "assay", "development_stage", "suspension_type"):
        value = dataset_entry.get(key)
        if value:
            parts.append(f"{key}: {_stringify_field(value)}")
    if dataset_entry.get("description"):
        parts.append(str(dataset_entry.get("description")))
    if dataset_entry.get("summary"):
        parts.append(str(dataset_entry.get("summary")))
    if collection_entry:
        collection_desc = collection_entry.get("description")
        if collection_desc:
            parts.append(f"collection_desc: {collection_desc}")
    if fallback:
        parts.append(fallback)
    text = "; ".join(part for part in parts if part)
    return text.strip("; ")


def _build_dataset_catalog(collections: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    for collection in collections:
        collection_id = collection.get("collection_id") or collection.get("id")
        collection_name = collection.get("name") or collection.get("title") or ""
        collection_description = collection.get("description") or ""
        for dataset in collection.get("datasets") or []:
            if not isinstance(dataset, dict):
                continue
            dataset_id = _extract_dataset_id(dataset)
            if not dataset_id:
                continue
            record = {
                "dataset_id": dataset_id,
                "dataset_title": _extract_dataset_title(dataset),
                "collection_id": str(collection_id) if collection_id else "",
                "collection_name": str(collection_name) if collection_name else "",
                "collection_description": str(collection_description) if collection_description else "",
                "organisms": _extract_organism_labels(dataset),
                "cell_count": dataset.get("cell_count"),
                "description": _build_dataset_description(
                    dataset,
                    collection_entry=collection,
                    fallback=_stringify_field(
                        dataset.get("tissue")
                        or dataset.get("disease")
                        or dataset.get("assay")
                    ),
                ),
            }
            datasets.append(record)
    return datasets


def _species_labels_from_datasets(datasets: Iterable[Dict[str, Any]]) -> List[str]:
    labels: set[str] = set()
    for dataset in datasets:
        for label in dataset.get("organisms") or []:
            text = str(label).strip()
            if text:
                labels.add(text)
    return sorted(labels, key=str.lower)


_SPECIES_LABELS: Dict[str, List[str]] = {
    "human": ["homo sapiens", "human"],
    "mouse": ["mus musculus", "mouse"],
}


def _candidate_species_labels(species_value: Optional[str]) -> List[str]:
    if not species_value:
        return []
    species_key = normalize_species(species_value)
    labels = _SPECIES_LABELS.get(species_key, [])
    if labels:
        return labels
    text = str(species_value).strip()
    return [text.lower()] if text else []


def _dataset_matches_species(dataset: Dict[str, Any], labels: Sequence[str]) -> bool:
    if not labels:
        return True
    dataset_labels = [str(lbl).lower() for lbl in dataset.get("organisms") or []]
    for label in labels:
        if label.lower() in dataset_labels:
            return True
    return False


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _rank_datasets(
    datasets: List[Dict[str, Any]],
    *,
    target_context: str,
    gene_symbol: str,
) -> List[Dict[str, Any]]:
    tokens = _tokenize(f"{target_context} {gene_symbol}")
    if not tokens:
        return list(datasets)
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for dataset in datasets:
        title = (dataset.get("dataset_title") or "").lower()
        haystack = " ".join(
            filter(
                None,
                [
                    title,
                    str(dataset.get("collection_name") or "").lower(),
                    str(dataset.get("description") or "").lower(),
                    str(dataset.get("collection_description") or "").lower(),
                ],
            )
        )
        score = sum(1 for token in tokens if token in haystack)
        score += 2 * sum(1 for token in tokens if token in title)
        scored.append((score, dataset))
    scored.sort(key=lambda item: (-item[0], item[1].get("dataset_title") or "", item[1].get("dataset_id") or ""))
    return [item[1] for item in scored]


def _select_dataset_with_llm(
    target_context: str,
    gene_symbol: str,
    species_value: Optional[str],
    datasets: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not datasets:
        return None
    payload = {
        "target_context": target_context or "",
        "gene_symbol": gene_symbol or "",
        "species": species_value or "",
        "datasets": [
            {
                "dataset_id": dataset.get("dataset_id"),
                "dataset_title": dataset.get("dataset_title"),
                "collection_name": dataset.get("collection_name"),
                "organisms": dataset.get("organisms"),
                "description": dataset.get("description"),
                "cell_count": dataset.get("cell_count"),
                "asset_filesize_bytes": dataset.get("asset_filesize_bytes"),
            }
            for dataset in datasets
        ],
    }
    try:
        result = Runner._run_sync(
            _dataset_selector_agent,
            input=json.dumps(payload, ensure_ascii=False),
            max_steps=1,
        )
    except Exception:
        result = None

    valid_ids = {dataset.get("dataset_id") for dataset in datasets if dataset.get("dataset_id")}
    choice = result.final_output if result is not None else None
    selected_id: Optional[str] = None
    if isinstance(choice, str):
        try:
            obj = json.loads(choice)
            if isinstance(obj, dict):
                selected_id = obj.get("dataset_id")
        except Exception:
            for dataset_id in valid_ids:
                if dataset_id and dataset_id in choice:
                    selected_id = dataset_id
                    break
    elif isinstance(choice, dict):
        selected_id = choice.get("dataset_id")

    if selected_id and selected_id in valid_ids:
        for dataset in datasets:
            if dataset.get("dataset_id") == selected_id:
                return dataset
    return None


def _select_census_context_with_llm(
    *,
    target_context: str,
    species_hint: str,
    available_species: List[Dict[str, Any]],
    available_tissues: Optional[List[Dict[str, Any]]] = None,
    available_cell_types: Optional[List[Dict[str, Any]]] = None,
    preferred_tissue: Optional[str] = None,
    preferred_cell_type: Optional[str] = None,
    error_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    payload = {
        "target_context": target_context or "",
        "species_hint": species_hint or "",
        "available_species": available_species or [],
        "available_tissues": available_tissues or [],
        "available_cell_types": available_cell_types or [],
        "preferred_tissue": preferred_tissue or "",
        "preferred_cell_type": preferred_cell_type or "",
        "error_hint": error_hint or "",
    }
    try:
        result = Runner._run_sync(
            _census_selector_agent,
            input=json.dumps(payload, ensure_ascii=False),
            max_steps=1,
        )
    except Exception:
        result = None
    choice = result.final_output if result is not None else None
    if isinstance(choice, str):
        try:
            obj = json.loads(choice)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None
    if isinstance(choice, dict):
        return choice
    return None


def _normalize_choice(value: Any, lookup: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return lookup.get(text.lower())


def _normalize_choice_list(values: Any, lookup: Dict[str, str], limit: int = 3) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    selected: List[str] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        mapped = lookup.get(text.lower())
        if mapped and mapped not in selected:
            selected.append(mapped)
            if len(selected) >= limit:
                break
    return selected


def _normalized_label_tokens(value: Any) -> List[str]:
    tokens: List[str] = []
    for token in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if token in {"cell", "cells", "type", "line", "derived", "like", "human", "mouse", "in", "vitro"}:
            continue
        tokens.append(token)
        if len(token) > 3 and token.endswith("s"):
            tokens.append(token[:-1])
    return list(dict.fromkeys(tokens))


def _preferred_tissue_match(preferred_tissue: Optional[str], available_tissues: List[Dict[str, Any]]) -> Optional[str]:
    preferred = str(preferred_tissue or "").strip()
    if not preferred:
        return None
    exact_lookup = {
        str(item.get("tissue") or "").strip().lower(): str(item.get("tissue") or "").strip()
        for item in available_tissues
        if str(item.get("tissue") or "").strip()
    }
    exact = exact_lookup.get(preferred.lower())
    if exact:
        return exact
    preferred_tokens = _normalized_label_tokens(preferred)
    if not preferred_tokens:
        return None
    for item in available_tissues:
        tissue = str(item.get("tissue") or "").strip()
        tissue_tokens = set(_normalized_label_tokens(tissue))
        if tissue and all(token in tissue_tokens for token in preferred_tokens):
            return tissue
    return None


def _ranked_context_tissue_matches(
    *,
    target_context: str,
    available_tissues: List[Dict[str, Any]],
    limit: int = 3,
) -> List[str]:
    context_tokens = set(_normalized_label_tokens(target_context))
    if not context_tokens:
        return []
    ranked: List[Tuple[int, int, str]] = []
    for item in available_tissues:
        tissue = str(item.get("tissue") or "").strip()
        if not tissue:
            continue
        tissue_tokens = set(_normalized_label_tokens(tissue))
        overlap = context_tokens & tissue_tokens
        if not overlap:
            continue
        score = 10 * len(overlap) + int(item.get("cell_count") or 0)
        ranked.append((score, int(item.get("cell_count") or 0), tissue))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [tissue for _, _, tissue in ranked[: max(1, limit)]]


def _select_census_tissue_with_llm(
    *,
    target_context: str,
    species_hint: str,
    available_species: List[Dict[str, Any]],
    available_tissues: List[Dict[str, Any]],
    preferred_tissue: Optional[str] = None,
    max_attempts: int = 5,
) -> Optional[str]:
    tissue_lookup = {
        str(tissue.get("tissue")).lower(): str(tissue.get("tissue"))
        for tissue in available_tissues
        if tissue.get("tissue")
    }
    preferred_match = _preferred_tissue_match(preferred_tissue, available_tissues)
    if preferred_match:
        return preferred_match
    error_hint = None
    for _ in range(max_attempts):
        choice = _select_census_context_with_llm(
            target_context=target_context,
            species_hint=species_hint,
            available_species=available_species,
            available_tissues=available_tissues,
            preferred_tissue=preferred_tissue,
            error_hint=error_hint,
        )
        selected_tissue = _normalize_choice((choice or {}).get("tissue"), tissue_lookup)
        if selected_tissue:
            return selected_tissue
        error_hint = "Return a tissue from available_tissues; do not invent."
    context_matches = _ranked_context_tissue_matches(
        target_context=target_context,
        available_tissues=available_tissues,
        limit=1,
    )
    if context_matches:
        return context_matches[0]
    return None


def _select_census_cell_types_with_llm(
    *,
    target_context: str,
    species_hint: str,
    available_species: List[Dict[str, Any]],
    available_cell_types: List[Dict[str, Any]],
    selected_tissue: str,
    preferred_cell_type: Optional[str] = None,
    max_attempts: int = 5,
) -> List[str]:
    cell_type_lookup = {
        str(ct.get("cell_type")).lower(): str(ct.get("cell_type"))
        for ct in available_cell_types
        if ct.get("cell_type")
    }
    preferred_matches = _preferred_cell_type_matches(preferred_cell_type, available_cell_types)
    if preferred_matches:
        return preferred_matches
    error_hint = None
    for _ in range(max_attempts):
        choice = _select_census_context_with_llm(
            target_context=target_context,
            species_hint=species_hint,
            available_species=available_species,
            available_tissues=[{"tissue": selected_tissue}],
            available_cell_types=available_cell_types,
            preferred_cell_type=preferred_cell_type,
            error_hint=error_hint,
        )
        selected_cell_types = _normalize_choice_list((choice or {}).get("cell_types"), cell_type_lookup, limit=3)
        if preferred_cell_type:
            compatible = [
                cell_type
                for cell_type in selected_cell_types
                if _cell_type_matches_preferred(cell_type, preferred_cell_type)
            ]
            if compatible:
                selected_cell_types = compatible
        if selected_cell_types:
            return selected_cell_types
        error_hint = "Return 1-3 cell_types from available_cell_types; do not invent."
    context_matches = _ranked_context_cell_type_matches(
        target_context=target_context,
        available_cell_types=available_cell_types,
        limit=3,
    )
    if context_matches:
        return context_matches
    fallback = [
        str(item.get("cell_type") or "").strip()
        for item in available_cell_types[:1]
        if str(item.get("cell_type") or "").strip()
    ]
    return fallback


def _ranked_context_cell_type_matches(
    *,
    target_context: str,
    available_cell_types: List[Dict[str, Any]],
    limit: int = 3,
) -> List[str]:
    context_tokens = set(_normalized_label_tokens(target_context))
    if not context_tokens:
        return []
    ranked: List[Tuple[int, int, str]] = []
    for item in available_cell_types:
        cell_type = str(item.get("cell_type") or "").strip()
        if not cell_type:
            continue
        cell_tokens = set(_normalized_label_tokens(cell_type))
        overlap = context_tokens & cell_tokens
        if not overlap:
            continue
        score = 10 * len(overlap) + int(item.get("cell_count") or 0)
        ranked.append((score, int(item.get("cell_count") or 0), cell_type))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [cell_type for _, _, cell_type in ranked[: max(1, limit)]]


def _cell_type_matches_preferred(cell_type: Any, preferred_cell_type: Any) -> bool:
    cell_text = str(cell_type or "").strip().lower()
    preferred_text = str(preferred_cell_type or "").strip().lower()
    if not cell_text or not preferred_text:
        return False
    if cell_text == preferred_text:
        return True
    preferred_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", preferred_text)
        if token not in {"cell", "cells", "type", "line", "derived"}
    ]
    if not preferred_tokens:
        return False
    return all(token in cell_text for token in preferred_tokens)


def _preferred_cell_type_matches(
    preferred_cell_type: Optional[str],
    available_cell_types: List[Dict[str, Any]],
    limit: int = 3,
) -> List[str]:
    preferred = str(preferred_cell_type or "").strip()
    if not preferred:
        return []
    exact_lookup = {
        str(item.get("cell_type") or "").strip().lower(): str(item.get("cell_type") or "").strip()
        for item in available_cell_types
        if str(item.get("cell_type") or "").strip()
    }
    exact = exact_lookup.get(preferred.lower())
    if exact:
        return [exact]
    matches: List[str] = []
    for item in available_cell_types:
        cell_type = str(item.get("cell_type") or "").strip()
        if not cell_type:
            continue
        if _cell_type_matches_preferred(cell_type, preferred):
            matches.append(cell_type)
            if len(matches) >= limit:
                break
    return matches


def _enrich_candidates_with_collection_details(
    api_base: str,
    candidates: List[Dict[str, Any]],
    *,
    timeout: int = CELLXGENE_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    by_collection: Dict[str, List[Dict[str, Any]]] = {}
    for dataset in candidates:
        collection_id = dataset.get("collection_id") or ""
        if not collection_id:
            continue
        by_collection.setdefault(collection_id, []).append(dataset)

    enriched: List[Dict[str, Any]] = []
    for collection_id, datasets in by_collection.items():
        detail = _fetch_collection_detail(api_base, collection_id, timeout=timeout, session=session)
        detail_datasets = detail.get("datasets") or []
        detail_map: Dict[str, Dict[str, Any]] = {}
        for entry in detail_datasets:
            if not isinstance(entry, dict):
                continue
            dataset_id = _extract_dataset_id(entry)
            if dataset_id:
                detail_map[dataset_id] = entry
        for dataset in datasets:
            dataset_id = dataset.get("dataset_id")
            entry = detail_map.get(dataset_id or "")
            if not entry and dataset_id:
                entry = _fetch_dataset_detail(api_base, dataset_id, timeout=timeout, session=session)
            if not entry:
                continue
            assets = _extract_assets(entry)
            h5ad_asset = _pick_h5ad_asset(assets)
            asset_url = _asset_url(h5ad_asset) if h5ad_asset else None
            asset_name = ""
            asset_filesize = None
            asset_filesize_bytes = None
            if isinstance(h5ad_asset, dict):
                asset_name = h5ad_asset.get("name") or h5ad_asset.get("file_name") or ""
                asset_filesize = h5ad_asset.get("filesize")
                asset_filesize_bytes = _coerce_positive_int(asset_filesize)
            dataset_title = (
                _extract_dataset_title(entry)
                or dataset.get("dataset_title")
                or dataset.get("collection_name")
                or ""
            )
            cell_count = entry.get("cell_count") or dataset.get("cell_count")
            description = _build_dataset_description(
                entry,
                collection_entry=detail,
                fallback=str(dataset.get("description") or ""),
            )
            enriched.append(
                {
                    **dataset,
                    "dataset_title": dataset_title,
                    "description": description,
                    "asset_url": asset_url,
                    "asset_name": asset_name,
                    "asset_filesize": asset_filesize,
                    "asset_filesize_bytes": asset_filesize_bytes,
                    "cell_count": cell_count,
                    "has_h5ad": bool(h5ad_asset),
                }
            )
    return enriched


def _fetch_dataset_detail(
    api_base: str,
    dataset_id: str,
    *,
    timeout: int = CELLXGENE_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    try:
        return _cxg_get_json(
            f"{api_base}/curation/v1/datasets/{dataset_id}",
            timeout=timeout,
            session=session,
        )
    except requests.RequestException:
        return {}


def _extract_assets(dataset_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("dataset_assets", "assets", "data_assets", "datasetAssets"):
        assets = dataset_payload.get(key)
        if isinstance(assets, list):
            return assets
    return []


def _asset_url(asset: Dict[str, Any]) -> Optional[str]:
    for key in ("url", "file_url", "fileUrl", "s3_uri", "s3Uri", "access_url"):
        value = asset.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _is_h5ad_asset(asset: Dict[str, Any]) -> bool:
    for key in ("filetype", "file_type", "fileType", "file_format", "fileFormat"):
        value = asset.get(key)
        if isinstance(value, str) and "h5ad" in value.lower():
            return True
    name = asset.get("name") or asset.get("filename") or asset.get("file_name")
    if isinstance(name, str) and name.lower().endswith(".h5ad"):
        return True
    return False


def _pick_h5ad_asset(assets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for asset in assets:
        if _is_h5ad_asset(asset):
            return asset
    return None


def _head_content_length(
    url: str,
    *,
    timeout: int = CELLXGENE_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> Optional[int]:
    requester = session or requests
    try:
        resp = requester.head(url, allow_redirects=True, timeout=timeout)
        if resp.status_code >= 400:
            raise requests.RequestException(f"status {resp.status_code}")
        length = resp.headers.get("Content-Length")
        if length:
            return int(length)
    except Exception:
        try:
            resp = requester.get(url, stream=True, allow_redirects=True, timeout=timeout)
            if resp.status_code >= 400:
                return None
            length = resp.headers.get("Content-Length")
            resp.close()
            if length:
                return int(length)
        except Exception:
            return None
    return None


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    try:
        size = float(value)
    except Exception:
        return "unknown"
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def _download_file(
    url: str,
    destination: Path,
    *,
    chunk_size: int = 1024 * 1024,
    max_bytes: Optional[int] = None,
) -> None:
    del chunk_size
    from .asset_store import download
    download(url, destination, kind="hdf5", max_bytes=max_bytes)


def _load_anndata(path: Path):
    try:
        import anndata as ad  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("anndata is required to read h5ad files") from exc
    return ad.read_h5ad(path, backed="r")


def _maybe_sample_cells(adata, max_cells: Optional[int], seed: Optional[int]) -> Tuple[Any, Optional[int]]:
    if max_cells is None:
        return adata, None
    try:
        max_cells_int = int(max_cells)
    except Exception:
        return adata, None
    if max_cells_int <= 0 or int(adata.n_obs) <= max_cells_int:
        return adata, None
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("numpy is required for cell sampling") from exc
    rng = np.random.default_rng(seed)
    indices = rng.choice(int(adata.n_obs), size=max_cells_int, replace=False)
    indices = np.sort(indices)
    return adata[indices, :], max_cells_int


def _census_organism_label(species_value: Optional[str], organism_labels: Optional[List[str]] = None) -> str:
    for label in organism_labels or []:
        low = str(label).lower()
        if "homo sapiens" in low or low == "human":
            return "Homo sapiens"
        if "mus musculus" in low or low == "mouse":
            return "Mus musculus"
    species_key = normalize_species(species_value)
    if species_key == "mouse":
        return "Mus musculus"
    return "Homo sapiens"


def _escape_filter_value(text: str) -> str:
    return text.replace("'", "\\'")


def _build_census_var_filter(tokens: Sequence[str]) -> str:
    clauses: List[str] = []
    for token in tokens:
        if not token:
            continue
        token_str = str(token).strip()
        if not token_str:
            continue
        token_safe = _escape_filter_value(token_str)
        if token_str.upper().startswith("ENS"):
            base = token_safe.split(".", 1)[0]
            clauses.append(f"feature_id == '{token_safe}'")
            if base and base != token_safe:
                clauses.append(f"feature_id == '{base}'")
        else:
            clauses.append(f"feature_name == '{token_safe}'")
    return " or ".join(clauses)


def _sample_census_obs_coords(
    census,
    organism: str,
    obs_filter: str,
    max_cells: Optional[int],
    seed: Optional[int],
) -> Tuple[Optional[List[int]], Optional[int], Optional[int]]:
    """Sample cells from Census. Uses fast range-based sampling instead of fetching all cells."""
    if max_cells is None:
        return None, None, None
    # Use fast range-based sampling: just take first N cells instead of
    # fetching all cell IDs and randomly sampling (which is extremely slow)
    return list(range(max_cells)), None, max_cells


def _is_broad_baseline_context(target_context: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(target_context or "").lower()).strip()
    if not normalized:
        return False
    broad_markers = (
        "general baseline",
        "general human baseline",
        "baseline context",
        "no specific tissue",
        "no specific cell",
        "no specific disease",
        "no specific model",
        "no tissue specified",
        "no cell type specified",
        "no disease specified",
        "no model system specified",
    )
    return any(marker in normalized for marker in broad_markers)


def _infer_tissue_and_cell_type(target_context: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer tissue_general and cell_type values from target_context string.

    Returns (tissue, cell_type) tuple. Both may be None if not inferrable.
    """
    context_lower = target_context.lower()

    # Map context terms to (tissue_general, cell_type) pairs
    # More specific patterns should come first
    context_mapping = [
        # Common cell lines
        ("hek293t", ("kidney", "kidney epithelial cell")),
        ("hek 293t", ("kidney", "kidney epithelial cell")),
        ("hek293", ("kidney", "kidney epithelial cell")),
        ("hek 293", ("kidney", "kidney epithelial cell")),
        ("293t", ("kidney", "kidney epithelial cell")),
        ("hepg2", ("liver", "hepatocyte")),
        ("huh7", ("liver", "hepatocyte")),
        ("k562", ("blood", "erythroid progenitor cell")),

        # Neurons - specific types first
        ("dopaminergic neuron", ("brain", "dopaminergic neuron")),
        ("dopaminergic", ("brain", "dopaminergic neuron")),
        ("cortical neuron", ("brain", "cerebral cortex neuron")),
        ("cortical neurons", ("brain", "cerebral cortex neuron")),
        ("motor neuron", ("brain", "neuron")),
        ("glutamatergic neuron", ("brain", "glutamatergic neuron")),
        ("gabaergic neuron", ("brain", "GABAergic neuron")),
        ("purkinje", ("brain", "Purkinje cell")),
        ("pyramidal neuron", ("brain", "pyramidal neuron")),
        ("interneuron", ("brain", "inhibitory interneuron")),
        ("neuron", ("brain", "neuron")),
        ("neural", ("brain", "neuron")),

        # Brain - non-neuronal
        ("astrocyte", ("brain", "astrocyte")),
        ("oligodendrocyte", ("brain", "oligodendrocyte")),
        ("microglia", ("brain", "microglial cell")),
        ("glia", ("brain", None)),  # Multiple glial types
        ("brain", ("brain", None)),
        ("cortex", ("brain", None)),
        ("hippocampus", ("brain", None)),
        ("cerebellum", ("brain", None)),
        ("cns", ("brain", None)),
        ("spinal", ("central nervous system", "neuron")),

        # Blood/immune
        ("t cell", ("blood", "T cell")),
        ("b cell", ("blood", "B cell")),
        ("macrophage", ("blood", "macrophage")),
        ("monocyte", ("blood", "monocyte")),
        ("lymphocyte", ("blood", None)),
        ("pbmc", ("blood", None)),
        ("immune", ("blood", None)),
        ("blood", ("blood", None)),

        # Heart
        ("cardiomyocyte", ("heart", "cardiomyocyte")),
        ("cardiac", ("heart", None)),
        ("heart", ("heart", None)),

        # Lung
        ("alveolar", ("lung", "alveolar cell")),
        ("airway", ("lung", None)),
        ("pulmonary", ("lung", None)),
        ("lung", ("lung", None)),

        # Liver
        ("hepatocyte", ("liver", "hepatocyte")),
        ("hepatic", ("liver", None)),
        ("liver", ("liver", None)),

        # Kidney
        ("podocyte", ("kidney", "podocyte")),
        ("renal", ("kidney", None)),
        ("kidney", ("kidney", None)),

        # Eye
        ("retinal ganglion", ("eye", "retinal ganglion cell")),
        ("photoreceptor", ("eye", "photoreceptor cell")),
        ("retina", ("eye", None)),
        ("retinal", ("eye", None)),
        ("eye", ("eye", None)),

        # Muscle
        ("myocyte", ("skeletal muscle organ", "skeletal muscle satellite cell")),
        ("muscle", ("skeletal muscle organ", None)),
        ("skeletal", ("skeletal muscle organ", None)),

        # Other
        ("dermal fibroblast", ("skin of body", "skin fibroblast")),
        ("fibroblast", ("skin of body", "fibroblast")),
        ("dermal", ("skin of body", "skin fibroblast")),
        ("skin", ("skin of body", None)),
        ("keratinocyte", ("skin of body", "keratinocyte")),
        ("beta cell", ("endocrine gland", "pancreatic beta cell")),
        ("islet", ("endocrine gland", None)),
        ("pancreas", ("endocrine gland", None)),
        ("colon", ("colon", None)),
        ("intestine", ("small intestine", None)),
        ("gut", ("small intestine", None)),
        ("enterocyte", ("small intestine", "enterocyte")),
    ]

    for pattern, (tissue, cell_type) in context_mapping:
        if pattern in context_lower:
            return tissue, cell_type

    return None, None


def _infer_tissue_from_context(target_context: str) -> Optional[str]:
    """Infer tissue_general value from target_context string (backward compat)."""
    tissue, _ = _infer_tissue_and_cell_type(target_context)
    return tissue


def _parse_gene_list(value: Optional[Any]) -> List[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        tokens = value
    else:
        tokens = re.split(r"[;,]", str(value))
    out: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        text = str(token).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_value_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_normalize_value_list(item))
        return out
    text = str(value).strip()
    if not text:
        return []
    if any(sep in text for sep in [",", ";"]):
        return [token.strip() for token in re.split(r"[;,]", text) if token.strip()]
    return [text]


def _build_column_maps(var_df, columns: Sequence[str]) -> List[Tuple[str, Dict[str, List[int]], Dict[str, List[int]]]]:
    maps: List[Tuple[str, Dict[str, List[int]], Dict[str, List[int]]]] = []
    for col in columns:
        values = var_df.index.astype(str) if col == "__index__" else var_df[col].astype(str)
        exact_map: Dict[str, List[int]] = {}
        lower_map: Dict[str, List[int]] = {}
        for idx, value in enumerate(values):
            for token in _normalize_value_list(value):
                if not token:
                    continue
                exact_map.setdefault(token, []).append(idx)
                lower_map.setdefault(token.lower(), []).append(idx)
                if token.upper().startswith("ENS") and "." in token:
                    base = token.split(".", 1)[0]
                    exact_map.setdefault(base, []).append(idx)
                    lower_map.setdefault(base.lower(), []).append(idx)
        maps.append((col, exact_map, lower_map))
    return maps


def _match_token_to_indices(
    token: str,
    column_maps: Sequence[Tuple[str, Dict[str, List[int]], Dict[str, List[int]]]],
) -> List[int]:
    if not token:
        return []
    token_text = str(token).strip()
    if not token_text:
        return []
    token_lower = token_text.lower()
    for _, exact_map, lower_map in column_maps:
        if token_text in exact_map:
            return exact_map[token_text]
        if token_lower in lower_map:
            return lower_map[token_lower]
    return []


def _mean_expression(matrix, indices: List[int], n_obs: int) -> Optional[float]:
    if not indices:
        return None
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception:
        sp = None
    col = matrix[:, indices]
    if sp is not None and sp.issparse(col):
        return float(col.mean()) if n_obs else None
    try:
        col_arr = col.toarray()
    except Exception:
        col_arr = col
    if hasattr(col_arr, "mean"):
        return float(col_arr.mean())
    return None


def _compute_expression_map(adata, tokens: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
    if not tokens:
        return {}, []
    candidate_columns = [
        "gene_id",
        "gene_ids",
        "feature_id",
        "ensembl_id",
        "gene_symbol",
        "gene_symbols",
        "symbol",
        "name",
        "feature_name",
        "__index__",
    ]
    columns = [col for col in candidate_columns if col == "__index__" or col in adata.var.columns]
    if "__index__" not in columns:
        columns.append("__index__")
    column_maps = _build_column_maps(adata.var, columns)
    matrix = adata.X
    n_obs = int(adata.n_obs)
    expression_map: Dict[str, float] = {}
    for token in tokens:
        indices = _match_token_to_indices(token, column_maps)
        mean_val = _mean_expression(matrix, indices, n_obs)
        if mean_val is not None:
            expression_map[token] = float(mean_val)
    return expression_map, columns


def _sum_expression_for_row(
    tokens: List[str],
    expression_map: Dict[str, float],
) -> Tuple[Optional[float], List[str]]:
    total: float = 0.0
    matched: List[str] = []
    for token in tokens:
        value = expression_map.get(token)
        if value is None:
            continue
        total += float(value)
        matched.append(token)
    if not matched:
        return None, []
    return total, matched


def _lookup_token_expression(token: Optional[str], expression_map: Dict[str, float]) -> Optional[float]:
    if not token:
        return None
    if token in expression_map:
        return float(expression_map[token])
    lowered = token.lower()
    for key, value in expression_map.items():
        if key.lower() == lowered:
            return float(value)
    return None


def _load_initial_filtering(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        rows = []
    return payload, rows


def _persist_updated_filtering(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


@function_tool
def annotate_off_target_expression(
    initial_filter_json_path: str,
    structured_goal_json: str,
    api_base: str = CELLXGENE_API_BASE,
    page_size: int = CELLXGENE_PAGE_SIZE,
    timeout: int = CELLXGENE_TIMEOUT,
    max_pages: Optional[int] = None,
    max_dataset_candidates: int = CELLXGENE_MAX_DATASET_CANDIDATES,
    cache_dir: str = CELLXGENE_CACHE_DIR,
    max_cells: Optional[int] = CELLXGENE_MAX_CELLS,
    default_sample_mode: str = CELLXGENE_SAMPLE_MODE,
    sample_seed: Optional[int] = CELLXGENE_SAMPLE_SEED,
) -> Dict[str, Any]:
    """
    Annotate initial_filtering_tiles.json with Cellxgene off-target gene expression.

    Adds columns:
      - Off_target_expression_sum: sum of mean expression for off-target genes.
      - Off_target_expression_tissue: dataset label used for expression lookup.
    Optionally random-sample cells when max_cells is set to limit IO.
    """
    if not initial_filter_json_path or not os.path.exists(initial_filter_json_path):
        return {"error": f"initial_filter_json_path not found: {initial_filter_json_path}"}

    try:
        structured_goal = json.loads(structured_goal_json) if structured_goal_json else {}
    except Exception as exc:
        return {"error": f"structured_goal_json is not valid JSON: {exc}"}

    target_symbol = (structured_goal.get("gene_symbol") or "").strip()
    target_gene_id = (structured_goal.get("gene_id") or "").strip()
    species_value = (structured_goal.get("species") or "").strip()
    settings_raw = structured_goal.get("off_target_expression") or structured_goal.get("cellxgene") or {}
    settings = settings_raw if isinstance(settings_raw, dict) else {}
    target_context = str(
        settings.get("target_context") or structured_goal.get("target_context") or ""
    ).strip()
    max_cells_override = _coerce_positive_int(
        settings.get("max_cells") or settings.get("sample_cells") or settings.get("cellxgene_max_cells")
    )
    sample_seed_override = _coerce_int(settings.get("sample_seed") or settings.get("cellxgene_sample_seed"))
    sample_mode_override = settings.get("sample_mode") or settings.get("sample_strategy") or settings.get("cell_sample_mode")
    if max_cells_override is not None:
        max_cells = max_cells_override
    if sample_seed_override is not None:
        sample_seed = sample_seed_override
    sample_mode = (
        str(sample_mode_override).strip().lower()
        if sample_mode_override
        else str(default_sample_mode or "").strip().lower() or None
    )
    use_census = _coerce_bool(settings.get("use_census"))
    if use_census is None:
        use_census = True
    if use_census is False:
        return {"error": "h5ad_fallback_disabled: set use_census true"}
    max_h5ad_bytes = None

    payload, rows = _load_initial_filtering(initial_filter_json_path)

    # Parse gene tokens from rows first (needed for expression queries)
    gene_id_lists: List[List[str]] = []
    gene_id_lists_by_mismatch: List[Dict[str, List[str]]] = []
    tokens: set[str] = set()
    for row in rows:
        gene_ids = _parse_gene_list(row.get("Off_target_gene_ids") or row.get("off_target_gene_ids"))
        gene_id_lists.append(gene_ids)
        tokens.update(gene_ids)
        per_bin: Dict[str, List[str]] = {}
        for label in ("0mm", "1mm", "2mm", "3mm"):
            bin_gene_ids = _parse_gene_list(
                row.get(f"Off_target_{label}_gene_ids")
                or row.get(f"off_target_{label}_gene_ids")
            )
            per_bin[label] = bin_gene_ids
            tokens.update(bin_gene_ids)
        by_mismatch = row.get("Off_target_gene_ids_by_mismatch") or row.get("off_target_gene_ids_by_mismatch")
        if isinstance(by_mismatch, dict):
            for label, raw_ids in by_mismatch.items():
                label_text = str(label).strip()
                if label_text not in per_bin:
                    continue
                parsed_ids = _parse_gene_list(raw_ids)
                if parsed_ids:
                    per_bin[label_text] = sorted(set(per_bin[label_text] + parsed_ids))
                    tokens.update(parsed_ids)
        gene_id_lists_by_mismatch.append(per_bin)
    if target_gene_id:
        tokens.add(target_gene_id)

    # Initialize result tracking
    asset_url = None
    dataset_id = ""
    selected_dataset = None
    total_cells: Optional[int] = None
    sampled_cells: Optional[int] = None
    expression_source = "census"
    expression_map: Dict[str, float] = {}
    expression_stats: Optional[Dict[str, Dict[str, float]]] = None
    feature_columns: List[str] = []

    # Census path only (no AnnData fallback)
    try:
        import cellxgene_census  # type: ignore
    except Exception as exc:
        return {"error": f"cellxgene_census_not_available: {exc}"}

    # Use default if not set
    if max_cells is None:
        max_cells = CELLXGENE_MAX_CELLS

    census_version = _resolve_cellxgene_census_version(cellxgene_census)
    try:
        with cellxgene_census.open_soma(census_version=census_version) as census:
            # Step 1: Select organism based on species
            available_species = _soma_list_species(census)
            print(f"[cellxgene] available species: {[s['organism'] for s in available_species]}", flush=True)

            # Map species to organism
            organism = _census_organism_label(species_value)
            if organism == "Homo sapiens":
                organism = "homo_sapiens"
            elif organism == "Mus musculus":
                organism = "mus_musculus"
            print(f"[cellxgene] selected organism: {organism}", flush=True)

            # Step 2: List tissues and select best match
            available_tissues = _soma_list_tissues(
                census,
                organism,
                limit=CELLXGENE_TISSUE_CANDIDATE_LIMIT,
            )
            print(f"[cellxgene] top tissues: {[t['tissue'] for t in available_tissues[:10]]}", flush=True)

            broad_baseline_context = _is_broad_baseline_context(target_context)
            # Infer tissue/cell type hints, but always rely on LLM selection unless the user
            # explicitly asked for a broad baseline context.
            inferred_tissue, inferred_cell_type = _infer_tissue_and_cell_type(target_context)
            selection_context = {
                "census_version": census_version,
                "organism": organism,
                "target_context": target_context,
                "broad_baseline_context": broad_baseline_context,
                "available_tissues": sorted(
                    str(item.get("tissue") or "")
                    for item in available_tissues
                    if str(item.get("tissue") or "").strip()
                ),
            }
            cached_selection = _load_cellxgene_selection_cache(cache_dir, selection_context)
            available_tissue_names = {
                str(item.get("tissue") or "").strip() for item in available_tissues
            }
            cached_tissue = str(cached_selection.get("tissue") or "").strip()
            selected_tissue = (
                "__all_tissues__"
                if broad_baseline_context
                else cached_tissue
                if cached_tissue in available_tissue_names
                else ""
            )
            selected_cell_types: List[str] = []
            if broad_baseline_context:
                print("[cellxgene] selected broad baseline sample across all tissues", flush=True)
            elif not selected_tissue:
                selected_tissue = _select_census_tissue_with_llm(
                    target_context=target_context,
                    species_hint=species_value,
                    available_species=available_species,
                    available_tissues=available_tissues,
                    preferred_tissue=inferred_tissue,
                )
            if not selected_tissue:
                return {"error": "cellxgene_tissue_selection_failed"}
            if selected_tissue == cached_tissue:
                print("[cellxgene] reused cached tissue/cell interpretation", flush=True)
            print(f"[cellxgene] selected tissue: {selected_tissue}", flush=True)

            # Step 3: List cell types in tissue and select top matches
            if not broad_baseline_context:
                available_cell_types = _soma_list_cell_types(
                    census,
                    organism,
                    selected_tissue,
                    limit=CELLXGENE_CELL_TYPE_CANDIDATE_LIMIT,
                )
                print(
                    f"[cellxgene] top cell types in {selected_tissue}: "
                    f"{[ct['cell_type'] for ct in available_cell_types[:10]]}",
                    flush=True,
                )

                if not available_cell_types:
                    return {"error": "cellxgene_cell_type_selection_failed"}

                available_cell_type_names = {
                    str(item.get("cell_type") or "").strip()
                    for item in available_cell_types
                    if str(item.get("cell_type") or "").strip()
                }
                cached_cell_types = [
                    str(item).strip()
                    for item in (cached_selection.get("cell_types") or [])
                    if str(item).strip()
                ]
                selected_cell_types = (
                    cached_cell_types
                    if cached_cell_types
                    and all(item in available_cell_type_names for item in cached_cell_types)
                    else []
                )
                if selected_cell_types and inferred_cell_type:
                    preferred_cached = [
                        cell_type
                        for cell_type in selected_cell_types
                        if _cell_type_matches_preferred(cell_type, inferred_cell_type)
                    ]
                    if preferred_cached:
                        selected_cell_types = preferred_cached
                    else:
                        selected_cell_types = []
                if not selected_cell_types:
                    selected_cell_types = _select_census_cell_types_with_llm(
                        target_context=target_context,
                        species_hint=species_value,
                        available_species=available_species,
                        available_cell_types=available_cell_types,
                        selected_tissue=selected_tissue,
                        preferred_cell_type=inferred_cell_type,
                    )
                if not selected_cell_types:
                    return {"error": "cellxgene_cell_type_selection_failed"}
            _save_cellxgene_selection_cache(
                cache_dir,
                selection_context,
                {
                    "tissue": selected_tissue,
                    "cell_types": selected_cell_types,
                    "selection_reason": (
                        "broad_baseline_context" if broad_baseline_context else "target_context_match"
                    ),
                },
            )
            print(f"[cellxgene] selected cell types: {selected_cell_types}", flush=True)

            # Step 4: Find genes
            gene_list = list(tokens)
            gene_info = _soma_find_genes(census, organism, gene_list)
            print(f"[cellxgene] found {len(gene_info)}/{len(gene_list)} genes", flush=True)

            if not gene_info:
                return {"error": "No genes found in Census for the provided gene list."}

            # Step 5: Reuse expression only for an identical Census and sampling context.
            normalized_sample_mode = (
                "first"
                if str(sample_mode or "").strip().lower() in {"first", "head", "top"}
                else "random"
            )
            effective_seed = None if normalized_sample_mode == "first" else (sample_seed if sample_seed is not None else 42)
            cache_context = {
                "census_version": census_version,
                "organism": organism,
                "tissue": selected_tissue,
                "cell_types": sorted(str(item) for item in selected_cell_types),
                "max_cells": int(max_cells),
                "sample_mode": normalized_sample_mode,
                "sample_seed": effective_seed,
                "broad_baseline_context": broad_baseline_context,
            }
            cached_stats = _load_cellxgene_expression_cache(cache_dir, cache_context)
            missing_gene_info = {
                gene: info for gene, info in gene_info.items() if gene not in cached_stats
            }
            cache_hits = len(gene_info) - len(missing_gene_info)
            if cache_hits:
                print(
                    f"[cellxgene] expression cache hit: {cache_hits}/{len(gene_info)} genes",
                    flush=True,
                )
            fetched_stats: Dict[str, Dict[str, float]] = {}
            if missing_gene_info:
                fetched_stats = _soma_get_expression(
                    census,
                    organism=organism,
                    tissue=selected_tissue,
                    cell_types=selected_cell_types,
                    gene_info=missing_gene_info,
                    max_cells=max_cells,
                    sample_seed=sample_seed,
                    sample_mode=sample_mode,
                )
                if fetched_stats:
                    _save_cellxgene_expression_cache(cache_dir, cache_context, fetched_stats)
            expression_stats = {
                gene: cached_stats[gene]
                for gene in gene_info
                if gene in cached_stats
            }
            expression_stats.update(fetched_stats)
            expression_map = {}
            for gene, stats in expression_stats.items():
                mean_expr = stats.get("mean_expression")
                if mean_expr is not None:
                    expression_map[gene] = float(mean_expr)
            feature_columns = ["feature_name", "feature_id"]

            # Get cell counts from expression results
            if expression_stats:
                first_gene = next(iter(expression_stats.values()))
                total_cells = first_gene.get("total_cells", 0)
                sampled_cells = (
                    first_gene.get("n_cells", 0)
                    if first_gene.get("n_cells", 0) < total_cells
                    else None
                )
            else:
                total_cells = 0
                sampled_cells = None

        # Create dataset record for results
        if broad_baseline_context:
            dataset_id_suffix = "broad_baseline"
            dataset_title = "CellXGene Census (broad baseline sample)"
        else:
            dataset_id_suffix = selected_tissue
            dataset_title = f"CellXGene Census ({selected_tissue})"
        selected_dataset = {
            "dataset_id": f"census_{organism}_{dataset_id_suffix}",
            "dataset_title": dataset_title,
            "collection_name": "CellXGene Census",
            "organisms": [organism],
            "cell_count": total_cells,
            "tissue": selected_tissue,
            "cell_types": selected_cell_types,
            "selection_reason": "broad_baseline_context" if broad_baseline_context else "target_context_match",
        }
    except Exception as exc:
        print(f"[cellxgene] census query failed: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": f"census_query_failed: {exc}"}

    if not selected_dataset:
        return {"error": "No selected dataset available after expression query."}

    dataset_id = selected_dataset.get("dataset_id") or dataset_id
    asset_url = selected_dataset.get("asset_url") or asset_url

    cells_loaded = sampled_cells if sampled_cells is not None else total_cells

    schema = payload.get("schema") or {}
    columns = schema.setdefault("columns", {})
    columns.setdefault("Off_target_gene_ids", "string")
    columns["Off_target_expression_sum"] = "number"
    columns["Off_target_expression_tissue"] = "string"
    columns["Target_gene_expression"] = "number"
    columns["Off_target_expression_ratio"] = "number"
    columns["Off_target_expression_details"] = "object"
    columns["Off_target_expression_sum_0mm"] = "number"
    columns["Off_target_expression_sum_1mm"] = "number"
    columns["Off_target_expression_sum_2mm"] = "number"
    columns["Off_target_expression_sum_3mm"] = "number"
    columns["Off_target_expression_weighted_risk"] = "number"
    columns["Off_target_expression_measured_sum"] = "number"
    columns["Off_target_expression_coverage"] = "object"

    updated_rows: List[Dict[str, Any]] = []
    annotated_count = 0
    target_expression_value: Optional[float] = None
    if target_gene_id:
        target_expression_value = _lookup_token_expression(target_gene_id, expression_map)

    for idx, row in enumerate(rows):
        gene_ids = gene_id_lists[idx]
        per_bin_gene_ids = gene_id_lists_by_mismatch[idx] if idx < len(gene_id_lists_by_mismatch) else {}
        total, matched = _sum_expression_for_row(gene_ids, expression_map)
        used_tokens = list(gene_ids)
        missing = [token for token in used_tokens if token not in matched]
        measured_total = total
        row['Off_target_expression_measured_sum'] = measured_total
        row['Off_target_expression_coverage'] = {
            'requested_genes':len(used_tokens), 'matched_genes':len(matched), 'missing_genes':missing,
            'status':'partial' if missing and matched else 'unavailable' if missing or not matched else 'complete',
            'scope':'measured_sum is the sum over matched genes only; sum and ratio are unknown when any requested gene lacks expression evidence.'}
        if missing:
            total = None
        per_bin_totals: Dict[str, Optional[float]] = {}
        for label in ("0mm", "1mm", "2mm", "3mm"):
            bin_genes = per_bin_gene_ids.get(label, [])
            bin_total, bin_matched = _sum_expression_for_row(bin_genes, expression_map)
            if any(gene not in bin_matched for gene in bin_genes):
                bin_total = None
            per_bin_totals[label] = bin_total
            row[f"Off_target_expression_sum_{label}"] = bin_total
        weighted_terms = [
            (per_bin_totals.get("0mm"), 100.0),
            (per_bin_totals.get("1mm"), 10.0),
            (per_bin_totals.get("2mm"), 3.0),
            (per_bin_totals.get("3mm"), 1.0),
        ]
        weighted_values = [value * weight for value, weight in weighted_terms if value is not None]
        row["Off_target_expression_weighted_risk"] = sum(weighted_values) if weighted_values and not missing else None

        if total is not None:
            annotated_count += 1
            row["Off_target_expression_sum"] = total
        else:
            row["Off_target_expression_sum"] = None

        dataset_title = selected_dataset.get("dataset_title") or ""
        row["Off_target_expression_tissue"] = dataset_title or dataset_id
        row["Target_gene_expression"] = target_expression_value
        if total is not None and target_expression_value and target_expression_value > 0:
            row["Off_target_expression_ratio"] = total / target_expression_value
        else:
            row["Off_target_expression_ratio"] = None

        details = [
            {
                "dataset_id": dataset_id,
                "dataset_title": dataset_title,
                "collection_id": selected_dataset.get("collection_id"),
                "collection_name": selected_dataset.get("collection_name"),
                "organisms": selected_dataset.get("organisms"),
                "asset_url": asset_url,
                "asset_filesize": selected_dataset.get("asset_filesize"),
                "asset_filesize_bytes": selected_dataset.get("asset_filesize_bytes"),
                "cell_count": selected_dataset.get("cell_count"),
                "cells_loaded": cells_loaded,
                "cells_total": total_cells,
                "cells_sampled": sampled_cells,
                "cells_requested": max_cells,
                "max_h5ad_bytes": max_h5ad_bytes,
                "expression_source": expression_source,
                "sum": total,
                "measured_sum": measured_total,
                "aggregate_scope": row['Off_target_expression_coverage'],
                "matched_genes": matched,
                "missing_genes": missing,
                "feature_columns": feature_columns,
                "tissue_id": dataset_id,
                "tissue_label": dataset_title,
            }
        ]
        row["Off_target_expression_details"] = details
        row["off_target_expression_details"] = details
        updated_rows.append(row)

    payload["rows"] = updated_rows
    _persist_updated_filtering(initial_filter_json_path, payload)

    result = {
        "rows_processed": len(rows),
        "rows_with_expression": annotated_count,
        "selected_dataset": selected_dataset,
        "asset_url": asset_url,
        "cells_total": total_cells,
        "cells_sampled": sampled_cells,
        "max_h5ad_bytes": max_h5ad_bytes,
        "expression_source": expression_source,
        "output_path": initial_filter_json_path,
    }
    return result


off_target_expression_agent = Agent(
    name="Off-Target Expression Agent",
    instructions=(
        "You annotate initial_filtering_tiles.json with Cellxgene expression for off-target genes.\n"
        "Call 'annotate_off_target_expression' exactly once using the provided structured goal JSON and file path.\n"
        "Return only the tool output JSON."
    ),
    tools=[annotate_off_target_expression],
    model="gpt-5-mini-2025-08-07",
)
