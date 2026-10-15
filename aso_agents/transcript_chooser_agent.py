from typing import Any, Dict, List, Optional, Set, Tuple
from copy import deepcopy
import gzip
import os
import time
import requests
import re
import json
from collections import defaultdict
from pathlib import Path
from requests import HTTPError
from urllib.parse import urlparse

from agents import Agent, function_tool, Runner, ModelSettings
from .llm_provider import (
    build_hosted_web_search_tool,
    create_response,
    get_provider_config,
    provider_credentials_available,
)
from .species_config import get_species_config, list_supported_species, normalize_species

# -----------------------
# Config
# -----------------------
ENSEMBL_REST = os.environ.get("ENSEMBL_REST", "https://rest.ensembl.org")
USER_AGENT = os.environ.get("HTTP_USER_AGENT", "aso-agent/0.1 (+https://example.org)")
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GTEX_BASE = os.environ.get("GTEX_API_BASE", "https://gtexportal.org/api/v2")
_ENSEMBL_CONFIG = ENSEMBL_REST.rstrip("/")
_ENSEMBL_PARSED = urlparse(_ENSEMBL_CONFIG if "://" in _ENSEMBL_CONFIG else f"https://{_ENSEMBL_CONFIG}")
ENSEMBL_PRIMARY_HOST = _ENSEMBL_PARSED.netloc or _ENSEMBL_PARSED.path
ENSEMBL_BASE_URL = f"https://{ENSEMBL_PRIMARY_HOST}"
EXPECTED_ENSEMBL_HOST = "rest.ensembl.org"
DEFAULT_LOOKUP_SPECIES = "homo_sapiens"
DEFAULT_LOOKUP_GENE = "OTUD4"
JSON_OBJECT_SETTINGS = ModelSettings(
    reasoning={"effort": "medium"},
    extra_body={"text": {"format": {"type": "json_object"}}},
)
ENSEMBL_TRANSCRIPT_ID_RE = re.compile(r"\b(ENS[A-Z]*T\d+(?:\.\d+)?)\b", re.IGNORECASE)
_ENSEMBL_GENE_LOOKUP_CACHE: Dict[Tuple[str, str, int], Optional[Dict[str, Any]]] = {}
_ENSEMBL_ID_LOOKUP_CACHE: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}
_LOCAL_GTF_GENE_SYMBOL_CACHE: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
_LOCAL_GTF_STABLE_ID_GENE_CACHE: Dict[str, Dict[str, Dict[str, str]]] = {}
_LOCAL_GTF_GENE_PAYLOAD_CACHE: Dict[Tuple[str, float, str], Dict[str, Any]] = {}
_LOCAL_GTF_ID_PAYLOAD_CACHE: Dict[Tuple[str, float, str], Dict[str, Any]] = {}
_LOCAL_REFSEQ_TRANSCRIPT_CACHE: Dict[str, Dict[str, Tuple[str, ...]]] = {}


class EnsemblServiceError(RuntimeError):
    """Raised when Ensembl REST fails (network/HTTP) and the pipeline should stop or retry."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status_code = status_code


class FrozenDependencyUnavailableError(RuntimeError):
    """Raised when a frozen run requests data absent from local references."""


def _frozen_dependency_mode() -> bool:
    return str(os.environ.get("ASO_DEPENDENCY_MODE") or "").strip().lower() == "frozen"


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _ensembl_rest_settings() -> Tuple[int, float, float]:
    """Return (attempts, backoff_seconds, timeout_seconds) from env, with safe defaults."""
    attempts = _env_int("ENSEMBL_REST_RETRIES", 3, minimum=1)
    backoff = _env_float("ENSEMBL_REST_BACKOFF", 1.5, minimum=0.0)
    timeout = _env_float("ENSEMBL_REST_TIMEOUT", 30.0, minimum=1.0)
    return attempts, backoff, timeout


def _ensembl_backoff_seconds(backoff: float, attempt: int) -> float:
    return float(backoff) * (2 ** int(attempt))


def _ensembl_unavailable_reason(
    last_resp: Optional[requests.Response],
    last_exc: Optional[Exception],
) -> Tuple[str, Optional[int]]:
    if last_resp is not None and (last_resp.status_code == 429 or last_resp.status_code >= 500):
        return f"HTTP {last_resp.status_code}", last_resp.status_code
    if last_exc is None:
        return "unknown error", None
    name = type(last_exc).__name__
    text = str(last_exc).lower()
    if "timeout" in name.lower() or "timeout" in text:
        return "timeout", None
    status = getattr(last_exc, "status_code", None)
    return name, status if isinstance(status, int) else None


# -----------------------
# HTTP helpers
# -----------------------
def _get_json(url: str, params: Optional[Dict[str, str]] = None) -> Dict:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    backoff = 1.5
    for attempt in range(3):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(backoff * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    # Last attempt: raise the final status
    resp.raise_for_status()
    return resp.json()

# ============================================================
# INTERNAL HELPERS
# ============================================================
def _assert_human_grch38(species_value: str, payload: Optional[Dict]) -> None:
    species_norm = (species_value or "").lower()
    if species_norm in {"human", "homo_sapiens"}:
        assembly = (payload or {}).get("assembly_name")
        if assembly and assembly != "GRCh38":
            raise EnsemblServiceError(
                f"Unexpected assembly {assembly!r}; expected GRCh38 for human.",
                retryable=False,
            )


def _get_ensembl_json(path: str, params: Optional[Dict[str, str]] = None) -> Tuple[requests.Response, Optional[Dict]]:
    """Perform an Ensembl REST GET with strict host enforcement and retries."""
    if _frozen_dependency_mode():
        raise FrozenDependencyUnavailableError(
            "Live Ensembl REST access is disabled in frozen dependency mode; "
            f"the local reference bundle did not satisfy {path}."
        )
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    attempts, backoff, timeout = _ensembl_rest_settings()
    url = f"{ENSEMBL_BASE_URL}{path}"
    last_exc: Optional[Exception] = None
    last_resp: Optional[requests.Response] = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=True)
            last_resp = resp
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(_ensembl_backoff_seconds(backoff, attempt))
            continue
        final_host = urlparse(resp.url).netloc
        if final_host != EXPECTED_ENSEMBL_HOST:
            raise EnsemblServiceError(
                f"Unexpected Ensembl host '{final_host}', expected '{EXPECTED_ENSEMBL_HOST}'.",
                retryable=False,
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            last_exc = HTTPError(f"HTTP {resp.status_code}")
            if attempt + 1 < attempts:
                time.sleep(_ensembl_backoff_seconds(backoff, attempt))
            continue
        if resp.status_code == 404:
            return resp, None
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise EnsemblServiceError(
                f"Ensembl REST request failed ({resp.status_code}): {resp.text or exc}",
                retryable=False,
                status_code=resp.status_code,
            ) from exc
        try:
            data = resp.json()
        except Exception as exc:
            raise EnsemblServiceError(f"Failed to parse Ensembl JSON: {exc}", retryable=False) from exc
        return resp, data
    reason, status_code = _ensembl_unavailable_reason(last_resp, last_exc)
    effective_url = last_resp.url if last_resp is not None else url
    print(
        f"[warn] Ensembl is likely down ({reason}). "
        "Gene and transcript lookups cannot continue until rest.ensembl.org recovers.",
        flush=True,
    )
    raise EnsemblServiceError(
        f"Ensembl is likely down. Ensembl REST is unavailable ({reason}) after {attempts} retries: {effective_url}",
        retryable=True,
        status_code=status_code,
    ) from last_exc


def _lookup_gene(symbol: str, species: str = "human", expand: int = 1) -> Optional[Dict]:
    species_key = normalize_species(species)
    cache_key = (species_key, str(symbol or "").strip(), int(expand))
    if cache_key in _ENSEMBL_GENE_LOOKUP_CACHE:
        return deepcopy(_ENSEMBL_GENE_LOOKUP_CACHE[cache_key])
    cfg = get_species_config(species_key)
    params = {"expand": str(expand)}
    path = f"/lookup/symbol/{cfg.ensembl_species}/{symbol}"
    try:
        _, result = _get_ensembl_json(path, params)
        if result:
            _assert_human_grch38(cfg.ensembl_species, result)
        payload = result if result else None
        _ENSEMBL_GENE_LOOKUP_CACHE[cache_key] = deepcopy(payload)
        return payload
    except EnsemblServiceError:
        raise
    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status and 400 <= status < 500 and status != 429:
            return None
        raise EnsemblServiceError(
            f"Ensembl REST is unavailable (HTTP {status}) during gene lookup: {exc}",
            retryable=True,
            status_code=status,
        ) from exc
    except Exception as exc:
        raise EnsemblServiceError(
            f"Ensembl REST is unavailable during gene lookup: {exc}",
            retryable=True,
        ) from exc


def check_ensembl_connectivity() -> Dict[str, Any]:
    """Probe Ensembl REST via ping (informational) and mandatory lookup."""
    results: Dict[str, Any] = {"expected_host": EXPECTED_ENSEMBL_HOST}

    # Informational ping
    try:
        resp, data = _get_ensembl_json("/info/ping")
        results["ping"] = {
            "ok": bool(isinstance(data, dict) and data.get("ping") == 1),
            "status_code": resp.status_code,
            "url": resp.url,
            "body": resp.text.strip(),
        }
    except Exception as exc:
        results["ping"] = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}

    # Primary gate: lookup must succeed and stay on the primary host
    try:
        resp, data = _get_ensembl_json(f"/lookup/symbol/{DEFAULT_LOOKUP_SPECIES}/{DEFAULT_LOOKUP_GENE}", params={"expand": "1"})
        _assert_human_grch38(DEFAULT_LOOKUP_SPECIES, data)
        results["lookup"] = {
            "ok": bool(data),
            "status_code": resp.status_code,
            "url": resp.url,
            "assembly": (data or {}).get("assembly_name"),
        }
    except Exception as exc:
        results["lookup"] = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}

    overall_ok = bool(results.get("lookup", {}).get("ok"))
    results["ok"] = overall_ok
    return results

def _lookup_id(stable_id: str, expand: int = 1) -> Optional[Dict]:
    cache_key = (str(stable_id or "").strip(), int(expand))
    # Resolve an explicit version from the installed annotation before asking
    # a current-release REST endpoint, which may no longer serve that version.
    if "." in str(stable_id or ""):
        local_payload = _lookup_local_gtf_id(stable_id)
        if local_payload and _source_identifier_matches(local_payload, stable_id):
            _ENSEMBL_ID_LOOKUP_CACHE[cache_key] = deepcopy(local_payload)
            return local_payload
    if _frozen_dependency_mode():
        local_payload = _lookup_local_gtf_id(stable_id)
        if local_payload:
            _ENSEMBL_ID_LOOKUP_CACHE[cache_key] = deepcopy(local_payload)
            return local_payload
        raise FrozenDependencyUnavailableError(
            "The requested Ensembl identifier is absent from the active frozen "
            f"reference bundle: {stable_id}."
        )
    if cache_key in _ENSEMBL_ID_LOOKUP_CACHE:
        return deepcopy(_ENSEMBL_ID_LOOKUP_CACHE[cache_key])
    params = {"expand": str(expand)}
    try:
        _, data = _get_ensembl_json(f"/lookup/id/{stable_id}", params)
        payload = data if data else None
        _ENSEMBL_ID_LOOKUP_CACHE[cache_key] = deepcopy(payload)
        return payload
    except EnsemblServiceError:
        local_payload = _lookup_local_gtf_id(stable_id)
        if local_payload:
            _ENSEMBL_ID_LOOKUP_CACHE[cache_key] = deepcopy(local_payload)
            return local_payload
        raise
    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status and 400 <= status < 500 and status != 429:
            return None
        raise EnsemblServiceError(
            f"Ensembl REST is unavailable (HTTP {status}) during ID lookup: {exc}",
            retryable=True,
            status_code=status,
        ) from exc
    except Exception as exc:
        raise EnsemblServiceError(
            f"Ensembl REST is unavailable during ID lookup: {exc}",
            retryable=True,
        ) from exc


def _lookup_gene_by_symbol(symbol: str, species_key: str, expand: int) -> Optional[Dict[str, Any]]:
    symbol_text = str(symbol or "").strip()
    if _frozen_dependency_mode():
        local_gene = _lookup_gene_by_local_gtf_symbol(
            symbol_text, species_key, expand=expand
        )
        if local_gene:
            return local_gene
        raise FrozenDependencyUnavailableError(
            "The requested gene symbol is absent from the active frozen reference "
            f"bundle for {species_key}: {symbol_text}."
        )
    variants: List[str] = []
    for candidate in (
        symbol_text,
        symbol_text.upper() if symbol_text else "",
        symbol_text.lower() if symbol_text else "",
    ):
        if candidate and candidate not in variants:
            variants.append(candidate)

    service_errors: List[EnsemblServiceError] = []
    for candidate in variants:
        try:
            gene = _lookup_gene(candidate, species=species_key, expand=expand)
        except EnsemblServiceError as exc:
            service_errors.append(exc)
            continue
        if gene:
            if candidate != symbol_text:
                gene["_matched_symbol_variant"] = candidate
            gene.setdefault("_resolution_source", "ensembl_lookup_symbol")
            return gene

    for candidate in variants:
        try:
            gene = _lookup_gene_by_xrefs(candidate, species_key, expand=expand)
        except EnsemblServiceError as exc:
            service_errors.append(exc)
            continue
        if gene:
            if candidate != symbol_text:
                gene["_matched_symbol_variant"] = candidate
            return gene

    try:
        local_gene = _lookup_gene_by_local_gtf_symbol(symbol_text, species_key, expand=expand)
    except EnsemblServiceError as exc:
        service_errors.append(exc)
    else:
        if local_gene:
            return local_gene

    if service_errors:
        permanent_errors = [
            exc for exc in service_errors
            if not bool(getattr(exc, "retryable", True))
        ]
        retryable = not permanent_errors and any(
            bool(getattr(exc, "retryable", True))
            for exc in service_errors
        )
        representative = permanent_errors[0] if permanent_errors else service_errors[0]
        failure_kind = (
            "Ensembl gene symbol resolution was rejected"
            if permanent_errors
            else "Ensembl REST is unavailable during gene symbol resolution"
        )
        raise EnsemblServiceError(
            failure_kind + " after lookup, xref, and local GTF fallbacks: "
            + "; ".join(str(exc) for exc in service_errors[:3]),
            retryable=retryable,
            status_code=getattr(representative, "status_code", None),
        ) from representative
    return None


def _parse_gtf_attributes(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for raw_attr in str(attr_text or "").split(";"):
        attr = raw_attr.strip()
        if not attr:
            continue
        if " " in attr:
            key, value = attr.split(" ", 1)
        elif "=" in attr:
            key, value = attr.split("=", 1)
        else:
            continue
        attrs[key.strip()] = value.strip().strip('"')
    return attrs


def _parse_gtf_attribute_values(attr_text: str) -> Dict[str, List[str]]:
    attrs: Dict[str, List[str]] = defaultdict(list)
    for raw_attr in str(attr_text or "").split(";"):
        attr = raw_attr.strip()
        if not attr:
            continue
        if " " in attr:
            key, value = attr.split(" ", 1)
        elif "=" in attr:
            key, value = attr.split("=", 1)
        else:
            continue
        key = key.strip()
        value = value.strip().strip('"')
        if key and value:
            attrs[key].append(value)
    return dict(attrs)


def _local_gtf_cache_identity(species_key: str) -> Optional[Tuple[str, float]]:
    gtf_path = get_species_config(species_key).gtf_path
    compressed_path = gtf_path.with_suffix(f"{gtf_path.suffix}.gz")
    if compressed_path.exists():
        gtf_path = compressed_path
    if not gtf_path.exists():
        return None
    return str(gtf_path.resolve()), float(gtf_path.stat().st_mtime)


def _open_gtf_text(gtf_path: str):
    return gzip.open(gtf_path, "rt", encoding="utf-8", errors="ignore") if gtf_path.endswith(".gz") else open(
        gtf_path,
        "rt",
        encoding="utf-8",
        errors="ignore",
    )


def _load_local_gtf_gene_symbol_index(species_key: str) -> Dict[str, List[Dict[str, str]]]:
    normalized_species = normalize_species(species_key)
    identity = _local_gtf_cache_identity(normalized_species)
    if identity is None:
        return {}
    cache_key = f"{normalized_species}:{identity[0]}:{identity[1]}"
    if (
        cache_key in _LOCAL_GTF_GENE_SYMBOL_CACHE
        and cache_key in _LOCAL_GTF_STABLE_ID_GENE_CACHE
    ):
        return _LOCAL_GTF_GENE_SYMBOL_CACHE[cache_key]

    cfg = get_species_config(normalized_species)
    index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    genes_by_id: Dict[str, Dict[str, str]] = {}
    transcript_gene_ids: Dict[str, str] = {}
    gtf_path = cfg.gtf_path
    with _open_gtf_text(str(gtf_path)) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            attrs = _parse_gtf_attributes(parts[8])
            gene_id = str(attrs.get("gene_id") or "").split(".", 1)[0].strip()
            gene_name = str(attrs.get("gene_name") or attrs.get("Name") or "").strip()
            if not gene_id or not gene_name:
                continue
            if parts[2] == "gene":
                match = {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "gene_biotype": attrs.get("gene_type") or attrs.get("gene_biotype") or "",
                    "seq_region_name": parts[0],
                    "start": parts[3],
                    "end": parts[4],
                    "strand": parts[6],
                }
                genes_by_id[gene_id] = match
                index[gene_name.lower()].append(match)
            transcript_id = str(attrs.get("transcript_id") or "").split(".", 1)[0].strip()
            if transcript_id:
                transcript_gene_ids[transcript_id] = gene_id

    cached = dict(index)
    stable_index = dict(genes_by_id)
    for transcript_id, gene_id in transcript_gene_ids.items():
        match = genes_by_id.get(gene_id)
        if match is not None:
            stable_index[transcript_id] = match
    _LOCAL_GTF_GENE_SYMBOL_CACHE[cache_key] = cached
    _LOCAL_GTF_STABLE_ID_GENE_CACHE[cache_key] = stable_index
    return cached


def _load_local_gtf_stable_id_index(
    species_key: str,
) -> Dict[str, Dict[str, str]]:
    normalized_species = normalize_species(species_key)
    identity = _local_gtf_cache_identity(normalized_species)
    if identity is None:
        return {}
    cache_key = f"{normalized_species}:{identity[0]}:{identity[1]}"
    _load_local_gtf_gene_symbol_index(normalized_species)
    return _LOCAL_GTF_STABLE_ID_GENE_CACHE.get(cache_key, {})


def local_reference_gene_transcripts(
    species_key: str,
    *,
    gene_id: str = "",
    gene_symbol: str = "",
) -> List[Dict[str, Any]]:
    """Return same-gene transcripts from the installed, versioned annotation.

    This is a reference-policy lookup, so a missing local annotation does not
    silently substitute a live Ensembl release with different transcript
    versions.
    """
    species = normalize_species(species_key)
    stable_gene_id = str(gene_id or "").split(".", 1)[0].strip()
    # A workbench may copy a gene symbol into both the symbol and ID slots.
    # Their equality is evidence of one label, not a stable-ID lock. Resolve
    # that label against the installed annotation before requiring an ID match.
    if (stable_gene_id and stable_gene_id.casefold() == str(gene_symbol or "").strip().casefold()
            and stable_gene_id not in _load_local_gtf_stable_id_index(species)):
        stable_gene_id = ""
    match = (
        _load_local_gtf_stable_id_index(species).get(stable_gene_id)
        if stable_gene_id else None
    )
    if match is None and gene_symbol:
        matches = _load_local_gtf_gene_symbol_index(species).get(
            str(gene_symbol).strip().casefold(), []
        )
        if len(matches) == 1:
            match = matches[0]
    if not isinstance(match, dict):
        return []
    payload = _local_gtf_gene_payload(match, species)
    if not isinstance(payload, dict):
        return []
    if stable_gene_id and str(payload.get("id") or "") != stable_gene_id:
        return []
    return [
        deepcopy(item)
        for item in payload.get("Transcript") or []
        if isinstance(item, dict)
    ]


def _local_gtf_gene_payload(
    match: Dict[str, str],
    species_key: str,
) -> Optional[Dict[str, Any]]:
    identity = _local_gtf_cache_identity(species_key)
    gene_id = str(match.get("gene_id") or "").split(".", 1)[0].strip()
    if identity is None or not gene_id:
        return None
    gtf_path, gtf_mtime = identity
    cache_key = (gtf_path, gtf_mtime, gene_id)
    cached = _LOCAL_GTF_GENE_PAYLOAD_CACHE.get(cache_key)
    if cached:
        return deepcopy(cached)

    cfg = get_species_config(species_key)
    gene: Dict[str, Any] = {
        "id": gene_id,
        "display_name": str(match.get("gene_name") or gene_id),
        "description": None,
        "object_type": "Gene",
        "biotype": str(match.get("gene_biotype") or ""),
        "species": cfg.ensembl_species,
        "assembly_name": cfg.ensembl_assembly,
        "seq_region_name": str(match.get("seq_region_name") or ""),
        "start": int(match.get("start") or 0),
        "end": int(match.get("end") or 0),
        "strand": -1 if str(match.get("strand") or "") == "-" else 1,
        "Transcript": [],
        "_resolution_source": "local_gtf_gene_symbol",
    }
    transcripts: Dict[str, Dict[str, Any]] = {}
    with _open_gtf_text(gtf_path) as handle:
        for raw in handle:
            if gene_id not in raw or raw.startswith("#"):
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            attr_values = _parse_gtf_attribute_values(parts[8])
            row_gene_id = str((attr_values.get("gene_id") or [""])[-1]).split(".", 1)[0]
            if row_gene_id != gene_id:
                continue
            feature = parts[2]
            if feature == "gene":
                source_gene_id = str((attr_values.get("gene_id") or [gene_id])[-1])
                gene["versioned_id"] = source_gene_id
                suffix = source_gene_id.rpartition(".")[2]
                if "." in source_gene_id and suffix.isdigit():
                    gene["version"] = int(suffix)
                gene["seq_region_name"] = parts[0]
                gene["start"] = int(parts[3])
                gene["end"] = int(parts[4])
                gene["strand"] = -1 if parts[6] == "-" else 1
                continue
            source_transcript_id = str((attr_values.get("transcript_id") or [""])[-1])
            transcript_id = source_transcript_id.split(".", 1)[0]
            if not transcript_id:
                continue
            tags = set(attr_values.get("tag") or [])
            transcript = transcripts.setdefault(
                transcript_id,
                {
                    "id": transcript_id,
                    "Parent": gene_id,
                    "gene_symbol": str((attr_values.get("gene_name") or [""])[-1]) or None,
                    "versioned_id": source_transcript_id,
                    "version": int(source_transcript_id.rsplit(".", 1)[1])
                    if "." in source_transcript_id and source_transcript_id.rsplit(".", 1)[1].isdigit()
                    else None,
                    "reference_source": "local_gtf",
                    "reference_annotation": gtf_path,
                    "species": cfg.ensembl_species,
                    "assembly_name": cfg.ensembl_assembly,
                    "display_name": str((attr_values.get("transcript_name") or [transcript_id])[-1]),
                    "description": None,
                    "biotype": str((attr_values.get("transcript_type") or [""])[-1]),
                    "seq_region_name": parts[0],
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": -1 if parts[6] == "-" else 1,
                    "is_canonical": 0,
                    "is_mane_select": False,
                    "is_mane_plus_clinical": False,
                    "Exon": [],
                    "_tags": set(),
                    "_coding_start": None,
                    "_coding_end": None,
                    "_coding_bp": 0,
                    "_protein_id": "",
                },
            )
            transcript["start"] = min(int(transcript.get("start") or parts[3]), int(parts[3]))
            transcript["end"] = max(int(transcript.get("end") or parts[4]), int(parts[4]))
            transcript["_tags"].update(tags)
            if feature == "exon":
                transcript["Exon"].append(
                    {
                        "id": str((attr_values.get("exon_id") or [""])[-1]).split(".", 1)[0],
                        "start": int(parts[3]),
                        "end": int(parts[4]),
                        "seq_region_name": parts[0],
                        "phase": None,
                        "end_phase": None,
                    }
                )
            elif feature == "CDS":
                cds_start = int(parts[3])
                cds_end = int(parts[4])
                transcript["_coding_start"] = (
                    cds_start
                    if transcript["_coding_start"] is None
                    else min(int(transcript["_coding_start"]), cds_start)
                )
                transcript["_coding_end"] = (
                    cds_end
                    if transcript["_coding_end"] is None
                    else max(int(transcript["_coding_end"]), cds_end)
                )
                transcript["_coding_bp"] += cds_end - cds_start + 1
                protein_id = str((attr_values.get("protein_id") or [""])[-1]).split(".", 1)[0]
                if protein_id:
                    transcript["_protein_id"] = protein_id

    for transcript in transcripts.values():
        tags = transcript.pop("_tags")
        transcript["is_mane_select"] = "MANE_Select" in tags
        transcript["is_mane_plus_clinical"] = "MANE_Plus_Clinical" in tags
        transcript["is_canonical"] = int("Ensembl_canonical" in tags)
        coding_start = transcript.pop("_coding_start")
        coding_end = transcript.pop("_coding_end")
        coding_bp = int(transcript.pop("_coding_bp") or 0)
        protein_id = str(transcript.pop("_protein_id") or "")
        if coding_start is not None and coding_end is not None:
            transcript["Translation"] = {
                "id": protein_id or None,
                "start": coding_start,
                "end": coding_end,
                "length": coding_bp // 3,
            }
        transcript["Exon"].sort(key=lambda exon: (int(exon["start"]), int(exon["end"])))
        _LOCAL_GTF_ID_PAYLOAD_CACHE[(gtf_path, gtf_mtime, transcript["id"])] = transcript
        _ENSEMBL_ID_LOOKUP_CACHE[(transcript["id"], 0)] = deepcopy(transcript)
        _ENSEMBL_ID_LOOKUP_CACHE[(transcript["id"], 1)] = deepcopy(transcript)

    gene["Transcript"] = sorted(transcripts.values(), key=lambda transcript: transcript["id"])
    _LOCAL_GTF_GENE_PAYLOAD_CACHE[cache_key] = gene
    _LOCAL_GTF_ID_PAYLOAD_CACHE[cache_key] = gene
    _ENSEMBL_ID_LOOKUP_CACHE[(gene_id, 0)] = deepcopy(gene)
    _ENSEMBL_ID_LOOKUP_CACHE[(gene_id, 1)] = deepcopy(gene)
    return deepcopy(gene)


def _lookup_local_gtf_id(stable_id: str) -> Optional[Dict[str, Any]]:
    stable = str(stable_id or "").split(".", 1)[0].strip()
    if not stable:
        return None
    for species_key in list_supported_species():
        identity = _local_gtf_cache_identity(species_key)
        if identity is None:
            continue
        cached = _LOCAL_GTF_ID_PAYLOAD_CACHE.get((identity[0], identity[1], stable))
        if cached and _source_identifier_matches(cached, stable_id):
            return deepcopy(cached)
        match = _load_local_gtf_stable_id_index(species_key).get(stable)
        if match:
            gene = _local_gtf_gene_payload(match, species_key)
            if not gene:
                continue
            if str(gene.get("id") or "") == stable:
                return gene
            cached = _LOCAL_GTF_ID_PAYLOAD_CACHE.get(
                (identity[0], identity[1], stable)
            )
            if cached and _source_identifier_matches(cached, stable_id):
                return deepcopy(cached)
    return None


def _source_identifier_matches(payload: Dict[str, Any], requested_id: str) -> bool:
    """Match a version lock only against a version reported by the reference."""
    requested = str(requested_id or "").strip().upper()
    source = str(payload.get("id") or payload.get("versioned_id") or "").strip().upper()
    if requested.split(".", 1)[0] != source.split(".", 1)[0]:
        return False
    if "." not in requested:
        return True
    versioned = str(payload.get("versioned_id") or source).upper()
    if "." not in versioned and payload.get("version") is not None:
        versioned = f"{source}.{payload['version']}"
    return requested == versioned


def _transcript_reference_identity(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Carry version provenance through the stable-ID metadata/structure APIs."""
    identity = {k: payload[k] for k in (
        "versioned_id", "version", "reference_source", "reference_annotation",
        "species", "assembly_name"
    ) if payload.get(k) is not None}
    source = str(payload.get("id") or "")
    if "versioned_id" not in identity and payload.get("version") is not None:
        identity["versioned_id"] = f"{source.split('.', 1)[0]}.{payload['version']}"
    return identity


def _lookup_gene_by_local_gtf_symbol(symbol: str, species_key: str, expand: int) -> Optional[Dict[str, Any]]:
    symbol_norm = str(symbol or "").strip().upper()
    if not symbol_norm:
        return None
    matches = _load_local_gtf_gene_symbol_index(species_key).get(symbol_norm.lower(), [])
    if not matches:
        return None
    cfg = get_species_config(species_key)
    sorted_matches = sorted(
        matches,
        key=lambda item: (
            0 if str(item.get("gene_name") or "").strip().upper() == symbol_norm else 1,
            0 if str(item.get("gene_biotype") or "").strip().lower() == "protein_coding" else 1,
            str(item.get("gene_id") or ""),
        ),
    )
    service_errors: List[str] = []
    for match in sorted_matches:
        stable_id = str(match.get("gene_id") or "").strip()
        if not stable_id:
            continue
        local_gene = _local_gtf_gene_payload(match, species_key)
        if local_gene:
            return local_gene
        try:
            gene = _lookup_id(stable_id, expand=expand)
        except EnsemblServiceError as exc:
            service_errors.append(str(exc))
            continue
        if not gene:
            continue
        obj_type = str(gene.get("object_type") or "").strip().lower()
        if obj_type and obj_type != "gene":
            continue
        species_name = str(gene.get("species") or "").strip().lower()
        if species_name and species_name != cfg.ensembl_species.lower():
            continue
        _assert_human_grch38(cfg.ensembl_species, gene)
        gene["_resolution_source"] = "local_gtf_gene_symbol"
        return gene
    if service_errors:
        raise EnsemblServiceError(
            "Ensembl REST gene ID lookup failed after local GTF symbol match: "
            + "; ".join(service_errors[:3])
        )
    return None


def _lookup_gene_by_xrefs(symbol: str, species_key: str, expand: int) -> Optional[Dict[str, Any]]:
    cfg = get_species_config(species_key)
    try:
        _, data = _get_ensembl_json(
            f"/xrefs/symbol/{cfg.ensembl_species}/{symbol}",
            {"object_type": "gene"},
        )
    except EnsemblServiceError:
        raise
    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status and 400 <= status < 500 and status != 429:
            return None
        raise EnsemblServiceError(
            f"Ensembl REST is unavailable (HTTP {status}) during gene xref lookup: {exc}",
            retryable=True,
            status_code=status,
        ) from exc
    except Exception as exc:
        raise EnsemblServiceError(
            f"Ensembl REST is unavailable during gene xref lookup: {exc}",
            retryable=True,
        ) from exc

    rows = data if isinstance(data, list) else []
    candidate_ids: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_type = str(row.get("type") or "").strip().lower()
        gene_id = str(row.get("id") or row.get("primary_id") or "").split(".", 1)[0].strip()
        if row_type and row_type != "gene":
            continue
        if gene_id and gene_id.upper().startswith("ENS") and gene_id not in candidate_ids:
            candidate_ids.append(gene_id)

    candidates: List[Dict[str, Any]] = []
    for gene_id in candidate_ids:
        gene = _lookup_id(gene_id, expand=expand)
        if not gene:
            continue
        obj_type = str(gene.get("object_type") or "").strip().lower()
        if obj_type and obj_type != "gene":
            continue
        species_name = str(gene.get("species") or "").strip().lower()
        if species_name and species_name != cfg.ensembl_species.lower():
            continue
        _assert_human_grch38(cfg.ensembl_species, gene)
        gene["_resolution_source"] = "ensembl_xrefs_symbol"
        candidates.append(gene)

    if not candidates:
        return None

    symbol_norm = str(symbol or "").strip().lower()
    candidates.sort(
        key=lambda gene: (
            0 if str(gene.get("display_name") or gene.get("name") or "").strip().lower() == symbol_norm else 1,
            0 if str(gene.get("biotype") or "").strip().lower() == "protein_coding" else 1,
            str(gene.get("id") or ""),
        )
    )
    return candidates[0]

TEST_ISOFORM_OVERRIDES: Dict[str, Dict[str, str]] = {
    "LMNA": {
        "lamin a": "ENST00000368300",
        "prelamin a": "ENST00000368300",
    }
}


def _env_flag(name: str) -> bool:
    value = str(os.environ.get(name) or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


ENABLE_TEST_ISOFORM_OVERRIDES = _env_flag("ASO_ENABLE_TEST_ISOFORM_OVERRIDES")


def _extract_stable_transcript_id(text: Any) -> Optional[str]:
    match = ENSEMBL_TRANSCRIPT_ID_RE.search(str(text or ""))
    if not match:
        return None
    return match.group(1).split(".", 1)[0].upper()


def _extract_stable_transcript_ids(text: Any) -> List[str]:
    seen: Set[str] = set()
    ids: List[str] = []
    for match in ENSEMBL_TRANSCRIPT_ID_RE.finditer(str(text or "")):
        tid = match.group(1).split(".", 1)[0].upper()
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)
    return ids


def _find_transcript_id(transcripts: List[Dict[str, Any]], candidate_id: str) -> Optional[str]:
    if not candidate_id:
        return None
    base = candidate_id.split(".", 1)[0]
    for tx in transcripts:
        tid = tx.get("id") or ""
        if not tid:
            continue
        tid_base = tid.split(".", 1)[0]
        if (tid == candidate_id or tid_base == base) and _source_identifier_matches(tx, candidate_id):
            return tx.get("id")
    return None


def _looks_like_external_transcript_accession(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or any(ch.isspace() for ch in text):
        return False
    upper = text.upper()
    base, separator, version = upper.partition(".")
    if separator and (not version or not version.isdigit()):
        return False
    if upper.startswith(("NM_", "NR_", "XM_", "XR_")):
        return base[3:].isdigit()
    if base.startswith("CCDS"):
        return base[4:].isdigit()
    if base.startswith("OTT"):
        transcript_marker = base.find("T", 3)
        return transcript_marker >= 3 and base[transcript_marker + 1 :].isdigit()
    if not base.startswith("ENS"):
        return False
    transcript_marker = base.find("T", 3)
    return transcript_marker >= 3 and base[transcript_marker + 1 :].isdigit()


def _local_refseq_cache_identity(species_key: str) -> Optional[Tuple[str, float]]:
    cfg = get_species_config(normalize_species(species_key))
    configured = getattr(cfg, "refseq_xref_path", None)
    if configured is None:
        return None
    path = Path(configured).expanduser()
    if not path.exists() and path.suffix == ".gz":
        uncompressed = path.with_suffix("")
        if uncompressed.exists():
            path = uncompressed
    if not path.is_file():
        return None
    return str(path.resolve()), float(path.stat().st_mtime)


def _open_metadata_text(path: str):
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        if path.endswith(".gz")
        else open(
            path,
            "rt",
            encoding="utf-8",
            errors="ignore",
        )
    )


def _load_local_refseq_transcript_index(
    species_key: str,
) -> Dict[str, Tuple[str, ...]]:
    identity = _local_refseq_cache_identity(species_key)
    if identity is None:
        return {}
    cache_key = f"{normalize_species(species_key)}:{identity[0]}:{identity[1]}"
    cached = _LOCAL_REFSEQ_TRANSCRIPT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows: Dict[str, Set[str]] = defaultdict(set)
    with _open_metadata_text(identity[0]) as handle:
        for raw in handle:
            columns = raw.rstrip("\n").split("\t")
            if len(columns) < 2:
                continue
            transcript_id = str(columns[0] or "").split(".", 1)[0].strip().upper()
            if not transcript_id:
                continue
            for raw_accession in columns[1:]:
                accession = str(raw_accession or "").strip().upper()
                if not accession.startswith(("NM_", "NR_", "XM_", "XR_")):
                    continue
                rows[accession].add(transcript_id)
                rows[accession.split(".", 1)[0]].add(transcript_id)
    index = {
        accession: tuple(sorted(transcript_ids))
        for accession, transcript_ids in rows.items()
    }
    _LOCAL_REFSEQ_TRANSCRIPT_CACHE[cache_key] = index
    return index


def _resolve_local_external_transcript_accession(
    accession: str,
    *,
    species_key: str,
    candidate_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    accession_text = str(accession or "").strip()
    accession_upper = accession_text.upper()
    candidates = candidate_by_id or {}

    if accession_upper.startswith("ENS"):
        transcript_id = accession_upper.split(".", 1)[0]
        candidate = candidates.get(transcript_id)
        if candidates and candidate is None:
            return None
        if transcript_id not in _load_local_gtf_stable_id_index(species_key):
            return None
        return {
            "transcript_id": str(candidate.get("id")) if candidate else transcript_id,
            "method": "local_gencode_transcript_id",
            "confidence": "high",
            "rationale": (
                f"Transcript accession {accession_text} resolved in the active frozen "
                "GENCODE annotation."
            ),
        }

    index = _load_local_refseq_transcript_index(species_key)
    mapped_ids = index.get(accession_upper)
    if not mapped_ids and "." in accession_upper:
        mapped_ids = index.get(accession_upper.split(".", 1)[0])
    if not mapped_ids:
        return None
    local_transcripts = _load_local_gtf_stable_id_index(species_key)
    mapped_ids = tuple(
        transcript_id
        for transcript_id in mapped_ids
        if transcript_id in local_transcripts
    )
    if candidates:
        mapped_ids = tuple(
            transcript_id for transcript_id in mapped_ids if transcript_id in candidates
        )
    if not mapped_ids:
        return None
    if len(mapped_ids) > 1:
        return {
            "transcript_id": None,
            "transcript_ids": list(mapped_ids),
            "ambiguous": True,
            "method": "local_gencode_refseq_xref",
            "confidence": "ambiguous",
            "rationale": (
                f"RefSeq accession {accession_text} maps to multiple transcripts in "
                "the active frozen GENCODE metadata."
            ),
        }
    transcript_id = mapped_ids[0]
    candidate = candidates.get(transcript_id)
    return {
        "transcript_id": str(candidate.get("id")) if candidate else transcript_id,
        "method": "local_gencode_refseq_xref",
        "confidence": "high",
        "rationale": (
            f"RefSeq accession {accession_text} resolved through the active frozen "
            f"GENCODE metadata to transcript {transcript_id}."
        ),
    }


def extract_transcript_accessions(value: Any) -> List[str]:
    """Collect transcript accessions from nested case data without interpreting prose."""

    strings: List[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                collect(nested)

    collect(value)
    accessions: List[str] = []
    seen: Set[str] = set()
    for string in strings:
        token_chars: List[str] = []
        tokens: List[str] = []
        for character in string:
            if character.isalnum() or character in {"_", "."}:
                token_chars.append(character)
            elif token_chars:
                tokens.append("".join(token_chars))
                token_chars = []
        if token_chars:
            tokens.append("".join(token_chars))
        for token in tokens:
            normalized = token.strip(".").upper()
            if not _looks_like_external_transcript_accession(normalized):
                continue
            if normalized not in seen:
                seen.add(normalized)
                accessions.append(normalized)
    return accessions


def preflight_frozen_transcript_accessions(accessions: List[str]) -> Dict[str, Any]:
    """Prove explicit transcript accessions are resolvable from frozen local assets."""

    supported_species = sorted(list_supported_species())
    checked: List[Dict[str, Any]] = []
    missing: List[str] = []
    for accession in accessions:
        matches: List[Dict[str, Any]] = []
        for species_key in supported_species:
            resolved = _resolve_local_external_transcript_accession(
                accession,
                species_key=species_key,
            )
            if resolved is not None:
                matches.append({"species": species_key, **resolved})
                break
        if matches:
            checked.append({"accession": accession, "matches": matches})
        else:
            missing.append(accession)
    return {
        "accessions": checked,
        "checked_count": len(accessions),
        "missing": missing,
        "ok": not missing,
    }


def resolve_external_transcript_accession(
    accession: str,
    *,
    species: str = "human",
    candidate_transcripts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve external transcript accessions, such as RefSeq mRNAs, to candidate Ensembl transcripts."""
    accession_text = str(accession or "").strip()
    if not accession_text:
        return None
    cfg = get_species_config(normalize_species(species))
    candidate_transcripts = candidate_transcripts if isinstance(candidate_transcripts, list) else []

    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    for tx in candidate_transcripts:
        if not isinstance(tx, dict):
            continue
        tid = str(tx.get("id") or tx.get("transcript_id") or "").strip()
        if tid:
            candidate_by_id[tid.split(".", 1)[0].upper()] = tx
        for value in tx.values():
            if isinstance(value, str) and value.strip().split(".", 1)[0].upper() == accession_text.split(".", 1)[0].upper():
                if tid:
                    return {
                        "transcript_id": tid,
                        "method": "candidate_metadata_xref",
                        "confidence": "high",
                        "rationale": f"External accession {accession_text} matched candidate transcript metadata.",
                    }
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip().split(".", 1)[0].upper() == accession_text.split(".", 1)[0].upper():
                        if tid:
                            return {
                                "transcript_id": tid,
                                "method": "candidate_metadata_xref",
                                "confidence": "high",
                                "rationale": f"External accession {accession_text} matched candidate transcript metadata.",
                            }

    if not _looks_like_external_transcript_accession(accession_text):
        return None

    # An exact installed cross-reference is authoritative in live mode too.
    # Avoid an unnecessary remote dependency while retaining accession versions
    # and ambiguous mappings. Never substitute a base-accession match here for
    # an explicitly requested version absent from the installed metadata.
    local_index = _load_local_refseq_transcript_index(normalize_species(species))
    if accession_text.upper() in local_index:
        local_match = _resolve_local_external_transcript_accession(
            accession_text, species_key=normalize_species(species), candidate_by_id=candidate_by_id)
        if local_match is not None:
            identity = _local_refseq_cache_identity(normalize_species(species))
            return {**local_match, 'requested_accession':accession_text,
                    'xref_source':identity[0] if identity else None,
                    'rationale':f'Exact accession {accession_text} found in installed annotation cross-reference metadata; mapping multiplicity is retained.'}

    if _frozen_dependency_mode():
        local_match = _resolve_local_external_transcript_accession(
            accession_text,
            species_key=normalize_species(species),
            candidate_by_id=candidate_by_id,
        )
        if local_match is not None:
            return local_match
        raise FrozenDependencyUnavailableError(
            "The requested external transcript accession is absent from the active "
            f"frozen reference bundle for {normalize_species(species)}: {accession_text}."
        )

    names = [accession_text]
    base = accession_text.split(".", 1)[0]
    if base and base != accession_text:
        names.append(base)
    seen_names: Set[str] = set()
    for name in names:
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        try:
            _, data = _get_ensembl_json(f"/xrefs/name/{cfg.ensembl_species}/{name}")
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status and 400 <= status < 500 and status != 429:
                continue
            raise EnsemblServiceError(f"Ensembl REST xref lookup failed ({status}): {exc}") from exc
        except Exception as exc:
            raise EnsemblServiceError(f"Ensembl REST xref lookup failed: {exc}") from exc
        rows = data if isinstance(data, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            stable_id = _extract_stable_transcript_id(row.get("id") or row.get("primary_id") or "")
            if not stable_id:
                continue
            candidate = candidate_by_id.get(stable_id.upper())
            if not candidate_by_id or candidate:
                return {
                    "transcript_id": candidate.get("id") if candidate else stable_id,
                    "method": "ensembl_xref",
                    "confidence": "high" if candidate else "medium",
                    "rationale": (
                        f"External accession {accession_text} resolved through Ensembl xrefs"
                        f" to transcript {candidate.get('id') if candidate else stable_id}."
                    ),
                    "xref": row,
                }
    return None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _transcript_cdna_length(transcript: Dict[str, Any]) -> Optional[int]:
    exons = transcript.get("Exon") or []
    total = 0
    for exon in exons:
        start = _int_or_none(exon.get("start"))
        end = _int_or_none(exon.get("end"))
        if start is None or end is None:
            continue
        total += abs(end - start) + 1
    return total or None


def _transcript_coding_bp_length(transcript: Dict[str, Any]) -> Optional[int]:
    translation = transcript.get("Translation") or {}
    coding_start = _int_or_none(translation.get("start"))
    coding_end = _int_or_none(translation.get("end"))
    if coding_start is None or coding_end is None:
        return None
    left = min(coding_start, coding_end)
    right = max(coding_start, coding_end)
    total = 0
    for exon in (transcript.get("Exon") or []):
        exon_start = _int_or_none(exon.get("start"))
        exon_end = _int_or_none(exon.get("end"))
        if exon_start is None or exon_end is None:
            continue
        lo = min(exon_start, exon_end)
        hi = max(exon_start, exon_end)
        overlap_start = max(lo, left)
        overlap_end = min(hi, right)
        if overlap_start <= overlap_end:
            total += overlap_end - overlap_start + 1
    return total or None


def _terminal_exon_metadata(transcript: Dict[str, Any]) -> Dict[str, Any]:
    exons = []
    for exon in (transcript.get("Exon") or []):
        start = _int_or_none(exon.get("start"))
        end = _int_or_none(exon.get("end"))
        if start is None or end is None:
            continue
        lo = min(start, end)
        hi = max(start, end)
        exons.append((lo, hi))
    if not exons:
        return {}
    exons.sort(key=lambda item: (item[0], item[1]))
    strand = _int_or_none(transcript.get("strand")) or 1
    five_prime = exons[0] if strand == 1 else exons[-1]
    three_prime = exons[-1] if strand == 1 else exons[0]
    return {
        "five_prime_terminal_exon_start": five_prime[0],
        "five_prime_terminal_exon_end": five_prime[1],
        "five_prime_terminal_exon_length": five_prime[1] - five_prime[0] + 1,
        "three_prime_terminal_exon_start": three_prime[0],
        "three_prime_terminal_exon_end": three_prime[1],
        "three_prime_terminal_exon_length": three_prime[1] - three_prime[0] + 1,
        "three_prime_terminal_exon_signature": f"{three_prime[0]}-{three_prime[1]}",
    }


def _extract_transcript_rows(gene: Dict) -> List[Dict]:
    from .transcript_metadata import reference_annotation_flags
    rows = []
    for t in (gene.get("Transcript") or []):
        if not t.get("id"):
            continue
        translation = t.get("Translation") or {}
        coding_bp_length = _transcript_coding_bp_length(t)
        protein_length = _int_or_none(translation.get("length"))
        if protein_length is None and coding_bp_length:
            protein_length = max(0, coding_bp_length // 3)
        row = {
            "id": t["id"],
            **_transcript_reference_identity({**{key: gene.get(key) for key in ("species", "assembly_name")}, **t}),
            "biotype": t.get("biotype"),
            "seq_region_name": t.get("seq_region_name"),
            "start": int(t.get("start", 0)),
            "end": int(t.get("end", 0)),
            "strand": int(t.get("strand", 1)),
            "exon_count": len(t.get("Exon") or []),
            "length": (int(t.get("end", 0)) - int(t.get("start", 0)) + 1),
            **reference_annotation_flags(t),
            "display_name": t.get("display_name"),
            "description": t.get("description"),
            "translation_id": translation.get("id"),
            "cdna_length": _transcript_cdna_length(t),
            "coding_bp_length": coding_bp_length,
            "protein_length_aa": protein_length,
        }
        row.update(_terminal_exon_metadata(t))
        rows.append(row)
    rows.sort(key=lambda r: ((r["biotype"] or ""), -r["length"], r["id"]))
    return rows


def _normalize_label(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(text))
    return " ".join(cleaned.split()).strip()


def _fallback_transcript_match(
    transcripts: List[Dict[str, Any]],
    isoform_label: str,
    objective: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    label_norm = _normalize_label(isoform_label).lower()
    tokens = [tok for tok in label_norm.split() if tok]
    objective_text = _normalize_label(json.dumps(objective or {}, ensure_ascii=False)).lower()
    prefer_alt_terminal = any(
        phrase in objective_text
        for phrase in (
            "terminal exon",
            "junction pattern",
            "alternative terminal",
            "alt terminal",
            "3 prime",
            "3p",
        )
    )
    prefer_shorter = any(
        phrase in objective_text
        for phrase in (
            "shorter",
            "truncated",
            "short isoform",
            "short protein",
        )
    )
    canonical_entry = next((tx for tx in transcripts if tx.get("is_canonical")), None)
    if canonical_entry is None:
        canonical_entry = next((tx for tx in transcripts if tx.get("is_mane_select")), None)
    canonical_terminal_signature = (
        str(canonical_entry.get("three_prime_terminal_exon_signature") or "").strip()
        if isinstance(canonical_entry, dict)
        else ""
    )
    best: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[int, int, int, int, int, int, int]] = None
    best_rationale = ""
    for tx in transcripts:
        tid = tx.get("id")
        if not tid:
            continue
        display = _normalize_label(tx.get("display_name") or "").lower()
        desc = _normalize_label(tx.get("description") or "").lower()
        translation_id = _normalize_label(tx.get("translation_id") or "").lower()
        protein_length = _int_or_none(tx.get("protein_length_aa")) or 0
        corpus_parts = [display, desc, translation_id]
        if protein_length:
            corpus_parts.append(f"{protein_length} aa")
            corpus_parts.append(f"{protein_length} amino acids")
        corpus = " ".join(part for part in corpus_parts if part).strip()
        score = 0
        matched_terms: List[str] = []
        signals: List[str] = []
        for tok in tokens:
            if tok and tok in corpus:
                score += 2
                if tok not in matched_terms:
                    matched_terms.append(tok)
        if tokens and tokens[0] and corpus.startswith(tokens[0]):
            score += 1
        if label_norm and label_norm in corpus:
            score += 3
            matched_terms.append(label_norm)
        alt_terminal = 0
        terminal_signature = str(tx.get("three_prime_terminal_exon_signature") or "").strip()
        if prefer_alt_terminal and terminal_signature and canonical_terminal_signature and terminal_signature != canonical_terminal_signature:
            alt_terminal = 1
            score += 3
            signals.append("alternate_3prime_terminal_exon")
        mane_plus = int(bool(tx.get("is_mane_plus_clinical")))
        mane_select = int(bool(tx.get("is_mane_select")))
        canonical = int(bool(tx.get("is_canonical")))
        if mane_plus:
            score += 3
            signals.append("MANE_Plus_Clinical")
        if tx.get("is_mane_select"):
            score += 2
            signals.append("MANE_Select")
        if tx.get("is_canonical"):
            score += 1
            signals.append("canonical")
        length = int(tx.get("length") or 0)
        protein_rank = -protein_length if prefer_shorter and protein_length else protein_length
        candidate_score = (score, alt_terminal, mane_plus, mane_select, canonical, protein_rank, length)
        if best is None or candidate_score > best_score:
            best = tx
            best_score = candidate_score
            reasons: List[str] = []
            if matched_terms:
                reasons.append(f"matched label terms {', '.join(sorted(set(matched_terms))[:4])}")
            if signals:
                reasons.append(", ".join(signals))
            if protein_length:
                reasons.append(f"{protein_length} aa protein")
            if not reasons:
                reasons.append("default transcript priority order")
            best_rationale = f"Fallback local ranking preferred {tid} based on " + "; ".join(reasons) + "."
    if not best:
        return None
    return {
        "transcript_id": best.get("id"),
        "confidence": "low",
        "method": "fallback",
        "rationale": best_rationale,
        "requires_confirmation": True,
    }


_transcript_id_extractor_agent = Agent(
    name="Transcript ID Extractor",
    instructions=(
        "You receive unstructured text (often from a web search snippet) and must pull out a single Ensembl transcript ID.\n"
        "Return strict JSON: {\"transcript_id\": <string|null>}.\n"
        "- Extract one Ensembl transcript stable ID such as ENST###########, ENSMUST###########, or ENSRNOT########### (optionally with a version suffix like .6).\n"
        "- When multiple IDs are present, choose the one that best matches the referenced gene if provided.\n"
        "- If no Ensembl transcript ID is present, set transcript_id to null. Do NOT invent IDs."
    ),
    model="gpt-5.2-2025-12-11",
)


_transcript_confirmation_interpreter_agent = Agent(
    name="Transcript Confirmation Answer Interpreter",
    instructions=(
        "You interpret a user's free-text answer to a transcript-confirmation question.\n"
        "Input JSON: {question, answer, gene_symbol, isoform_label, suggested_transcript_id, suggested_display_name, "
        "suggested_description, method, confidence, rationale, evidence_summary, candidate_transcripts:[{id, display_name?, description?, "
        "is_canonical?, is_mane_select?, is_mane_plus_clinical?, protein_length_aa?, translation_id?, three_prime_terminal_exon_signature?}]}\n"
        "Output strict JSON: {\"decision\": \"accept\"|\"user_selected\"|\"skip\"|\"upload\"|\"clarify\", \"transcript_id\": string|null, \"rationale\": string}.\n"
        "Rules:\n"
        "- Interpret the answer semantically, not literally.\n"
        "- If the user accepts the suggestion, return decision='accept' and transcript_id=suggested_transcript_id.\n"
        "- If the user rejects the suggestion and indicates another transcript, return decision='user_selected' and choose transcript_id only from candidate_transcripts.\n"
        "- Partial references like suffixes, prefixes, transcript nicknames, or 'the one ending in 89' should resolve to a transcript_id only when exactly one candidate fits.\n"
        "- If the user wants to omit the isoform, return decision='skip'.\n"
        "- If the user wants to switch to a SnapGene/file-based reference, return decision='upload'.\n"
        "- If the answer is still ambiguous after using the provided candidates and context, return decision='clarify' with transcript_id=null.\n"
        "- Never invent transcript IDs or choose an ID not present in candidate_transcripts unless candidate_transcripts is empty and the user explicitly gave a full Ensembl transcript ID.\n"
    ),
    model="gpt-5.2-2025-12-11",
    model_settings=JSON_OBJECT_SETTINGS,
)


def _extract_transcript_id_from_text(text: str, gene_symbol: Optional[str] = None) -> Optional[str]:
    if not text or not text.strip():
        return None
    payload = {
        "text": text,
        "gene_symbol": gene_symbol,
    }
    try:
        result = Runner._run_sync(
            _transcript_id_extractor_agent,
            input=json.dumps(payload, ensure_ascii=False),
            max_steps=3,
        )
    except Exception:
        return None
    output = result.final_output
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            return None
    if isinstance(output, dict):
        tid = output.get("transcript_id")
        if isinstance(tid, str) and tid.strip():
            return tid.strip()
    return None


def _fallback_interpret_transcript_confirmation_answer(
    answer: Any,
    suggested_transcript_id: str = "",
) -> Dict[str, Optional[str]]:
    raw_answer = str(answer or "").strip()
    lowered = raw_answer.lower()
    transcript_id = _extract_stable_transcript_id(raw_answer)
    if transcript_id:
        return {
            "decision": "user_selected",
            "transcript_id": transcript_id,
            "rationale": "Explicit transcript ID found in the answer.",
        }
    if not raw_answer or lowered in {"y", "yes", "accept", "accepted", "keep", "default"}:
        transcript_id = str(suggested_transcript_id or "").strip()
        return {
            "decision": "accept",
            "transcript_id": transcript_id.split(".", 1)[0].upper() if transcript_id else None,
            "rationale": "User accepted the suggested transcript.",
        }
    if lowered in {"skip", "omit"}:
        return {"decision": "skip", "transcript_id": None, "rationale": "User chose to skip this isoform."}
    if lowered in {"upload", "snapgene"}:
        return {"decision": "upload", "transcript_id": None, "rationale": "User requested a SnapGene reference."}
    return {"decision": "clarify", "transcript_id": None, "rationale": "Answer remained ambiguous."}


def _answer_accepts_contextual_transcript_override(
    answer: Any,
    question: str,
    suggested_transcript_id: str,
    candidates: List[Dict[str, Any]],
) -> Optional[str]:
    raw_answer = str(answer or "").strip()
    lowered = raw_answer.lower()
    if raw_answer and lowered not in {"y", "yes", "accept", "accepted", "keep", "default", "ok", "okay", "correct"}:
        return None
    suggested_base = _extract_stable_transcript_id(suggested_transcript_id)
    override_ids = [
        tid
        for tid in _extract_stable_transcript_ids(question)
        if not suggested_base or tid != suggested_base
    ]
    if len(override_ids) != 1:
        return None
    override_id = override_ids[0]
    if candidates:
        return _find_transcript_id(candidates, override_id)
    return override_id


def _apply_contextual_transcript_override(
    parsed: Dict[str, Optional[str]],
    *,
    answer: Any,
    question: str,
    suggested_transcript_id: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    if str(parsed.get("decision") or "").strip().lower() != "accept":
        return parsed
    override_id = _answer_accepts_contextual_transcript_override(
        answer,
        question,
        suggested_transcript_id,
        candidates,
    )
    if not override_id:
        return parsed
    updated = dict(parsed)
    updated["decision"] = "user_selected"
    updated["transcript_id"] = override_id
    base_rationale = str(parsed.get("rationale") or "").strip()
    override_note = "User accepted the clarified transcript override."
    updated["rationale"] = f"{base_rationale} {override_note}".strip()
    return updated


_additional_transcript_selections_agent = Agent(
    name="Additional Explicit Transcript Selections",
    instructions=(
        "Read user-authored answers for explicit transcript selections for known_isoforms. "
        "A reply to one confirmation may also select references for other named isoforms. "
        "Extract every affirmative, unambiguous label-to-full-ID selection, including coordinated lists. "
        "Do not infer selections from biological facts, absence, comparison, negation, examples, "
        "assistant questions, or an unscoped yes. Do not choose between alternative IDs. "
        "Keep exact version suffixes. Use only labels in known_isoforms and IDs explicitly in the answer. "
        "Return strict JSON: {selections:[{isoform_label:string,transcript_id:string,"
        "answer_index:integer,evidence_quote:string}]}. answer_index is zero-based. "
        "Each evidence_quote must be an exact contiguous substring of that user's answer containing "
        "the label and ID and the context supporting the selection; include a preceding shared selection "
        "verb when interpreting a coordinated list. Return an empty list when no explicit selection exists. "
        "If repair_errors and previous_response are supplied, repair the extraction against the same answers."
    ),
    model="gpt-5.2-2025-12-11",
    model_settings=JSON_OBJECT_SETTINGS,
    tools=[],
)


def interpret_additional_transcript_selections(
    answers: List[Dict[str, Any]], isoform_labels: List[str],
) -> List[Dict[str, Any]]:
    """Extract additional explicit choices, with verbatim user evidence."""
    identifier = re.compile(r"\b(?:ENS[A-Z]*T\d+(?:\.\d+)?|[NX][MR]_\d+(?:\.\d+)?)\b", re.I)
    mentioned = [label for label in isoform_labels if any(
        re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", str(item.get("answer") or ""), re.I)
        and identifier.search(str(item.get("answer") or "")) for item in answers)]
    if not mentioned:
        return []
    payload: Dict[str, Any] = {"answers": answers, "known_isoforms": sorted(set(mentioned))}
    for _attempt in range(3):
        response = Runner._run_sync(
            _additional_transcript_selections_agent,
            input=json.dumps(payload, ensure_ascii=False), max_steps=3,
        )
        raw = response.final_output if isinstance(response.final_output, str) else ""
        parsed = _parse_json_object_text(raw)
        errors = []
        records: Dict[str, Dict[str, Any]] = {}
        selections = parsed.get("selections") if isinstance(parsed, dict) else None
        if not isinstance(selections, list):
            errors.append("selections must be a list.")
        for item in selections if isinstance(selections, list) else []:
            if not isinstance(item, dict):
                errors.append("Each selection must be an object.")
                continue
            label, tid = item.get("isoform_label"), item.get("transcript_id")
            index, quote = item.get("answer_index"), item.get("evidence_quote")
            if (label not in mentioned or not isinstance(tid, str) or not identifier.fullmatch(tid)
                    or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(answers)
                    or not isinstance(quote, str) or not quote
                    or quote not in str(answers[index].get("answer") or "")
                    or str(label).casefold() not in quote.casefold()
                    or tid not in {match.group() for match in identifier.finditer(quote)}):
                errors.append("Every selection needs a known label and full exact ID in a verbatim user-answer quote.")
                continue
            if label in records and records[label]["transcript_id"] != tid:
                errors.append("Conflicting choices for one label must not be resolved by choosing an ID.")
                continue
            records[label] = {
                "isoform_label": label, "transcript_id": tid, "decision": "user_selected",
                "raw_answer": str(answers[index].get("answer") or ""),
                "method": "explicit_additional_user_selection", "confidence": "user_selected",
                "note": "Explicit user selection: " + quote,
            }
        if not errors:
            return list(records.values())
        payload.update(previous_response=raw, repair_errors=errors)
    raise ValueError("Additional transcript selections could not be bound to explicit user evidence.")


def interpret_transcript_confirmation_answer(
    answer: Any,
    *,
    question: str = "",
    gene_symbol: str = "",
    isoform_label: str = "",
    suggested_transcript_id: str = "",
    suggested_display_name: str = "",
    suggested_description: str = "",
    method: str = "",
    confidence: str = "",
    rationale: str = "",
    evidence_summary: str = "",
    candidate_transcripts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Optional[str]]:
    candidates = candidate_transcripts if isinstance(candidate_transcripts, list) else []
    payload = {
        "question": str(question or "").strip(),
        "answer": str(answer or "").strip(),
        "gene_symbol": str(gene_symbol or "").strip(),
        "isoform_label": str(isoform_label or "").strip(),
        "suggested_transcript_id": str(suggested_transcript_id or "").strip(),
        "suggested_display_name": str(suggested_display_name or "").strip(),
        "suggested_description": str(suggested_description or "").strip(),
        "method": str(method or "").strip(),
        "confidence": str(confidence or "").strip(),
        "rationale": str(rationale or "").strip(),
        "evidence_summary": str(evidence_summary or "").strip(),
        "candidate_transcripts": _sanitize_transcript_records(candidates, max_description=120, max_display=80),
    }
    try:
        result = Runner._run_sync(
            _transcript_confirmation_interpreter_agent,
            input=json.dumps(payload, ensure_ascii=False),
            max_steps=3,
        )
        output_text = result.final_output if isinstance(result.final_output, str) else ""
        parsed = _parse_json_object_text(output_text) or {}
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict) or not parsed:
        fallback = _fallback_interpret_transcript_confirmation_answer(answer, suggested_transcript_id)
        return _apply_contextual_transcript_override(
            fallback,
            answer=answer,
            question=question,
            suggested_transcript_id=suggested_transcript_id,
            candidates=candidates,
        )

    decision = str(parsed.get("decision") or "").strip().lower()
    transcript_id = _extract_stable_transcript_id(parsed.get("transcript_id"))
    if candidates and transcript_id:
        transcript_id = _find_transcript_id(candidates, transcript_id)
    if decision == "accept":
        transcript_id = _find_transcript_id(candidates, suggested_transcript_id) or _extract_stable_transcript_id(
            suggested_transcript_id
        )
    if decision == "user_selected" and not transcript_id:
        return {
            "decision": "clarify",
            "transcript_id": None,
            "rationale": str(parsed.get("rationale") or "").strip() or "Could not map the reply to one transcript.",
        }
    if decision not in {"accept", "user_selected", "skip", "upload", "clarify"}:
        fallback = _fallback_interpret_transcript_confirmation_answer(answer, suggested_transcript_id)
        return _apply_contextual_transcript_override(
            fallback,
            answer=answer,
            question=question,
            suggested_transcript_id=suggested_transcript_id,
            candidates=candidates,
        )
    result_payload = {
        "decision": decision,
        "transcript_id": transcript_id,
        "rationale": str(parsed.get("rationale") or "").strip(),
    }
    return _apply_contextual_transcript_override(
        result_payload,
        answer=answer,
        question=question,
        suggested_transcript_id=suggested_transcript_id,
        candidates=candidates,
    )

# --- PubMed ---

def _pubmed_search(query: str, retmax: int = 10) -> List[Dict]:
    es_params = {"db": "pubmed", "term": query, "retmax": str(retmax), "retmode": "json", "sort": "relevance"}
    try:
        data = _get_json(f"{NCBI_EUTILS}/esearch.fcgi", es_params)
        ids = data.get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []
    if not ids:
        return []
    esum_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    try:
        summ = _get_json(f"{NCBI_EUTILS}/esummary.fcgi", esum_params)
    except Exception:
        return []
    out = []
    for pmid in ids:
        rec = summ.get("result", {}).get(pmid, {})
        title = rec.get("title")
        journal = rec.get("fulljournalname") or rec.get("source")
        pubdate = rec.get("pubdate") or rec.get("sortpubdate")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if title:
            out.append({"pmid": pmid, "title": title, "journal": journal, "pubdate": pubdate, "url": url})
    return out

# --- GTEx helpers ---

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def _gtex_list_tissues_internal() -> List[Dict]:
    params = {"datasetId": "gtex_v8", "itemsPerPage": 100000}
    data = _get_json(f"{GTEX_BASE}/dataset/tissueSiteDetail", params)
    return data.get("data", []) if isinstance(data, dict) else []

# LLM-backed tissue selection agent
_tissue_selector_agent = Agent(
    name="GTEx Tissue Selector",
    instructions=(
        "You choose all GTEx tissues strongly related to a user request.\n"
        "Input: a JSON object with keys {target_context, tissues:[{id,label},...]} where 'id' is tissueSiteDetailId and 'label' is samplingSite.\n"
        "Rules:\n"
        "- Select every tissue that is clearly relevant to the target_context (1-5 entries). If only one is relevant, return a single entry.\n"
        "- Use the user's target_context semantics to decide and prefer specific tissues over general ones when appropriate.\n"
        "- If target_context is empty or ambiguous, pick the most disease-relevant or general tissue likely involved; do not invent new ids.\n"
        "Output: strictly JSON: {\"tissueSiteDetailIds\": [\"<ID FROM LIST>\", ...]}. No extra text."
    ),
    tools=[],
    model="gpt-5-mini-2025-08-07",
)

def _gtex_pick_best_tissue(tissue_hint: str, tissues: List[Dict]) -> List[Dict[str, str]]:
    """Return a ranked list of tissues (id + label) that align with the hint."""
    if not tissues:
        return []

    # Prepare compact input for the LLM
    items: List[Dict[str, str]] = []
    for t in tissues:
        tid = t.get("tissueSiteDetailId") or ""
        lab = t.get("samplingSite") or ""
        if tid:
            items.append({"id": tid, "label": lab})

    payload = {
        "target_context": (tissue_hint or ""),
        "tissues": items,
    }
    try:
        result = Runner._run_sync(
            _tissue_selector_agent,
            input=json.dumps(payload, ensure_ascii=False),
            max_steps=1,
        )
    except Exception:
        result = None

    def _extract_ids(obj: Any) -> List[str]:
        found: List[str] = []
        if isinstance(obj, dict):
            ids = obj.get("tissueSiteDetailIds")
            if isinstance(ids, list):
                for item in ids:
                    if isinstance(item, str) and item.strip():
                        found.append(item.strip())
            single = obj.get("tissueSiteDetailId")
            if isinstance(single, str) and single.strip():
                found.append(single.strip())
        elif isinstance(obj, list):
            for entry in obj:
                found.extend(_extract_ids(entry))
        elif isinstance(obj, str) and obj.strip():
            found.append(obj.strip())
        return found

    choice_ids: List[str] = []
    valid_ids = {item["id"] for item in items}
    choice = result.final_output if result is not None else None
    if isinstance(choice, str):
        try:
            obj = json.loads(choice)
            choice_ids.extend(_extract_ids(obj))
        except Exception:
            for t in items:
                if t["id"] in choice:
                    choice_ids.append(t["id"])
    elif isinstance(choice, dict):
        choice_ids.extend(_extract_ids(choice))

    # Deduplicate while preserving order, clamp to 5
    seen = set()
    normalized_ids: List[str] = []
    for cid in choice_ids:
        base = cid.strip()
        if base and base in valid_ids and base not in seen:
            seen.add(base)
            normalized_ids.append(base)
        if len(normalized_ids) >= 5:
            break

    selected: List[Dict[str, str]] = []
    for tid in normalized_ids:
        for t in tissues:
            if t.get("tissueSiteDetailId") == tid:
                selected.append(
                    {
                        "id": tid,
                        "label": t.get("samplingSite") or t.get("tissueSiteDetailName") or "",
                    }
                )
                break
    return selected

def _gtex_gene_gencode_id(symbol: str) -> Optional[str]:
    try:
        data = _get_json(f"{GTEX_BASE}/reference/features/{symbol}")
        feats = data.get("features", []) if isinstance(data, dict) else []
        # Prefer featureType == 'gene' matching the symbol
        for f in feats:
            if (f.get("featureType") or "").lower() == "gene":
                gc = f.get("gencodeId") or f.get("gencodeIdV26") or f.get("gencodeIdV29")
                if gc:
                    return gc
        # Fallback: first with gencodeId
        for f in feats:
            gc = f.get("gencodeId")
            if gc:
                return gc
    except Exception:
        return None
    return None

# ============================================================
# TOOLS (Agent-facing)
# ============================================================
@function_tool
def lookup_gene_id(symbol: str, species: str = "human") -> Dict:
    """
    Return Ensembl gene ID for a gene symbol without fetching transcript listings.
    """
    species_key = normalize_species(species)
    cfg = get_species_config(species_key)
    gene = _lookup_gene_by_symbol(symbol, species_key, expand=0)
    if not gene:
        return {
            "error": f"Gene {symbol!r} not found for species {species_key!r}.",
            "status": "not_found",
            "species": species_key,
        }
    _assert_human_grch38(cfg.ensembl_species, gene)
    return {
        "gene_symbol": symbol,
        "species": species_key,
        "ensembl_species": cfg.ensembl_species,
        "ucsc_build": cfg.ucsc_build,
        "gene_id": gene.get("id"),
        "resolution_source": gene.get("_resolution_source") or "ensembl_lookup_symbol",
        "matched_symbol_variant": gene.get("_matched_symbol_variant"),
        "display_name": gene.get("display_name") or gene.get("name") or symbol,
        "description": gene.get("description"),
        "assembly": gene.get("assembly_name", cfg.ucsc_build),
    }


@function_tool
def lookup_gene_by_id(gene_id: str) -> Dict:
    """
    Return gene metadata for an Ensembl gene ID.
    """
    gene = _lookup_id(gene_id, expand=0)
    if not gene:
        return {"error": f"Gene ID {gene_id!r} not found.", "status": "not_found"}
    obj_type = str(gene.get("object_type") or "").lower()
    if obj_type and obj_type != "gene":
        return {
            "error": f"Ensembl ID {gene_id!r} is not a gene (object_type={gene.get('object_type')}).",
            "status": "not_gene",
        }
    return {
        "gene_id": gene.get("id") or gene_id,
        "display_name": gene.get("display_name") or gene.get("name"),
        "description": gene.get("description"),
        "species": gene.get("species"),
        "assembly": gene.get("assembly_name"),
        "biotype": gene.get("biotype"),
        "object_type": gene.get("object_type"),
    }


def _ortholog_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _ortholog_symbol_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _ortholog_auto_min_identity() -> float:
    try:
        return float(os.environ.get("ASO_CROSS_SPECIES_ORTHOLOG_AUTO_MIN_IDENTITY", "50"))
    except Exception:
        return 50.0


def resolve_ortholog_gene(gene_id: str, target_species: str) -> Dict[str, Any]:
    """Resolve an Ensembl ortholog by stable gene ID without assuming symbol parity."""
    source_id = str(gene_id or "").split(".", 1)[0].strip()
    species_key = normalize_species(target_species)
    cfg = get_species_config(species_key)
    if not source_id:
        return {
            "status": "error",
            "error": "source_gene_id_missing",
            "target_species": species_key,
        }
    source_metadata = lookup_gene_by_id(source_id)
    if source_metadata.get("error"):
        return {
            "status": "error",
            "error": f"source_gene_metadata_unavailable:{source_metadata.get('error')}",
            "source_gene_id": source_id,
            "target_species": species_key,
        }
    source_species = str(source_metadata.get("species") or "").strip().lower()
    if not source_species:
        return {
            "status": "error",
            "error": "source_gene_species_unavailable",
            "source_gene_id": source_id,
            "target_species": species_key,
        }
    try:
        _, payload = _get_ensembl_json(
            f"/homology/id/{source_species}/{source_id}",
            {
                "target_species": cfg.ensembl_species,
                "type": "orthologues",
            },
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"ortholog_lookup_failed:{exc}",
            "source_gene_id": source_id,
            "target_species": species_key,
        }

    if not isinstance(payload, dict):
        return {
            "status": "error",
            "error": "ortholog_endpoint_returned_no_payload",
            "source_gene_id": source_id,
            "source_species": source_species,
            "target_species": species_key,
        }

    data = payload.get("data") if isinstance(payload, dict) else []
    homologies: List[Dict[str, Any]] = []
    for entry in data or []:
        if not isinstance(entry, dict):
            continue
        for homology in entry.get("homologies") or []:
            if not isinstance(homology, dict):
                continue
            target = homology.get("target") if isinstance(homology.get("target"), dict) else {}
            target_id = str(target.get("id") or "").split(".", 1)[0].strip()
            if target_id:
                homologies.append(homology)
    if not homologies:
        return {
            "status": "no_ortholog",
            "source_gene_id": source_id,
            "target_species": species_key,
            "ensembl_species": cfg.ensembl_species,
            "resolution_source": "ensembl_homology",
        }

    type_rank = {
        "ortholog_one2one": 0,
        "ortholog_one2many": 1,
        "ortholog_many2many": 2,
    }

    def _rank(homology: Dict[str, Any]) -> Tuple[int, float, str]:
        target = homology.get("target") if isinstance(homology.get("target"), dict) else {}
        homology_type = str(homology.get("type") or "").strip().lower()
        try:
            identity = float(target.get("perc_id") or 0.0)
        except Exception:
            identity = 0.0
        return (
            type_rank.get(homology_type, 9),
            -identity,
            str(target.get("id") or ""),
        )

    sorted_homologies = sorted(homologies, key=_rank)
    candidates: List[Dict[str, Any]] = []
    for rank, homology in enumerate(sorted_homologies, start=1):
        target = homology.get("target") if isinstance(homology.get("target"), dict) else {}
        source = homology.get("source") if isinstance(homology.get("source"), dict) else {}
        target_id = str(target.get("id") or "").split(".", 1)[0].strip()
        if not target_id:
            continue
        metadata: Dict[str, Any] = {}
        try:
            maybe_metadata = lookup_gene_by_id(target_id)
            if isinstance(maybe_metadata, dict) and not maybe_metadata.get("error"):
                metadata = maybe_metadata
        except Exception:
            metadata = {}
        candidates.append(
            {
                "rank": rank,
                "species": species_key,
                "ensembl_species": cfg.ensembl_species,
                "gene_id": target_id,
                "gene_symbol": metadata.get("display_name") or target.get("display_id") or target_id,
                "description": metadata.get("description") or target.get("description"),
                "homology_type": homology.get("type"),
                "percent_identity": target.get("perc_id"),
                "source_percent_identity": source.get("perc_id"),
                "source_gene_id": source_id,
                "source_gene_symbol": source_metadata.get("display_name"),
                "target_protein_id": target.get("protein_id"),
                "source_protein_id": source.get("protein_id"),
                "resolution_source": "ensembl_homology",
            }
        )
    if not candidates:
        return {
            "status": "no_ortholog",
            "source_gene_id": source_id,
            "target_species": species_key,
            "ensembl_species": cfg.ensembl_species,
            "resolution_source": "ensembl_homology",
        }

    selected_candidate = candidates[0]
    selected = sorted_homologies[0]
    target = selected.get("target") if isinstance(selected.get("target"), dict) else {}
    target_id = str(selected_candidate.get("gene_id") or "").split(".", 1)[0].strip()
    metadata = lookup_gene_by_id(target_id)
    if metadata.get("error"):
        return {
            "status": "error",
            "error": f"ortholog_metadata_unavailable:{metadata.get('error')}",
            "source_gene_id": source_id,
            "target_gene_id": target_id,
            "target_species": species_key,
        }
    selected_type = str(selected_candidate.get("homology_type") or "").strip().lower()
    selected_identity = _ortholog_float(selected_candidate.get("percent_identity"))
    min_identity = _ortholog_auto_min_identity()
    ambiguity_reasons: List[str] = []
    if selected_type != "ortholog_one2one":
        ambiguity_reasons.append("homology_type_not_one2one")
    if len(candidates) > 1:
        ambiguity_reasons.append("multiple_ortholog_candidates")
    if selected_identity is None or selected_identity < min_identity:
        ambiguity_reasons.append(f"percent_identity_below_{min_identity:g}")
    source_symbol_key = _ortholog_symbol_key(source_metadata.get("display_name"))
    target_symbol_key = _ortholog_symbol_key(selected_candidate.get("gene_symbol"))
    if source_symbol_key and target_symbol_key and source_symbol_key != target_symbol_key and selected_identity is not None and selected_identity < 70:
        ambiguity_reasons.append("symbol_mismatch_low_identity")

    result = {
        "status": "resolved",
        "source_gene_id": source_id,
        "source_gene_symbol": source_metadata.get("display_name"),
        "target_species": species_key,
        "ensembl_species": cfg.ensembl_species,
        "target_gene_id": target_id,
        "target_gene_symbol": metadata.get("display_name") or target.get("display_id"),
        "homology_type": selected.get("type"),
        "percent_identity": target.get("perc_id"),
        "source_percent_identity": (selected.get("source") or {}).get("perc_id") if isinstance(selected.get("source"), dict) else None,
        "resolution_source": "ensembl_homology",
        "ortholog_candidates": candidates,
        "selected_candidate": selected_candidate,
        "requires_clarification": bool(ambiguity_reasons),
        "ambiguous": bool(ambiguity_reasons),
    }
    if ambiguity_reasons:
        result["ambiguity_reasons"] = list(dict.fromkeys(ambiguity_reasons))
        result["confidence"] = "requires_user_confirmation"
    else:
        result["confidence"] = "auto_resolved"
    return result


@function_tool
def list_transcripts(symbol: str, species: str = "human", gene_id: Optional[str] = None) -> Dict:
    """
    Return all Ensembl transcript IDs for a gene with minimal metadata.
    The agent will decide which transcript to use; do NOT enforce MANE/canonical.
    If gene_id is supplied, transcripts are fetched using that Ensembl gene ID.
    """
    species_key = normalize_species(species)
    cfg = get_species_config(species_key)
    gene = None
    if gene_id:
        gene = _lookup_id(gene_id, expand=1)
        if gene:
            obj_type = str(gene.get("object_type") or "").lower()
            if obj_type and obj_type != "gene":
                gene = None
        if not gene:
            return {
                "error": f"Gene ID {gene_id!r} not found for species {species_key!r}.",
                "status": "not_found",
                "species": species_key,
            }
    if not gene:
        gene = _lookup_gene_by_symbol(symbol, species_key, expand=1)
    if not gene:
        return {"error": f"Gene {symbol!r} not found for species {species_key!r}.", "status": "not_found", "species": species_key}
    rows = _extract_transcript_rows(gene)
    return {
        "gene_symbol": gene.get("display_name") or symbol,
        "species": species_key,
        "ensembl_species": cfg.ensembl_species,
        "ucsc_build": cfg.ucsc_build,
        "gene_id": gene.get("id"),
        "assembly": gene.get("assembly_name", cfg.ucsc_build),
        "transcripts": rows,
        "count": len(rows),
    }

@function_tool
def search_pubmed(query: str, retmax: int = 10) -> Dict:
    results = _pubmed_search(query, retmax=retmax)
    return {"query": query, "results": results, "count": len(results)}

@function_tool
def gather_pubmed_for_transcripts(symbol: str, transcript_ids: List[str], retmax: int = 5) -> Dict:
    out = []
    for tx in transcript_ids:
        q1 = f'{symbol} "{tx}" antisense oligonucleotide'
        q2 = f'{symbol} "{tx}" ASO'
        q3 = f'{symbol} antisense oligonucleotide {tx}'
        results = _pubmed_search(q1, retmax=retmax) or _pubmed_search(q2, retmax=retmax) or _pubmed_search(q3, retmax=retmax)
        out.append({"transcript_id": tx, "queries": [q1, q2, q3], "results": results})
        time.sleep(0.34)
    return {"gene_symbol": symbol, "evidence": out}

@function_tool
def gtex_list_tissues() -> Dict:
    """List GTEx v8 tissueSiteDetail entries."""
    tissues = _gtex_list_tissues_internal()
    return {"dataset": "gtex_v8", "count": len(tissues), "tissues": tissues}

@function_tool
def gtex_transcript_expression(structured_goal_json: str) -> Dict:
    """
    Input JSON string with keys at least: { gene_symbol, target_context? }
    - Picks all GTEx tissues matching target_context via an LLM chooser (from the provided GTEx tissue list).
    - Resolves gene_symbol to gencodeId.
    - Returns median transcript expression for the gene across the selected tissues.
    Output:
    {
      gene_symbol,
      gencode_id,
      tissue: first selected tissue (for backwards compatibility),
      tissues: [ { id, label, expressions: [...], count } ],
      expressions: [ { transcript_id, median, tissues: [...] }, ... ],
      count
    }
    """
    try:
        goal = json.loads(structured_goal_json)
    except Exception as e:
        return {"error": f"structured_goal_json is not valid JSON: {e}"}

    species_key = normalize_species(goal.get("species"))
    if species_key != "human":
        return {
            "error": "gtex_unsupported_species",
            "species": species_key,
            "message": "GTEx is human-only; skipping expression lookup.",
        }

    symbol = (goal.get("gene_symbol") or "").strip()
    if not symbol:
        return {"error": "structured_goal.gene_symbol is required"}
    tissue_hint = (goal.get("target_context") or "").strip()

    tissues = _gtex_list_tissues_internal()
    selected_tissues = _gtex_pick_best_tissue(tissue_hint, tissues)
    if not selected_tissues:
        return {"error": "Failed to resolve GTEx tissue via selector."}

    gencode_id = _gtex_gene_gencode_id(symbol)
    if not gencode_id:
        return {"error": f"Failed to resolve gencodeId for symbol {symbol!r}"}

    def _to_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except Exception:
            return None

    combined: Dict[str, Dict[str, Any]] = {}
    tissue_payloads: List[Dict[str, Any]] = []

    for tissue_entry in selected_tissues:
        tissue_id = tissue_entry.get("id")
        tissue_label = tissue_entry.get("label")
        if not tissue_id:
            continue
        params = {
            "gencodeId": [gencode_id],
            "tissueSiteDetailId": [tissue_id],
            "datasetId": "gtex_v8",
            "itemsPerPage": 1000,
            "page": 0,
        }
        try:
            data = _get_json(f"{GTEX_BASE}/expression/medianTranscriptExpression", params)
        except Exception as exc:
            tissue_payloads.append(
                {
                    "id": tissue_id,
                    "label": tissue_label,
                    "expressions": [],
                    "count": 0,
                    "error": str(exc),
                }
            )
            continue

        rows: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            for r in (data.get("data") or []):
                tx = r.get("transcriptId")
                med = r.get("median")
                if tx is None or med is None:
                    continue
                med_float = _to_float(med)
                expression_row = {"transcript_id": tx, "median": med_float if med_float is not None else med}
                rows.append(expression_row)

                entry = combined.setdefault(
                    tx,
                    {
                        "transcript_id": tx,
                        "median": med_float if med_float is not None else med,
                        "tissues": [],
                    },
                )
                entry["tissues"].append(
                    {
                        "id": tissue_id,
                        "label": tissue_label,
                        "median": med_float if med_float is not None else med,
                    }
                )
                if med_float is not None:
                    current = entry.get("median")
                    current_val = current if isinstance(current, (int, float)) else _to_float(current)
                    if current_val is None or med_float > current_val:
                        entry["median"] = med_float

        rows.sort(
            key=lambda x: (
                -(x["median"] if isinstance(x.get("median"), (int, float)) else 0),
                x.get("transcript_id") or "",
            )
        )
        tissue_payloads.append(
            {
                "id": tissue_id,
                "label": tissue_label,
                "expressions": rows,
                "count": len(rows),
            }
        )

    expressions = list(combined.values())
    expressions.sort(
        key=lambda x: (
            -(x["median"] if isinstance(x.get("median"), (int, float)) else 0),
            x.get("transcript_id") or "",
        )
    )

    return {
        "gene_symbol": symbol,
        "gencode_id": gencode_id,
        "tissue": selected_tissues[0] if selected_tissues else None,
        "tissues": tissue_payloads,
        "expressions": expressions,
        "count": len(expressions),
    }


def _transcript_structure_from_payload(tx: Optional[Dict[str, Any]], transcript_id: str) -> Dict[str, Any]:
    from .transcript_metadata import reference_annotation_flags
    if not tx or not tx.get("id") or not tx.get("Exon"):
        return {"error": f"Transcript {transcript_id!r} not found or missing exons."}
    if not _source_identifier_matches(tx, transcript_id):
        return {"error": f"Transcript version {transcript_id!r} is not established by the reference.",
                "requested_transcript_id": transcript_id,
                "reference_identity": _transcript_reference_identity(tx)}

    strand = int(tx.get("strand", 1))
    chrom = tx.get("seq_region_name")
    exons_src = tx["Exon"][:]

    exons_genomic_raw = sorted(exons_src, key=lambda e: int(e.get("start", 0)))
    exons_genomic = [
        {
            "start": int(e.get("start", 0)),
            "end": int(e.get("end", 0)),
            "seq_region_name": e.get("seq_region_name", chrom),
        }
        for e in exons_genomic_raw
    ]

    if strand == 1:
        exons_tx_raw = exons_genomic_raw
    else:
        exons_tx_raw = list(reversed(exons_genomic_raw))

    translation = tx.get("Translation") or {}
    try:
        translation_start_gen = int(translation.get("start")) if translation.get("start") is not None else None
        translation_end_gen = int(translation.get("end")) if translation.get("end") is not None else None
    except Exception:
        translation_start_gen = translation_end_gen = None

    if translation_start_gen is not None and translation_end_gen is not None:
        coding_gen_lo = min(translation_start_gen, translation_end_gen)
        coding_gen_hi = max(translation_start_gen, translation_end_gen)
    else:
        coding_gen_lo = coding_gen_hi = None

    cdna_cursor = 1
    exons_tx: List[Dict[str, Any]] = []
    for idx, exon in enumerate(exons_tx_raw, 1):
        exon_start_gen = int(exon.get("start", 0))
        exon_end_gen = int(exon.get("end", 0))
        length = exon_end_gen - exon_start_gen + 1
        cdna_start = cdna_cursor
        cdna_end = cdna_cursor + max(length, 0) - 1
        cdna_cursor = cdna_end + 1

        coding_start_cdna: Optional[int] = None
        coding_end_cdna: Optional[int] = None
        coding_start_genomic: Optional[int] = None
        coding_end_genomic: Optional[int] = None
        contains_coding = False
        if coding_gen_lo is not None and coding_gen_hi is not None:
            overlap_start_gen = max(exon_start_gen, coding_gen_lo)
            overlap_end_gen = min(exon_end_gen, coding_gen_hi)
            if overlap_start_gen <= overlap_end_gen:
                contains_coding = True
                coding_start_genomic = overlap_start_gen
                coding_end_genomic = overlap_end_gen
                if strand == 1:
                    coding_start_cdna = cdna_start + (overlap_start_gen - exon_start_gen)
                    coding_end_cdna = cdna_start + (overlap_end_gen - exon_start_gen)
                else:
                    coding_start_cdna = cdna_start + (exon_end_gen - overlap_end_gen)
                    coding_end_cdna = cdna_start + (exon_end_gen - overlap_start_gen)

        exons_tx.append(
            {
                "index": idx,
                "start": int(exon.get("start", 0)),
                "end": int(exon.get("end", 0)),
                "seq_region_name": exon.get("seq_region_name", chrom),
                "phase": exon.get("phase"),
                "end_phase": exon.get("end_phase"),
                "cdna_start": cdna_start,
                "cdna_end": cdna_end,
                "coding_start": coding_start_cdna,
                "coding_end": coding_end_cdna,
                "coding_start_genomic": coding_start_genomic,
                "coding_end_genomic": coding_end_genomic,
                "contains_coding": contains_coding,
            }
        )

    exon_count = len(exons_tx)
    if strand == 1:
        tx_to_genomic = {i: (i - 1) for i in range(1, exon_count + 1)}
    else:
        tx_to_genomic = {i: (exon_count - i) for i in range(1, exon_count + 1)}

    return {
        "id": tx.get("id"),
        **_transcript_reference_identity(tx),
        "gene_id": tx.get("Parent") or tx.get("gene_id"),
        "gene_symbol": tx.get("gene_symbol") or tx.get("gene_name"),
        "annotation_features": reference_annotation_flags(tx),
        "seq_region_name": chrom,
        "strand": strand,
        "start": int(tx.get("start", 0)),
        "end": int(tx.get("end", 0)),
        "exon_count": exon_count,
        "exons_tx": exons_tx,
        "exons_genomic": exons_genomic,
        "tx_to_genomic": tx_to_genomic,
    }


def get_gene_transcript_structures(
    symbol: str,
    species: str = "human",
    gene_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return metadata and structures from the configured annotation when available."""
    species_key = normalize_species(species)
    # The bulk path must use the same annotation/provenance as the single-
    # transcript path. A live gene lookup can silently bypass that reference.
    stable_gene_id = str(gene_id or "").strip()
    local_by_id = _lookup_local_gtf_id(stable_gene_id) if stable_gene_id else None
    if (stable_gene_id and stable_gene_id.casefold() == str(symbol or "").strip().casefold()
            and local_by_id is None):
        stable_gene_id = ""
    gene = (local_by_id if stable_gene_id
            else _lookup_gene_by_local_gtf_symbol(symbol, species_key, expand=1))
    if gene and (str(gene.get("object_type") or "").lower() != "gene"
                 or normalize_species(gene.get("species") or species_key) != species_key
                 or (stable_gene_id and not _source_identifier_matches(gene, stable_gene_id))):
        gene = None
    if not gene and stable_gene_id:
        gene = _lookup_id(stable_gene_id, expand=1)
    if not gene:
        gene = _lookup_gene_by_symbol(symbol, species_key, expand=1)
    if not gene:
        return {
            "error": f"Gene {symbol!r} not found for species {species_key!r}.",
            "status": "not_found",
            "species": species_key,
        }

    rows = _extract_transcript_rows(gene)
    structures: Dict[str, Dict[str, Any]] = {}
    for transcript in gene.get("Transcript") or []:
        if not isinstance(transcript, dict):
            continue
        transcript_id = str(transcript.get("id") or "").strip()
        if not transcript_id:
            continue
        reference_transcript = {**{key: gene.get(key) for key in ("species", "assembly_name")}, **transcript}
        structure = _transcript_structure_from_payload(reference_transcript, transcript_id)
        if not structure.get("error"):
            structures[transcript_id] = structure
    return {
        "gene_symbol": gene.get("display_name") or symbol,
        "gene_id": gene.get("id"),
        "species": species_key,
        "transcripts": rows,
        "structures": structures,
        "count": len(rows),
    }


@function_tool
def get_transcript_structure(transcript_id: str, use_reference: Optional[str] = None) -> Dict:
    """Return transcript structure with exons in transcript and genomic order."""
    del use_reference
    # Structure must agree with the configured annotation used by downstream
    # mapping. A live stable-ID lookup can silently change versions between
    # calls, even while the genome and splice predictors remain unchanged.
    # Explicit versions still require an exact reference identity.
    local = _lookup_local_gtf_id(transcript_id)
    if local and _source_identifier_matches(local, transcript_id):
        return _transcript_structure_from_payload(local, transcript_id)
    tx = _lookup_id(transcript_id, expand=1)
    if (not tx or not tx.get("id") or not tx.get("Exon")) and "." in (transcript_id or ""):
        base_id = transcript_id.split(".", 1)[0]
        tx = _lookup_id(base_id, expand=1)
    return _transcript_structure_from_payload(tx, transcript_id)


def _extract_output_text(blocks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    text_parts: List[str] = []
    citations: List[Dict[str, Any]] = []
    for item in blocks:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "output_text":
            text_parts.append(item.get("text", ""))
            citations.extend(item.get("annotations") or [])
        elif kind == "message":
            for segment in item.get("content") or []:
                if isinstance(segment, dict) and segment.get("type") == "output_text":
                    text_parts.append(segment.get("text", ""))
                    citations.extend(segment.get("annotations") or [])
    return "\n".join(part for part in text_parts if part), citations


def _normalize_confidence_label(value: Any, default: str = "low") -> str:
    text = str(value or "").strip().lower()
    if text in {"high", "medium", "low", "none"}:
        return text
    return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_web_evidence(
    evidence: Any,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []

    def _add_item(source: Any, url: Any, note: Any = None, title: Any = None) -> None:
        item: Dict[str, str] = {}
        source_text = str(source or "").strip()
        url_text = str(url or "").strip()
        note_text = str(note or "").strip()
        title_text = str(title or "").strip()
        if source_text:
            item["source"] = source_text
        if url_text:
            item["url"] = url_text
        if title_text:
            item["title"] = title_text
        if note_text:
            item["note"] = note_text
        if not item:
            return
        if item not in normalized:
            normalized.append(item)

    if isinstance(evidence, list):
        for entry in evidence:
            if isinstance(entry, dict):
                _add_item(
                    entry.get("source") or entry.get("domain"),
                    entry.get("url") or entry.get("link"),
                    entry.get("note") or entry.get("summary"),
                    entry.get("title"),
                )
            elif isinstance(entry, str) and entry.strip():
                _add_item(None, None, entry.strip(), None)

    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        url = (
            citation.get("url")
            or citation.get("uri")
            or citation.get("link")
            or ((citation.get("source") or {}).get("url") if isinstance(citation.get("source"), dict) else None)
        )
        title = (
            citation.get("title")
            or ((citation.get("source") or {}).get("title") if isinstance(citation.get("source"), dict) else None)
            or citation.get("text")
        )
        source = (
            citation.get("source_name")
            or citation.get("domain")
            or ((citation.get("source") or {}).get("name") if isinstance(citation.get("source"), dict) else None)
        )
        _add_item(source, url, None, title)
        if len(normalized) >= 4:
            break

    return normalized[:4]


def _evidence_summary(evidence: Any, max_items: int = 2) -> str:
    pieces: List[str] = []
    for entry in _normalize_web_evidence(evidence)[:max_items]:
        source = str(entry.get("source") or entry.get("title") or "source").strip()
        note = str(entry.get("note") or entry.get("title") or "").strip()
        if note and note != source:
            pieces.append(f"{source}: {note}")
        else:
            pieces.append(source)
    return "; ".join(piece for piece in pieces if piece)


def _parse_json_object_text(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


@function_tool
def web_search(query: str) -> Dict:
    """Run a hosted web search for transcript evidence using primary sources."""

    prompt = (
        "Resolve a transcript/isoform request using web search.\n"
        "Prioritize primary sources only: Ensembl, NCBI Gene/RefSeq, UniProt, and HGNC.\n"
        "You may perform up to three focused web searches before answering.\n"
        "If the query provides a candidate transcript list, you MUST choose transcript_id only from that list.\n"
        "When primary sources do not name a candidate ENST directly, use source-backed alias evidence"
        " (for example protein length, transcript/protein accession mapping, MANE annotation, or alternative terminal-exon usage)"
        " to map the alias back onto the provided candidate features.\n"
        "If the evidence is ambiguous or unsupported, return transcript_id=null instead of guessing.\n"
        "Return exactly one JSON object in plain text with keys:\n"
        "{"
        "\"transcript_id\": string|null, "
        "\"confidence\": \"high\"|\"medium\"|\"low\"|\"none\", "
        "\"rationale\": string, "
        "\"requires_confirmation\": boolean, "
        "\"evidence\": [{\"source\": string, \"url\": string, \"note\": string}]"
        "}.\n"
        "Do not wrap the JSON in markdown fences or add extra commentary.\n"
        "When transcript_id is not null, use an Ensembl transcript stable ID (ENST...) without a version suffix.\n\n"
        f"Query context:\n{query}"
    )
    # Reasoning tokens share this budget.  Leave enough headroom for both the
    # hosted search work and the compact, cited JSON answer at high/max effort.
    tokens = 8192
    token_ceiling = 32768
    last_response: Optional[Dict[str, Any]] = None

    try:
        provider_config = get_provider_config()
        hosted_search_tool = build_hosted_web_search_tool(provider_config, max_uses=3)
    except Exception as exc:
        return {"error": f"web_search failed: {exc}"}

    for attempt in range(3):
        payload = {
            "model": "gpt-5.2-2025-12-11",
            "input": prompt,
            "tools": [hosted_search_tool],
            "max_output_tokens": tokens,
            "max_tool_calls": 3,
            "parallel_tool_calls": False,
        }
        try:
            data = create_response(payload, provider_config)
        except Exception as exc:
            return {"error": f"web_search failed: {exc}"}

        last_response = data
        if data.get("status") == "incomplete" and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
            tokens = min(tokens * 2, token_ceiling)
            continue
        if data.get("status") == "incomplete":
            return {
                "query": query,
                "output_text": "",
                "citations": [],
                "raw": data,
                "error": f"web_search incomplete: {data.get('incomplete_details')}",
            }
        break

    if last_response is None:
        return {"error": "web_search failed: empty response"}

    blocks = last_response.get("output") or last_response.get("content") or []
    output_text, citations = _extract_output_text(blocks)
    if not output_text:
        output_text = str(last_response.get("output_text") or "").strip()
    parsed = _parse_json_object_text(output_text) or {}
    extracted_id = None
    if "transcript_id" in parsed:
        extracted_id = _extract_stable_transcript_id(parsed.get("transcript_id"))
    elif output_text:
        extracted_id = _extract_stable_transcript_id(output_text)
    evidence = _normalize_web_evidence(parsed.get("evidence"), citations)
    rationale = str(parsed.get("rationale") or "").strip()
    if not rationale and output_text.strip():
        rationale = output_text.strip().splitlines()[0][:280]

    return {
        "query": query,
        "output_text": output_text.strip(),
        "citations": citations,
        "raw": last_response,
        "extracted_transcript_id": extracted_id,
        "confidence": _normalize_confidence_label(parsed.get("confidence"), "medium" if extracted_id else "none"),
        "rationale": rationale,
        "evidence": evidence,
        "evidence_summary": _evidence_summary(evidence),
        "requires_confirmation": _coerce_bool(parsed.get("requires_confirmation"), default=bool(extracted_id)),
    }


def _sanitize_transcript_records(
    transcripts: List[Dict[str, Any]],
    max_description: int = 160,
    max_display: int = 80,
) -> List[Dict[str, Any]]:
    from .transcript_metadata import reference_annotation_flags
    sanitized: List[Dict[str, Any]] = []
    for entry in transcripts:
        tid = entry.get("id")
        if not tid:
            continue
        sanitized.append(
            {
                "id": tid,
                "display_name": (entry.get("display_name") or "")[:max_display],
                "description": (entry.get("description") or "")[:max_description],
                **reference_annotation_flags(entry),
                "length": entry.get("length"),
                "cdna_length": entry.get("cdna_length"),
                "coding_bp_length": entry.get("coding_bp_length"),
                "protein_length_aa": entry.get("protein_length_aa"),
                "exon_count": entry.get("exon_count"),
                "translation_id": entry.get("translation_id"),
                "three_prime_terminal_exon_start": entry.get("three_prime_terminal_exon_start"),
                "three_prime_terminal_exon_end": entry.get("three_prime_terminal_exon_end"),
                "three_prime_terminal_exon_length": entry.get("three_prime_terminal_exon_length"),
                "three_prime_terminal_exon_signature": entry.get("three_prime_terminal_exon_signature"),
            }
        )
    return sanitized


def _build_candidate_constrained_transcript_query(
    gene_symbol: str,
    isoform_label: str,
    transcripts: List[Dict[str, Any]],
    *,
    objective: Optional[Dict[str, Any]] = None,
    species: str = "human",
) -> str:
    candidate_rows = _sanitize_transcript_records(
        transcripts,
        max_description=120,
        max_display=80,
    )
    objective_payload = objective if isinstance(objective, dict) else {}
    lines = [
        "Resolve the isoform label to one transcript from the candidate list only.",
        "Use primary-source web evidence to infer isoform-specific traits, then map those traits back to the candidate features below.",
        "Do not rely on direct string overlap alone. Labels like protein aliases may correspond via protein length, MANE/RefSeq/UniProt mappings, or alternative 3' terminal exon usage.",
        f"Gene: {gene_symbol or 'unknown'}",
        f"Species: {species or 'unknown'}",
        f"Isoform label: {isoform_label or 'unknown'}",
    ]
    if objective_payload:
        lines.append(f"Objective hints: {json.dumps(objective_payload, ensure_ascii=False)}")
    lines.append(
        "Candidate feature notes: length=genomic span bp; cdna_length=spliced transcript length bp; "
        "protein_length_aa=predicted protein length; three_prime_terminal_exon_signature=genomic span of the 3' terminal exon."
    )
    lines.append(
        "Candidate transcripts (you may choose only from these IDs; if none are defensible, return transcript_id=null):"
    )
    lines.append(json.dumps(candidate_rows, ensure_ascii=False))
    return "\n".join(lines)


def _attempt_transcript_via_web_search(
    gene_symbol: str,
    isoform_label: str,
    transcripts: List[Dict[str, Any]],
    *,
    objective: Optional[Dict[str, Any]] = None,
    species: str = "human",
) -> Optional[Dict[str, Any]]:
    try:
        search_available = provider_credentials_available()
    except Exception:
        search_available = False
    if not search_available:
        return None
    query_text = _build_candidate_constrained_transcript_query(
        gene_symbol,
        isoform_label,
        transcripts,
        objective=objective,
        species=species,
    )
    try:
        result = web_search(query_text)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    extracted = result.get("extracted_transcript_id")
    if not extracted:
        return None
    matched_id = _find_transcript_id(transcripts, extracted)
    if not matched_id:
        return None
    rationale_line = str(result.get("rationale") or "").strip()
    evidence_summary = str(result.get("evidence_summary") or "").strip()
    evidence_refs = list(
        dict.fromkeys(
            str(entry.get("url") or "").strip()
            for entry in result.get("evidence") or []
            if isinstance(entry, dict) and str(entry.get("url") or "").strip()
        )
    )
    if evidence_summary:
        if rationale_line:
            rationale_line = f"{rationale_line} Sources: {evidence_summary}"
        else:
            rationale_line = f"Matched via primary-source web search. Sources: {evidence_summary}"
    if not rationale_line:
        rationale_line = "Matched via primary-source candidate-constrained web search."
    return {
        "transcript_id": matched_id,
        "confidence": _normalize_confidence_label(result.get("confidence"), "medium"),
        "method": "web_search_primary",
        "rationale": rationale_line,
        "evidence_summary": evidence_summary or None,
        "evidence_refs": evidence_refs,
        "requires_confirmation": True,
    }


_isoform_mapper_agent = Agent(
    name="Isoform Transcript Mapper",
    instructions=(
        "You map an isoform nickname to exactly one transcript ID from the provided list.\n"
        "If the isoform label already matches a transcript ID in the list, return it directly without calling tools.\n"
        "When you must call web_search, include the candidate transcript list in the query and ask it to use primary sources only.\n"
        "Use the provided candidate metadata aggressively: display_name/description, MANE flags, protein_length_aa, translation_id, and 3' terminal exon features.\n"
        "If external sources describe an alias indirectly, map that evidence back onto the candidate features instead of relying only on direct text overlap.\n"
        "If web_search fails or the evidence is ambiguous, fall back to the provided transcripts and choose the best match deterministically.\n"
        "The web_search tool response includes 'extracted_transcript_id', 'confidence', 'rationale', and 'requires_confirmation'. Prefer those fields after verifying the ID appears in the provided transcript list.\n"
        "Input JSON: {gene_symbol, isoform_label, transcripts:[{id, display_name?, description?, is_canonical?, is_mane_select?, is_mane_plus_clinical?, protein_length_aa?, translation_id?, three_prime_terminal_exon_signature?}, ...], objective?, species?}.\n"
        "Return strict JSON: {\"transcript_id\": string|null, \"confidence\": \"high\"|\"medium\"|\"low\"|\"none\", \"rationale\": string, \"requires_confirmation\": boolean, \"method\": \"llm\"|\"llm_web_search\"|\"fallback\"}.\n"
        "You must always return a transcript_id from the provided list when transcripts are supplied, even if confidence is low.\n"
        "If you cannot resolve the label, choose the best available transcript deterministically by strongest candidate-specific evidence first, then MANE Plus Clinical, then MANE Select, then canonical, then longest, and mark requires_confirmation=true.\n"
        "Never invent IDs or return transcript_id=null unless the transcript list is empty."
    ),
    tools=[web_search],
    model="gpt-5.2-2025-12-11",
    log_tool_calls=True,
)


def auto_match_isoform_transcript(
    gene_symbol: str,
    isoform_label: str,
    transcripts: List[Dict],
    objective: Optional[Dict] = None,
    species: str = "human",
) -> Dict[str, Any]:
    species_key = normalize_species(species)
    if not isoform_label or not transcripts:
        return {
            "transcript_id": None,
            "confidence": "none",
            "method": None,
            "rationale": "Missing isoform label or transcripts.",
            "requires_confirmation": True,
        }

    # If the label already matches an ID in the transcript list, return it immediately
    label_norm = isoform_label.strip()
    direct_match = _find_transcript_id(transcripts, label_norm)
    if direct_match:
        return {
            "transcript_id": direct_match,
            "confidence": "high",
            "method": "direct_id",
            "rationale": "Isoform label matches a transcript ID provided by the user.",
            "requires_confirmation": False,
        }

    # Optional local test overrides. Disabled by default.
    if ENABLE_TEST_ISOFORM_OVERRIDES:
        gene_key = (gene_symbol or "").upper()
        label_key = label_norm.lower()
        override_id = TEST_ISOFORM_OVERRIDES.get(gene_key, {}).get(label_key)
        if override_id:
            matched_override = _find_transcript_id(transcripts, override_id)
            if matched_override:
                return {
                    "transcript_id": matched_override,
                    "confidence": "high",
                    "method": "override",
                    "rationale": "Test override matched to known transcript without web search.",
                    "requires_confirmation": True,
                }

    deterministic_match = _attempt_transcript_via_web_search(
        gene_symbol,
        isoform_label,
        transcripts,
        objective=objective if isinstance(objective, dict) else None,
        species=species_key,
    )
    if deterministic_match:
        return deterministic_match

    payload_transcripts = _sanitize_transcript_records(transcripts)

    payload = {
        "gene_symbol": gene_symbol,
        "isoform_label": isoform_label,
        "transcripts": payload_transcripts,
        "objective": objective or {},
        "species": species_key,
    }

    if not payload_transcripts:
        return {
            "transcript_id": None,
            "confidence": "none",
            "method": None,
            "rationale": "Transcript list is empty.",
            "requires_confirmation": True,
        }

    try:
        result = Runner._run_sync(_isoform_mapper_agent, input=json.dumps(payload, ensure_ascii=False), max_steps=4)
    except Exception as exc:
        fallback_match = _fallback_transcript_match(
            transcripts,
            isoform_label,
            objective=objective if isinstance(objective, dict) else None,
        )
        if fallback_match:
            fallback_match["rationale"] = (
                f"Isoform mapper failed: {exc}. " + str(fallback_match.get("rationale") or "").strip()
            ).strip()
            return fallback_match
        return {
            "transcript_id": None,
            "confidence": "none",
            "method": None,
            "rationale": f"Isoform mapper failed: {exc}",
            "requires_confirmation": True,
        }

    output = result.final_output
    raw_output = output.strip() if isinstance(output, str) else ""
    parsed_output = _parse_json_object_text(raw_output) if raw_output else None
    transcript_id = None
    confidence = None
    rationale = None
    method = "llm"
    requires_confirmation = True
    if isinstance(parsed_output, dict):
        transcript_id = parsed_output.get("transcript_id")
        confidence = _normalize_confidence_label(parsed_output.get("confidence"), "low")
        rationale = str(parsed_output.get("rationale") or "").strip() or None
        parsed_method = str(parsed_output.get("method") or "").strip().lower()
        if parsed_method in {"llm", "llm_web_search", "fallback"}:
            method = parsed_method
        requires_confirmation = _coerce_bool(parsed_output.get("requires_confirmation"), default=True)
    if transcript_id is None and raw_output and parsed_output is None:
        match = re.findall(r"ENST\\d+(?:\\.\\d+)?", raw_output, flags=re.IGNORECASE)
        if match:
            transcript_id = match[0]

    matched_id = _find_transcript_id(transcripts, transcript_id) if transcript_id else None
    if not matched_id:
        fallback_match = _fallback_transcript_match(
            transcripts,
            isoform_label,
            objective=objective if isinstance(objective, dict) else None,
        )
        if fallback_match:
            fallback_match["rationale"] = (
                "No valid transcript from mapper. " + str(fallback_match.get("rationale") or "").strip()
            ).strip()
            return fallback_match
        return {
            "transcript_id": None,
            "confidence": "none",
            "method": None,
            "rationale": "Transcript list is empty.",
            "requires_confirmation": True,
        }

    return {
        "transcript_id": matched_id,
        "confidence": confidence or "low",
        "method": method or "llm",
        "rationale": rationale or "Selected transcript from mapper output.",
        "requires_confirmation": requires_confirmation,
    }

@function_tool
def build_bed6(transcript_id: str, structured_goal_json: str) -> Dict:
    """
    structured_goal_json: JSON string with keys like
      {gene_symbol, species, intent, desired_effect?, exon_label?, flanks?, region_flanks?}
    Returns:
      {
        bed6: [[chr,start0,end,name,score,strand], ...],
        note: str,
        transcript_span: [tx_start, tx_end]
      }
    """
    try:
        structured_goal = json.loads(structured_goal_json)
    except Exception as e:
        return {"error": f"structured_goal_json is not valid JSON: {e}"}

    tx = get_transcript_structure(transcript_id)
    if tx.get("error"):
        return tx

    chrom = tx["seq_region_name"]
    strand_val = int(tx["strand"])
    strand = "-" if strand_val == -1 else "+"
    ex_tx = tx["exons_tx"]          # transcript order
    ex_g  = tx["exons_genomic"]     # genomic order
    to_g  = tx["tx_to_genomic"]
    exon_count = tx["exon_count"]

    goal_intent = (structured_goal.get("intent") or "").lower()
    exon_label = structured_goal.get("exon_label")
    flanks = structured_goal.get("flanks") or structured_goal.get("region_flanks")  # accept either key

    bed_rows: List[List] = []
    name_prefix = f"{structured_goal.get('gene_symbol','GENE')}|{transcript_id}"

    def bed6_row(start_bp: int, end_bp: int, label: str) -> List:
        # BED: 0-based start, 1-based end (half-open)
        start0 = max(0, start_bp - 1)
        return [chrom, start0, end_bp, f"{name_prefix}|{label}", 0, strand]

    if goal_intent == "splice_modulation":
        if not exon_label:
            return {"error": "structured_goal.exon_label is required for splice_modulation"}
        try:
            target_idx = int(str(exon_label))  # 1-based transcript-index
        except Exception:
            return {"error": f"Invalid exon_label: {exon_label!r}"}
        if target_idx < 1 or target_idx > exon_count:
            return {"error": f"exon_label {target_idx} out of range (1..{exon_count})"}

        exon = ex_tx[target_idx - 1]
        g_idx = to_g[target_idx]  # 0-based genomic index

        if isinstance(flanks, dict):
            up = int(flanks.get("up", 0))
            down = int(flanks.get("down", 0))
            if strand == "+":
                start_bp = exon["start"] - up
                end_bp   = exon["end"] + down
            else:
                # upstream in transcript space is to the RIGHT genomically
                start_bp = exon["start"] - down
                end_bp   = exon["end"] + up
            bed_rows.append(bed6_row(start_bp, end_bp, f"exon{target_idx}_±{up}/{down}"))
            note = "splice_modulation with numeric flanks"
        else:
            # Default: entire introns on both sides - use GENOMIC neighbors
            if g_idx == 0:
                left = tx["start"]
            else:
                left = ex_g[g_idx - 1]["end"] + 1
            if g_idx == len(ex_g) - 1:
                right = tx["end"]
            else:
                right = ex_g[g_idx + 1]["start"] - 1
            # Ensure the exon body is included (edge cases when exons abut)
            left  = min(left, exon["start"])
            right = max(right, exon["end"])
            # Guard against pathological inversions
            if left > right:
                return {"error": f"Computed intronic span is invalid: left({left}) > right({right})"}
            bed_rows.append(bed6_row(left, right, f"exon{target_idx}_with_introns"))
            note = "splice_modulation with entire introns (default)"

    elif goal_intent == "knockdown":
        # Exon-only tiling: one BED row per exon (mature mRNA parts)
        for ex in ex_tx:
            bed_rows.append(bed6_row(ex["start"], ex["end"], "exon_only"))
        note = "knockdown: exon-only tiling (one row per exon)"
    else:
        return {"error": f"Unknown or missing intent: {goal_intent!r}"}

    # Final sanity: starts <= ends
    for r in bed_rows:
        if r[1] > r[2]:
            return {"error": f"BED row has start > end: {r}"}

    return {"bed6": bed_rows, "note": note, "transcript_span": [tx["start"], tx["end"]]}


# ============================================================
# Conserved exon helper
# ============================================================
@function_tool
def conserved_exons_for_transcript(
    gene_symbol: str,
    transcript_id: str,
    species: str = "human",
    top_n: int = 5,
    target_context: Optional[str] = None,
    tissue_site_detail_ids: Optional[List[str]] = None,
) -> Dict:
    """Identify exons of transcript_id that are most conserved (shared) across isoforms for the gene, optionally weighted by GTEx transcript expression in a target tissue.

    Returns
    {
      gene_symbol,
      transcript_id,
      total_transcripts,
      top_exons: [ {exon_index, start, end, frequency, fraction, expression_fraction?} ... ],
      exon_frequency_map: { "exon_index": fraction, ... },
      exon_expression_fraction_map: { "exon_index": expression_fraction, ... }  # when expression data available
    }
    """
    gene_symbol = (gene_symbol or "").strip()
    if not gene_symbol:
        return {"error": "gene_symbol is required"}

    transcript_id = (transcript_id or "").strip()
    if not transcript_id:
        return {"error": "transcript_id is required"}

    tx_struct = get_transcript_structure(transcript_id)
    if tx_struct.get("error"):
        return tx_struct

    exons_tx: List[Dict[str, int]] = tx_struct.get("exons_tx") or []
    if not exons_tx:
        return {"error": f"Transcript {transcript_id} has no exon annotations"}

    species_key = normalize_species(species)
    gene = _lookup_gene_by_symbol(gene_symbol, species_key, expand=1)
    if not gene or not gene.get("Transcript"):
        return {"error": f"Unable to fetch transcripts for {gene_symbol!r}"}

    transcripts = [t for t in (gene.get("Transcript") or []) if t.get("id")]
    total_transcripts = len(transcripts)
    if total_transcripts == 0:
        return {"error": f"No transcripts found for gene {gene_symbol!r}"}

    target_base = tx_struct.get("id", transcript_id).split(".", 1)[0]

    # Identify canonical transcript spans so only canonical exons are considered.
    canonical_spans: set[Tuple[int, int]] = set()
    try:
        canonical_tx = next((t for t in transcripts if t.get("is_canonical")), None)
    except Exception:
        canonical_tx = None
    if canonical_tx and canonical_tx.get("Exon"):
        for ex in canonical_tx.get("Exon"):
            try:
                span = (int(ex.get("start")), int(ex.get("end")))
            except Exception:
                continue
            canonical_spans.add(span)
    if canonical_spans:
        total_transcripts = len(transcripts)
    else:
        print("[warn] Canonical transcript not found or missing exons; falling back to all exons.")

    # Map exon genomic span -> index (1-based) for target transcript
    span_to_index: Dict[Tuple[int, int], int] = {}
    exon_frequency = defaultdict(int)
    for exon in exons_tx:
        idx = int(exon.get("index", 0))
        start_bp = int(exon.get("start"))
        end_bp = int(exon.get("end"))
        span = (start_bp, end_bp)
        if canonical_spans and span not in canonical_spans:
            continue
        span_to_index[span] = idx
        exon_frequency[idx] = 1  # count the target transcript itself

    # Count occurrences across other transcripts (unique per transcript)
    for t in transcripts:
        tid = t.get("id") or ""
        base_tid = tid.split(".", 1)[0]
        if base_tid == target_base:
            continue
        seen: set = set()
        for ex in t.get("Exon") or []:
            try:
                span = (int(ex.get("start")), int(ex.get("end")))
            except Exception:
                continue
            if canonical_spans and span not in canonical_spans:
                continue
            if span in span_to_index and span not in seen:
                exon_frequency[span_to_index[span]] += 1
                seen.add(span)

    # Optional expression weighting (GTEx)
    expression_weights: Dict[str, float] = {}
    expression_tissues: List[Dict[str, Any]] = []
    selected_tissues: List[Dict[str, Any]] = []

    if tissue_site_detail_ids:
        tissue_lookup = {}
        for entry in _gtex_list_tissues_internal():
            tid = entry.get("tissueSiteDetailId")
            if tid:
                tissue_lookup[tid] = entry.get("samplingSite")
        for tid in tissue_site_detail_ids:
            if not tid:
                continue
            selected_tissues.append({"id": tid, "label": tissue_lookup.get(tid)})
    else:
        selected_tissues = _gtex_pick_best_tissue(target_context or "", _gtex_list_tissues_internal())

    if selected_tissues:
        gencode_id = _gtex_gene_gencode_id(gene_symbol)

        def _to_float(val: Any) -> Optional[float]:
            try:
                return float(val)
            except Exception:
                return None

        if gencode_id:
            for tissue in selected_tissues:
                tissue_id = tissue.get("id")
                if not tissue_id:
                    continue
                params = {
                    "gencodeId": [gencode_id],
                    "tissueSiteDetailId": [tissue_id],
                    "datasetId": "gtex_v8",
                    "itemsPerPage": 1000,
                    "page": 0,
                }
                try:
                    data = _get_json(f"{GTEX_BASE}/expression/medianTranscriptExpression", params)
                except Exception as exc:
                    expression_tissues.append({"id": tissue_id, "label": tissue.get("label"), "error": str(exc)})
                    continue
                for row in (data.get("data") or []) if isinstance(data, dict) else []:
                    tx_id = row.get("transcriptId")
                    med = _to_float(row.get("median"))
                    if not tx_id or med is None:
                        continue
                    base_id = tx_id.split(".", 1)[0]
                    current = expression_weights.get(base_id)
                    if current is None or med > current:
                        expression_weights[base_id] = med
                expression_tissues.append({"id": tissue_id, "label": tissue.get("label")})

    entries: List[Dict[str, float]] = []
    exon_expr_weights: Dict[int, float] = {}
    total_expr_weight = 0.0

    def _weight_for_transcript(tid: str) -> Optional[float]:
        if not expression_weights:
            return None
        base = (tid or "").split(".", 1)[0]
        return expression_weights.get(base)

    if expression_weights:
        target_weight = _weight_for_transcript(tx_struct.get("id", transcript_id))
        for exon in exons_tx:
            idx = int(exon.get("index", 0))
            if target_weight is not None:
                exon_expr_weights[idx] = target_weight
                total_expr_weight += target_weight

        for t in transcripts:
            tid = t.get("id") or ""
            weight = _weight_for_transcript(tid)
            if weight is None or weight <= 0:
                continue
            seen_spans: set = set()
            for ex in t.get("Exon") or []:
                try:
                    span = (int(ex.get("start")), int(ex.get("end")))
                except Exception:
                    continue
                if span in span_to_index and span not in seen_spans:
                    idx = span_to_index[span]
                    exon_expr_weights[idx] = exon_expr_weights.get(idx, 0.0) + weight
                    total_expr_weight += weight
                    seen_spans.add(span)

    for exon in exons_tx:
        idx = int(exon.get("index", 0))
        start_bp = int(exon.get("start"))
        end_bp = int(exon.get("end"))
        freq = exon_frequency.get(idx, 0)
        fraction = float(freq) / float(total_transcripts) if total_transcripts else 0.0
        expr_weight = exon_expr_weights.get(idx)
        expr_fraction = None
        if expression_weights and total_expr_weight > 0 and expr_weight is not None:
            expr_fraction = float(expr_weight) / float(total_expr_weight)
        entries.append({
            "exon_index": idx,
            "start": start_bp,
            "end": end_bp,
            "frequency": freq,
            "fraction": fraction,
            "expression_fraction": expr_fraction,
        })

    entries.sort(
        key=lambda x: (
            -(x["expression_fraction"] if isinstance(x.get("expression_fraction"), (int, float)) else -1),
            -x["frequency"],
            x["exon_index"],
        )
    )
    top = entries[:top_n] if top_n and top_n > 0 else entries

    exon_freq_map = {str(item["exon_index"]): item["fraction"] for item in entries}
    exon_expr_fraction_map = {
        str(item["exon_index"]): item["expression_fraction"]
        for item in entries
        if item.get("expression_fraction") is not None
    }

    return {
        "gene_symbol": gene_symbol,
        "transcript_id": tx_struct.get("id", transcript_id),
        "species": species_key,
        "total_transcripts": total_transcripts,
        "top_exons": top,
        "exon_frequency_map": exon_freq_map,
        "exon_expression_fraction_map": exon_expr_fraction_map,
        "expression_tissues": expression_tissues,
        "expression_weighted": bool(exon_expr_fraction_map),
    }
# ============================================================
# Agent Instance
# ============================================================
transcript_chooser_agent = Agent(
    name="ASO Transcript Chooser",
    instructions=(
        "Task: Given a gene and a structured user goal, PROPOSE transcript options without building BED.\n"
        "If structured_goal.transcript_policy == 'user_locked', reflect the locked transcript as 'locked' and still propose the canonical transcript.\n"
        "For explicit user-locked transcripts (kind in {'explicit_id','reference_accession'}): VERIFY PubMed evidence using tools.\n"
        "- First, try gather_pubmed_for_transcripts(symbol, [transcript_id]).\n"
        "- If zero results, try search_pubmed on the gene (e.g., '<symbol> antisense oligonucleotide').\n"
        "- Include a short note with evidence counts; do not ask the user anything further.\n"
        "Otherwise:\n"
        "  - Call list_transcripts to enumerate candidates.\n"
        "  - Call gtex_transcript_expression with the structured goal ONLY when species is human; skip GTEx for non-human species.\n"
        "  - Optionally, consult PubMed (search_pubmed or gather_pubmed_for_transcripts) for disease/context hints.\n"
        "  - Choose one recommended transcript based on the user's context (do NOT default to MANE/canonical unless justified).\n"
        "  - Each option MUST include a clear rationale explaining why the isoform fits the request (expression, disease evidence, localization, etc.).\n"
        "  - For knockdown requests, identify exons shared across most isoforms (call conserved_exons_for_transcript; GTEx weighting applies only for human) and summarize them.\n"
        "Also identify the canonical transcript from list_transcripts (is_canonical==True) and explain why it matters.\n"
        "OUTPUT STRICT JSON ONLY:\n"
        "{\n"
        "  \"options\": {\n"
        "    \"recommended\": { \"transcript_id\": \"ENST...\", \"rationale\": \"...\" },\n"
        "    \"canonical\":   { \"transcript_id\": \"ENST...\", \"rationale\": \"...\" }\n"
        "  },\n"
        "  \"notes\": \"short note\"\n"
        "}\n"
        "No extra commentary. Do not output BED here."
    ),
    tools=[list_transcripts, gtex_list_tissues, gtex_transcript_expression, search_pubmed, gather_pubmed_for_transcripts, get_transcript_structure, build_bed6, conserved_exons_for_transcript],
    model="gpt-5-mini-2025-08-07",
)
