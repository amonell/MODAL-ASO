from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from urllib.parse import quote


from .paths import reference_root

DEFAULT_DATA_ROOT = reference_root()
DATA_ROOT = Path(os.environ.get("ASO_REFERENCE_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()
ENSEMBL_REST = os.environ.get("ENSEMBL_REST", "https://rest.ensembl.org").rstrip("/")
USER_AGENT = os.environ.get("HTTP_USER_AGENT", "aso-agent/0.1 (+https://example.org)")


@dataclass(frozen=True)
class SpeciesConfig:
    """Filesystem and identifier metadata for a supported species."""

    key: str
    label: str
    ensembl_species: str
    ensembl_assembly: Optional[str]
    ucsc_build: str
    genome_fasta: Path
    genome_fasta_gz: Optional[Path]
    gtf_path: Path
    chrom_sizes_path: Path
    phastcons_path: Path
    phylop_path: Path
    bowtie_prefix: Path
    refseq_xref_path: Optional[Path] = None
    conservation_mode: str = "available"
    reference_source: str = "static"


def _build_static_species_data(data_root: Path) -> Dict[str, SpeciesConfig]:
    root = Path(data_root).expanduser()
    return {
        "human": SpeciesConfig(
            key="human",
            label="Human (GRCh38/hg38)",
            ensembl_species="homo_sapiens",
            ensembl_assembly="GRCh38",
            ucsc_build="hg38",
            genome_fasta=root / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
            genome_fasta_gz=root / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fna.gz",
            gtf_path=root / "gencode.v42.annotation.gtf",
            chrom_sizes_path=root / "hg38.chrom.sizes",
            phastcons_path=root / "hg38.phastCons30way.bw",
            phylop_path=root / "hg38.phyloP30way.bw",
            bowtie_prefix=root / "bowtie" / "hg38",
            refseq_xref_path=root / "gencode.v42.metadata.RefSeq.gz",
        ),
        "mouse": SpeciesConfig(
            key="mouse",
            label="Mouse (GRCm39/mm39)",
            ensembl_species="mus_musculus",
            ensembl_assembly="GRCm39",
            ucsc_build="mm39",
            genome_fasta=root / "GRCm39.primary_assembly.genome.fa",
            genome_fasta_gz=root / "GRCm39.primary_assembly.genome.fa.gz",
            gtf_path=root / "gencode.vM35.primary_assembly.annotation.gtf",
            chrom_sizes_path=root / "mm39.chrom.sizes",
            phastcons_path=root / "mm39.phastCons35way.bw",
            phylop_path=root / "mm39.phyloP35way.bw",
            bowtie_prefix=root / "bowtie" / "mm39",
            refseq_xref_path=root / "gencode.vM35.metadata.RefSeq.gz",
        ),
    }


_SPECIES_DATA: Dict[str, SpeciesConfig] = _build_static_species_data(DATA_ROOT)

_SPECIES_ALIASES: Dict[str, str] = {
    "human": "human",
    "homo_sapiens": "human",
    "homo sapiens": "human",
    "grch38": "human",
    "hg38": "human",
    "mouse": "mouse",
    "mus musculus": "mouse",
    "mus_musculus": "mouse",
    "grcm39": "mouse",
    "mm39": "mouse",
    "rat": "rattus_norvegicus",
    "rattus": "rattus_norvegicus",
    "rattus norvegicus": "rattus_norvegicus",
    "rattus_norvegicus": "rattus_norvegicus",
    "rnor": "rattus_norvegicus",
    "cyno": "macaca_fascicularis",
    "cynomolgus": "macaca_fascicularis",
    "cynomolgus macaque": "macaca_fascicularis",
    "macaca fascicularis": "macaca_fascicularis",
    "macaca_fascicularis": "macaca_fascicularis",
    "crab eating macaque": "macaca_fascicularis",
    "crab-eating macaque": "macaca_fascicularis",
}

_REQUESTED_SPECIES_ALIASES: Dict[str, str] = dict(_SPECIES_ALIASES)
_ENSEMBL_SPECIES_CACHE: Optional[List[Dict[str, Any]]] = None
_ENSEMBL_SPECIES_CACHE_AT = 0.0


def _slug_species(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _primary_species_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return next((item for item in value if item not in (None, "")), None)
    if isinstance(value, set):
        return next((item for item in sorted(value, key=str) if item not in (None, "")), None)
    return value


def _composite_species_alias(value: str) -> Optional[str]:
    """Resolve labels composed only of aliases for the same supported species."""
    tokens = tuple(token for token in _slug_species(value).split("_") if token)
    if not tokens:
        return None
    alias_tokens = [
        (tuple(token for token in _slug_species(alias).split("_") if token), species_key)
        for alias, species_key in _SPECIES_ALIASES.items()
        if alias
    ]
    states: set[tuple[int, Optional[str]]] = {(0, None)}
    while states:
        index, selected_species = states.pop()
        if index == len(tokens):
            return selected_species
        for candidate_tokens, species_key in alias_tokens:
            if not candidate_tokens:
                continue
            end = index + len(candidate_tokens)
            if end > len(tokens) or tokens[index:end] != candidate_tokens:
                continue
            if selected_species is not None and selected_species != species_key:
                continue
            states.add((end, selected_species or species_key))
    return None


def normalize_species(value: Any) -> str:
    """Return canonical species key, defaulting only unspecified requests to human."""
    value = _primary_species_value(value)
    if not value:
        return "human"
    text = str(value).strip().lower()
    key = _SPECIES_ALIASES.get(text)
    if key:
        return key
    composite = _composite_species_alias(text)
    if composite:
        return composite
    return _slug_species(text) or "human"


def try_normalize_species(value: Any) -> Optional[str]:
    """Return a canonical species key, preserving explicit Ensembl-style organisms."""
    value = _primary_species_value(value)
    if not value:
        return "human"
    text = str(value).strip().lower()
    key = _SPECIES_ALIASES.get(text)
    if key:
        return key
    composite = _composite_species_alias(text)
    if composite:
        return composite
    return _slug_species(text) or None


def normalize_requested_species(value: Any) -> str:
    """Normalize species names for request requirements without defaulting unsupported species to human."""
    value = _primary_species_value(value)
    if not value:
        return "human"
    text = str(value).strip().lower()
    supported = _SPECIES_ALIASES.get(text)
    if supported:
        return supported
    alias = _REQUESTED_SPECIES_ALIASES.get(text)
    if alias:
        return alias
    composite = _composite_species_alias(text)
    if composite:
        return composite
    return _slug_species(text)


def _remember_resolved_species_aliases(requested: str, species_key: str, match: Dict[str, Any]) -> None:
    """Cache Ensembl names/common names as runtime aliases after a successful lookup."""
    candidates = {
        requested,
        _slug_species(requested),
        species_key,
        species_key.replace("_", " "),
        str(match.get("name") or ""),
        str(match.get("display_name") or ""),
        str(match.get("common_name") or ""),
    }
    candidates.update(str(alias or "") for alias in (match.get("aliases") or []))
    for candidate in candidates:
        text = str(candidate or "").strip().lower()
        if not text:
            continue
        _SPECIES_ALIASES.setdefault(text, species_key)
        slug = _slug_species(text)
        if slug:
            _SPECIES_ALIASES.setdefault(slug, species_key)
            _REQUESTED_SPECIES_ALIASES.setdefault(slug, species_key)
        _REQUESTED_SPECIES_ALIASES.setdefault(text, species_key)


def _dynamic_species_config(key: str) -> SpeciesConfig:
    species_key = normalize_requested_species(key)
    species_dir = DATA_ROOT / "ensembl" / species_key
    return SpeciesConfig(
        key=species_key,
        label=f"{species_key.replace('_', ' ').title()} (Ensembl)",
        ensembl_species=species_key,
        ensembl_assembly=None,
        ucsc_build=species_key,
        genome_fasta=species_dir / f"{species_key}.genome.fa",
        genome_fasta_gz=species_dir / f"{species_key}.genome.fa.gz",
        gtf_path=species_dir / f"{species_key}.annotation.gtf",
        chrom_sizes_path=species_dir / f"{species_key}.chrom.sizes",
        phastcons_path=species_dir / f"{species_key}.phastCons.unavailable.bw",
        phylop_path=species_dir / f"{species_key}.phyloP.unavailable.bw",
        bowtie_prefix=species_dir / "bowtie" / species_key,
        refseq_xref_path=species_dir / f"{species_key}.metadata.RefSeq.gz",
        conservation_mode="disabled",
        reference_source="ensembl_dynamic",
    )


def set_reference_data_root(path: str | Path) -> Path:
    """Switch reference-data paths for the current process."""
    global DATA_ROOT, _SPECIES_DATA, _ENSEMBL_SPECIES_CACHE, _ENSEMBL_SPECIES_CACHE_AT
    root = Path(path).expanduser().resolve()
    DATA_ROOT = root
    os.environ["ASO_REFERENCE_DATA_ROOT"] = str(root)
    _SPECIES_DATA = _build_static_species_data(root)
    _ENSEMBL_SPECIES_CACHE = None
    _ENSEMBL_SPECIES_CACHE_AT = 0.0
    return root


def get_species_config(value: Optional[str]) -> SpeciesConfig:
    """Return SpeciesConfig for requested species, using dynamic Ensembl paths when needed."""
    key = normalize_species(value)
    if key in _SPECIES_DATA:
        return _SPECIES_DATA[key]
    return _dynamic_species_config(key)


def list_supported_species() -> Dict[str, SpeciesConfig]:
    """Return copy of statically supported species mapping."""
    return dict(_SPECIES_DATA)


def available_reference_files() -> List[Dict[str, Any]]:
    """Describe configured local files without retrieval or identity claims."""
    resources = []
    for species, config in list_supported_species().items():
        files = [("genome_fasta", config.genome_fasta), ("annotation_gtf", config.gtf_path)]
        if config.genome_fasta.is_file():
            files.append(("genome_fasta_index", Path(str(config.genome_fasta) + ".fai")))
        for role, path in files:
            if path.is_file():
                resources.append({
                    "species": species, "role": role, "path": str(path),
                    "expected_assembly": config.ensembl_assembly,
                    "identity_verified": False,
                })
    return resources


def is_static_species(value: Optional[str]) -> bool:
    """Return whether a species has curated static reference metadata."""
    return normalize_species(value) in _SPECIES_DATA


def list_ensembl_species(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Return Ensembl REST species metadata with a short in-memory and ignored on-disk cache."""
    global _ENSEMBL_SPECIES_CACHE, _ENSEMBL_SPECIES_CACHE_AT
    now = time.time()
    if _ENSEMBL_SPECIES_CACHE is not None and not force_refresh and now - _ENSEMBL_SPECIES_CACHE_AT < 24 * 3600:
        return list(_ENSEMBL_SPECIES_CACHE)

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "modal-aso"
    cache_path = cache_root / "ensembl_species_cache.json"
    # Read installed metadata without writing into the shared reference installation.
    legacy_cache_path = DATA_ROOT / "ensembl_species_cache.json"
    if not cache_path.exists() and legacy_cache_path.exists():
        read_cache_path = legacy_cache_path
    else:
        read_cache_path = cache_path
    if not force_refresh and read_cache_path.exists():
        try:
            payload = json.loads(read_cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("species"), list):
                _ENSEMBL_SPECIES_CACHE = list(payload["species"])
                _ENSEMBL_SPECIES_CACHE_AT = float(payload.get("cached_at") or now)
                return list(_ENSEMBL_SPECIES_CACHE)
        except Exception:
            pass

    url = f"{ENSEMBL_REST}/info/species"
    response = requests.get(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        params={"content-type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    species = payload.get("species") if isinstance(payload, dict) else None
    if not isinstance(species, list):
        raise RuntimeError("Ensembl species response did not contain a species list.")
    _ENSEMBL_SPECIES_CACHE = [entry for entry in species if isinstance(entry, dict)]
    _ENSEMBL_SPECIES_CACHE_AT = now
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"cached_at": now, "species": _ENSEMBL_SPECIES_CACHE}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return list(_ENSEMBL_SPECIES_CACHE)


def resolve_ensembl_species(value: Optional[str]) -> Dict[str, Any]:
    """Resolve a user/request species string against Ensembl species metadata."""
    requested = str(value or "").strip()
    if not requested:
        requested = "human"
    normalized = normalize_requested_species(requested)
    if normalized in _SPECIES_DATA:
        cfg = _SPECIES_DATA[normalized]
        return {
            "status": "resolved",
            "species_key": normalized,
            "ensembl_species": cfg.ensembl_species,
            "display_name": cfg.label,
            "static": True,
            "matches": [],
        }

    query_variants = {
        requested.lower(),
        normalized,
        normalized.replace("_", " "),
    }
    try:
        species = list_ensembl_species()
    except Exception as exc:
        return {
            "status": "error",
            "species_key": normalized,
            "error": f"ensembl_species_lookup_failed:{exc}",
            "matches": [],
        }

    exact: List[Dict[str, Any]] = []
    fuzzy: List[Dict[str, Any]] = []
    for entry in species:
        names = {
            str(entry.get("name") or "").lower(),
            str(entry.get("display_name") or "").lower(),
            str(entry.get("common_name") or "").lower(),
        }
        names.update(str(alias or "").lower() for alias in (entry.get("aliases") or []))
        names = {name for name in names if name}
        slugs = {_slug_species(name) for name in names}
        if normalized in slugs or names.intersection(query_variants):
            exact.append(entry)
        elif any(normalized and normalized in slug for slug in slugs):
            fuzzy.append(entry)

    matches = exact or fuzzy
    taxonomy_evidence = None
    if not matches:
        # Common names and synonyms are taxonomic data, not an application alias list.
        query = requested.replace("_", " ")
        url = f"{ENSEMBL_REST}/taxonomy/name/{quote(query, safe='')}"
        try:
            response = requests.get(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                                    params={"content-type": "application/json"})
            response.raise_for_status()
            taxa = response.json()
            taxon_ids = {str(item.get("id")) for item in taxa if isinstance(item, dict)}
            matches = [entry for entry in species if str(entry.get("taxon_id")) in taxon_ids]
            taxonomy_evidence = {"source_url": response.url, "query": query, "response": taxa}
        except Exception:
            # An unavailable reference is unresolved, never a guessed organism.
            pass
    if len(matches) == 1:
        match = matches[0]
        species_key = normalize_requested_species(str(match.get("name") or normalized))
        _remember_resolved_species_aliases(requested, species_key, match)
        return {
            "status": "resolved",
            "species_key": species_key,
            "ensembl_species": species_key,
            "display_name": match.get("display_name") or match.get("common_name") or species_key,
            "assembly": match.get("assembly"),
            "release": match.get("release"),
            "taxon_id": match.get("taxon_id"),
            "static": False,
            "matches": [match],
            "taxonomy_evidence": taxonomy_evidence,
        }
    if len(matches) > 1:
        return {
            "status": "ambiguous",
            "species_key": normalized,
            "matches": [
                {
                    "name": entry.get("name"),
                    "display_name": entry.get("display_name"),
                    "common_name": entry.get("common_name"),
                    "assembly": entry.get("assembly"),
                }
                for entry in matches[:10]
            ],
        }
    return {
        "status": "not_found",
        "species_key": normalized,
        "matches": [],
    }
