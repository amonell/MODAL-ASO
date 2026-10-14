#!/usr/bin/env python3
"""Download generic Ensembl reference assets for MODAL-ASO dynamic species."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aso_agents.species_config import DATA_ROOT, get_species_config, resolve_ensembl_species  # noqa: E402


try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - exercised only when tqdm is absent
    tqdm = None


ENSEMBL_FTP = os.environ.get("ENSEMBL_FTP", "https://ftp.ensembl.org").rstrip("/")
USER_AGENT = os.environ.get("HTTP_USER_AGENT", "aso-agent/0.1 (+https://example.org)")


def _get_text(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
    response.raise_for_status()
    return response.text


def _hrefs(url: str) -> List[str]:
    return re.findall(r'href="([^"]+)"', _get_text(url))


def _sort_key(name: str) -> List[Any]:
    parts: List[Any] = []
    for item in re.split(r"(\d+)", name):
        if item.isdigit():
            parts.append(int(item))
        else:
            parts.append(item)
    return parts


def _pick_fasta_href(species_key: str) -> str:
    base = f"{ENSEMBL_FTP}/pub/current_fasta/{species_key}/dna/"
    hrefs = [href for href in _hrefs(base) if href.endswith(".fa.gz")]
    preferred = [
        href for href in hrefs
        if ".dna.toplevel.fa.gz" in href and "dna_rm" not in href and "dna_sm" not in href
    ]
    if preferred:
        return base + sorted(preferred, key=_sort_key)[0]
    preferred = [
        href for href in hrefs
        if ".dna.primary_assembly.fa.gz" in href and "dna_rm" not in href and "dna_sm" not in href
    ]
    if preferred:
        return base + sorted(preferred, key=_sort_key)[0]
    raise RuntimeError(
        f"No single Ensembl toplevel/primary assembly FASTA found for {species_key}. "
        "Split per-chromosome FASTA assemblies are not supported by this downloader yet."
    )


def _pick_gtf_href(species_key: str, release: Optional[Any]) -> str:
    release_text = str(release or "").strip()
    candidate_bases = []
    if release_text:
        candidate_bases.append(f"{ENSEMBL_FTP}/pub/release-{release_text}/gtf/{species_key}/")
    candidate_bases.append(f"{ENSEMBL_FTP}/pub/current_gtf/{species_key}/")
    last_error = ""
    for base in candidate_bases:
        try:
            hrefs = [href for href in _hrefs(base) if href.endswith(".gtf.gz")]
        except Exception as exc:
            last_error = str(exc)
            continue
        preferred = [
            href for href in hrefs
            if "abinitio" not in href.lower() and ".chr." not in href.lower()
        ]
        if preferred:
            return base + sorted(preferred, key=_sort_key)[0]
        fallback = [href for href in hrefs if "abinitio" not in href.lower()]
        if fallback:
            return base + sorted(fallback, key=_sort_key)[0]
    raise RuntimeError(f"No Ensembl GTF found for {species_key}. Last error: {last_error}")


def _progress(total: int, initial: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total or None, initial=initial, unit="B", unit_scale=True, desc=desc)


def _download(url: str, output: Path) -> None:
    import zlib
    from aso_agents.asset_store import download, attest, verify
    if output.is_file() and not verify(output):
        # Adopt a complete existing archive, then reuse its integrity record.
        try:
            attest(output, source=url, kind="gzip")
        except (OSError, ValueError, EOFError, zlib.error):
            pass
    download(url, output, kind="gzip")


def _gunzip(source: Path, output: Path) -> None:
    from aso_agents.asset_store import decompress
    decompress(source, output, kind="fasta" if output.suffix in {".fa", ".fasta", ".fna"} else None)


def _generate_chrom_sizes(fasta: Path, output: Path) -> int:
    if output.exists() and output.stat().st_size > 0:
        print(f"Already exists: {output}", flush=True)
        return sum(1 for _ in output.open("r", encoding="utf-8"))
    count = 0
    current_name = ""
    current_len = 0
    with fasta.open("r", encoding="utf-8", errors="replace") as src, output.open("w", encoding="utf-8") as dst:
        for raw in src:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name:
                    dst.write(f"{current_name}\t{current_len}\n")
                    count += 1
                current_name = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
        if current_name:
            dst.write(f"{current_name}\t{current_len}\n")
            count += 1
    return count


def _bowtie_index_complete(prefix: Path) -> bool:
    return all(Path(f"{prefix}.{suffix}.ebwt").exists() for suffix in ("1", "2", "3", "4", "rev.1", "rev.2"))


def _build_bowtie_index(fasta: Path, prefix: Path, skip: bool = False) -> Dict[str, Any]:
    if _bowtie_index_complete(prefix):
        return {"status": "complete", "prefix": str(prefix), "already_present": True}
    if skip:
        return {"status": "skipped", "prefix": str(prefix), "reason": "skip_bowtie_build"}
    executable = shutil.which("bowtie-build")
    if not executable:
        from aso_agents.runtime_binaries import ensure_bowtie_1

        ensure_bowtie_1()
        executable = shutil.which("bowtie-build")
    if not executable:
        from aso_agents.runtime_binaries import bowtie_1_install_hint

        return {
            "status": "error",
            "prefix": str(prefix),
            "error": "bowtie-build not found in PATH",
            "hint": bowtie_1_install_hint() + f" Then run: bowtie-build {fasta} {prefix}",
        }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building Bowtie 1 index for {fasta.name}; this can take a while.", flush=True)
    proc = subprocess.run([executable, str(fasta), str(prefix)], text=True)
    if proc.returncode != 0:
        return {"status": "error", "prefix": str(prefix), "error": f"bowtie-build exited {proc.returncode}"}
    return {"status": "complete", "prefix": str(prefix), "already_present": False}


def _link_or_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    target.symlink_to(source)


def _link_manifest_to_configured_paths(manifest: Dict[str, Any], species_key: str) -> None:
    cfg = get_species_config(species_key)
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    for key, target in (
        ("genome_fasta", cfg.genome_fasta),
        ("genome_fasta_gz", cfg.genome_fasta_gz),
        ("gtf_path", cfg.gtf_path),
        ("chrom_sizes_path", cfg.chrom_sizes_path),
    ):
        source_text = paths.get(key)
        if not source_text or target is None:
            continue
        source = Path(str(source_text)).expanduser().resolve()
        if source.exists():
            _link_or_replace(source, Path(target))
    source_prefix = Path(str(paths.get("bowtie_prefix") or ""))
    target_prefix = Path(cfg.bowtie_prefix)
    if source_prefix:
        for suffix in ("1", "2", "3", "4", "rev.1", "rev.2"):
            source = Path(f"{source_prefix}.{suffix}.ebwt").expanduser()
            if source.exists():
                _link_or_replace(source.resolve(), Path(f"{target_prefix}.{suffix}.ebwt"))


def download_species_assets(
    species: str,
    species_dir: Optional[Path] = None,
    *,
    skip_bowtie_build: bool = False,
    link_configured_paths: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    resolved = resolve_ensembl_species(species)
    if resolved.get("status") != "resolved":
        return {"status": "error", "error": "species_not_resolved", "species_resolution": resolved}
    species_key = str(resolved.get("species_key") or "").strip()
    if not species_key:
        return {"status": "error", "error": "species_key_missing", "species_resolution": resolved}
    cfg = get_species_config(species_key)
    out_dir = (species_dir or cfg.genome_fasta.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    release = resolved.get("release")
    fasta_url = _pick_fasta_href(species_key)
    gtf_url = _pick_gtf_href(species_key, release)
    genome_gz = out_dir / f"{species_key}.genome.fa.gz"
    genome_fasta = out_dir / f"{species_key}.genome.fa"
    gtf_gz = out_dir / f"{species_key}.annotation.gtf.gz"
    gtf_path = out_dir / f"{species_key}.annotation.gtf"
    chrom_sizes = out_dir / f"{species_key}.chrom.sizes"
    bowtie_prefix = out_dir / "bowtie" / species_key
    manifest_path = out_dir / "reference_manifest.json"
    manifest: Dict[str, Any] = {
        "status": "planned" if dry_run else "running",
        "species": species_key,
        "display_name": resolved.get("display_name"),
        "assembly": resolved.get("assembly"),
        "release": release,
        "conservation_mode": "disabled",
        "urls": {"genome_fasta_gz": fasta_url, "gtf_gz": gtf_url},
        "paths": {
            "species_dir": str(out_dir),
            "genome_fasta_gz": str(genome_gz),
            "genome_fasta": str(genome_fasta),
            "gtf_gz": str(gtf_gz),
            "gtf_path": str(gtf_path),
            "chrom_sizes_path": str(chrom_sizes),
            "bowtie_prefix": str(bowtie_prefix),
        },
    }
    if dry_run:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    _download(fasta_url, genome_gz)
    _gunzip(genome_gz, genome_fasta)
    _download(gtf_url, gtf_gz)
    _gunzip(gtf_gz, gtf_path)
    chrom_count = _generate_chrom_sizes(genome_fasta, chrom_sizes)
    bowtie = _build_bowtie_index(genome_fasta, bowtie_prefix, skip=skip_bowtie_build)
    manifest["chromosome_count"] = chrom_count
    manifest["bowtie"] = bowtie
    manifest["status"] = "complete" if bowtie.get("status") == "complete" else "partial"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if link_configured_paths and out_dir != cfg.genome_fasta.parent:
        _link_manifest_to_configured_paths(manifest, species_key)
    return manifest


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True, help="Ensembl species name or alias, e.g. rattus_norvegicus.")
    parser.add_argument(
        "--species-dir",
        default="",
        help="Exact species asset directory. Defaults to the configured MODAL-ASO species directory.",
    )
    parser.add_argument("--skip-bowtie-build", action="store_true", help="Download assets but do not build Bowtie index.")
    parser.add_argument("--link-configured-paths", action="store_true", help="Symlink configured MODAL-ASO paths to --species-dir outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve URLs and write a manifest without downloading.")
    parser.add_argument("--output-json", default="", help="Optional path to write the manifest JSON.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    species_dir = Path(args.species_dir).expanduser() if args.species_dir else None
    try:
        manifest = download_species_assets(
            args.species,
            species_dir=species_dir,
            skip_bowtie_build=bool(args.skip_bowtie_build),
            link_configured_paths=bool(args.link_configured_paths),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        manifest = {"status": "error", "error": str(exc), "species": args.species}
    if args.output_json:
        Path(args.output_json).expanduser().write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest.get("status") in {"complete", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
