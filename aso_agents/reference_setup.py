"""Check local reference availability before transcript resolution."""

from pathlib import Path
import os
import shlex
from typing import Any

from .species_config import get_species_config


def _available(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def inspect_resolution_references(species: str) -> dict[str, Any]:
    config = get_species_config(species)
    annotation = config.gtf_path
    compressed = annotation.with_suffix(annotation.suffix + ".gz")
    if _available(compressed):
        annotation = compressed
    missing = []
    if not _available(annotation):
        missing.append("transcript annotation")
    if not _available(config.genome_fasta):
        missing.append("genome sequence")
    message = ""
    if missing:
        label = species.replace("_", " ")
        lines = [f"Local {label} reference files are unavailable: {' and '.join(missing)}."]
        if "transcript annotation" in missing:
            lookup_note = (
                "This run uses frozen references, so isoform mapping requires the local annotation."
                if str(os.environ.get("ASO_DEPENDENCY_MODE") or "").strip().lower() == "frozen"
                else "Transcript mapping will use Ensembl or a supplied reference. Requested versions may be unavailable online."
            )
            lines += [
                lookup_note,
                f"Expected annotation: {config.gtf_path}",
            ]
        if "genome sequence" in missing:
            lines += [
                "Genome-based design and off-target screening require a local reference genome.",
                f"Expected genome: {config.genome_fasta}",
            ]
        lines.append(
            "If these files are already installed, set ASO_REFERENCE_DATA_ROOT to their folder before starting or resuming the run."
        )
        if species in {"human", "mouse"}:
            lines.append(f"To install them, run: modal-aso setup --species {shlex.quote(species)} --features references")
        message = "\n".join(lines)
    return {
        "species": species,
        "status": "missing" if missing else "available",
        "annotation": str(annotation),
        "genome": str(config.genome_fasta),
        "missing": missing,
        "message": message,
    }
