"""Recover masked exonic sequence from the matching published annotation.

Only reference data are downloaded. Each run computes its own matches, and
unknown bases remain unresolved if the replacement fails identity checks.
"""
from __future__ import annotations

import gzip
from pathlib import Path

from .asset_store import attest, download, read_receipt
from .paths import cache_root
from .reference_assets import CATALOG, reference_checksums
from .species_config import get_species_config


def load_transcript_sequences(species: str, entries: dict) -> tuple[dict, dict]:
    """Read only the requested, version-matched transcripts from a pinned FASTA."""
    config = get_species_config(species)
    catalog = CATALOG.get(species)
    if not catalog or Path(config.gtf_path).name != catalog["gtf"]:
        raise ValueError("No matching published transcript reference is configured for this annotation")
    name = f"gencode.v{catalog['annotation']}.transcripts.fa.gz"
    expected = reference_checksums().get(name, {}).get("sha256")
    if not expected:
        raise ValueError("Published transcript reference checksum is missing")
    path = cache_root() / "reference_transcripts" / name
    url = f"{catalog['gencode_url']}/{name}"
    download(url, path, expected_sha256=expected, kind="gzip",
             mirror_urls=(url.replace("https://", "http://", 1),))
    # download() can verify a pre-existing pinned file without a receipt.
    # Record that verification so subsequent runs avoid hashing it again.
    if read_receipt(path).get("sha256") != expected:
        attest(path, source=url, expected_sha256=expected, kind="gzip")
    wanted = {entry.get("transcript_id_version"): key for key, entry in entries.items()
              if entry.get("transcript_id_version")}
    records = {}
    current, chunks = None, []

    def save():
        if current:
            key = wanted[current]
            if key in records:
                raise ValueError("Duplicate transcript in published reference")
            records[key] = "".join(chunks).upper()

    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line in stream:
            if line.startswith(">"):
                save()
                fields = line[1:].split()[0].split("|")
                current = fields[0] if fields[0] in wanted else None
                if current:
                    entry = entries[wanted[current]]
                    if len(fields) < 2 or fields[1] != entry.get("gene_id_version"):
                        raise ValueError("Transcript and annotation gene identities differ")
                chunks = []
            elif current:
                chunks.append(line.strip())
        save()
    return records, {"source": url, "sha256": expected, "annotation": catalog["annotation"],
                     "assembly": catalog["assembly"], "transcripts_found": len(records),
                     "verified": read_receipt(path).get("sha256") == expected}


def recover_exons(sequence: str, genome_exons: list[str], lengths: list[int]) -> list[str]:
    """Require exon lengths and every available genomic base to agree."""
    if not sequence or len(sequence) != sum(lengths):
        raise ValueError("Transcript length differs from the annotated exons")
    result, offset = [], 0
    for original, length in zip(genome_exons, lengths):
        recovered = sequence[offset:offset + length]
        offset += length
        if original and len(original) != length:
            raise ValueError("Genome exon length differs from the annotation")
        if any(a in "ACGT" and a != b for a, b in zip(original, recovered)):
            raise ValueError("Published transcript differs from known genomic bases")
        result.append(recovered)
    return result
