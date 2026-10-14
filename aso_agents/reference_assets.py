"""Install and verify the reference assemblies used by MODAL-ASO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .asset_store import (asset_lock, atomic_json, attest, decompress, download,
                          file_lock, fingerprint, read_receipt, verify)
from .paths import reference_root, data_root, resource_root

UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath"
GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode"
CATALOG = {
    "human": {
        "assembly": "GRCh38", "build": "hg38", "annotation": "42",
        "fasta": "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
        "compressed": "GRCh38_no_alt_analysis_set_GCA_000001405.15.fna.gz",
        "genome_url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz",
        "gtf": "gencode.v42.annotation.gtf",
        "gencode_url": f"{GENCODE}/Gencode_human/release_42",
        "xref": "gencode.v42.metadata.RefSeq.gz", "alignment": "30way",
        "anchors": {"chr1": 248956422, "chrX": 156040895, "chrM": 16569},
    },
    "mouse": {
        "assembly": "GRCm39", "build": "mm39", "annotation": "M35",
        "fasta": "GRCm39.primary_assembly.genome.fa",
        "compressed": "GRCm39.primary_assembly.genome.fa.gz",
        "genome_url": f"{GENCODE}/Gencode_mouse/release_M35/GRCm39.primary_assembly.genome.fa.gz",
        "gtf": "gencode.vM35.primary_assembly.annotation.gtf",
        "gencode_url": f"{GENCODE}/Gencode_mouse/release_M35",
        "xref": "gencode.vM35.metadata.RefSeq.gz", "alignment": "35way",
        "anchors": {"chr1": 195154279, "chrX": 169476592, "chrM": 16299},
    },
}
INDEX_SUFFIXES = ("1", "2", "3", "4", "rev.1", "rev.2")


def assets(species: str) -> list[dict]:
    c = CATALOG[species]
    build, alignment = c["build"], c["alignment"]
    items = [
        {"name": c["compressed"], "url": c["genome_url"], "kind": "gzip", "extract": c["fasta"], "extracted_kind": "fasta"},
        {"name": c["gtf"] + ".gz", "url": f"{c['gencode_url']}/{c['gtf']}.gz", "kind": "gzip", "extract": c["gtf"]},
        {"name": c["xref"], "url": f"{c['gencode_url']}/{c['xref']}", "kind": "gzip"},
        {"name": build + ".chrom.sizes", "url": f"{UCSC}/{build}/bigZips/{build}.chrom.sizes"},
        *[{"name": f"{build}.{track}{alignment}.bw", "url": f"{UCSC}/{build}/{track}{alignment}/{build}.{track}{alignment}.bw", "kind": "bigwig"}
          for track in ("phastCons", "phyloP")],
    ]
    checksums = reference_checksums()
    for item in items:
        item["sha256"] = checksums.get(item["name"], {}).get("sha256")
        if "extract" in item:
            item["extracted_sha256"] = checksums.get(item["extract"], {}).get("sha256")
    return items


def reference_checksums() -> dict:
    return json.loads((resource_root() / "reference-checksums.json").read_text())


def binary(name: str) -> str:
    managed = data_root() / "runtime" / "bin" / name
    value = str(managed) if managed.is_file() else shutil.which(name)
    if not value:
        raise RuntimeError(f"{name} is required. Run: modal-aso setup --features tools")
    return value


def _fai(fasta: Path) -> dict:
    index = Path(str(fasta) + ".fai")
    source = read_receipt(fasta).get("sha256")
    binding = "fasta:sha256:" + str(source)
    if not (verify(index) and read_receipt(index).get("source") == binding):
        # Explicit output keeps imported, read-only FASTAs untouched.
        temporary = index.with_name(index.name + ".building")
        subprocess.run([binary("samtools"), "faidx", "--fai-idx", str(temporary), str(fasta)], check=True)
        temporary.replace(index)
        attest(index, source=binding)
    return {row.split("\t")[0]: int(row.split("\t")[1]) for row in index.read_text().splitlines()}


def validate_reference(species: str, root: Path) -> dict:
    c = CATALOG[species]
    lengths = _fai(root / c["fasta"])
    if any(lengths.get(name) != size for name, size in c["anchors"].items()):
        raise ValueError(f"FASTA does not match the {c['assembly']} assembly")
    header = []
    with (root / c["gtf"]).open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            header.append(line)
    annotation_header = "".join(header)
    if c["assembly"] not in annotation_header or ("version " + c["annotation"]) not in annotation_header:
        raise ValueError(f"Expected GENCODE {c['annotation']} on {c['assembly']}; annotation header differs")
    return {"assembly": c["assembly"], "annotation": c["annotation"], "contigs": len(lengths)}


def _index_files(prefix: Path) -> list[Path]:
    for extension in ("ebwt", "ebwtl"):
        files = [Path(f"{prefix}.{part}.{extension}") for part in INDEX_SUFFIXES]
        if all(p.is_file() for p in files):
            return files
    return []


def _sequence_digests(stream) -> dict:
    result, current, digest = {}, None, None
    alphabet = bytes(v if v in b"ACGT" else ord("N") for v in range(256))
    for line in stream:
        if line.startswith(b">"):
            if current is not None:
                result[current] = digest.hexdigest()
            current = line[1:].split()[0].decode()
            if current in result:
                raise ValueError("Duplicate FASTA sequence identifier")
            digest = hashlib.sha256()
        elif digest is not None:
            # Bowtie stores ambiguity symbols as N, and sequences in uppercase.
            normalized = line.strip().upper().translate(alphabet)
            digest.update(normalized)
    if current is not None:
        result[current] = digest.hexdigest()
    return result


def build_index(species: str, root: Path, *, progress=print) -> dict:
    c = CATALOG[species]
    fasta, prefix = root / c["fasta"], root / "bowtie" / c["build"]
    binding = "fasta:sha256:" + read_receipt(fasta)["sha256"]
    with file_lock(asset_lock(prefix)):
        existing = _index_files(prefix)
        if existing and all(verify(p) and read_receipt(p).get("source") == binding for p in existing):
            progress(f"Ready: {c['build']} Bowtie index")
            return {"status": "ready", "prefix": str(prefix)}
        prefix.parent.mkdir(parents=True, exist_ok=True)
        progress(f"Building the {c['build']} Bowtie index (one-time setup)...")
        with tempfile.TemporaryDirectory(prefix=c["build"] + "-", dir=prefix.parent) as temporary:
            staged = Path(temporary) / c["build"]
            subprocess.run([binary("bowtie-build"), "--threads", str(min(8, os.cpu_count() or 1)), str(fasta), str(staged)], check=True)
            files = _index_files(staged)
            if not files:
                raise RuntimeError("Bowtie returned an incomplete index")
            for path in files:
                destination = prefix.parent / path.name
                path.replace(destination)
                attest(destination, source=binding)
        return {"status": "built", "prefix": str(prefix)}


def install_reference(species: str, root: Path | None = None, *, progress=print) -> dict:
    root = Path(root) if root else reference_root()
    c = CATALOG[species]
    with file_lock(asset_lock(root / (species + ".setup"))):
        steps = []
        for item in assets(species):
            output = root / item.get("extract", item["name"])
            # Imported or already extracted verified files also satisfy setup.
            expected = item.get("extracted_sha256") if "extract" in item else item.get("sha256")
            if verify(output, expected_sha256=expected):
                progress(f"Ready: {output.name}")
                steps.append({"status": "ready", "path": str(output)})
                continue
            packed = root / item["name"]
            steps.append(download(item["url"], packed, expected_sha256=item.get("sha256"), kind=item.get("kind"), progress=progress))
            if "extract" in item:
                steps.append(decompress(packed, output, kind=item.get("extracted_kind"), expected_sha256=expected))
        validation = validate_reference(species, root)
        steps.append(build_index(species, root, progress=progress))
        result = {"status": "ready", "species": species, **validation, "steps": steps}
        atomic_json(root / ".modal-aso" / (species + ".json"), result)
        return result


def import_reference(species: str, source: Path, root: Path | None = None, *, progress=print) -> dict:
    """Verify existing files and link them into a writable managed reference store."""
    source, root = Path(source).expanduser().resolve(), Path(root) if root else reference_root()
    if source == root.resolve():
        raise ValueError("Use a separate managed data directory when importing existing references")
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(asset_lock(root / (species + ".setup"))):
        for item in assets(species):
            name = item.get("extract", item["name"])
            original, destination = source / name, root / name
            if not original.is_file():
                raise FileNotFoundError(f"Missing reference: {original}")
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != original:
                    raise ValueError(f"Managed file already exists: {destination}")
            else:
                destination.symlink_to(original)
            expected = item.get("extracted_sha256") if "extract" in item else item.get("sha256")
            if not verify(destination, expected_sha256=expected) or not read_receipt(destination):
                attest(destination, source="import:" + str(original), expected_sha256=expected,
                       kind=item.get("extracted_kind") or (item.get("kind") if "extract" not in item else None))
        validation = validate_reference(species, root)
        c = CATALOG[species]
        original_prefix = source / "bowtie" / c["build"]
        originals = _index_files(original_prefix)
        if originals:
            progress("Checking existing Bowtie index against the imported genome...")
            with (root / c["fasta"]).open("rb") as fasta:
                expected = _sequence_digests(fasta)
            proc = subprocess.Popen([binary("bowtie-inspect"), str(original_prefix)], stdout=subprocess.PIPE)
            try:
                actual = _sequence_digests(proc.stdout)
            finally:
                proc.stdout.close()
                result = proc.wait()
            if result or actual != expected:
                raise ValueError("Existing Bowtie index does not match the imported genome")
            target_dir = root / "bowtie"
            target_dir.mkdir(exist_ok=True)
            binding = "fasta:sha256:" + read_receipt(root / c["fasta"])["sha256"]
            for original in originals:
                destination = target_dir / original.name
                if not destination.exists():
                    destination.symlink_to(original)
                elif destination.resolve() != original:
                    raise ValueError(f"Managed index already exists: {destination}")
                attest(destination, source=binding)
        else:
            build_index(species, root, progress=progress)
        result = {"status": "imported", "species": species, "source": str(source), **validation}
        atomic_json(root / ".modal-aso" / (species + ".json"), result)
        return result


def inventory(species: str, *, full=False) -> list[dict]:
    root, c = reference_root(), CATALOG[species]
    files = [root / item.get("extract", item["name"]) for item in assets(species)]
    files += [Path(str(root / c["fasta"]) + ".fai")]
    files += _index_files(root / "bowtie" / c["build"]) or [root / "bowtie" / (c["build"] + ".1.ebwt")]
    checksums = reference_checksums()
    return [{"path": str(p), "status": "verified" if verify(p, expected_sha256=checksums.get(p.name, {}).get("sha256"), full=full) else "missing" if not p.exists() else "unverified",
             "bytes": p.stat().st_size if p.exists() else 0} for p in files]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=list(CATALOG), required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(install_reference(args.species, args.data_dir), indent=2))


if __name__ == "__main__":
    main()
