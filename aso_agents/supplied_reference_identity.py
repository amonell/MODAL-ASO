"""Bind supplied files to typed references and verify their accession identity.

User prose cannot establish identity. An accession binding requires an exact
resolver result, the complete ordered exon structure and sequence verification
against the installed reference, allowing recorded patient substitutions.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .gapmer_selectivity import _complete_structure, _reference_context_compatible
from .supplied_reference_role import artifact_sha256


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _structure_digest(entry):
    return _digest({"structure": _complete_structure(entry),
                    "ordinals": [exon.get("index") for exon in entry.get("exons_tx") or []]})


def canonical_supplied_paths(paths, goal):
    """Collapse byte-identical copies, keeping every original path as provenance."""
    by_digest, aliases, canonical = {}, {}, []
    for value in paths:
        path = str(Path(value).expanduser().resolve())
        digest = artifact_sha256(path)
        if digest not in by_digest:
            by_digest[digest] = path
            canonical.append(path)
        aliases[path] = by_digest[digest]
    if any(a != b for a, b in aliases.items()):
        goal.setdefault("supplied_reference_path_aliases", {}).update(aliases)
    # Only explicit file fields are normalized; labels and prose remain intact.
    records = [goal, *(goal.get("splice_isoform_objectives") or []),
               *(goal.get("evaluation_targets") or []),
               *(goal.get("supplied_sequence_roles") or []),
               *(goal.get("artifact_refs") or [])]
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in ("snapgene_path", "artifact_ref_path", "path"):
            if isinstance(record.get(key), str) and record[key]:
                path = str(Path(record[key]).expanduser().resolve())
                if path in aliases:
                    record[key] = aliases[path]
    return canonical


def single_declared_reference(goal):
    """Return a unique typed transcript identity, without inspecting its name."""
    identities = set()
    lock = goal.get("transcript_lock") or {}
    if goal.get("transcript_policy") == "user_locked" and isinstance(lock, dict):
        value = lock.get("value") or lock.get("transcript_id") or lock.get("reference_accession")
        if isinstance(value, str) and value.strip():
            identities.add(value.strip())
    for record in [goal, *(goal.get("design_programs") or [])]:
        if not isinstance(record, dict):
            continue
        for value in [record.get("transcript_id"), *(record.get("transcript_ids") or []),
                      (record.get("splice_event") or {}).get("reference_transcript_id")]:
            if isinstance(value, str) and value.strip():
                identities.add(value.strip())
    return next(iter(identities)) if len(identities) == 1 else None


def _annotation_for_entry(entry, goal):
    path = (entry.get("metadata") or {}).get("snapgene_path")
    if not path:
        return None
    path = str(Path(path).resolve())
    annotations = [goal.get("snapgene_annotation"),
                   *(goal.get("snapgene_annotations") or {}).values()]
    matches = [a for a in annotations if isinstance(a, dict) and a.get("snapgene_path")
               and str(Path(a["snapgene_path"]).resolve()) == path]
    unique = {_digest(a): a for a in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def attest_supplied_reference(entry, reference_entry, provenance, goal):
    """Verify the supplied sequence before adding an accession alias to it.

    Return a new receipt; neither catalog entry nor goal is mutated on failure.
    Whole transcript exon identity is required, including order and strand.
    """
    from Bio.Seq import Seq
    from pyfaidx import Fasta
    from .snapgene_ingestion import _read_sequence_record

    annotation = _annotation_for_entry(entry, goal)
    shape = _complete_structure(entry)
    if (not annotation or not shape or shape != _complete_structure(reference_entry)
            or not _reference_context_compatible(entry, reference_entry)):
        raise ValueError("supplied_reference_structure_unverified")
    path = Path(annotation["snapgene_path"])
    file_hash = artifact_sha256(path)
    if file_hash != annotation.get("input_file_sha256"):
        raise ValueError("supplied_reference_file_changed")
    sequence = str(_read_sequence_record(path).seq).upper()
    sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
    if sequence_hash != annotation.get("sequence_sha256"):
        raise ValueError("supplied_reference_sequence_changed")
    exons = sorted(annotation.get("exons") or [], key=lambda e: e.get("order", 0))
    if (len(exons) != len(shape[2])
            or [e.get("index") for e in entry.get("exons_tx") or []] != list(range(1, len(exons) + 1))):
        raise ValueError("supplied_reference_exons_incomplete")
    variants = {(str(v.get("chrom") or "").removeprefix("chr"), v.get("pos"),
                 v.get("ref"), v.get("alt")) for v in annotation.get("patient_variants") or []}
    substitutions, transcript = [], []
    with Fasta(annotation["reference_fasta_path"], sequence_always_upper=True) as fasta:
        chrom = next((c for c in (shape[0], "chr" + shape[0]) if c in fasta), None)
        if not chrom:
            raise ValueError("supplied_reference_chromosome_unverified")
        for ordinal, (exon, interval) in enumerate(zip(exons, shape[2]), 1):
            start, end = interval
            local_start, local_end = exon.get("local_start"), exon.get("local_end")
            if (exon.get("order") != ordinal
                    or (exon.get("genomic_start"), exon.get("genomic_end")) != interval
                    or str(exon.get("chrom") or "").removeprefix("chr") != shape[0]
                    or exon.get("strand") != shape[1]
                    or type(local_start) is not int or type(local_end) is not int
                    or not 0 <= local_start < local_end <= len(sequence)
                    or local_end - local_start != end - start + 1
                    or exon.get("feature_strand") not in (1, -1)):
                raise ValueError("supplied_reference_exon_mapping_unverified")
            read = sequence[local_start:local_end]
            if exon["feature_strand"] == -1:
                read = str(Seq(read).reverse_complement())
            if read != exon.get("sequence"):
                raise ValueError("supplied_reference_exon_sequence_changed")
            transcript.append(read)
            genomic = read if shape[1] == "+" else str(Seq(read).reverse_complement())
            baseline = str(fasta[chrom][start - 1:end])
            if len(baseline) != len(genomic) or set(genomic + baseline) - set("ACGT"):
                raise ValueError("supplied_reference_sequence_unverified")
            for offset, (ref, alt) in enumerate(zip(baseline, genomic)):
                if ref == alt:
                    continue
                if (shape[0], start + offset, ref, alt) not in variants:
                    raise ValueError("supplied_reference_unrecorded_sequence_difference")
                substitutions.append({"chrom": chrom, "pos": start + offset, "ref": ref, "alt": alt})
    source = reference_entry.get("metadata") or {}
    return {**deepcopy(provenance), "supplied_reference_identity": {
        "method": "complete_exon_structure_and_sequence",
        "supplied_transcript_id": entry["transcript_id"],
        "reference_transcript_id": reference_entry["transcript_id"],
        "reference_versioned_id": source.get("versioned_id") or reference_entry["transcript_id"],
        "reference_source": source.get("reference_source"),
        "reference_annotation": source.get("reference_annotation"),
        "input_path": str(path.resolve()), "input_sha256": file_hash,
        "sequence_sha256": sequence_hash, "structure_sha256": _structure_digest(entry),
        "transcript_sequence_sha256": hashlib.sha256("".join(transcript).encode()).hexdigest(),
        "matched_exon_count": len(exons), "exonic_substitutions": substitutions,
    }}


def verified_supplied_receipt(entry, reference):
    """Reject evidence if the file, structure, accession or selected identity changed."""
    metadata = entry.get("metadata") or {}
    receipts = [metadata.get("reference_accession_resolution"),
                *(metadata.get("reference_accession_resolutions") or [])]
    shape = _complete_structure(entry)
    for receipt in receipts:
        if (not isinstance(receipt, dict) or receipt.get("exact_accession_verified") is not True
                or receipt.get("requested_accession") != reference):
            continue
        proof = receipt.get("supplied_reference_identity") or {}
        if (not shape or proof.get("method") != "complete_exon_structure_and_sequence"
                or proof.get("structure_sha256") != _structure_digest(entry)
                or proof.get("supplied_transcript_id") != entry.get("transcript_id")
                or proof.get("matched_exon_count") != len(shape[2])
                or not metadata.get("snapgene_path")):
            continue
        path = Path(metadata["snapgene_path"]).resolve()
        try:
            if str(path) == proof.get("input_path") and artifact_sha256(path) == proof.get("input_sha256"):
                return receipt
        except OSError:
            continue
    return None
