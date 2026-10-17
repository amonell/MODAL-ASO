"""Measured target-RNA accessibility, separate from oligonucleotide self-folding."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def unpaired_probabilities(sequence, maximum_length, output_dir):
    executable = shutil.which("RNAplfold")
    if not executable:
        raise RuntimeError("target_accessibility_requires_RNAplfold")
    sequence = sequence.upper().replace("T", "U")
    if set(sequence) - set("ACGU"):
        raise ValueError("target_accessibility_requires_unambiguous_reference_sequence")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    window = min(len(sequence), max(70, maximum_length))
    command = [executable, "-W", str(window), "-L", str(window), "-u", str(maximum_length), "-T", "37"]
    version = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    metadata = {"method": "RNAplfold", "version": version, "command": command,
                "window_nt": window, "maximum_pair_span_nt": window, "temperature_celsius": 37,
                "maximum_unpaired_length_nt": maximum_length,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(), "sequence_length": len(sequence),
                "measurement": "mean probability that the full binding window is unpaired",
                "coordinate_convention": "1-based inclusive; matrix row=end, column=length",
                "scope": "Sequence-based local equilibrium prediction; no tissue-specific or experimental accessibility measurement."}
    manifest = directory / "metadata.json"
    matrix = directory / "target_lunp"
    if not (manifest.exists() and matrix.exists() and json.loads(manifest.read_text()) == metadata):
        fasta = ">target\n" + sequence + "\n"
        (directory / "input.fa").write_text(fasta)
        result = subprocess.run(command, input=fasta, cwd=directory, capture_output=True, text=True, check=True)
        (directory / "stdout.txt").write_text(result.stdout)
        (directory / "stderr.txt").write_text(result.stderr)
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    values = {}
    for line in matrix.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        end = int(fields[0])
        for length, text in enumerate(fields[1:], 1):
            try:
                value = float(text)
            except ValueError:
                continue
            if math.isfinite(value) and 0 <= value <= 1 and end >= length:
                values[(end, length)] = value
    return values, {**metadata, "metadata_path": str(manifest), "matrix_path": str(matrix)}


def annotate_target_accessibility(tiles_path, goal, runtime, primary_catalog):
    if goal.get("target_accessibility") is not True:
        return {"status": "not_requested"}
    path = Path(tiles_path)
    payload = json.loads(path.read_text())
    rows = payload.get("rows") or []
    if not rows:
        return {"status": "empty_candidate_set"}
    species = runtime._cross_species_requested_species(goal)
    primary = runtime.normalize_species(goal.get("species"))
    references = {primary: runtime._build_isoform_cdna_sequences(
        primary_catalog, primary, reference_fasta=runtime._resolve_patient_reference_fasta(goal))}
    reference_records = {primary: {key: (primary_catalog[key].get("sequence_evidence") or {})
                                   for key in references[primary]}}
    targets = runtime._resolve_cross_species_gene_targets(goal, species) if len(species) > 1 else {}
    errors = {}
    for key in species:
        if key == primary:
            continue
        target = targets.get(key) or {}
        if target.get("status") != "resolved":
            errors[key] = target.get("error") or "target_gene_unresolved"
            continue
        reference_records[key] = {}
        sequences, error = runtime._resolve_cross_species_cdna_sequences(
            target.get("gene_symbol"), key, gene_id=target.get("gene_id"), canonical_only=True,
            reference_records=reference_records[key],
            **({"transcript_ids": target["transcript_ids"]} if target.get("transcript_ids") else {}))
        if error:
            errors[key] = error
        references[key] = sequences
    maximum_length = max(len(str(row.get("Sequence") or "")) for row in rows)
    predictions = {}
    for key in species:
        if not references.get(key):
            errors.setdefault(key, "reference_cdna_unavailable")
        for transcript, sequence in (references.get(key) or {}).items():
            record_id = hashlib.sha256((key + transcript + sequence).encode()).hexdigest()
            try:
                values, evidence = unpaired_probabilities(sequence, maximum_length, path.parent / "target_accessibility" / record_id)
                evidence["reference_sequence"] = reference_records.get(key, {}).get(transcript) or {}
                predictions[(key, transcript)] = (values, evidence)
            except Exception as exc:
                errors[key] = str(exc)
    complete_count = 0
    for row in rows:
        target_sequence = str(row.get("Sequence") or "").upper().replace("U", "T")
        measurements = {}
        species_scores = {}
        for key in species:
            hits = []
            for transcript, sequence in (references.get(key) or {}).items():
                prediction = predictions.get((key, transcript))
                if not prediction or not target_sequence:
                    continue
                values, evidence = prediction
                start = sequence.find(target_sequence)
                while start >= 0:
                    end = start + len(target_sequence)
                    probability = values.get((end, len(target_sequence)))
                    if probability is not None:
                        hits.append({"transcript_id": transcript, "start": start + 1, "end": end,
                                     "full_window_unpaired_probability": probability, "evidence": evidence})
                    start = sequence.find(target_sequence, start + 1)
            measurements[key] = hits
            if hits:
                species_scores[key] = max(hit["full_window_unpaired_probability"] for hit in hits)
        complete = bool(species) and len(species_scores) == len(species)
        row["target_accessibility"] = {
            "status": "measured" if complete else "unavailable_or_no_exact_target",
            "primary_species": primary, "measurements": measurements,
            "species_scores": species_scores,
            "aggregation": "Best exact target site in each recorded species transcript scope, then minimum across required species.",
            "species_errors": errors,
        }
        if complete:
            row["target_accessibility_min_probability"] = min(species_scores.values())
            complete_count += 1
    columns = payload.setdefault("schema", {}).setdefault("columns", {})
    columns.update(target_accessibility="object", target_accessibility_min_probability="number")
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return {"status": "complete" if not errors else "incomplete", "annotated_count": len(rows),
            "complete_match_count": complete_count, "species_errors": errors,
            "reference_transcripts": {key: list(value) for key, value in references.items()},
            "method": "RNAplfold full binding-window unpaired probability; relative ranking, no efficacy threshold"}


def select_panel_rows(ranked_rows, limit, diversify=False, minimum=None):
    """Retain unique molecules, filling a diverse panel only to its requested minimum."""
    unique = []
    seen = set()
    for row in ranked_rows:
        molecule = str(row.get("Synthesis") or row.get("synthesized_antisense_sequence") or row.get("Sequence") or row.get("sequence") or "")
        identity = (molecule, str(row.get("IDT") or row.get("chemistry_label") or ""))
        if molecule and identity not in seen:
            seen.add(identity)
            unique.append(row)
    if not diversify:
        return unique[:limit]
    def coordinate(value):
        if type(value) is int:
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isascii() and stripped.isdigit():
                return int(stripped)
        return None

    def intervals(row):
        # Ranked tiling rows use Genomic_* fields, while delivered candidates
        # can carry explicit binding blocks. Prefer the physical target blocks
        # over accessibility hits across other transcripts.
        blocks = row.get("binding_intervals_bed0_half_open")
        if isinstance(blocks, list) and blocks:
            parsed = [(part[0], coordinate(part[1]) + 1, coordinate(part[2])) for part in blocks
                      if isinstance(part, (list, tuple)) and len(part) >= 3
                      and isinstance(part[0], str) and coordinate(part[1]) is not None
                      and coordinate(part[2]) is not None]
            if parsed:
                return parsed
        segments = row.get("Genomic_segments")
        if isinstance(segments, str):
            try:
                segments = json.loads(segments)
            except (TypeError, ValueError):
                segments = None
        if isinstance(segments, list) and segments:
            parsed = [(part["chrom"], coordinate(part["start"]), coordinate(part["end"])) for part in segments
                      if isinstance(part, dict) and isinstance(part.get("chrom"), str)
                      and coordinate(part.get("start")) is not None
                      and coordinate(part.get("end")) is not None]
            if parsed:
                return parsed
        chrom = row.get("Genomic_chrom") or row.get("Chrom") or row.get("chrom")
        start = row.get("Genomic_start", row.get("Start", row.get("start")))
        end = row.get("Genomic_end", row.get("End", row.get("end")))
        start, end = coordinate(start), coordinate(end)
        if chrom is not None and start is not None and end is not None:
            return [(chrom, start, end)]
        access = row.get("target_accessibility") or {}
        hits = (access.get("measurements") or {}).get(access.get("primary_species")) or []
        if hits:
            return [(hit["transcript_id"], hit["start"], hit["end"]) for hit in hits]
        return []
    selected = []
    covered = []
    deferred = []
    for row in unique:
        spans = intervals(row)
        overlap = any(chrom == old_chrom and start <= old_end and old_start <= end
                      for chrom, start, end in spans for old_chrom, old_start, old_end in covered)
        if overlap or not spans:
            deferred.append(row)
        else:
            selected.append(row)
            covered.extend(spans)
        if len(selected) == limit:
            return selected
    # A range such as four to six does not require filling the upper bound
    # with redundant binding windows. Once enough disjoint physical targets
    # exist, retain that panel. If they do not meet the requested minimum,
    # nearby alternatives may still fill the shortfall. Callers without an
    # explicit minimum keep the historical upper-bound behavior.
    if type(minimum) is int and minimum > 0:
        needed = min(minimum, limit)
        if len(selected) >= needed:
            return selected
        return (selected + deferred)[:needed]
    return (selected + deferred)[:limit]
