"""Content checks for a requested design, independent of successful report writing."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from .candidate_presentation import molecule_key, ordered_rows, distinct_ranked_rows


def _physical_element_explanation(elements):
    descriptions = []
    for element in elements or []:
        if not isinstance(element, dict):
            continue
        kind = element.get("element")
        if kind not in {"acceptor", "donor"}:
            continue
        exon = element.get("exon_ordinal")
        intron = element.get("adjacent_intron_ordinal")
        transcript = element.get("transcript_id")
        if not exon or not transcript:
            continue
        descriptions.append(
            f"{transcript} annotated exon {exon} {kind} "
            f"(intron {intron}/exon {exon} boundary); physical target spans "
            f"{element.get('intronic_overlap_bp')} intronic and "
            f"{element.get('exonic_overlap_bp')} exonic base(s)"
        )
    return "; ".join(dict.fromkeys(descriptions))


def _physical_splice_site_rationale(row, transcript_site, intronic_site=None):
    """Describe attested binding geometry without inventing motif or activity evidence."""
    modality = str(row.get("source_modality") or "").lower()
    boundary = _physical_element_explanation(row.get("physical_splice_elements"))
    if boundary:
        if modality and modality not in {"splice_mod", "steric"}:
            return boundary
        return (f"{boundary}. Steric occupancy at this annotated boundary could alter "
                "splice-site recognition; occupancy, direction and magnitude of the "
                "effect remain unmeasured.")
    if modality not in {"splice_mod", "steric"}:
        return ""
    accession = transcript_site.get("transcript_accession")
    exon = transcript_site.get("exon_ordinal")
    start = transcript_site.get("cdna_start")
    end = transcript_site.get("cdna_end")
    if accession and exon is not None and start is not None and end is not None:
        return (f"Physical target lies inside {accession} exon {exon}, mature-RNA "
                f"cDNA {start}–{end}. Steric masking could perturb exon recognition, "
                "but no site-specific regulatory motif or occupancy is established; "
                "the splice effect remains unmeasured.")
    if intronic_site:
        ordinal = intronic_site["intron_ordinal"]
        donor_start = intronic_site["donor_distance_start"]
        donor_end = intronic_site["donor_distance_end"]
        acceptor_start = intronic_site["acceptor_distance_start"]
        acceptor_end = intronic_site["acceptor_distance_end"]
        return (f"Physical target lies wholly inside {intronic_site['transcript_accession']} "
                f"intron {ordinal}, {donor_start}–{donor_end} intronic bases downstream "
                f"of exon {ordinal}'s donor and {acceptor_start}–{acceptor_end} "
                f"intronic bases upstream of exon {ordinal + 1}'s acceptor in "
                "transcript orientation. Steric occupancy "
                "could perturb splice-site recognition, but no regulatory motif, "
                "occupancy or direction of effect is established; the splice effect "
                "remains unmeasured.")
    return ""


def _bound_route_prediction_rationale(row):
    """Describe a selected molecule's own scored site without calling it ASO activity."""
    from .splice_recommendations import finite_number, prediction_evidence

    if str(row.get("source_modality") or "").lower() not in {"splice_mod", "steric"}:
        return ""
    annotations = row.get("source_annotations")
    site = row.get("predicted_response_site")
    if not isinstance(annotations, dict) or not isinstance(site, dict):
        return ""
    kind = site.get("site_kind")
    position = site.get("position")
    chrom = site.get("chrom")
    if (kind not in {"acceptor", "donor"} or not str(position).isdigit()
            or not isinstance(chrom, str) or not chrom.strip()):
        return ""
    if (annotations.get("primary_label") != f"{kind}@{position}"
            or str(annotations.get("selected_site_kind") or kind) != kind
            or str(annotations.get("selected_site_position") or position) != str(position)):
        return ""
    scored_chrom = annotations.get("selected_site_chrom")
    if (scored_chrom and chrom and str(scored_chrom).removeprefix("chr")
            != str(chrom).removeprefix("chr")):
        return ""
    direction = annotations.get("primary_direction")
    values = [finite_number(annotations.get("primary_delta_deletion")),
              finite_number(annotations.get("primary_delta_replacement"))]
    if direction not in {"increase", "decrease"} or any(value is None for value in values):
        return ""
    state = prediction_evidence(values, direction)["state"]
    interpretations = {
        "direction_supported": f"both have the sign of the requested {direction}",
        "opposing": f"both oppose the requested {direction}",
        "discordant": "their signs disagree",
        "zero_observed_change": "neither shows a signed change",
    }
    if state not in interpretations:
        return ""
    return (
        f"For this molecule's scored {kind} at {chrom}:{position}, the recorded local "
        f"sequence-deletion and replacement splice-score changes are {values[0]:+.6g} "
        f"and {values[1]:+.6g}; {interpretations[state]}. This candidate-specific "
        "perturbation proxy motivates testing the physical binding site, but it does "
        "not model ASO occupancy or establish an inclusion effect."
    )


def _physical_sequence_metrics(sequence):
    seq = str(sequence or "").upper()
    if not seq or set(seq) - set("ACGT"):
        return None, None
    longest = run = 0
    previous = None
    for base in seq:
        run = run + 1 if base == previous else 1
        longest = max(longest, run)
        previous = base
    return (seq.count("G") + seq.count("C")) / len(seq), longest


def _single_physical_interval_1based(row):
    """Convert one attested BED0 binding block to an inclusive genomic site."""
    intervals = row.get("binding_intervals_bed0_half_open") or []
    if len(intervals) != 1 or not isinstance(intervals[0], (list, tuple)):
        return None
    block = intervals[0]
    if (len(block) < 3 or not isinstance(block[0], str)
            or type(block[1]) is not int or type(block[2]) is not int
            or block[1] < 0 or block[2] <= block[1]):
        return None
    return block[0], block[1] + 1, block[2]


def _resolved_accession_and_record(goal, row=None):
    lock = goal.get("transcript_lock") if isinstance(goal.get("transcript_lock"), dict) else {}
    accession = str(goal.get("transcript_id") or lock.get("value") or "").strip()
    records = goal.get("isoform_transcripts") or goal.get("all_transcripts_full") or {}
    if not isinstance(records, dict):
        return accession, None
    def attested(value, fallback):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        versioned = str(metadata.get("versioned_id") or "").strip()
        base = str(value.get("transcript_id") or fallback).split(".", 1)[0]
        return versioned if versioned.split(".", 1)[0] == base and versioned else str(fallback)
    def attested_cross_reference(value, key, requested):
        """Accept a RefSeq alias only with a verified, structurally matching xref."""
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        resolutions = metadata.get("exon_nomenclature_resolutions") or {}
        if not isinstance(resolutions, dict):
            return False
        canonical = attested(value, key)
        exons = value.get("exons_tx") or []
        if not isinstance(exons, list) or not exons:
            return False
        lengths = [exon.get("end") - exon.get("start") + 1
                   for exon in exons if isinstance(exon, dict)
                   and type(exon.get("start")) is int and type(exon.get("end")) is int
                   and exon["end"] >= exon["start"]]
        if len(lengths) != len(exons):
            return False
        for resolution in resolutions.values():
            if not isinstance(resolution, dict) or resolution.get("status") != "verified_alternate_mapping":
                continue
            alias = str(resolution.get("refseq_accession") or "")
            if not alias or alias.split(".", 1)[0] != requested.split(".", 1)[0]:
                continue
            if "." in requested and requested != alias:
                continue
            source_id = str(resolution.get("source_ensembl_transcript_id") or "")
            mapped_id = str(resolution.get("ensembl_transcript_id") or "")
            if source_id != canonical or mapped_id != canonical.split(".", 1)[0]:
                continue
            if any(not isinstance(resolution.get(name), str)
                   or len(resolution[name]) != 64
                   or any(character not in "0123456789abcdef" for character in resolution[name].lower())
                   for name in ("xref_sha256", "source_sha256")):
                continue
            if (resolution.get("ensembl_exon_count") != len(exons)
                    or resolution.get("refseq_exon_count") != len(exons)
                    or resolution.get("ensembl_exon_lengths") != lengths
                    or resolution.get("refseq_exon_lengths") != lengths):
                continue
            return True
        return False
    explicit = str(goal.get("transcript_id") or "").strip()
    if not explicit and lock.get("kind") in {"explicit_id", "reference_accession", "accession", "transcript_id", "ensembl_transcript"}:
        explicit = str(lock.get("value") or "").strip()
    if explicit:
        matches = [(key, value) for key, value in records.items()
                   if str(key).split(".", 1)[0] == explicit.split(".", 1)[0]
                   and isinstance(value, dict)]
        if len(matches) == 1:
            key, value = matches[0]
            resolved = attested(value, key)
            # An explicitly requested accession version must not be silently
            # replaced by a different reference version with the same base ID.
            if "." in explicit:
                if "." in resolved and resolved != explicit:
                    return explicit, None
                resolved = explicit
            return resolved, value
        if not matches:
            aliases = [(key, value) for key, value in records.items()
                       if isinstance(value, dict)
                       and attested_cross_reference(value, key, explicit)]
            if len(aliases) == 1:
                key, value = aliases[0]
                return attested(value, key), value
        return explicit, None
    # A selected tiling job is stronger than a verbal transcript description:
    # it records the exact transcript whose physical windows produced this row.
    job_id = str((row or {}).get("job_id") or "")
    tiling = goal.get("gapmer_tiling_results_by_job") or {}
    job = tiling.get(job_id) if isinstance(tiling, dict) else None
    job_transcript = str((job or {}).get("transcript_id") or "") if isinstance(job, dict) else ""
    if job_transcript:
        matches = [(key, value) for key, value in records.items()
                   if str(key).split(".", 1)[0] == job_transcript.split(".", 1)[0]
                   and isinstance(value, dict)]
        if len(matches) == 1:
            key, value = matches[0]
            return attested(value, key), value
    if accession in records and isinstance(records[accession], dict):
        return attested(records[accession], accession), records[accession]
    if accession:
        matches = [value for key, value in records.items()
                   if str(key).split(".", 1)[0] == accession.split(".", 1)[0]
                   and isinstance(value, dict)]
        if len(matches) == 1:
            return attested(matches[0], accession), matches[0]
    if not accession and len(records) == 1:
        record = next(iter(records.values()))
        if isinstance(record, dict):
            return attested(record, next(iter(records))), record
    return accession, None


def _resolved_transcript_site(goal, row):
    """Map complete physical blocks to one contiguous mature-transcript site."""
    accession, record = _resolved_accession_and_record(goal, row)
    unresolved = {"transcript_accession": accession or None, "exon_ordinal": None,
                  "cdna_start": None, "cdna_end": None, "complete_coding_exon_site": None}
    if not accession or not isinstance(record, dict):
        return unresolved
    intervals = row.get("binding_intervals_bed0_half_open")
    if intervals:
        if not all(isinstance(block, (list, tuple)) and len(block) >= 3
                   and type(block[1]) is int and type(block[2]) is int
                   and 0 <= block[1] < block[2] for block in intervals):
            return unresolved
        blocks = [(block[0], block[1]+1, block[2]) for block in intervals]
    else:
        physical = _single_physical_interval_1based(row)
        blocks = [physical or (row.get("chrom"), row.get("start"), row.get("end"))]
    projected = []
    strand = record.get("strand")
    for chrom, start, end in blocks:
        if type(start) is not int or type(end) is not int or start > end:
            return unresolved
        covered = 0
        for exon in record.get("exons_tx") or []:
            if not isinstance(exon, dict):
                continue
            exon_chrom = str(exon.get("seq_region_name") or record.get("chrom") or "")
            if str(chrom).removeprefix("chr") != exon_chrom.removeprefix("chr"):
                continue
            left, right, cdna = exon.get("start"), exon.get("end"), exon.get("cdna_start")
            if not all(type(value) is int for value in (left, right, cdna)):
                continue
            a, b = max(start, left), min(end, right)
            if a > b:
                continue
            covered += b-a+1
            ca = cdna + (a-left if strand in (1, "+") else right-b)
            cb = ca+b-a
            coding_left, coding_right = exon.get("coding_start_genomic"), exon.get("coding_end_genomic")
            coding = (min(coding_left, coding_right) <= a <= b <= max(coding_left, coding_right)
                      if type(coding_left) is int and type(coding_right) is int else False)
            projected.append((ca, cb, exon.get("index"), coding))
        if covered != end-start+1:
            return unresolved
    projected.sort()
    if not projected or any(left[1]+1 != right[0] for left, right in zip(projected, projected[1:])):
        return unresolved
    ordinals = {item[2] for item in projected}
    return {"transcript_accession": accession,
            "exon_ordinal": next(iter(ordinals)) if len(ordinals) == 1 else None,
            "cdna_start": projected[0][0], "cdna_end": projected[-1][1],
            "complete_coding_exon_site": all(item[3] for item in projected)}


def _resolved_intronic_site(goal, row):
    """Name a wholly intronic physical site only between attested adjacent exons."""
    accession, record = _resolved_accession_and_record(goal, row)
    physical = _single_physical_interval_1based(row)
    if not accession or not isinstance(record, dict) or not physical:
        return None
    chrom, start, end = physical
    strand = record.get("strand")
    if strand not in (1, -1, "+", "-"):
        return None
    exons = []
    for exon in record.get("exons_tx") or []:
        if not isinstance(exon, dict):
            continue
        ordinal, left, right = exon.get("index"), exon.get("start"), exon.get("end")
        exon_chrom = str(exon.get("seq_region_name") or record.get("chrom") or "")
        if (type(ordinal) is not int or ordinal < 1 or type(left) is not int
                or type(right) is not int or left > right or not exon_chrom
                or chrom.removeprefix("chr") != exon_chrom.removeprefix("chr")):
            continue
        exons.append((ordinal, left, right))
    if len({ordinal for ordinal, _, _ in exons}) != len(exons):
        return None
    by_ordinal = {ordinal: (left, right) for ordinal, left, right in exons}
    matches = []
    for ordinal, (left, right) in by_ordinal.items():
        following = by_ordinal.get(ordinal + 1)
        if not following:
            continue
        next_left, next_right = following
        if strand in (1, "+") and right < start <= end < next_left:
            matches.append((ordinal, start - right, end - right,
                            next_left - end, next_left - start))
        elif strand in (-1, "-") and next_right < start <= end < left:
            matches.append((ordinal, left - end, left - start,
                            start - next_right, end - next_right))
    if len(matches) != 1:
        return None
    ordinal, donor_start, donor_end, acceptor_start, acceptor_end = matches[0]
    return {"transcript_accession": accession, "intron_ordinal": ordinal,
            "donor_distance_start": donor_start, "donor_distance_end": donor_end,
            "acceptor_distance_start": acceptor_start,
            "acceptor_distance_end": acceptor_end}


def translation_start_panel_evidence(goal, panel_csv):
    """Describe a start-codon panel using resolved annotation and physical rows.

    This deliberately reports only coordinate facts established by the selected
    transcript and CSV.  It does not infer occupancy or experimental inhibition.
    """
    if not isinstance(goal, dict) or goal.get("target_region") != "translation_start":
        return ""
    panel_csv = Path(panel_csv) if panel_csv else None
    if not panel_csv or not panel_csv.is_file():
        return ""
    with panel_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ""
    accession, record = _resolved_accession_and_record(goal)
    if not accession or not isinstance(record, dict):
        return ""
    if any(str(row.get("transcript_accession") or "") != accession for row in rows):
        return ""
    exons = [exon for exon in record.get("exons_tx") or [] if isinstance(exon, dict)]
    starts = [exon["coding_start"] for exon in exons
              if type(exon.get("coding_start")) is int and exon["coding_start"] > 0]
    if not starts:
        return ""
    codon_start = min(starts)
    codon_end = codon_start + 2
    codon_exon = next((exon for exon in exons
                       if type(exon.get("cdna_start")) is int
                       and type(exon.get("cdna_end")) is int
                       and exon["cdna_start"] <= codon_start <= codon_end <= exon["cdna_end"]), None)
    strand = record.get("strand")
    chrom = str(record.get("chrom") or "")
    if codon_exon and type(codon_exon.get("start")) is int and type(codon_exon.get("end")) is int:
        offset = codon_start - codon_exon["cdna_start"]
        if strand in (-1, "-"):
            genomic_end = codon_exon["end"] - offset
            genomic_start = genomic_end - 2
            strand_label = "-"
        elif strand in (1, "+"):
            genomic_start = codon_exon["start"] + offset
            genomic_end = genomic_start + 2
            strand_label = "+"
        else:
            genomic_start = genomic_end = None
            strand_label = "unknown"
    else:
        genomic_start = genomic_end = None
        strand_label = "unknown"
    sites = []
    for row in rows:
        try:
            start, end = int(row["transcript_cdna_start"]), int(row["transcript_cdna_end"])
        except (KeyError, TypeError, ValueError):
            return ""
        if start < 1 or end < start:
            return ""
        sites.append((start, end))
    unique_sites = set(sites)
    codon_overlaps = sum(start <= codon_end and end >= codon_start for start, end in sites)
    nonoverlap_pairs = sum(left_end < right_start or right_end < left_start
                           for index, (left_start, left_end) in enumerate(sites)
                           for right_start, right_end in sites[index + 1:])
    flanks = goal.get("flanks") if isinstance(goal.get("flanks"), dict) else {}
    try:
        upstream, downstream = int(flanks["up"]), int(flanks["down"])
    except (KeyError, TypeError, ValueError):
        upstream = downstream = None
    if upstream is not None and downstream is not None and upstream >= 0 and downstream >= 0:
        window_start, window_end = max(1, codon_start - upstream), codon_end + downstream
        within = sum(start >= window_start and end <= window_end for start, end in sites)
        window_text = (f"The resolved requested window is cDNA {window_start}–{window_end}; "
                       f"{within}/{len(sites)} selected binding windows are fully contained in it. ")
    else:
        window_text = ""
    genomic_text = (f" and {chrom}:{genomic_start}–{genomic_end} ({strand_label} strand)"
                    if chrom and genomic_start is not None else "")
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    annotation = str(metadata.get("reference_annotation") or "resolved transcript annotation")
    preview = sites[:20]
    site_lines = ["", "| CSV candidate | Mature-RNA binding site (1-based) | Overlaps start codon |",
                  "|---|---|---|"]
    site_lines.extend(
        f"| {index} | cDNA {start}–{end} | {'yes' if start <= codon_end and end >= codon_start else 'no'} |"
        for index, (start, end) in enumerate(preview, 1)
    )
    if len(sites) > len(preview):
        site_lines.extend(["", f"The CSV contains all {len(sites)} rows; this table shows the first {len(preview)}."])
    return (
        "## Translation-start evidence for the selected panel\n\n"
        f"Resolved transcript: `{accession}`. Its annotated translation-start codon "
        f"occupies mature-RNA cDNA {codon_start}–{codon_end}{genomic_text} "
        f"according to `{annotation}`. "
        f"{window_text}"
        f"The exported panel has {len(sites)} physical rows at {len(unique_sites)} "
        f"distinct mature-RNA intervals spanning cDNA {min(start for start, _ in sites)}–"
        f"{max(end for _, end in sites)}. {codon_overlaps}/{len(sites)} windows overlap "
        f"the annotated start codon; {nonoverlap_pairs} of "
        f"{len(sites) * (len(sites) - 1) // 2} site pairs do not overlap. "
        "These counts describe site diversity and geometry, not measured target "
        "occupancy, protein inhibition, efficacy, or safety.\n"
        + "\n".join(site_lines) + "\n"
    )


def _complete_chemistry_description(chemistry):
    if isinstance(chemistry, str):
        return chemistry
    if not isinstance(chemistry, dict):
        return None
    recommendation = chemistry.get("recommendation")
    if isinstance(recommendation, dict):
        chemistry = recommendation
    fields = (("architecture", "CHEMISTRY"), ("wing_sugar", "wing_chemistry"),
              ("gap_sugar", "gap_chemistry"), ("backbone", "backbone"),
              ("platform", "platform"))
    parts = [f"{label}={chemistry[key]}" for label, key in fields
             if chemistry.get(key) not in (None, "", "none")]
    return "; ".join(parts) or None


def _candidate_chemistry_description(row, goal):
    """Use the producing branch's chemical profile, never a sibling's summary."""
    program_id = row.get("program_id")
    profiles = goal.get("chemistry_profiles") or {}
    profile = (profiles.get("by_program") or {}).get(program_id)
    if profile:
        return _complete_chemistry_description(profile)
    programs = [p for p in goal.get("design_programs") or [] if isinstance(p, dict)]
    matches = [p for p in programs if program_id and p.get("program_id") == program_id]
    if len(matches) == 1 and matches[0].get("chemistry"):
        return _complete_chemistry_description(matches[0]["chemistry"])
    if len(programs) > 1:
        return None
    return _complete_chemistry_description(goal.get("chemistry"))


def unexecuted_design_branches(goal, branches):
    """Requested work cannot be replaced by a report; conditional fallbacks stay optional."""
    required = {key: value for key, value in branches.items() if value.get("requested")}
    pending = []
    for key, branch in required.items():
        if branch.get("status") in {"unknown", "skipped"}:
            programs = [program for program in goal.get("design_programs", []) if program.get("modality") == key]
            conditional = bool(programs) and all(
                (program.get("execution_policy") or {}).get("role") == "fallback" for program in programs)
            another_branch_delivered = any(other != key and value.get("status") == "outputs_present"
                                           for other, value in required.items())
            if not (conditional and another_branch_delivered):
                pending.append(key)
    return pending


def design_delivery(goal, candidate_set, branches, *, binding_sets=None, intervals_for_row=None):
    required = {key: value for key, value in branches.items() if value.get("requested")}
    errors = [f"requested_design_not_executed:{key}" for key in unexecuted_design_branches(goal, branches)]
    warnings = []
    for branch in required.values():
        if branch.get("status") == "evidence_unavailable":
            errors.append("requested_evidence_source_unavailable")
    selected_rows = list(candidate_set.get("rows") or [])
    rows = list(selected_rows)
    for group in (binding_sets or {}).get("sets") or []:
        rows.extend(group.get("rows") or [])
    molecules = set()
    for row in rows:
        target = str(row.get("target_binding_sequence") or row.get("sequence") or "").upper()
        molecule = str(row.get("synthesized_antisense_sequence") or row.get("Synthesis") or "").upper()
        if not molecule or set(molecule) - set("ACGT"):
            errors.append("candidate_synthesis_sequence_missing_or_invalid")
            continue
        if target and molecule != target.translate(str.maketrans("ACGT", "TGCA"))[::-1]:
            errors.append("candidate_orientation_inconsistent")
        molecules.add(molecule)
        if goal.get("target_accessibility") is True:
            accessibility = row.get("target_accessibility") or {}
            if accessibility.get("status") != "measured":
                warnings.append("requested_target_accessibility_not_measured")
    if required:
        minimum = goal.get("candidate_minimum")
        if minimum is None:
            minimum = 1
        if len(molecules) < minimum:
            errors.append("requested_candidate_count_not_delivered")
        if minimum == 0 and not molecules and any(
                branch.get("status") not in {"no_candidates_after_constraints", "no_candidates_found"}
                for branch in required.values()):
            errors.append("empty_result_search_completion_unestablished")
        maximum = goal.get("candidate_limit")
        selected_molecules = {str(row.get("synthesized_antisense_sequence") or row.get("Synthesis") or "").upper()
                              for row in selected_rows}
        selected_molecules.discard("")
        if maximum is not None and len(selected_molecules) > maximum:
            errors.append("requested_candidate_limit_exceeded")
    from .candidate_minimums import region_minimum_delivery
    region_delivery = region_minimum_delivery(goal, candidate_set, binding_sets,
        intervals_for_row or (lambda row, **_kwargs: row.get("binding_intervals_bed0_half_open") or []))
    errors.extend(region_delivery["errors"])
    return {"status": "unfulfilled" if errors else "fulfilled_with_limitations" if warnings else "fulfilled", "requested_design_branches": list(required),
            "candidate_rows": len(selected_rows), "delivered_binding_records": len(rows), "distinct_oligos": len(molecules),
            "per_site_minimum_delivery": region_delivery,
            "errors": list(dict.fromkeys(errors)), "warnings": list(dict.fromkeys(warnings))}


def write_design_panel(candidate_set, goal, output_directory):
    """A concise export generated directly from selected candidate evidence."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "design_panel.csv"
    markdown_path = directory / "design_panel.md"
    rows = ordered_rows(candidate_set.get("rows") or [])
    distinct = distinct_ranked_rows(rows)
    preview = distinct if len(distinct) <= 20 else distinct[:10]
    exported = []
    lines = ["# Candidate panel", "", f"Selected candidate rows: {len(rows)}. Distinct molecules (antisense sequence plus chemistry): {len(distinct)}. Preview: {len(preview)} molecules in recorded rank order.", "",
             "Sequences are written 5′→3′. These are computational design candidates; experimental activity and in-cell accessibility have not been established.", "",
             "| Candidate | Native rank / scope | Antisense oligo (5′→3′) | Physical binding interval(s), BED0 | Target genomic strand | Physical splice element | Predicted response site | Genomic screening |",
             "|---|---|---|---|---|---|---|---|"]
    for index, row in enumerate(rows, 1):
        annotations = row.get("source_annotations") or {}
        access = row.get("target_accessibility") or {}
        physical_sequence = row.get("synthesized_antisense_sequence")
        physical_gc, homopolymer = _physical_sequence_metrics(physical_sequence)
        transcript_site = _resolved_transcript_site(goal, row)
        intronic_site = _resolved_intronic_site(goal, row)
        splice_rationale = _physical_splice_site_rationale(row, transcript_site, intronic_site)
        route_prediction_rationale = _bound_route_prediction_rationale(row)
        if route_prediction_rationale:
            splice_rationale = (splice_rationale + " " + route_prediction_rationale).strip()
        physical_interval = _single_physical_interval_1based(row)
        genomic_start, genomic_end = ((physical_interval[1], physical_interval[2]) if physical_interval else
                                      (None, None) if row.get("binding_intervals_bed0_half_open") else
                                      (row.get("start"), row.get("end")))
        if type(genomic_start) is not int or type(genomic_end) is not int:
            genomic_start = genomic_end = None
        chemistry = _candidate_chemistry_description(row, goal)
        record = {"candidate": index, "antisense_5to3": row.get("synthesized_antisense_sequence"),
                  "program_id": row.get("program_id"), "source_modality": row.get("source_modality"),
                  "target_5to3": row.get("target_binding_sequence"), "length": row.get("length"),
                  "gc_fraction": physical_gc,
                  "max_homopolymer_run_nt": homopolymer,
                  "chemistry_description": chemistry,
                  "gapmer_preservation_assessment": json.dumps(row.get("gapmer_preservation_assessment") or annotations.get("gapmer_preservation_assessment") or {}, sort_keys=True),
                  "transcript_accession": transcript_site["transcript_accession"],
                  "transcript_cdna_start": transcript_site["cdna_start"],
                  "transcript_cdna_end": transcript_site["cdna_end"],
                  "exon_ordinal": transcript_site["exon_ordinal"],
                  "intron_ordinal": intronic_site["intron_ordinal"] if intronic_site else None,
                  "intron_donor_distance_start_1based": intronic_site["donor_distance_start"] if intronic_site else None,
                  "intron_donor_distance_end_1based": intronic_site["donor_distance_end"] if intronic_site else None,
                  "intron_acceptor_distance_start_1based": intronic_site["acceptor_distance_start"] if intronic_site else None,
                  "intron_acceptor_distance_end_1based": intronic_site["acceptor_distance_end"] if intronic_site else None,
                  "complete_coding_exon_site": transcript_site["complete_coding_exon_site"],
                  "genomic_chrom": (physical_interval[0] if physical_interval else
                      next(iter({block[0] for block in row.get("binding_intervals_bed0_half_open") or []}), None)
                      if len({block[0] for block in row.get("binding_intervals_bed0_half_open") or []}) == 1
                      else None if row.get("binding_intervals_bed0_half_open") else row.get("chrom")),
                  "genomic_start_1based": genomic_start,
                  "genomic_end_1based": genomic_end,
                  "target_genomic_strand": row.get("strand") if row.get("strand") in {"+", "-"} else None,
                  "native_rank": row.get("selection_rank"),
                  "ranking_scope": row.get("ranking_scope") or row.get("job_id") or row.get("source_path"),
                  "physical_binding_intervals_bed0": json.dumps(row.get("binding_intervals_bed0_half_open") or [], sort_keys=True),
                  "binding_region_membership": json.dumps(row.get("binding_region_membership") or [], sort_keys=True),
                  "physical_splice_elements": json.dumps(row.get("physical_splice_elements") or [], sort_keys=True),
                  "physical_element_explanation": splice_rationale,
                  "predicted_response_site": json.dumps(row.get("predicted_response_site") or {}, sort_keys=True),
                  "recommendation_status": row.get("recommendation_status"),
                  "recommendation_note": row.get("recommendation_note") or splice_rationale,
                  "idt_chemistry": row.get("IDT") or annotations.get("IDT") or annotations.get("chemistry_idt"),
                  "target_accessibility_min_probability": row.get("target_accessibility_min_probability"),
                  "target_accessibility_by_species": json.dumps(access.get("species_scores") or {}, sort_keys=True),
                  "target_binding_sites": json.dumps({key: [{k: value for k, value in hit.items() if k != "evidence"} for hit in hits]
                                                     for key, hits in (access.get("measurements") or {}).items()}, sort_keys=True),
                  "cross_species_matches": json.dumps(row.get("cross_species_matches") or {}, sort_keys=True),
                  "cross_species_match_placements": json.dumps(row.get("cross_species_match_placements") or {}, sort_keys=True),
                  "cross_species_pass": row.get("cross_species_pass"),
                  "off_target_screening_status_by_species": json.dumps(row.get("off_target_screening_status_by_species") or {}, sort_keys=True),
                  "off_target_search_provenance_by_species": json.dumps(row.get("off_target_search_provenance_by_species") or {}, sort_keys=True),
                  "off_target_by_species": json.dumps(row.get("off_target_by_species") or {}, sort_keys=True),
                  "off_target_search_status": row.get("off_target_search_status") or annotations.get("off_target_search_status"),
                  "off_target_mismatch_budget": row.get("Off_target_mismatch_budget") if row.get("Off_target_mismatch_budget") is not None else annotations.get("Off_target_mismatch_budget"),
                  "off_target_exact_count": row.get("Off_target_exact_count") if row.get("Off_target_exact_count") is not None else annotations.get("Off_target_exact_count"),
                  "off_target_1mm_count": row.get("Off_target_1mm_count") if row.get("Off_target_1mm_count") is not None else annotations.get("Off_target_1mm_count"),
                  "off_target_2mm_count": row.get("Off_target_2mm_count") if row.get("Off_target_2mm_count") is not None else annotations.get("Off_target_2mm_count"),
                  "off_target_3mm_count": row.get("Off_target_3mm_count") if row.get("Off_target_3mm_count") is not None else annotations.get("Off_target_3mm_count"),
                  "off_target_exact_count_unit": row.get("off_target_exact_count_unit") or annotations.get("off_target_exact_count_unit"),
                  "off_target_position_convention": row.get("off_target_position_convention") or annotations.get("off_target_position_convention"),
                  "requested_exon_overlap_bp": row.get("requested_exon_overlap_bp"),
                  "requested_anchor_distance_bp": row.get("requested_anchor_distance_bp")}
        exported.append(record)
        if any(row is item for item in preview):
            scope = row.get("ranking_scope") or row.get("job_id") or "source ordering"
            rank = row.get("selection_rank") if row.get("selection_rank") is not None else "unavailable"
            screen = str(record['off_target_search_status'] or 'unknown')
            budget = record['off_target_mismatch_budget']
            exact = record['off_target_exact_count']
            one_mm = record['off_target_1mm_count']
            if budget is not None:
                screen += f"; searched ≤{budget} mismatch(es)"
            if exact is not None:
                screen += f"; exact={exact} {record['off_target_exact_count_unit'] or 'hits'}"
            if one_mm is not None:
                screen += f"; 1mm={one_mm} recorded hits"
            lines.append(f"| {index} | {rank} / {scope} | {record['antisense_5to3'] or 'unverified'} | {record['physical_binding_intervals_bed0']} | {record['target_genomic_strand'] or 'unresolved'} | {record['physical_element_explanation'] or 'No annotated boundary crossing established'} | {record['predicted_response_site']} | {screen} |")
    if exported:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
            writer.writeheader(); writer.writerows(exported)
    else:
        csv_path.write_text("candidate,antisense_5to3,target_5to3\n")
    lines.extend(["", "## Selection and evidence", "",
                  "Selection policy: " + str((candidate_set.get("candidate_selection") or {}).get("method") or "recorded native ranking") + ".",
                  "", "The preview contains one row per physical molecule; the full CSV retains every binding/route observation. Native ranks apply within their recorded scope and do not establish experimental potency. Physical splice-element labels describe annotated exon/intron geometry crossed by the ASO, not motif occupancy or measured activity. Predicted response coordinates identify the site whose perturbation was scored; physical binding coordinates identify where the ASO binds.",
                  "", "Full sequences, transcript positions, chemical notation, exact species matches and the actual off-target screening coverage are in [design_panel.csv](design_panel.csv). The complete per-candidate evidence and source paths are in [generated_aso_set.json](generated_aso_set.json). Missing screening data must not be interpreted as zero off-targets."])
    protected_rows = [(index, row.get("gapmer_preservation_assessment") or
                       (row.get("source_annotations") or {}).get("gapmer_preservation_assessment"))
                      for index, row in enumerate(rows, 1)]
    if any(assessment for _, assessment in protected_rows):
        lines.extend(["", "## Protected-transcript sequence discrimination", "",
                      "Each selected molecule was compared against the complete resolved protected transcripts. "
                      "The table gives all equally closest ungapped alignments, with mismatch positions counted "
                      "in the synthesized ASO's 5′→3′ direction. The conservative shortlist excludes perfect "
                      "protected-transcript complements and closest alignments with an entirely matched DNA gap. "
                      "A mismatch in the gap supports sequence discrimination, not demonstrated RNase H or cellular "
                      "selectivity. Bulges, binding kinetics and activity are not measured; both desired knockdown "
                      "and protected-isoform sparing require experimental testing.", "",
                      "| Candidate | Protected transcript | Closest site (cDNA, 1-based) | Mismatches | ASO mismatch positions | DNA-gap mismatch positions | Actual DNA gap |",
                      "|---|---|---|---|---|---|---|"])
        for index, assessment in protected_rows:
            if not assessment:
                continue
            for reference in assessment.get("protected_references") or []:
                for site in reference.get("closest_sites") or []:
                    lines.append(f"| {index} | {reference['transcript_label']} | "
                                 f"{site['transcript_cdna_start_1based']}–{site['transcript_cdna_end_1based']} | "
                                 f"{site['mismatch_count']} | {site['mismatch_positions_aso_1based']} | "
                                 f"{site['dna_gap_mismatch_positions_aso_1based']} | {assessment['dna_gap_positions_aso_1based']} |")
    if any(row.get("cross_species_match_placements") for row in rows):
        lines.extend(["", "## Exact cross-species target placements", "",
                      "Reference positions are one-based. Genomic blocks are shown only when mapped through the exons of the same searched transcript; an unavailable mapping is not inferred from a cDNA hit. Exact sequence match does not establish activity.", "",
                      "| Candidate | Species | Reference | Reference position | Scope | Assembly / strand | Genomic blocks (1-based) | Mapping status |",
                      "|---|---|---|---|---|---|---|---|"])
        for index, row in enumerate(rows, 1):
            for species, placements in (row.get("cross_species_match_placements") or {}).items():
                for placement in placements:
                    blocks = placement.get("genomic_segments_1based_inclusive") or []
                    block_text = "; ".join(
                        f"{block.get('chrom')}:{block.get('start')}–{block.get('end')}"
                        for block in blocks
                    ) or "unavailable"
                    lines.append(
                        f"| {index} | {species} | {placement.get('transcript_versioned_id') or placement.get('reference_id')} | "
                        f"{placement.get('start_1based')}–{placement.get('end_1based')} | {placement.get('coordinate_scope')} | "
                        f"{placement.get('assembly') or 'unresolved'} / {placement.get('target_strand') or 'unresolved'} | "
                        f"{block_text} | {placement.get('genomic_mapping_status') or 'not_applicable'} |"
                    )
    search_sources = next((row.get("off_target_search_provenance_by_species") for row in rows
                           if row.get("off_target_search_provenance_by_species")), None)
    if isinstance(search_sources, dict):
        lines.extend(["", "## Off-target search references", "",
                      "The recorded zero-mismatch search used these species-specific references; this is a search-scope statement, not evidence of experimental specificity.", "",
                      "| Species | Assembly | Genomic index | Exon annotation | Junction search |",
                      "|---|---|---|---|---|"])
        for species, source in sorted(search_sources.items()):
            if not isinstance(source, dict):
                continue
            lines.append(
                f"| {species} | {source.get('assembly') or 'unresolved'} | "
                f"{Path(str(source.get('genomic_index') or '')).name or 'unresolved'} | "
                f"{Path(str(source.get('exon_annotation') or '')).name or 'unresolved'} | "
                f"{source.get('junction_search') or 'unresolved'} |"
            )
    lines.append("Genomic screening counts report the recorded query, mismatch budget, and hit unit; they do not establish binding to transcribed RNA or cellular specificity. Empty count cells mean unsearched or unavailable, not zero. Exon overlap and anchor distance describe physical binding, not the scored splice response coordinate.")
    if goal.get("target_accessibility") is True:
        measured = sum((row.get("target_accessibility") or {}).get("status") == "measured" for row in rows)
        if measured == len(rows) and rows:
            lines.extend(["", "RNA accessibility was estimated with [ViennaRNA RNAplfold](https://viennarna.readthedocs.io/en/latest/man/RNAplfold.html) on the recorded full transcript sequences. The score is the probability that the entire target binding window is unpaired. Parameters, tool version, sequence hashes and raw matrices are preserved with each measurement. No empirical efficacy cutoff is imposed."])
        else:
            lines.extend(["", f"Target-RNA accessibility was measured for {measured} of {len(rows)} selected rows; missing measurements remain unavailable and cannot support accessibility ranking."])
    markdown_path.write_text("\n".join(lines) + "\n")
    return {"design_panel_csv": str(csv_path), "design_panel_md": str(markdown_path)}
