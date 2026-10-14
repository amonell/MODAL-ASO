"""Source-backed comparison of submitted physical ASO sets against one target."""
from __future__ import annotations

from typing import Any, Dict, List
import csv
import json
from pathlib import Path


SUBMITTED_SET_COLUMNS = (
    "set_id", "candidate_id", "physical_aso_5to3", "chemistry", "length",
    "gc_fraction", "max_homopolymer_run_nt", "written_molecule_assessment",
    "requested_reference_site_or_nearest_hypothesis", "antisense_mismatches",
    "same_strand_exact_site", "same_strand_site", "separate_reverse_complement_aso_5to3",
    "reverse_complement_alternative_site", "reverse_complement_matches_submitted_candidate",
    "exploratory_priority", "off_target_exact_count",
    "off_target_count_unit", "off_target_mismatch_budget", "off_target_count_status",
    "off_target_count_scope",
)


def write_submitted_set_table(path: Path, evidence: List[Dict[str, Any]]) -> None:
    """Export one row per submitted physical molecule from the checked report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMITTED_SET_COLUMNS)
        writer.writeheader()
        for item in evidence:
            scope = item.get("off_target_scope") if isinstance(item.get("off_target_scope"), dict) else {}
            writer.writerow({
                "set_id": item.get("set_id"), "candidate_id": item.get("name"),
                "physical_aso_5to3": item.get("sequence"), "chemistry": item.get("chemistry"),
                "length": item.get("length"), "gc_fraction": item.get("gc_fraction"),
                "max_homopolymer_run_nt": item.get("homopolymer"),
                "written_molecule_assessment": item.get("status"),
                "requested_reference_site_or_nearest_hypothesis": item.get("site"),
                "antisense_mismatches": item.get("mismatch"),
                "same_strand_exact_site": item.get("same_strand_exact"),
                "same_strand_site": item.get("same_strand_site"),
                "separate_reverse_complement_aso_5to3": item.get("reverse_complement_alternative"),
                "reverse_complement_alternative_site": item.get("reverse_complement_alternative_site"),
                "reverse_complement_matches_submitted_candidate": item.get("reverse_complement_submitted_match"),
                "exploratory_priority": item.get("exploratory_rank"),
                "off_target_exact_count": item.get("off_target_exact_count"),
                "off_target_count_unit": scope.get("count_unit"),
                "off_target_mismatch_budget": item.get("off_target_mismatch_budget"),
                "off_target_count_status": item.get("off_target_count_status"),
                "off_target_count_scope": json.dumps(scope, sort_keys=True, ensure_ascii=False),
            })


def _cell(value: Any) -> str:
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def _target_hit(row: Dict[str, Any], transcript: str, orientation: str) -> Dict[str, Any]:
    return next((hit for hit in row.get("mapping_hits") or []
                 if isinstance(hit, dict)
                 and str(hit.get("transcript_id") or "").split(".", 1)[0] == transcript
                 and hit.get("mapping_scope") == "spliced_cdna"
                 and hit.get("match_orientation") == orientation), {})


def _near_hit(row: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    hits = [hit for hit in row.get("near_match_candidates") or []
            if isinstance(hit, dict)
            and str(hit.get("transcript_id") or "").split(".", 1)[0] == transcript
            and hit.get("rna_context") == "mature_spliced_cdna"
            and hit.get("match_orientation") == "reverse_complement"
            and type(hit.get("mismatch_count")) is int]
    return min(hits, key=lambda hit: hit["mismatch_count"]) if hits else {}


def _site(hit: Dict[str, Any], row: Dict[str, Any], *, antisense: bool) -> str:
    if not hit:
        return "not established"
    identity = hit.get("reference_identity") or {}
    transcript = identity.get("transcript_id") or identity.get("versioned_id") or hit.get("transcript_id")
    cdna_start, cdna_end = hit.get("cdna_start"), hit.get("cdna_end")
    cdna = f"cDNA {cdna_start}–{cdna_end}" if all(type(v) is int for v in (cdna_start, cdna_end)) else "cDNA position unavailable"
    segments = hit.get("genomic_segments") or []
    genome = ", ".join(f"{part.get('chrom')}:{part.get('start')}–{part.get('end')}"
                       for part in segments if isinstance(part, dict)) or "genomic segments unavailable"
    exon_text = ""
    membership = row.get("principal_reference_exon_membership")
    if antisense and isinstance(membership, dict):
        recorded_transcript = str(membership.get("transcript_id") or "").split(".", 1)[0]
        expected_mismatches = hit.get("mismatch_count", 0)
        if (membership.get("status") == "resolved"
                and recorded_transcript == str(transcript or "").split(".", 1)[0]
                and type(membership.get("alignment_mismatches")) is int
                and membership["alignment_mismatches"] == expected_mismatches):
            exons = membership.get("exons") if isinstance(membership.get("exons"), list) else []
            ordinals = [str(item["transcript_exon_ordinal"]) for item in exons
                        if isinstance(item, dict) and item.get("transcript_exon_ordinal") is not None]
            if ordinals:
                exon_text = f"; exon ordinal(s) {', '.join(ordinals)}"
    return f"{transcript}; {cdna}{exon_text}; {genome} ({hit.get('strand') or 'strand unknown'})"


def submitted_set_response(compare_sets: List[Dict[str, Any]], evaluation: Dict[str, Any],
                           comparison: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return a bounded factual report, or defer if reference evidence is unavailable.

    This path applies only to distinct user-supplied sets. It never interprets
    reverse complements as the submitted physical sequence, and it treats
    near matches as exploratory hypotheses rather than confirmed binding.
    """
    if len(compare_sets) < 2 or not all(
        item.get("source_type") == "provided" and isinstance(item.get("rows"), list) and item["rows"]
        for item in compare_sets
    ):
        return None
    rows = evaluation.get("provided_asos")
    if not isinstance(rows, list) or not rows:
        return None
    reference = evaluation.get("context_summary") or {}
    transcript = str(reference.get("transcript_id") or "").split(".", 1)[0]
    if not transcript or not (evaluation.get("mapping_strategy") or {}).get("searched_spliced_cdna"):
        return None

    evidence = []
    for item in compare_sets:
        for row in item.get("rows") or []:
            if not isinstance(row, dict):
                continue
            exact = _target_hit(row, transcript, "reverse_complement")
            same_strand = _target_hit(row, transcript, "input_sequence")
            near = _near_hit(row, transcript) if not exact else {}
            if exact:
                tier = (0, 0)
                status = "exact antisense site on the requested mature transcript"
                site = _site(exact, row, antisense=True)
                mismatch = "0"
            elif near and near["mismatch_count"] > 0:
                tier = (1, near["mismatch_count"])
                status = ("same-strand exact site exists; only a separate antisense near-match hypothesis is present; binding is unverified"
                          if same_strand else "near-match hypothesis; binding is unverified")
                site = _site(near, row, antisense=True)
                mismatch = str(near["mismatch_count"])
            elif same_strand:
                tier = (2, 0)
                status = "same-strand exact site only; physical ASO is not a confirmed antisense binder"
                site = _site(same_strand, row, antisense=False)
                mismatch = "not an antisense exact match"
            else:
                tier = (3, 0)
                status = "no supported antisense site in the searched mature transcript"
                site = "not established"
                mismatch = "unknown"
            metadata = row.get("input_metadata") if isinstance(row.get("input_metadata"), dict) else {}
            sequence = str(row.get("sequence") or "").upper()
            alternate = (sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                         if same_strand and sequence and set(sequence) <= set("ACGT") else None)
            sense_site = _site(same_strand, row, antisense=False) if same_strand else None
            evidence.append({
                "set_id": item.get("set_id"), "name": metadata.get("candidate_id") or row.get("name"),
                "sequence": row.get("sequence"), "chemistry": row.get("chemistry"),
                "length": row.get("length"), "gc_fraction": row.get("gc_fraction"),
                "homopolymer": row.get("maximum_homopolymer_run_nt"),
                "status": status, "site": site, "mismatch": mismatch,
                "same_strand_exact": bool(same_strand), "tier": tier,
                "same_strand_site": sense_site,
                "reverse_complement_alternative": alternate,
                "reverse_complement_alternative_site": sense_site if alternate else None,
                "off_target_exact_count": row.get("Off_target_exact_count"),
                "off_target_count_status": row.get("off_target_count_status"),
                "off_target_mismatch_budget": row.get("Off_target_mismatch_budget"),
                "off_target_scope": row.get("off_target_count_scope") or {},
            })
    if not evidence:
        return None
    submitted_sequences = {}
    for item in evidence:
        submitted_sequences.setdefault(str(item["sequence"] or "").upper(), []).append(
            f"{item['set_id']}/{item['name']}")
    for item in evidence:
        alternate = item.get("reverse_complement_alternative")
        item["reverse_complement_submitted_match"] = (
            ", ".join(submitted_sequences.get(alternate, [])) if alternate else None)
    ordered = sorted(evidence, key=lambda item: (item["tier"], str(item["set_id"]), str(item["name"])))
    for index, item in enumerate(ordered, 1):
        item["exploratory_rank"] = (index if index == 1 or item["tier"] != ordered[index - 2]["tier"]
                                     else ordered[index - 2]["exploratory_rank"])
    lines = ["# Submitted ASO set comparison", "",
             f"Requested reference: {reference.get('gene_symbol') or 'target gene'}, "
             f"{transcript} (versioned accession is recorded per verified alignment below).",
             "",
             "Exact set overlap counts refer to the submitted physical 5′→3′ molecules; reverse complements are separate diagnostic hypotheses."]
    for pair in comparison.get("pairwise_overlaps") or []:
        lines.append(f"- {_cell(pair.get('set_a_label'))} vs {_cell(pair.get('set_b_label'))}: "
                     f"{int(pair.get('shared_count') or 0)} identical submitted sequence(s).")
    lines.extend(["", "| Set | Candidate | Submitted ASO 5′→3′ | Written-molecule assessment | Requested-reference site or nearest hypothesis | Mismatches | Length | GC | Max homopolymer | Exploratory priority |",
                  "|---|---|---|---|---|---|---:|---:|---:|---:|"])
    for item in evidence:
        lines.append("| " + " | ".join(_cell(value) for value in (
            item["set_id"], item["name"], item["sequence"], item["status"], item["site"],
            item["mismatch"], item["length"], item["gc_fraction"], item["homopolymer"],
            item["exploratory_rank"])) + " |")
    alternatives = [item for item in evidence if item.get("same_strand_site")]
    if alternatives:
        lines.extend(["", "## Exact same-strand sites and separate ASO alternatives", "",
                      "A same-strand match is not binding evidence for the written physical ASO. "
                      "The listed reverse complement is a different hypothetical molecule unless it was separately submitted."])
        for item in alternatives:
            lines.append(f"- {_cell(item['set_id'])}/{_cell(item['name'])}: written-molecule same-strand site "
                         f"{_cell(item['same_strand_site'])}; separate reverse-complement ASO 5′→3′ "
                         f"{_cell(item['reverse_complement_alternative'])} would be exactly antisense at that site. "
                         f"Submitted match: {_cell(item['reverse_complement_submitted_match'] or 'none')}. "
                         "Do not substitute this alternative for the supplied molecule.")
    lines.extend(["", "Priority uses verified antisense complementarity to the requested mature transcript first, then the smallest recorded near-match distance. Equal evidence tiers are tied. This is not a measured potency ranking.",
                  "", "## Chemistry and screening scope"])
    for item in evidence:
        scope = item["off_target_scope"] if isinstance(item["off_target_scope"], dict) else {}
        lines.append(f"- {_cell(item['set_id'])}/{_cell(item['name'])}: supplied chemistry "
                     f"{_cell(item['chemistry'])}; exact non-target count "
                     f"{_cell(item['off_target_exact_count'])} "
                     f"({_cell(scope.get('count_unit') or 'declared count unit')}); mismatch budget "
                     f"{_cell(item['off_target_mismatch_budget'])}; count status "
                     f"{_cell(item['off_target_count_status'])}. "
                     "Any higher mismatch bins or biological effects not measured by that scope remain unknown.")
    lines.extend(["", "The screening counts are limited to their recorded reference universe and do not establish genome-wide safety. No knockdown or cellular activity was measured by this comparison."])
    summary = (f"Compared {len(compare_sets)} submitted sets and {len(evidence)} physical molecules "
               f"against {transcript}; {sum(item['tier'][0] == 0 for item in evidence)} "
               "have an exact antisense site on the requested mature transcript. "
               "Exploratory priorities and unmeasured scope are shown per candidate.")
    return {"answer_summary": summary, "markdown": "\n".join(lines),
            "assumptions": [], "limitations": [
                "An exact or near-match sequence site does not establish biological efficacy.",
                "Screening claims apply only to the recorded reference and mismatch budget."],
            "evidence_refs": ["provided_aso_evaluation", "comparison_summary"],
            "suggested_followups": [], "submitted_set_evidence": evidence}
