"""Presentation views of saved evaluations; never merge incompatible alignments.

The reference catalog is a search denominator, not a same-gene isoform census.
Absence of an exact site does not change the submitted molecule's identity.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import json


def _dict(value):
    return value if isinstance(value, dict) else {}


def alignment_export_observation(hit):
    """Keep recorded alignment facts together without expanding reference catalogs."""
    return {key: deepcopy(hit[key]) for key in (
        'transcript_id', 'transcript_version', 'versioned_transcript_id',
        'reference_build', 'reference_source', 'reference_annotation',
        'mapping_scope', 'rna_context', 'match_orientation', 'strand', 'coordinate_system',
        'cdna_start', 'cdna_end', 'locus_relative_start', 'locus_relative_end',
        'genomic_segments', 'genomic_span', 'matched_target_sequence', 'matched_window_sequence',
        'mismatch_count', 'mismatch_positions_in_supplied_sequence_1based',
        'mismatch_positions_in_target_sequence_1based', 'mismatch_details_in_target_orientation',
        'mismatch_annotation_coordinate_system', 'search_scope', 'binding_interpretation',
    ) if key in hit}


def reference_groups(row, catalog):
    """Partition *all* recorded alignments by reference, RNA context and strand."""
    identities = _dict(_dict(catalog).get("transcript_identities"))
    groups = {}
    for kind, records in (("exact", row.get("mapping_hits")),
                          ("near", row.get("near_match_candidates"))):
        for index, hit in enumerate(records or []):
            if not isinstance(hit, dict):
                continue
            identity = {**_dict(identities.get(hit.get("transcript_id"))), **_dict(hit.get("reference_identity"))}
            key = (str(identity.get("transcript_id") or hit.get("transcript_id") or "unknown"),
                   str(identity.get("species") or "unknown"),
                   str(identity.get("assembly") or "unknown"),
                   str(hit.get("mapping_scope") or "unknown"),
                   str(hit.get("match_orientation") or "unknown"))
            group = groups.setdefault(key, dict(zip(
                ("transcript_id", "species", "assembly", "mapping_scope", "match_orientation"), key),
                exact_hit_count=0, near_hit_count=0, best_near_mismatch_count=None,
                best_near_alignments=[], near_alignments=[], exact_alignments=[]))
            # Geometry remains attached to its own source observation. Never
            # combine a mature-RNA count with a locus coordinate or other strand.
            observation = alignment_export_observation(hit)
            observation["source_pointer"] = f"/{'mapping_hits' if kind == 'exact' else 'near_match_candidates'}/{index}"
            if kind == "exact":
                group["exact_hit_count"] += 1
                group["exact_alignments"].append(observation)
            else:
                group["near_hit_count"] += 1
                group["near_alignments"].append(observation)
                count = hit.get("mismatch_count")
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                    best = group["best_near_mismatch_count"]
                    if best is None or count < best:
                        group["best_near_mismatch_count"] = count
                        group["best_near_alignments"] = [observation]
                    elif count == best:
                        group["best_near_alignments"].append(observation)
    return list(groups.values())


def evaluation_projection(payload):
    """One supplied-observation row plus complete context-specific binding views."""
    catalog = _dict(payload.get("reference_catalog"))
    identities = _dict(catalog.get("transcript_identities"))
    result = []
    for source in payload.get("provided_asos") or []:
        if not isinstance(source, dict):
            continue
        orientation = _dict(source.get("orientation_interpretation"))
        scope = _dict(source.get("off_target_count_scope"))
        groups = reference_groups(source, catalog)
        target_input = orientation.get("provided_sequence_role") == "target_site_sequence"
        physical_orientation = "input_sequence" if target_input else "reverse_complement"
        row = {
            "aso_id": source.get("aso_id") or source.get("name"),
            "input_row_id": source.get("input_row_id"),
            "source_file": source.get("source_file"), "source_row": source.get("source_row"),
            "row_identity_scope": "One recorded input observation; repeated sequences are not automatically merged across chemistry or source identity.",
            "input_sequence_5to3": source.get("sequence"),
            "evaluated_molecule_sequence_5to3": orientation.get("evaluated_molecule_sequence") or source.get("sequence"),
            "physical_identity_policy": "confirmed target-site input" if target_input else "as-received molecule; no sequence replacement",
            "requires_sequence_confirmation": bool(orientation.get("requires_confirmation")),
            "length": source.get("length"), "gc_fraction": source.get("gc_fraction"),
            "longest_homopolymer": source.get("maximum_homopolymer_run_nt"),
            "chemistry": source.get("chemistry"),
            "recorded_screened_gene_count": source.get("off_target_gene_count"),
            "recorded_screened_gene_ids": source.get("Off_target_gene_ids"),
            "recorded_screened_gene_symbols": source.get("Off_target_genes"),
            "screen_exclusion_applied": scope.get("exclusion_applied"),
            "screen_excluded_gene_id": scope.get("excluded_gene_id"),
            "screen_count_status": source.get("off_target_count_status"),
            "screen_query_sequence": source.get("off_target_query_sequence"),
            "screen_query_role": source.get("off_target_query_role"),
            "screen_scope": deepcopy(scope),
            "near_match_search_scope": deepcopy(source.get("near_match_search")),
            "reference_alignment_groups": groups,
            "reference_catalog_scope": "supplied search catalog; not a whole-gene isoform census",
        }
        for key, identity in identities.items():
            identity = _dict(identity)
            tid = str(identity.get("transcript_id") or key)
            prefix = str(identity.get("species") or "unknown") + "_" + tid + "_mature_antisense"
            selected = [g for g in groups if g["transcript_id"] == tid
                        and g["mapping_scope"] == "spliced_cdna"
                        and g["match_orientation"] == physical_orientation]
            row[prefix + "_recorded_exact_hits"] = sum(g["exact_hit_count"] for g in selected)
            near = [g["best_near_mismatch_count"] for g in selected
                    if g["best_near_mismatch_count"] is not None]
            row[prefix + "_best_recorded_mismatches"] = min(near) if near else None
            row[prefix + "_evidence_status"] = (
                "exact" if row[prefix + "_recorded_exact_hits"] else
                "mismatched only" if near else "no recorded site; search completeness must be checked")
        result.append(row)
    return result


def _cell(value):
    if value is None:
        return "not established"
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", " ")


def projection_markdown(payload):
    rows = evaluation_projection(payload)
    lines = ["# Computational ASO evaluation", "", str(payload.get("request_summary") or "Saved candidate evidence"), "",
             "Input molecule identity is preserved independently of binding success. No exact site does not imply an incorrectly supplied sequence. "
             "An explicitly required orientation confirmation remains required. Counts below refer to recorded alignments; mature RNA, locus RNA and same-strand matches are separate.", "",
             "| ASO | Input bases 5′→3′ | Evaluated molecule 5′→3′ | Length | GC fraction | Longest homopolymer | Chemistry |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[k]) for k in (
            "aso_id", "input_sequence_5to3", "evaluated_molecule_sequence_5to3",
            "length", "gc_fraction", "longest_homopolymer", "chemistry")) + " |")
    lines += ["", "## Reference-specific binding evidence", "",
              "| ASO | Species | Reference | RNA context | Query orientation | Recorded exact sites | Best recorded near-site mismatches |",
              "|---|---|---|---|---|---:|---:|"]
    for row in rows:
        for group in row["reference_alignment_groups"]:
            lines.append("| " + " | ".join(_cell(v) for v in (row["aso_id"], group["species"], group["transcript_id"],
                group["mapping_scope"], group["match_orientation"], group["exact_hit_count"], group["best_near_mismatch_count"])) + " |")
    lines += ["", "The supplied transcript catalog is not a same-gene isoform census; a cross-species catalog fraction does not measure gene-wide isoform coverage.",
              "Mismatched sites are binding hypotheses, not exact conservation or measured activity. The complete alignment observations and their source pointers are retained in the projection table."]
    for row in rows:
        lines += ["", "## " + _cell(row["aso_id"])]
        mature = [(k, v) for k, v in row.items() if k.endswith("_evidence_status")]
        for key, value in mature:
            stem = key.removesuffix("_evidence_status")
            lines.append(f"- {stem}: {_cell(value)}; recorded exact sites {row[stem + '_recorded_exact_hits']}; "
                         f"best recorded mismatches {_cell(row[stem + '_best_recorded_mismatches'])}.")
        lines.append(f"- Sequence identity: {row['physical_identity_policy']}; confirmation required: {row['requires_sequence_confirmation']}.")
        lines.append(f"- Recorded screening gene count: {_cell(row['recorded_screened_gene_count'])}; "
                     f"observed symbols: {_cell(row['recorded_screened_gene_symbols'])}; status: {_cell(row['screen_count_status'])}.")
        lines.append(f"- Actual target exclusion applied: {_cell(row['screen_exclusion_applied'])}; "
                     f"excluded gene ID: {_cell(row['screen_excluded_gene_id'])}. "
                     "When exclusion was not applied, this raw screening count is target-inclusive and must not be labeled an other-gene count.")
        lines.append(f"- Recorded screening query: {_cell(row['screen_query_sequence'])}; role: {_cell(row['screen_query_role'])}.")
        for key, value in row["screen_scope"].items():
            # Intended policy is not evidence that an exclusion was applied.
            if key not in {"metric", "intended_target_exclusion", "exclusion_applied", "excluded_gene_id"}:
                lines.append(f"- {key.replace('_', ' ')}: {_cell(value)}")
    lines += ["", "These are saved computational observations. Missing or capped searches remain unknown; no potency, safety or experimental efficacy is established."]
    return "\n".join(lines) + "\n"


def metric_evaluation_response(tasks, blocked_tasks=(), request_envelope=None):
    """Handle only completed reference evaluation + independently saved metrics.

    Unknown/mixed scientific task schemas retain the normal synthesis path.
    """
    if blocked_tasks or {t.get("task_type") for t in tasks} != {"evaluate_existing_aso_set", "sandbox_analysis"}:
        return None
    evaluations = [t.get("payload") for t in tasks if t.get("task_type") == "evaluate_existing_aso_set"]
    if len(evaluations) != 1 or not isinstance(evaluations[0], dict):
        return None
    payload = evaluations[0]
    sources = payload.get("provided_asos") or []
    if not sources or payload.get("splice_related_request") or not payload.get("reference_catalog"):
        return None
    identities = Counter((r.get("aso_id") or r.get("name"), r.get("sequence")) for r in sources)
    additions = []
    for task in tasks:
        if task.get("task_type") != "sandbox_analysis":
            continue
        metrics = _dict(task.get("payload"))
        supported_keys = {"status", "task_id", "task_type", "analysis_plan", "inputs", "input_files",
                          "results", "per_aso", "report_tables", "messages", "scope", "metric_definitions",
                          "aggregate", "consistency_checks", "warnings", "assumptions", "limitations", "output_files"}
        if set(metrics) - supported_keys:
            return None  # A new scientific schema must not lose its additional analyses.
        rows = metrics.get("per_aso") or metrics.get("results")
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(r, dict) or not {"sequence", "length", "gc_fraction"}.issubset(r) for r in rows)
                or Counter((r.get("aso_id") or r.get("name"), r.get("sequence")) for r in rows) != identities):
            return None
        # Preserve every computed metric and its semantics, not a selected
        # preview. This applies to renamed molecules and arbitrary panel sizes.
        fields = list(dict.fromkeys(k for r in rows for k in r))
        additions += ["", "## Independently computed sequence properties", "",
                      "| " + " | ".join(_cell(k) for k in fields) + " |",
                      "| " + " | ".join("---" for _ in fields) + " |"]
        additions += ["| " + " | ".join(_cell(r.get(k)) for k in fields) + " |" for r in rows]
        for key in ("inputs", "input_files", "metric_definitions", "scope", "aggregate", "consistency_checks", "messages", "warnings", "assumptions", "limitations"):
            if metrics.get(key):
                additions += ["", key.replace("_", " ") + ": " + _cell(metrics[key])]
    for task in _dict(request_envelope).get("tasks") or []:
        if task.get("task_type") == "evaluate_existing_aso_set" and _dict(task.get("context")).get("chemistry"):
            additions += ["", "Intended chemistry (declared task context, not a measured property): " +
                          _cell(task["context"]["chemistry"])]
    return {
        "answer_summary": f"Evaluated {len(sources)} supplied molecules using context-specific saved alignments and independent sequence properties.",
        "markdown": projection_markdown(payload) + "\n".join(additions),
        "source_backed_reference_projection": True,
        "reference_projection_rows": evaluation_projection(payload),
        "assumptions": [], "limitations": ["Recorded reference and screening evidence does not measure efficacy."],
        "evidence_refs": [str(_dict(payload.get("reference_catalog")).get("snapshot_path") or "")],
        "suggested_followups": [],
    }
