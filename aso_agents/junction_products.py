"""Construct requested splice products from typed boundary changes."""
from copy import deepcopy
import hashlib
import json

from .gapmer_selectivity import resolve_catalog_label


def boundary_position(change):
    """Return the exonic boundary base in one-based genomic coordinates."""
    kind, strand = change.get("site_kind"), change.get("strand")
    convention, role = change.get("coordinate_convention"), change.get("coordinate_role")
    if (kind not in {"acceptor", "donor"} or strand not in {"+", "-"}
            or convention not in {"one_based_inclusive", "bed0_half_open"}):
        raise ValueError("junction_product_coordinate_definition_incomplete")
    if role == "exon_boundary":
        position = change.get("position")
        if type(position) is not int or any(change.get(key) is not None for key in ("start", "end")):
            raise ValueError("junction_product_boundary_requires_single_position")
        position += convention == "bed0_half_open"
    elif role == "splice_dinucleotide":
        start, end = change.get("start"), change.get("end")
        if (type(start) is not int or type(end) is not int
                or change.get("position") is not None):
            raise ValueError("junction_product_dinucleotide_requires_interval")
        start += convention == "bed0_half_open"
        if start < 1 or end - start != 1:
            raise ValueError("junction_product_dinucleotide_must_span_two_bases")
        position = end + 1 if (kind, strand) in {("acceptor", "+"), ("donor", "-")} else start - 1
    else:
        raise ValueError("junction_product_coordinate_role_required")
    if position < 1:
        raise ValueError("junction_product_boundary_out_of_range")
    return position


def changed_exon_spans(runtime, objective, base_label, base_entry, goal):
    """Change only the specified boundary of an identified reference exon."""
    exons = base_entry.get("exons_tx") or []
    strand = runtime._strand_char(base_entry.get("strand"))
    chrom = str(base_entry.get("chrom") or (base_entry.get("metadata") or {}).get("seq_region_name") or "")
    targets, seen = {}, set()
    changes = objective.get("junction_changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("junction_product_changes_required")
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("junction_product_invalid_change")
        if (str(change.get("chrom") or "").removeprefix("chr") != chrom.removeprefix("chr")
                or ("strand" in change and change["strand"] != strand)):
            raise ValueError("junction_product_reference_orientation_conflict")
        requested = change.get("exon_label")
        if type(requested) is not int or requested < 1:
            raise ValueError("junction_product_exon_label_required")
        ordinal = runtime._resolve_authoritative_transcript_exon_ordinal(base_entry, requested, goal)
        matches = [e for e in exons if e.get("index") == ordinal]
        if len(matches) != 1:
            raise ValueError("junction_product_exon_unresolved")
        exon = matches[0]
        kind = change.get("site_kind")
        if (ordinal, kind) in seen:
            raise ValueError("junction_product_duplicate_boundary")
        seen.add((ordinal, kind))
        if (kind == "acceptor" and ordinal == 1) or (kind == "donor" and ordinal == len(exons)):
            raise ValueError("junction_product_terminal_boundary_has_no_splice_site")
        target = targets.setdefault(ordinal, {
            "chrom": chrom, "strand": strand, "start": exon["start"], "end": exon["end"],
            "exon_label": ordinal, "requested_exon_label": requested,
            "base_isoform_label": base_label, "source": "typed_junction_change",
        })
        endpoint = "start" if (kind, strand) in {("acceptor", "+"), ("donor", "-")} else "end"
        target[endpoint] = boundary_position({**change, "strand": strand})
    for ordinal, target in targets.items():
        original = next(e for e in exons if e.get("index") == ordinal)
        start, end = target["start"], target["end"]
        if start > end or end < original["start"] or start > original["end"]:
            raise ValueError("junction_product_boundary_removes_identified_exon")
        if any(e.get("index") != ordinal and e["start"] <= end and e["end"] >= start for e in exons):
            raise ValueError("junction_product_boundary_crosses_another_exon")
        if (start, end) == (original["start"], original["end"]):
            raise ValueError("junction_product_changes_leave_reference_unchanged")
    return list(targets.values())


def materialize_junction_products(runtime, goal, final_payload=None):
    objectives = [o for o in goal.get("splice_isoform_objectives") or []
                  if isinstance(o, dict) and o.get("junction_changes")]
    if not objectives:
        return False
    from .declared_reference_catalog import complete_target_references
    working = deepcopy(goal)
    complete_target_references(runtime, working)
    catalog = working.get("isoform_transcripts") or {}
    for objective in objectives:
        reference = objective.get("reference_accession")
        label = objective.get("isoform_label")
        if not reference or not label or label == reference or objective.get("transcript_id"):
            raise ValueError("junction_product_requires_distinct_label_and_parent_reference")
        base_label = resolve_catalog_label(reference, catalog, catalog)
        if base_label is None:
            raise ValueError("junction_product_parent_reference_unresolved")
        base = deepcopy(catalog[base_label])
        targets = changed_exon_spans(runtime, objective, base_label, base, working)
        entry, _ = runtime._build_modified_explicit_target_transcript(
            base_label=base_label, base_entry=base, modified_label=label, targets=targets)
        fingerprint = hashlib.sha256(json.dumps(
            [reference, base.get("transcript_id"), base.get("exons_tx"), targets], sort_keys=True).encode()).hexdigest()
        product_id = str(base.get("transcript_id") or base_label) + "::junction_product_" + fingerprint[:16]
        metadata = entry["metadata"]
        entry.pop("explicit_target_modified", None)
        metadata.pop("explicit_target_modified", None)
        entry["transcript_id"] = product_id
        metadata.update(id=product_id, display_name=label, source="junction_defined_product",
                        reference_source="junction_defined_product", junction_defined_product=True,
                        junction_changes=deepcopy(objective["junction_changes"]),
                        junction_exon_changes=deepcopy(targets),
                        parent_reference_accession=reference, geometry_sha256=fingerprint,
                        description="Constructed splice product using the requested junction changes and the verified parent reference.")
        existing = catalog.get(label)
        if existing and (existing.get("metadata") or {}).get("source") != "junction_defined_product":
            raise ValueError("junction_product_label_conflicts_with_existing_reference")
        catalog[label] = entry
    goal["isoform_transcripts"] = catalog
    goal["declared_target_reference_resolution"] = working.get("declared_target_reference_resolution") or []
    if isinstance(final_payload, dict):
        final_payload["isoform_transcripts"] = deepcopy(catalog)
        final_payload["declared_target_reference_resolution"] = deepcopy(goal["declared_target_reference_resolution"])
    return True
