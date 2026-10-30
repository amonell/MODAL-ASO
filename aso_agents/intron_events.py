"""Resolve intron events on an attested transcript, independently of exon effects."""
from copy import deepcopy

from .gapmer_selectivity import resolve_catalog_label


def resolve_intron_event(program, catalog):
    event = program.get("splice_event")
    if not event:
        return None
    kind = event.get("event_type")
    ordinal = event.get("intron_number")
    if kind not in {"intron_retention", "intron_excision"} or type(ordinal) is not int or ordinal < 1:
        raise ValueError("intron_event_requires_type_and_positive_transcript_intron_number")
    if program.get("modality") != "steric" or program.get("intent") != "splice_modulation":
        raise ValueError("intron_event_requires_steric_splice_modulation")
    if program.get("desired_effect") in {"inclusion", "skipping"}:
        raise ValueError("intron_event_is_not_an_exon_inclusion_or_skipping_effect")
    reference = event.get("reference_transcript_id") or program.get("transcript_id")
    if not reference:
        raise ValueError("intron_event_requires_reference_transcript")
    label = resolve_catalog_label(reference, catalog, catalog)
    if label is None:
        raise ValueError("intron_event_reference_unresolved: " + reference)
    entry = catalog[label]
    exons = entry.get("exons_tx") or []
    strand = str(entry.get("strand") or "")
    chrom = entry.get("chrom") or (entry.get("metadata") or {}).get("seq_region_name")
    if not chrom or strand not in {"+","-","1","-1"} or ordinal >= len(exons):
        raise ValueError("intron_event_reference_structure_or_ordinal_invalid")
    minus = strand in {"-","-1"}
    ordered = sorted(exons, key=lambda exon: exon["start"], reverse=minus)
    upstream, downstream = ordered[ordinal-1:ordinal+1]
    donor = upstream["start"] if minus else upstream["end"]
    acceptor = downstream["end"] if minus else downstream["start"]
    start, end = min(donor, acceptor) + 1, max(donor, acceptor) - 1
    if end < start or (donor < acceptor if minus else donor > acceptor):
        raise ValueError("intron_event_requires_separated_adjacent_exons")
    direction = "decrease" if kind == "intron_retention" else "increase"
    return {"event_type":kind, "intron_number":ordinal, "reference_transcript_id":reference,
        "catalog_label":label, "chrom":str(chrom), "strand":"-" if minus else "+",
        "intron_start_1based":start,"intron_end_1based":end,
        "sites":[{"site":"donor","position":donor,"exon_index":ordinal,"desired_delta":direction},
                 {"site":"acceptor","position":acceptor,"exon_index":ordinal+1,"desired_delta":direction}],
        "interpretation":"Flanking splice-site perturbation is a mechanistic proxy; it does not establish intron retention/excision, frame consequences, NMD or protein outcome. Other junction changes remain relevant."}


def _identity(chrom, site, position, strand):
    return str(chrom).removeprefix("chr"), site, position, "-" if str(strand) in {"-","-1"} else "+"


def _memberships(event, site, catalog):
    present, absent, indices, unresolved = [], [], {}, []
    target = _identity(event["chrom"],site["site"],site["position"],event["strand"])
    for label,record in catalog.items():
        exons = record.get("exons_tx") or []
        chrom = record.get("chrom") or (record.get("metadata") or {}).get("seq_region_name")
        strand = str(record.get("strand") or "")
        if not chrom or strand not in {"+","-","1","-1"} or not exons:
            unresolved.append(label)
            continue
        minus = strand in {"-","-1"}
        ordered = sorted(exons,key=lambda exon:exon["start"],reverse=minus)
        matched = None
        for ordinal,exon in enumerate(ordered,1):
            kind = site["site"]
            if (kind == "acceptor" and ordinal == 1) or (kind == "donor" and ordinal == len(ordered)):
                continue
            position = exon["start"] if (kind == "donor") == minus else exon["end"]
            if _identity(chrom,kind,position,strand) == target:
                matched = ordinal
                break
        if matched is None:
            absent.append(label)
        else:
            present.append(label)
            indices[label] = matched
    return present,absent,indices,unresolved


def bind_intron_events(plan, comparisons, goal, catalog):
    events = [resolve_intron_event(program,catalog) for program in goal.get("design_programs") or []
              if isinstance(program,dict) and program.get("splice_event")]
    if not events:
        return plan, comparisons
    bound = deepcopy(plan)
    index = {_identity(row.get("chrom"),row.get("site"),row.get("position"),row.get("strand")):row for row in bound}
    requested = {}
    for event in events:
        for site in event["sites"]:
            key = _identity(event["chrom"],site["site"],site["position"],event["strand"])
            if key in requested and requested[key] != site["desired_delta"]:
                raise ValueError("intron_events_request_conflicting_directions_at_same_site")
            requested[key] = site["desired_delta"]
            row = index.get(key)
            if row is None:
                # The older generic planner omits noncoding exon boundaries.
                # An explicitly requested intron is still bound by the complete
                # verified exon chain, including UTR introns. Materialize those
                # actual boundaries rather than substituting a coding exon.
                label = event["catalog_label"]
                present,absent,indices,unresolved = _memberships(event,site,catalog)
                row = {"chrom":event["chrom"],"strand":event["strand"],
                    "site":site["site"],"position":site["position"],
                    "present_in":present,"transcript_ids":present,
                    "absent_in":absent,"exon_indices":indices,
                    "unresolved_membership":unresolved,
                    "covers_all_isoforms":not absent and not unresolved,
                    "uniqueness":"unassessed" if unresolved else "unique" if len(present)==1 else "shared",
                    "delta_per_isoform":{key:site["desired_delta"] if key==label else "context" for key in present},
                    "reference_binding_status":"resolved","source":"verified_transcript_intron_boundaries"}
                bound.append(row)
                index[key] = row
            if row.get("delta_direction") in {"increase","decrease"} and row["delta_direction"] != site["desired_delta"]:
                raise ValueError("intron_event_conflicts_with_existing_splice_objective")
            row.update(delta_direction=site["desired_delta"], intron_event=event,
                       direction_source="explicit_intron_event", objective_kind=event["event_type"])
    comparison_rows = deepcopy(comparisons)
    comparison_keys = {_identity(row.get("chrom"),row.get("site"),row.get("position"),row.get("strand")) for row in comparison_rows}
    comparison_rows.extend(deepcopy(row) for key,row in index.items() if key in requested and key not in comparison_keys)
    for row in comparison_rows:
        key = _identity(row.get("chrom"),row.get("site"),row.get("position"),row.get("strand"))
        if key in requested:
            row["delta_direction"] = requested[key]
    return bound, comparison_rows


def program_intron_sites(program, catalog, plan):
    event = resolve_intron_event(program,catalog)
    if event is None:
        return None
    targets = {_identity(event["chrom"],site["site"],site["position"],event["strand"]) for site in event["sites"]}
    rows = [deepcopy(row) for row in plan if _identity(row.get("chrom"),row.get("site"),row.get("position"),row.get("strand")) in targets]
    if len(rows) != 2:
        raise ValueError("intron_event_requires_both_resolved_flanking_sites")
    return rows
