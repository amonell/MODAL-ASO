"""Reference-bounded splice-site sharing; independent of design objectives."""


def _chrom(value):
    return str(value or "").removeprefix("chr")


def _strand(value):
    return {"1": "+", "+": "+", "-1": "-", "-": "-"}.get(str(value))


def measurement_site_exon_membership(site, catalog):
    """Resolve missing measurement-plan membership from physical junctions."""
    kind, position = site.get('site'), site.get('position')
    chrom, strand = _chrom(site.get('chrom')), _strand(site.get('strand'))
    if not strand:
        strands = {_strand(entry.get('strand')) for entry in catalog.values()
                   if isinstance(entry, dict)} - {None}
        strand = next(iter(strands)) if len(strands) == 1 else None
    if (kind not in {'acceptor', 'donor'} or not isinstance(position, int)
            or isinstance(position, bool) or not chrom or not strand):
        return {}
    resolved = {}
    for label, entry in catalog.items():
        if not isinstance(entry, dict):
            continue
        entry_chrom = _chrom(entry.get('chrom') or entry.get('seq_region_name')
                             or (entry.get('metadata') or {}).get('seq_region_name'))
        if entry_chrom != chrom or _strand(entry.get('strand')) != strand:
            continue
        exons = entry.get('exons_tx') or []
        if not exons or any(not isinstance(exon, dict)
                            or not isinstance(exon.get('index'), int)
                            or isinstance(exon.get('index'), bool) for exon in exons):
            continue
        exons = sorted(exons, key=lambda exon: exon['index'])
        endpoint = 'start' if (kind == 'acceptor') == (strand == '+') else 'end'
        matches = [exon['index'] for ordinal, exon in enumerate(exons)
                   if (ordinal > 0 if kind == 'acceptor' else ordinal < len(exons)-1)
                   and exon.get(endpoint) == position]
        if len(matches) == 1:
            resolved[label] = matches[0]
    return resolved


def splice_site_reference_scope(site, goal):
    selected = goal.get("isoform_transcripts") or {}
    catalog = goal.get("all_transcripts_full") or selected
    scope = "available_gene_transcript_catalog" if goal.get("all_transcripts_full") else "selected_design_transcripts"
    kind = site.get("site") or site.get("site_kind")
    position = site.get("position")
    chrom, strand = _chrom(site.get("chrom")), _strand(site.get("strand"))
    if not strand:
        strands = {_strand(e.get("strand")) for e in selected.values() if isinstance(e, dict)} - {None}
        strand = next(iter(strands)) if len(strands) == 1 else None
    compared, present, unresolved = set(), set(), set()
    for label, entry in catalog.items():
        if not isinstance(entry, dict):
            unresolved.add(str(label))
            continue
        tid = str(entry.get("transcript_id") or label)
        exons = entry.get("exons_tx") or []
        entry_chrom = _chrom(entry.get("chrom") or entry.get("seq_region_name")
                             or (entry.get("metadata") or {}).get("seq_region_name"))
        entry_strand = _strand(entry.get("strand"))
        if (kind not in {"acceptor", "donor"} or not isinstance(position, int)
                or isinstance(position, bool) or not chrom or not strand
                or not entry_chrom or not entry_strand or not exons
                or any(not isinstance(x.get(k), int) or isinstance(x.get(k), bool)
                       for x in exons for k in ("index", "start", "end"))):
            unresolved.add(tid)
            continue
        indices = [x["index"] for x in exons]
        if sorted(indices) != list(range(1, len(exons) + 1)):
            unresolved.add(tid)
            continue
        compared.add(tid)
        if chrom != entry_chrom or strand != entry_strand:
            continue
        for exon in exons:
            internal = exon["index"] > 1 if kind == "acceptor" else exon["index"] < len(exons)
            endpoint = "start" if (kind == "acceptor") == (strand == "+") else "end"
            if internal and exon[endpoint] == position:
                present.add(tid)
    if len(present) > 1:
        status = "shared"
    elif len(present) == 1 and len(compared) > 1 and not unresolved:
        status = "single_catalog_match"
    else:
        status = "specificity_unresolved"
    description = f"{len(present)} matching transcripts among {len(compared)} inspected ({scope}); outside-catalog specificity unknown"
    return {"scope": scope, "status": status, "compared_transcript_ids": sorted(compared),
            "present_in": sorted(present), "absent_in": sorted(compared - present),
            "unresolved_transcript_ids": sorted(unresolved), "complete_for_gene": False,
            "description": description}


def _objective_membership_risks(scope, goal):
    """Bind objective labels to actual catalog membership, never missing flags."""
    aliases = {}
    for label, entry in (goal.get('isoform_transcripts') or {}).items():
        if not isinstance(entry, dict):
            continue
        identity = str(entry.get('transcript_id') or label)
        for alias in (str(label), identity):
            aliases[alias.casefold()] = identity
    def identity(value):
        text = str(value or '')
        return aliases.get(text.casefold(), text)
    increase, decrease = set(), set()
    for item in goal.get('splice_isoform_objectives') or []:
        if not isinstance(item, dict):
            continue
        label = item.get('isoform_label') or item.get('transcript_id') or item.get('label')
        direction = item.get('desired_abundance')
        if label and direction in {'increase','decrease'}:
            (increase if direction == 'increase' else decrease).add(identity(label))
    for key, target in [('increase_isoforms',increase),('decrease_isoforms',decrease)]:
        if isinstance(goal.get(key), list):
            target.update(identity(x) for x in goal[key] if isinstance(x,str) and x)
    if not increase and not decrease:
        return {}
    present, compared = set(scope['present_in']), set(scope['compared_transcript_ids'])
    raised, reduced = present & increase, present & decrease
    complete = bool(compared) and (increase | decrease) <= compared
    return {'beneficial_isoforms':sorted(raised),'pathogenic_isoforms':sorted(reduced),
            'shared_with_beneficial':True if len(present)>1 and raised else False if complete else None,
            'conflict_between_objectives':True if raised and reduced else False if complete else None,
            'objective_membership_complete':complete,
            'unresolved_objective_transcript_ids':sorted((increase | decrease) - compared),
            'objective_membership_scope':'physically_inspected_transcript_catalog'}


def scoped_splice_evidence(payload, site, goal):
    """Separate selected-objective metadata from physical catalog membership."""
    result = dict(payload)
    keys = ("uniqueness", "covers_all_isoforms", "unique_to_pathogenic", "unique_to_beneficial",
            "present_in", "absent_in", "net_effect_hint", "rationale_hint")
    context = {key: result.pop(key) for key in keys if key in result}
    context["scope"] = "selected_design_transcripts"
    context["reference_verified"] = False
    context["interpretation"] = "Planner-supplied annotations; checked physical membership is in reference_scope."
    result["selected_transcript_context"] = context
    result["reference_scope"] = splice_site_reference_scope(site, goal)
    result.update(_objective_membership_risks(result['reference_scope'], goal))
    return result


def bind_safeguard_reference(candidate, site, goal):
    """Keep guard policy while checking any attached transcript-sharing claim."""
    result = dict(candidate)
    scope = splice_site_reference_scope(site, goal)
    risks = _objective_membership_risks(scope, goal)
    if not scope['compared_transcript_ids'] and not risks:
        return result
    result.update(risks)
    result.update(reference_scope=scope, present_in=scope['present_in'], absent_in=scope['absent_in'])
    if risks:
        result['unique_to_pathogenic'] = (scope['status'] == 'single_catalog_match'
                                          and bool(risks['pathogenic_isoforms'])
                                          and not risks['beneficial_isoforms'])
        result['unique_to_beneficial'] = (scope['status'] == 'single_catalog_match'
                                          and bool(risks['beneficial_isoforms'])
                                          and not risks['pathogenic_isoforms'])
        result['reference_scope_note'] = (
            'Shared with beneficial isoforms in the inspected reference catalog; outside-catalog specificity unknown.'
            if risks.get('shared_with_beneficial') else scope['description'])
    return result
