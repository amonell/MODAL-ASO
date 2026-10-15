"""Geometry shared by reference-derived splice windows and delivery checks."""


def splice_site_window(position, strand, site, context, up, down):
    """Return one-based inclusive bounds with the native junction convention."""
    if type(position) is not int or position < 1:
        return None
    if strand not in {"+", "-"} or site not in {"acceptor", "donor"}:
        return None
    if context == "upstream_intron" and site != "acceptor":
        context = "site_window"
    if context == "downstream_intron" and site != "donor":
        context = "site_window"
    if context == "upstream_intron":
        start, end = (position - up, position) if strand == "+" else (position, position + down)
    elif context == "downstream_intron":
        start, end = (position, position + down) if strand == "+" else (position - up, position)
    elif context == "exon_body":
        toward_right = (site == "acceptor") == (strand == "+")
        start, end = (position, position + down) if toward_right else (position - up, position)
    else:
        start, end = position - up, position + down
    start = max(1, int(start))
    return start, max(start + 1, int(end))


def verified_reference_window_scope(goal, job):
    """Resolve a compiled exon window without waiving unresolved user constraints.

    A policy label alone is not physical evidence. Require reference exon
    boundaries and exactly the windows produced from those boundaries. This
    establishes spatial scope only, not the direction or efficacy of an ASO.
    """
    meta = job.get("selection_meta") or {}
    if (meta.get("sources") != ["splice_site_window"] or job.get("allow_full_copy")
            or meta.get("fallback") or meta.get("constraint_exhausted")):
        return False
    materialized = goal.get("materialized_region_constraints") or {}
    if (job.get("region_constraints") or goal.get("region_constraints")
            or materialized.get("global")
            or (materialized.get("by_program") or {}).get(job.get("program_id"))):
        return False
    targets = meta.get("resolved_site_targets")
    catalog = goal.get("isoform_transcripts") or {}
    if (not isinstance(targets, list) or not targets or not isinstance(catalog, dict) or not catalog
            or type(job.get("exon_label")) is not int or job["exon_label"] < 1):
        return False
    up = down = 200
    for flanks in (job.get("flanks"), goal.get("flanks")):
        if not isinstance(flanks, dict):
            continue
        values = [flanks.get("up"), flanks.get("down")]
        if any(type(value) is int and value > 0 for value in values):
            up, down = [value if type(value) is int and value > 0 else 200 for value in values]
            break
    context = meta.get("site_context") or "site_window"
    if context not in {"site_window", "upstream_intron", "downstream_intron", "exon_body"}:
        return False
    expected = set()
    for target in targets:
        if not isinstance(target, dict):
            return False
        ordinals = target.get("resolved_exon_ordinals") or {}
        if (not isinstance(ordinals, dict) or not ordinals
                or target.get("requested_exon_label") != job.get("exon_label")):
            return False
        chrom = str(target.get("chrom") or "").removeprefix("chr")
        position, strand, site = target.get("position"), target.get("strand"), target.get("site")
        if not chrom or strand not in {"+", "-"} or site not in {"acceptor", "donor"}:
            return False
        for label, ordinal in ordinals.items():
            reference = catalog.get(label) or {}
            reference_strand = str(reference.get("strand"))
            reference_strand = {"1": "+", "-1": "-"}.get(reference_strand, reference_strand)
            reference_chrom = reference.get("chrom") or (reference.get("metadata") or {}).get("seq_region_name")
            if str(reference_chrom or "").removeprefix("chr") != chrom or reference_strand != strand:
                return False
            exons = [exon for exon in reference.get("exons_tx") or [] if exon.get("index") == ordinal]
            if len(exons) != 1:
                return False
            # The acceptor is the transcript-oriented exon start; the donor
            # is its end. Both use the annotated exon boundary, not intronic GT/AG.
            boundary = "start" if (site == "acceptor") == (strand == "+") else "end"
            if type(position) is not int or exons[0].get(boundary) != position:
                return False
        window = splice_site_window(position, strand, site, context, up, down)
        if window is None:
            return False
        expected.add((chrom, window[0] - 1, window[1], strand))
    rows = job.get("tile_filter_rows") or job.get("region_rows") or []
    observed = set()
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            return False
        if type(row[1]) is not int or type(row[2]) is not int or not 0 <= row[1] < row[2]:
            return False
        observed.add((str(row[0]).removeprefix("chr"), row[1], row[2], str(row[5])))
    return bool(expected) and observed == expected
