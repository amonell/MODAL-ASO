"""Keep observed physical placements distinct from complete mapping inventories."""
from copy import deepcopy
import hashlib
import json
import re


def strand_value(value):
    return {'1':'+', '-1':'-', '+':'+', '-':'-'}.get(str(value).strip())


def locus_key(hit):
    blocks = hit.get('genomic_segments') or [hit]
    try:
        geometry = tuple(sorted((str(b['chrom']).removeprefix('chr').lower(),
                                 int(b['start']), int(b['end'])) for b in blocks))
        if not geometry or any(a < 1 or b < a for _, a, b in geometry):
            return None
        return geometry
    except (TypeError, ValueError, KeyError):
        return None


def row_placement(row):
    blocks = row.get('Genomic_segments') or row.get('genomic_segments')
    if isinstance(blocks, str):
        try:
            blocks = json.loads(blocks)
        except (ValueError, TypeError):
            return None
    if not blocks:
        chrom = row.get('Genomic_chrom') or row.get('Chrom') or row.get('chrom')
        if row.get('Genomic_start') is not None:
            start, end = row.get('Genomic_start'), row.get('Genomic_end')
            offset = 0  # Native genomic projections are one-based inclusive.
        else:
            start = row.get('Start', row.get('start'))
            end = row.get('End', row.get('end'))
            offset = 0 if row.get('coordinate_system') == '1based_inclusive' else 1
            if str(row.get('source') or '') in {'gapmer_cdna', 'junction_gapmer'}:
                return None  # A transcript-space span is not a genomic locus.
        try:
            blocks = [dict(chrom=chrom, start=int(start)+offset, end=int(end))]
        except (TypeError, ValueError):
            return None
    if not isinstance(blocks, list) or not locus_key({'genomic_segments': blocks}):
        return None
    chroms = {str(b.get('chrom')) for b in blocks}
    if len(chroms) != 1 or not next(iter(chroms)) or next(iter(chroms)) == 'None':
        return None
    return {'chrom':blocks[0]['chrom'], 'start':min(int(b['start']) for b in blocks),
            'end':max(int(b['end']) for b in blocks), 'genomic_segments':deepcopy(blocks),
            'strand':strand_value(row.get('Strand') or row.get('strand')), 'mismatch_bin':'exact',
            'evidence_source':'reported_candidate_placement', 'coordinate_system':'1based_inclusive'}


def observed_placements(rows):
    """Index explicit row geometry, retaining overlapping placements and strands."""
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sequence = str(row.get('Sequence') or row.get('sequence') or row.get('seq') or '').upper().replace('U','T')
        hit = row_placement(row)
        if not sequence or not hit:
            continue
        result.setdefault(sequence, {})[(locus_key(hit), hit.get('strand'))] = hit
    return result


def mapping_inventory(row, observed, parse_loci):
    """A missing, capped or partly parsed inventory never establishes zero hits.

    Existing candidate placements are positive evidence only. They are not a
    substitute for an exhaustive genome/transcriptome search.
    """
    raw = next((row[k] for k in ('Mapped_to','mapped_to','mappedto')
                if k in row and row[k] is not None), None)
    recorded = row.get('Mapped_loci')
    fingerprint = hashlib.sha256(json.dumps(
        {'raw':raw, 'recorded':recorded if raw is None else None},
        sort_keys=True, default=str).encode()).hexdigest()
    hits, complete, basis = [], False, 'mapping_inventory_missing'
    if raw is not None:
        text = str(raw).strip()
        text = re.sub(r'^mappedto=', '', text, flags=re.I)
        tokens = [token.strip() for token in text.split(',') if token.strip()]
        hits = parse_loci(raw)
        complete = bool(tokens) and len(hits) == len(tokens)
        # Empty textual cells and overflow sentinels carry no completion proof.
        basis = 'reported_mapped_to_inventory' if complete else 'mapping_inventory_incomplete'
    elif isinstance(recorded, list):
        hits = [deepcopy(h) for h in recorded if isinstance(h, dict) and locus_key(h)]
        complete = len(hits) == len(recorded)
        basis = 'reported_structured_mapping_inventory' if complete else 'mapping_inventory_incomplete'
    # Recompute from current observations. A previous derived missingness flag
    # must not override a subsequently delivered complete mapper inventory.
    previous = row.get('repeat_inventory_evidence') or {}
    if (row.get('repeat_inventory_status') in {'partial','unavailable'}
            and (not isinstance(previous, dict)
                 or previous.get('input_mapping_sha256') in {None, fingerprint})):
        complete = False
    sequence = str(row.get('Sequence') or row.get('sequence') or row.get('seq') or '').upper().replace('U','T')
    strand = strand_value(row.get('Strand') or row.get('strand'))
    merged = {locus_key(h):deepcopy(h) for h in hits if locus_key(h)}
    for (key, observed_strand), hit in observed.get(sequence, {}).items():
        if strand and observed_strand and strand != observed_strand:
            continue
        if key not in merged:
            if complete:
                complete = False
                basis = 'reported_mapping_inventory_omits_an_observed_placement'
            merged[key] = deepcopy(hit)
    primary = row_placement(row)
    primary_key = locus_key(primary) if primary else None
    if not primary_key:
        complete = False
        basis = 'primary_genomic_placement_unresolved'
    alternatives = [hit for key, hit in merged.items() if key != primary_key]
    status = 'complete' if complete else 'partial' if alternatives else 'unavailable'
    return alternatives, {'status':status, 'complete':complete, 'basis':basis,
        'input_mapping_sha256':fingerprint,
        'scope':'Distinct physical genomic block sets; overlapping placements remain distinct; primary placement excluded. Observed candidate rows supply positive evidence only. No genome-wide absence claim from an incomplete inventory.',
        'coordinate_system':'1based_inclusive'}
