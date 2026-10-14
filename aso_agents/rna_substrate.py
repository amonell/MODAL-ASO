"""Bounded RNA-locus admission for genomic steric candidates.

A coordinate conversion is not evidence of an RNA substrate. Both mature exons
and intronic pre-mRNA are legitimate; this check does not require cDNA identity.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def transcript_loci(catalog):
    loci = []
    for label, entry in (catalog or {}).items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get('metadata') or {}
        chrom = entry.get('chrom') or meta.get('seq_region_name')
        strand = {1: '+', -1: '-', '1': '+', '-1': '-'}.get(entry.get('strand'), entry.get('strand'))
        exons = entry.get('exons_tx') or []
        if not chrom or strand not in {'+', '-'} or not exons:
            continue
        if any(not isinstance(e, dict) or any(not isinstance(e.get(k), int) or isinstance(e.get(k), bool)
               for k in ('start', 'end')) or e['start'] < 1 or e['end'] < e['start'] for e in exons):
            continue
        loci.append({'transcript_id': entry.get('transcript_id') or label,
                     'chrom': str(chrom).removeprefix('chr'), 'strand': strand,
                     'start0': min(e['start'] for e in exons) - 1,
                     'end0': max(e['end'] for e in exons),
                     'reference_source': meta.get('reference_source') or entry.get('reference_source'),
                     'reference_annotation': meta.get('reference_annotation') or entry.get('reference_annotation')})
    return loci


def candidate_rna_support(row, loci, intervals_for_row):
    blocks = intervals_for_row(row, output_space='genomic')
    strand = row.get('Strand') or row.get('strand')
    strand = {1: '+', -1: '-', '1': '+', '-1': '-'}.get(strand, strand)
    if not loci or not blocks or strand not in {'+', '-'}:
        return {'status': 'unverified', 'reason': 'RNA catalog, candidate blocks or target strand unavailable'}
    hits = [locus for locus in loci if strand == locus['strand'] and all(
        str(chrom).removeprefix('chr') == locus['chrom'] and locus['start0'] <= start < end <= locus['end0']
        for chrom, start, end in blocks)]
    return {'status': 'supported_transcript_locus' if hits else 'outside_recorded_target_rna_loci',
            'supporting_transcripts': [hit['transcript_id'] for hit in hits],
            'candidate_blocks_bed0_half_open': deepcopy(blocks), 'target_rna_strand': strand,
            'scope': 'Full candidate containment on the target strand in at least one recorded transcript locus, including intronic pre-mRNA. This is substrate geometry, not measured binding or activity. An exclusion concerns this bounded RNA catalog, not every possible transcript or a DNA-targeting mechanism.'}


def filter_target_rna_candidates(source_path, catalog, intervals_for_row):
    source = Path(source_path)
    content = source.read_bytes()
    payload = json.loads(content)
    loci = transcript_loci(catalog)
    eligible, excluded = [], []
    for raw in payload.get('rows') or []:
        row = deepcopy(raw)
        evidence = candidate_rna_support(row, loci, intervals_for_row)
        row['target_rna_evidence'] = evidence
        (excluded if evidence['status'] == 'outside_recorded_target_rna_loci' else eligible).append(row)
    destination = source.with_name(source.stem + '_target_rna_eligible.json')
    rejected = source.with_name(source.stem + '_outside_target_rna.json')
    scope = {'reference_transcript_loci': loci, 'source_path': str(source),
             'source_sha256': hashlib.sha256(content).hexdigest(),
             'policy': 'Exclude demonstrated target-RNA locus/strand contradictions; preserve unknown evidence as unknown. Coordinate conversion alone does not establish target-RNA identity.'}
    for path, rows in ((destination, eligible), (rejected, excluded)):
        path.write_text(json.dumps({**payload, 'rows': rows, 'target_rna_admission': scope}, indent=2))
    return {**scope, 'status': 'complete' if loci else 'unverified_reference_catalog',
            'eligible_path': str(destination), 'excluded_path': str(rejected),
            'input_count': len(eligible) + len(excluded), 'eligible_count': len(eligible),
            'excluded_count': len(excluded),
            'unverified_count': sum(row['target_rna_evidence']['status'] == 'unverified' for row in eligible)}
