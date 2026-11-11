"""Resolve exon names from structured RefSeqGene/LRG records.

The small reference index and individual XML records are retrieved on demand.
Successful downloads are held in process memory; runs need no installed cache.
Only the evidence summary is saved with a run, never genomic sequence.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
import time
import xml.etree.ElementTree as ET

INDEX_URL = 'https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/RefSeqGene/LRG_RefSeqGene'
LRG_URL = 'https://ftp.ebi.ac.uk/pub/databases/lrgex/{}.xml'
RESOLVER_VERSION = 1
VERIFIED_NOMENCLATURE_STATUSES = frozenset({'verified_alternate_mapping', 'verified_curated_mapping'})


def _strand(value):
    return {'1': '+', '-1': '-', '+': '+', '-': '-'}.get(str(value))


def _chrom(value):
    return str(value or '').removeprefix('chr')


def _assembly(value):
    value = {'hg38': 'GRCh38', 'hg19': 'GRCh37'}.get(str(value), str(value or ''))
    return value.partition('.p')[0]


def reference_context(entry):
    metadata = entry.get('metadata') if isinstance(entry.get('metadata'), dict) else {}
    identifier = str(entry.get('transcript_id') or entry.get('id') or metadata.get('id') or '')
    versioned = metadata.get('versioned_id') or entry.get('versioned_id')
    if versioned and versioned.split('.', 1)[0] == identifier.split('.', 1)[0]:
        identifier = versioned
    elif '.' not in identifier and metadata.get('version') is not None:
        identifier += '.' + str(metadata['version'])
    return {'transcript_id': identifier,
            'modified': bool(entry.get('explicit_target_modified') or metadata.get('explicit_target_modified')),
            'assembly': entry.get('assembly_name') or metadata.get('assembly_name'),
            'chrom': entry.get('chrom') or entry.get('seq_region_name') or metadata.get('seq_region_name'),
            'strand': _strand(entry.get('strand', metadata.get('strand'))),
            'species': entry.get('species') or metadata.get('species'),
            'gene_symbol': entry.get('gene_symbol') or metadata.get('gene_symbol') or metadata.get('gene_name')}


def reference_fingerprint(transcript_id, transcript_exons, context):
    identity = [transcript_id, _assembly(context.get('assembly')), _chrom(context.get('chrom')),
                _strand(context.get('strand')), context.get('species'), context.get('gene_symbol'), context.get('modified'),
                sorted([(e.get('index'), e.get('start'), e.get('end'),
                         _chrom(e.get('seq_region_name') or context.get('chrom')))
                        if isinstance(e, dict) else (None, None, None, str(e))
                        for e in transcript_exons], key=lambda row: str(row[0]))]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=64)
def _download(url, timeout, hour):
    import requests
    try:
        response = requests.get(url, headers={'User-Agent': 'MODAL-ASO exon-nomenclature resolver'}, timeout=timeout)
        response.raise_for_status()
        body = response.text
        return body, hashlib.sha256(body.encode()).hexdigest(), None
    except requests.RequestException as exc:
        # A source outage incurs one timeout per process window, across genes
        # and transcript isoforms. The exception carries no record contents.
        return None, None, type(exc).__name__


def _read(url, timeout):
    body, sha, error = _download(url, timeout, int(time.time() // 300))
    if error:
        raise OSError(error)
    return body, sha


def _identifier_matches(requested, actual):
    return requested == actual if '.' in requested else requested == actual.split('.', 1)[0]


def _coordinates(element, system):
    matches = [node for node in element.findall('coordinates') if node.get('coord_system') == system]
    if len(matches) != 1:
        raise ValueError('A source feature has ambiguous coordinates.')
    node = matches[0]
    start, end, strand = int(node.get('start')), int(node.get('end')), _strand(node.get('strand'))
    if start < 1 or end < start or strand is None:
        raise ValueError('A source feature has invalid coordinates.')
    return start, end, strand


def _map_interval(start, end, strand, span):
    """Apply explicit LRG alignment differences, including intronic indels."""
    lo, hi = int(span.get('lrg_start')), int(span.get('lrg_end'))
    other_lo, other_hi = int(span.get('other_start')), int(span.get('other_end'))
    orientation = _strand(span.get('strand'))
    if not (lo <= start <= end <= hi) or orientation is None or other_lo > other_hi:
        return None
    differences = []
    for diff in span.findall('diff'):
        kind = diff.get('type')
        if kind == 'mismatch':
            continue
        if kind not in {'lrg_ins', 'other_ins'}:
            return None
        a, b = int(diff.get('lrg_start')), int(diff.get('lrg_end'))
        c, d = int(diff.get('other_start')), int(diff.get('other_end'))
        if not lo <= a <= b <= hi or c > d:
            return None
        if kind == 'lrg_ins':
            if max(start, a) <= min(end, b):
                return None
            differences.append((b, -(b-a+1)))
        else:
            if b != a+1 or start <= a < end:
                return None
            differences.append((a, d-c+1))
    if other_hi-other_lo != hi-lo+sum(delta for _, delta in differences):
        return None
    def position(point):
        offset = point-lo+sum(delta for boundary, delta in differences if point > boundary)
        return other_lo+offset if orientation == '+' else other_hi-offset
    endpoints = [position(start), position(end)]
    mapped_strand = strand if orientation == '+' else ('-' if strand == '+' else '+')
    return min(endpoints), max(endpoints), mapped_strand


def _project_interval(coordinates, mapping):
    matches = {_map_interval(*coordinates, span) for span in mapping.findall('mapping_span')}
    matches.discard(None)
    return next(iter(matches)) if len(matches) == 1 else None


def parse_lrg_nomenclature(xml_text, *, transcript_id, transcript_exons, context):
    """Match the exact transcript structure before assigning any curated labels."""
    root = ET.fromstring(xml_text)
    fixed = root.find('fixed_annotation')
    if fixed is None:
        raise ValueError('The LRG record has no fixed annotation.')
    lrg_id = fixed.findtext('id')
    organism = fixed.find('organism')
    if organism is None or organism.get('taxon') != '9606':
        raise ValueError('The LRG organism differs from the human reference.')
    ordered = sorted(transcript_exons, key=lambda exon: exon['index'])
    if ([e['index'] for e in ordered] != list(range(1, len(ordered)+1)) or
            any(type(e[key]) is not int for e in ordered for key in ('index', 'start', 'end')) or
            any(e['start'] < 1 or e['end'] < e['start'] or
                _chrom(e.get('seq_region_name') or context['chrom']) != _chrom(context['chrom']) for e in ordered)):
        raise ValueError('Transcript ordinals or coordinates are invalid.')
    strand = _strand(context['strand'])
    if sorted(ordered, key=lambda e: e['start'], reverse=strand == '-') != ordered:
        raise ValueError('Exon order disagrees with the transcript strand.')
    genomic = sorted(ordered, key=lambda e: e['start'])
    if any(left['end'] >= right['start'] for left, right in zip(genomic, genomic[1:])):
        raise ValueError('Transcript exon intervals overlap.')
    mappings = [m for m in root.findall('./updatable_annotation/annotation_set/mapping')
                if _assembly(m.get('coord_system')) == _assembly(context['assembly'])
                and _chrom(context['chrom']) in {_chrom(m.get('other_name')), m.get('other_id'), m.get('other_id_syn')}
                and (m.get('type') == 'main_assembly' or context['chrom'] in {m.get('other_id'), m.get('other_id_syn')})]
    transcripts = [t for t in root.findall('./updatable_annotation/annotation_set/features/gene/transcript')
                   if _identifier_matches(transcript_id, t.get('accession', ''))]
    # LRG transcript identifiers address the fixed section directly.
    transcripts += [t for t in fixed.findall('transcript') if transcript_id == lrg_id+t.get('name', '')]
    structures = []
    expected = [(e['start'], e['end'], strand) for e in ordered]
    for transcript in transcripts:
        coordinates = [_coordinates(e, lrg_id) for e in transcript.findall('exon')]
        for mapping in mappings:
            actual = [_project_interval(c, mapping) for c in coordinates]
            if actual == expected:
                structures.append((transcript, coordinates, mapping))
    if not structures:
        raise ValueError('The LRG record has no exact transcript, assembly, strand and exon-coordinate match.')
    resolutions = []
    for transcript, coordinates, mapping in structures:
        fixed_id = transcript.get('fixed_id') or transcript.get('name')
        fixed_transcripts = [t for t in fixed.findall('transcript') if t.get('name') == fixed_id] if fixed_id else [
            t for t in fixed.findall('transcript') if set(coordinates).intersection(
                _coordinates(e, lrg_id) for e in t.findall('exon'))]
        if len(fixed_transcripts) != 1:
            continue
        fixed_id = fixed_transcripts[0].get('name')
        standard = {_coordinates(e, lrg_id): e.get('label') for e in fixed_transcripts[0].findall('exon')}
        if not any(standard.get(c) for c in coordinates):
            continue
        naming = []
        for annotation in root.findall('./updatable_annotation/annotation_set/fixed_transcript_annotation'):
            if annotation.get('name') != fixed_id:
                continue
            for other in annotation.findall('other_exon_naming'):
                # Coordinates, labels and attribution are explicit XML fields.
                labels = {}
                for exon in other.findall('exon'):
                    coordinate = _coordinates(exon, lrg_id)
                    label = (exon.findtext('label') or '').strip()
                    if coordinate in labels and labels[coordinate] != label:
                        raise ValueError('Curated exon names conflict at the same coordinates.')
                    labels[coordinate] = label
                if any(not labels.get(c) for c in standard):
                    raise ValueError('The alternative naming scheme is incomplete for this transcript.')
                if set(labels) - set(standard):
                    raise ValueError('The alternative naming scheme contains coordinates outside its fixed transcript.')
                description, refs = other.get('description'), [u.text for u in other.findall('url') if u.text]
                if not description or not refs:
                    raise ValueError('The alternative naming scheme lacks attribution.')
                naming.append((labels, description, refs))
        if not naming:
            naming = [(standard, f'{lrg_id} exon numbering', [])]
        # Preserve a RefSeq alias only when the XML explicitly pairs it and
        # its complete exon structure agrees with the selected transcript.
        paired_ids = {node.get('accession') for node in transcript.findall('db_xref')
                      if node.get('source') == 'RefSeq'}
        if transcript.get('accession', '').startswith(('NM_', 'NR_')):
            paired_ids.add(transcript.get('accession'))
        refseq_ids = sorted({t.get('accession') for t in root.findall(
            './updatable_annotation/annotation_set/features/gene/transcript')
            if t.get('accession') in paired_ids and t.get('fixed_id') == fixed_id
            and [_coordinates(e, lrg_id) for e in t.findall('exon')] == coordinates})
        for labels, description, refs in naming:
            names = [labels.get(c) for c in coordinates]
            available = [name for name in names if name]
            if len({name.casefold() for name in available}) != len(available):
                raise ValueError('An exon name occurs more than once in this transcript.')
            resolutions.append({'lrg_id': lrg_id, 'fixed_transcript_id': lrg_id+fixed_id,
                'source_transcript_id': transcript.get('accession') or transcript_id,
                'refseq_accession': refseq_ids[0] if len(refseq_ids) == 1 else None,
                'naming_system': description, 'naming_refs': refs, 'assembly_mapping': dict(mapping.attrib),
                'ordered_biological_labels': names,
                'named_exon_count': len(available), 'total_exon_count': len(names),
                'exons': [{**{key: exon[key] for key in ('index', 'start', 'end')},
                           'label': name, 'lrg_label': standard.get(c), 'lrg_start': c[0], 'lrg_end': c[1]}
                          for exon, name, c in zip(ordered, names, coordinates)]})
    signatures = {json.dumps(r['ordered_biological_labels']) for r in resolutions}
    if len(signatures) != 1:
        raise ValueError('Curated exon naming is unavailable or conflicts between source schemes.')
    return resolutions[0]


def fetch_curated_exon_nomenclature(*, transcript_id, transcript_exons, context, timeout=15):
    context = dict(context or {})
    result = {'status': 'unresolved', 'source': 'refseqgene_lrg', 'resolver_version': RESOLVER_VERSION,
              'reference_transcript_id': transcript_id,
              'reference_fingerprint': reference_fingerprint(transcript_id, transcript_exons, context),
              'reference': {key: context.get(key) for key in ('assembly', 'chrom', 'strand')}}
    if (context.get('species') not in {'human', 'homo_sapiens', 'Homo sapiens', '9606', 9606} or
            not context.get('assembly') or not context.get('chrom') or not _strand(context.get('strand')) or
            not transcript_exons or context.get('modified') or not (transcript_id.startswith(('ENS', 'NM_', 'NR_', 'LRG_'))) or
            transcript_id.startswith('ENS') and not context.get('gene_symbol') or
            any(marker in transcript_id for marker in ('|', ':', '__'))):
        return {**result, 'status': 'not_attempted', 'reason': 'Curated naming requires a complete human reference identity.'}
    try:
        text, index_sha = _read(INDEX_URL, timeout)
        reader = csv.DictReader(io.StringIO(text), delimiter='\t')
        if not {'#tax_id', 'Symbol', 'LRG', 'RNA', 'RSG'}.issubset(reader.fieldnames or []):
            raise ValueError('The reference index has an unexpected schema.')
        candidates = {}
        for row in reader:
            lrg = row.get('LRG') or ''
            if row.get('#tax_id') != '9606' or not lrg.startswith('LRG_') or not lrg[4:].isdigit():
                continue
            if (_identifier_matches(transcript_id, row.get('RNA') or '') or
                    transcript_id == lrg+(row.get('t') or '') or
                    context.get('gene_symbol') and context['gene_symbol'].casefold() == row.get('Symbol', '').casefold()):
                candidates.setdefault(lrg, []).append(row)
        if not candidates:
            return {**result, 'reason': 'No LRG record is listed for this reference.', 'evidence_refs': [INDEX_URL], 'xref_sha256': index_sha}
        found, failures = [], []
        for lrg, rows in sorted(candidates.items()):
            url = LRG_URL.format(lrg)
            try:
                xml_text, source_sha = _read(url, timeout)
                parsed = parse_lrg_nomenclature(xml_text, transcript_id=transcript_id,
                    transcript_exons=transcript_exons, context=context)
                if parsed['lrg_id'] != lrg:
                    raise ValueError('The downloaded record has a different LRG identifier.')
                found.append({**result, **parsed, 'status': 'verified', 'source_url': url,
                    'evidence_refs': [INDEX_URL, url, *parsed['naming_refs']],
                    'xref_sha256': index_sha, 'source_sha256': source_sha,
                    'refseqgene_accessions': sorted({r['RSG'] for r in rows}),
                    **({'source_ensembl_transcript_id': transcript_id,
                        'ensembl_transcript_id': transcript_id.split('.', 1)[0],
                        'ensembl_exon_count': len(transcript_exons), 'refseq_exon_count': len(transcript_exons),
                        'ensembl_exon_lengths': [e['end']-e['start']+1 for e in parsed['exons']],
                        'refseq_exon_lengths': [e['end']-e['start']+1 for e in parsed['exons']]}
                       if transcript_id.startswith('ENS') and parsed.get('refseq_accession') else {}),
                    'reason': 'Curated names are bound to exact exon coordinates within the selected reference.'})
            except (OSError, ValueError, ET.ParseError, TypeError, KeyError) as exc:
                failures.append({'source_url': url, 'reason': str(exc) if isinstance(exc, ValueError) else type(exc).__name__})
        if found and len({json.dumps(r['ordered_biological_labels']) for r in found}) == 1:
            return found[0]
        return {**result, 'reason': 'Curated naming could not be verified for this reference.',
                'evidence_refs': [INDEX_URL], 'xref_sha256': index_sha, 'source_failures': failures}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {**result, 'reason': 'The curated naming source could not be read.', 'exception_type': type(exc).__name__}


def ensure_entry_nomenclature(entry):
    """Attach a run-local receipt, invalidating it when reference identity changes."""
    context = reference_context(entry)
    exons = entry.get('exons_tx') or []
    fingerprint = reference_fingerprint(context['transcript_id'], exons, context)
    if not isinstance(entry.get('metadata'), dict):
        entry['metadata'] = {}
    metadata = entry['metadata']
    saved = metadata.get('curated_exon_nomenclature') or {}
    if saved.get('reference_fingerprint') != fingerprint or saved.get('resolver_version') != RESOLVER_VERSION:
        saved = fetch_curated_exon_nomenclature(transcript_id=context['transcript_id'],
            transcript_exons=exons, context=context)
        metadata['curated_exon_nomenclature'] = saved
    return saved


def annotate_goal_exon_nomenclature(goal):
    catalog = goal.get('isoform_transcripts')
    if not isinstance(catalog, dict):
        return
    for entry in catalog.values():
        if isinstance(entry, dict):
            ensure_entry_nomenclature(entry)


def resolve_curated_label(receipt, requested_label):
    requested = str(requested_label).strip().casefold().removeprefix('exon ').strip()
    if receipt.get('status') != 'verified':
        return {**deepcopy(receipt), 'requested_exon_label': str(requested_label)}
    matches = [exon for exon in receipt['exons'] if str(exon.get('label') or '').casefold() == requested]
    if len(matches) != 1:
        return {**deepcopy(receipt), 'status': 'unresolved', 'requested_exon_label': str(requested_label),
                'reason': 'The requested exon name is absent or ambiguous in the curated naming scheme.'}
    exon = matches[0]
    return {**deepcopy(receipt), 'status': 'verified_curated_mapping',
            'requested_exon_label': str(requested_label), 'resolved_transcript_ordinal': exon['index']}


def verified_mapping_for_entry(resolution, entry):
    if not isinstance(resolution, dict) or resolution.get('status') not in VERIFIED_NOMENCLATURE_STATUSES:
        return False
    if resolution['status'] == 'verified_alternate_mapping':
        return True  # Existing independently reviewed source contract.
    context = reference_context(entry)
    return (resolution.get('reference_transcript_id') == context['transcript_id'] and
            resolution.get('reference_fingerprint') == reference_fingerprint(
                context['transcript_id'], entry.get('exons_tx') or [], context))
