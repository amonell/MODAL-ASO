"""Sequence-evidenced transfer of published UTR sites to a resolved reference."""
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import zipfile


def _rows(path):
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if not n.endswith('/'))
            with archive.open(name) as stream:
                yield from csv.DictReader(io.TextIOWrapper(stream), delimiter='\t')
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            yield from csv.DictReader(stream, delimiter='\t')


def source_utr_record(path, taxon, transcript_id):
    """Index one species once; preserve source identity and the ungapped sequence.

    The TargetScan download calls its transcript column 'Refseq ID', including
    Ensembl IDs. Offsets in predictions count ungapped, spliced UTR bases.
    """
    path = Path(path).resolve()
    stat = path.stat()
    signature = f'{stat.st_size}:{stat.st_mtime_ns}:{taxon}'
    index = path.with_name(path.name + '.' + hashlib.sha256(signature.encode()).hexdigest()[:16] + '.sqlite3')
    if not index.exists():
        with index.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not index.exists():
                temp = index.with_suffix(f'.{os.getpid()}.tmp')
                temp.unlink(missing_ok=True)
                db = sqlite3.connect(temp)
                try:
                    db.execute('CREATE TABLE records (transcript TEXT PRIMARY KEY, body TEXT NOT NULL)')
                    csv.field_size_limit(max(csv.field_size_limit(), 64*1024*1024))
                    with path.open('rb') as handle:
                        source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
                    for row in _rows(path):
                        if str(row.get('Species ID') or '').strip() != str(taxon):
                            continue
                        tid = str(row.get('Transcript ID') or row.get('Refseq ID') or '').strip()
                        sequence = str(row.get('UTR sequence') or '').upper().replace('U', 'T').replace('-', '').replace('.', '')
                        if tid and sequence:
                            body = {'transcript_id':tid, 'taxon_id':str(taxon), 'gene_id':row.get('Gene ID'),
                                    'gene_symbol':row.get('Gene Symbol'), 'ungapped_utr_sequence':sequence,
                                    'source_path':str(path), 'source_sha256':source_hash,
                                    'utr_sha256':hashlib.sha256(sequence.encode()).hexdigest()}
                            db.execute('INSERT INTO records VALUES (?,?)', (tid,json.dumps(body)))
                    db.commit()
                except Exception:
                    db.close(); temp.unlink(missing_ok=True)
                    raise
                db.close(); temp.replace(index)
    with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as db:
        row = db.execute('SELECT body FROM records WHERE transcript=?', (transcript_id,)).fetchone()
    return json.loads(row[0]) if row else None


def project_source_utr_site(source, cdna, utr_start, start, end, flank=50):
    """Require an exact, unique sequence anchor containing the complete site.

    Prefer the entire UTR. When UTR ends or distant bases changed, use the site
    with up to 50 source bases on either side (at least 50 bases in total).
    This validates local coordinate transfer, not whole-version equivalence.
    """
    sequence = str((source or {}).get('ungapped_utr_sequence') or '')
    cdna = str(cdna or '').upper().replace('U', 'T')
    if not sequence or not cdna or utr_start is None:
        return {'status':'unresolved', 'reason':'source_or_reference_sequence_missing'}
    start, end = min(start,end), max(start,end)
    if not (1 <= start <= end <= len(sequence)):
        return {'status':'unresolved', 'reason':'source_site_outside_utr'}
    current_utr = cdna[int(utr_start)-1:]
    offset = current_utr.find(sequence) if set(sequence) <= set('ACGT') else -1
    source_left, source_right = 0, len(sequence)
    anchor = sequence
    method = 'unique_full_source_utr_exact_match_in_reference_utr'
    if offset < 0:
        source_left = max(0,start-1-flank)
        source_right = min(len(sequence),end+flank)
        anchor = sequence[source_left:source_right]
        offset = current_utr.find(anchor) if len(anchor)>=50 and set(anchor)<=set('ACGT') else -1
        method = 'unique_exact_source_site_and_flanks_in_reference_utr'
    if offset < 0 or current_utr.find(anchor, offset+1) >= 0:
        return {'status':'unresolved', 'reason':'source_utr_not_uniquely_identical_in_reference_utr',
                'source_utr_length':len(sequence), 'reference_utr_length':len(current_utr)}
    first = int(utr_start) + offset + start - 1 - source_left
    last = int(utr_start) + offset + end - 1 - source_left
    return {'status':'verified', 'method':method,
            'source_transcript_id':source['transcript_id'], 'source_utr_sha256':source['utr_sha256'],
            'source_utr_length':len(sequence),
            'source_anchor_bed0':[source_left,source_right], 'source_anchor_sequence_5to3':anchor,
            'reference_anchor_utr_bed0':[offset,offset+len(anchor)],
            'transfer_scope':'Exact source site and recorded surrounding sequence; distal UTR changes and source prediction scores are not revalidated.',
            'reference_cdna_sha256':hashlib.sha256(cdna.encode()).hexdigest(),
            'reference_utr_cdna_start_1based':int(utr_start),
            'cdna_start_1based':first, 'cdna_end_1based':last,
            'site_sequence_5to3':sequence[start-1:end]}
