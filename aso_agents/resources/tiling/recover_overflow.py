#!/usr/bin/env python
"""Finish censored Bowtie queries with disk-backed counting and bounded previews.

The initial capped search is fast for ordinary queries. Only its overflow
queries are searched again, with all placements requested. Duplicate sequences
share one search; each original query keeps its own intended-site exclusion.
"""
import argparse
import bisect
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile


def read_fasta(path):
    records = {}
    opener = gzip.open if str(path).endswith('.gz') else open
    name, parts = None, []
    def save():
        if name is not None:
            sequence = ''.join(parts).upper()
            if not sequence or re.search('[^ACGTN]', sequence):
                raise ValueError('Invalid query sequence')
            if name in records and records[name] != sequence:
                raise ValueError('Conflicting query sequences')
            records[name] = sequence
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.startswith('>'):
                save()
                name, parts = line[1:].split()[0], []
            elif line.strip():
                if name is None:
                    raise ValueError('Missing FASTA query name')
                parts.append(line.strip())
    save()
    return records


def gene_index(path):
    """Inclusive gene spans, with prefix maxima for efficient overlap lookup."""
    index = {}
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            attrs = dict(re.findall(r'(gene_name|gene_id) "([^"]*)"', fields[8]))
            index.setdefault(fields[0], []).append((int(fields[3]), int(fields[4]),
                attrs.get('gene_name', ''), attrs.get('gene_id', '')))
    result = {}
    for chrom, rows in index.items():
        rows.sort()
        maximum, ends = 0, []
        for row in rows:
            maximum = max(maximum, row[1])
            ends.append(maximum)
        result[chrom] = ([row[0] for row in rows], ends, rows)
    return result


def overlapping_genes(index, chrom, start, end):
    if chrom not in index:
        return
    starts, maximum_ends, rows = index[chrom]
    i = bisect.bisect_right(starts, end) - 1
    while i >= 0 and maximum_ends[i] >= start:
        if rows[i][1] >= start:
            yield rows[i][2], rows[i][3]
        i -= 1


def target_start(name):
    match = re.search(r'(?:^|\|)pos=([^:|]+):(\d+)-', name)
    return (match[1], int(match[2])) if match else (None, None)


def aggregate_sam(stream, database, sequences, budget):
    """Validate every record and keep unique loci on disk, including MAPQ zero."""
    database.execute('CREATE TABLE hits (query TEXT, mm INTEGER, chrom TEXT, pos INTEGER, '
                     'PRIMARY KEY(query, mm, chrom, pos)) WITHOUT ROWID')
    seen, unmapped, batch = set(), set(), []
    digest = hashlib.sha256()
    for line in stream:
        digest.update(line.encode())
        if line.startswith('@') or not line.strip():
            continue
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 11 or parts[0] not in sequences:
            raise ValueError('Unexpected or malformed alignment record')
        name, flag = parts[0], int(parts[1])
        sequence = parts[9].upper()
        if flag & 16:
            sequence = sequence.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]
        if sequence != sequences[name]:
            raise ValueError('Alignment sequence disagrees with query')
        if flag & 4:
            unmapped.add(name)
            continue
        tags = [tag[5:] for tag in parts[11:] if tag.startswith('NM:i:')]
        if len(tags) != 1:
            raise ValueError('Missing alignment mismatch count')
        mm, pos, chrom = int(tags[0]), int(parts[3]), parts[2]
        if mm not in range(budget + 1) or pos < 1 or chrom == '*' or parts[5] != f'{len(sequence)}M':
            raise ValueError('Alignment falls outside the requested search')
        seen.add(name)
        batch.append((name, mm, chrom, pos))
        if len(batch) >= 4096:
            database.executemany('INSERT OR IGNORE INTO hits VALUES (?, ?, ?, ?)', batch)
            batch.clear()
    database.executemany('INSERT OR IGNORE INTO hits VALUES (?, ?, ?, ?)', batch)
    database.commit()
    if seen & unmapped or seen | unmapped != set(sequences):
        raise ValueError('Missing or conflicting query evidence')
    return digest.hexdigest()


def summarize(database, queries, identifiers, budget, annotations, preview_limit):
    rows = []
    for name, sequence in queries.items():
        counts = [0] * 4
        positions, genes, gene_ids = ([[] for _ in range(4)], [set() for _ in range(4)], [set() for _ in range(4)])
        exact_total, exact_preview = 0, []
        intended = target_start(name)
        for mm, chrom, pos in database.execute(
                'SELECT mm, chrom, pos FROM hits WHERE query=? ORDER BY mm, chrom, pos',
                (identifiers[sequence],)):
            location = f'{chrom}:{pos}-{pos + len(sequence) - 1}'
            if mm == 0:
                exact_total += 1
                if len(exact_preview) < preview_limit:
                    exact_preview.append(location)
            if (chrom, pos) == intended:
                continue
            counts[mm] += 1
            if len(positions[mm]) < preview_limit:
                positions[mm].append(location)
            for symbol, identifier in overlapping_genes(annotations, chrom, pos, pos + len(sequence) - 1):
                if symbol:
                    genes[mm].add(symbol)
                if identifier:
                    gene_ids[mm].add(identifier)
        metadata = {}
        for mm in range(4):
            label = 'exact' if mm == 0 else f'{mm}mm'
            metadata[f'off_target_{label}_count'] = str(counts[mm]) if mm <= budget else 'NA'
            metadata[f'off_target_{mm}mm_positions'] = (','.join(positions[mm]) or 'None') if mm <= budget else 'NA'
            metadata[f'off_target_{mm}mm_genes'] = ','.join(sorted(genes[mm]))
            metadata[f'off_target_{mm}mm_gene_ids'] = ','.join(sorted(gene_ids[mm]))
        metadata.update({
            f'off_target_total_upto_{budget}mm_count': str(sum(counts)),
            'off_target_mismatch_budget': str(budget),
            'off_target_weighted_risk_score': str(sum(c * w for c, w in zip(counts, (100, 10, 3, 1)))),
            'off_target_positions_truncated': str(any(c > preview_limit for c in counts)).lower(),
            'off_target_position_preview_limit': str(preview_limit),
            'off_target_gene_annotation_status': 'complete',
            'off_target_search_method': 'uncapped_overflow_recovery',
        })
        rows.append({'qname': name, 'qseq': sequence, 'metadata': metadata,
                     'exact_mapped_count': exact_total, 'exact_mapped_preview': exact_preview})
    return rows


def recover(overflow, index, budget, annotation, output, *, bowtie='bowtie', preview_limit=100):
    if budget not in range(4) or preview_limit < 1:
        raise ValueError('Invalid mismatch budget or preview limit')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'schema_version': 1, 'status': 'failed', 'rows': [],
               'mismatch_budget': budget, 'index': str(Path(index).resolve()),
               'input_sha256': hashlib.sha256(Path(overflow).read_bytes()).hexdigest()}
    try:
        queries = read_fasta(overflow)
        receipt['query_count'] = len(queries)
        if not queries:
            receipt['status'] = 'complete'
        else:
            annotations = gene_index(annotation)
            identifiers = {seq: f'q{i}' for i, seq in enumerate(dict.fromkeys(queries.values()))}
            sequences = {identifier: seq for seq, identifier in identifiers.items()}
            receipt['unique_sequence_count'] = len(sequences)
            with tempfile.TemporaryDirectory(prefix='.overflow-', dir=output.parent) as work:
                work = Path(work)
                fasta = work / 'queries.fasta'
                fasta.write_text(''.join(f'>{name}\n{seq}\n' for name, seq in sequences.items()))
                command = [bowtie, '-a', '-f', '-v', str(budget), '-y', '--sam', '--sam-nohead',
                           '--seed', '1', str(index), str(fasta)]
                receipt['command'] = command
                with sqlite3.connect(work / 'placements.sqlite') as database, (work / 'bowtie.stderr').open('w+') as errors:
                    database.execute('PRAGMA journal_mode=OFF')
                    database.execute('PRAGMA synchronous=OFF')
                    database.execute('PRAGMA cache_size=-8192')
                    database.execute('PRAGMA temp_store=FILE')
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True) as process:
                        try:
                            receipt['alignment_stream_sha256'] = aggregate_sam(process.stdout, database, sequences, budget)
                        except BaseException:
                            process.terminate()
                            raise
                        finally:
                            process.stdout.close()
                        receipt['returncode'] = process.wait()
                    errors.seek(0)
                    receipt['aligner_summary'] = errors.read()[-8000:]
                    if receipt['returncode']:
                        raise RuntimeError(f'Bowtie exited with status {receipt["returncode"]}')
                    receipt['rows'] = summarize(database, queries, identifiers, budget, annotations, preview_limit)
                    receipt['status'] = 'complete'
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        # Keep the capped evidence available if recovery itself cannot finish.
        receipt['rows'] = []
        receipt['error'] = f'{type(error).__name__}: {error}'
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(receipt, separators=(',', ':')) + '\n')
    temporary.replace(output)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overflow', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--max-mismatches', type=int, required=True)
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = recover(args.overflow, args.index, args.max_mismatches, args.annotation, args.output)
    print(f'Overflow recovery: {result["status"]}; {result.get("query_count", 0)} queries')
