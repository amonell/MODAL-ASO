"""Reference-sequence consequences of complete exon removal; not ASO efficacy."""
import hashlib


def simulate_exon_skips(transcript, genome, exon_keys=None):
    if genome is None:
        return {}
    strand = '-' if str(transcript.get('strand')) in {'-', '-1'} else '+'
    reverse = lambda seq: seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]
    pieces, coding = [], []
    offset = 0
    for exon in transcript.get('exons_tx') or []:
        start, end = int(exon['start']), int(exon['end'])
        chrom = str(exon.get('seq_region_name') or transcript.get('chrom') or transcript.get('seq_region_name') or '')
        chroms = list(dict.fromkeys([chrom, chrom.removeprefix('chr'), 'chr' + chrom.removeprefix('chr')]))
        selected = next((c for c in chroms if c in genome), None)
        if selected is None:
            raise KeyError(f'Reference chromosome absent: {chrom}')
        sequence = str(genome[selected][start - 1:end]).upper()
        if len(sequence) != end - start + 1:
            raise ValueError('Incomplete reference exon sequence')
        if strand == '-':
            sequence = reverse(sequence)
        key = (selected, strand, start, end)
        piece = {'key':key, 'sequence':sequence, 'start0':offset, 'end0':offset + len(sequence), 'index':exon.get('index')}
        pieces.append(piece)
        a, b = exon.get('coding_start_genomic'), exon.get('coding_end_genomic')
        if exon.get('contains_coding') and isinstance(a, int) and isinstance(b, int):
            lo, hi = min(a,b), max(a,b)
            if not (start <= lo <= hi <= end):
                raise ValueError('Coding interval outside its reference exon')
            coding.append((offset + (lo-start if strand == '+' else end-hi),
                           offset + (hi-start+1 if strand == '+' else end-lo+1)))
        offset += len(sequence)
    if not coding:
        return {}
    cdna = ''.join(p['sequence'] for p in pieces)
    cds_start, cds_end = min(a for a,b in coding), max(b for a,b in coding)
    def first_stop(sequence, start):
        return next((i for i in range(start, len(sequence)-2, 3) if sequence[i:i+3] in {'TAA','TAG','TGA'}), None)
    baseline_stop = first_stop(cdna, cds_start)
    baseline_valid = cdna[cds_start:cds_start+3] == 'ATG' and baseline_stop in {cds_end-3, cds_end}
    output = {}
    for index, piece in enumerate(pieces):
        key = piece['key']
        if exon_keys is not None and key not in exon_keys:
            continue
        removed_coding = max(0, min(piece['end0'], cds_end) - max(piece['start0'], cds_start))
        result = {'skipped_exon_index':piece['index'], 'skipped_genomic_interval_1based_inclusive':list(key),
            'removed_exon_length':len(piece['sequence']), 'removed_coding_length':removed_coding,
            'coding_frame_shift':bool(removed_coding % 3), 'reference_cdna_sha256':hashlib.sha256(cdna.encode()).hexdigest(),
            'baseline_cds_start_cdna0':cds_start, 'baseline_stop_start_cdna0':baseline_stop,
            'nmd_predicted':None, 'premature_stop':None, 'stop_pos_nt':None,
            'method':'Complete exon removal on the attested reference cDNA, original translation start, standard genetic code, and a >55-nt stop-end-to-last-junction heuristic.',
            'method_reference':'https://pubmed.ncbi.nlm.nih.gov/9710612/',
            'limitations':'Conditional sequence consequence only. Does not predict ASO occupancy, actual exon skipping, alternative initiation, cryptic splicing, EJC deposition, tissue-specific NMD or knockdown; NMD exceptions are not modeled.'}
        if not baseline_valid:
            result['status'] = 'unverified_reference_cds'
        elif piece['start0'] <= cds_start < piece['end0']:
            result['status'] = 'translation_start_removed'
        elif baseline_stop is not None and piece['start0'] <= baseline_stop < piece['end0']:
            result['status'] = 'normal_termination_region_removed'
        else:
            alternate = cdna[:piece['start0']] + cdna[piece['end0']:]
            alt_start = cds_start - (len(piece['sequence']) if piece['end0'] <= cds_start else 0)
            stop = first_stop(alternate, alt_start)
            expected_stop = baseline_stop - (len(piece['sequence']) if piece['end0'] <= baseline_stop else 0)
            remaining = [p for i,p in enumerate(pieces) if i != index]
            last_junction = sum(len(p['sequence']) for p in remaining[:-1]) if len(remaining) > 1 else None
            premature = stop is not None and stop < expected_stop
            distance = last_junction - (stop + 3) if last_junction is not None and stop is not None else None
            result.update(status='computed', alternate_cdna_sha256=hashlib.sha256(alternate.encode()).hexdigest(),
                alt_cds_length=(stop+3-alt_start if stop is not None else len(alternate)-alt_start),
                stop_pos_nt=stop-alt_start if stop is not None else None,
                stop_pos_aa=(stop-alt_start)//3 if stop is not None else None,
                alternate_stop_start_cdna0=stop, alternate_last_junction_cdna0=last_junction,
                stop_end_to_last_junction_nt=distance, premature_stop=premature,
                nmd_predicted=bool(premature and distance is not None and distance > 55),
                coordinate_system='cDNA offsets and stop_pos_nt are zero-based; genomic interval is one-based inclusive')
        output[key] = result
    return output


def requested_skip_reports(goal, catalog, genome_path):
    """Assess named single-exon skipping events on unambiguous saved references."""
    events = []
    containers = [goal, *(goal.get('design_programs') or []), *(goal.get('evaluation_targets') or []),
                  *(goal.get('splice_isoform_objectives') or [])]
    for item in containers:
        if not isinstance(item, dict) or item.get('desired_effect') != 'skipping':
            continue
        labels = item.get('exon_labels') or [item.get('exon_label')]
        if not isinstance(labels, list):
            labels = [labels]
        identity = item.get('transcript_id') or item.get('isoform_label') or (goal.get('transcript_lock') or {}).get('value')
        for exon in labels:
            if isinstance(exon, int) and (identity, exon) not in events:
                events.append((identity, exon))
    for item in goal.get('conditional_exon_skip_events') or []:
        if not isinstance(item, dict):
            continue
        exon = item.get('exon_label')
        identity = item.get('transcript_id') or (goal.get('transcript_lock') or {}).get('value')
        if isinstance(exon, int) and exon > 0 and (identity, exon) not in events:
            events.append((identity, exon))
    if not events:
        return []
    from pyfaidx import Fasta
    reports = []
    with Fasta(str(genome_path), rebuild=False) as genome:
        cache = {}
        for identity, exon in events:
            labels = []
            for label, entry in catalog.items():
                meta = entry.get('metadata') or {}
                if identity and identity in {label, entry.get('transcript_id'), entry.get('versioned_id'), meta.get('versioned_id')}:
                    labels.append(label)
            if not identity and len(catalog) == 1:
                labels = list(catalog)
            if len(labels) != 1:
                reports.append({'requested_reference':identity, 'requested_exon':exon, 'status':'unresolved_reference', 'nmd_predicted':None})
                continue
            label = labels[0]; entry = catalog[label]
            if label not in cache:
                cache[label] = simulate_exon_skips(entry, genome)
            matches = [value for value in cache[label].values() if value['skipped_exon_index'] == exon]
            reports.append({'requested_reference':identity, 'catalog_label':label,
                'reference_transcript_id':(entry.get('metadata') or {}).get('versioned_id') or entry.get('transcript_id'),
                'requested_exon':exon, 'event_scope':'single complete exon removal; not joint multi-exon deletion',
                'consequence':matches[0] if len(matches)==1 else {'status':'missing_exon_or_coding_annotation','nmd_predicted':None}})
    return reports
