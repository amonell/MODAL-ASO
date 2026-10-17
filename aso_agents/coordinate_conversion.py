"""Conservative assembly conversion with retained source coordinates and evidence.

Only one complete ungapped chain mapping is accepted. Split, partial and
ambiguous mappings remain unresolved; their bounding boxes are never tiled.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import gzip
import hashlib
import re
import tempfile

import requests
from Bio import Align


_BUILD_ALIASES = {'hg19':'GRCh37', 'hg38':'GRCh38', 'mm10':'GRCm38', 'mm39':'GRCm39'}


def coordinate_convention_question(request_text, qa_history, envelope):
    """Do not turn untyped user-written intervals into assumed BED coordinates.

    A model-generated coordinate_convention field is not user evidence. This
    guard concerns explicit intervals in the user's request, not coordinates
    obtained from annotated transcript features or declared-format attachments.
    """
    if not re.search(r'\bchr[0-9A-Za-z_.]+\s*:\s*\d[\d,]*\s*[-–]\s*\d', str(request_text), re.I):
        return None
    goals = [envelope.get('structured_goal') or {}]
    goals += [(task.get('context') or {}).get('structured_goal') or {}
              for task in envelope.get('tasks') or [] if isinstance(task, dict)]
    constraints = [constraint for goal in goals
                   for container in [goal, *(goal.get('design_programs') or [])]
                   for constraint in container.get('region_constraints') or []]
    if not any(c.get('source_type') == 'explicit_rows' and c.get('coordinate_space') == 'genomic'
               and c.get('intervals_bed0_half_open') for c in constraints):
        return None
    answers = [str(item.get('answer') or '') for item in qa_history or [] if isinstance(item, dict)]
    evidence = '\n'.join([str(request_text), *answers])
    if re.search(r'\b(?:[01]|zero|one)[ -]?based\b|\bBED(?:[036])?\b', evidence, re.I):
        return None
    return ('Are the written genomic intervals zero-based half-open (BED) or one-based inclusive? '
            'Please specify the convention for each interval if they differ; their original values and builds will be retained.')


def assembly_name(value):
    text = str(value or '').strip()
    for alias, assembly in _BUILD_ALIASES.items():
        if text.lower() in {alias.lower(), assembly.lower(), (assembly+'/'+alias).lower(), (alias+'/'+assembly).lower()}:
            return assembly
    return text or None


def _ucsc_name(value):
    assembly = assembly_name(value)
    result = next((k for k,v in _BUILD_ALIASES.items() if v == assembly), str(value or ''))
    if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9]*', result):
        raise ValueError('Unsupported assembly identifier for UCSC chain retrieval')
    return result


@lru_cache(maxsize=8)
def _chain_alignments(path, sha256):
    """Load once per chain snapshot; Biopython handles reverse coordinates."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise ValueError('Assembly chain changed while loading')
    by_chrom = {}
    with gzip.open(path, 'rt') as handle:
        for alignment in Align.parse(handle, 'chain'):
            by_chrom.setdefault(alignment.target.id, []).append(alignment)
    return by_chrom


def map_interval(alignments, row):
    chrom, start, end, name, score, strand = row
    if int(start) < 0 or int(end) <= int(start):
        raise ValueError('Invalid BED0 source interval')
    matches, overlapping = [], 0
    for alignment in alignments.get(str(chrom), []):
        coords = alignment.coordinates
        if max(coords[0]) <= start or min(coords[0]) >= end:
            continue
        overlapping += 1
        for i in range(coords.shape[1]-1):
            a, b = int(coords[0,i]), int(coords[0,i+1])
            c, d = int(coords[1,i]), int(coords[1,i+1])
            if not a <= start < end <= b or b-a != abs(d-c):
                continue
            direction = 1 if d > c else -1
            mapped_start, mapped_end = c + direction*(start-a), c + direction*(end-a)
            mapped_strand = strand if direction == 1 or strand not in {'+', '-'} else ('-' if strand == '+' else '+')
            matches.append({'row':[alignment.query.id, min(mapped_start,mapped_end), max(mapped_start,mapped_end), name, score, mapped_strand],
                            'chain_id':alignment.annotations.get('id'), 'chain_score':alignment.score})
    if len(matches) != 1 or overlapping != 1:
        return None, {'status':'ambiguous_or_incomplete_mapping', 'complete_mappings':len(matches), 'overlapping_chains':overlapping,
                      'reason':'Requires one complete ungapped chain mapping; no silent selection, partial interval or gap-spanning bounding box.'}
    return matches[0]['row'], {'status':'converted', **{k:v for k,v in matches[0].items() if k != 'row'}}


def convert_rows(rows, constraint, goal, run_dir):
    from .species_config import get_species_config, normalize_species
    config = get_species_config(normalize_species(goal.get('species') or 'human'))
    target = config.ensembl_assembly
    originals = constraint.get('intervals_bed0_half_open') or [{} for _ in rows]
    result, records = [], []
    if len(originals) != len(rows):
        return [], {'status':'unresolved', 'reason':'Some original intervals were invalid; source-to-row identity cannot be verified.'}
    for row, original in zip(rows, originals):
        source = assembly_name(original.get('genome_build') or constraint.get('genome_build'))
        record = {'original_row_bed0_half_open':deepcopy(row), 'original_build':source,
                  'target_build':target, 'coordinate_system':'0based_half_open'}
        if not source or source == target:
            record.update(status='native_reference' if source else 'native_reference_default', mapped_row_bed0_half_open=deepcopy(row))
            result.append(row)
        elif constraint.get('allow_coordinate_conversion') is not True:
            record.update(status='conversion_requires_confirmation', reason='Source and installed reference assemblies differ; original coordinates retained.')
        else:
            try:
                src, dst = _ucsc_name(source), _ucsc_name(target)
                filename = src + 'To' + dst[0].upper() + dst[1:] + '.over.chain.gz'
                url = f'https://hgdownload.soe.ucsc.edu/goldenPath/{src}/liftOver/{filename}'
                folder = Path(run_dir)/'coordinate_conversion'; folder.mkdir(parents=True, exist_ok=True)
                path = folder/filename
                if not path.exists():
                    response = requests.get(url, timeout=(10, 45)); response.raise_for_status()
                    # Validate compression before caching. Download failure is technical, not biological infeasibility.
                    gzip.decompress(response.content)
                    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
                        handle.write(response.content); temporary = Path(handle.name)
                    temporary.replace(path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                lookup_row = deepcopy(row)
                lookup_row[0] = 'chr' + str(row[0]).removeprefix('chr')
                mapped, status = map_interval(_chain_alignments(str(path), sha), lookup_row)
                record.update(status, chain_url=url, chain_path=str(path), chain_sha256=sha)
                if mapped:
                    result.append(mapped); record['mapped_row_bed0_half_open'] = mapped
            except (requests.RequestException, OSError, ValueError) as exc:
                record.update(status='conversion_evidence_unavailable', reason=str(exc), technical_failure=True)
        records.append(record)
    catalog = goal.get('all_transcripts_full') or goal.get('isoform_transcripts') or {}
    spans = []
    for label, entry in catalog.items():
        exons = entry.get('exons_tx') or []
        if not exons:
            continue
        chrom = entry.get('chrom') or entry.get('seq_region_name') or exons[0].get('seq_region_name')
        if chrom:
            spans.append({'chrom':str(chrom).removeprefix('chr'), 'start':min(int(e['start']) for e in exons)-1,
                          'end':max(int(e['end']) for e in exons), 'transcript_id':entry.get('transcript_id') or label})
    if spans and goal.get('gene_symbol'):
        allowed = []
        for record in records:
            mapped = record.get('mapped_row_bed0_half_open')
            if not mapped:
                continue
            matches = [s['transcript_id'] for s in spans if s['chrom']==str(mapped[0]).removeprefix('chr') and mapped[1]<s['end'] and s['start']<mapped[2]]
            record['resolved_gene_context'] = {'gene_symbol':goal['gene_symbol'], 'overlapping_reference_spans':matches, 'reference_spans_bed0_half_open':spans}
            if matches:
                allowed.append(mapped)
            else:
                record['design_status'] = 'outside_resolved_gene_reference'
                record['design_reason'] = 'The supplied/mapped interval is outside all resolved transcript spans for the requested gene; it is retained as evidence but is not an RNA-target design region for this gene.'
        result = allowed
    return result, {'status':'materialized' if len(result) == len(rows) else 'partially_materialized' if result else 'unresolved',
                    'target_build':target, 'coordinate_conversions':records,
                    'method':'UCSC assembly chains parsed by Bio.Align; unique complete ungapped interval mapping, with original intervals retained.',
                    'method_reference':'https://genome.ucsc.edu/goldenPath/help/chain.html'}


def conversion_plan_builds(goal):
    """Return declared builds only when all foreign intervals can reach conversion.

    Admission is not evidence of successful mapping. The materializer must
    independently produce chain evidence before any converted interval is used.
    """
    from .species_config import get_species_config, normalize_species
    try:
        target = get_species_config(normalize_species(goal.get('species') or 'human')).ensembl_assembly
    except ValueError:
        return None  # Native reference provisioning handles unsupported species.
    cleaned = deepcopy(goal)
    builds = {target}
    containers = [cleaned, *(cleaned.get('design_programs') or [])]
    found = False
    source_intervals = set()
    for container in containers:
        for constraint in container.get('region_constraints') or []:
            intervals = constraint.get('intervals_bed0_half_open')
            if not intervals:
                continue
            for interval in intervals:
                source = assembly_name(interval.get('genome_build') or constraint.get('genome_build'))
                if not source:
                    return None
                builds.add(source)
                if isinstance(interval.get('start'), int) and isinstance(interval.get('end'), int):
                    chrom = str(interval.get('chrom') or '').removeprefix('chr')
                    # These are two representations of the same source interval,
                    # not calculated liftover coordinates.
                    source_intervals.add((source, chrom, interval['start'], interval['end']))
                    source_intervals.add((source, chrom, interval['start'] + 1, interval['end']))
                if source != target:
                    if constraint.get('allow_coordinate_conversion') is not True or not constraint.get('coordinate_convention'):
                        return None
                    found = True
                interval.pop('genome_build', None)
                # This is a validation-only copy: a display label repeats the
                # preserved source coordinates, not an additional operative
                # interval. Keep all original labels on the conversion input.
                interval.pop('label', None)
            constraint.pop('genome_build', None)
            constraint.pop('label', None)
    if found:
        _mask_source_coordinate_aliases(cleaned, source_intervals)
    return (builds, cleaned, target) if found else None


def _mask_source_coordinate_aliases(value, source_intervals):
    """Validation-only removal of proven source-coordinate repetitions in prose.

    Original data are never modified. Unmatched intervals and independent typed
    assembly declarations remain visible to mixed-build validation.
    """
    pattern = re.compile(r'\b(?:chr)?(?P<chrom>[0-9]{1,2}|X|Y|MT|M)\s*:\s*(?P<start>\d[\d,]*)\s*[-–]\s*(?P<end>\d[\d,]*)\b', re.I)
    build_pattern = re.compile(r'\b(?:hg19|hg38|GRCh37|GRCh38|mm10|mm39|GRCm38|GRCm39)\b', re.I)
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'label', 'objective', 'target_region'} and isinstance(item, str):
                def replace(match):
                    preceding = list(build_pattern.finditer(item[:match.start()]))
                    anywhere = list(build_pattern.finditer(item))
                    build = assembly_name(preceding[-1].group() if preceding else anywhere[0].group()) if preceding or len(anywhere) == 1 else None
                    identity = (match['chrom'], int(match['start'].replace(',', '')), int(match['end'].replace(',', '')))
                    candidates = [entry for entry in source_intervals if entry[1:] == identity and (build is None or entry[0] == build)]
                    return '[coordinate alias of retained source interval]' if candidates else match.group()
                value[key] = pattern.sub(replace, item)
            elif isinstance(item, (dict, list)):
                _mask_source_coordinate_aliases(item, source_intervals)
    elif isinstance(value, list):
        for item in value:
            _mask_source_coordinate_aliases(item, source_intervals)
