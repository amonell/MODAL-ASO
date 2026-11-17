"""Read-only exclusion history shared by design and supplied-candidate reports.

Use explicit rejection buckets, admission receipts and sequence-stage traces.
Never interpret a file called ``filtered_tiles`` (the survivors), absence from
a shortlist, or a failed predictive safeguard as an exclusion. Sequence-only
traces cannot establish the chemistry, locus or orientation of an ambiguous ASO.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .report_explorer_data import (collection, coordinates, notation, number,
                                   physical_identity, read_record, sequence, source_row)
from .run_layout import resolve_run_path


# These are rejection buckets, not the many native "filtered" survivor files.
BUCKETS = {
    'filtered_no_homopolymer': ('Initial sequence filters', 'Excluded by the recorded homopolymer filter.'),
    'filtered_gc_out_of_range': ('Initial sequence filters', 'GC is outside the recorded permitted range.'),
    'filtered_binding_sites_exceed': ('Binding-site filter', 'Binding-site count exceeds the recorded limit.'),
    'filtered_outside_target_rna': ('Target-RNA admission', 'Not fully contained on the target strand in any recorded target-RNA locus.'),
    'cross_species_excluded': ('Required species matching', 'Failed the required cross-species matching check.'),
}
TRACE_STAGES = {
    'filtered_during_initial_filter': ('Initial sequence filters', 'Present in the tiling inventory but absent after initial filtering.'),
    'filtered_during_scoring_or_ranking': ('Before or during scoring', 'Passed the initial filters but is absent from the scored inventory.'),
}


def _bucket(path):
    stem = Path(path).stem
    if stem in BUCKETS:
        return stem
    if stem.endswith('_outside_target_rna'):
        return 'filtered_outside_target_rna'
    if stem.endswith('.cross_species_excluded'):
        return 'cross_species_excluded'
    return None


def _reason(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return '; '.join(filter(None, (_reason(v) for v in value)))
    if isinstance(value, dict):
        return _reason(value.get('reason') or value.get('message') or value.get('code'))
    return ''


def _normalize(row, path):
    """Use declared antisense roles and the native evaluation mapping contract."""
    row = source_row(row)
    if not sequence(row) and row.get('declared_sequence_role') == 'antisense_oligo_sequence':
        row = {**row, 'antisense_5to3': row.get('sequence')}
    if (path and path.name in {'existing_aso_evaluation.json', 'provided_aso_evaluation.json', 'evaluation_evidence.json'}
            and row.get('genomic_segments') and not row.get('coordinate_system')):
        row = {**row, 'coordinate_system': 'genomic_1based_inclusive'}
    return row


def project_filter_history(run_dir, *, goal=None):
    """Return compact, scoped display records without changing analysis artifacts.

    Future producers can expose any admission rule through an ``excluded_path``
    receipt, a ``filtered_<rule>: {path, reason}`` bucket, or ``excluded_rows`` in
    a supplied-candidate evaluation. No gene, mechanism or species is assumed.
    Only files inside this run are followed; missing evidence stays unavailable.
    """
    if not run_dir:
        return None
    root = Path(run_dir).resolve()
    if not root.is_dir():
        return None
    def local(value):
        if not isinstance(value, (str, Path)) or not str(value):
            return None
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path.resolve() if path.is_file() and path.resolve().is_relative_to(root) else None
    def saved(name):
        return local(resolve_run_path(root, name))
    def relative(path):
        return str(path.relative_to(root))
    goal = goal if isinstance(goal, dict) else read_record(saved('structured_goal.json'))
    jobs = {j['job_id']: j for name in ('steric_design_jobs', 'gapmer_design_jobs')
            for j in goal.get(name) or [] if isinstance(j, dict) and j.get('job_id')}
    programs = {p['program_id']: p for p in goal.get('design_programs') or []
                if isinstance(p, dict) and p.get('program_id')}
    sources, inline = {}, []
    def add(path, code=None, stage='', reason='', job=''):
        path = local(path)
        if not path:
            return
        code = code or _bucket(path)
        if not code:
            return
        key = (path, str(job or ''))
        prior = sources.get(key) or {}
        sources[key] = dict(path=path, code=code, stage=stage or prior.get('stage', ''),
                            reason=reason or prior.get('reason', ''), job=job)
    def receipts(value, path, context='', job='', *, candidate_rows=False):
        if isinstance(value, list):
            for child in value:
                receipts(child, path, context, job, candidate_rows=candidate_rows)
        elif isinstance(value, dict):
            job = value.get('job_id') or job
            for key, child in value.items():
                if candidate_rows and key in {'excluded_rows', 'rejected_rows'} and isinstance(child, list):
                    for row in child:
                        if isinstance(row, dict):
                            inline.append((row, dict(path=path, code='recorded_exclusion',
                                stage=value.get('stage') or 'Candidate admission',
                                reason=_reason(value.get('exclusion_reason')), job=job)))
                elif key in {'excluded_path', 'rejected_path'}:
                    add(child, _bucket(str(child)) or context or 'recorded_exclusion',
                        value.get('stage') or '', _reason(value.get('reason')), job)
                elif key.startswith('filtered_') and isinstance(child, dict) and child.get('path'):
                    add(child['path'], key, child.get('stage') or '', _reason(child.get('reason')), job)
                elif key.endswith('_path') and isinstance(child, str) and _bucket(child):
                    add(child, job=job)
                if isinstance(child, (dict, list)) and key not in {'rows', 'records', 'excluded_rows', 'rejected_rows',
                        'all_transcripts_full', 'isoform_transcripts', 'source_annotations', 'sequence_variants'}:
                    receipts(child, path, key, key if key in jobs else job, candidate_rows=candidate_rows)
    goal_path = saved('structured_goal.json')
    receipts(goal, goal_path)
    # Intake contains execution receipts in older layouts. Bounded traversal
    # ignores the large sequence/reference catalogs and candidate row bodies.
    intake_path = saved('goal_intake.json')
    if intake_path:
        receipts(read_record(intake_path), intake_path)
    directories = {root, root/'artifacts/candidates', root/'artifacts/intake'}
    for base in (root, root/'artifacts/design'):
        for folder in ('tiling', 'gapmer_tiling', 'junction_gapmer_tiling'):
            directories.add(base/folder)
        for folder in ('steric_jobs', 'gapmer_jobs', 'v2_tasks', 'v3_tasks'):
            directories.update((base/folder).glob('*'))
    evaluation_names = {'existing_aso_evaluation.json', 'provided_aso_evaluation.json', 'evaluation_evidence.json',
                        'candidate_exclusions.json'}
    for directory in sorted(directories):
        for path in sorted(directory.glob('*.json')):
            if not local(path):
                continue
            code = _bucket(path)
            if code:
                add(path, code)
            elif path.name in evaluation_names:
                document = read_record(path)
                receipts(document, path, candidate_rows=True)
                # A dedicated rejection artifact declares its own row schema.
                if path.name == 'candidate_exclusions.json':
                    for row in document.get('rows') or []:
                        if isinstance(row, dict):
                            inline.append((row, dict(path=path, code='recorded_exclusion',
                                stage='Candidate admission', reason='', job='')))
    assessment_path = saved('candidate_assessments.json')
    for row in read_record(assessment_path).get('rows') or []:
        if not isinstance(row, dict):
            continue
        assessment = row.get('candidate_assessment')
        if not isinstance(assessment, dict):
            continue
        if assessment.get('selection_eligible') is False:
            motifs = assessment.get('motifs') or {}
            reasons = [flag.get('reason') or flag.get('rule_id') for flag in motifs.get('flags') or []
                       if isinstance(flag, dict) and flag.get('action') == 'exclude']
            inline.append((row, dict(path=assessment_path, code='selection_ineligible',
                stage='Recommendation eligibility', job='', reason=_reason(reasons)
                or 'Recorded as ineligible for recommendation; a more specific reason is unavailable.')))
    def scope(row, path, declared=''):
        src = source_row(row)
        job = src.get('job_id') or declared
        if not job and path and path.parent.parent.name in {'steric_jobs', 'gapmer_jobs'}:
            job = path.parent.name
        program = src.get('program_id') or (jobs.get(job) or {}).get('program_id') or ''
        return str(job or ''), str(program)
    def identity(row):
        src = source_row(row)
        # Exact physical identity + binding geometry + reference. No locus or
        # chemistry merges when those dimensions were not recorded.
        fallback = None
        if not sequence(row) or not coordinates(row) or not notation(row):
            fallback = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        return json.dumps([physical_identity(row), coordinates(row), src.get('Strand') or src.get('strand'),
                           src.get('reference_id'), src.get('transcript_id'), src.get('chemistry_group_id'),
                           src.get('genome_build'), src.get('species'), src.get('reference_sha256'), fallback], sort_keys=True)
    def compact(row, spec):
        row = _normalize(row, spec['path'])
        src = source_row(row); code = spec['code']
        job, program = scope(row, spec['path'], spec['job'])
        default = ('Excluded by the recorded ' + code.removeprefix('filtered_').replace('_', ' ') + ' rule.'
                   if code.startswith('filtered_') else 'Excluded by a recorded admission rule.')
        stage, default = BUCKETS.get(code, (spec['stage'] or 'Candidate admission', default))
        reason = _reason(src.get('exclusion_reasons') or src.get('exclusion_reason') or src.get('reason')) or spec['reason'] or default
        if code == 'filtered_no_homopolymer' and not spec['reason']:
            motifs = collection(src.get('Homopolymer_excluded_motifs'))
            if isinstance(motifs, list) and src.get('Homopolymer_sequence_role') == 'synthesized_antisense_sequence':
                hits = [m for m in motifs if isinstance(m, str) and m and m in sequence(row)]
                if hits:
                    reason = 'Excluded homopolymer in synthesized ASO: ' + ', '.join(hits)
        intervals = coordinates(row)
        gc = number(src.get('GC', src.get('gc_fraction')))
        if gc is None and number(src.get('gc_percent')) is not None:
            gc = number(src['gc_percent']) / 100
        if gc is not None and not 0 <= gc <= 1:
            gc = None
        chem = src.get('chemistry_label') or src.get('chemistry_group_label') or src.get('chemistry_platform') or ''
        program_label = (programs.get(program) or {}).get('label') or program
        return dict(aso=sequence(row), location='; '.join(f"{p['chrom']}:{p['start']:,}–{p['end']:,}" for p in intervals),
                    intervals=intervals, strand=src.get('Strand') or src.get('strand') or '', gc=gc,
                    gc_display=f'{gc*100:.0f}%' if gc is not None else '', status='Filtered out',
                    stage=str(src.get('stage') or spec['stage'] or stage), reason=reason,
                    group=' · '.join(v for v in [program_label, job] if v) or 'Recorded candidate pool',
                    job=job, program=program, chemistry=str(chem), notation=notation(row),
                    sources=[relative(spec['path'])] if spec['path'] else [],
                    comparison_sequence=str(src.get('Sequence') or src.get('comparison_sequence') or ''),
                    _identity=identity(row), _code=code)
    candidates = []
    specs_by_path = defaultdict(list)
    for spec in sources.values():
        specs_by_path[spec['path']].append(spec)
    for path, specs in specs_by_path.items():
        # Several jobs may cite the same admission artifact. Read the scientific
        # table once, even when each producing scope needs its own display row.
        for row in read_record(path).get('rows') or []:
            if isinstance(row, dict):
                for spec in specs:
                    if spec['code'] not in BUCKETS and not any(key in row for key in (
                            'Sequence', 'sequence', 'Synthesis', 'antisense_5to3', 'synthesized_antisense_sequence',
                            'molecule_specification', 'candidate_id', 'input_row_id')):
                        # Generic receipts may reject reference intervals or input
                        # annotations. They are not automatically ASO candidates.
                        continue
                    candidates.append(compact(row, spec))
    candidates.extend(compact(row, spec) for row, spec in inline)
    # Consolidate duplicate aggregate/job copies without collapsing distinct
    # producing jobs or attributing an unscoped record to an arbitrary job.
    grouped = defaultdict(list)
    for row in candidates:
        key = (row['_identity'], row['_code'], row['stage'])
        if not row['aso'] or not row['intervals'] or not row['notation']:
            key += (tuple(row['sources']),)
        grouped[key].append(row)
    merged = []
    for rows in grouped.values():
        scoped = {(r['job'], r['program']) for r in rows if r['job'] or r['program']}
        by_scope = {}
        for row in rows:
            keys = [(row['job'], row['program'])]
            if keys == [('', '')] and scoped:
                keys = sorted(scoped)
            for key in keys:
                if key not in by_scope:
                    by_scope[key] = {**row, 'job':key[0], 'program':key[1], 'sources':[]}
                item = by_scope[key]
                if row['job'] or row['program']:
                    item['group'] = row['group']
                item['sources'] = sorted(set(item['sources'] + row['sources']))
                if row['reason'] != item['reason'] and row['reason'] not in item['reason']:
                    item['reason'] += '; ' + row['reason']
        merged.extend(by_scope.values())
    trace_path = saved('generated_filter_trace.json')
    trace = read_record(trace_path)
    drops = [r for r in trace.get('rows') or [] if isinstance(r, dict)
             and r.get('terminal_classification') in TRACE_STAGES]
    if drops:
        # Index raw sequence evidence once. Avoid O(trace rows × raw rows) joins.
        inventory_paths = set()
        for name in ('raw_generated', 'initial_filter'):
            for value in (trace.get('sources') or {}).get(name) or []:
                path = local(value)
                if path:
                    inventory_paths.add(path)
        for directory in directories:
            path = local(directory/'tiles.json')
            if path:
                inventory_paths.add(path)
        index = defaultdict(dict)
        wanted = {str(r.get('sequence') or '').upper() for r in drops}
        for path in sorted(inventory_paths):
            for row in read_record(path).get('rows') or []:
                if not isinstance(row, dict):
                    continue
                src = source_row(row)
                comparison = src.get('Sequence') or src.get('comparison_sequence')
                if comparison and str(comparison).upper() in wanted:
                    key = identity(row)
                    # Raw aggregate copies may lack a job. A single physical
                    # candidate can be resolved only if all explicit scopes agree.
                    # Keep only display inputs; large off-target and prediction
                    # arrays stay in their source file, not the report index.
                    minimal = {k: v for k, v in src.items() if k in {
                        'Synthesis', 'synthesized_antisense_sequence', 'antisense_5to3', 'IDT', 'ordering_notation',
                        'molecule_specification', 'Chrom', 'Start', 'End', 'Strand', 'chrom', 'start', 'end', 'strand',
                        'Genomic_segments', 'genomic_segments', 'coordinate_system', 'genomic_coordinate_system',
                        'binding_intervals_bed0_half_open', 'GC', 'gc_fraction', 'gc_percent', 'job_id', 'program_id',
                        'chemistry_label', 'chemistry_group_label', 'chemistry_platform', 'Sequence', 'comparison_sequence'}}
                    item = index[str(comparison).upper()].setdefault(key, dict(row=minimal, paths=set(), scopes=set()))
                    item['paths'].add(path)
                    job = scope(row, path)
                    if job != ('', ''):
                        item['scopes'].add(job)
        exclusions_by_comparison = defaultdict(lambda: defaultdict(set))
        for row in merged:
            exclusions_by_comparison[row['comparison_sequence'].upper()][row['_identity']].add((row['job'], row['program']))
        for entry in drops:
            comparison = str(entry.get('sequence') or '').upper()
            matches = index.get(comparison) or {}
            covered = exclusions_by_comparison[comparison]
            if matches and all(key in covered and (not item['scopes'] or item['scopes'] <= covered[key])
                               for key, item in matches.items()):
                continue
            resolved = next(iter(matches.values())) if len(matches) == 1 else None
            if resolved and len(resolved['scopes']) > 1:
                resolved = None
            path = sorted(resolved['paths'])[0] if resolved else trace_path
            stage, message = TRACE_STAGES[entry['terminal_classification']]
            spec = dict(path=path, code='trace_removal', stage=stage, reason='',
                        job=next(iter(resolved['scopes']))[0] if resolved and resolved['scopes'] else '')
            row = compact(resolved['row'] if resolved else {}, spec)
            row.update(status='Removal reason unavailable', reason=message + ' No candidate-specific exclusion reason was recorded.',
                       comparison_sequence=comparison, sources=sorted({relative(trace_path), *row['sources']}))
            if not resolved:
                row['reason'] += ' The sequence-level trace does not identify a unique ASO, chemistry and binding context.'
            merged.append(row)
    if not merged:
        return None
    for row in merged:
        row.pop('_identity'); row.pop('_code')
        row['id'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
    merged.sort(key=lambda r: (r['status'], r['stage'],
        tuple((p['chrom'], p['start'], p['end']) for p in r['intervals']), r['group'], r['aso'], r['id']))
    return {'version': 1, 'records': merged}
