"""Small, reference-scoped presentation records over saved run evidence.

Rendering never predicts, selects molecules, repairs chemistry, or reads genome
assets. Each row retains its producing job and response site. Joins use physical
binding geometry, antisense sequence and ordering chemistry together.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def number(value):
    if value is None or isinstance(value, bool) or value == '':
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def collection(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value


def read_record(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def source_row(row):
    return {**(row.get('source_annotations') or {}), **row}


def sequence(row):
    molecule = row.get('molecule_specification') or {}
    row = source_row(row)
    return str(molecule.get('antisense_5to3') or row.get('synthesized_antisense_sequence')
               or row.get('antisense_5to3') or row.get('Synthesis') or '').strip().upper()


def notation(row):
    molecule = row.get('molecule_specification') or {}
    row = source_row(row)
    value = molecule.get('ordering_notation') or row.get('ordering_notation') or row.get('IDT')
    # Historical shortlist identity is (sequence, JSON-encoded chemistry).
    identity = row.get('molecule_identity')
    if not value and isinstance(identity, (list, tuple)) and len(identity) == 2:
        saved = collection(identity[1])
        if isinstance(saved, str):
            value = saved
    return value if isinstance(value, str) else ''


def physical_identity(row):
    molecule = row.get('molecule_specification') or {}
    if molecule.get('physical_molecule_sha256'):
        return str(molecule['physical_molecule_sha256'])
    identity = [sequence(row), notation(row), molecule.get('nucleotide_sugars_5to3'),
                molecule.get('base_modifications_5to3'), molecule.get('linkages_5to3'),
                molecule.get('terminal_modifications'), molecule.get('conjugates')]
    if not notation(row) and not molecule.get('nucleotide_sugars_5to3'):
        source = source_row(row)
        identity.append({key:source.get(key) for key in ('chemistry_platform','chemistry_label',
            'chemistry','job_id','program_id','input_row_id','candidate_id','name')})
    if not identity[0]:
        return None
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def coordinates(row):
    """One-based genomic display coordinates; unknown spaces stay unplotted."""
    source = source_row(row)
    bed = collection(row.get('binding_intervals_bed0_half_open'))
    result = []
    if isinstance(bed, list):
        for item in bed:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                a, b = number(item[1]), number(item[2])
                if a is not None and b is not None and b > a >= 0:
                    result.append({'chrom':str(item[0]), 'start':int(a)+1, 'end':int(b)})
        return result
    segments = collection(source.get('Genomic_segments') or source.get('genomic_segments'))
    convention = source.get('coordinate_system') or source.get('genomic_coordinate_system')
    if isinstance(segments, list) and convention in {'1based_inclusive', 'genomic_1based_inclusive'}:
        for item in segments:
            if isinstance(item, dict) and item.get('chrom') and number(item.get('start')) is not None and number(item.get('end')) is not None:
                if 1 <= float(item['start']) <= float(item['end']):
                    result.append({'chrom':str(item['chrom']), 'start':int(item['start']), 'end':int(item['end'])})
        return result
    chrom = source.get('Chrom') or source.get('chrom')
    a = number(source.get('Start', source.get('start')))
    b = number(source.get('End', source.get('end')))
    if not chrom or a is None or b is None:
        return []
    if convention in {'1based_inclusive', 'genomic_1based_inclusive'} and 1 <= a <= b:
        return [{'chrom':str(chrom),'start':int(a),'end':int(b)}]
    # Native tiler Chrom/Start/End fields have a BED0 contract. Generic input
    # start/end fields require their own declared convention.
    if (all(key in source for key in ('Chrom','Start','End')) or convention in
            {'bed0_half_open','0based_half_open','genomic_0based_half_open'}) and 0 <= a < b:
        return [{'chrom':str(chrom),'start':int(a)+1,'end':int(b)}]
    return []


def join_key(row):
    source = source_row(row)
    return (sequence(row), str(source.get('Strand') or source.get('strand') or source.get('Genomic_strand') or ''),
            tuple((p['chrom'].removeprefix('chr'),p['start'],p['end']) for p in coordinates(row)))


def chemistry(row):
    from aso_molecule import parse_ordering_notation
    molecule = row.get('molecule_specification') or {}
    original = notation(row)
    parsed, issues = {}, []
    if original:
        try:
            parsed = parse_ordering_notation(original)
        except ValueError:
            issues.append('The supplier needs to confirm the recorded ordering notation.')
    else:
        issues.append('Ordering notation is unavailable for this ASO.')
    if parsed and molecule:
        for key, description in [('base_modifications_5to3','base modifications'),
                                 ('linkages_5to3','backbone chemistry')]:
            if molecule.get(key) and parsed.get(key) != molecule[key]:
                issues.append(f'The ordering notation differs from the recorded {description}.')
        if molecule.get('nucleotide_sugars_5to3') and parsed.get('sugars_5to3') != molecule['nucleotide_sugars_5to3']:
            issues.append('The ordering notation differs from the recorded sugar modifications.')
    if parsed and parsed.get('sequence') != sequence(row).replace('U','T'):
        issues.append('The ordering notation differs from the recorded ASO sequence.')
    sugars = molecule.get('nucleotide_sugars_5to3') or parsed.get('sugars_5to3') or []
    links = molecule.get('linkages_5to3') or parsed.get('linkages_5to3') or []
    labels = {'2MOE':'2′-MOE','2OME':'2′-O-methyl','DNA':'DNA','LNA':'LNA','MORPHOLINO':'morpholino'}
    sugar_text = ', '.join(labels.get(s,s) for s in dict.fromkeys(sugars))
    linkage_text = ', '.join(dict.fromkeys(links))
    platform = molecule.get('platform') or source_row(row).get('chemistry_platform')
    architecture = molecule.get('architecture') or source_row(row).get('chemistry_label')
    if platform == 'gapmer':
        title, mechanism = f'{len(sequence(row))}-nt gapmer', 'Designed to recruit RNase H for RNA cleavage.'
    elif platform in {'steric_blocker','pmo'} or architecture in {'steric-hindrance','steric-2ome-ps','morpholino'}:
        title, mechanism = f'{len(sequence(row))}-nt steric blocker', 'Designed to block access to the bound RNA site.'
    else:
        title, mechanism = (f'{len(sequence(row))}-nt ASO' if sequence(row) else 'Chemistry unavailable'), ''
    details = ' · '.join(part for part in [sugar_text + ' sugars' if sugar_text else '',
                            linkage_text + ' backbone' if linkage_text else '',
                            architecture if platform == 'gapmer' else ''] if part)
    return {'label':title,'detail':details,'mechanism':mechanism,'notation':original,'issues':issues}


def site_name(label, sites):
    site = sites.get(label) or {}
    kind = site.get('site') or site.get('site_kind')
    position = site.get('position')
    if not kind and '@' in str(label):
        kind, _, point = str(label).partition('@')
        position = number(point)
    if kind not in {'acceptor','donor'}:
        return str(label or 'Splice site')
    # Names may be supplied only by the same reference that anchors the site.
    exon = site.get('verified_exon_label')
    if exon is not None:
        return f'Exon {exon} {kind}'
    return kind.capitalize() + (f' at {int(position):,}' if number(position) is not None else '')


def safeguard_summary(row, sites):
    assessment = row.get('candidate_assessment') or row.get('assessment') or {}
    reports = row.get('safeguard_report') or (assessment.get('splice_safeguards') or {}).get('reports') or []
    details, failures, unknown, passed, observed = [], [], 0, 0, 0
    for guard in reports:
        name = site_name(guard.get('label'), sites)
        values = [number(guard.get(key)) for key in ('delta_deletion','delta_replacement')]
        if all(v is None for v in values):
            values = [number(v) for v in guard.get('values') or []]
        values = [v for v in values if v is not None]
        if guard.get('policy') == 'observe':
            observed += 1
            text = name + ': changes recorded for context.'
        elif guard.get('passes') is False or guard.get('status') in {'fail','failed'}:
            breaches = []
            for value in values:
                for field, magnitude in [('max_abs_delta',abs(value)),('max_increase',value),('max_decrease',-value)]:
                    limit = number(guard.get(field))
                    if limit is not None and magnitude > limit + 1e-9:
                        breaches.append((magnitude-limit,value,limit))
            if breaches:
                _, value, limit = max(breaches)
                text = f'{name} {"rose" if value > 0 else "fell"} {abs(value):.3g}; limit {limit:.3g}.'
            else:
                text = name + ': outside its recorded limit.'
            failures.append(text)
        elif guard.get('passes') is True or guard.get('status') in {'pass','passed'}:
            passed += 1
            text = name + ': within the specified limits.'
        else:
            unknown += 1
            text = name + ': prediction unavailable.'
        review = guard.get('review') or {}
        rationale = review.get('rationale') or (review.get('decision') or {}).get('rationale') or guard.get('note')
        details.append({'site':name,'text':text,'reason':rationale or '',
                        'review_incomplete':review.get('status') == 'unavailable'})
    if failures:
        summary = failures[0] + (f' {len(failures)-1} other checks exceeded a limit.' if len(failures)>1 else '')
        status = 'review'
    elif unknown:
        summary, status = f'{unknown} splice-site prediction{"s" if unknown!=1 else ""} unavailable.', 'incomplete'
    elif passed:
        summary, status = f'{passed} splice check{"s" if passed!=1 else ""} within limits.', 'passed'
    elif observed:
        summary, status = 'Splice changes recorded for context.', 'observed'
    else:
        summary, status = 'Splice checks unavailable.' if row.get('route_label') else 'No splice checks requested.', 'unavailable'
    if any(d['review_incomplete'] for d in details):
        summary += ' Model review incomplete; existing checks retained.'
    return {'status':status,'summary':summary,'details':details}


def _effects(row, label, direction):
    source = source_row(row)
    values = None
    for entry in row.get('selected_effect_evidence') or []:
        if entry.get('label') == label:
            values = (entry.get('prediction') or {}).get('values')
    if values is None:
        for entry in collection(source.get('splice_site_deltas')) or []:
            if not isinstance(entry, dict) or label not in {entry.get('canonical_label'), entry.get('display_label'), entry.get('label')}:
                continue
            kind = entry.get('site_kind')
            if kind in {'acceptor', 'donor'}:
                values = [entry.get(f'delta_{kind}_deletion'), entry.get(f'delta_{kind}_replacement')]
                break
    names = source.get('delta_names') or []
    if isinstance(names,str):
        names = names.split(',')
    if values is None and label in names:
        index = names.index(label)
        values = []
        for key in ('delta_deletion','delta_replacement'):
            column = source.get(key)
            if not isinstance(column,(list,tuple)):
                column = str(column if column is not None else '').split(',')
            values.append(column[index] if index<len(column) else None)
    if values is None and source.get('selected_label') == label:
        values = [source.get('selected_delta_deletion'),source.get('selected_delta_replacement')]
    values = [number(v) for v in (values or [])[:2]]
    mean = sum(values)/2 if len(values)==2 and all(v is not None for v in values) else None
    effect = mean * (1 if direction=='increase' else -1) if mean is not None and direction in {'increase','decrease'} else None
    return effect, values


def _record(row, context, sites, order, panel_rank=None):
    source = source_row(row)
    aso = sequence(row)
    if not aso:
        return None
    chemistry_record = chemistry(row)
    identity = physical_identity(row)
    intervals = coordinates(row)
    effect, values = _effects(row, context.get('site'), context.get('direction'))
    assessment = row.get('candidate_assessment') or row.get('assessment') or {}
    screen = assessment.get('off_target') or {}
    counts = screen.get('counts_by_mismatch')
    if not isinstance(counts,list):
        counts = [source.get(k) for k in ('Off_target_exact_count','Off_target_1mm_count','Off_target_2mm_count')]
    counts = [number(v) for v in counts[:3]]
    counts += [None]*(3-len(counts))
    budget = number(screen.get('mismatch_budget',source.get('Off_target_mismatch_budget')))
    if budget is not None:
        counts = [value if index<=budget else None for index,value in enumerate(counts)]
    status = screen.get('status') or source.get('off_target_count_status') or source.get('off_target_search_status') or 'unavailable'
    complete = status in {'complete','completed','complete_within_declared_scope'} and not source.get('off_target_mapping_capped')
    gc = number(source.get('gc_fraction', source.get('GC')))
    guards = safeguard_summary(row, sites)
    if context.get('site') and not guards['details']:
        guards['summary'] = 'Splice checks unavailable for this ASO.'
    rank = number(row.get('selection_rank') or row.get('native_rank'))
    origin = {'program':row.get('program_id') or context.get('program'), 'job':row.get('job_id') or context.get('job'),
              'route':context.get('site') or row.get('route_label'), 'source':Path(str(row.get('source_path') or '')).name}
    row_id = hashlib.sha256(json.dumps([identity,context['id'],intervals,order],sort_keys=True).encode()).hexdigest()[:20]
    exclusion = source.get('exclusion_reasons') or source.get('filter_failure_reasons') or []
    if isinstance(exclusion,str):
        exclusion = [exclusion]
    if assessment and assessment.get('selection_eligible') is False:
        exclusion = [*exclusion,'Excluded by the recorded sequence-motif rules.']
    return {'id':row_id,'molecule':identity,'context':context['id'],'name':str(source.get('candidate_id') or source.get('Name') or source.get('name') or f'ASO {order}'),
        'aso':aso,'target':source.get('target_binding_sequence') or source.get('Sequence') or '',
        'chemistry':chemistry_record,'intervals':intervals,'strand':source.get('strand') or source.get('Strand') or source.get('Genomic_strand') or '',
        'location':'; '.join(f"{p['chrom']}:{p['start']:,}–{p['end']:,}" for p in intervals) or 'Location unavailable',
        'start':intervals[0]['start'] if intervals else None,'effect':effect,'predictions':values,
        'gc':gc,'off_targets':counts,'off_target_status':status,'off_target_complete':complete,
        'off_target_budget':budget,
        'off_target_unit':screen.get('unit') or source.get('off_target_exact_count_unit') or 'recorded matches',
        'guard':guards,'native_rank':rank,'order':order,'panel_rank':panel_rank,'origin':origin,
        'selection_note':'; '.join(str(v) for v in exclusion) if exclusion else ('Selected for the panel.' if panel_rank else
            'Assessed in the design pool.' if assessment else 'Scored in the search pool; final selection checks are available for retained candidates.'),
        'motifs':[f.get('rule_id') for f in (assessment.get('motifs') or {}).get('flags') or [] if isinstance(f,dict)]}


def _objective(program, gene):
    text = program.get('objective')
    if isinstance(text,str) and 0<len(text.strip())<=150:
        return text.strip()
    event = (program.get('splice_event') or {}).get('event_type')
    action = {'intron_retention':'Increase retention of the selected intron', 'intron_excision':'Increase removal of the selected intron'}.get(event)
    if not action:
        exon = 'exon ' + str(program['exon_label']) if program.get('exon_label') is not None else 'the selected exon'
        action = {'inclusion':f'Increase inclusion of {exon}', 'skipping':f'Increase skipping of {exon}'}.get(program.get('desired_effect'))
    if not action:
        action = 'Reduce RNA abundance' if program.get('modality')=='gapmer' else 'Block access to the selected RNA region'
    return f'{action} for {gene}.' if gene else action + '.'


def _root_file(root, name):
    from .run_layout import resolve_run_path
    path = resolve_run_path(root, name)
    return path if path.is_file() and path.resolve().is_relative_to(root) else None


def project_run(run_dir, *, goal=None, kind='design'):
    """Project every recorded producer; preserve all source data on disk."""
    if not run_dir:
        return None
    report_kind = kind
    root = Path(run_dir).resolve()
    if not root.is_dir():
        return None
    goal = goal or read_record(_root_file(root,'structured_goal.json'))
    pool = read_record(_root_file(root,'candidate_assessments.json')).get('rows') or []
    from .candidate_export_evidence import with_native_ranks
    pool = with_native_ranks(pool)
    selected = read_record(_root_file(root,'recommended_shortlist.json'))
    panel = selected.get('records') or read_record(_root_file(root,'generated_aso_set.json')).get('rows') or []
    programs = {p['program_id']:p for p in goal.get('design_programs') or [] if isinstance(p,dict) and p.get('program_id')}
    jobs = {j['job_id']:j for j in [*(goal.get('steric_design_jobs') or []),*(goal.get('gapmer_design_jobs') or [])]
            if isinstance(j,dict) and j.get('job_id')}
    catalog = goal.get('isoform_transcripts') or {}
    sites = {f"{s.get('site')}@{s.get('position')}":s for s in goal.get('splice_site_plan') or [] if isinstance(s,dict)}
    by_job = defaultdict(list)
    for row in pool:
        by_job[str(row.get('job_id') or row.get('program_id') or 'saved')].append(row)
    paths = []
    for folder in ('steric_jobs','gapmer_jobs'):
        for name in ('scored_tiles.json','gapmer_scored_uniqueness.json'):
            paths.extend((root/'artifacts/design'/folder).glob('*/'+name))
    for name in ('scored_tiles.json','gapmer_scored_uniqueness.json'):
        path = _root_file(root,name)
        if path and not paths:
            paths.append(path)
    raw_by_job = defaultdict(list)
    for path in sorted(set(paths)):
        if not path.resolve().is_relative_to(root):
            continue
        job_id = path.parent.name if path.parent.name in jobs or path.parent.name in by_job else None
        if job_id is None and len(by_job)==1:
            job_id = next(iter(by_job))
        if job_id is None:
            job_id = path.parent.name
        rows = read_record(path).get('rows') or []
        for row in rows:
            if isinstance(row,dict):
                raw_by_job[job_id].append({**row,'job_id':job_id,'source_path':str(path)})
    if not pool and not raw_by_job:
        return None
    panel_by_sequence = defaultdict(list)
    for rank,row in enumerate(panel,1):
        panel_by_sequence[sequence(row)].append((row.get('recommendation_order') or rank,row))
    contexts, records = [], []
    for job_id in dict.fromkeys([*by_job,*raw_by_job]):
        assessed = by_job[job_id]
        job = jobs.get(job_id) or {}
        program_id = job.get('program_id') or next((r.get('program_id') for r in assessed if r.get('program_id')),None)
        program = programs.get(program_id) or (next(iter(programs.values())) if len(programs)==1 else {})
        routes = {}
        for row in assessed:
            source = source_row(row)
            label = row.get('route_label') or source.get('selected_label') or ''
            direction = source.get('selected_direction') or source.get('primary_direction')
            routes[label] = direction or program.get('desired_delta')
        if not routes:
            routes[''] = None
        lookup = defaultdict(list)
        for row in assessed:
            lookup[join_key(row)].append(row)
        for route,direction in routes.items():
            context_id = hashlib.sha256(json.dumps([job_id,route]).encode()).hexdigest()[:16]
            label = str(program.get('label') or next((r.get('job_label') for r in assessed if r.get('job_label')),None) or 'ASO candidates')
            if route:
                label += ' · ' + site_name(route,sites)
            context = {'id':context_id,'label':label,'program':program_id,'job':job_id,'site':route,'direction':direction,
                'objective':_objective(program,goal.get('gene_symbol')),'metric':'Mean splice prediction' if route else 'Recorded ranking',
                'metric_note':'Mean of the deletion and replacement predictions, oriented so a larger value supports the requested direction. Scores use the model’s native units.' if route else
                    'Candidates follow the saved ordering within this design group. Each ASO’s details retain its selection evidence.',
                'exons':[],'sites':[],'transcript':''}
            transcript = program.get('transcript_id') or goal.get('transcript_id')
            matches = [(name,entry) for name,entry in catalog.items() if transcript and transcript in
                       {name,entry.get('transcript_id'),(entry.get('metadata') or {}).get('versioned_id')}]
            if not matches and len(catalog)==1:
                matches = list(catalog.items())
            if len(matches)==1:
                name,entry = matches[0]
                context.update(transcript=(entry.get('metadata') or {}).get('declared_reference') or name,reference_label=name,
                               reference_chrom=entry.get('chrom') or (entry.get('metadata') or {}).get('seq_region_name'),
                               reference_strand=entry.get('strand'),exons=[{k:e[k] for k in ('index','start','end') if k in e}
                                  for e in entry.get('exons_tx') or []])
            # Exon names come from exact reference-bound junction records.
            # Planner prose and proximity alone never establish exon numbering.
            context_sites = dict(sites)
            for candidate in [*raw_by_job[job_id], *assessed]:
                for s in collection(source_row(candidate).get('splice_site_deltas')) or []:
                    if not isinstance(s,dict):
                        continue
                    key = s.get('canonical_label') or s.get('display_label') or s.get('label')
                    kind = s.get('site_kind'); point = s.get(kind) or {}
                    annotated = s.get('reference_binding_status') == 'resolved_catalog_identity'
                    context_sites[key] = {**context_sites.get(key,{}), **s, **point,
                        'verified_exon_label':s.get('exon_index') if annotated else None}
            if route:
                context['label'] = label.rsplit(' · ',1)[0] + ' · ' + site_name(route,context_sites)
                action = {'increase':'Increase use of', 'decrease':'Decrease use of'}.get(direction)
                if action:
                    context['objective'] = f'{action} the {site_name(route,context_sites).lower()} for {goal.get("gene_symbol") or "the selected target"}.'
            combined, seen = [], set()
            candidates = raw_by_job[job_id] or assessed
            for raw in candidates:
                key = join_key(raw)
                options = [r for r in lookup.get(key,[]) if (r.get('route_label') or source_row(r).get('selected_label') or '') == route
                           and (not notation(raw) or notation(r)==notation(raw))]
                if len(options)==1:
                    item = {**raw,**options[0]}
                else:
                    item = dict(raw)
                    common = lookup.get(key,[])
                    if len({physical_identity(r) for r in common})==1 and common:
                        item['molecule_specification'] = common[0].get('molecule_specification') or {}
                identity = (key,notation(item))
                if identity not in seen:
                    combined.append(item); seen.add(identity)
            for row in assessed:
                if (row.get('route_label') or source_row(row).get('selected_label') or '') != route:
                    continue
                identity = (join_key(row),notation(row))
                if identity not in seen:
                    combined.append(row); seen.add(identity)
            for index,row in enumerate(combined,1):
                rank = None
                for panel_rank,chosen in panel_by_sequence.get(sequence(row),[]):
                    if physical_identity(row)==physical_identity(chosen) or (notation(row) and notation(row)==notation(chosen)):
                        rank = panel_rank; break
                record = _record(row,context,context_sites,index,rank)
                if record:
                    records.append(record)
                    source = source_row(row)
                    for s in collection(source.get('splice_site_deltas')) or []:
                        if not isinstance(s,dict):
                            continue
                        kind = s.get('site_kind'); point = s.get(kind) or {}
                        site_label = s.get('canonical_label') or s.get('label')
                        if point.get('position') and point.get('chrom'):
                            context['sites'].append({'label':site_name(site_label,context_sites),
                                'position':point['position'],'chrom':point['chrom'],'target':site_label==route})
            context['sites'] = list({json.dumps(s,sort_keys=True):s for s in context['sites']}.values())
            duplicates = {s['label'] for s in context['sites'] if sum(t['label']==s['label'] for t in context['sites'])>1}
            for site in context['sites']:
                if site['label'] in duplicates:
                    site['label'] += f" · {int(site['position']):,}"
            target = next((site for site in context['sites'] if site['target']),None)
            if target:
                context['label'] = label.rsplit(' · ',1)[0] + ' · ' + target['label']
            contexts.append(context)
    if not records:
        return None
    all_chemistries = {(r['chemistry']['label'],r['chemistry']['detail'],r['chemistry']['mechanism']) for r in records}
    molecules = {r['molecule'] for r in records}
    recommended = {r['molecule'] for r in records if r['panel_rank'] is not None}
    facts = [{'label':'Gene','value':goal.get('gene_symbol') or 'Recorded targets',
              'note':goal.get('transcript_id') or (contexts[0].get('transcript') if len({c.get('transcript') for c in contexts})==1 else f'{len(contexts)} design groups')},
             {'label':'ASOs assessed','value':f'{len(molecules):,}','note':f'{len(recommended):,} in the selected panel'},
             {'label':'Target','value':contexts[0]['label'].split(' · ')[-1] if len(contexts)==1 else f'{len(contexts)} target groups',
              'note':'Choose a target below' if len(contexts)>1 else contexts[0].get('transcript','')}]
    variant = goal.get('patient_variant') or {}
    if variant:
        zygosity = {'homozygous':'both copies', 'heterozygous':'one copy', '1/1':'both copies',
                    '1|1':'both copies', '0/1':'one copy', '1/0':'one copy', '0|1':'one copy', '1|0':'one copy'}.get(
                    str(variant.get('zygosity') or variant.get('genotype') or ''))
        # Read only the small declared variant file, never patient sequence.
        vcf = (goal.get('patient_reference') or {}).get('vcf_path')
        if not zygosity and vcf and Path(vcf).is_file() and Path(vcf).resolve().is_relative_to(root) and Path(vcf).stat().st_size < 128000:
            for line in Path(vcf).read_text().splitlines():
                fields = line.split('\t')
                if (line.startswith('#') or len(fields)<10 or number(fields[1])!=number(variant.get('pos'))
                        or fields[0].removeprefix('chr')!=str(variant.get('chr')).removeprefix('chr')
                        or variant.get('ref') and fields[3]!=variant['ref']):
                    continue
                sample = dict(zip(fields[8].split(':'), fields[9].split(':')))
                alternatives = fields[4].split(',')
                if variant.get('alt') in alternatives:
                    allele = str(alternatives.index(variant['alt'])+1)
                    genotype = str(sample.get('GT') or '').replace('|','/').split('/')
                    if '.' not in genotype:
                        copies = genotype.count(allele)
                        zygosity = 'both copies' if copies==2 and len(genotype)==2 else 'one copy' if copies==1 else None
        position = f"{variant.get('chr')}:{int(variant['pos']):,}" if variant.get('chr') and number(variant.get('pos')) is not None else ''
        facts.insert(1,{'label':'Patient variant','value':variant.get('notation') or position or 'Supplied variant',
                        'note':' · '.join(v for v in [zygosity,position] if v)})
    if len(all_chemistries)==1:
        chem,details,mechanism = next(iter(all_chemistries))
        facts.append({'label':'Chemistry','value':chem,'note':details + (' · '+mechanism if mechanism else '')})
    else:
        facts.append({'label':'Chemistry','value':f'{len(all_chemistries)} specifications','note':'The exact chemistry stays attached to each ASO'})
    result = {'version':1,'kind':report_kind,'contexts':contexts,'records':records,'facts':facts,
              'objective':contexts[0]['objective'],
              'panel_label':'Recommended panel' if selected.get('records') else 'Selected panel',
              'scope':'Saved scored and assessed candidates; each binding record retains its design group.'}
    result['id'] = hashlib.sha256(json.dumps(result,sort_keys=True,default=str).encode()).hexdigest()[:24]
    return result


def project_records(rows, *, title='Supplied ASOs', objective='Review the supplied ASOs and their recorded evidence.', kind='evaluation'):
    """Adapt attested molecule records; supplied sequences retain input order."""
    context = {'id':'supplied', 'label':title, 'objective':objective, 'metric':'Recorded evidence',
               'metric_note':'ASOs follow their saved input order. Open a row for its chemistry and checks.',
               'exons':[], 'sites':[], 'transcript':''}
    records = []
    for order, original in enumerate(rows,1):
        row = dict(original)
        if not sequence(row):
            if row.get('declared_sequence_role') != 'antisense_oligo_sequence':
                continue
            row['antisense_5to3'] = row.get('sequence')
        if row.get('genomic_segments'):
            # Evaluation mapping_hits have an explicit genomic 1-based contract.
            row['coordinate_system'] = 'genomic_1based_inclusive'
        record = _record(row,context,{},order)
        if record:
            record['selection_note'] = 'Supplied for evaluation.' if kind=='evaluation' else 'Included in the supplied molecule set.'
            record['guard']['summary'] = 'Open the analysis below for the complete evaluation.' if kind=='evaluation' else record['guard']['summary']
            records.append(record)
    if not records:
        return None
    result = {'version':1,'kind':kind,'contexts':[context], 'records':records, 'objective':objective,
              'panel_label':'Selected panel','scope':'Supplied molecule records',
              'facts':[{'label':'ASOs supplied','value':str(len(records)),'note':title}]}
    specifications = {(r['chemistry']['label'],r['chemistry']['detail']) for r in records}
    if len(specifications)==1:
        label, detail = next(iter(specifications))
        if detail:
            result['facts'].append({'label':'Chemistry','value':label,'note':detail})
    result['id'] = hashlib.sha256(json.dumps(result,sort_keys=True,default=str).encode()).hexdigest()[:24]
    return result


def project_analysis(report_dir):
    """Discover native evaluation artifacts in the producing report directory.

    Free-form sandbox tables retain their own schemas and the shared reader
    layout. They are never coerced into an ASO design or ordering contract.
    """
    root = Path(report_dir).resolve()
    paths = [root/'provided_aso_evaluation.json']
    if not paths[0].is_file():
        paths = [root/'evaluation_evidence.json']
    if not paths[0].is_file():
        paths = sorted(root.glob('artifacts/design/v3_tasks/*/provided_aso_evaluation.json'))
    present = [p for p in paths if p.is_file() and p.resolve().is_relative_to(root)]
    if len(present)!=1:
        return None
    source = read_record(present[0])
    summary = source.get('response_evaluation_summary') or source.get('evaluation_summary') or source
    context = summary.get('context_summary') or {}
    gene = context.get('gene_symbol')
    result = project_records(summary.get('provided_asos') or [], title=f'{gene} evaluation' if gene else 'Supplied ASOs',
                             objective=f'Review the supplied ASOs for {gene}.' if gene else 'Review the supplied ASOs and their recorded evidence.')
    if result and gene:
        result['facts'].insert(0,{'label':'Gene','value':gene,'note':context.get('transcript_id') or ''})
    return result
