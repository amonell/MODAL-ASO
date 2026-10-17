"""Separate physical binding-region membership from predicted splice response."""
from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json

from .parameter_constraints import evaluate_row_against_constraints
from .region_overlap import candidate_matches_regions
from .reporting_contracts import generated_sequence_fields
from .candidate_presentation import molecule_key
from .candidate_minimums import genomic_regions_for_constraint
from .region_projection import genomic_sources
from .region_overlap import candidate_matches_region_groups


def recorded_screening_summary(goal, final):
    """Describe measured screens without confusing missing values with zeros."""
    paths = set()
    for source in (goal, final):
        for key in ('tiling_results','gapmer_tiling_results'):
            result = source.get(key) or {}
            if result.get('scored_tiles_path'):
                paths.add(str(result['scored_tiles_path']))
    summaries = []
    for raw in sorted(paths):
        try:
            content=Path(raw).read_bytes(); rows=json.loads(content).get('rows') or []
        except (OSError,ValueError,TypeError) as exc:
            summaries.append({'path':raw,'status':'unavailable','reason':str(exc)}); continue
        fields={}
        for name, aliases in {'exact_off_target_count':('Off_target_exact_count','off_target_exact_count'),
                'one_mismatch_off_target_count':('Off_target_1mm_count','off_target_1mm_count'),
                'alternate_exact_repeat_count':('repeat_hit_count_exact',)}.items():
            values=[]
            for row in rows:
                value=next((row[k] for k in aliases if row.get(k) not in (None,'')),None)
                try: values.append(int(value))
                except (TypeError,ValueError): pass
            fields[name]={'measured_rows':len(values),'unknown_rows':len(rows)-len(values),
                'rows_with_hits':sum(v>0 for v in values),'minimum':min(values) if values else None,
                'maximum':max(values) if values else None}
        summaries.append({'path':raw,'sha256':hashlib.sha256(content).hexdigest(),'row_count':len(rows),
            'measurements':fields,'scope':'This scored source table, not automatically the final selected subset. Off-target hits, distinct genes and alternate repeats have different definitions; zero repeats does not mean zero genomic off-target hits.'})
    return summaries


def constraint_groups(goal):
    materialized = goal.get('materialized_region_constraints') or {}
    return [(None,c) for c in materialized.get('global') or []] + [
        (program,c) for program,entries in (materialized.get('by_program') or {}).items() for c in entries]


def memberships(row, goal, intervals_for_row):
    intervals = intervals_for_row(row, output_space='genomic')
    result = []
    for program, constraint in constraint_groups(goal):
        if constraint.get('operator') == 'exclude':
            continue
        regions = genomic_regions_for_constraint(goal, program, constraint)
        rule = constraint.get('aso_overlap_requirement') or 'any_overlap'
        sources = genomic_sources(goal, constraint) or [
            {"source_row": region, "coordinate_space": "genomic", "genomic_rows": [region]} for region in regions]
        for source in sources:
            region = source["source_row"]
            if not candidate_matches_region_groups(intervals, [source["genomic_rows"]], rule, int(constraint.get('minimum_overlap_bp') or 1)):
                continue
            # Report every source interval, including nested/overlapping labels.
            # Merge intersection blocks before counting to avoid duplicate mapping
            # records inflating the overlap length.
            blocks = sorted({(max(int(a), int(region[1])), min(int(b), int(region[2])))
                             for chrom, a, b in intervals for region in source["genomic_rows"]
                             if str(chrom).removeprefix('chr') == str(region[0]).removeprefix('chr')
                             and max(int(a), int(region[1])) < min(int(b), int(region[2]))})
            merged = []
            for start, end in blocks:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            result.append({'program_id':program, 'constraint_id':constraint.get('constraint_id'),
                           'label':region[3] if len(region) > 3 and region[3] else constraint.get('label'),
                           'constraint_label':constraint.get('label'),
                           'source_interval_bed0_half_open':list(region),
                           'source_coordinate_space':source['coordinate_space'],
                           'genomic_segments_bed0_half_open':source['genomic_rows'],
                           'overlap_bp':sum(end-start for start,end in merged),
                           'aso_overlap_requirement':rule})
    return result


def physical_splice_element_overlaps(intervals, transcript_catalog):
    """Name only annotated splice boundaries physically crossed by a target.

    BED0 target intervals and one-based inclusive exon coordinates are kept
    separate. A response-site prediction is never used as a binding location.
    """
    if not isinstance(transcript_catalog, dict):
        return []
    elements, seen = [], set()
    for label, transcript in transcript_catalog.items():
        if not isinstance(transcript, dict):
            continue
        strand = str(transcript.get("strand") or "")
        exons = transcript.get("exons_tx")
        if strand not in {"+", "-"} or not isinstance(exons, list) or len(exons) < 2:
            continue
        transcript_id = str(transcript.get("transcript_id") or label)
        metadata = transcript.get("metadata") if isinstance(transcript.get("metadata"), dict) else {}
        versioned_id = str(metadata.get("versioned_id") or "")
        if versioned_id.startswith(transcript_id + "."):
            transcript_id = versioned_id
        for position, exon in enumerate(exons, 1):
            if not isinstance(exon, dict):
                continue
            try:
                exon_start, exon_end = int(exon["start"]), int(exon["end"])
                ordinal = int(exon.get("index") or position)
            except (KeyError, TypeError, ValueError):
                continue
            chrom = str(exon.get("seq_region_name") or transcript.get("chrom") or "")
            if not chrom or exon_start > exon_end:
                continue
            boundaries = []
            for kind, neighbor_index, intron_ordinal in (
                ("acceptor", position - 2, ordinal - 1),
                ("donor", position, ordinal),
            ):
                if not 0 <= neighbor_index < len(exons):
                    continue
                neighbor = exons[neighbor_index]
                if not isinstance(neighbor, dict):
                    continue
                neighbor_chrom = str(neighbor.get("seq_region_name") or transcript.get("chrom") or "")
                if neighbor_chrom.removeprefix("chr").upper() != chrom.removeprefix("chr").upper():
                    continue
                try:
                    neighbor_start, neighbor_end = int(neighbor["start"]), int(neighbor["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if kind == "acceptor" and strand == "+":
                    boundary, intron_start, intron_end = exon_start, neighbor_end + 1, exon_start - 1
                elif kind == "acceptor":
                    boundary, intron_start, intron_end = exon_end, exon_end + 1, neighbor_start - 1
                elif strand == "+":
                    boundary, intron_start, intron_end = exon_end, exon_end + 1, neighbor_start - 1
                else:
                    boundary, intron_start, intron_end = exon_start, neighbor_end + 1, exon_start - 1
                if intron_start <= intron_end:
                    boundaries.append((kind, boundary, intron_start, intron_end, intron_ordinal))
            for interval in intervals or []:
                if not isinstance(interval, (list, tuple)) or len(interval) < 3:
                    continue
                try:
                    target_start, target_end = int(interval[1]) + 1, int(interval[2])
                except (TypeError, ValueError):
                    continue
                if (str(interval[0]).removeprefix("chr").upper() != chrom.removeprefix("chr").upper()
                        or target_start > target_end):
                    continue
                for kind, boundary, intron_start, intron_end, intron_ordinal in boundaries:
                    intronic_bp = max(0, min(target_end, intron_end) - max(target_start, intron_start) + 1)
                    exonic_bp = max(0, min(target_end, exon_end) - max(target_start, exon_start) + 1)
                    if not intronic_bp or not exonic_bp:
                        continue
                    key = (transcript_id, chrom, strand, ordinal, kind, boundary,
                           target_start, target_end)
                    if key in seen:
                        continue
                    seen.add(key)
                    elements.append({
                        "transcript_id": transcript_id, "chrom": chrom, "strand": strand,
                        "exon_ordinal": ordinal, "element": kind,
                        "adjacent_intron_ordinal": intron_ordinal,
                        "boundary_1based": boundary,
                        "target_interval_1based_inclusive": [target_start, target_end],
                        "exonic_overlap_bp": exonic_bp, "intronic_overlap_bp": intronic_bp,
                        "evidence_scope": "annotated_genomic_geometry_only",
                    })
    return elements


def build_binding_region_report(goal, final, run_dir, intervals_for_row, constraints):
    groups = constraint_groups(goal)
    if not groups or not run_dir:
        return {}
    sources = []
    jobs = {j.get('job_id'):j for j in final.get('steric_design_jobs') or goal.get('steric_design_jobs') or []}
    for job_id, result in (final.get('steric_tiling_results_by_job') or goal.get('steric_tiling_results_by_job') or {}).items():
        path = result.get('scored_tiles_path') or result.get('uniqueness_scored_path')
        if path:
            sources.append((jobs.get(job_id,{}).get('program_id'), path))
    aggregate = final.get('tiling_results') or goal.get('tiling_results') or {}
    path = aggregate.get('scored_tiles_path') or aggregate.get('uniqueness_scored_path')
    if not sources and path:
        sources.append((None,path))
    loaded, source_evidence, failures = [], [], []
    for program, raw in sources:
        path = Path(raw)
        try:
            content = path.read_bytes(); payload = json.loads(content)
            rows = payload.get('rows') or []
            loaded.extend((program, row, str(path)) for row in rows if isinstance(row,dict))
            source_evidence.append({'path':str(path),'sha256':hashlib.sha256(content).hexdigest(),'row_count':len(rows),'program_id':program})
        except (OSError,ValueError,TypeError) as exc:
            failures.append({'path':str(path),'error':str(exc)})
    if not sources:
        return {}  # Gapmer/report-only work keeps its existing candidate views.
    rna_filter = aggregate.get('target_rna_evidence_filtering') or {}
    excluded_rows = []
    if rna_filter.get('excluded_path'):
        excluded_rows = json.loads(Path(rna_filter['excluded_path']).read_text()).get('rows') or []
    sets = []
    for program, constraint in groups:
        if constraint.get('operator') == 'exclude':
            continue
        regions = genomic_regions_for_constraint(goal, program, constraint)
        rule = constraint.get('aso_overlap_requirement') or 'any_overlap'
        qualified, seen = [], set()
        for row_program, row, source in loaded:
            if program and row_program and program != row_program:
                continue
            intervals = intervals_for_row(row, output_space='genomic')
            if not candidate_matches_region_groups(intervals,
                    [item['genomic_rows'] for item in genomic_sources(goal, constraint)] or [[region] for region in regions],
                    rule, int(constraint.get('minimum_overlap_bp') or 1)):
                continue
            keep, _ = evaluate_row_against_constraints(row, constraints)
            if not keep:
                continue
            chemistry = str(row.get('IDT') or row.get('chemistry_label') or '')
            key = (tuple(intervals),str(row.get('Synthesis') or row.get('Sequence')),chemistry)
            if key in seen:
                continue
            seen.add(key)
            qualified.append({**deepcopy(row), **generated_sequence_fields(row),
                'binding_intervals_bed0_half_open':intervals,'requested_binding_region':constraint.get('label'),
                'binding_overlap_requirement':rule,'source_scored_table':source})
        def rank(row):
            try: return float(row.get('selection_rank'))
            except (TypeError,ValueError):
                try: return -float(row.get('Score'))
                except (TypeError,ValueError): return float('inf')
        qualified.sort(key=rank)
        outside_rna = [row for row in excluded_rows if candidate_matches_regions(
            intervals_for_row(row, output_space='genomic'), regions, rule,
            int(constraint.get('minimum_overlap_bp') or 1))]
        sets.append({'program_id':program,'constraint_id':constraint.get('constraint_id'),'label':constraint.get('label'),
            'intervals_bed0_half_open':regions,'aso_overlap_requirement':rule,'candidate_count':len(qualified),
            'minimum_overlap_bp':int(constraint.get('minimum_overlap_bp') or 1),
            'binding_record_count':len(qualified),
            'distinct_molecule_count':len({molecule_key(row) for row in qualified}),
            'status':'candidates_available' if qualified else 'no_supported_target_rna_candidates' if outside_rna else 'no_qualifying_candidates_in_recorded_scored_pool',
            'excluded_outside_target_rna_count':len(outside_rna),
            'target_rna_exclusion_evidence':rna_filter.get('excluded_path') if outside_rna else None,
            'target_rna_scope':'Converted coordinates alone do not establish an RNA target. Excluded candidates fall outside the recorded target transcript loci or have the wrong target strand; this is bounded annotation evidence, not proof that no transcript could ever exist there.',
            'materialization_summary':constraint.get('materialization_summary'), 'rows':qualified})
    # Collapse only equivalent report views: identical physical regions, rules
    # and binding populations. Preserve every original program/constraint label.
    views = {}
    for group in sets:
        region_geometry = sorted((str(row[0]), int(row[1]), int(row[2]), str(row[5]) if len(row)>5 else '.')
                                 for row in group['intervals_bed0_half_open'])
        population = sorted((molecule_key(row), tuple(row['binding_intervals_bed0_half_open'])) for row in group['rows'])
        signature = json.dumps([region_geometry, group['aso_overlap_requirement'], group['minimum_overlap_bp'], population], sort_keys=True)
        identity = {key:group.get(key) for key in ('program_id','constraint_id','label')}
        if signature in views:
            views[signature]['source_groups'].append(identity)
        else:
            group['source_groups'] = [identity]
            views[signature] = group
    sets = list(views.values())
    result = {'sets':sets,'source_tables':source_evidence,'source_errors':failures,
        'interpretation':'Each set satisfies its own physical binding-region rule and the recorded parameter filters. A molecule may belong to multiple sets. Splice-response routes describe the junction being evaluated, not necessarily where the ASO binds. These site-specific candidates are computational design hypotheses, not experimentally validated activity or an assertion that every candidate passes a splice-response threshold.',
        'ranking':'Recorded selection_rank when available, otherwise descending native Score; if neither field is recorded, retain source-file order. This is a within-pool computational ordering, not cross-gene potency or a direction-specific splice-efficacy ranking.'}
    from .candidate_minimums import region_minimum_delivery
    result['minimum_delivery'] = region_minimum_delivery(goal, {}, result, intervals_for_row)
    exception_records = sum(bool(row.get('default_gc_preference_exception'))
                            for group in sets for row in group['rows'])
    result['default_gc_preference_exception_records'] = exception_records
    if exception_records:
        result['default_gc_preference_note'] = (
            'Some candidates fall outside an unspecified default GC preference and were retained to meet an explicit minimum count. '
            'Their observed GC and the preferred interval are recorded per row. Explicit GC bounds and other required filters remain enforced; '
            'all retained candidates undergo the ordinary downstream evidence checks. This is not evidence of equivalent affinity or activity.')
    if result['minimum_delivery']['status'] != 'not_requested':
        csv_path = Path(run_dir) / 'requested_binding_site_sets.csv'
        fields = ['program_id', 'region', 'antisense_5to3', 'target_5to3', 'idt_chemistry',
                  'physical_binding_intervals_bed0', 'gc_fraction', 'default_gc_preference_exception',
                  'off_target_exact_count', 'off_target_1mm_count']
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for group in sets:
                for row in group['rows']:
                    writer.writerow({'program_id': group.get('program_id'), 'region': group.get('label'),
                        'antisense_5to3': row.get('synthesized_antisense_sequence'),
                        'target_5to3': row.get('target_binding_sequence'), 'idt_chemistry': row.get('IDT'),
                        'physical_binding_intervals_bed0': json.dumps(row['binding_intervals_bed0_half_open']),
                        'gc_fraction': row.get('GC'),
                        'default_gc_preference_exception': json.dumps(row.get('default_gc_preference_exception')),
                        'off_target_exact_count': row.get('Off_target_exact_count'),
                        'off_target_1mm_count': row.get('Off_target_1mm_count')})
        result['complete_table_relative_path'] = csv_path.name
    path = Path(run_dir)/'requested_binding_site_sets.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    # Full measurement rows remain in the artifact; keep model/report context bounded.
    preview_keys = ('synthesized_antisense_sequence','target_binding_sequence','sequence_role',
                    'binding_intervals_bed0_half_open','Off_target_exact_count','off_target_exact_count',
                    'off_target_exact_count_unit','Score','IDT','requested_binding_region','binding_overlap_requirement',
                    'GC','default_gc_preference_exception')
    previews = [{**group,'rows':[{k:row[k] for k in preview_keys if k in row} for row in group['rows'][:10]]}
                for group in sets]
    return {**result, 'sets':previews, 'artifact_path':str(path), 'preview_scope':'At most ten candidates per site; complete rows are in the artifact.'}


def binding_region_markdown(result):
    if not result:
        return ''
    lines = ['## Requested binding-region candidate sets','',result['interpretation'],result['ranking'],
             'Counts distinguish binding records from distinct molecules (antisense sequence plus chemistry). Full containment and any overlap define different populations; counts under those rules are not interchangeable.',
             f"Full candidate sets and measured annotations: `{result['artifact_path']}`."]
    minimum = result.get('minimum_delivery') or {}
    if minimum.get('status') not in (None, 'not_requested'):
        lines += ['', 'Requested minimum count status: **' + minimum['status'] + '**.']
        for group in minimum.get('groups') or []:
            lines.append(f"- {group.get('label') or group['group_id']}: {group['distinct_molecules']} distinct molecules; minimum {group['minimum']}; shortfall {group['shortfall']}.")
    if result.get('complete_table_relative_path'):
        lines += ['', '[Complete candidate table for every requested region](' + result['complete_table_relative_path'] + ').']
    if result.get('default_gc_preference_note'):
        lines += ['', result['default_gc_preference_note']]
    for group in result['sets']:
        label = group.get('label') or group.get('constraint_id')
        program = group.get('program_id')
        heading = f"{label} (design program: {program})" if program else str(label)
        lines += ['',f"### {heading}",'',
                  f"Rule: {group['aso_overlap_requirement']}; minimum overlap {group.get('minimum_overlap_bp', 1)} bp; {group.get('binding_record_count', group['candidate_count'])} binding records; {group.get('distinct_molecule_count', 'unknown')} distinct molecules. Status: {group['status']}.",
                  'BED0 half-open source region(s): '+json.dumps(group['intervals_bed0_half_open']), '',
                  '| ASO (5′–3′) | binding interval(s), BED0 | exact off-target count | native score |',
                  '|---|---|---|---|']
        if group.get('excluded_outside_target_rna_count'):
            lines[-3:-3] = [f"{group['excluded_outside_target_rna_count']} candidates excluded for lack of target-RNA locus support. {group['target_rna_scope']} Evidence: `{group['target_rna_exclusion_evidence']}`.", '']
        for row in group['rows'][:10]:
            seq = row.get('synthesized_antisense_sequence') or 'sequence role unverified; inspect source'
            count = row.get('Off_target_exact_count', row.get('off_target_exact_count','unknown'))
            lines.append(f"| {seq} | {json.dumps(row['binding_intervals_bed0_half_open'])} | {count} | {row.get('Score','unknown')} |")
        if group['candidate_count'] > 10:
            lines.append('First ten in the stated ordering shown; all candidates remain in the linked artifact.')
        if len(group.get('source_groups') or []) > 1:
            lines.append('Equivalent source groups: ' + '; '.join(str(item.get('label') or item.get('constraint_id')) for item in group['source_groups']) + '.')
    return '\n'.join(lines)
