"""Steric binding designs on mature RNA, using typed reference geometry."""
from copy import deepcopy
import json
from pathlib import Path

from .gapmer_selectivity import resolve_catalog_label
from .region_overlap import candidate_matches_region_groups, matches_constraint_filters
from .region_projection import genomic_sources


def uses_mature_rna(program):
    return (isinstance(program, dict)
            and program.get('modality') == 'steric'
            and program.get('coordinate_space') in {'cdna', 'transcript', 'mature_transcript'}
            and program.get('intent') != 'splice_modulation')


def merge_rows(rows):
    """Merge only adjacent/overlapping pieces of the same transcript and strand."""
    groups = {}
    for row in rows:
        key = (str(row[0]), str(row[5]) if len(row) > 5 else '+')
        groups.setdefault(key, []).append(list(row))
    result = []
    for pieces in groups.values():
        merged = []
        for row in sorted(pieces, key=lambda r:(int(r[1]),int(r[2]))):
            if merged and int(row[1]) <= int(merged[-1][2]):
                merged[-1][2] = max(int(merged[-1][2]), int(row[2]))
            else:
                merged.append(row)
        result.extend(merged)
    return result


def compile_regions(runtime, goal, program, base_rows):
    catalog = goal.get('isoform_transcripts') or {}
    primary, _ = runtime._select_constraint_primary_isoform(goal)
    label = runtime._select_design_program_isoform_label(catalog, program, primary, goal)
    if label not in catalog:
        raise ValueError('mature_steric_reference_unresolved')
    entry = catalog[label]
    transcript = entry.get('transcript_id') or entry.get('id') or label
    base = merge_rows(runtime._map_genomic_bed_rows_to_cdna(base_rows, entry, label))
    filters = []
    sources = runtime._constraint_entries_for_scope(goal, program=program)
    if not base_rows and sources and runtime._constraints_can_seed_without_base(sources):
        # A constraint-only program has no independent base interval. Its
        # predicates define eligibility on the resolved mature transcript;
        # intersecting feature coordinates here would erase legal longer ASOs.
        # A supplied base that fails projection is never replaced by this domain.
        exons = entry.get('exons_tx') or []
        if exons and all(isinstance(exon, dict)
                and type(exon.get('start')) is int and type(exon.get('end')) is int
                and 1 <= exon['start'] <= exon['end'] for exon in exons):
            length = sum(exon['end']-exon['start']+1 for exon in exons)
            base = [[transcript, 0, length, str(label), 0,
                     '-' if str(entry.get('strand')) in {'-', '-1'} else '+']]
    from .mature_binding import compile_scope
    shared_binding = compile_scope(runtime, goal, program, label, base, sources)
    if shared_binding:
        global_sources = (goal.get('materialized_region_constraints') or {}).get('global') or []
        if global_sources and sources[:len(global_sources)] != global_sources:
            sources = [*global_sources, *sources]
            shared_binding = compile_scope(runtime, goal, program, label, base, sources)
        projected_base = runtime._map_cdna_bed_rows_to_genomic(base, entry, label)
        return projected_base, {'constraint_count': len(sources), 'output_space': 'genomic',
            'tile_filter_rows': projected_base, 'tile_overlap_requirement': 'full_containment',
            'tile_constraint_filters': [],
            'constraint_enforcement': 'reference_verified_mature_binding',
            'mature_rna': {'transcript_label': label, 'transcript_id': transcript,
                'search_rows_cdna': base, 'constraint_filters_cdna': [],
                'shared_binding': shared_binding}}
    for constraint in sources:
        rows = runtime._constraint_rows_in_output_space(structured_goal=goal, constraint=constraint,
            output_space='cdna', base_rows=base, isoform_transcripts=catalog, iso_label_primary=label)
        normalized = []
        for row in rows:
            if resolve_catalog_label(row[0], catalog, catalog) != label:
                continue
            normalized.append([transcript, *row[1:]])
        filters.append({'constraint_id':constraint.get('constraint_id'),
            'rows':merge_rows(normalized), 'operator':constraint.get('operator') or 'union',
            'aso_overlap_requirement':constraint.get('aso_overlap_requirement') or 'any_overlap',
            'minimum_overlap_bp':constraint.get('minimum_overlap_bp') or 1})
    projected_filters = []
    for rule in filters:
        projected = genomic_sources(goal, {'coordinate_space':'cdna', 'materialized_rows':rule['rows']})
        projected_filters.append({**rule,
            'rows':[row for item in projected for row in item['genomic_rows']],
            'region_groups':[item['genomic_rows'] for item in projected]})
    projected_base = runtime._map_cdna_bed_rows_to_genomic(base, entry, label)
    return projected_base, {'constraint_count':len(filters), 'output_space':'genomic',
        'tile_filter_rows':projected_base, 'tile_overlap_requirement':'full_containment',
        'tile_constraint_filters':projected_filters,
        'mature_rna':{'transcript_label':label, 'transcript_id':transcript,
                      'search_rows_cdna':base, 'constraint_filters_cdna':filters}}


def build_tiles(runtime, goal, job, chemistry, output_dir):
    scope = ((job.get('selection_meta') or {}).get('region_constraints') or {}).get('mature_rna')
    if not scope:
        return None
    label = scope['transcript_label']
    catalog = goal.get('isoform_transcripts') or {}
    entry = catalog[label]
    sequences = runtime._build_isoform_cdna_sequences({label:entry}, goal.get('species') or 'human',
        reference_fasta=runtime._resolve_patient_reference_fasta(goal))
    sequence = sequences.get(label)
    if not sequence:
        raise ValueError('mature_steric_reference_sequence_missing')
    recommendation = chemistry.get('recommendation') or chemistry
    length = int(recommendation.get('length') or job.get('length') or 0)
    chemistry_label = recommendation.get('CHEMISTRY') or recommendation.get('chemistry')
    if length <= 0:
        raise ValueError('mature_steric_length_missing')
    transcript = scope['transcript_id']
    strand = '-' if str(entry.get('strand')) in {'-', '-1'} else '+'
    limits = runtime._resolve_homopolymer_per_base_limits(goal)
    binding = None
    if scope.get('shared_binding'):
        from .mature_binding import BindingContext
        binding = BindingContext(runtime, goal, scope['shared_binding'])
    rows = []
    for start in range(len(sequence)-length+1):
        interval = [(transcript,start,start+length)]
        if not candidate_matches_region_groups(interval, [scope['search_rows_cdna']], 'full_containment'):
            continue
        if not matches_constraint_filters(interval, scope['constraint_filters_cdna']):
            continue
        target = sequence[start:start+length]
        binding_evidence = binding.evaluate(target, start) if binding else None
        if binding and not binding_evidence:
            continue
        idt, synthesis, warning = runtime._format_idt_and_synthesis(target, chemistry_label)
        if warning:
            raise ValueError('mature_steric_chemistry_unresolved: '+str(warning))
        segments = runtime._map_cdna_range_to_genomic_segments(start+1,start+length,entry)
        if sum(b-a+1 for _,a,b in segments) != length:
            raise ValueError('mature_steric_projection_incomplete')
        rows.append({'id':f"{job['job_id']}_cdna_{start+1}",
            'Name':f'{transcript}_cdna{start+1}-{start+length}',
            'Chrom':transcript,'Start':start+1,'End':start+length,'Strand':strand,
            'Sequence':target,'IDT':idt,'Synthesis':synthesis,
            'GC':round((target.count('G')+target.count('C'))/length,4),
            'No_homopolymer':not limits or runtime.sequence_passes_homopolymer_limits(synthesis,limits),
            'source':'steric_cdna','coordinate_system':'1based_inclusive',
            'Genomic_segments':json.dumps([{'chrom':c,'start':a,'end':b} for c,a,b in segments]),
            'mature_rna_reference':transcript, 'design_job_id':job['job_id'],
            **({'mature_rna_binding': binding_evidence} if binding_evidence else {})})
    runtime._annotate_tile_rows_with_rnafold(rows)
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    result = runtime._write_tiles_json_payload(rows, output_dir, 'mature_steric_tiles',
        'Steric ASOs enumerated on reference mature RNA; exon segments retain genomic provenance.')
    if rows:
        from .pipeline_tools import _merge_parameter_constraints
        mapping = runtime._annotate_gapmer_off_targets(result['json_path'],goal,
            _merge_parameter_constraints(goal))
        if not isinstance(mapping,dict) or mapping.get('error'):
            raise ValueError('mature_steric_off_target_mapping_failed: '+str(mapping))
        from .pipeline_tools import _enrich_exact_off_target_bins_for_tiles, exact_gene_count_scope
        scope = exact_gene_count_scope(runtime.get_species_config(runtime.normalize_species(goal.get('species'))),
                                       goal.get('gene_id'), mapping)
        _enrich_exact_off_target_bins_for_tiles(result['json_path'], scope)
        result['off_target_mapping'] = mapping
        result['off_target_count_scope'] = scope
    result['row_count'] = len(rows)
    return result
