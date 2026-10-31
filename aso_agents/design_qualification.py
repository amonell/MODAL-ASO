"""Versioned, composable evidence contracts for existing typed design programs.

This describes computational completion, not biological efficacy. It neither
creates new programs nor infers a mechanism from a gene name or a paper panel.
"""
from collections import Counter

from .candidate_presentation import molecule_key
from .gapmer_selectivity import resolve_catalog_label

VERSION = 'modal-design-qualification/1'


def strategy_contract(program):
    event = (program.get('splice_event') or {}).get('event_type')
    constraints = program.get('region_constraints') or []
    required = ['resolved_reference_geometry', 'physical_binding_and_chemistry',
                'requested_constraints', 'declared_off_target_scope']
    if event in {'intron_retention', 'intron_excision'}:
        strategy = event
        required += ['both_flanking_junctions', 'scoped_perturbation_evidence']
        limit = 'Junction perturbations do not establish retention/excision, frame consequences, NMD or protein outcome.'
    elif program.get('modality') == 'gapmer':
        strategy = 'isoform_selective_gapmer' if program.get('preservation_targets') else 'gapmer_knockdown'
        required += ['mature_RNA_target_binding']
        if strategy == 'isoform_selective_gapmer':
            required += ['protected_reference_binding_assessment']
        limit = 'Mature-transcript selectivity does not establish precursor selectivity, cleavage, potency or protein reduction.'
    elif program.get('desired_effect') in {'inclusion', 'skipping'}:
        strategy = 'splice_' + program['desired_effect']
        required += ['physical_exon_event', 'scoped_perturbation_evidence']
        if program.get('intent') == 'knockdown':
            required += ['separately_qualified_frame_and_NMD_hypothesis']
        limit = 'Local sequence perturbations are proxies, not ASO occupancy or measured exon/isoform abundance; knockdown via splicing additionally needs frame/NMD evidence.'
    elif any(rule.get('source_type') == 'transcript_feature' and rule.get('feature_name') == 'start_codon'
             for rule in constraints if isinstance(rule, dict)):
        strategy = 'translation_start_blockade'
        required += ['annotated_start_codon_overlap']
        limit = 'Annotated start-site overlap does not establish translational inhibition or protein reduction.'
    elif (program.get('rbp_binding') or program.get('mirna_binding')
          or any(rule.get('source_type') in {'rbp_binding', 'mirna_binding'}
                 for rule in constraints if isinstance(rule, dict))):
        strategy = 'regulatory_site_blockade'
        required += ['reference_bound_regulatory_feature', 'requested_site_overlap']
        limit = 'Feature overlap is not proof of occupancy or a downstream expression effect in the requested tissue.'
    else:
        strategy = 'steric_binding'
        limit = 'Physical binding-site eligibility alone does not establish a downstream biological effect.'
    return {'version': VERSION, 'strategy': strategy, 'required_evidence': required,
            'interpretation_limit': limit, 'biological_validation': 'not_established'}


def _program_id(row):
    return row.get('program_id') or (row.get('source_annotations') or {}).get('program_id')


def _summary(rows):
    statuses, unavailable, sites = Counter(), Counter(), set()
    for row in rows:
        assessment = row.get('candidate_assessment') or {}
        statuses[assessment.get('recommendation_status') or 'unassessed'] += 1
        unavailable.update(assessment.get('unavailable_evidence') or [])
        for site in row.get('physical_splice_elements') or []:
            sites.add((site.get('transcript_id'), site.get('exon_ordinal'), site.get('element')))
    return {'binding_record_count': len(rows), 'distinct_molecule_count': len({molecule_key(row) for row in rows}),
            'recommendation_status_by_binding_record': dict(statuses),
            'unavailable_evidence_by_binding_record': dict(unavailable),
            'physical_splice_elements': [list(site) for site in sorted(sites, key=str)]}


def qualify_panel(goal, pool, selected, selection):
    catalog = goal.get('isoform_transcripts') or {}
    programs = [program for program in goal.get('design_programs') or [] if isinstance(program, dict)]
    contracts = []
    for program in programs:
        identifier = program.get('program_id')
        # Inherit only typed global preservation/feature contracts, never prose.
        scope = {**{key: goal[key] for key in ('preservation_targets', 'region_constraints', 'rbp_binding', 'mirna_binding')
                    if key in goal}, **program}
        references = list(dict.fromkeys(value for value in [scope.get('transcript_id'),
            (scope.get('splice_event') or {}).get('reference_transcript_id'),
            *(scope.get('transcript_ids') or [])] if value))
        resolved = [{'requested': value, 'catalog_label': resolve_catalog_label(value, catalog, catalog)}
                    for value in references]
        members = lambda rows: [row for row in rows if _program_id(row) == identifier
                               or len(programs) == 1 and not _program_id(row)]
        chosen, candidates = members(selected), members(pool)
        contracts.append({'program_id': identifier, **strategy_contract(scope),
            'references': resolved, 'reference_status': 'unresolved' if not resolved
                or any(item['catalog_label'] is None for item in resolved) else 'resolved_catalog_identity',
            'requested_outcomes': {key: scope[key] for key in ('intent', 'desired_delta', 'desired_effect', 'splice_event', 'preservation_targets') if key in scope},
            'pool': _summary(candidates), 'selected': _summary(chosen),
            'completion_status': 'no_selected_candidates' if not chosen else 'computational_panel_exported'})
    return {'version': VERSION, 'scope': 'computational_evidence_and_panel_coverage',
            'programs': contracts, 'pool': _summary(pool), 'selected': _summary(selected),
            'selection': selection, 'explicit_priorities': goal.get('candidate_panel_objectives') or [],
            'exon_conversion_bindings': goal.get('exon_conversion_bindings') or [],
            'unassigned_selected_records': sum(not _program_id(row) for row in selected) if len(programs) > 1 else 0,
            'interpretation': 'Successful export does not verify all mechanism requirements. Missing evidence stays explicit; counts describe binding records and distinct physical molecules separately. Model perturbations and manuscript sequence overlap are not biological validation.'}
