"""Reconcile scoped design obligations against verbatim user evidence."""
from copy import deepcopy
import re


def _quote(value, source, key='quote'):
    text = value.get(key) if isinstance(value, dict) else None
    return isinstance(text, str) and bool(text.strip()) and text.strip().casefold() in source.casefold()


def _references(value, source):
    references = value.get('transcript_ids')
    if (not isinstance(references, list) or not references
            or not _quote(value, source, 'references_quote')):
        return None
    quote = value['references_quote']
    result = []
    for reference in references:
        if (not isinstance(reference, str) or not reference.strip()
                or re.search(r'(?<![\w.])' + re.escape(reference) + r'(?![\w.])', quote) is None):
            return None
        if reference not in result:
            result.append(reference)
    return result


def _positive(value):
    return type(value) is int and value > 0


def apply_program_obligations(goal, audit, source):
    """Do not infer scopes from labels, gene membership, or unquoted prose.

    Counts need a unique typed program selector. Shared binding and feature
    rules apply only when independently extracted as full-panel obligations.
    Unsupported or ambiguous positive obligations remain explicit blockers.
    """
    receipt = {'status': 'not_requested', 'applied': [], 'unresolved': []}
    programs = goal.get('design_programs')
    if not isinstance(programs, list) or not all(isinstance(p, dict) for p in programs):
        return receipt
    working = deepcopy(programs)
    cohort = audit.get('shared_binding_cohort')
    references = None
    binding_programs = bool(working) and all(
        p.get('modality') == 'gapmer' or
        (p.get('modality') == 'steric' and (p.get('intent') or goal.get('intent')) == 'binding')
        for p in working)
    if cohort is not None:
        references = _references(cohort, source) if isinstance(cohort, dict) else None
        if (not _quote(cohort, source) or not references or len(references) < 2
                or cohort.get('applies_to_all_candidates') is not True
                or cohort.get('reference_molecule') != 'mature_rna' or not binding_programs):
            receipt['unresolved'].append({'kind': 'shared_binding_cohort', 'reason': 'scope_or_evidence_unverified'})
            references = None
        else:
            for program in working:
                program['transcript_ids'] = list(references)
                program['coordinate_space'] = 'cdna'
                program.setdefault('target_type', 'transcript_cohort')
            receipt['applied'].append({'kind': 'shared_binding_cohort', 'transcript_ids': references,
                                       'quote': cohort['quote'], 'references_quote': cohort['references_quote']})

    features = audit.get('feature_requirements') or []
    if not isinstance(features, list):
        receipt['unresolved'].append({'kind': 'feature_requirements', 'reason': 'invalid_list'})
        features = []
    for item in features:
        if (not isinstance(item, dict) or not _quote(item, source)
                or item.get('applies_to_all_candidates') is not True or not binding_programs
                or item.get('feature_name') not in {'start_codon', 'cds'}
                or item.get('aso_overlap_requirement') not in {'any_overlap', 'contains_feature', 'full_containment'}
                or not _positive(item.get('minimum_overlap_bp', 1))):
            receipt['unresolved'].append({'kind': 'feature_requirement', 'reason': 'scope_or_evidence_unverified'})
            continue
        reference_scope = item.get('reference_scope')
        if reference_scope == 'shared_binding_cohort' and references:
            targets = references
        elif reference_scope == 'primary_reference':
            targets = [None]
        elif reference_scope == 'explicit_references':
            targets = _references(item, source)
        else:
            targets = None
        if not targets:
            receipt['unresolved'].append({'kind': 'feature_requirement', 'reason': 'reference_scope_unverified'})
            continue
        for program in working:
            existing = program.get('region_constraints')
            if existing is not None and (not isinstance(existing, list) or not all(isinstance(c, dict) for c in existing)):
                receipt['unresolved'].append({'kind': 'feature_requirement', 'reason': 'existing_constraints_invalid'})
                continue
            constraints = deepcopy(existing or [])
            for reference in targets:
                constraint = {'source_type': 'transcript_feature', 'feature_name': item['feature_name'],
                              'operator': 'intersect', 'coordinate_space': 'cdna',
                              'aso_overlap_requirement': item['aso_overlap_requirement'],
                              'minimum_overlap_bp': item.get('minimum_overlap_bp', 1),
                              'source': 'quoted_user_requirement', 'notes': item['quote']}
                if reference:
                    constraint['reference_transcript_id'] = reference
                identity = ('source_type', 'feature_name', 'reference_transcript_id',
                            'operator', 'aso_overlap_requirement', 'minimum_overlap_bp')
                if not any(all(c.get(key, 1 if key == 'minimum_overlap_bp' else None) == constraint.get(key)
                                   for key in identity) for c in constraints):
                    constraints.append(constraint)
            program['region_constraints'] = constraints
            program['coordinate_space'] = 'cdna'
        receipt['applied'].append({'kind': 'feature_requirement', 'feature_name': item['feature_name'],
                                   'reference_scope': reference_scope, 'quote': item['quote']})

    counts = audit.get('program_candidate_counts') or []
    if not isinstance(counts, list):
        receipt['unresolved'].append({'kind': 'program_candidate_counts', 'reason': 'invalid_list'})
        counts = []
    assigned = {}
    for item in counts:
        if (not isinstance(item, dict) or not _quote(item, source)
                or not _positive(item.get('count')) or not _positive(item.get('length_nt'))
                or item.get('kind') not in {'exact', 'minimum', 'maximum'}
                or re.search(r'(?<!\d)' + str(item['length_nt']) + r'(?!\d)', item['quote']) is None
                or item.get('modality') not in {None, 'gapmer', 'steric'}):
            receipt['unresolved'].append({'kind': 'program_candidate_count', 'reason': 'selector_or_evidence_unverified'})
            continue
        matches = [i for i, program in enumerate(working)
                   if program.get('length') == item['length_nt']
                   and (item.get('modality') is None or program.get('modality') == item['modality'])]
        if len(matches) != 1:
            receipt['unresolved'].append({'kind': 'program_candidate_count', 'reason': 'selector_not_unique',
                                          'length_nt': item['length_nt'], 'matching_programs': len(matches)})
            continue
        index = matches[0]
        signature = (item['count'], item['kind'])
        if index in assigned and assigned[index] != signature:
            receipt['unresolved'].append({'kind': 'program_candidate_count', 'reason': 'conflicting_counts'})
            continue
        assigned[index] = signature
        program = working[index]
        if item['kind'] != 'maximum':
            program['candidate_minimum'] = item['count']
        if item['kind'] != 'minimum':
            program['candidate_limit'] = item['count']
        else:
            program.pop('candidate_limit', None)
        if (program.get('candidate_minimum') or 0) > (program.get('candidate_limit') or float('inf')):
            receipt['unresolved'].append({'kind': 'program_candidate_count', 'reason': 'incompatible_count_range'})
        receipt['applied'].append({'kind': 'program_candidate_count', 'program_id': program.get('program_id'),
                                   'length_nt': item['length_nt'], 'count': item['count'], 'quote': item['quote']})
    if receipt['unresolved']:
        receipt['status'] = 'unresolved'
        receipt['applied'] = []
    elif receipt['applied']:
        goal['design_programs'] = working
        receipt['status'] = 'applied'
    return receipt
