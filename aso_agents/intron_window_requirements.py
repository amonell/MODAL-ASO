"""Bind unambiguous user window instructions before intron-event execution.

This omission repair uses user evidence, never model objective prose or a
gene-specific coordinate. Complex or ambiguous requests need typed constraints.
"""
import re


def bind_missing_intron_windows(goal, request, overlap_requirement=None):
    programs = [p for p in goal.get('design_programs') or [] if isinstance(p, dict)]
    missing = [p for p in programs if p.get('splice_event')
               and p.get('targeting_policy') == 'explicit_user_region'
               and not p.get('region_constraints') and not goal.get('region_constraints')]
    if not missing:
        return {'status': 'not_required', 'changed': False}
    text = str(request or '')
    # Admit a single symmetric window in a single-program request. Multiple
    # programs/widths/references must retain their own explicitly typed scopes.
    patterns = [
        r'(?P<width>\d+)\s*(?:nt|bp|nucleotides?|bases?)\s+(?:on\s+)?each\s+side\s+of\s+(?:each|both|the)\s+(?:annotated\s+)?(?:intron\s+)?(?:junctions?|boundaries|boundary)',
        r'(?:window|windows)\s*(?:of\s*)?(?:\+\s*/\s*-|±)\s*(?P<width>\d+)\s*(?:nt|bp|nucleotides?|bases?)\b',
    ]
    matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
    widths = {int(m['width']) for m in matches}
    both = bool(re.search(r'\bboth\s+(?:annotated\s+)?(?:intron\s+)?(?:boundaries|junctions)\b|\beach\s+junction\b', text, re.I))
    references = set(re.findall(r'\b(?:ENS[A-Z]*T\d+(?:\.\d+)?|[NX]M_\d+(?:\.\d+)?)\b', text))
    ambiguous = any(re.search(r'\b(?:not|never|whether|either|or|instead)\b',
                    text[max(text.rfind('\n', 0, m.start()), text.rfind(';', 0, m.start())) + 1:m.start()], re.I)
                    for m in matches)
    # A bare request to overlap a named junction window means an intersection;
    # crossing the junction or containing the feature is a different predicate.
    overlap = overlap_requirement
    if not overlap and re.search(r'\brequire\s+overlap\s+with\s+(?:the\s+)?(?:corresponding\s+)?(?:junction|boundary|splice)\s+window\b', text, re.I):
        overlap = 'any_overlap'
    special_overlap = bool(re.search(r'\bcross(?:ing)?\s+(?:the\s+)?(?:splice\s+)?(?:junction|boundar)|\bcontains?\s+(?:the\s+)?(?:feature|junction)\b', text, re.I))
    overlap_counts = {int(m[1]) for m in re.finditer(r'\b(\d+)\s*(?:nt|bp|bases?)\s+(?:of\s+)?overlap\b', text, re.I)}
    exact_scope = (len(programs) == len(missing) == 1 and len(widths) == 1
                   and next(iter(widths), 0) > 0 and both and len(references) <= 1
                   and not ambiguous and not special_overlap and not (overlap_counts - {1})
                   and overlap in {'any_overlap', 'full_containment'})
    if not exact_scope:
        return {'status': 'needs_clarification', 'changed': False,
            'program_ids': [p.get('program_id') for p in missing],
            'questions': ['Please confirm the upstream and downstream search-window sizes at each requested intron boundary, '
                          'and whether each ASO may overlap the window or must lie fully within it. '
                          'For multiple introns or transcripts, give the window for each.'],
            'reason': 'Explicit intron search windows are not represented by executable region constraints.'}
    width = next(iter(widths))
    program = missing[0]
    event = program['splice_event']
    ordinal = event.get('intron_number')
    reference = event.get('reference_transcript_id') or program.get('transcript_id')
    if type(ordinal) is not int or ordinal < 1 or not reference:
        return {'status': 'not_required', 'changed': False}  # Reference/event validation owns this error.
    program['region_constraints'] = [
        {'constraint_id': f'intron_boundary_{kind}', 'source_type': 'splice_relative_window',
         'label': f'intron {ordinal} {kind} window', 'operator': 'union',
         'aso_overlap_requirement': overlap, 'minimum_overlap_bp': 1,
         'splice_relative_window': {'reference_transcript_id': reference,
             'exon_label': exon, 'anchor_site': kind,
             'relative_start_nt': -width, 'relative_end_nt': width}}
        for kind, exon in [('donor', ordinal), ('acceptor', ordinal + 1)]]
    receipt = {'status': 'bound_from_user_evidence', 'changed': True,
        'program_id': program.get('program_id'), 'window_source_quotes': [m.group(0) for m in matches],
        'width_each_side_nt': width, 'aso_overlap_requirement': overlap,
        'reference_transcript_id': reference, 'intron_number': ordinal}
    goal['intron_window_resolution'] = receipt
    return receipt
