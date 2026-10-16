"""Keep physical molecule identity separate from a dual-orientation search."""
from __future__ import annotations

import hashlib

SEQUENCE_ROLES = frozenset({'antisense_oligo_sequence', 'target_site_sequence', 'unknown'})


def molecule_key(row: dict) -> str:
    identity = str(row.get('aso_id') or row.get('name') or '')
    sequence = str(row.get('sequence') or '').upper()
    return hashlib.sha256((identity + '\0' + sequence).encode()).hexdigest()


def preferred_binding_hits(row: dict, hits: list[dict]) -> list[dict]:
    """Only confirmed target strings authorize evaluation of their complement."""
    role = row.get('sequence_role_confirmation', {}).get('role') or row.get('declared_sequence_role')
    orientation = 'input_sequence' if role == 'target_site_sequence' else 'reverse_complement'
    palindrome = row.get('sequence') == row.get('reverse_complement_sequence')
    return [hit for hit in hits if hit.get('match_orientation') == orientation or palindrome]


def interpret_orientation(row: dict) -> dict:
    sequence = str(row.get('sequence') or row.get('raw_sequence') or '').upper()
    if not sequence:
        return {}
    complement = str(row.get('reverse_complement_sequence') or '').upper()
    hits = [h for h in (row.get('mapping_hits') or []) + (row.get('reference_candidate_hits') or [])
            if isinstance(h, dict) and h.get('matched_target_sequence')]
    binding = preferred_binding_hits(row, hits)
    confirmation = row.get('sequence_role_confirmation') or {}
    declared_role = str(row.get('declared_sequence_role') or '').strip()
    role = confirmation.get('role') or (
        declared_role if declared_role in {'antisense_oligo_sequence', 'target_site_sequence'} else 'unknown'
    )
    same_strand = [h for h in hits if h.get('match_orientation') == 'input_sequence']
    # An exact same-strand site is a search observation, not evidence that a
    # declared or confirmed physical ASO was supplied backwards.
    likely_reversed = bool(same_strand and not binding and complement and complement != sequence
                           and role == 'unknown')
    needs_confirmation = bool(same_strand and not binding and role == 'unknown' and not confirmation)
    match = binding[0] if binding else (same_strand[0] if same_strand else {})
    explicitly_unknown = confirmation.get('role') == 'unknown' or (not confirmation and declared_role == 'unknown')
    if binding and role == 'unknown' and not explicitly_unknown:
        role = 'antisense_oligo_sequence'
    actual = complement if role == 'target_site_sequence' else sequence
    if role == 'target_site_sequence':
        note = 'The supplied string is declared or confirmed as a target site. Its reverse complement is the evaluated antisense molecule.'
    elif binding and explicitly_unknown:
        note = 'An exact reverse-complement match supports a conditional antisense interpretation; the supplied sequence role remains explicitly unresolved.'
    elif binding:
        note = 'An antisense binding match supports evaluating the original supplied molecule; its identity is unchanged.'
    elif same_strand:
        note = ('The supplied string matches the transcript-oriented strand. This is not an exact antisense binding match for the original molecule. '
                'The reverse complement is a separate conditional hypothesis and has not been substituted. '
                'A missing exact match does not establish inactivity.')
        if likely_reversed:
            note += (' Likely reversed entry relative to this exact transcript-oriented site: '
                     'the reverse-complement interpretation has an exact antisense site, '
                     'but the submitted molecule remains evaluated as written.')
    else:
        note = 'No exact binding orientation was established in the searched context. Preserve the original molecule and report binding uncertainty.'
        if role == 'antisense_oligo_sequence':
            note += ' The supplied string remains the declared antisense molecule; a missing exact site does not change its input role.'
    if confirmation.get('role') in {'antisense_oligo_sequence', 'target_site_sequence'}:
        note += ' The user has explicitly confirmed the sequence role; that question is resolved. Any remaining uncertainty concerns biological binding or activity, not which molecule was supplied.'
    result = {
        'provided_sequence_role': role, 'input_sequence': sequence,
        'evaluated_molecule_sequence': actual,
        'sequence_role_confirmation': confirmation,
        'sequence_role_source': 'user_confirmation' if confirmation else row.get('sequence_role_source'),
        'requires_confirmation': needs_confirmation,
        'original_molecule_exact_binding': bool(preferred_binding_hits({**row, 'sequence_role_confirmation': {},
            'declared_sequence_role': 'antisense_oligo_sequence'}, hits)),
        'likely_reversed_entry': likely_reversed,
        'interpretation_note': note,
    }
    if match:
        result['match_orientation'] = match.get('match_orientation')
        result['matched_target_sequence'] = match.get('matched_target_sequence')
    if role == 'antisense_oligo_sequence' or role == 'target_site_sequence':
        result['synthesized_antisense_sequence'] = actual
    if same_strand and not binding:
        result['conditional_reverse_complement_sequence'] = complement
    return result
