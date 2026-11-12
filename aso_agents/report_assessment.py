"""Plain reading labels for the saved per-molecule assessment.

No assessment is rerun here. Unknown fields keep their recorded identifiers.
"""
from __future__ import annotations

from .report_candidate_view import MOTIF_LABELS


EVIDENCE_LABELS = {
    'physical_sequence_motif_screen': 'sequence motif check',
    'complete_off_target_screen': 'complete off-target search',
    'off_target_reference_identity': 'off-target reference identity',
    'attested_physical_chemistry': 'verified sugar and backbone chemistry',
    'target_RNA_accessibility': 'prediction of how exposed the target RNA site is',
    'splice_safeguard_measurements': 'predictions for the protected splice sites',
    'splice_safeguard_review': 'biological review of the splice safeguards',
}


def evidence_gaps(record):
    assessment = record.get('assessment') or {}
    missing = assessment.get('unavailable_evidence')
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        return 'Missing-check details unavailable'
    return '; '.join(EVIDENCE_LABELS.get(item, item) for item in missing) or 'None recorded'


def motif_notes(record):
    flags = record.get('motif_flags')
    if not isinstance(flags, list):
        return 'Motif results unavailable'
    assessed = {flag.get('rule_id'): flag for flag in ((record.get('assessment') or {}).get('motifs') or {}).get('flags') or []}
    notes = []
    for flag in flags:
        label = MOTIF_LABELS.get(flag, str(flag))
        applicability = (assessed.get(flag) or {}).get('applicability')
        if applicability in {'outside_studied_architecture', 'outside_studied_chemistry'}:
            label += '; this ASO uses a different chemistry'
        elif applicability == 'context_dependent':
            label += '; effect depends on chemistry and biological context'
        notes.append(label)
    return '; '.join(notes) or 'None flagged'


def prediction_note(record):
    assessment = record.get('assessment') or {}
    native = (assessment.get('native_prediction') or {}).get('directional_status')
    notes = {
        'opposing_predictions': 'Splicing predictions favor the opposite direction.',
        'no_directional_support': 'The recorded predictions give no support for the requested splicing change.',
        'mixed_or_partial_predictions': 'Splicing predictions give mixed or partial support.',
        'incomplete_prediction_evidence': 'Splicing predictions for this ASO are incomplete.',
    }
    parts = [notes[native]] if native in notes else []
    guards = (assessment.get('splice_safeguards') or {}).get('status')
    if guards == 'failed':
        parts.append('Predicted changes at protected splice sites exceed the specified limits.')
    elif guards == 'unmeasured':
        parts.append('Predictions for protected splice sites are unavailable.')
    if any((report.get('review') or {}).get('status') == 'unavailable'
           for report in (assessment.get('splice_safeguards') or {}).get('reports') or []):
        parts.append('Safeguard review is incomplete; the recorded default checks were used.')
    return ' '.join(parts)


def render_shortlist_markdown(shortlist, pool_molecule_count, *, include_rationale=True):
    """Render the recorded order, missing checks and chemistry-scoped flags.

    Reports with recommendation cards can render checks alone by setting
    include_rationale=False. Both views retain the same evidence table.
    """
    records = shortlist.get('records') or []
    reviews = any(prediction_note(record) for record in records)
    guarded = any(record.get('safeguard_status') != 'not_requested' for record in records)
    headers = ['Order', 'ASO (5′→3′)', 'Status', 'Evidence gaps']
    if reviews:
        headers.append('Review note')
    if guarded:
        headers.append('Splice safeguards')
    headers += ['Off-target screen', 'Motif flags', 'Reference Tm (°C)', 'Reference ΔG (kcal/mol)']
    review = shortlist.get('review') or {}
    introduction = []
    if include_rationale and review.get('summary'):
        introduction += [review['summary'], '', review.get('size_reason') or '', '']
    for record in records:
        item = record.get('recommendation_review') or {}
        if include_rationale and item:
            introduction += [f"## {record['recommendation_order']}. {record['antisense_5to3']}", '',
                item['why_chosen'], '', '**Tradeoff:** ' + item['tradeoff'], '',
                '**Panel role:** ' + item['panel_role'], '']
    lines = ['# Recommended shortlist' if include_rationale else '## Selection checks', '', *introduction,
        (f'{len(records)} distinct molecules selected; {pool_molecule_count} in the assessed eligible pool.'
         if include_rationale else f'Checks for {len(records)} recommended ASO{"s" if len(records) != 1 else ""}.'), '',
        'This table gives the selection checks for the ASOs in panel order. Evidence gaps name the checks still missing.', '',
        'Off-targets are other sequences an ASO may also bind.' + (' Splice safeguards check whether predicted changes at protected splice sites stay within the specified limits.' if guarded else ''), '',
        'Motif flags identify short sequence patterns. Each flag names its study context; its relevance depends on the ASO chemistry.', '',
        '| ' + ' | '.join(headers) + ' |',
        '|' + '---|' * len(headers)]
    statuses = {'provisional': 'Some evidence unavailable', 'computational_candidate': 'Computational checks available',
                'exploratory': 'Prediction requires review'}
    guards = {'passed': 'Within the specified limits', 'failed': 'Outside the specified limits',
              'unmeasured': 'Measurements unavailable', 'not_requested': 'Outside this analysis',
              'monitoring_only': 'Changes recorded for context'}
    screens = {'complete_within_declared_scope': 'Completed', 'incomplete_or_unavailable': 'Incomplete or unavailable'}
    def cell(value):
        return str(value if value is not None else 'unavailable').replace('|', '\\|').replace('\n', ' ')
    for record in records:
        values = [record.get('recommendation_order'), record.get('antisense_5to3'),
                  statuses.get(record.get('recommendation_status'), record.get('recommendation_status')),
                  evidence_gaps(record)]
        if reviews:
            values.append(prediction_note(record) or 'No additional review note')
        if guarded:
            values.append(guards.get(record.get('safeguard_status'), record.get('safeguard_status')))
        values += [screens.get(record.get('off_target_status'), record.get('off_target_status')), motif_notes(record),
                   record.get('reference_tm_c'), record.get('reference_delta_g_kcal_mol')]
        lines.append('| ' + ' | '.join(cell(value) for value in values) + ' |')
    lines.extend(['', 'Tm (melting temperature) and ΔG (binding free energy) describe a perfectly matched, unmodified RNA/DNA duplex at 1 M sodium. These values provide a common sequence reference. The modified ASO’s affinity requires chemistry-specific measurements.'])
    if include_rationale:
        lines.extend(['', 'Selection applies the requested filters, then the design ranking, comparable off-target checks and relevant sequence motifs. Requested panel roles take priority. Full CSV/JSON records include the exact statuses, reference conditions and any predicted changes at unannotated splice sites.'])
    if shortlist.get('status') == 'awaiting_size_preference':
        lines.extend(['', 'Choose a shortlist size to complete selection. The full assessed pool is available.'])
    elif shortlist.get('minimum_count_shortfall'):
        lines.extend(['', f"The requested minimum is unmet by {shortlist['minimum_count_shortfall']} distinct molecules under the retained constraints."])
    return '\n'.join(lines) + '\n'
