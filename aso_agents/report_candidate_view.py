"""A sequence-first reading view of the exact selected-panel table.

This module only projects recorded fields. It never infers a sequence strand,
joins molecules by sequence, changes selection order, or fills missing results.
The caller retains the complete source table alongside this view.
"""
from __future__ import annotations

import html
import json
import math
import re

from .report_orientation import section_tables


def _position(value):
    """Spell out BED0 without converting coordinates or guessing the assembly."""
    try:
        intervals = json.loads(value)
    except (TypeError, ValueError):
        return value
    if (not isinstance(intervals, list) or not intervals
            or not all(isinstance(row, list) and len(row) == 3
                       and isinstance(row[0], str) and type(row[1]) is int
                       and type(row[2]) is int and 0 <= row[1] < row[2] for row in intervals)):
        return value
    return '; '.join(f'{chrom}:{start}–{end}' for chrom, start, end in intervals) + ' (zero-based; end excluded)'


def screening_fields(value):
    """Expand the native genome-screen record without inventing missing bins."""
    parts = value.split('; ')
    statuses = {
        'complete': 'The off-target search finished.',
        'incomplete': 'The off-target search is incomplete.',
        'unknown': 'This report does not say whether the off-target search finished.',
        'unavailable': 'This report is missing the off-target search result.',
        'failed': 'The off-target search failed.',
    }
    if parts[0] not in statuses:
        return None
    fields = [('Search for other sequences this ASO may bind', statuses[parts[0]])]
    for part in parts[1:]:
        limit = re.fullmatch(r'searched ≤(\d+) mismatch\(es\)', part)
        exact = re.fullmatch(r'exact=(\d+) (.+)', part)
        near = re.fullmatch(r'(\d+)mm=(\d+) recorded hits', part)
        if limit:
            fields.append(('Recorded search limit', f'Up to {limit[1]} base differences'))
        elif exact:
            fields.append(('Exact matches', f'{exact[1]} {exact[2]}'))
        elif near:
            fields.append((f'Matches with {near[1]} base ' + ('difference' if near[1] == '1' else 'differences'), f'{near[2]} recorded hits'))
        else:
            return None
    return fields


def candidate_reading_view(section):
    """Return no projection when a panel's format or sequence role is unclear."""
    tables = section_tables(section)
    if len(tables) != 1:
        return ''
    table = tables[0]
    required = {'candidate', 'antisense oligo (5′→3′)', 'physical binding interval(s), bed0', 'genomic screening'}
    if not required <= set(table['headers']) or not 0 < len(table['rows']) <= 24:
        return ''
    records = [dict(zip(table['headers'], row)) for row in table['rows']]
    if not all(re.fullmatch(r'[ACGTUacgtu]+', row['antisense oligo (5′→3′)']) and row['candidate'] for row in records):
        return ''
    esc = html.escape
    cards = []
    for row in records:
        label = 'ASO ' + row['candidate'] if row['candidate'].isdigit() else row['candidate']
        sequence = row['antisense oligo (5′→3′)']
        fields = [('Binding position in the genome', _position(row['physical binding interval(s), bed0']))]
        fields.extend(screening_fields(row['genomic screening']) or [('Genome search record', row['genomic screening'])])
        if row.get('target genomic strand'):
            fields.append(('Target genomic strand', row['target genomic strand']))
        cards.append('<details class="reader-candidate"><summary><span>' + esc(label)
                     + '</span><code class="reader-aso-sequence">' + esc(sequence) + '</code></summary>'
                     + '<div><dl>' + ''.join('<dt>' + esc(key) + '</dt><dd>' + esc(value) + '</dd>' for key, value in fields)
                     + '</dl></div></details>')
    return ('<div class="reader-candidates"><p class="reader-reading-note">ASO sequences, 5′→3′, in panel order. '
            'Open an ASO for its binding position and search details.</p>' + ''.join(cards) + '</div>')


def prediction_reading_view(section):
    """Group a model table by its recorded candidate, strand and assay conditions.

    Identical estimates can be displayed once; differing values remain separate
    in the complete table. No average or cross-condition range is calculated.
    """
    tables = section_tables(section)
    required = {'candidate', 'aso (5′→3′)', 'reference / context', 'dose (nm) / delivery',
                'predicted inhibition (%)', 'status'}
    if len(tables) != 1 or not required <= set(tables[0]['headers']):
        return ''
    groups = {}
    for values in tables[0]['rows']:
        row = dict(zip(tables[0]['headers'], values))
        if not re.fullmatch('[ACGTUacgtu]+', row['aso (5′→3′)']) or not row['candidate']:
            return ''
        key = (row['candidate'], row['aso (5′→3′)'], row['dose (nm) / delivery'])
        groups.setdefault(key, []).append(row)
    if not 0 < len(groups) <= 24:
        return ''
    cards = []
    esc = html.escape
    for (candidate, sequence, conditions), rows in groups.items():
        estimates = {row['predicted inhibition (%)'] for row in rows}
        statuses = {row['status'] for row in rows}
        estimate = next(iter(estimates))
        try:
            valid_number = math.isfinite(float(estimate))
        except ValueError:
            valid_number = False
        if statuses == {'predicted'} and len(estimates) == 1 and valid_number:
            finding = estimate + '% predicted RNA inhibition'
        elif statuses == {'predicted'}:
            finding = 'Predictions differ across reference contexts.'
        else:
            finding = 'Includes unavailable predictions.'
        label = re.sub(r'^candidate_(\d+)$', r'Candidate \1', candidate)
        conditions = html.unescape(conditions)
        assay = re.fullmatch(r'([\d.]+) / ([^(]+) \(dose (specified|assumed reference scenario); delivery (specified|assumed reference scenario)\)', conditions)
        if assay:
            conditions = assay[1] + ' nM with ' + assay[2].strip().lower()
            if assay[3] == assay[4] == 'assumed reference scenario':
                conditions += '; reference assay conditions assumed'
            else:
                conditions += '; dose ' + assay[3] + '; delivery ' + assay[4]
        count = len(rows)
        # The model's candidate IDs can differ from the panel's display order.
        # Use the strand as the reading label when it is unambiguous here, and
        # retain the exact candidate identity in the complete prediction table.
        unique_sequence = sum(key[1] == sequence for key in groups) == 1
        heading = '<code>' + esc(sequence) + '</code> (5′→3′)'
        if not unique_sequence:
            heading = esc(label) + ' · ' + heading
        cards.append('<div class="reader-prediction"><h3>' + heading + '</h3><p>' + esc(finding) + '</p>'
                     + '<p class="reader-reading-note">Assay conditions: ' + esc(conditions)
                     + '. ' + str(count) + (' RNA reference record.' if count == 1 else ' RNA reference records.') + '</p></div>')
    return ('<div class="reader-predictions"><p>OligoAI activity predictions, grouped by candidate and assay conditions. '
            'Each prediction record identifies the RNA reference used.</p>' + ''.join(cards) + '</div>')
