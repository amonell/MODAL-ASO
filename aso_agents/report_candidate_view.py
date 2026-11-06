"""A sequence-first reading view of the exact selected-panel table.

This module only projects recorded fields. It never infers a sequence strand,
joins molecules by sequence, changes selection order, or fills missing results.
The caller retains the complete source table alongside this view.
"""
from __future__ import annotations

import html
import ast
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
        'unknown': 'The search completion record is missing.',
        'unavailable': 'This report is missing the off-target search result.',
        'failed': 'The off-target search failed.',
    }
    if parts[0] not in statuses:
        return None
    fields = [('Search for other sequences this ASO may bind', statuses[parts[0]])]
    budget, bins = None, set()
    for part in parts[1:]:
        limit = re.fullmatch(r'searched ≤(\d+) mismatch\(es\)', part)
        exact = re.fullmatch(r'exact=(\d+) (.+)', part)
        near = re.fullmatch(r'(\d+)mm=(\d+) recorded hits', part)
        if limit:
            fields.append(('Recorded search limit', f'Up to {limit[1]} base differences'))
            budget = int(limit[1])
        elif exact:
            fields.append(('Exact matches', f'{exact[1]} {exact[2]}'))
            bins.add(0)
        elif near:
            fields.append((f'Matches with {near[1]} base ' + ('difference' if near[1] == '1' else 'differences'), f'{near[2]} recorded hits'))
            bins.add(int(near[1]))
        else:
            return None
    if budget is not None and 0 <= budget <= 3:
        for differences in range(budget + 1):
            if differences not in bins:
                label = 'Exact matches' if differences == 0 else f'Matches with {differences} base ' + ('difference' if differences == 1 else 'differences')
                fields.append((label, 'Count absent from this summary'))
    return fields


MOTIF_LABELS = {
    'lna_383_tgc': 'TGC motif (study in LNA 3–8–3 gapmers)',
    'lna_383_tcc': 'TCC motif (study in LNA 3–8–3 gapmers)',
    'cpg_immune_context': 'CG (CpG) motif',
}


def panel_check_records(section, sections):
    """Attach checks only when the whole ordered panel agrees, with unique strands.

    A shared sequence alone cannot identify molecules with different chemistry.
    Ambiguous, reordered, partial or repeated panels retain separate tables.
    """
    panels = section_tables(section)
    checks = [table for item in sections if item['heading'].casefold() == 'recommended shortlist'
              for table in section_tables(item)]
    if len(panels) != 1 or len(checks) != 1:
        return []
    panel, check = panels[0], checks[0]
    if not {'candidate', 'antisense oligo (5′→3′)'} <= set(panel['headers']) or not {'order', 'aso (5′→3′)'} <= set(check['headers']):
        return []
    rows = [dict(zip(panel['headers'], row)) for row in panel['rows']]
    records = [dict(zip(check['headers'], row)) for row in check['rows']]
    strands = [row['antisense oligo (5′→3′)'] for row in rows]
    if (len(rows) != len(records) or len(set(strands)) != len(strands)
            or any((row['candidate'], row['antisense oligo (5′→3′)']) !=
                   (record['order'], record['aso (5′→3′)']) for row, record in zip(rows, records))):
        return []
    return records


def _motifs(value):
    try:
        flags = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    if isinstance(flags, list) and all(isinstance(flag, str) for flag in flags):
        return '; '.join(MOTIF_LABELS.get(flag, flag) for flag in flags) or 'None flagged'
    return value


def screening_summary(value):
    fields = screening_fields(value)
    if not fields:
        return 'Open for the recorded genome search results.'
    exact = next((text for label, text in fields if label == 'Exact matches' and text != 'Count absent from this summary'), '')
    near = [label + ': ' + text + '.' for label, text in fields
            if label.startswith('Matches with ') and re.fullmatch(r'[1-9]\d* recorded hits', text)]
    return ' '.join([fields[0][1], *(['Exact matches: ' + exact + '.'] if exact else []), *near])


def candidate_reading_view(section, *, checks=()):
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
    for index, row in enumerate(records):
        label = 'ASO ' + row['candidate'] if row['candidate'].isdigit() else row['candidate']
        sequence = row['antisense oligo (5′→3′)']
        fields = [('Binding position in the genome', _position(row['physical binding interval(s), bed0']))]
        if row.get('physical splice element'):
            fields.append(('RNA region covered', row['physical splice element']))
        if row.get('predicted response site') not in {None, '', '{}'}:
            fields.append(('Splice site whose effect was scored', row['predicted response site']))
        fields.extend(screening_fields(row['genomic screening']) or [('Genome search record', row['genomic screening'])])
        if row.get('target genomic strand'):
            fields.append(('Target genomic strand', row['target genomic strand']))
        check = checks[index] if len(checks) == len(records) else {}
        if check.get('motif flags'):
            fields.append(('Sequence motifs to review', _motifs(check['motif flags'])))
        if check.get('evidence gaps'):
            fields.append(('Missing checks', check['evidence gaps']))
        if check.get('review note'):
            fields.append(('Prediction review', check['review note']))
        if check.get('splice safeguards'):
            from .report_values import VALUE_LABELS
            guards = {'passed': 'Within the specified limits', 'failed': 'Outside the specified limits',
                      'unmeasured': 'Measurements unavailable'}
            fields.append(('Checks on other splice sites', guards.get(check['splice safeguards'],
                           VALUE_LABELS.get(check['splice safeguards'], check['splice safeguards']))))
        glance = screening_summary(row['genomic screening'])
        if check.get('review note') and check['review note'] != 'No additional review note':
            glance += ' ' + check['review note']
        elif check.get('status') in {'exploratory', 'Prediction requires review'}:
            glance += ' Splicing predictions require review.'
        if check.get('evidence gaps') and check['evidence gaps'] != 'None recorded':
            glance += ' Missing: ' + check['evidence gaps'].rstrip('.') + '.'
        from .report_values import parse_collection, value_html
        def field_value(key, value):
            if key == 'Splice site whose effect was scored':
                structured = parse_collection(value)
                if structured is not None:
                    return value_html(structured)
            return esc(value)
        cards.append(('<details class="reader-candidate"><summary><span>' + esc(label)
                      + '</span><code class="reader-aso-sequence">' + esc(sequence) + '</code>',
                      glance, '</summary><div><dl>'
                      + ''.join('<dt>' + esc(key) + '</dt><dd>' + field_value(key, value) + '</dd>' for key, value in fields)
                      + '</dl></div></details>'))
    shared = len(cards) > 1 and len({card[1] for card in cards}) == 1
    common = ('<p class="reader-panel-checks"><strong>For each ASO shown:</strong> ' + esc(cards[0][1]) + '</p>') if shared else ''
    rendered = ''.join(start + ('' if shared else '<span class="reader-candidate-check">' + esc(note) + '</span>') + end
                       for start, note, end in cards)
    return ('<div class="reader-candidates"><p class="reader-reading-note">ASO sequences run from 5′ to 3′, in panel order. '
            'Open a sequence for its checks and binding position. Off-targets are other sequences an ASO may also bind.</p>'
            + common + rendered + '</div>')


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
