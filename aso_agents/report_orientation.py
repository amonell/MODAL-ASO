"""A short introduction for readers, derived only from saved report records.

Counts describe the table that supplies them, never successful fulfillment of
the request. Unrecognized or inconsistent records do not receive guessed facts.
This is display logic; it neither chooses ASOs nor evaluates their biology.
"""
from __future__ import annotations

import html
import re
from collections import Counter


def _cells(line):
    return [cell.strip() for cell in re.split(r'(?<!\\)\|', line.strip()[1:-1])]


def _key(value):
    return html.unescape(value).strip('`* ').casefold()


def section_tables(section):
    """Admit complete rectangular Markdown tables, outside fenced code."""
    lines = section['lines']; tables = []; fence = None; index = 0
    while index < len(lines):
        text = lines[index].strip(); marker = re.match(r'^(`{3,}|~{3,})', text)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            index += 1; continue
        if marker:
            fence = marker[1]; index += 1; continue
        if not (text.startswith('|') and text.endswith('|') and index + 1 < len(lines)):
            index += 1; continue
        divider = lines[index + 1].strip()
        if not (divider.startswith('|') and divider.endswith('|')
                and all(re.fullmatch(r':?-{3,}:?', cell) for cell in _cells(divider))):
            index += 1; continue
        headers = _cells(text); rows = []; valid = len(_cells(divider)) == len(headers); index += 2
        while index < len(lines) and lines[index].strip().startswith('|') and lines[index].strip().endswith('|'):
            row = _cells(lines[index]); valid = valid and len(row) == len(headers)
            rows.append(row); index += 1
        keys = [_key(value) for value in headers]
        if valid and len(set(keys)) == len(keys):
            tables.append({'headers': keys, 'rows': rows})
    return tables


def _fact(label, value, source, note=''):
    return {'label': label, 'value': value, 'source': source, 'note': note}


def _audit_summary(sections, tables):
    audits = [(index, table) for index, table in tables
              if {'sequence', 'recomputed_length', 'retention_status'} <= set(table['headers'])]
    if not audits:
        return None
    summaries = [(i, s) for i, s in enumerate(sections) if _key(s['heading']) == 'outcome summary']
    if not summaries:
        return None
    if len(audits) != 1 or len(summaries) != 1:
        return {'valid': False}
    index, section = summaries[0]; counts = {}
    for line in section['lines']:
        match = re.fullmatch(r'\s*-\s*(Total input rows|Retained|Excluded|Indeterminate):\s*(\d+)\s*', line)
        if match:
            if match[1] in counts:
                return {'valid': False}
            counts[match[1]] = int(match[2])
    names = ['Total input rows', 'Retained', 'Excluded', 'Indeterminate']
    if any(name not in counts for name in names):
        return {'valid': False}
    _, table = audits[0]; status_col = table['headers'].index('retention_status')
    # Both vocabularies occur in saved sequence audits. Translate only this
    # specific retention column, and check the separate retained flag if present.
    labels = {'passed': 'retained', 'failed': 'excluded'}
    statuses = [labels.get(_key(row[status_col]), _key(row[status_col])) for row in table['rows']]
    observed = Counter(statuses)
    flags_agree = True
    if 'retained' in table['headers']:
        retained_col = table['headers'].index('retained')
        flags_agree = all(_key(row[retained_col]) == ('true' if status == 'retained' else 'false')
                          for row, status in zip(table['rows'], statuses) if status != 'indeterminate')
    valid = (counts['Total input rows'] == len(table['rows'])
             and sum(counts[name] for name in names[1:]) == counts['Total input rows']
             and flags_agree
             and set(observed) <= {'retained', 'excluded', 'indeterminate'}
             and all(counts[name] == observed[name.casefold()] for name in names[1:]))
    if not valid:
        return {'valid': False}
    source = f'reader-section-{index}'
    return {'valid': True, 'facts': [
        _fact('Sequences checked', counts['Total input rows'], source),
        _fact('Pass the listed checks', counts['Retained'], source),
        _fact('Do not pass the checks', counts['Excluded'], source),
    ], 'note': (f"{counts['Indeterminate']} could not be classified." if counts['Indeterminate'] else ''), 'source': source}


def _candidate_summary(sections, tables):
    for index, section in enumerate(sections):
        if _key(section['heading']) != 'selected physical panel':
            continue
        count = re.search(r'Selected candidate rows: (\d+)\. Distinct molecules \(antisense sequence plus chemistry\): (\d+)\. Preview: (\d+) molecules in recorded rank order\.', '\n'.join(section['lines']))
        own = [table for i, table in tables if i == index and 'candidate' in table['headers']
               and any(key.startswith('antisense oligo') for key in table['headers'])]
        if not count or len(own) != 1:
            return None
        total, distinct, shown = map(int, count.groups())
        if not (0 <= distinct <= total and shown == len(own[0]['rows']) and shown <= total):
            return None
        other = any(i != index and table['rows'] and any(key.startswith(('aso (', 'antisense oligo')) for key in table['headers']) for i, table in tables)
        return {'total': total, 'shown': shown, 'other_tables': other, 'source': f'reader-section-{index}'}
    return None


def _activity_summary(evaluations):
    # A combined report can repeat evaluations from different contexts. Do not
    # combine their counts or infer a shared denominator.
    if not isinstance(evaluations, (list, tuple)) or len(evaluations) != 1 or not isinstance(evaluations[0], dict):
        return None
    value = evaluations[0]; total = value.get('candidate_count'); predicted = value.get('predicted_count'); rows = value.get('candidates')
    if not (type(total) is int and type(predicted) is int and total > 0 and 0 <= predicted <= total
            and isinstance(rows, list) and len(rows) == total
            and all(isinstance(row, dict) and isinstance(row.get('status'), str) for row in rows)):
        return None
    statuses = Counter(row.get('status') for row in rows)
    if set(statuses) - {'predicted', 'not_predicted', 'unavailable', 'error'} or statuses['predicted'] != predicted:
        return None
    return {'total': total, 'predicted': predicted}


def build_orientation(sections, context=None):
    """Build explainable display facts with links back to their source section."""
    context = context or {}; kind = context.get('kind') or 'analysis'
    tables = [(index, table) for index, section in enumerate(sections) for table in section_tables(section)]
    audit = _audit_summary(sections, tables)
    comparison = next(((index, table) for index, table in tables if
                       {'set', 'sequence', 'pooled_nondominated_frontier'} <= set(table['headers'])), None)
    assessments = [(index, table) for index, table in tables if
                   {'aso', 'recomputed properties', 'assessment'} <= set(table['headers'])]
    if kind == 'analysis' and audit is not None:
        kind = 'sequence_checks'
    elif kind == 'analysis' and comparison is not None:
        kind = 'comparison'
    elif kind == 'analysis' and assessments:
        kind = 'evaluation'
    result = {'kind': kind, 'facts': [], 'notices': [], 'source_links': []}
    if kind == 'design':
        gene = str(context.get('gene') or '').strip()
        result['purpose'] = 'This report contains ASO design results' + (f' for {gene}.' if gene else '.')
        result['meaning'] = 'A candidate is an ASO proposed for further study. Check the prediction details to see whether they support the requested effect. A rank compares entries within one list; it does not measure activity in an experiment.'
        candidate = _candidate_summary(sections, tables)
        if candidate:
            note = 'Entries in the main candidate table.'
            if candidate['shown'] != candidate['total']:
                note = f"{candidate['shown']} are shown here out of {candidate['total']} reported entries."
            result['facts'].append(_fact('Design candidates listed', candidate['shown'], candidate['source'], note))
            result['source_links'].append(('See the candidate table', candidate['source']))
            if candidate['total'] == 0:
                result['notices'].append('The main candidate table is empty.' + (' Other tables contain ASO entries; check those tables and the requested files before choosing a list.' if candidate['other_tables'] else ' Check the detailed results for the reason and any other available outputs.'))
        activity = _activity_summary(context.get('evaluations') or [])
        if activity:
            missing = activity['total'] - activity['predicted']
            source = next((f'reader-section-{i}' for i, section in enumerate(sections)
                           if _key(section['heading']) == 'oligoai gapmer efficacy prediction'), '')
            result['facts'].append(_fact('Activity estimates available', f"{activity['predicted']} of {activity['total']}", source, 'OligoAI model predictions, not measurements.'))
            if missing:
                result['notices'].append(f"{missing} candidate{'s have' if missing != 1 else ' has'} no OligoAI activity estimate. Missing predictions do not mean zero activity.")
    elif kind == 'sequence_checks':
        result['purpose'] = 'This report checks the supplied sequences against the stated sequence rules.'
        result['meaning'] = 'Passing these checks means meeting the rules used in this analysis. It does not establish how well an ASO will work.'
        if audit['valid']:
            result['facts'] = audit['facts']; result['source_links'].append(('See the check results', audit['source']))
            if audit['note']:
                result['notices'].append(audit['note'])
        else:
            result['notices'].append('Use the full sequence results below. These results could not be summarized reliably.')
    elif kind == 'evaluation':
        result['purpose'] = 'This report assesses the supplied ASOs using the available sequence checks.'
        result['meaning'] = 'Sequence matching asks where an ASO could pair with RNA. It does not show that the ASO binds or works in an experiment. An off-target search looks for matches beyond the intended target; read its search limits alongside any count.'
        if len(assessments) == 1:
            index, table = assessments[0]
            source = f'reader-section-{index}'
            result['facts'].append(_fact('Entries assessed', len(table['rows']), source, 'Entries in the assessment table.'))
            result['source_links'].append(('See the assessments', source))
    elif kind == 'comparison':
        result['purpose'] = 'This report compares the supplied ASO entries using the measures provided.'
        result['meaning'] = 'An option can look better on one measure and worse on another. A trade-off option has no alternative in the comparison that is at least as good on every measure and better on at least one.'
        if comparison:
            index, table = comparison; flag = table['headers'].index('pooled_nondominated_frontier'); group = table['headers'].index('set')
            flags = [_key(row[flag]) for row in table['rows']]
            if flags and all(value in {'true', 'false'} for value in flags) and all(row[group] for row in table['rows']):
                source = f'reader-section-{index}'
                result['facts'] = [_fact('Entries shown', len(table['rows']), source),
                                   _fact('Groups shown', len({row[group] for row in table['rows']}), source),
                                   _fact('Trade-off options shown', flags.count('true'), source, 'Called “nondominated” in the full results.')]
                result['source_links'].append(('See the comparison', source))
    elif kind == 'combined':
        result['purpose'] = 'This page brings together several ASO design reports and their shared analysis.'
        result['meaning'] = 'Start with the combined findings. Open an individual design report for its ASO sequences and checks. Ranks within separate reports are not a shared ranking of biological activity.'
        labels = context.get('design_labels') or []
        if labels:
            result['facts'].append(_fact('Design reports available', len(labels), '', 'Open the named report tabs to explore each design.'))
    else:
        result['purpose'] = 'This report presents the results of an ASO analysis.'
        result['meaning'] = 'Start with the findings and the questions that remain open. The detailed tables show which conclusions the supplied data support.'
    result['basics'] = 'An antisense oligonucleotide (ASO) is a short synthetic strand designed to bind RNA. The letters in an ASO sequence describe that strand.'
    return result


def orientation_html(orientation, *, findings, downloads=None, limitations=None):
    esc = html.escape
    cards = ''.join('<div class="reader-fact"><span class="reader-fact-value">' + esc(str(fact['value']))
                    + '</span><span class="reader-fact-label">' + esc(fact['label']) + '</span>'
                    + ('<span class="reader-fact-note">' + esc(fact['note']) + '</span>' if fact['note'] else '')
                    + ('<a class="reader-fact-source" href="#' + esc(fact['source'], quote=True) + '">See these results</a>' if fact['source'] else '')
                    + '</div>' for fact in orientation['facts'])
    notices = ('<div class="reader-attention"><h3>What needs your attention</h3><ul>'
               + ''.join('<li>' + esc(note) + '</li>' for note in orientation['notices']) + '</ul></div>') if orientation['notices'] else ''
    actions = '<a class="reader-action" href="#' + esc(findings, quote=True) + '">Read the findings</a>'
    if limitations:
        actions += '<a href="#' + esc(limitations, quote=True) + '">See limitations</a>'
    if downloads:
        actions += '<a href="#' + esc(downloads, quote=True) + '">Get the files</a>'
    return ('<section class="reader-orientation" aria-labelledby="reader-start"><p class="reader-eyebrow">Start here</p>'
            + '<h2 id="reader-start">The short version</h2><p class="reader-purpose">' + esc(orientation['purpose']) + '</p>'
            + ('<div class="reader-facts">' + cards + '</div>' if cards else '')
            + notices + '<h3>How to understand the result</h3><p>' + esc(orientation['meaning']) + '</p>'
            + '<details class="reader-basics"><summary>What is an ASO?</summary><p>' + esc(orientation['basics']) + '</p></details>'
            + '<nav class="reader-actions" aria-label="Start reading">' + actions + '</nav></section>')
