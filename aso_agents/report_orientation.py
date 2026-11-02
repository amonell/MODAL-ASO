"""A short introduction for readers, derived only from saved report records.

Counts describe the table that supplies them, never successful fulfillment of
the request. Unrecognized or inconsistent records do not receive guessed facts.
This is display logic; it neither chooses ASOs nor evaluates their biology.
"""
from __future__ import annotations

import html
import math
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
        count = re.search(r'Selected candidate rows: (\d+)\. Distinct molecules \(antisense sequence plus chemistry\): (\d+)\. Preview: (\d+) molecules in (?:recorded rank|requested role|panel) order\.', '\n'.join(section['lines']))
        own = [table for i, table in tables if i == index and 'candidate' in table['headers']
               and any(key.startswith('antisense oligo') for key in table['headers'])]
        if not count or len(own) != 1:
            return None
        total, distinct, shown = map(int, count.groups())
        if not (0 <= distinct <= total and shown == len(own[0]['rows']) and shown <= total):
            return None
        other = any(i != index and table['rows'] and any(key.startswith(('aso (', 'antisense oligo')) for key in table['headers']) for i, table in tables)
        unranked = 0
        if 'native rank / scope' in own[0]['headers']:
            rank_col = own[0]['headers'].index('native rank / scope')
            unranked = sum(_key(re.split(r'\s+/\s+', row[rank_col], maxsplit=1)[0]) in
                           {'unavailable', 'not recorded'} for row in own[0]['rows'])
        return {'total': total, 'distinct': distinct, 'shown': shown, 'other_tables': other, 'source': f'reader-section-{index}',
                'unranked': unranked}
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
    summary = {'total': total, 'predicted': predicted}
    conditions = value.get('conditions') or {}
    if not isinstance(conditions, dict):
        return summary
    contexts = [c for row in rows for c in row.get('contexts') or [] if isinstance(c, dict)]
    if (predicted == total and all(row.get('contexts') and row.get('conditions') == conditions for row in rows)
            and contexts and all(c.get('context_basis') == 'genomic_pre_mrna' for c in contexts)):
        estimates = [c.get('predicted_inhibition_percent') for c in contexts]
        dose = conditions.get('dosage_nm'); method = conditions.get('transfection_method')
        if (all(type(n) in (int, float) and math.isfinite(n) for n in estimates)
                and type(dose) in (int, float) and math.isfinite(dose) and dose > 0 and method):
            assumption = 'assuming' if 'assumed_reference_scenario' in (conditions.get('dose_basis'), conditions.get('method_basis')) else 'at'
            summary['prediction'] = f"OligoAI predicts {min(estimates):.0f}–{max(estimates):.0f}% RNA inhibition, {assumption} {dose:g} nM with {str(method).lower()}."
    return summary


def _recorded_cautions(sections, tables):
    """Surface known missing-output and search-limit records before folded data.

    No general keyword sentiment analysis: prose and unrecognized records keep
    their original interpretation. Prediction rows may repeat molecules across
    contexts, so never turn the number of error rows into a candidate count.
    """
    cautions = []
    for index, section in enumerate(sections):
        key = _key(section['heading']); source = f'reader-section-{index}'
        if key == 'requested output files':
            missing = sum(bool(re.fullmatch(r'- .+: file unavailable in this report bundle\.', line.strip()))
                          for line in section['lines'])
            if missing:
                cautions.append((f"{missing} requested file{'s are' if missing != 1 else ' is'} unavailable in this report bundle.", source))
        elif key == 'recommended shortlist':
            own = [table for i, table in tables if i == index and 'off-target screen' in table['headers']]
            if len(own) == 1:
                table = own[0]; col = table['headers'].index('off-target screen')
                incomplete = sum(_key(row[col]) == 'incomplete_or_unavailable' for row in table['rows'])
                if incomplete:
                    affected = f'all {incomplete} ASOs' if incomplete == len(table['rows']) else f"{incomplete} of {len(table['rows'])} ASOs"
                    cautions.append((f"Off-targets are other sequences an ASO may also bind. The search is incomplete for {affected}.", source))
        elif key == 'oligoai gapmer efficacy prediction':
            own = [table for i, table in tables if i == index and
                   {'candidate', 'status', 'predicted inhibition (%)'} <= set(table['headers'])]
            statuses = [row[table['headers'].index('status')] for table in own for row in table['rows']]
            if any(value.startswith('error:') for value in statuses):
                if all(value.startswith('error:') for value in statuses):
                    note = 'Activity prediction failed for every entry in the linked table.'
                    if all('OligoAI runtime is not installed' in value for value in statuses):
                        note += ' OligoAI was not installed.'
                else:
                    note = 'Some activity predictions failed; see the affected entries in the linked table.'
                cautions.append((note, source))
    # Repeated prediction sections stay separate below; one identical alert is
    # enough here. No results or scientific limitations are removed.
    unique = {}
    for text, source in cautions:
        unique.setdefault(text, source)
    return list(unique.items())


def design_reader_context(goal, routing=None):
    """Carry typed intent into the view; never infer a goal from a gene name."""
    goal = goal if isinstance(goal, dict) else {}
    programs = [p for p in goal.get('design_programs') or [] if isinstance(p, dict)]
    fields = ('intent', 'desired_effect', 'exon_label', 'target_region')
    context = {}
    for field in fields:
        values = {str(p[field]) for p in programs if p.get(field) is not None}
        if len(values) == 1:
            context[field] = next(iter(values))
        elif not values and goal.get(field) is not None:
            context[field] = str(goal[field])
    context['routes'] = [row for row in (routing or {}).get('goals') or [] if isinstance(row, dict)]
    return context


def _design_purpose(context, gene):
    intent = context.get('intent'); effect = context.get('desired_effect')
    target = gene or 'the target'
    exon = str(context.get('exon_label') or '').strip()
    exon_name = f'exon {exon}' if exon else 'the requested exon'
    if intent == 'knockdown':
        return f'Designed to lower {target} RNA.'
    if intent == 'binding' and context.get('target_region') == 'translation_start':
        return f'Designed to block protein production by binding near the start of the {target} coding sequence.'
    if intent == 'splice_modulation' and effect in {'inclusion', 'exon_inclusion'}:
        return f'Designed to help {target} RNA retain {exon_name} during splicing.'
    if intent == 'splice_modulation' and effect in {'skipping', 'exon_skipping'}:
        return f'Designed to make {target} RNA skip {exon_name} during splicing.'
    if intent == 'splice_modulation':
        return f'Designed to change how {target} RNA is spliced.'
    if intent == 'binding':
        return f'Designed to bind the requested sites in {target} RNA.'
    return f'ASO designs for {target}.'


def build_orientation(sections, context=None):
    """Build explainable display facts with links back to their source section."""
    context = context or {}; kind = context.get('kind') or 'analysis'
    tables = [(index, table) for index, section in enumerate(sections) for table in section_tables(section)]
    audit = _audit_summary(sections, tables)
    comparison = next(((index, table) for index, table in tables if
                       {'set', 'sequence', 'pooled_nondominated_frontier'} <= set(table['headers'])), None)
    assessments = [(index, table) for index, table in tables if
                   ({'aso', 'recomputed properties', 'assessment'} <= set(table['headers'])
                    or {'aso', 'evaluated molecule (5′→3′)', 'exact antisense records', 'screening status'} <= set(table['headers']))]
    if kind == 'analysis' and audit is not None:
        kind = 'sequence_checks'
    elif kind == 'analysis' and comparison is not None:
        kind = 'comparison'
    elif kind == 'analysis' and assessments:
        kind = 'evaluation'
    result = {'kind': kind, 'facts': [], 'notices': [], 'source_links': []}
    if kind == 'design':
        gene = str(context.get('gene') or '').strip()
        result['purpose'] = _design_purpose(context, gene)
        result['meaning'] = ''
        candidate = _candidate_summary(sections, tables)
        if candidate:
            result['title'] = f"{candidate['distinct']} ASOs" + (f' for {gene}' if gene else '')
            note = ''
            if candidate['shown'] != candidate['total']:
                note = f"The table previews {candidate['shown']} of {candidate['total']} entries."
            result['meaning'] = note
            result['source_links'].append(('View the ASOs', candidate['source']))
            for index, section in enumerate(sections):
                if _key(section['heading']) != 'recommended shortlist':
                    continue
                match = re.search(r'(?m)^(\d+) distinct molecules selected; (\d+) in the assessed eligible pool\.$', '\n'.join(section['lines']))
                own = [t for i, t in tables if i == index and {'order', 'off-target screen'} <= set(t['headers'])]
                if (match and len(own) == 1 and int(match[1]) == len(own[0]['rows']) == candidate['distinct']
                        and int(match[2]) >= candidate['distinct']):
                    result['meaning'] = f"Selected from {int(match[2]):,} designs that passed the filters." + (f' {note}' if note else '')
            if candidate['total'] == 0:
                result['notices'].append('The main candidate table is empty.' + (' Other tables contain ASO entries; check those tables and the requested files before choosing a list.' if candidate['other_tables'] else ' Check the detailed results for the reason and any other available outputs.'))
        activity = _activity_summary(context.get('evaluations') or [])
        if activity:
            missing = activity['total'] - activity['predicted']
            source = next((f'reader-section-{i}' for i, section in enumerate(sections)
                           if _key(section['heading']) == 'oligoai gapmer efficacy prediction'), '')
            if missing:
                result['notices'].append(f"OligoAI activity predictions are unavailable for {missing} of {activity['total']} ASOs.")
            elif source and activity.get('prediction'):
                result['meaning'] = ' '.join(part for part in (result['meaning'], activity['prediction']) if part)
        low_routes = [r for r in context.get('routes') or [] if r.get('rank') == 1 and r.get('ranking_confidence') == 'low']
        if low_routes:
            notice = 'Confidence in the leading splicing approach is low.'
            routes = [(i, t) for i, t in tables if _key(sections[i]['heading']) == 'predicted splice-response candidate sets'
                      and {'rank', 'directional delta', 'opposite delta', 'confidence'} <= set(t['headers'])]
            if len(routes) == 1:
                _, table = routes[0]; keys = table['headers']
                leading = [dict(zip(keys, row)) for row in table['rows'] if row[keys.index('rank')] == '1']
                if len(leading) == 1 and leading[0]['confidence'] == 'low':
                    try:
                        directional, opposite = (float(leading[0][field]) for field in ('directional delta', 'opposite delta'))
                    except (ValueError, TypeError):
                        pass
                    else:
                        if math.isfinite(opposite) and math.isfinite(directional) and opposite > directional >= 0:
                            notice = 'Across the tested designs, the strongest splice-site prediction favors the opposite direction. Confidence in the chosen approach is low.'
            result['notices'].append(notice)
            source = next((f'reader-section-{i}' for i, s in enumerate(sections) if _key(s['heading']) == 'practical interpretation'), '')
            result.setdefault('notice_sources', {})[result['notices'][-1]] = source
    elif kind == 'sequence_checks':
        result['purpose'] = 'This report checks the supplied sequences against the stated sequence rules.'
        result['meaning'] = 'The table shows which sequences meet each rule and which require review.'
        if audit['valid']:
            result['facts'] = audit['facts']; result['source_links'].append(('See the check results', audit['source']))
            if audit['note']:
                result['notices'].append(audit['note'])
        else:
            result['notices'].append('Use the full sequence results below. These results could not be summarized reliably.')
    elif kind == 'evaluation':
        result['purpose'] = 'This report assesses the supplied ASOs using the available sequence checks.'
        result['meaning'] = 'Sequence matching finds places an ASO may bind. Off-targets are other sequences it may also bind.'
        if len(assessments) == 1:
            index, table = assessments[0]
            source = f'reader-section-{index}'
            result['facts'].append(_fact('Entries assessed', len(table['rows']), source, 'Entries in the assessment table.'))
            result['source_links'].append(('See the assessments', source))
            if 'exact antisense records' in table['headers'] and table['rows']:
                col = table['headers'].index('exact antisense records')
                counts = [row[col] for row in table['rows']]
                if all(re.fullmatch(r'\d+', count) for count in counts):
                    matched = sum(int(count) > 0 for count in counts)
                    result['facts'].append(_fact('Entries with recorded exact matches', f'{matched} of {len(counts)}',
                                                source, 'Within the references represented in this table.'))
                    if matched < len(counts):
                        missing = len(counts) - matched
                        result['notices'].append(f"{missing} {'entry has' if missing == 1 else 'entries have'} no recorded exact match. Read the assessments for search status and any near matches.")
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
        result['meaning'] = 'Open a design report for its ASO sequences, predictions and checks.'
        labels = context.get('design_labels') or []
        if labels:
            result['facts'].append(_fact('Design reports available', len(labels), '', 'Open the named report tabs to explore each design.'))
    else:
        result['purpose'] = 'This report presents the results of an ASO analysis.'
        result['meaning'] = 'Start with the findings and the questions that remain open. The detailed tables show which conclusions the supplied data support.'
    result['basics'] = 'An antisense oligonucleotide (ASO) is a short synthetic strand designed to bind RNA.'
    result.setdefault('notice_sources', {})
    for text, source in _recorded_cautions(sections, tables):
        result['notices'].append(text)
        result['notice_sources'][text] = source
    return result


def orientation_html(orientation, *, findings, downloads=None, limitations=None):
    esc = html.escape
    cards = ''.join('<div class="reader-fact"><span class="reader-fact-value">' + esc(str(fact['value']))
                    + '</span><span class="reader-fact-label">' + esc(fact['label']) + '</span>'
                    + ('<span class="reader-fact-note">' + esc(fact['note']) + '</span>' if fact['note'] else '')
                    + ('<a class="reader-fact-source" href="#' + esc(fact['source'], quote=True) + '">See these results</a>' if fact['source'] else '')
                    + '</div>' for fact in orientation['facts'])
    notices = ('<div class="reader-attention"><h3>Limitations</h3><ul>'
               + ''.join('<li>' + esc(note) + (' <a href="#' + esc(orientation['notice_sources'][note], quote=True)
                   + '">Details</a>' if note in orientation.get('notice_sources', {}) else '') + '</li>'
                   for note in orientation['notices']) + '</ul></div>') if orientation['notices'] else ''
    first_link = next(iter(orientation.get('source_links') or []), ('Explore the results', findings))
    actions = '<a class="reader-action" href="#' + esc(first_link[1], quote=True) + '">' + esc(first_link[0]) + '</a>'
    if first_link[1] != findings and orientation['kind'] != 'design':
        actions += '<a href="#' + esc(findings, quote=True) + '">All results</a>'
    if limitations:
        actions += '<a href="#' + esc(limitations, quote=True) + '">See limitations</a>'
    if downloads:
        actions += '<a href="#' + esc(downloads, quote=True) + '">Get the files</a>'
    return ('<section class="reader-orientation" aria-labelledby="reader-start">'
            + '<h2 id="reader-start">' + esc(orientation.get('title', 'Results')) + '</h2><p class="reader-purpose">' + esc(orientation['purpose']) + '</p>'
            + ('<div class="reader-facts">' + cards + '</div>' if cards else '')
            + ('<p>' + esc(orientation['meaning']) + '</p>' if orientation['meaning'] else '') + notices
            + '<nav class="reader-actions" aria-label="Start reading">' + actions + '</nav></section>')
