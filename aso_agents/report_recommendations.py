"""A compact, progressively disclosed view of the saved recommendation review."""
import html

from .report_values import field_label, value_html


def recommendations_html(data):
    review = data.get('recommendation_review') or {}
    if not review:
        return ''
    esc = lambda value: html.escape(str(value or ''), quote=True)
    completed = review.get('status') == 'completed'
    title = 'Recommended ASOs' if completed else 'Provisional panel'
    if review.get('status') == 'no_eligible_candidates':
        title = 'Recommendation outcome'
    panel = {}
    for row in data['records']:
        if row.get('panel_rank') is not None:
            panel.setdefault(row['panel_rank'], row)
    parts = ['<section class="report-recommendations" aria-label="ASO recommendations" data-role="recommendations" '
        'id="report-' + esc(data['id']) + '-recommendations" tabindex="-1">',
        '<h2>' + title + '</h2><p>' + esc(review.get('summary')) + '</p>']
    if completed:
        coverage = review.get('review_coverage') or {}
        if coverage:
            count = coverage.get('indexed_molecule_count', review.get('eligible_molecule_count', 0))
            parts.append('<p class="report-sub">' + f'{len(panel):,} recommended from {count:,} eligible ASOs.</p>')
        else:
            count = review.get('reviewed_molecule_count', 0)
            parts.append('<p class="report-sub">' + f'{len(panel):,} recommended from {count:,} ASOs reviewed.</p>')
    if panel:
        parts.append('<ol class="report-recommendation-list">')
        for rank, row in sorted(panel.items())[:8]:
            item = (row.get('recommendation') or {}) if completed else {}
            parts.extend(['<li><div class="report-recommendation-heading"><strong>' + str(rank)
                + '. <span class="report-sequence">' + esc(row['aso']) + '</span></strong>',
                '<span class="report-sub">' + esc(row['chemistry']['label']) + '</span></div>',
                '<p>' + esc(item.get('why_chosen') if completed else
                    'Provisional ASO selected using the saved analysis ranking and panel constraints.') + '</p>',
                '<details><summary>' + ('Why this ASO belongs in the panel' if completed else 'Recorded evidence') + '</summary>',
                '<p><strong>Panel role:</strong> ' + esc(item.get('panel_role') if completed else 'Awaiting recommendation review') + '</p>',
                '<p><strong>Tradeoff:</strong> ' + esc(item.get('tradeoff') if completed else (row.get('guard') or {}).get('summary')) + '</p>',
                '<p class="report-sub">' + esc(row.get('location')) + '</p>'])
            citations = (item.get('evidence') or []) + (item.get('comparison_evidence') or [])
            if citations:
                parts.append('<details><summary>Evidence behind this choice</summary><dl>')
            for evidence in citations:
                path = evidence.get('path') or ''
                label = ' · '.join(field_label(part.replace('~1','/').replace('~0','~'))
                    for part in path.split('/')[3:] if not part.isdigit())
                if evidence.get('candidate_id') != item.get('candidate_id'):
                    label = 'Comparison with ' + str(evidence.get('antisense_5to3') or 'another assessed ASO') + ' · ' + label
                parts.append('<dt>' + esc(label) + '</dt><dd>'
                             + value_html(evidence.get('value'), path.split('/')[-1]) + '</dd>')
            if citations:
                parts.append('</dl></details>')
            parts.extend(['</details>', '<button type="button" class="report-button" '
                'data-recommendation-pick="' + esc(row['id']) + '">Add ASO ' + str(rank) + ' to basket</button></li>'])
        parts.append('</ol>')
        if panel:
            parts.append('<button type="button" class="report-button" data-action="add-recommendations">Add '
                         + ('recommended' if completed else 'provisional') + ' panel to basket</button>')
            parts.append(' <button type="button" class="report-button" data-action="show-recommendations">Explore '
                         + ('recommended' if completed else 'provisional') + ' ASOs</button>')
        if len(panel) > 8:
            parts.append('<p class="report-sub">The first 8 panel entries are shown here. '
                'Explore the panel to read the reasons and tradeoffs for all ' + str(len(panel)) + ' ASOs.</p>')
    if not completed and panel:
        parts.append('<p class="report-sub">' + str(len(panel)) + ' provisional ASOs retained. '
                     'Their sequence, chemistry and saved checks are available in the ASO table.</p>')
        draft_count = review.get('draft_molecule_count')
        if type(draft_count) is int and draft_count > 0:
            parts.append('<p>The ' + str(draft_count) + '-ASO review draft remains unapproved. '
                         'The provisional panel follows the saved ranking and requested selection constraints.</p>')
    if completed:
        reviewer = review.get('reviewer') or {}
        process = review.get('selection_process') or {}
        parts.append('<details><summary>How these ASOs were chosen</summary><p>'
            + esc(review.get('size_reason')) + '</p>')
        if process:
            parts.append('<p>' + esc(process.get('summary')) + '</p>')
            coverage = review.get('review_coverage') or {}
            labels = [('indexed_molecule_count', 'ASOs included in the library survey'),
                      ('matched_molecule_count', 'ASOs matching the chosen filters'),
                      ('compared_molecule_count', 'ASOs individually compared'),
                      ('fully_reviewed_molecule_count', 'ASOs with complete evidence reviewed')]
            parts.append('<dl>')
            for key, label in labels:
                count = coverage.get(key)
                if type(count) is int and count >= 0:
                    parts.append('<dt>' + label + '</dt><dd>' + f'{count:,}' + '</dd>')
            parts.append('</dl>')
            steps = process.get('steps') or []
            if steps:
                parts.append('<details><summary>Selection criteria and alternatives</summary><ol>')
                for step in steps:
                    parts.append('<li><p>' + esc(step.get('rationale')) + '</p>')
                    counts = []
                    for key, label in [('matched_count', 'matched'), ('unknown_count', 'had unavailable measurements')]:
                        count = step.get(key)
                        if type(count) is int and count >= 0:
                            counts.append(f'{count:,} ASOs ' + label)
                    if counts:
                        parts.append('<p class="report-sub">' + '; '.join(counts) + '.</p>')
                    parts.append('</li>')
                parts.append('</ol></details>')
        else:
            parts.append('<p>The model reviewed the eligible ASOs in groups, compared the leading choices, '
                'and checked each explanation against the saved evidence. Requested counts, regions, '
                'spacing and explicit ranking priorities were applied during selection.</p>')
        parts.append('<p class="report-sub">' + esc(reviewer.get('model')) + '</p></details>')
    parts.append('</section>')
    return ''.join(parts)
