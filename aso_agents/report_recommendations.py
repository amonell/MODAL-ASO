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
    parts = ['<section class="report-recommendations" aria-label="ASO recommendations" data-role="recommendations">',
        '<h2>' + title + '</h2><p>' + esc(review.get('summary')) + '</p>']
    if completed:
        count = review.get('reviewed_molecule_count', 0)
        parts.append('<p class="report-sub">' + f'{len(panel):,} recommended from {count:,} ASOs reviewed.</p>')
    if completed:
        parts.append('<ol class="report-recommendation-list">')
        for rank, row in sorted(panel.items())[:8]:
            item = row.get('recommendation') or {}
            if not item:
                continue
            parts.extend(['<li><div class="report-recommendation-heading"><strong>' + str(rank)
                + '. <span class="report-sequence">' + esc(row['aso']) + '</span></strong>',
                '<span class="report-sub">' + esc(row['chemistry']['label']) + '</span></div>',
                '<p>' + esc(item.get('why_chosen')) + '</p>',
                '<details><summary>Why this ASO belongs in the panel</summary>',
                '<p><strong>Panel role:</strong> ' + esc(item.get('panel_role')) + '</p>',
                '<p><strong>Tradeoff:</strong> ' + esc(item.get('tradeoff')) + '</p>',
                '<p class="report-sub">' + esc(row.get('location')) + '</p>',
                '<details><summary>Evidence behind this choice</summary><dl>'])
            for evidence in (item.get('evidence') or []) + (item.get('comparison_evidence') or []):
                path = evidence.get('path') or ''
                label = ' · '.join(field_label(part.replace('~1','/').replace('~0','~'))
                    for part in path.split('/')[3:] if not part.isdigit())
                if evidence.get('candidate_id') != item.get('candidate_id'):
                    label = 'Comparison with ' + str(evidence.get('antisense_5to3') or 'another assessed ASO') + ' · ' + label
                parts.append('<dt>' + esc(label) + '</dt><dd>'
                             + value_html(evidence.get('value'), path.split('/')[-1]) + '</dd>')
            parts.extend(['</dl></details></details>', '<button type="button" class="report-button" '
                'data-recommendation-pick="' + esc(row['id']) + '">Add ASO ' + str(rank) + ' to basket</button></li>'])
        parts.append('</ol>')
        if panel:
            parts.append('<button type="button" class="report-button" data-action="add-recommendations">Add recommended panel to basket</button>')
            parts.append(' <button type="button" class="report-button" data-action="show-recommendations">Explore recommended ASOs</button>')
        if len(panel) > 8:
            parts.append('<p class="report-sub">The first 8 recommendations are shown here. '
                'Explore the panel to read the reasons and tradeoffs for all ' + str(len(panel)) + ' ASOs.</p>')
        reviewer = review.get('reviewer') or {}
        parts.append('<details><summary>About the recommendation review</summary><p>'
            + esc(review.get('size_reason')) + '</p><p>'
            'The model reviewed the eligible ASOs in groups, compared the leading choices, '
            'and checked each explanation against the saved evidence. Requested counts, regions, '
            'spacing and explicit ranking priorities were applied during selection.</p><p class="report-sub">'
            + esc(reviewer.get('model')) + '</p></details>')
    parts.append('</section>')
    return ''.join(parts)
