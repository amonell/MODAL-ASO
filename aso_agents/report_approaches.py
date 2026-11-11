"""Present saved approach rankings within the design jobs that produced them."""
from collections import defaultdict
import html


def project_approaches(goal, contexts, routing=None):
    from .report_explorer_data import collection, number, site_name

    routing = routing or goal.get('strategy_routing') or {}
    by_job = goal.get('strategy_routing_by_job') or {}
    saved_goals = routing.get('goals')
    goals = saved_goals if isinstance(saved_goals, list) else [
        {**item, 'job_id': job_id} for job_id, result in by_job.items()
        for item in result.get('goals') or [] if isinstance(item, dict)]
    context_jobs = {c.get('job') for c in contexts if c.get('job')}
    jobs = {j['job_id']: j for j in [*(goal.get('steric_design_jobs') or []),
                                     *(goal.get('gapmer_design_jobs') or [])]
            if isinstance(j, dict) and j.get('job_id')}
    ranking = (routing.get('planner_plan') or {}).get('ranking') or {}
    rankings = {item['job_id']: item.get('ranking') or {}
                for item in ranking.get('job_rankings') or [] if item.get('job_id')}
    for job_id, result in by_job.items():
        if (result.get('planner_plan') or {}).get('ranking'):
            rankings.setdefault(job_id, result['planner_plan']['ranking'])
    job_ids = context_jobs | set(jobs) | set(by_job) | set(rankings)
    job_ids.update(item['job_id'] for item in goals if isinstance(item, dict) and item.get('job_id'))
    job_ids.update(item['job_id'] for item in routing.get('jobs') or []
                   if isinstance(item, dict) and item.get('job_id'))
    saved_steps = routing.get('steps')
    steps = saved_steps if isinstance(saved_steps, list) else [
        {**step, 'job_id': job_id} for job_id, result in by_job.items()
        for step in result.get('steps') or [] if isinstance(step, dict)]
    groups = {}
    for item in goals:
        if not isinstance(item, dict):
            continue
        job_id = item.get('job_id')
        if not job_id and len(job_ids) == 1:
            job_id = next(iter(job_ids))
        # A jobless historical ranking stays separate when several jobs exist.
        key = job_id or 'unassigned'
        job = jobs.get(job_id) or {}
        policy = rankings.get(job_id) or (ranking if ranking.get('mode') != 'per_job'
                                         and len(job_ids) <= 1 else {})
        group = groups.setdefault(key, {
            'id': key, 'job': job_id,
            'label': item.get('job_label') or job.get('label') or 'Saved approach ranking',
            'summary': policy.get('summary') or '', 'method': policy.get('mode') or '',
            'model': policy.get('model') or item.get('ranking_model') or '', 'approaches': []})
        targets = {}
        for target in item.get('primary_targets') or []:
            if not isinstance(target, (str, dict)):
                continue
            label = target if isinstance(target, str) else target.get('label') or target.get('site_label')
            if label:
                targets[label] = target.get('direction') if isinstance(target, dict) else None
        # The executed step supplies typed directions without interpreting prose.
        matching_steps = [s for s in steps if s.get('step_index') == item.get('step_index')
                          and item.get('step_index') is not None and
                          (s.get('job_id') == job_id or not s.get('job_id') and len(job_ids) <= 1)]
        if len(matching_steps) == 1:
            args = matching_steps[0].get('args') or {}
            specifications = collection(args.get('primary_labels_json')) or args.get('primary_targets') or []
            for spec in specifications:
                if isinstance(spec, dict):
                    label = spec.get('label') or spec.get('site_label')
                    if label in targets:
                        targets[label] = spec.get('direction') or targets[label]
            if args.get('primary_label') in targets:
                targets[args['primary_label']] = args.get('primary_direction') or targets[args['primary_label']]
        matches = [c for c in contexts if job_id and c.get('job') == job_id
                   and c.get('site') in targets and
                   (not targets[c['site']] or c.get('direction') == targets[c['site']])]
        # Unknown directions cannot attach one rationale to opposite objectives.
        ambiguous = {label for label, direction in targets.items() if not direction and
                     len({c.get('direction') for c in matches if c.get('site') == label}) > 1}
        matches = [c for c in matches if c.get('site') not in ambiguous]
        names = {}
        for context in matches:
            target = next((s for s in context.get('sites') or [] if s.get('target')), None)
            names[context['site']] = target['label'] if target else site_name(context['site'], {})
        label = str(item.get('goal_label') or item.get('primary_effect') or 'Recorded approach')
        if targets and all(direction in {'increase', 'decrease'} for direction in targets.values()):
            label = '; '.join(direction.capitalize() + ' use of ' + (names.get(target) or site_name(target, {}))
                              for target, direction in targets.items())
        else:
            for target in sorted(targets, key=len, reverse=True):
                label = label.replace(target, names.get(target) or site_name(target, {}))
        if len(targets) == len(matches) == 1 and matches[0].get('objective'):
            label = matches[0]['objective']
        label = label[:1].upper() + label[1:]
        rank = number(item.get('rank'))
        count = number(item.get('count'))
        group['approaches'].append({
            'rank': int(rank) if rank is not None and rank > 0 and rank.is_integer() else None,
            'label': label, 'rationale': item.get('rationale') or '',
            'confidence': item.get('ranking_confidence') or '',
            'retained_count': int(count) if count is not None and count >= 0 and count.is_integer() else None,
            'context_ids': [c['id'] for c in matches]})
    for group in groups.values():
        group['approaches'].sort(key=lambda a: a['rank'] if a['rank'] is not None else float('inf'))
    return list(groups.values())


def context_approaches(data):
    matched = defaultdict(list)
    for group in data.get('approach_rankings') or []:
        for approach in group['approaches']:
            for context_id in approach['context_ids']:
                matched[context_id].append(approach)
    return matched


def ordered_contexts(data):
    """Follow saved priorities inside each job; preserve job order and ties."""
    matched = context_approaches(data)
    if not matched:
        return list(data['contexts'])
    jobs = defaultdict(list)
    for context in data['contexts']:
        jobs[context.get('job') or context['id']].append(context)
    def rank(context):
        choices = matched[context['id']]
        return choices[0]['rank'] if len(choices) == 1 and choices[0]['rank'] is not None else float('inf')
    return [context for group in jobs.values() for context in sorted(group, key=rank)]


def approach_html(data):
    if data.get('kind') != 'design':
        return ''
    esc = lambda value: html.escape(str(value), quote=True)
    groups = data.get('approach_rankings') or []
    matched = context_approaches(data)
    contexts = data['contexts']
    by_id = {c['id']: c for c in contexts}
    parts = ['<div class="report-approaches" data-role="approaches">']
    for index, context in enumerate(contexts):
        choices = matched[context['id']]
        parts.append('<div class="report-approach-summary" data-approach-context="' + esc(context['id'])
                     + '"' + (' hidden' if index else '') + '>')
        if len(choices) == 1:
            choice = choices[0]
            rank = choice['rank']
            heading = 'Why it ranks first' if rank == 1 else f'Why it has approach rank {rank}' if rank else 'Saved approach reasoning'
            confidence = choice['confidence']
            parts.append('<p class="report-approach-name"><strong>' + esc(choice['label']) + '</strong></p>'
                         '<h3>' + esc(heading) + (' <span class="report-confidence">' + esc(str(confidence).capitalize())
                         + ' confidence</span>' if confidence else '') + '</h3><p>'
                         + esc(choice['rationale'] or 'The reason for this position is unavailable in the saved results.') + '</p>')
        elif choices:
            parts.append('<p>This target appears in several ranked approaches. Their reasons are listed below.</p>')
        else:
            parts.append('<p><strong>' + esc(context['label']) + '</strong></p><p>'
                         + esc('This run evaluated one target and approach.' if len(contexts) == 1 and not groups else
                                     'A comparison rank for this approach is unavailable in the saved results.') + '</p>')
        parts.append('</div>')
    parts.append('<div data-approach-context="all" hidden><p>'
                 + ('Each design objective keeps its own approach ranking below.' if groups else
                    'A comparison ranking across these approaches is unavailable in the saved results.') + '</p></div>')
    if groups:
        count = sum(len(g['approaches']) for g in groups)
        parts.append('<p class="report-sub">Approach ranks compare targets within each design objective. '
                     'Analysis ranks in the ASO table compare candidates within an approach.</p>'
                     '<details class="report-approach-comparison"><summary>'
                     + (f'Compare all {count} ranked approaches' if count > 1 else 'Approach ranking details') + '</summary>')
        for group in groups:
            parts.append('<section class="report-approach-group"><h3>' + esc(group['label']) + '</h3>')
            if group['summary']:
                parts.append('<p class="report-sub">' + esc(group['summary']) + '</p>')
            if group['model']:
                parts.append('<p class="report-sub">Ranking model: ' + esc(group['model']) + '.</p>')
            parts.append('<ol class="report-approach-list">')
            for choice in group['approaches']:
                rank = f'Approach rank {choice["rank"]}' if choice['rank'] is not None else 'Rank unavailable'
                confidence = choice['confidence']
                parts.append('<li><div class="report-approach-title"><strong>' + esc(rank + ' · ' + choice['label'])
                             + '</strong>' + ('<span class="report-confidence">' + esc(str(confidence).capitalize())
                             + ' confidence</span>' if confidence else '') + '</div><p>'
                             + esc(choice['rationale'] or 'The ranking reason is unavailable in the saved results.') + '</p>')
                if choice['retained_count'] is not None:
                    parts.append('<span class="report-sub">' + f'{choice["retained_count"]:,}' + ' candidates retained.</span> ')
                for context_id in choice['context_ids']:
                    label = 'View ASOs' if len(choice['context_ids']) == 1 else 'View ' + by_id[context_id]['label']
                    parts.append('<button type="button" class="report-button" data-approach-select="' + esc(context_id)
                                 + '">' + esc(label) + '</button>')
                parts.append('</li>')
            parts.append('</ol></section>')
        parts.append('</details>')
    parts.append('</div>')
    return ''.join(parts)
