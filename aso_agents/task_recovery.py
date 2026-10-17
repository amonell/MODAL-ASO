"""Invalidate task results for a restart without discarding prerequisite data."""
from copy import deepcopy


def reset_task_and_dependents(state, task_id):
    graph = state.get('task_graph') or {}
    tasks = {str(t.get('task_id') or ''): t for t in graph.get('tasks') or []
             if isinstance(t, dict) and t.get('task_id')}
    if task_id not in tasks:
        raise ValueError('Cannot restart an unknown task')
    edges = {(str(e.get('upstream_task_id') or ''), str(e.get('downstream_task_id') or ''))
             for e in graph.get('dependencies') or [] if isinstance(e, dict)}
    for tid, task in tasks.items():
        for upstream in task.get('depends_on') or []:
            edges.add((str(upstream), tid))
    affected = {task_id}
    while True:
        expanded = affected | {down for up, down in edges if up in affected and down in tasks}
        if expanded == affected:
            break
        affected = expanded
    statuses = state.setdefault('task_statuses', {})
    artifacts = state.setdefault('task_artifacts', {})
    previous = {tid: {'status': deepcopy(statuses.get(tid)), 'artifacts': deepcopy(artifacts.get(tid))}
                for tid in sorted(affected)}
    for tid in affected:
        statuses[tid] = {'task_type': tasks[tid].get('task_type'),
                         'native': tasks[tid].get('native', True), 'status': 'pending'}
        artifacts.pop(tid, None)
    state.setdefault('task_restart_history', []).append({
        'task_id': task_id, 'invalidated_task_ids': sorted(affected), 'previous_results': previous,
        'policy': 'Prior files retained; invalidated results are not usable until their tasks complete again.'})
    return sorted(affected)
