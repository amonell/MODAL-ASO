"""Share saved recommendation status across the terminal, GUI and report."""
from __future__ import annotations

import json
from pathlib import Path


def review_notice(review: dict, count: int | None = None) -> str:
    status = review.get('status')
    if status in {None, 'completed', 'no_eligible_candidates'}:
        return ''
    panel = (f' {count} provisional ASO{" is" if count == 1 else "s are"} available in the report.' if count else
             ' The report contains the available candidate evidence.')
    if status == 'skipped':
        return 'Recommendation review was skipped at your request.' + panel
    return 'Recommendation review remains incomplete.' + panel


def completion_review(run_dir, artifacts=None) -> dict:
    """Read a declared shortlist or its canonical/legacy run location."""
    from .run_layout import resolve_run_file
    declared = (artifacts or {}).get('recommended_shortlist_json')
    path = Path(declared) if declared else resolve_run_file(run_dir, 'recommended_shortlist.json') if run_dir else None
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {'status': 'unavailable', 'completion_notice': 'Recommendation results could not be read. Open the report for the saved analysis.'}
    if not isinstance(payload, dict):
        return {'status': 'unavailable', 'completion_notice': 'Recommendation results could not be read. Open the report for the saved analysis.'}
    review = payload.get('review') or {}
    if not isinstance(review, dict) or not review:
        return {}
    records = payload.get('records')
    count = len(records) if isinstance(records, list) else None
    return {**{key: review[key] for key in ('status', 'failure_code', 'draft_molecule_count',
                'reviewed_molecule_count', 'eligible_molecule_count') if key in review},
            'panel_count': count, 'completion_notice': review_notice(review, count)}
