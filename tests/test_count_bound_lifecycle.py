"""Count bounds retain their meaning through audit, legacy parsing and save."""
import asyncio
from copy import deepcopy
import json

import pytest

import run_goal_intake as intake
from aso_agents import pipeline_tools as pipeline
from aso_agents.shortlist_clarification import size_question


@pytest.mark.parametrize('request_text, expected', [
    ('Return at least 7 candidates per site.', {'min_asos_per_site': 7}),
    ('For each window, return no fewer than 7 ASOs.', {'min_asos_per_site': 7}),
    ('Return exactly 7 candidates per site.', {'min_asos_per_site': 7, 'candidate_limit_per_site': 7}),
    ('Return up to 7 oligos for each route.', {'candidate_limit_per_site': 7}),
    ('For each site, at most 7 candidates.', {'candidate_limit_per_site': 7}),
    ('Return at least 7 candidates.', {'candidate_minimum': 7}),
    ('Return exactly 7 candidates.', {'candidate_minimum': 7, 'candidate_limit': 7}),
    ('Select no more than 7 ASOs.', {'candidate_limit': 7}),
    ('Use at least 18 nucleotides. Return 7 candidates per site.',
     {'min_asos_per_site': 7, 'candidate_limit_per_site': 7}),
])
def test_literal_bound_stays_with_its_count(request_text, expected):
    goal = {'design_programs': [{'program_id': 'panel', 'modality': 'steric',
             'region_constraints': [{'splice_relative_window': {'anchor_site': 'acceptor'}}]}]}
    final = {'structured_goal': goal}
    assert pipeline._apply_candidate_count_from_request_text_to_goal({'request': request_text}, goal, final)
    actual = {k: v for k, v in goal.items() if k != 'design_programs'}
    assert actual == expected
    for key, value in expected.items():
        assert final[key] == value
    if not any('limit' in key for key in expected):
        assert 'candidate_limit' not in goal['design_programs'][0]
        assert size_question(goal, True)['kind'] == 'shortlist_size'


@pytest.mark.parametrize('per_site', [False, True])
def test_audited_minimum_survives_fallback_and_persisted_goal(monkeypatch, tmp_path, per_site):
    request = 'Return at least 7 ASOs' + (' per site.' if per_site else '.')
    count = {'count': 7, 'scope': 'per_site' if per_site else 'global',
             'kind': 'minimum', 'quote': request}
    async def respond(*args, **kwargs):
        return type('Response', (), {'final_output': json.dumps({'candidate_count': count})})()
    monkeypatch.setattr(intake.Runner, 'run', respond)
    draft = {'status': 'ready', 'structured_goal': {'design_programs': [{'program_id': 'panel'}],
        'candidate_limit_per_site' if per_site else 'candidate_limit': 7}}
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    goal = result['structured_goal']
    before = deepcopy(goal)
    assert not pipeline._apply_candidate_count_from_request_text_to_goal({'request': request}, goal, result)
    assert goal == before
    pipeline._save_goal_payload(str(tmp_path), {'request': request}, goal, result)
    stored = json.loads(next(tmp_path.glob('**/structured_goal.json')).read_text())
    assert stored['min_asos_per_site' if per_site else 'candidate_minimum'] == 7
    assert not stored.get('candidate_limit') and not stored.get('candidate_limit_per_site')
    assert size_question(stored, True)['restart_stage'] == 'report'


def test_scoped_audit_keeps_global_exact_and_per_site_minimum_independent():
    goal = {'candidate_limit': 11, 'candidate_minimum': 11, 'min_asos_per_site': 3}
    final = {'structured_goal': goal, 'request_obligation_audit': {'status': 'accepted',
        'candidate_count': {'count': 11, 'scope': 'global', 'kind': 'exact'},
        'per_site_count': {'count': 3, 'kind': 'minimum'}}}
    before = deepcopy(goal)
    assert not pipeline._apply_candidate_count_from_request_text_to_goal(
        {'request': 'Return 11 ASOs total with at least 3 ASOs per site.'}, goal, final)
    assert goal == before
    assert size_question(goal, True) is None
