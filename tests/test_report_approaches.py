"""Approach explanations retain ranking scope, direction and missing evidence."""
from copy import deepcopy
import json

from aso_agents.report_approaches import approach_html, ordered_contexts, project_approaches
from aso_agents.report_explorer import explorer_html
from aso_agents.report_explorer_data import project_run
from test_report_explorer import Packet, run_fixture, save


def context(key, job='j', site='acceptor@110', direction='increase'):
    return {'id': key, 'job': job, 'site': site, 'direction': direction,
            'label': key, 'sites': [{'target': True, 'label': 'Exon 2 acceptor'}]}


def ranked(rank, job='j', site='acceptor@110', reason='Recorded tradeoff.', **extra):
    return {'job_id': job, 'rank': rank, 'goal_label': 'increase ' + site,
            'primary_targets': [site], 'rationale': reason, 'ranking_confidence': 'low', **extra}


def test_same_site_in_separate_jobs_keeps_its_own_rank_and_reason():
    contexts = [context('a2'), context('a1', site='donor@120'), context('b1', job='k')]
    routing = {'goals': [ranked(2), ranked(1, site='donor@120'), ranked(1, job='k', reason='Other objective.')],
               'planner_plan': {'ranking': {'mode': 'per_job', 'job_rankings': [
                   {'job_id': 'j', 'ranking': {'summary': 'First objective.', 'model': 'model-a'}},
                   {'job_id': 'k', 'ranking': {'summary': 'Second objective.', 'model': 'model-b'}}]}}}
    before = deepcopy(routing)
    groups = project_approaches({}, contexts, routing)
    assert [g['summary'] for g in groups] == ['First objective.', 'Second objective.']
    assert groups[1]['approaches'][0]['context_ids'] == ['b1']
    assert groups[1]['approaches'][0]['rationale'] == 'Other objective.'
    assert [c['id'] for c in ordered_contexts({'contexts': contexts, 'approach_rankings': groups})] == ['a1','a2','b1']
    assert routing == before


def test_jobless_ranking_does_not_attach_to_one_of_several_jobs():
    route = ranked(1); route.pop('job_id')
    groups = project_approaches({}, [context('a'), context('b', job='k')], {'goals': [route]})
    assert groups[0]['job'] is None
    assert groups[0]['approaches'][0]['context_ids'] == []
    # A second job with no candidate pool still makes a jobless join ambiguous.
    groups = project_approaches({'steric_design_jobs':[{'job_id':'j'},{'job_id':'k'}]},
                               [context('a')], {'goals':[route]})
    assert groups[0]['approaches'][0]['context_ids'] == []


def test_executed_site_direction_controls_the_join():
    contexts = [context('up'), context('down', direction='decrease')]
    routing = {'goals': [ranked(1, step_index=0)], 'steps': [{'job_id': 'j', 'step_index': 0,
        'args': {'primary_labels_json': json.dumps([{'label':'acceptor@110','direction':'decrease'}])}}]}
    groups = project_approaches({'desired_delta':'increase'}, contexts, routing)
    assert groups[0]['approaches'][0]['context_ids'] == ['down']
    routing.pop('steps')
    assert project_approaches({}, contexts, routing)[0]['approaches'][0]['context_ids'] == []


def test_empty_approaches_and_all_reasons_remain_available():
    groups = project_approaches({}, [context('a')], {'goals': [
        ranked(i, site=f'donor@{200+i}', reason=f'Reason {i}.', count=0) for i in range(1,13)]})
    document = approach_html({'kind':'design','contexts':[context('a')], 'approach_rankings':groups})
    assert 'Compare all 12 ranked approaches' in document and 'Reason 12.' in document
    assert document.count('0 candidates retained.') == 12
    assert 'data-approach-select=' not in document
    assert 'This run evaluated one target and approach.' not in document


def test_current_summary_takes_precedence_over_older_embedded_planning():
    goal = {'strategy_routing_by_job':{'j':{'goals':[ranked(1)],
        'planner_plan':{'ranking':{'summary':'Older ordering.'}}}}}
    assert project_approaches(goal,[context('a')],{'goals':[]}) == []
    routing = {'goals':[ranked(2)], 'planner_plan':{'ranking':{'mode':'per_job',
        'job_rankings':[{'job_id':'j','ranking':{'summary':'Current ordering.'}}]}}}
    groups = project_approaches(goal,[context('a')],routing)
    assert groups[0]['summary'] == 'Current ordering.'
    assert groups[0]['approaches'][0]['rank'] == 2


def test_missing_rank_or_rationale_are_not_filled_from_order_or_other_notes():
    route = ranked(None, reason=''); route['notes'] = 'Preliminary planning note.'
    groups = project_approaches({}, [context('a')], {'goals':[route]})
    data = {'kind':'design','contexts':[context('a')], 'approach_rankings':groups}
    document = approach_html(data)
    assert 'Why it ranks first' not in document and 'Preliminary planning note' not in document
    assert 'ranking reason is unavailable' in document
    assert 'This run evaluated one target and approach.' in approach_html({**data,'approach_rankings':[]})
    assert approach_html({**data,'kind':'evaluation'}) == ''


def test_saved_summary_is_projected_and_rendered_without_changing_source_data(tmp_path):
    run_fixture(tmp_path)
    route = ranked(3, reason='Supported <tradeoff> & retained uncertainty.', count=0)
    save(tmp_path,'artifacts/design/strategy_router_summary.json',{'goals':[route]})
    before = {str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
    data = project_run(tmp_path)
    assert data['approach_rankings'][0]['approaches'][0]['rank'] == 3
    document = explorer_html(data)
    assert 'Approach rank 3' in document and 'Why it has approach rank 3' in document
    assert 'Supported &lt;tradeoff&gt; &amp; retained uncertainty.' in document
    packet = json.loads(Packet(document).text)
    assert packet['approach_rankings'] == data['approach_rankings']
    assert before == {str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
