from copy import deepcopy
import csv
import json
import pytest

from aso_agents.ranking_contracts import scoped_strategy_routing_plan
import run_goal_intake as runtime


def plan_fixture():
    return {'mode':'per_job','job_count':2,'goal_count':2,
            'ranking':{'summary':'Only route Alpha is targetable.'},
            'jobs':[{'job_id':'job_a','job_label':'Design Alpha','planner_plan':{'ranking':{'model':'neutral','summary':'Only route Alpha is targetable.'}}},
                    {'job_id':'job_b','job_label':'Design Beta','planner_plan':{'ranking':{'model':'neutral','summary':'Only route Beta is targetable.'}}}]}


def test_independent_rankings_preserve_scope_without_promoting_first_job():
    original=plan_fixture();before=deepcopy(original)
    actual=scoped_strategy_routing_plan(original)
    summary=actual['ranking']['summary']
    assert 'Design Alpha: Only route Alpha is targetable.' in summary
    assert 'Design Beta: Only route Beta is targetable.' in summary
    assert 'not a comparative efficacy ranking' in summary
    assert actual['ranking']['scope']=='within_each_design_job'
    assert actual['jobs']==original['jobs'] and original==before
    assert scoped_strategy_routing_plan(actual)==actual


def test_html_compaction_keeps_every_job_and_scoped_summary():
    result=runtime._compact_strategy_routing_for_html({'planner_plan':plan_fixture(),'goals':[{'goal_label':'Alpha','rank':1},{'goal_label':'Beta','rank':2}]})
    ranking=result['planner_plan']['ranking']
    assert {r['job_id'] for r in ranking['job_rankings']}=={'job_a','job_b'}
    assert len(result['goals'])==2
    assert ranking['summary'].startswith('Independently ranked design jobs')


def test_single_global_plan_is_unchanged():
    plan={'ranking':{'summary':'One joint ranking.','model':'neutral'},'steps':[]}
    assert scoped_strategy_routing_plan(plan)==plan


@pytest.mark.parametrize('values,expected,peak',[
    ('-0.4, NA, 0.1','-0.4;;0.1','-0.4'),
    ('NA, NA, NA',';;',''),
    ('NaN, -0.2, inf',';-0.2;','-0.2'),
    ('0, -0.2, 0.1','0;-0.2;0.1','-0.2'),
])
def test_missing_route_deltas_keep_their_site_positions(tmp_path,values,expected,peak):
    path=tmp_path/'route.json'
    path.write_text(json.dumps({'rows':[{'Sequence':'ACGTCATGAGTTCGATCGTA','delta_names':'site_A, site_B, site_C',
                                     'delta_deletion':values,'delta_replacement':values}]}))
    output,error=runtime._write_strategy_route_csv(str(path))
    assert error is None
    with open(output) as stream:row=next(csv.DictReader(stream))
    assert row['delta_labels']=='site_A;site_B;site_C'
    assert row['delta_deletion_values']==expected and row['delta_replacement_values']==expected
    assert row['delta_peak_deletion']==peak and row['delta_peak_replacement']==peak
