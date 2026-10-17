import copy,json
import pytest
import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline
from aso_agents.preservation import required_routing_safeguards
from test_preservation_measurement_scope import fixture


@pytest.mark.parametrize('guard',[
    {'max_increase':.2},{'max_decrease':.3},{'max_abs_delta':.25},
    {'max_increase':.2,'max_decrease':.4}])
def test_preservation_fills_only_unbounded_directions(guard):
    result=required_routing_safeguards([{'label':'donor@230',**guard}],{'donor@230'})[0]
    assert result['required'] is True
    assert all(result[k]==v for k,v in guard.items())
    if 'max_abs_delta' not in guard:
        assert result['max_increase']==guard.get('max_increase',.1)
        assert result['max_decrease']==guard.get('max_decrease',.1)


def test_legacy_and_atomic_apply_same_resolved_preservation_policy(tmp_path,monkeypatch):
    goal=fixture()
    goal['splice_site_plan']=[{'site':'acceptor','position':201,'chrom':'chr8','strand':'+','delta_direction':'decrease'},
                             {'site':'donor','position':250,'chrom':'chr8','strand':'+','delta_direction':'decrease'}]
    goal['splice_strategy_analysis']={'analysis':{'control':True}}
    scored=tmp_path/'scored.json'
    scored.write_text(json.dumps({'rows':[{'Name':'safe','GC':.5,'Sequence':'ACGTACGTACGTACGT',
        'delta_names':'acceptor@201,donor@250,donor@230','delta_deletion':'0,-.8,0','delta_replacement':'0,-.8,0'},
        {'Name':'unsafe','GC':.5,'Sequence':'ACGTACGTACGTACGT',
        'delta_names':'acceptor@201,donor@250,donor@230','delta_deletion':'-.9,-.8,0','delta_replacement':'-.9,-.8,0'}]}))
    goal['tiling_results']={'scored_tiles_path':str(scored)}
    entries=[{**site,'label':runtime._canonical_site_label(site),'decision_direction':'decrease'} for site in goal['splice_site_plan']]
    monkeypatch.setattr(pipeline,'_routing_target_entries',lambda *a,**k:entries)
    monkeypatch.setattr(pipeline,'_rank_routing_entries_with_llm',lambda **k:{'ranked_entries':entries})
    atomic=pipeline._build_atomic_routing_plan(structured_goal=copy.deepcopy(goal),scored_path=str(scored),
                                              run_dir=str(tmp_path/'atomic'),flags={})['planner_plan']
    legacy=runtime._run_splice_strategy_routing(copy.deepcopy(goal),str(tmp_path/'legacy'))
    assert not legacy.get('error')
    for plan in (atomic,legacy['planner_plan']):
        assert [s['args']['primary_label'] for s in plan['steps']]==['donor@250']
        assert plan['excluded_preservation_routes'][0]['label']=='acceptor@201'
        assert plan['route_audits']['routes'][0]['label']=='acceptor@201'
        guards = json.loads(plan['steps'][0]['args']['safeguards_json'])
        assert {guard['label'] for guard in guards if guard.get('required')} == {'acceptor@201','donor@230'}
    assert json.loads(atomic['steps'][0]['args']['safeguards_json'])==json.loads(legacy['planner_plan']['steps'][0]['args']['safeguards_json'])
    selected=json.loads(open(legacy['goals'][0]['output_path']).read())['rows']
    assert [row['Name'] for row in selected]==['safe']


@pytest.mark.parametrize('label_key',['label','canonical_label'])
def test_explicit_primary_bound_survives_candidate_selection(label_key):
    plan=[{'site':'acceptor','position':100}]
    guard={label_key:'acceptor@100','required':True,'max_increase':.2}
    selected=runtime._build_safeguard_candidates_by_target(plan,[guard])
    assert runtime._default_safeguards_from_candidates(selected['acceptor@100'])==[
        {'label':'acceptor@100','required':True,'max_increase':.2}]
