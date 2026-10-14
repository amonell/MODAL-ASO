import csv
import json

from aso_agents.route_audit import write_route_audit
from aso_agents import pipeline_tools as pipeline


def test_excluded_route_keeps_signed_evidence_without_entering_design_panel(tmp_path):
    source=tmp_path/'scored.json'
    source.write_text(json.dumps({'rows':[
        {'Name':'a','sequence':'ACGT','Synthesis':'ACGT',
         'delta_names':['acceptor@100','donor@200'],
         'delta_deletion':[-0.2,0.5],'delta_replacement':[0.1,0.4]},
        {'Name':'b','Sequence':'TGCA','Chrom':'chr2','Start':10,'End':13,'Strand':'-',
         'delta_names':'donor@200,acceptor@100',
         'delta_deletion':'0.3,0.6','delta_replacement':'0.2,0.7'},
    ]}))
    original=source.read_bytes()
    directory=pipeline._write_run_path(tmp_path,'strategy_routes');directory.mkdir(parents=True)
    result=write_route_audit(source,[{'label':'acceptor@100','direction':'decrease',
                                     'reason':'Shared with an isoform requested to increase.'}],directory)
    rows=list(csv.DictReader(open(result['routes'][0]['path'])))
    assert [r['delta_deletion'] for r in rows] == ['-0.2','0.6']
    assert [r['delta_replacement'] for r in rows] == ['0.1','0.7']
    assert {r['recommendation_status'] for r in rows} == {'not_recommended'}
    assert (rows[1]['target_sequence'],rows[1]['chrom'],rows[1]['start'],rows[1]['strand']) == ('TGCA','chr2','10','-')
    assert pipeline._strategy_route_candidate_paths(tmp_path) == []
    assert source.read_bytes() == original


def test_no_audit_claims_when_source_is_missing(tmp_path):
    result=write_route_audit(tmp_path/'missing.json',[
        {'label':'donor@100','direction':'increase','reason':'Weak support.'}],tmp_path)
    assert result['status']=='unavailable' and result['routes']==[]


def test_native_planner_preserves_conflicting_secondary_measurements(monkeypatch,tmp_path):
    first={'site':'acceptor','position':100,'delta_direction':'decrease',
           'decision_direction':'decrease','delta_per_isoform':{'reduce':'decrease'}}
    shared={'site':'acceptor','position':9000,'delta_direction':'decrease',
            'decision_direction':'decrease','delta_per_isoform':{'reduce':'decrease','raise':'decrease'}}
    goal={'splice_site_plan':[first,shared], 'splice_isoform_objectives':[
        {'isoform_label':'reduce','desired_abundance':'decrease'},
        {'isoform_label':'raise','desired_abundance':'increase'}]}
    source=tmp_path/'scored.json';source.write_text(json.dumps({'rows':[
        {'delta_names':['acceptor@100','acceptor@9000'],
         'delta_deletion':[-0.6,-0.8], 'delta_replacement':[-0.5,-0.7]}]}))
    monkeypatch.setattr(pipeline,'_routing_target_entries',lambda *a,**k:[first,shared])
    monkeypatch.setattr(pipeline,'_rank_routing_entries_with_llm',lambda **k:{'ranked_entries':[first]})
    result=pipeline._build_atomic_routing_plan(structured_goal=goal,scored_path=str(source),
                                               run_dir=str(tmp_path),flags={})
    plan=result['planner_plan']
    assert [s['args']['primary_label'] for s in plan['steps']] == ['acceptor@100']
    audit=plan['route_audits']['routes'][0]
    assert audit['label']=='acceptor@9000' and audit['recommendation_status']=='not_recommended'
    assert audit['isoform_objective_alignment']['conflicting_isoforms']==['raise']
    assert audit['row_count']==1
