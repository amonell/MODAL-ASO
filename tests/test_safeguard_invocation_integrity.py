import json
from copy import deepcopy
import pytest
import run_goal_intake as runtime
from test_route_objective_evidence_scope import goal


def plans():
    return [{'site':'acceptor','position':100,'chrom':'chr8','strand':'+'},
            {'site':'donor','position':40,'chrom':'chr8','strand':'+'}]


def stale_candidates():
    return [{'label':label,'present_in':['reduced'],'absent_in':['raised'],
             'pathogenic_isoforms':['reduced'],'suggested_guard':'no_increase',
             'suggested_guard_reason':'pathogenic_only'}
            for label in ['acceptor@100','donor@40']]


def test_reference_reconciliation_rebuilds_stale_directional_policy(monkeypatch):
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',lambda *a,**k:{})
    value=goal();selected=runtime._build_safeguard_candidates_by_target(plans(),stale_candidates(),value)
    guards=runtime._default_safeguards_from_candidates(selected['acceptor@100'])
    assert guards[0].get('max_abs_delta') == .1
    assert guards[0].get('max_increase') is None
    assert 'reconciled' in selected['acceptor@100'][0]['guard_policy_source']


def test_absent_candidate_cache_does_not_disable_resolved_objective_guards(monkeypatch):
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',lambda *a,**k:{})
    value=goal();selected=runtime._build_safeguard_candidates_by_target(plans(),[],value)
    assert runtime._default_safeguards_from_candidates(selected['acceptor@100'])
    assert value['safeguard_selection_audit']['candidate_sources']['reconstructed'] == 2


def test_failed_decider_is_a_visible_conservative_fallback(monkeypatch):
    def fail(*a,**k):raise RuntimeError('neutral provider failure')
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',fail)
    value=goal();selected=runtime._build_safeguard_candidates_by_target(plans(),stale_candidates(),value)
    assert selected['acceptor@100']
    assert value['safeguard_selection_audit']['decider_status']=='failed_conservative_keep'


def route(tmp_path,guard,metrics=None):
    import importlib
    router=importlib.import_module('aso_agents.splice_strategy_router_agent')
    source=tmp_path/'scored.json';output=tmp_path/'route.json'
    row={'Sequence':'ACGTACGTACGTACGT','Chrom':'chr8','Start':80,'End':96,
         'Strand':'+','GC':.5,'delta_names':'acceptor@100',
         'delta_deletion':'-.7','delta_replacement':'-.7'}
    if metrics is not None:
        row['delta_names']+=',donor@40'
        for key in ('delta_deletion','delta_replacement'):
            row[key]+=','+str(metrics.get(key,'') if metrics.get(key) is not None else '')
    source.write_text(json.dumps({'rows':[row]}))
    result=router.auto_threshold_constrained_site_metrics(json_path=str(source),
        primary_label='acceptor@100',primary_direction='decrease',
        safeguards_json=json.dumps([guard]),min_count=1,min_gc=.1,max_gc=.9,output_path=str(output))
    assert not result.get('error'),result
    return json.loads(output.read_text())


def test_missing_advisory_measurements_are_unknown_not_passed(tmp_path):
    data=route(tmp_path,{'label':'donor@40','max_abs_delta':.1})
    assert data['rows'][0]['safeguards_passed'] is None
    assert data['rows'][0]['safeguards_status']=='not_evaluated'
    assert data['rows'][0]['safeguard_report'][0]['passes'] is None


def test_explicit_bound_is_not_silently_replaced_by_default(tmp_path):
    data=route(tmp_path,{'label':'donor@40','max_abs_delta':.2},
               {'delta_deletion':.15,'delta_replacement':.15})
    assert data['safeguards'][0]['max_abs_delta']==.2
    assert data['rows'][0]['safeguards_passed'] is True


def test_reconstruction_is_scoped_to_measured_sites(monkeypatch):
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',lambda *a,**k:{})
    value=goal()
    selected=runtime._build_safeguard_candidates_by_target(plans(),[],value,
                                                          measured_site_labels={'acceptor@100'})
    assert not selected['acceptor@100']
    assert value['safeguard_selection_audit']['candidate_sources']['reconstructed']==1


def test_missing_measurement_does_not_erase_existing_required_guard(monkeypatch):
    value=goal()
    guard={'label':'donor@40','required':True,'max_abs_delta':.25}
    selected=runtime._build_safeguard_candidates_by_target(plans(),[guard],value,
                                                          measured_site_labels={'acceptor@100'})
    assert runtime._default_safeguards_from_candidates(selected['acceptor@100'])==[
        {'label':'donor@40','required':True,'max_abs_delta':.25,
         'note':'Shared with beneficial isoforms in the inspected reference catalog; outside-catalog specificity unknown.'}]


def test_partial_decider_retains_unanswered_guards_and_records_provider_decisions(monkeypatch):
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',lambda *a,**k:{
        ('acceptor@100','donor@40'):{'action':'keep','rationale':'Distinct measured junction'}})
    value=goal();selected=runtime._build_safeguard_candidates_by_target(plans(),stale_candidates(),value)
    assert selected['acceptor@100'] and selected['donor@40']
    audit=value['safeguard_selection_audit']
    assert audit['decider_status']=='partial_conservative_keep'
    assert {x['source'] for x in audit['selections']}=={'decider','conservative_fallback'}


def test_incomplete_objective_reference_does_not_silently_discard_cached_guard(monkeypatch):
    value=goal();value['isoform_transcripts'].pop('raised')
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',lambda *a,**k:{})
    selected=runtime._build_safeguard_candidates_by_target(plans(),stale_candidates(),value)
    guard=selected['acceptor@100'][0]
    assert guard['objective_membership_complete'] is False
    assert runtime._default_safeguards_from_candidates([guard])[0]['max_abs_delta']==.1


def test_catalog_missing_all_objective_references_keeps_guard_with_uncertainty(monkeypatch):
    value=goal()
    value['all_transcripts_full']={'other':{**deepcopy(value['isoform_transcripts']['raised']),
                                           'transcript_id':'REF_OTHER'}}
    value['all_transcripts_full']['other']['exons_tx'][1]['start']=101
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',lambda *a,**k:{})
    selected=runtime._build_safeguard_candidates_by_target(plans(),stale_candidates(),value)
    assert selected['acceptor@100']
    assert selected['acceptor@100'][0]['unresolved_objective_transcript_ids']==['REF_A','REF_B']


@pytest.mark.parametrize('mode',['empty_catalog','no_exons'])
@pytest.mark.parametrize('cached',[False,True])
def test_no_inspectable_transcript_still_records_objective_uncertainty(monkeypatch,mode,cached):
    value=goal()
    if mode=='empty_catalog':value['isoform_transcripts']={}
    else:
        for entry in value['isoform_transcripts'].values():entry['exons_tx']=[]
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',lambda *a,**k:{})
    selected=runtime._build_safeguard_candidates_by_target(plans(),stale_candidates() if cached else [],value,
                                                           measured_site_labels={'acceptor@100','donor@40'})
    guard=selected['acceptor@100'][0]
    assert guard['objective_membership_complete'] is False
    assert len(guard['unresolved_objective_transcript_ids'])==2
    actual=runtime._default_safeguards_from_candidates([guard])[0]
    assert actual['max_abs_delta']==.1 and 'unresolved' in actual['note']


@pytest.mark.parametrize('bounds,observed,expected',[
    ({'max_abs_delta':.1},.05,True),({'max_abs_delta':.1},-.3,False),
    ({'max_increase':.1},-.8,True),({'max_increase':.1},.3,False),
    ({'max_decrease':.1},.8,True),({'max_decrease':.1},-.3,False),
    ({'max_abs_delta':.2},.15,True),({'max_abs_delta':0},0,True)])
def test_signed_guard_controls(tmp_path,bounds,observed,expected):
    data=route(tmp_path,{'label':'donor@40',**bounds},
               {'delta_deletion':observed,'delta_replacement':observed})
    assert data['rows'][0]['safeguards_passed'] is expected


@pytest.mark.parametrize('missing',[None,float('nan'),float('inf')])
@pytest.mark.parametrize('observed,expected',[(.05,None),(.3,False)])
def test_partial_measurement_keeps_unknown_and_real_breach_distinct(tmp_path,missing,observed,expected):
    data=route(tmp_path,{'label':'donor@40','max_abs_delta':.1},
               {'delta_deletion':missing,'delta_replacement':observed})
    row=data['rows'][0]
    assert row['safeguards_passed'] is expected
    csv_path,error=runtime._write_strategy_route_csv(str(tmp_path/'route.json'));assert not error
    import csv
    with open(csv_path) as stream:flat=next(csv.DictReader(stream))
    assert flat['safeguards_status']==('fail' if expected is False else 'not_evaluated')
    assert flat['safeguards_deletion_values']==''
    assert flat['safeguards_replacement_values']==str(observed)
    assert flat['safeguards_fail_count']==('1' if expected is False else '0')


def test_required_missing_guard_filters_candidate(tmp_path):
    data=route(tmp_path,{'label':'donor@40','max_abs_delta':.1,'required':True})
    assert data['rows']==[]
