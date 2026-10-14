from copy import deepcopy
from types import SimpleNamespace

from aso_agents import pipeline_tools as pipeline
from aso_agents.splice_reference_scope import scoped_splice_evidence


def goal():
    tx={'chrom':'chr8','strand':'+','exons_tx':[
        {'index':1,'start':10,'end':40},{'index':2,'start':100,'end':140}]}
    return {'isoform_transcripts':{'raised':{**deepcopy(tx),'transcript_id':'REF_A'},
                                   'reduced':{**deepcopy(tx),'transcript_id':'REF_B'}},
            'splice_isoform_objectives':[{'isoform_label':'raised','desired_abundance':'increase'},
                                         {'isoform_label':'reduced','desired_abundance':'decrease'}]}


def test_shared_risk_uses_reference_membership_not_missing_or_false_flags():
    site={'site':'acceptor','position':100,'chrom':'chr8','strand':'+'}
    result=scoped_splice_evidence({'shared_with_beneficial':False,'conflict_between_objectives':False},site,goal())
    assert result['shared_with_beneficial'] is True
    assert result['conflict_between_objectives'] is True
    assert result['beneficial_isoforms'] == ['REF_A']


def test_missing_reference_does_not_become_false_shared_risk():
    value=goal();value['isoform_transcripts']={}
    result=scoped_splice_evidence({'shared_with_beneficial':False},
                                  {'site':'acceptor','position':100},value)
    assert result.get('shared_with_beneficial') is None


def test_declared_objectives_are_not_serialized_as_predicted_isoform_effects(monkeypatch):
    monkeypatch.setattr(pipeline,'_goal_runtime_module',lambda:SimpleNamespace(
        _canonical_site_label=lambda s:f"{s['site']}@{s['position']}"))
    site={'site':'acceptor','position':100,'chrom':'chr8','strand':'+',
          'delta_direction':'context','decision_direction':'decrease',
          'delta_per_isoform':{'raised':'increase','reduced':'decrease'},
          'delta_per_isoform_scope':'desired_objectives_not_predicted_effects',
          'isoform_abundance_objectives':{'raised':'increase','reduced':'decrease'}}
    result=pipeline._serialize_routing_candidate_for_llm(site,runtime=pipeline._goal_runtime_module(),structured_goal=goal())
    assert result is not None and 'delta_per_isoform' not in result
    assert result['isoform_abundance_objectives']==site['isoform_abundance_objectives']
    assert result.get('isoform_objective_alignment',{}).get('status') != 'aligned'


def test_candidate_projection_retains_site_plan_objective_scope(monkeypatch):
    monkeypatch.setattr(pipeline,'_goal_runtime_module',lambda:SimpleNamespace(
        _canonical_site_label=lambda s:f"{s['site']}@{s['position']}"))
    candidate={'site':'acceptor','position':100,'delta_direction':'context'}
    plan={**candidate,'delta_per_isoform':{'raised':'increase','reduced':'decrease'},
          'delta_per_isoform_scope':'desired_objectives_not_predicted_effects',
          'isoform_abundance_objectives':{'raised':'increase','reduced':'decrease'}}
    result=pipeline._routing_catalog_entries({'steric_site_candidates':[candidate],'splice_site_plan':[plan]})
    assert result[0]['delta_per_isoform_scope'] == plan['delta_per_isoform_scope']
    assert result[0]['isoform_abundance_objectives'] == plan['isoform_abundance_objectives']
    assert 'delta_per_isoform_scope' not in candidate
