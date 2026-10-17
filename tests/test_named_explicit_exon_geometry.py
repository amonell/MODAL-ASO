from copy import deepcopy

import pytest
import run_goal_intake as native
from aso_agents.explicit_target_identity import reconcile_declared_exon_spans


def span():
    return {'chrom':'chr8','start':150,'end':170,'strand':'+',
            'transcript_id':'TX.2','requested_exon_label':'2b','desired_delta':'decrease'}


def test_named_exon_never_becomes_its_numeric_prefix():
    assert native._normalize_exon_label_collection(['2b','12x',2,'exon 3'])==[2,3]
    source={'basis':'exon_presence','exon_label':'2b','presence':'included'}
    assert native._normalize_splice_aggregate_class(source)==source
    assert native._splice_aggregate_class_key({'aggregate_class':source})=='aggregate:exon_presence:exon_2b:included'


def test_exact_boundaries_coalesce_only_against_declared_complete_span():
    whole=span()
    targets=[{**whole,'start':p,'end':p} for p in (150,170,150,170)]
    result=reconcile_declared_exon_spans(targets,[whole])
    assert len(result)==1 and (result[0]['start'],result[0]['end'])==(150,170)
    assert len(result[0]['coalesced_boundary_targets'])==2
    assert reconcile_declared_exon_spans(result,[whole])==result
    assert reconcile_declared_exon_spans(targets,[])==targets
    for changed in ({'requested_exon_label':'2'}, {'transcript_id':'TX.3'}, {'strand':'-'}, {'chrom':'chr9'}):
        assert reconcile_declared_exon_spans(targets,[{**whole,**changed}])==targets
    assert reconcile_declared_exon_spans(targets,[whole,deepcopy(whole)])==targets


@pytest.mark.parametrize('strand',['+','-'])
def test_materialized_named_exon_keeps_one_span_and_actual_boundary_sites(strand,monkeypatch):
    whole={**span(),'strand':strand,'target_type':'exon','exon_label':'2b'}
    exons=[{'index':1,'start':50,'end':100,'seq_region_name':'chr8','contains_coding':True},
           {'index':2,'start':250,'end':300,'seq_region_name':'chr8','contains_coding':True}]
    if strand=='-':
        exons.reverse()
        for index,exon in enumerate(exons,1):exon['index']=index
    entry={'transcript_id':'TX','chrom':'chr8','strand':strand,'exons_tx':exons,
           'metadata':{'id':'TX','versioned_id':'TX.2'}}
    goal={'intent':'splice_modulation','desired_effect':'skipping','exon_label':'2b',
          'focus_exons_only':True,'isoform_transcripts':{'TX':entry},
          'evaluation_targets':[whole], 'explicit_splice_targets':[
              {**whole,'start':p,'end':p,'isoform_label':'TX'} for p in (150,170,150,170)]}
    native._materialize_modified_isoforms_for_explicit_targets(goal,{})
    model=goal['isoform_transcripts']['TX::explicit_target_model']
    assert len(model['exons_tx'])==3
    assert [(x['start'],x['end']) for x in model['exons_tx'] if x.get('source')=='explicit_splice_target']==[(150,170)]
    normalized=native._normalize_explicit_splice_targets(goal,goal['isoform_transcripts'])
    assert len(normalized)==1
    sites=native._explicit_splice_plan_from_targets(normalized,goal['isoform_transcripts'])
    expected={'acceptor':150 if strand=='+' else 170,'donor':170 if strand=='+' else 150}
    assert {s['site']:s['position'] for s in sites}==expected
    assert all(s['delta_direction']=='decrease' for s in sites)
    native._materialize_modified_isoforms_for_explicit_targets(goal,{})
    assert len(goal['isoform_transcripts']['TX::explicit_target_model']['exons_tx'])==3
