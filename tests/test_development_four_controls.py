from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from Bio.Seq import Seq

from aso_agents import pipeline_tools, genbank_export, tiling_agent
from aso_agents.comparison_evidence import response_comparison_context
from aso_agents.coordinate_conversion import coordinate_convention_question
from aso_agents.evaluation_evidence import attach_measurement_references, bind_assessment_measurements
from aso_agents.response_contracts import response_contract_errors
from aso_agents.target_geometry import transcript_feature_geometry
from aso_agents.regulatory_projection import source_utr_record, project_source_utr_site
from aso_agents.preservation import protected_junctions
from aso_agents import report_explainer
router = importlib.import_module('aso_agents.splice_strategy_router_agent')
expression = importlib.import_module('aso_agents.off_target_expression_agent')
import run_goal_intake as runtime
import hashlib
import sys
from types import SimpleNamespace


def mapping_module():
    path = Path(__file__).parents[1]/'aso_agents/resources/tiling/filter_mapped.py'
    spec = importlib.util.spec_from_file_location('filter_mapped_controls', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_repetitive_mapq_zero_hits_are_counted_and_reverse_flag_bits_preserved(tmp_path):
    module = mapping_module()
    q = 'id=neutral|pos=chr8:101-108:+'
    rows = [(q,0,'chr8',101,255,'AACCGGTA'),(q,272,'chr9',601,0,'TACCGGTT'),(q,4,'*',0,0,'AACCGGTA')]
    sam=tmp_path/'reads.sam'
    sam.write_text('\n'.join('\t'.join(map(str,[name,flag,chrom,pos,mapq,'8M','*',0,0,seq,'*','NM:i:0'])) for name,flag,chrom,pos,mapq,seq in rows)+'\n')
    df=module.load_sam_alignments(str(sam))
    assert len(df)==2 and set(df.qseq)=={'AACCGGTA'}
    meta=module.build_mismatch_metadata(df,1)[(q,'AACCGGTA')]
    assert meta['off_target_exact_count']=='1'
    assert meta['off_target_0mm_positions']=='chr9:601-608'
    assert meta['off_target_1mm_count']=='0' and meta['off_target_2mm_count']=='NA'


def test_censored_near_search_keeps_independent_exact_hits_and_unknown_bins():
    module=mapping_module();key=('q','AACCGGTA')
    df=pd.DataFrame([{'qname':'q','qseq':'AACCGGTA','position':'chr8:101-108','mismatches':0,'is_intended_target':True},
                     {'qname':'q','qseq':'AACCGGTA','position':'chr9:601-608','mismatches':0,'is_intended_target':False}])
    result=module.reconcile_mapping_metadata(df,pd.DataFrame(),[key],1,near_available=True,near_overflow={'q'})[key]
    assert result['off_target_exact_count']=='1'
    assert result['off_target_1mm_count']==result['off_target_total_upto_1mm_count']==result['off_target_weighted_risk_score']=='NA'
    unknown=module.reconcile_mapping_metadata(df,pd.DataFrame(),[key],1,near_available=False,same_reference=False)[key]
    assert unknown['off_target_exact_count']=='NA'
    censored=module.reconcile_mapping_metadata(df,pd.DataFrame(),[key],1,near_available=True,near_overflow={'q'},exact_overflow={'q'})[key]
    assert censored['off_target_exact_count']=='NA'


@pytest.mark.parametrize('value,expected', [('NA',None),('',None),('nan',None),('inf',None),('-1',None),('0',0),('34',34)])
def test_unknown_measurements_never_become_zero(value,expected):
    assert tiling_agent._measurement_int_or_unknown(value) == expected


@pytest.mark.parametrize('strand',[1,-1])
def test_steric_genbank_feature_extracts_exact_target_bases(tmp_path,strand):
    genome=Seq('GGGAACCGGTATTTC')
    # The genomic record starts at chr8:101 (one-based). The target is BED [103,111).
    target=genome[3:11] if strand==1 else genome[3:11].reverse_complement()
    path=tmp_path/'route.csv';path.write_text('Start,End,Strand,Sequence,Synthesis\n103,111,'+('+' if strand==1 else '-')+','+str(target)+','+str(target.reverse_complement())+'\n')
    features=genbank_export.build_aso_features([{'count':1,'csv_path':str(path),'goal_label':'neutral'}],101,114)
    assert len(features)==1 and features[0].extract(genome)==target
    assert (int(features[0].location.start),int(features[0].location.end))==(3,11)


def test_large_comparison_prompt_is_bounded_without_losing_sets_or_zero_values():
    payload={'unique_by_set':{name:[{'sequence':'ACGT','huge_provenance':'x'*1000} for _ in range(3000)] for name in ('a','b','c')},
             'set_count':3,'candidate_evidence_by_set':[{'set_id':name,'rows':[{'score':0,'missing':None}]} for name in ('a','b','c')]}
    before=deepcopy(payload);result=response_comparison_context(payload)
    assert len(json.dumps(result))<10000 and payload==before
    assert all(r['row_count']==3000 and len(r['identifier_sample'])==20 for r in result['unique_by_set'].values())
    assert result['candidate_evidence_by_set']==payload['candidate_evidence_by_set']


def test_assumed_model_convention_is_not_a_user_answer():
    goal={'region_constraints':[{'source_type':'explicit_rows','coordinate_space':'genomic','coordinate_convention':'1based_inclusive',
                               'intervals_bed0_half_open':[{'chrom':'chr8','start':99,'end':140}]}]}
    envelope={'structured_goal':goal};request='Use chr8:100-140.'
    assert coordinate_convention_question(request,[],envelope)
    assert coordinate_convention_question(request,[{'question':'BED or one-based?', 'answer':'Please convert the build.'}],envelope)
    assert coordinate_convention_question(request,[{'answer':'Both intervals use zero-based half-open coordinates.'}],envelope) is None
    assert coordinate_convention_question('Target transcript exon 3.',[],envelope) is None


@pytest.mark.parametrize('strand',['+','-'])
def test_reference_geometry_defines_upstream_without_inventing_regulatory_identity(strand):
    exons=[{'index':i+1,'start':a,'end':b} for i,(a,b) in enumerate([(101,130),(201,230),(301,330)] if strand=='+' else [(301,330),(201,230),(101,130)])]
    anchor=201 if strand=='+' else 230
    record={'resolved_target':{'transcript_id':'TX.2','anchor_position':anchor,'anchor_site':'acceptor','materialized_rows':[['chr8',190,241,'window',0,strand]]}}
    catalog={'tx':{'transcript_id':'TX.2','chrom':'8','strand':strand,'metadata':{'reference_source':'local_gtf','reference_annotation':'neutral.gtf'},'exons_tx':exons}}
    result=transcript_feature_geometry(record,catalog)
    assert result['selected_feature']['exon_index']==2
    assert all(f['relation']=='upstream' for f in result['annotated_junctions'] if f['exon_index']==1)
    assert all(f['relation']=='downstream' for f in result['annotated_junctions'] if f['exon_index']==3)
    assert len(result['annotated_junctions'])==4
    proof={'record_index':0,'resolved_target':{'evidence_capabilities':['annotated_junction_geometry']}}
    finding={'record_index':0,'status':'satisfied','requirement_kind':'annotated_junction_relation'}
    assert not pipeline_tools._resolved_target_finding_capability_errors([finding],[proof],positive_status='satisfied')
    finding['requirement_kind']='spatial_exclusion'
    assert pipeline_tools._resolved_target_finding_capability_errors([finding],[proof],positive_status='satisfied')
    record['requested_target']={'reference_transcript_id':'TX.3'}
    assert transcript_feature_geometry(record,catalog) is None
    record.pop('requested_target')
    record['resolved_target']['anchor_position']+=1
    assert transcript_feature_geometry(record,catalog) is None


def test_mismatch_summary_keeps_rna_context_and_zero_in_cited_evidence(tmp_path):
    rows=[{'name':'A','near_match_candidates':[{'mapping_scope':scope,'match_orientation':'reverse_complement','mismatch_count':n}
            for scope,n in [('spliced_cdna',8),('pre_mrna_locus',6),('pre_mrna_locus',0)]]}]
    summary=attach_measurement_references({'provided_asos':rows})
    row=summary['provided_asos'][0]
    minima={r['rna_context']:r['minimum_recorded_mismatches'] for r in row['near_match_measurements']}
    assert minima=={'mature_spliced_cdna':8,'unspliced_transcript_locus_pre_mrna':0}
    token=row['reference_tokens']['near_match_unspliced_transcript_locus_pre_mrna_reverse_complement']
    response={'evaluation_assessments':[{'name':'A','mechanistic_plausibility':token+'; activity remains unresolved.'}]}
    assert not response_contract_errors(response,{'task_type':'evaluate_existing_aso_set','evaluation_summary':summary})
    rendered=bind_assessment_measurements(response,summary)
    from aso_agents.evaluation_report import write_evaluation_evidence
    write_evaluation_evidence(tmp_path, {**rendered, 'response_evaluation_summary': summary})
    cited = token[2:-2]
    assert 'evaluation_evidence.html#' + cited in rendered['markdown']
    assert 'id="' + cited + '"' in (tmp_path/'evaluation_evidence.html').read_text()
    saved = json.loads((tmp_path/'evaluation_evidence.json').read_text())
    assert {v['rna_context']: v['minimum_recorded_mismatches'] for v in
            saved['provided_asos'][0]['near_match_measurements']} == minima
    # Prose factual contradictions are handled by the evidence reviewer; this
    # deterministic contract checks reference integrity, not keyword patterns.
    response['evaluation_assessments'][0]['mechanistic_plausibility']='{{missing_measurement}}'
    assert response_contract_errors(response,{'task_type':'evaluate_existing_aso_set','evaluation_summary':summary})


def test_locus_retention_does_not_let_sense_hits_hide_antisense_hypotheses(monkeypatch):
    monkeypatch.setattr(pipeline_tools,'_fetch_transcript_locus_sequence',lambda **kwargs:{
        'transcript_oriented_sequence':'AACCGGTANNNNTACAGGTT','chrom':'chr8','strand':'+',
        'locus_start':101,'locus_end':120,'transcript_id':kwargs.get('transcript_id') or kwargs.get('isoform_label'),'isoform_label':kwargs.get('isoform_label')})
    candidates=pipeline_tools._best_locus_near_matches(entry={'sequence':'AACCGGTA','reverse_complement_sequence':'TACCGGTT'},
        isoform_transcripts={f'TX{i}':{'transcript_id':f'TX{i}'} for i in range(10)},runtime=None,genome=True,preferred_transcript_id='TX9',max_results=3)
    assert {c['match_orientation'] for c in candidates}=={'input_sequence','reverse_complement'}
    anti=[c for c in candidates if c['match_orientation']=='reverse_complement']
    assert min(c['mismatch_count'] for c in anti)==1 and any(c['transcript_id']=='TX9' for c in anti)


def test_published_utr_projection_requires_exact_version_and_sequence_evidence(tmp_path):
    path=tmp_path/'utr.tsv'
    path.write_text('Refseq ID\tGene ID\tGene Symbol\tSpecies ID\tUTR sequence\nTX.3\tG\tNEUTRAL\t9606\tAA-CCUUAG\nTX.3\tG\tNEUTRAL\t10090\tCCCC\n')
    source=source_utr_record(path,'9606','TX.3')
    assert source['ungapped_utr_sequence']=='AACCTTAG'
    assert source_utr_record(path,'9606','TX.4') is None
    assert source['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    projection=project_source_utr_site(source,'CATGGGAACCTTAG',5,2,5)
    assert (projection['cdna_start_1based'],projection['cdna_end_1based'])==(8,11)
    assert projection['site_sequence_5to3']=='ACCT'
    assert project_source_utr_site(source,'CATGGGAACCATAG',5,2,5)['status']=='unresolved'
    assert project_source_utr_site(source,'AACCTTAGAACCTTAG',1,2,5)['status']=='unresolved'
    assert project_source_utr_site(source,'CATGGGAACCTTAG',5,2,50)['status']=='unresolved'


def preservation_goal(strand='+'):
    exons=[{'index':i+1,'start':a,'end':b} for i,(a,b) in enumerate([(101,130),(201,230),(301,330)] if strand=='+' else [(301,330),(201,230),(101,130)])]
    return {'gene_symbol':'NEUTRAL','preservation_targets':[{'action':'preserve','isoform_label':'normal','transcript_id':'TX.3'}],
            'isoform_transcripts':{'normal':{'transcript_id':'TX.3','chrom':'8','strand':strand,'exons_tx':exons}}}


@pytest.mark.parametrize('strand',['+','-'])
def test_preservation_uses_complete_internal_junctions_and_rejects_version_conflicts(strand):
    goal=preservation_goal(strand);protected=protected_junctions(goal)
    assert set(protected)==({'donor@130','donor@230','acceptor@201','acceptor@301'} if strand=='+' else {'donor@301','donor@201','acceptor@230','acceptor@130'})
    goal['preservation_targets'][0]['transcript_id']='TX.9'
    assert not protected_junctions(goal)
    goal['preservation_targets'][0].update(transcript_id='TX.3',gene_symbol='OTHER')
    assert not protected_junctions(goal)


@pytest.mark.parametrize('zero_fallback',[False,True])
def test_required_preservation_cannot_be_bypassed_by_threshold_or_fallback(tmp_path,zero_fallback):
    effect='0' if zero_fallback else '-0.8'
    rows=[{'id':name,'delta_names':'donor@250,donor@230','delta_deletion':effect+','+v,'delta_replacement':effect+','+w}
          for name,v,w in [('safe','0','0'),('bad','-0.9','-0.9'),('missing','0',''),('nan','nan','nan')]]
    path=tmp_path/'input.json';path.write_text(json.dumps({'rows':rows}))
    output=tmp_path/'out.json'
    common=dict(json_path=str(path),primary_label='donor@250',primary_direction='decrease',start_threshold=.5,min_threshold=.01,
                min_count=1,output_path=str(output),allow_zero_delta_fallback=zero_fallback)
    result=router.auto_threshold_constrained_site_metrics(**common,safeguards_json=json.dumps([{'label':'donor@230','required':True,'max_abs_delta':.1}]))
    assert result['count']==1 and {row['id'] for row in json.loads(output.read_text())['rows']}=={'safe'}
    advisory=router.auto_threshold_constrained_site_metrics(**common,safeguards_json=json.dumps([{'label':'donor@230','max_abs_delta':.1}]))
    assert advisory['count']>1
    assert router._parse_site_metrics({'delta_names':'donor@230','delta_deletion':0,'delta_replacement':0})[0]['delta_deletion']==0


def test_atomic_plan_excludes_direct_protected_suppression_but_keeps_cryptic_route(tmp_path,monkeypatch):
    goal=preservation_goal()
    (tmp_path/'scores.json').write_text(json.dumps({'rows':[{
        'delta_names':'donor@230,donor@250','delta_deletion':'-.8,-.8','delta_replacement':'-.8,-.8'}]}))
    goal['splice_site_plan']=[{'label':'donor@230','site':'donor','position':230}, {'label':'donor@250','site':'donor','position':250}]
    entries=[{'label':label,'site':'donor','position':pos,'decision_direction':'decrease'} for label,pos in [('donor@230',230),('donor@250',250)]]
    monkeypatch.setattr(pipeline_tools,'_routing_target_entries',lambda *a,**k:entries)
    monkeypatch.setattr(pipeline_tools,'_rank_routing_entries_with_llm',lambda **k:{'ranked_entries':entries})
    result=pipeline_tools._build_atomic_routing_plan(structured_goal=goal,scored_path=str(tmp_path/'scores.json'),run_dir=str(tmp_path),flags={})
    assert 'planner_plan' in result, result
    plan=result['planner_plan']
    assert [x['args']['primary_label'] for x in plan['steps']]==['donor@250']
    assert plan['excluded_preservation_routes'][0]['label']=='donor@230'
    guards=json.loads(plan['steps'][0]['args']['safeguards_json'])
    assert any(g['label']=='donor@230' and g['required'] for g in guards)


def test_knockdown_exon_placeholder_resolves_complete_locked_context_before_ptc(monkeypatch):
    goal={'gene_symbol':'NEUTRAL','species':'human','intent':'knockdown','run_splice_eval':True,'run_splice_router':True,
          'transcript_policy':'user_locked','transcript_lock':{'kind':'explicit_id','value':'ENST00000000009'},
          'splice_isoform_objectives':[{'objective_scope':'default_exon_context','transcript_id':'ENST00000000009','exon_label':2,'desired_effect':'skipping'}]}
    exons=[{'index':i+1,'start':101+i*100,'end':130+i*100,'contains_coding':True,'cdna_start':1+i*30,'cdna_end':30+i*30} for i in range(3)]
    monkeypatch.setattr(runtime,'_default_splice_context_transcript_listing',lambda *a,**k:{'transcripts':[{'id':'ENST00000000009','is_canonical':True}],
        'structures':{'ENST00000000009':{'id':'ENST00000000009','strand':1,'seq_region_name':'8','exons_tx':exons}}})
    monkeypatch.setattr(runtime,'_generate_splice_strategy_analysis',lambda *a,**k:{})
    result=runtime._ensure_default_splice_context(goal,{})
    assert len(result['isoform_transcripts']['ENST00000000009']['exons_tx'])==3
    assert {i for s in result['splice_site_plan'] for i in s['exon_indices'].values()}=={2}
    plain={'gene_symbol':'NEUTRAL','intent':'knockdown','run_splice_eval':False}
    assert runtime._ensure_default_splice_context(plain,{})=={}


def test_missing_expression_keeps_measured_zero_separate_from_unknown_total(tmp_path,monkeypatch):
    class Census:
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setitem(sys.modules,'cellxgene_census',SimpleNamespace(open_soma=lambda **k:Census()))
    monkeypatch.setattr(expression,'_resolve_cellxgene_census_version',lambda *a:'fixed')
    monkeypatch.setattr(expression,'_soma_list_species',lambda *a:[])
    monkeypatch.setattr(expression,'_soma_list_tissues',lambda *a,**k:[])
    monkeypatch.setattr(expression,'_soma_find_genes',lambda *a:{'G1':{},'TARGET':{}})
    monkeypatch.setattr(expression,'_soma_get_expression',lambda *a,**k:{'G1':{'mean_expression':0,'total_cells':10,'n_cells':10},'TARGET':{'mean_expression':2}})
    path=tmp_path/'rows.json';path.write_text(json.dumps({'rows':[{'Off_target_gene_ids':['G1','G2'],'Off_target_0mm_gene_ids':['G1','G2']},
                                                               {'Off_target_gene_ids':['G1'],'Off_target_0mm_gene_ids':['G1']}]}))
    result=expression.annotate_off_target_expression(str(path),json.dumps({'gene_symbol':'N','gene_id':'TARGET','species':'human','target_context':'general human baseline'}),cache_dir=str(tmp_path/'cache'))
    assert not result.get('error')
    partial,complete=json.loads(path.read_text())['rows']
    assert partial['Off_target_expression_measured_sum']==0 and partial['Off_target_expression_sum'] is None
    assert partial['Off_target_expression_ratio'] is None and partial['Off_target_expression_sum_0mm'] is None
    assert complete['Off_target_expression_sum']==0 and complete['Off_target_expression_ratio']==0


def test_local_utr_projection_tolerates_distal_changes_but_not_site_mismatch_or_ambiguity():
    import random
    rng=random.Random(25)
    sequence=''.join(rng.choice('ACGT') for _ in range(600))
    source={'transcript_id':'TX.3','ungapped_utr_sequence':sequence,'utr_sha256':hashlib.sha256(sequence.encode()).hexdigest()}
    current='G'*20+sequence[100:500]
    result=project_source_utr_site(source,current,21,201,208)
    assert result['status']=='verified' and result['method']=='unique_exact_source_site_and_flanks_in_reference_utr'
    assert current[result['cdna_start_1based']-1:result['cdna_end_1based']]==sequence[200:208]
    mutated=current[:120]+('A' if current[120]!='A' else 'T')+current[121:]
    assert project_source_utr_site(source,mutated,21,201,208)['status']=='unresolved'
    assert project_source_utr_site(source,current+current[20:],21,201,208)['status']=='unresolved'


def test_recorded_guide_keeps_zero_initial_filter_counts_and_delivered_exports():
    context={'run_metadata':{'gene_symbol':'NEUTRAL','user_request':'Design splice-switching ASOs.'},
             'branch_outcomes':{'steric':{'status':'no_candidates_after_constraints'}},
             'report_manifest':{'payload_kind':'report_manifest','tiling':{'initial_filtering':{'total_rows':37,'passed':{'count':0}}},
                                'steric_jobs':[{'run_splice_eval':True}], 'branch_outcomes':{'steric':{'status':'no_candidates_after_constraints'}}},
             'output_inventory':[{'key':'scored_tiles','title':'Scored candidates','summary':'0 scored candidates; 37 initial windows.'},
                                 {'key':'genbank','title':'GenBank export','summary':'Saved annotation file.'}],
             'important_artifacts':[{'path':'aso_export.gb'}]}
    result=report_explainer.build_recorded_report_explainer(context)
    descriptions=[node['description'] for node in result['workflow_graph']['nodes']]
    assert 'Steric tiling kept 0 of 37 candidates after initial filtering.' in descriptions
    assert next(g for g in result['output_guides'] if g['key']=='genbank')['what_it_is']=='Saved annotation file.'
    assert result['source']=='recorded_output_navigation'
