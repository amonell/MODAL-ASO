"""Mature-RNA constraints must retain transcript identity and physical bases."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import run_goal_intake as native
from aso_agents import pipeline_tools as pipeline
from aso_agents.region_projection import genomic_sources
from aso_agents.region_overlap import candidate_matches_region_groups, matches_constraint_filters
from aso_agents.candidate_region_gate import filter_final_rows_by_global_regions
from aso_agents.binding_region_reports import memberships
from aso_agents.mature_steric import compile_regions, build_tiles, uses_mature_rna


def fixture(strand='+'):
    exons=[{'index':1,'start':1001,'end':1020},{'index':2,'start':2001,'end':2040}]
    if strand=='-':
        exons=list(reversed(exons))
        for i,e in enumerate(exons,1): e['index']=i
    entry={'transcript_id':'TXA','metadata':{'versioned_id':'TXA.3'},'chrom':'chr8','strand':strand,'exons_tx':exons}
    constraint={'constraint_id':'window','source_type':'explicit_rows','coordinate_space':'cdna',
        'materialized_rows':[['TXA.3',10,50,'window',0,strand]],'aso_overlap_requirement':'full_containment'}
    goal={'isoform_transcripts':{'primary':entry},'species':'human',
          'materialized_region_constraints':{'global':[constraint]}}
    program={'program_id':'panel','modality':'steric','intent':'binding','transcript_id':'TXA.3',
        'coordinate_space':'cdna','coordinate_convention':'one_based_inclusive',
        'target_region':'translation_start','start':11,'end':50,'length':20}
    return goal,program,constraint


@pytest.mark.parametrize('strand,expected',[('+',[(1010,1020),(2000,2030)]),('-',[(2000,2030),(1010,1020)])])
def test_projection_preserves_source_across_exons(strand,expected):
    goal,program,c=fixture(strand)
    source=genomic_sources(goal,c)[0]
    assert [(r[1],r[2]) for r in source['genomic_rows']]==expected
    assert source['source_row']==c['materialized_rows'][0]
    assert sum(r[2]-r[1] for r in source['genomic_rows'])==40
    c['materialized_rows'][0][0]='TXA.4'
    assert genomic_sources(goal,c)==[]


@pytest.mark.parametrize('strand',['+','-'])
def test_global_gate_and_report_share_exact_split_source(strand):
    goal,program,c=fixture(strand)
    sources=genomic_sources(goal,c)
    regions=sources[0]['genomic_rows']
    row={'intervals':[(r[0],r[1],r[2]) for r in regions]}
    get=lambda row,**kw:row['intervals']
    kept,audit=filter_final_rows_by_global_regions([row],goal,get)
    assert kept==[row] and audit['excluded_count']==0
    evidence=memberships(row,goal,get)
    assert len(evidence)==1 and evidence[0]['overlap_bp']==40
    assert evidence[0]['source_coordinate_space']=='cdna'
    wrong={'intervals':[('chr8',1010,2030)]}
    assert filter_final_rows_by_global_regions([wrong],goal,get)[0]==[]


def test_group_overlap_counts_unique_bases_across_exons_only_for_same_source():
    regions=[['chr8',10,15],['chr8',30,35]]
    blocks=[('8',10,15),('chr8',10,15),('8',30,35)]
    assert candidate_matches_region_groups(blocks,[regions],'any_overlap',8)
    assert not candidate_matches_region_groups(blocks,[[regions[0]],[regions[1]]],'any_overlap',8)
    assert not candidate_matches_region_groups(blocks,[regions],'any_overlap',11)
    assert candidate_matches_region_groups(blocks,[regions],'contains_feature',1)
    assert not candidate_matches_region_groups([('8',10,15)],[regions],'contains_feature',1)


@pytest.mark.parametrize('strand',['+','-'])
def test_mature_compile_keeps_outer_window_when_small_feature_is_intersected(strand):
    goal,program,c=fixture(strand)
    codon={**deepcopy(c),'constraint_id':'codon','operator':'intersect',
           'materialized_rows':[['TXA.3',24,27,'codon',0,strand]],'aso_overlap_requirement':'contains_feature'}
    goal['materialized_region_constraints']['global'].append(codon)
    base,meta=native._build_steric_program_region_rows(goal,program)
    regions,compiled=compile_regions(native,goal,program,base)
    scope=compiled['mature_rna']
    assert scope['search_rows_cdna'][0][1:3]==[10,50]
    assert sum(r[2]-r[1] for r in regions)==40
    filters=scope['constraint_filters_cdna']
    assert matches_constraint_filters([('TXA',10,30)],filters)
    assert not matches_constraint_filters([('TXA',27,47)],filters)
    assert not matches_constraint_filters([('TXA',0,20)],filters)
    genomic_filter=compiled['tile_constraint_filters']
    candidate=genomic_sources(goal,{'coordinate_space':'cdna','materialized_rows':[['TXA',10,30]]})[0]
    assert matches_constraint_filters([(r[0],r[1],r[2]) for r in candidate['genomic_rows']],genomic_filter)


@pytest.mark.parametrize('strand',['+','-'])
def test_explicit_any_overlap_window_permits_only_legal_flanks(strand):
    goal,program,c=fixture(strand)
    program.update(start=26,end=35,length=20)
    c.update(materialized_rows=[['TXA.3',25,35,'window',0,strand]],aso_overlap_requirement='any_overlap',minimum_overlap_bp=8)
    base,_=native._build_steric_program_region_rows(goal,program)
    _,meta=compile_regions(native,goal,program,base)
    scope=meta['mature_rna']
    assert scope['search_rows_cdna'][0][1:3]==[13,47]
    filters=scope['constraint_filters_cdna']
    assert matches_constraint_filters([('TXA',13,33)],filters)
    assert not matches_constraint_filters([('TXA',12,32)],filters)
    assert matches_constraint_filters([('TXA',27,47)],filters)
    assert not matches_constraint_filters([('TXA',28,48)],filters)


@pytest.mark.parametrize('strand',['+','-'])
@pytest.mark.parametrize('chemistry',['steric-hindrance','steric-2ome-ps'])
def test_builder_preserves_actual_steric_chemistry_and_screens_junctions(tmp_path,monkeypatch,strand,chemistry):
    goal,program,c=fixture(strand)
    base,_=native._build_steric_program_region_rows(goal,program)
    _,meta=compile_regions(native,goal,program,base)
    job={**program,'job_id':'one','selection_meta':{'region_constraints':meta}}
    seq='ACGT'*15
    monkeypatch.setattr(native,'_build_isoform_cdna_sequences',lambda *a,**kw:{'primary':seq})
    monkeypatch.setattr(native,'_annotate_tile_rows_with_rnafold',lambda rows:None)
    seen=[]
    def screen(path,goal,constraints):
        seen.extend(json.loads(Path(path).read_text())['rows']);return {'status':'complete'}
    monkeypatch.setattr(native,'_annotate_gapmer_off_targets',screen)
    result=build_tiles(native,goal,job,{'recommendation':{'CHEMISTRY':chemistry,'length':20}},str(tmp_path))
    rows=json.loads(Path(result['json_path']).read_text())['rows']
    assert len(rows)==21 and len(seen)==21
    assert any(len(json.loads(r['Genomic_segments']))==2 for r in rows)
    for row in rows:
        target=seq[row['Start']-1:row['End']]
        idt,synthesis,warning=native._format_idt_and_synthesis(target,chemistry)
        assert row['IDT']==idt and row['Synthesis']==synthesis and warning is None
        assert row['coordinate_system']=='1based_inclusive'
        assert sum(seg['end']-seg['start']+1 for seg in json.loads(row['Genomic_segments']))==20
    assert result['off_target_mapping']['status']=='complete'


def test_splice_program_keeps_genomic_path():
    goal,program,c=fixture()
    assert uses_mature_rna(program)
    program['intent']='splice_modulation'
    assert not uses_mature_rna(program)
    program.update(intent='binding',coordinate_space='genomic')
    assert not uses_mature_rna(program)


def test_cross_species_cohort_selects_anchor_only_with_attested_secondary_reference():
    goal,program,_=fixture()
    program.update(modality='gapmer',intent='knockdown',target_type='transcript_cohort',
        transcript_ids=['TXA.3','OTHER.7'])
    program.pop('transcript_id');program.pop('target_region');program.pop('coordinate_space')
    goal['cross_species_targets']={'mouse':{'status':'resolved','transcript_ids':['OTHER.7'],
        'reference_evidence':[{'id':'OTHER','versioned_id':'OTHER.7','species':'mus_musculus'}]}}
    rows,meta=native._build_gapmer_program_region_rows(goal,goal['isoform_transcripts'],'primary',program)
    assert rows[0][0:3]==['TXA',0,60] and meta['isoform_label']=='primary'
    assert program['transcript_ids']==['TXA.3','OTHER.7']
    goal['cross_species_targets']['mouse']['reference_evidence'][0]['versioned_id']='OTHER.8'
    assert native._build_gapmer_program_region_rows(goal,goal['isoform_transcripts'],'primary',program)[0]==[]
    goal['cross_species_targets']['mouse']['reference_evidence'][0]['versioned_id']='OTHER.7'
    goal['cross_species_targets']['human']=goal['cross_species_targets'].pop('mouse')
    assert native._build_gapmer_program_region_rows(goal,goal['isoform_transcripts'],'primary',program)[0]==[]


def test_comparison_can_consume_declared_design_and_evaluation_without_duplicate_user_input(tmp_path):
    csv=tmp_path/'submitted.csv';csv.write_text('name,sequence\nrecord,ACGTACGTACGTACGTACGT\n')
    envelope={'schema_version':'v3','request_mode':'composite','tasks':[
        {'task_id':'design','task_type':'design_new_asos','context':{'gene_symbol':'TEST','species':'human','target_context':'cultured cells'}},
        {'task_id':'evaluate','task_type':'evaluate_existing_aso_set','context':{'gene_symbol':'TEST','species':'human',
            'provided_aso_files':[{'path':str(csv)}]}},
        {'task_id':'compare','task_type':'compare_aso_sets','depends_on':['design','evaluate'],'context':{}}]}
    validation=pipeline._request_validation_summary(envelope)
    assert 'provided_asos_missing' not in validation['tasks']['compare']['issues']
    envelope['tasks'][2]['depends_on']=['design']
    validation=pipeline._request_validation_summary(envelope)
    assert 'provided_asos_missing' in validation['tasks']['compare']['issues']


def test_evaluation_loader_cannot_substitute_unrelated_first_task(tmp_path):
    first=tmp_path/'unrelated.json';first.write_text(json.dumps({'provided_asos':[{'name':'wrong'}]}))
    second=tmp_path/'declared.json';second.write_text(json.dumps({'provided_asos':[{'name':'right'}]}))
    state={'request_envelope':{'tasks':[
        {'task_id':'unrelated','task_type':'evaluate_existing_aso_set'},
        {'task_id':'declared','task_type':'evaluate_existing_aso_set'}]},
        'task_artifacts':{'unrelated':{'json':str(first)},'declared':{'json':str(second)}}}
    state['task_graph'] = {'tasks': state['request_envelope']['tasks']}
    assert pipeline._load_task_payload_by_type(state,'evaluate_existing_aso_set',task_ids={'declared'})['provided_asos'][0]['name']=='right'
    assert pipeline._load_task_payload_by_type(state,'evaluate_existing_aso_set',task_ids=set())=={}


def test_recollection_preserves_every_compiled_chemistry_and_coordinate_source(tmp_path):
    groups=[]
    for index,source in enumerate(('steric_cdna','steric')):
        path=tmp_path/f'group-{index}.json'
        path.write_text(json.dumps({'rows':[{'id':str(index),'source':source,'chemistry_group_id':str(index)}]}))
        groups.append({'json_path':str(path)})
    result=pipeline._collect_steric_group_results(groups,str(tmp_path/'all.json'))
    rows=json.loads(Path(result['json_path']).read_text())['rows']
    assert [(r['source'],r['chemistry_group_id']) for r in rows]==[('steric_cdna','0'),('steric','1')]
    groups.append({'json_path':str(tmp_path/'missing.json')})
    assert pipeline._collect_steric_group_results(groups,str(tmp_path/'partial.json'))['error']=='steric_group_result_missing'


@pytest.mark.parametrize('answer,status',[
    ('I cannot provide that detail; please choose a reasonable documented default.', 'ready'),
    ('I do not know whether these are zero-based or one-based and cannot authorize an assumption.', 'need_clarification'),
])
def test_coordinate_answers_reach_semantic_review_without_keyword_adoption(monkeypatch,answer,status):
    envelope={'structured_goal':{'region_constraints':[{'source_type':'explicit_rows',
        'coordinate_space':'genomic','intervals_bed0_half_open':[{'chrom':'chr8','start':99,'end':140}]}]}}
    seen=[]
    async def review(evidence,planned):
        seen.append((json.loads(evidence),planned))
        return {**planned,'status':status,'questions':[] if status=='ready' else planned['questions']}
    monkeypatch.setattr(pipeline,'_review_request_clarifications_if_needed',review)
    request='Use chr8:100-140.'
    assert pipeline._review_coordinate_convention_question(request,[],envelope)['status']=='need_clarification'
    assert seen==[]
    history=[{'question':'Which convention?', 'answer':answer}]
    result=pipeline._review_coordinate_convention_question(request,history,envelope)
    assert result['status']==status
    assert seen[0][0]['clarification_history']==history
    assert seen[0][1]['request_envelope']==envelope
    assert result['request_envelope']==envelope

@pytest.mark.parametrize('strand',['+','-'])
def test_constraint_only_program_uses_reference_domain_and_all_predicates(strand):
    goal,program,outer=fixture(strand)
    for key in ('target_region','start','end'):program.pop(key)
    inner={**deepcopy(outer),'constraint_id':'inner','operator':'intersect',
           'materialized_rows':[['TXA.3',24,27,'inner',0,strand]],
           'aso_overlap_requirement':'contains_feature'}
    program['region_constraints']=[{**rule,'rows':deepcopy(rule['materialized_rows'])} for rule in (outer,inner)]
    goal['materialized_region_constraints']={'global':[], 'by_program':{'panel':[outer,inner]}}
    base,seed=native._build_steric_program_region_rows(goal,program)
    assert base==[] and seed['direct_region_constraints_seed']
    _,compiled=compile_regions(native,goal,program,base)
    scope=compiled['mature_rna']
    assert scope['search_rows_cdna'][0][1:3]==[0,60]
    allowed=[start for start in range(41) if matches_constraint_filters(
        [('TXA',start,start+20)],scope['constraint_filters_cdna'])]
    assert allowed==list(range(10,25))
    # A real independent base with no transcript projection remains exhausted.
    _,invalid=compile_regions(native,goal,program,[['chr99',0,60,'wrong',0,strand]])
    assert invalid['mature_rna']['search_rows_cdna']==[]
