"""Neutral controls for typed geometry, scoped branches and output identity."""
from copy import deepcopy
import json
import importlib
import asyncio

import pytest

import run_goal_intake as native
from aso_agents import pipeline_tools as pipeline
from aso_agents.binding_region_reports import memberships
from aso_agents.candidate_minimums import requested_region_minimums, belongs_to_region
from aso_agents.gapmer_selectivity import annotate_preservation
from aso_agents.goal_contract import canonical_task_context_schema
chooser = importlib.import_module('aso_agents.transcript_chooser_agent')


def intervals(row, **kwargs):
    return row['intervals']


def test_union_membership_keeps_every_source_and_measured_overlap():
    rows = [['chr8', 100, 130, 'A'], ['chr8', 115, 160, 'B'], ['chr8', 180, 200, 'C']]
    goal = {'materialized_region_constraints': {'global': [{
        'source_type':'explicit_rows', 'coordinate_space':'genomic', 'label':'union',
        'materialized_rows':rows, 'aso_overlap_requirement':'any_overlap'}]}}
    evidence = memberships({'intervals':[('8', 110, 125), ('8', 115, 125)]}, goal, intervals)
    assert [(x['label'], x['overlap_bp']) for x in evidence] == [('A', 15), ('B', 10)]
    assert [x['source_interval_bed0_half_open'] for x in evidence] == rows[:2]


def test_global_named_windows_get_independent_minimums_and_overlap_thresholds():
    source = {'source_type':'explicit_rows', 'coordinate_space':'genomic', 'label':'union',
              'materialized_rows':[['chr8', 100, 120, 'A'], ['chr8', 150, 180, 'B']],
              'aso_overlap_requirement':'any_overlap', 'minimum_overlap_bp':8}
    goal = {'min_asos_per_site':2, 'materialized_region_constraints':{'global':[source]},
            'steric_design_jobs':[{'program_id':'design', 'selection_meta':{}}]}
    groups = requested_region_minimums(goal)
    assert [x['label'] for x in groups] == ['A','B']
    assert all(x['minimum']==2 for x in groups)
    assert not belongs_to_region({'intervals':[('chr8', 113, 140)]}, groups[0], intervals)
    assert belongs_to_region({'intervals':[('chr8', 112, 140)]}, groups[0], intervals)
    assert not belongs_to_region({'intervals':[('chr8', 112, 140)]}, groups[1], intervals)


@pytest.mark.parametrize('strand', ['+','-'])
def test_exact_transcript_identity_overrides_mistyped_chromosome_without_prose(tmp_path, strand):
    entry = {'transcript_id':'TXA', 'metadata':{'versioned_id':'TXA.3'}, 'chrom':'chr8', 'strand':strand,
             'exons_tx':[{'index':1,'start':1001,'end':1100}]}
    constraint = {'source_type':'explicit_rows','coordinate_space':'genomic',
                  'intervals_bed0_half_open':[{'chrom':'TXA.3','start':10,'end':45}],
                  'aso_overlap_requirement':'full_containment'}
    result = native._materialize_region_constraint_entry(structured_goal={}, final_payload={},
        run_dir=str(tmp_path), constraint=constraint, isoform_transcripts={'alias':entry})
    assert result['coordinate_space']=='cdna'
    assert result['materialized_rows'][0][:3]==['TXA.3',10,45]
    assert result['materialization_summary']['original_declared_coordinate_space']=='genomic'
    constraint['coordinate_space']='cdna';constraint['intervals_bed0_half_open'][0]['chrom']='TXA.4'
    rejected=native._materialize_region_constraint_entry(structured_goal={}, final_payload={},
        run_dir=str(tmp_path), constraint=constraint, isoform_transcripts={'alias':entry})
    assert rejected['materialized_rows']==[]


@pytest.mark.parametrize('strand,expected', [('+',[1010,1041]), ('-',[1059,1090])])
def test_explicit_start_window_is_not_replaced_with_approximate_flanks(strand, expected):
    entry={'transcript_id':'TXA.3','chrom':'chr8','strand':strand,
           'exons_tx':[{'index':1,'start':1001,'end':1100}]}
    goal={'isoform_transcripts':{'TXA.3':entry}}
    program={'program_id':'design','intent':'binding','modality':'steric','target_region':'translation_start',
             'transcript_id':'TXA.3','coordinate_space':'cdna','coordinate_convention':'one_based_inclusive',
             'start':11,'end':41,'flanks':{'up':12,'down':12},'length':20}
    rows,meta=native._build_steric_program_region_rows(goal,program)
    assert rows[0][1:3]==expected
    assert meta['tile_overlap_requirement']=='full_containment'
    program['end']=140
    rows,meta=native._build_steric_program_region_rows(goal,program)
    assert rows==[] and meta['error']=='explicit_transcript_window_outside_resolved_reference'


def test_fallback_abundance_target_cannot_change_primary_exon_objective_scope():
    goal={'design_programs':[
        {'program_id':'primary','label':'primary','modality':'steric','intent':'splice_modulation',
         'objective':'skip an exon','target_type':'exon','transcript_id':'TXA','exon_label':4,
         'desired_effect':'skipping','desired_delta':'decrease','site_kind':'both',
         'run_splice_router':True,'run_splice_eval':True},
        {'program_id':'fallback','label':'fallback','modality':'gapmer','intent':'knockdown',
         'objective':'knockdown','target_type':'transcript','transcript_id':'TXA','desired_delta':'decrease'}],
        'evaluation_targets':[{'objective_type':'route_target','target_type':'transcript',
                               'transcript_id':'TXA','desired_delta':'decrease'}],
        'splice_isoform_objectives':[{'objective_scope':'default_exon_context','transcript_id':'TXA',
            'exon_label':4,'desired_effect':'skipping','desired_abundance':'decrease'}]}
    assert not pipeline._canonical_design_goal_contract_issues(goal)
    goal['design_programs'][0]['target_type']='transcript'
    assert 'context.structured_goal.splice_isoform_objectives[0].objective_scope' in pipeline._canonical_design_goal_contract_issues(goal)


def test_beneficial_objective_from_other_panel_is_not_a_gapmer_preservation_filter(tmp_path):
    path=tmp_path/'molecules.json';path.write_text(json.dumps({'rows':[]}))
    result=annotate_preservation(path, {'splice_isoform_objectives':[{'isoform_label':'B','desired_abundance':'increase'}]},
                                {'beneficial_labels':['B']},{},None)
    assert result['status']=='not_required'


def test_live_external_accession_uses_exact_installed_xref_and_keeps_ambiguity(monkeypatch):
    monkeypatch.setattr(chooser, '_load_local_refseq_transcript_index', lambda *_:{'NM_000123.4':('TXA','TXB')})
    monkeypatch.setattr(chooser, '_resolve_local_external_transcript_accession', lambda *a,**kw:{
        'transcript_id':None,'transcript_ids':['TXA','TXB'],'ambiguous':True})
    monkeypatch.setattr(chooser, '_local_refseq_cache_identity', lambda *_:('/reference/xrefs.tsv',1))
    monkeypatch.setattr(chooser, '_get_ensembl_json', lambda *_:pytest.fail('remote lookup despite exact local xref'))
    result=chooser.resolve_external_transcript_accession('NM_000123.4')
    assert result['ambiguous'] and result['requested_accession']=='NM_000123.4'
    assert result['xref_source']=='/reference/xrefs.tsv'


def test_comparison_schema_exposes_implemented_observation_and_order_policies():
    properties=canonical_task_context_schema()['properties']['comparison_contract']['properties']
    assert properties['observation_policy']['enum']==['preserve_all_origins']
    assert properties['ranking_order']['enum']==['ascending']


def test_version_absent_from_live_catalog_resolves_exact_same_gene_installed_reference(monkeypatch):
    requested='ENST00000000888.3'
    goal={'gene_symbol':'FIXTURE','species':'human', 'splice_isoform_objectives':[
        {'transcript_id':requested,'isoform_label':requested}]}
    remote=[{'id':'ENST00000000888','version':4}]
    local={'id':'ENST00000000888','versioned_id':requested,'reference_annotation':'/reference/annotation.gtf'}
    monkeypatch.setattr(native,'local_reference_gene_transcripts',lambda *a,**kw:[local])
    final={}
    native._reconcile_inferred_splice_reference(goal,final,remote,'Use '+requested)
    assert goal['splice_isoform_objectives'][0]['transcript_id']==requested
    assert native._find_transcript_by_id(remote,requested)==local
    assert final['isoform_resolution_notes']['exact_local_references'][0]['reference_annotation']==local['reference_annotation']
    goal['splice_isoform_objectives'][0].update(transcript_id='ENST00000000888.2',isoform_label='ENST00000000888.2')
    with pytest.raises(RuntimeError,match='user-supplied transcript accession conflicts'):
        native._reconcile_inferred_splice_reference(goal,{},remote,'Use ENST00000000888.2')


def test_global_and_per_window_exact_counts_survive_audit_and_save(monkeypatch, tmp_path):
    request='Design eight molecules total, with two in each window.'
    audit={'candidate_count':{'count':8,'scope':'global','kind':'exact','quote':'eight molecules total'},
           'per_site_count':{'count':2,'kind':'exact','quote':'two in each window'}}
    async def fake_run(*args,**kwargs):
        return type('Response',(),{'final_output':json.dumps(audit)})()
    monkeypatch.setattr(native.Runner,'run',fake_run)
    draft={'status':'ready','structured_goal':{'design_programs':[
        {'program_id':'panel','modality':'steric','intent':'binding','objective':'panel'}]}}
    result=asyncio.run(native._audit_request_obligations(draft,request,[]))
    goal=result['structured_goal']
    assert (goal['candidate_limit'],goal['candidate_minimum'])==(8,8)
    assert (goal['candidate_limit_per_site'],goal['min_asos_per_site'])==(2,2)
    pipeline._save_goal_payload(str(tmp_path),{'request':request},goal,result)
    stored=json.loads(next(tmp_path.glob('**/structured_goal.json')).read_text())
    assert stored['min_asos_per_site']==2 and stored['candidate_limit_per_site']==2


def test_minimum_per_site_is_not_an_upper_bound(monkeypatch):
    request='Design at least three per window.'
    audit={'candidate_count':{'count':3,'scope':'per_site','kind':'minimum','quote':'at least three per window'}}
    async def fake_run(*args,**kwargs):
        return type('Response',(),{'final_output':json.dumps(audit)})()
    monkeypatch.setattr(native.Runner,'run',fake_run)
    draft={'status':'ready','structured_goal':{'design_programs':[{'program_id':'panel'}]}}
    goal=asyncio.run(native._audit_request_obligations(draft,request,[]))['structured_goal']
    assert goal['min_asos_per_site']==3 and 'candidate_limit_per_site' not in goal


def test_per_site_minimum_keeps_compiled_exclusions():
    constraint={'source_type':'explicit_rows','coordinate_space':'genomic','label':'site',
                'materialized_rows':[['chr8',100,140,'site']],'aso_overlap_requirement':'any_overlap'}
    goal={'min_asos_per_site':1,'materialized_region_constraints':{'global':[constraint]},
          'steric_design_jobs':[{'program_id':'design','selection_meta':{'region_constraints':{
              'tile_constraint_filters':[{'operator':'exclude','rows':[['chr8',100,120]]}]}}}]}
    group=requested_region_minimums(goal)[0]
    assert not belongs_to_region({'intervals':[('chr8',101,121)]},group,intervals)
    assert belongs_to_region({'intervals':[('chr8',120,140)]},group,intervals)


def test_late_binding_repair_cannot_inject_an_unsupported_overlap_enum():
    goal={'status':'ready','structured_goal':{'intent':'binding'}}
    with pytest.raises(ValueError,match='binding_constraint_repair_contract_invalid'):
        native._merge_binding_constraint_repair(goal,{'status':'updated','structured_goal_patch':{
            'aso_overlap_requirement':'at_least_8_bases'}})
    result=native._merge_binding_constraint_repair(goal,{'status':'updated','structured_goal_patch':{
        'aso_overlap_requirement':'any_overlap','minimum_overlap_bp':8}})
    assert result['structured_goal']['minimum_overlap_bp']==8
    assert 'aso_overlap_requirement' not in goal['structured_goal']


def test_binding_repair_retries_schema_fault_then_preserves_cdna_bounds(monkeypatch):
    responses=[{'status':'updated','structured_goal_patch':{'aso_overlap_requirement':'unsupported'}},
               {'status':'updated','structured_goal_patch':{'region_constraints':[{
                   'source_type':'explicit_rows','coordinate_space':'cdna','label':'window',
                   'aso_overlap_requirement':'any_overlap','minimum_overlap_bp':8,
                   'intervals_bed0_half_open':[{'chrom':'TXA.3','start':19,'end':80}]}]}}]
    calls=[]
    async def fake_run(*args,**kwargs):
        calls.append(kwargs['input'])
        return type('Response',(),{'final_output':json.dumps(responses[len(calls)-1])})()
    monkeypatch.setattr(native.Runner,'run',fake_run)
    raw={'status':'ready','structured_goal':{'intent':'binding','transcript_id':'TXA.3'}}
    result=asyncio.run(native._repair_binding_site_constraints_if_needed('Use the declared mature-RNA window.',raw))
    assert len(calls)==2
    constraint=result['structured_goal']['region_constraints'][0]
    assert constraint['coordinate_space']=='cdna' and constraint['minimum_overlap_bp']==8
    assert constraint['intervals_bed0_half_open'][0]['start']==19
    assert 'flanks' not in result['structured_goal']
