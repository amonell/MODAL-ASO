from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
from aso_agents import pipeline_tools as pipeline


@pytest.mark.parametrize('stage', ['eval', 'router'])
@pytest.mark.parametrize('persist_jobs', [False, True])
def test_requested_splice_branch_is_not_disabled_by_aggregate_flag(monkeypatch, stage, persist_jobs):
    programs = [
        {'program_id':'cleavage','modality':'gapmer','intent':'knockdown',
         'run_splice_eval':False,'run_splice_router':False},
        {'program_id':'splice','modality':'steric','intent':'splice_modulation',
         'run_splice_eval':True,'run_splice_router':True},
    ]
    goal = {'design_programs':programs,'run_splice_eval':False,'run_splice_router':False}
    final = {'run_splice_eval':False,'run_splice_router':False}
    if persist_jobs:
        final['steric_design_jobs'] = [{**programs[1], 'job_id':'job-splice'}]
    context = {'structured_goal':goal,'final_payload':final,'payload':{}}
    before = deepcopy(context)
    monkeypatch.setattr(pipeline, '_goal_payload_context', lambda _:context)
    assert getattr(pipeline, '_resolve_run_splice_'+stage)('unused') is True
    assert context == before


@pytest.mark.parametrize('stage', ['eval', 'router'])
def test_disabled_local_splice_branch_is_not_enabled_by_sibling_context(monkeypatch, stage):
    program = {'program_id':'occupancy','modality':'steric','intent':'binding',
               'target_region':'utr3','run_splice_eval':False,'run_splice_router':False}
    context = {'structured_goal':{'design_programs':[program],
                                 'run_splice_eval':True,'run_splice_router':True},
               'final_payload':{},'payload':{}}
    monkeypatch.setattr(pipeline, '_goal_payload_context', lambda _:context)
    assert getattr(pipeline, '_resolve_run_splice_'+stage)('unused') is False


def test_plural_transcript_contract_requires_exact_binding_to_every_member(tmp_path):
    program = {'program_id':'shared','modality':'gapmer','intent':'knockdown',
               'transcript_ids':['TXA.1','TXB.2'],'transcript_id':'TXA.1'}
    rows = [{'Name':'shared','isoform_uniqueness_labels':['TXA','TXB']},
            {'Name':'anchor-only','isoform_uniqueness_labels':['TXA']}]
    path=tmp_path/'scored.json';path.write_text(json.dumps({'rows':rows}))
    result=pipeline._apply_gapmer_constraints_to_result(
        gapmer_results={'scored_tiles_path':str(path)},
        structured_goal={'design_programs':[program]}, program_scope=program,
        runtime=SimpleNamespace(filter_tiles_json=lambda p, _: {'applied':False,'output_path':p}))
    assert 'error' not in result
    actual=json.loads(open(result['gapmer_results']['scored_tiles_path']).read())['rows']
    assert [row['Name'] for row in actual] == ['shared']


def test_selected_panel_chemistry_stays_with_producing_program(tmp_path):
    import csv
    from aso_agents.design_delivery import write_design_panel
    goal={'chemistry':'steric-hindrance','design_programs':[
        {'program_id':'cleavage','modality':'gapmer','chemistry':'5-10-5'},
        {'program_id':'occupancy','modality':'steric','chemistry':'steric-hindrance'}],
        'chemistry_profiles':{'by_program':{
            'cleavage':{'recommendation':{'CHEMISTRY':'5-10-5','wing_chemistry':'2MOE','gap_chemistry':'DNA','backbone':'PS'}},
            'occupancy':{'recommendation':{'CHEMISTRY':'steric-hindrance','wing_chemistry':'2MOE','gap_chemistry':'none','backbone':'PS'}}}}}
    rows=[{'program_id':pid,'sequence':'ACGTACGTACGTACGTACGT',
           'target_binding_sequence':'ACGTACGTACGTACGTACGT',
           'synthesized_antisense_sequence':'ACGTACGTACGTACGTACGT','length':20}
          for pid in ['cleavage','occupancy','unknown']]
    write_design_panel({'rows':rows},goal,tmp_path)
    exported=list(csv.DictReader(open(tmp_path/'design_panel.csv')))
    assert 'gap_sugar=DNA' in exported[0]['chemistry_description']
    assert 'steric-hindrance' not in exported[0]['chemistry_description']
    assert 'steric-hindrance' in exported[1]['chemistry_description']
    assert exported[2]['chemistry_description'] == ''
