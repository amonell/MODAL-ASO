from copy import deepcopy
import json

import pytest
import run_goal_intake as runtime
from aso_agents import pipeline_tools
from aso_agents.binding_region_reports import build_binding_region_report, memberships, binding_region_markdown, recorded_screening_summary


def test_region_sets_keep_containment_and_overlap_separate_from_response_route(tmp_path):
    constraints = [{'constraint_id':name,'label':name,'aso_overlap_requirement':rule,
                    'materialized_rows':[['chr8',a,b,name,0,'+']]}
                   for name,a,b,rule in [('acceptor',100,130,'full_containment'),
                                         ('donor',180,220,'full_containment'),('regulatory',200,240,'any_overlap')]]
    goal = {'materialized_region_constraints':{'by_program':{'p':constraints}}}
    rows = [{'Chrom':'chr8','Start':a,'End':b,'Sequence':'AACC','Synthesis':'GGTT','Score':str(score),
             'Off_target_exact_count':0,'selected_route_label':'acceptor@100'}
            for a,b,score in [(110,114,3),(210,214,2),(238,242,1)]]
    path = tmp_path/'scored.json';path.write_text(json.dumps({'rows':rows}))
    result = build_binding_region_report(goal,{'tiling_results':{'scored_tiles_path':str(path)}},tmp_path,
        runtime._tile_row_overlap_intervals,{})
    assert [g['candidate_count'] for g in result['sets']] == [1,1,2]
    assert result['sets'][0]['rows'][0]['binding_intervals_bed0_half_open'] == [('chr8',110,114)]
    assert memberships(rows[2],goal,runtime._tile_row_overlap_intervals)[0]['constraint_id'] == 'regulatory'
    assert len(memberships(rows[1],goal,runtime._tile_row_overlap_intervals)) == 2
    assert 'GGTT' in binding_region_markdown(result)
    assert all(g['rows'][0]['Off_target_exact_count'] == 0 for g in result['sets'])
    full=json.loads((tmp_path/'requested_binding_site_sets.json').read_text())
    assert full['sets'][2]['rows'][1]['selected_route_label'] == 'acceptor@100'


def test_report_distinguishes_same_region_label_in_different_design_programs():
    base = {'constraint_id': 'shared-window', 'label': 'shared-window',
            'aso_overlap_requirement': 'any_overlap', 'minimum_overlap_bp': 1,
            'intervals_bed0_half_open': [['chr8', 100, 130]], 'rows': [],
            'candidate_count': 0, 'binding_record_count': 0,
            'distinct_molecule_count': 0, 'status': 'no_qualifying_candidates_in_recorded_scored_pool'}
    report = {'artifact_path': 'sets.json', 'interpretation': 'Per-program pools.',
              'ranking': 'Native rank.', 'sets': [{**base, 'program_id': 'first'},
                                               {**base, 'program_id': 'second'}]}
    text = binding_region_markdown(report)
    assert '### shared-window (design program: first)' in text
    assert '### shared-window (design program: second)' in text


@pytest.mark.parametrize('strand', ['+','-'])
def test_existing_exact_exon_does_not_create_modified_reference_or_false_absence(strand):
    exons=[{'index':i,'start':a,'end':b,'seq_region_name':'8','contains_coding':True}
           for i,(a,b) in enumerate([(100,130),(200,230),(300,330)] if strand=='+' else [(300,330),(200,230),(100,130)],1)]
    catalog={'normal':{'transcript_id':'TX.2','chrom':'8','strand':strand,'exons_tx':exons},
             'alternate':{'transcript_id':'TY.3','chrom':'8','strand':strand,'exons_tx':deepcopy(exons)}}
    target={'chrom':'chr8','start':200,'end':230,'strand':strand,'isoform_label':'normal',
            'requested_exon_label':2,'desired_delta':'increase'}
    goal={'isoform_transcripts':catalog,'explicit_splice_targets':[target]}
    runtime._materialize_modified_isoforms_for_explicit_targets(goal,{})
    assert set(goal['isoform_transcripts']) == {'normal','alternate'}
    assert target['synthetic_exon_created'] is False
    plan=runtime._explicit_splice_plan_from_targets([target],catalog)
    assert len(plan)==2 and all(set(p['present_in'])=={'normal','alternate'} and p['absent_in']==[] for p in plan)


def test_reference_descriptions_do_not_guess_conversion_direction_from_accession_order():
    text='Increase exon inclusion using ENST00000000001 exon numbering; ENST00000000002 lacks this exon.'
    assert pipeline_tools._request_isoform_conversion_preseed_from_text(text) == {}
    text='Use ENST00000000001 as the skipped isoform to decrease and ENST00000000002 as the included isoform to increase.'
    result=pipeline_tools._normalized_design_objective_preseed(pipeline_tools._request_isoform_conversion_preseed_from_text(text))
    objectives={o['transcript_id']:o['desired_abundance'] for o in result['splice_isoform_objectives']}
    assert objectives=={'ENST00000000001':'decrease','ENST00000000002':'increase'}


def test_no_repeats_does_not_hide_recorded_off_targets_or_missing_screens(tmp_path):
    path=tmp_path/'scored.json';path.write_text(json.dumps({'rows':[
        {'Off_target_exact_count':2,'Off_target_1mm_count':0,'repeat_hit_count_exact':0},
        {'Off_target_exact_count':0,'repeat_hit_count_exact':0},
        {'Off_target_exact_count':None,'repeat_hit_count_exact':0}]}))
    result=recorded_screening_summary({}, {'tiling_results':{'scored_tiles_path':str(path)}})[0]['measurements']
    assert result['exact_off_target_count']['rows_with_hits']==1
    assert result['exact_off_target_count']['unknown_rows']==1
    assert result['one_mismatch_off_target_count']['unknown_rows']==2
    assert result['alternate_exact_repeat_count']['rows_with_hits']==0
