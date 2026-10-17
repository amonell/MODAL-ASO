"""Output/reference paths must not override a semantic new-design request."""
import pytest
from aso_agents import pipeline_tools

@pytest.mark.parametrize('action',[
    'Design antisense oligonucleotide candidates',
    'Generate a panel of antisense oligonucleotides',
    'Create five gapmers',
    'Design ASO candidates',
])
def test_design_request_with_output_path_is_not_forced_to_analysis(tmp_path,action):
    output=tmp_path/'results';output.mkdir()
    request=f'{action} for human MAPT. Return sequences and target locations in CSV files under {output}.'
    assert not pipeline_tools._request_text_is_analysis_only_artifact_request(request)
    assert pipeline_tools._deterministic_artifact_analysis_envelope(request_text=request,request_id='test',raw_envelope={}) is None

def test_ordinary_analysis_is_left_to_semantic_intake(tmp_path):
    assert not pipeline_tools._request_text_is_analysis_only_artifact_request(f'Review and rank the results in {tmp_path}.')

def test_explicit_no_design_instruction_keeps_existing_artifact_guard(tmp_path):
    assert pipeline_tools._request_text_is_analysis_only_artifact_request(f'Evaluate the existing output in {tmp_path}. Do not design new ASOs or rerun tiling.')

@pytest.mark.parametrize('task_index',[0,1,2])
def test_nested_array_repair_is_not_confused_with_whole_task_replacement(task_index):
    from copy import deepcopy
    task={'task_id':'design','task_type':'design_new_asos','context':{'structured_goal':{
        'source_attribution':{'required':True,'sources':['GENCODE']*(task_index+1)}}}}
    envelope={'tasks':[deepcopy(task) for _ in range(task_index+1)]}
    pointer=f'/tasks/{task_index}/context/structured_goal/source_attribution/sources/{task_index}'
    result=pipeline_tools._apply_goal_audit_patches(envelope,[{
        'op':'replace','path':pointer,'value':'GENCODE v42','evidence_refs':['request'],
        'evidence_quote':'GENCODE v42','reason':'Preserve the requested annotation release.'}],
        issue_records=[{'json_pointer':pointer}],evidence_sources=[{'evidence_id':'request','kind':'user_request','text':'Use GENCODE v42.'}])
    assert result['rejected']==[]
    assert result['request_envelope']['tasks'][task_index]['context']['structured_goal']['source_attribution']['sources'][task_index]=='GENCODE v42'
    assert result['request_envelope']['tasks'][task_index]['task_id']=='design'
