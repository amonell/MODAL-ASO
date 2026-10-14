from copy import deepcopy
import json

import pytest
import run_goal_intake as runtime


def whole_gene_goal():
    return {'gene_symbol':'example', 'intent':'knockdown',
        'transcript_policy':'user_locked', 'transcript_lock':{'value':'reference.3'},
        'evaluation_targets':[
            {'target_type':'gene', 'objective_type':'primary_outcome', 'desired_delta':'decrease', 'gene_symbol':'example'},
            {'target_type':'exon', 'objective_type':'identity_anchor', 'transcript_id':'reference.3', 'exon_label':4}]}


def test_coordinate_lock_does_not_override_typed_whole_gene_biological_objective():
    goal = whole_gene_goal(); before = deepcopy(goal)
    assert runtime._resolve_uniqueness_target_label(goal, 'reference') is None
    assert runtime._resolve_gapmer_uniqueness_context(goal, 'reference')['mode'] == 'gene_wide'
    assert goal == before


@pytest.mark.parametrize('override', ['explicit_scope','transcript_outcome','isoform_outcome','splice_objective','protected','other_gene','legacy'])
def test_true_selective_objectives_and_legacy_scope_remain_unchanged(override):
    goal = whole_gene_goal()
    if override == 'explicit_scope':
        goal['target_scope'] = 'isoform_specific'
    elif override in {'transcript_outcome','isoform_outcome'}:
        goal['evaluation_targets'][0].update(target_type=override.split('_')[0], transcript_id='reference.3')
    elif override == 'splice_objective':
        goal['splice_isoform_objectives'] = [{'isoform_label':'reference','desired_abundance':'decrease','role':'pathogenic'}]
    elif override == 'protected':
        goal['preservation_targets'] = [{'action':'preserve','preservation_target_type':'transcript','transcript_id':'protected.1'}]
    elif override == 'other_gene':
        goal['evaluation_targets'][0]['gene_symbol'] = 'different_gene'
    else:
        goal.pop('evaluation_targets')
    assert runtime._resolve_uniqueness_target_label(goal, 'reference') == 'reference'


def test_increasing_or_monitoring_a_gene_does_not_imply_whole_gene_knockdown_scope():
    goal = whole_gene_goal();goal['evaluation_targets'][0]['desired_delta']='increase'
    assert runtime._resolve_uniqueness_target_label(goal,'reference') == 'reference'
    goal['evaluation_targets'][0]['objective_type']='monitoring_safeguard'
    goal['evaluation_targets'][0]['desired_delta']='decrease'
    assert runtime._resolve_uniqueness_target_label(goal,'reference') == 'reference'


def test_ranking_prefers_catalog_breadth_for_gene_goal_and_specificity_for_isoform_goal(tmp_path):
    goal = whole_gene_goal()
    rows = [{'Name':'broad', 'Sequence':'AAAA', 'Synthesis':'TTTT', 'GC':0.5},
            {'Name':'specific', 'Sequence':'CCCC', 'Synthesis':'GGGG', 'GC':0.5}]
    kmer_index = {4:{'AAAA':{'reference','other_1','other_2'},'CCCC':{'reference'}}}
    source = tmp_path/'tiles.json';source.write_text(json.dumps({'rows':rows}))
    outputs = {}
    for mode in ('gene_wide','isoform_specific'):
        local = deepcopy(goal)
        if mode == 'isoform_specific':local['target_scope'] = mode
        context = runtime._resolve_gapmer_uniqueness_context(local,'reference')
        path = runtime._annotate_gapmer_uniqueness(str(source),kmer_index,context['target_label'],4,output_name=mode+'.json')
        outputs[mode] = json.loads(open(path).read())['rows']
    assert [r['Name'] for r in outputs['gene_wide']] == ['broad','specific']
    assert [r['isoform_uniqueness_score'] for r in outputs['gene_wide']] == [0.75,0.25]
    assert [r['Name'] for r in outputs['isoform_specific']] == ['specific','broad']
    assert [r['isoform_uniqueness_score'] for r in outputs['isoform_specific']] == [1.0,1/3]
    assert json.loads(source.read_text())['rows'] == rows
