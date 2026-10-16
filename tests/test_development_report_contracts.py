from copy import deepcopy
import hashlib
import json

import pytest
import run_goal_intake as runtime
from aso_agents.evaluation_evidence import attach_measurement_references, bind_assessment_measurements
from aso_agents.reference_provenance import evaluation_catalog_provenance
from aso_agents import pipeline_tools


@pytest.mark.parametrize('modality', ['steric', 'gapmer'])
def test_single_modality_export_is_not_counted_as_both(modality, tmp_path, monkeypatch):
    folder = tmp_path / 'candidates'; folder.mkdir()
    path = folder / (modality + '_uniqueness_ranked.csv'); path.write_text('fixture')
    monkeypatch.setattr(runtime, '_load_ranked_asos_csv', lambda p: [{'sequence': 'ACGT', 'source': str(p)}])
    monkeypatch.setattr(runtime, 'resolve_run_path', lambda root, name: folder/'generated_aso_set.json' if name == 'generated_aso_set.json' else tmp_path/name)
    own, other = ('tiling_results', 'gapmer_tiling_results') if modality == 'steric' else ('gapmer_tiling_results', 'tiling_results')
    goal = {own: {'uniqueness_csv': str(path)}}
    assert len(runtime._collect_ranked_asos(goal, str(tmp_path), own)) == 1
    assert runtime._collect_ranked_asos(goal, str(tmp_path), other) == []
    second = folder / ('other_uniqueness_ranked.csv'); second.write_text('fixture')
    goal[other] = {'uniqueness_csv': str(second)}
    # The same bases in two real chemical modalities remain two separate designs.
    assert len(runtime._collect_ranked_asos(goal, str(tmp_path), other)) == 1


def test_reference_export_attests_each_transcript_and_keeps_unknown_versions(tmp_path):
    catalog = {'primary': {'transcript_id': 'TX', 'metadata': {'versioned_id': 'TX.2'}},
               'alternate': {'transcript_id': 'TY', 'metadata': {'versioned_id': 'TY.8'}},
               'unknown': {'transcript_id': 'TZ'}}
    sequences = {'primary': 'ACGT', 'alternate': 'AAAACCCC'}
    rows = [{'mapping_hits': [{'isoform_label': 'alternate', 'transcript_id': 'TY'}],
             'near_match_candidates': [{'transcript_id': 'TZ', 'mismatch_count': 2}]}]
    before = deepcopy(catalog)
    result = evaluation_catalog_provenance(catalog, sequences, rows, tmp_path)
    assert rows[0]['mapping_hits'][0]['reference_identity']['transcript_id'] == 'TY.8'
    assert not rows[0]['near_match_candidates'][0]['reference_identity']['version_attested']
    snapshot = (tmp_path/'evaluation_reference_catalog.json').read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == result['snapshot_sha256']
    assert json.loads(snapshot)['entries']['alternate']['mature_cdna_sequence'] == 'AAAACCCC'
    assert catalog == before
    assert 'efficacy' not in rows[0]['mapping_hits'][0]


def test_reference_tokens_and_confirmed_sequence_role_share_one_report_source():
    summary = attach_measurement_references({'provided_asos': [{'name': 'A',
        'requested_feature_membership': [{'feature': 'exon', 'exon_index': 2, 'chrom': '9', 'start': 1234,
                                         'end': 1279, 'transcript_id': 'TX.2', 'coordinate_system': '1based_inclusive'}],
        'orientation_interpretation': {'sequence_role_confirmation': {'role': 'antisense_oligo_sequence'},
            'evaluated_molecule_sequence': 'AACC', 'requires_confirmation': False}}]})
    token = summary['provided_asos'][0]['reference_tokens']['feature_1']
    result = bind_assessment_measurements({'evaluation_assessments': [{'name': 'A',
        'mechanistic_plausibility': 'Conditional hypothesis at ' + token}], 'markdown': ''}, summary)
    assert '1279' in result['evaluation_assessments'][0]['mechanistic_plausibility']
    assert 'user_confirmed' in result['markdown'] and 'AACC' in result['markdown']
    assert summary['provided_asos'][0]['resolved_sequence_identity']['role'] == 'antisense_oligo_sequence'


def test_top_level_chemistry_matches_common_program_length_without_collapsing_mixed_lengths():
    goal = {'intent':'binding', 'chemistry':{'recommendation':{'CHEMISTRY':'steric-hindrance','length':20}},
        'design_programs':[{'program_id':'a','modality':'steric','length':18}]}
    final = {'chemistry_profiles':{'by_program':{'a':{'recommendation':{'CHEMISTRY':'steric-hindrance','length':20}}}}}
    profiles = pipeline_tools._ensure_program_chemistry_profiles(structured_goal=goal, final_payload=final, modality='steric')
    assert goal['chemistry']['recommendation']['length'] == 18
    assert profiles['by_program']['a']['recommendation']['length'] == 18
    goal['design_programs'].append({'program_id':'b','modality':'steric','length':22})
    profiles = pipeline_tools._ensure_program_chemistry_profiles(structured_goal=goal, final_payload=final, modality='steric')
    assert profiles['by_modality']['steric']['mixed']
    assert {p['recommendation']['length'] for p in profiles['by_program'].values()} == {18,22}


def test_unexecuted_conditional_branch_is_not_reported_as_outputs_present():
    goal = {'design_programs':[{'program_id':'fallback','modality':'gapmer','intent':'knockdown'}]}
    result = pipeline_tools._build_branch_outcomes_for_report(goal,{})
    assert result['gapmer']['status'] == 'skipped'


def test_sandbox_analysis_result_is_available_to_composite_report(tmp_path):
    result = tmp_path / 'analysis.json'
    result.write_text(json.dumps({'results': [{'candidate_id': 'sample', 'status': 'indeterminate'}]}))
    assert pipeline_tools._load_task_artifact_payload({'result_json': str(result)}) == {
        'results': [{'candidate_id': 'sample', 'status': 'indeterminate'}]
    }


def test_mixed_analysis_and_design_uses_complete_composite_path():
    tasks = [
        {'task_id': 'audit', 'task_type': 'sandbox_analysis', 'payload': {'results': [{'id': 'sample'}]}},
        {'task_id': 'design', 'task_type': 'design_new_asos', 'payload': {
            'child_run_dir': 'child_runs/design', 'design_outcome': {'outcome_status': 'completed_with_candidates',
                                                                     'candidate_count': 2}}},
    ]
    assert pipeline_tools._build_deterministic_composite_response({}, {}, completed_tasks=tasks) is None


def test_requested_export_receives_bounded_native_evaluation_evidence(monkeypatch):
    row = {'name': 'molecule', 'sequence': 'AACCGG', 'principal_reference_exon_membership':
           {'status': 'resolved', 'exons': [{'transcript_exon_ordinal': 3}]},
           'mapping_hits': [{'transcript_id': 'TX.2', 'cdna_start': 11, 'cdna_end': 16,
                             'reference_identity': {'sequence_evidence': {'mature_cdna_sequence': 'A' * 1000}}}]}
    monkeypatch.setattr(pipeline_tools, '_collect_completed_task_outputs', lambda state: ({}, [
        {'task_id': 'evaluate', 'task_type': 'evaluate_existing_aso_set',
         'payload': {'evaluation_summary': {'context_summary': {'transcript_id': 'TX.2'},
                                            'provided_asos': [row]}}}]))
    evidence = pipeline_tools._bounded_completed_evaluation_evidence({})
    assert evidence[0]['context_summary']['transcript_id'] == 'TX.2'
    assert evidence[0]['rows'][0]['principal_reference_exon_membership']['exons'][0]['transcript_exon_ordinal'] == 3
    assert evidence[0]['rows'][0]['mapping_hits'] == [{'transcript_id': 'TX.2', 'cdna_start': 11, 'cdna_end': 16}]
    assert evidence[0]['rows'][0]['mapping_hits_total'] == 1
    assert evidence[0]['rows'][0]['mapping_hits_preview_truncated'] is False


def test_multi_set_comparison_composes_evidence_and_survives_into_parent_report(monkeypatch, tmp_path):
    sets = [{'set_id':key,'label':key,'source_type':'generated', 'sequence_set':{seq},'by_sequence':{},
             'rows':[{'Sequence':seq,'Score':0,'Off_target_exact_count':0,'Off_target_1mm_count':None,
                      'isoform_uniqueness_score':0.5,'isoform_uniqueness_class':'target_specific'}]}
            for key,seq in [('a','AACC'),('b','CCAA')]]
    prepared = {'status':'ok','state_path':str(tmp_path/'state.json'),'state':{'request':'Compare strengths.'},
        'run_dir':str(tmp_path),'task_id':'compare','task':{'task_type':'compare_aso_sets'},
        'task_request_envelope':{'request_text':'Compare strengths.'},'task_context':{}}
    monkeypatch.setattr(pipeline_tools,'_prepare_v2_task_execution',lambda **_: prepared)
    monkeypatch.setattr(pipeline_tools,'_load_task_payload_by_type',lambda *_, **__: {})
    monkeypatch.setattr(pipeline_tools,'_collect_aso_compare_sets',lambda **_: sets)
    calls, written = [], {}
    write_native = pipeline_tools._write_v2_task_result_files
    def compose(**kw):
        calls.append(kw)
        return {'markdown':'Both sets tie on the measured exact-match criterion; efficacy cannot be ranked.'}
    def write(**kw):
        written.update(kw)
        return write_native(**kw)
    monkeypatch.setattr(pipeline_tools,'_build_compare_aso_sets_response',compose)
    monkeypatch.setattr(pipeline_tools,'_write_v2_task_result_files',write)
    monkeypatch.setattr(pipeline_tools,'_complete_v2_task',lambda **kw: kw)
    pipeline_tools.tool_compare_aso_sets(str(tmp_path/'state.json'))
    assert len(calls) == 1
    row = calls[0]['comparison_payload']['candidate_evidence_by_set'][0]['top_rows_in_source_order'][0]
    assert row['score'] == 0 and row['off_target_exact_count'] == 0
    assert 'off_target_1mm_count' not in row
    assert row['recorded_scientific_fields']['isoform_uniqueness_class'] == 'target_specific'
    response = pipeline_tools._build_deterministic_composite_response({}, {}, completed_tasks=[
        {'task_id':'compare','task_type':'compare_aso_sets','payload':written['payload']}])
    assert 'Both sets tie' in response['markdown']
