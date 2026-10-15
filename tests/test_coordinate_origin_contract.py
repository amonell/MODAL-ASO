from copy import deepcopy

from aso_agents.coordinate_contract import coordinate_execution_summary
from aso_agents.region_projection import genomic_sources


def _goal(space='genomic'):
    return {'transcript_lock': {'value': 'TX.4'}, 'design_programs': [{
        'program_id': 'region', 'modality': 'steric', 'intent': 'binding',
        'transcript_id': 'TX.4', 'region_constraints': [{
            'source_type': 'explicit_rows', 'coordinate_space': space,
            'coordinate_convention': 'bed0_half_open',
            'aso_overlap_requirement': 'any_overlap',
            'intervals_bed0_half_open': [{'chrom': 'chr8', 'start': 15, 'end': 35}]}]}],
        'evaluation_targets': [{'target_type': 'region', 'transcript_id': 'TX.4',
                                'chrom': 'chr8', 'start': 16, 'end': 35}]}


def test_single_interval_exposes_actual_origin_without_guessing_from_small_numbers():
    goal = _goal()
    before = deepcopy(goal)
    summary = coordinate_execution_summary(goal)
    assert summary['constraints'][0]['native_coordinate_space'] == 'genomic'
    assert summary['constraints'][0]['native_reference_kind'] == 'chromosome'
    assert summary['constraints'][0]['intervals_bed0_half_open'][0]['start'] == 15
    assert '/design_programs/0/region_constraints/0/coordinate_space' in summary['audit_scope_pointers']
    assert '/evaluation_targets/0/start' in summary['audit_scope_pointers']
    assert '/evaluation_targets' not in summary['audit_scope_pointers']
    assert summary['genomic_evaluation_targets'] == goal['evaluation_targets']
    assert goal == before


def test_coordinate_focus_keeps_evidence_and_allows_coherent_target_repairs():
    from aso_agents import pipeline_tools as p
    goal = _goal()
    prefix = '/tasks/0/context/structured_goal'
    evidence = [{'evidence_id': 'request', 'kind': 'user_request',
                 'text': 'Target mature-RNA positions 16–35 on TX.4, one-based inclusive.'}]
    envelope = {'tasks': [{'task_id': 'design', 'task_type': 'design_new_asos',
                           'context': {'structured_goal': goal}}]}
    scope = coordinate_execution_summary(goal)['audit_scope_pointers']
    prompt = p._goal_audit_prompt(request_envelope=envelope, validation_issues=[],
        issue_records=[], evidence_sources=evidence, semantic_audit=True,
        semantic_scope_pointers=[prefix + path for path in scope])
    protocol = p._goal_audit_protocol_payload(prompt)
    destinations = {(item['operation'], item['json_pointer'])
                    for item in p._semantic_goal_audit_patch_destinations(protocol)}
    assert ('replace', prefix + '/design_programs/0/region_constraints/0/coordinate_space') in destinations
    assert ('remove', prefix + '/evaluation_targets/0/start') in destinations
    assert ('remove', prefix + '/design_programs/0/region_constraints/0/intervals_bed0_half_open/0/chrom') not in destinations
    assert not any(path.endswith(('/exon_index', '/site_kind', '/label', '/operator', '/allow_coordinate_conversion'))
                   for _, path in destinations)
    assert not any(path.endswith(('/coordinate_convention', '/transcript_id')) for _, path in destinations)
    assert not any(path.endswith(('/intervals_bed0_half_open', '/region_constraints', '/evaluation_targets'))
                   for _, path in destinations)
    assert all(path == prefix + '/evaluation_targets' or
               path == prefix + '/design_programs/0/region_constraints' or
               path.startswith(prefix + '/evaluation_targets/') or
               path.startswith(prefix + '/design_programs/0/region_constraints/')
               for _, path in destinations)
    assert protocol['evidence_sources'][0]['text'] == evidence[0]['text']
    assert protocol['coordinate_execution'][0]['constraints'][0]['native_reference_kind'] == 'chromosome'


def test_mixed_coordinate_scopes_and_program_windows_remain_distinct():
    goal = _goal('cdna')
    goal['design_programs'].append({'start': 101, 'end': 120, 'coordinate_space': 'genomic',
                                    'coordinate_convention': 'one_based_inclusive'})
    summary = coordinate_execution_summary(goal)
    assert summary['constraints'][0]['native_reference_kind'] == 'transcript'
    assert summary['program_windows'][0]['coordinate_space'] == 'genomic'
    assert '/design_programs/1/coordinate_space' in summary['audit_scope_pointers']
    assert coordinate_execution_summary({'transcript_lock': {'value': 'TX.4'}}) is None


def test_explicit_transcript_origin_uses_attested_exons_in_both_orientations():
    for strand, exons in [('+', [{'start': 101, 'end': 120}, {'start': 501, 'end': 540}]),
                          ('-', [{'start': 521, 'end': 540}, {'start': 101, 'end': 140}])]:
        goal = {'isoform_transcripts': {'TX.4': {'chrom': 'chr8', 'strand': strand, 'exons_tx': exons}}}
        source = genomic_sources(goal, {'coordinate_space': 'cdna',
                                       'materialized_rows': [['TX.4', 15, 35]]})[0]
        assert sum(row[2] - row[1] for row in source['genomic_rows']) == 20
        assert len(source['genomic_rows']) == 2
        assert source['source_row'] == ['TX.4', 15, 35]
        assert all(row[5] == strand for row in source['genomic_rows'])
