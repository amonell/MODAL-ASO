from copy import deepcopy

import pytest

from aso_agents.region_overlap import effective_program_constraint_summary


def rule(start, end, operator='intersect', relation='any_overlap'):
    return {'source_type': 'explicit_rows', 'coordinate_space': 'cdna',
            'intervals_bed0_half_open': [{'chrom': 'TRANSCRIPT.8', 'start': start, 'end': end}],
            'operator': operator, 'aso_overlap_requirement': relation}


@pytest.mark.parametrize('operator', ['union', 'include', 'intersect', 'exclude'])
@pytest.mark.parametrize('length', [18, 23, 26])
def test_effective_predicates_include_global_and_local_order(operator, length):
    goal = {'region_constraints': [rule(3, 93), rule(31, 35)],
            'design_programs': [{'program_id': 'branch', 'length': length,
                                 'region_constraints': [rule(61, 65, operator)]}]}
    before = deepcopy(goal)
    summary = effective_program_constraint_summary(goal)[0]
    pointers = ['/region_constraints/0', '/region_constraints/1',
                '/design_programs/0/region_constraints/0']
    assert [x['constraint_pointer'] for x in summary['ordered_predicates']] == pointers
    assert summary['inherited_constraint_pointers'] == pointers[:2]
    assert summary['local_constraint_pointers'] == pointers[2:]
    # Independent three-predicate Boolean oracle; a local union can broaden a
    # global restriction, while an exclusion cannot restore a failed global rule.
    def expected(bits):
        outer = bits[0] and bits[1]
        if operator in {'union', 'include'}:
            return outer or bits[2]
        return outer and (not bits[2] if operator == 'exclude' else bits[2])
    assert summary['allows_candidate_matching_no_constraints'] == expected([False]*3)
    assert summary['constraints_allowed_to_fail_when_all_others_match'] == [
        pointers[i] for i in range(3) if expected([j != i for j in range(3)])]
    assert summary['constraints_sufficient_alone_for_admission'] == [
        pointers[i] for i in range(3) if expected([j == i for j in range(3)])]
    assert {x['candidate_length_nt'] for x in summary['explicit_interval_geometry']} == {length}
    assert goal == before


def test_inheritance_reaches_programs_without_local_rules_and_preserves_scope():
    goal = {'region_constraints': [rule(20, 25, relation='contains_feature')],
            'design_programs': [{'program_id': 'first', 'length': 4},
                                {'program_id': 'second', 'length': 12}]}
    result = effective_program_constraint_summary(goal)
    assert [r['program_id'] for r in result] == ['first', 'second']
    assert [r['local_constraint_pointers'] for r in result] == [[], []]
    assert [r['explicit_interval_geometry'][0]['single_interval_relation_possible']
            for r in result] == [False, True]


def test_local_rules_are_not_inherited_by_siblings_or_promoted_to_global():
    goal = {'design_programs': [
        {'program_id': 'one', 'region_constraints': [rule(10, 20, 'exclude')]},
        {'program_id': 'two'},
        {'program_id': 'three', 'region_constraints': [rule(30, 50)]}]}
    result = effective_program_constraint_summary(goal)
    assert [r['program_scope'] for r in result] == ['/design_programs/0', '/design_programs/2']
    assert all(r['inherited_constraint_pointers'] == [] for r in result)
    assert result[0]['allows_candidate_matching_no_constraints'] is True
    assert result[1]['allows_candidate_matching_no_constraints'] is False


@pytest.mark.parametrize('focused', ['global', 'local'])
def test_focused_audit_keeps_cross_scope_context_without_expanding_patch_permission(focused):
    from aso_agents import pipeline_tools as tools
    goal = {'gene_symbol': 'GENERIC', 'species': 'human',
            'region_constraints': [rule(5, 95), rule(39, 45)],
            'design_programs': [{'program_id': 'screen', 'modality': 'steric',
                                 'region_constraints': [dict(rule(39, 45, 'exclude'), minimum_overlap_bp=4)]},
                                {'program_id': 'other', 'modality': 'steric'}]}
    envelope = {'tasks': [{'task_id': 'd', 'task_type': 'design_new_asos',
                           'context': {'structured_goal': goal}}]}
    prefix = '/tasks/0/context/structured_goal'
    scope = prefix + ('/region_constraints' if focused == 'global' else '/design_programs/0/region_constraints')
    before = deepcopy(envelope)
    prompt = tools._goal_audit_prompt(request_envelope=envelope, validation_issues=[],
        issue_records=[], evidence_sources=[{'evidence_id': 'request', 'kind': 'user_request',
        'text': 'The first branch may overlap the small feature by at most three bases.'}],
        semantic_audit=True, semantic_scope_pointers=[scope])
    protocol = tools._goal_audit_protocol_payload(prompt)
    effective = protocol['effective_program_region_constraints'][0]['programs']
    assert len(effective) == 2
    assert effective[0]['ordered_predicates'][-1]['minimum_overlap_bp'] == 4
    assert effective[0]['inherited_constraint_pointers'] == ['/region_constraints/0', '/region_constraints/1']
    assert effective[1]['inherited_constraint_pointers'] == effective[0]['inherited_constraint_pointers']
    destinations = tools._semantic_goal_audit_patch_destinations(protocol)
    assert destinations and all(d['json_pointer'] == scope or d['json_pointer'].startswith(scope + '/')
                                for d in destinations)
    assert protocol['current_request_envelope'] == before == envelope


def test_diagnostics_do_not_resolve_unbound_features_or_infer_program_ids():
    goal = {'region_constraints': [{'source_type': 'transcript_feature', 'feature_name': 'cds',
                                    'operator': 'intersect'}],
            'design_programs': [{'region_constraints': [rule(10, 12)]}]}
    result = effective_program_constraint_summary(goal)[0]
    assert result['program_id'] is None
    assert len(result['ordered_predicates']) == 2
    assert len(result['explicit_interval_geometry']) == 1
    assert result['explicit_interval_geometry'][0]['candidate_length_nt'] is None
