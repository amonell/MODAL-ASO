"""Distinct junctions sharing transcripts still have independent predictions."""
import pytest
import run_goal_intake as runtime


def candidate(label, direction='decrease', guard='no_increase'):
    return {'label':label, 'present_in':['TX_A'], 'absent_in':['TX_B'],
            'pathogenic_isoforms':['TX_A'], 'decision_direction':direction,
            'suggested_guard':guard}


@pytest.mark.parametrize('direction,guard', [('decrease','no_increase'),('increase','no_decrease')])
def test_reciprocal_directional_guard_is_not_redundant_with_same_membership(direction,guard):
    first=candidate('acceptor@100',direction,guard)
    second=candidate('donor@160',direction,guard)
    action,reason=runtime._classify_safeguard_candidate_for_target(first,second)
    assert action == 'keep'
    assert 'reciprocal' in reason


def test_neutral_same_bundle_guard_requires_a_decision_not_silent_deletion():
    first=candidate('acceptor@100')
    second=candidate('donor@160',guard='neutral')
    assert runtime._classify_safeguard_candidate_for_target(first,second)[0] == 'ambiguous'


def test_same_physical_junction_is_not_its_own_advisory_guard():
    first=candidate('acceptor@100')
    assert runtime._classify_safeguard_candidate_for_target(first,first)[0] == 'drop'


def test_mapped_reciprocal_guard_retains_threshold_and_advisory_policy():
    first=candidate('acceptor@100')
    second=candidate('donor@160')
    plan=[{'site':'acceptor','position':100}, {'site':'donor','position':160}]
    selected=runtime._build_safeguard_candidates_by_target(plan,[first,second])
    guards=runtime._default_safeguards_from_candidates(selected['acceptor@100'])
    assert guards == [{'label':'donor@160','max_increase':0.1}]
    assert not guards[0].get('required',False)


def test_contextual_decider_receives_the_actual_advisory_guard_kind():
    result=runtime._build_ambiguous_safeguard_case(
        candidate('acceptor@100'), candidate('donor@160',guard='neutral'),
        'same_bundle_guard_requires_context')
    assert result['candidate']['guard_kind'] == 'neutral'
    assert result['candidate']['guard_policy'] == 'advisory'
