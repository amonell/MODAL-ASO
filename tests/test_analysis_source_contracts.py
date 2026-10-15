import hashlib
import json
import pytest

from aso_agents.pipeline_tools import _build_sandbox_declared_inputs


def completed_design(tmp_path, task_id, goal=None):
    child = tmp_path / task_id
    child.mkdir()
    source = child / 'generated_aso_set.json'
    source.write_text(json.dumps({'rows': [{'synthesized_antisense_sequence': 'ACGT'}]}))
    if goal is not None:
        reference = child / 'artifacts/intake/structured_goal.json'
        reference.parent.mkdir(parents=True)
        reference.write_text(json.dumps(goal))
    return {'child_status': 'run_complete', 'child_run_dir': str(child),
            'generated_aso_set_json': str(source)}


def test_completed_design_contracts_preserve_distinct_scopes_and_zero_values(tmp_path):
    goals = {
        'first': {'gene_symbol': 'GENERIC', 'candidate_minimum': 3,
                  'constraints': {'gc_min': 0.25, 'gc_max': 0.65},
                  'design_programs': [{'program_id': 'part_a', 'candidate_limit': 2,
                                       'constraints': {'max_mismatches': 0}},
                                      {'program_id': 'part_b', 'candidate_limit': 1,
                                       'chemistry': 'steric-hindrance'}]},
        'second': {'gene_symbol': 'OTHER', 'candidate_minimum': 6,
                   'constraints': {'gc_min': 0.40, 'gc_max': 0.60}},
    }
    state = {'task_graph': {'tasks': [{'task_id': name, 'task_type': 'design_new_asos'}
                                     for name in goals]},
             'task_artifacts': {name: completed_design(tmp_path, name, goal)
                                for name, goal in goals.items()}}
    # New analysis criteria remain distinct; they must not overwrite history.
    envelope = {'tasks': [], 'artifact_refs': [], 'task_context': {'gc_min': 0.1}}
    declared = _build_sandbox_declared_inputs(state, envelope)
    for candidate in declared['completed_design_candidates']:
        contract = candidate['source_design_contract']
        goal = goals[candidate['task_id']]
        assert contract['structured_goal'] == goal
        raw = (tmp_path / candidate['task_id'] / 'artifacts/intake/structured_goal.json').read_bytes()
        assert contract['source_sha256'] == hashlib.sha256(raw).hexdigest()
        assert json.loads(raw) == goal
    declared['completed_design_candidates'][0]['source_design_contract']['structured_goal']['constraints']['gc_min'] = 0
    assert json.loads((tmp_path / 'first/artifacts/intake/structured_goal.json').read_text()) == goals['first']
    assert envelope['task_context'] == {'gc_min': 0.1}


def test_unrecorded_design_contract_remains_unknown(tmp_path):
    state = {'task_graph': {'tasks': [{'task_id': 'old_run', 'task_type': 'design_new_asos'}]},
             'task_artifacts': {'old_run': completed_design(tmp_path, 'old_run')}}
    declared = _build_sandbox_declared_inputs(state, {'tasks': [], 'artifact_refs': []})
    contract = declared['completed_design_candidates'][0]['source_design_contract']
    assert contract['structured_goal'] is None
    assert contract['source_path'] is None
    assert contract['source_sha256'] is None
    assert 'gc_min' not in contract and 'candidate_limit' not in contract


def test_malformed_recorded_contract_is_not_replaced_with_empty_constraints(tmp_path):
    state = {'task_graph': {'tasks': [{'task_id': 'bad_run', 'task_type': 'design_new_asos'}]},
             'task_artifacts': {'bad_run': completed_design(tmp_path, 'bad_run', [])}}
    with pytest.raises(ValueError, match='Recorded design contract must be a JSON object'):
        _build_sandbox_declared_inputs(state, {'tasks': [], 'artifact_refs': []})
