"""Custom output writers retain task-specific source paths and schemas."""
import json

from aso_agents import analysis_execution
from aso_agents import pipeline_tools as pipeline


def test_task_scoped_supplied_files_reach_writer_without_merging_producers(tmp_path, monkeypatch):
    first, second = tmp_path / 'first.csv', tmp_path / 'second.csv'
    first.write_text('candidate_id,sequence,chemistry\nsame,ACGT,modified\n')
    second.write_text('candidate_id,sequence,chemistry\nsame,TGCA,unmodified\n')
    coordinates = tmp_path / 'coordinates.csv'
    coordinates.write_text('reference,block,start1,end1\nTX.2,1,10,13\n')
    goal = tmp_path / 'artifacts/intake/structured_goal.json'
    goal.parent.mkdir(parents=True); goal.write_text('{}')
    output = tmp_path / 'result.json'
    output.write_text('{}')
    envelope = {'schema_version': 'v3', 'tasks': [
        {'task_id': 'first', 'task_type': 'evaluate_existing_aso_set',
         'context': {'provided_aso_files': [{'path': str(first), 'file_format': 'csv'}],
                     'artifact_refs': [{'path': str(coordinates), 'artifact_type': 'csv'},
                                       {'path': str(output), 'artifact_type': 'output_file'}]}},
        {'task_id': 'second', 'task_type': 'evaluate_existing_aso_set',
         'context': {'provided_aso_files': [{'path': str(second), 'file_format': 'csv'}]}}],
        'final_outputs': [{'kind': 'custom', 'task_id': 'first', 'path': str(output)}]}
    captured = []

    def execute(**kwargs):
        captured.append(kwargs)
        for path in kwargs['required_outputs']:
            path.write_text('{"observations": 2}')
        return {'ok': True, 'attempts': []}

    monkeypatch.setattr(analysis_execution, 'execute_analysis_program', execute)
    result = pipeline._materialize_requested_final_outputs(
        {'request': 'Export both panels, preserving source identities.', 'request_envelope': envelope}, str(tmp_path))
    assert result['ok']
    declared = captured[0]['payload']['declared_inputs']
    tasks = {entry['task_id']: entry['declared_inputs'] for entry in declared['task_inputs']}
    assert tasks['first']['provided_input_files'][0]['path'] == str(first)
    assert tasks['second']['provided_input_files'][0]['path'] == str(second)
    assert tasks['first']['provided_input_files'][0]['table_summary']['columns'] == ['candidate_id', 'sequence', 'chemistry']
    assert tasks['first']['provided_input_files'][0]['table_summary']['sample_rows'][0]['sequence'] == 'ACGT'
    assert tasks['second']['provided_input_files'][0]['table_summary']['sample_rows'][0]['sequence'] == 'TGCA'
    assert tasks['first']['artifact_refs'] == [{'path': str(coordinates), 'exists': True},
                                             {'path': str(output), 'exists': True}]
    assert tasks['second']['artifact_refs'] == []
    assert {first, second, coordinates}.issubset(set(captured[0]['readonly_outputs']))
    assert output not in captured[0]['readonly_outputs']
    saved = json.loads((tmp_path / 'requested_artifacts/declared_inputs.json').read_text())
    assert saved['task_inputs'] == declared['task_inputs']
