import json

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.analysis_input_paths import declared_input_path_errors


def test_required_manifest_paths_are_reported_together_with_real_keys(tmp_path):
    code = """import json
with open('declared_inputs.json') as f:
    declared = json.load(f)
envelope = declared['request_envelope']
goal = envelope['structured_goal']
artifacts = declared['declared_inputs']['task_artifacts']
"""
    errors = declared_input_path_errors(code, tmp_path, {
        "request_envelope": {"tasks": []}, "completed_run": {"run_dir": "/output"}})
    assert any("['request_envelope']['structured_goal']" in error and "tasks" in error for error in errors)
    assert any("['declared_inputs']" in error and "completed_run" in error for error in errors)


def test_path_read_text_list_index_and_alias_reassignment(tmp_path):
    code = """from pathlib import Path
import json
data = json.loads(Path('declared_inputs.json').read_text())
first = data['rows'][0]
value = first['value']
missing = data['rows'][2]
data = {'other': 1}
other = data['other']
"""
    errors = declared_input_path_errors(code, tmp_path, {"rows": [{"value": 0}]})
    assert len(errors) == 1 and "['rows'][2]" in errors[0]


def test_optional_conditional_dynamic_and_other_file_accesses_are_not_guessed(tmp_path):
    code = """import json
data = json.load(open('declared_inputs.json'))
optional = data.get('absent')
safe = 'absent' in data and data['absent']
also_safe = data['absent'] if 'absent' in data else None
rows = [data[key] for key in data]
if 'absent' in data:
    value = data['absent']
other = json.load(open('different.json'))
unresolved = other['not_static']
"""
    assert declared_input_path_errors(code, tmp_path, {}) == []


def test_manifest_mutation_invalidates_static_knowledge(tmp_path):
    for mutation in ["data.update({'new': 1})", "if True:\n    data['new'] = 1"]:
        code = "import json\ndata=json.load(open('declared_inputs.json'))\n" + mutation + "\nvalue=data['new']\n"
        assert declared_input_path_errors(code, tmp_path, {}) == []


def test_nested_other_file_handle_does_not_inherit_manifest_identity(tmp_path):
    code = """import json
with open('declared_inputs.json') as f:
    with open('different.json') as f:
        other = json.load(f)
        value = other['unknown_shape']
"""
    assert declared_input_path_errors(code, tmp_path, {}) == []


def test_preflight_bundles_independent_faults_before_spending_a_repair_attempt(tmp_path):
    (tmp_path / "declared_inputs.json").write_text(json.dumps({"completed_run": {}}))
    code = """import json
data=json.load(open('declared_inputs.json'))
envelope=data['missing_envelope']
value = float('0.2')
eligible = 0.35 <= value <= 0.65
undefined = no_such_global
"""
    result = execute_analysis_program(directory=tmp_path, payload={}, required_outputs=[],
                                      generate=lambda _: {"python_code": code}, max_attempts=1)
    assert not result["ok"]
    error = result["attempts"][0]["error"]
    assert "missing_envelope" in error and "no_such_global" in error and "decimal-boundary" in error
