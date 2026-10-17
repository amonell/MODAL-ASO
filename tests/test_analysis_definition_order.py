import json
from pathlib import Path
import subprocess
import sys

import pytest

from aso_agents import analysis_runtime
from aso_agents.analysis_execution import execute_analysis_program


def run_program(tmp_path, code):
    program = tmp_path / 'generated_code.py'
    diagnostic = tmp_path / 'runtime_error.json'
    program.write_text(code)
    result = subprocess.run([sys.executable, str(Path(analysis_runtime.__file__)),
                             str(program), str(diagnostic)], cwd=tmp_path,
                            capture_output=True, text=True)
    return result.returncode, json.loads(diagnostic.read_text()) if diagnostic.exists() else {}


@pytest.mark.parametrize('code,use_line,module_line,definition_line,name', [
    ('for value in [2]:\n    result = normalize(value)\ndef normalize(value):\n    return value\n',
     2, 2, 3, 'normalize'),
    ('def transform(value):\n    return normalize(value)\nresult = transform(2)\ndef normalize(value):\n    return value\n',
     2, 3, 4, 'normalize'),
    ('result = Record()\nclass Record:\n    pass\n', 1, 1, 2, 'Record'),
])
def test_actual_missing_global_reports_later_definition(tmp_path, code, use_line, module_line,
                                                      definition_line, name):
    exit_code, record = run_program(tmp_path, code)
    assert exit_code == 1 and record['exception_type'] == 'NameError'
    context = record['definition_order']
    assert context['name'] == name
    assert context['use_line'] == use_line
    assert context['module_execution_line'] == module_line
    assert context['definition_line'] == definition_line
    assert 'before' in context['repair_instruction']
    assert 'value' not in context  # No local values enter this diagnostic.


@pytest.mark.parametrize('code', [
    'missing(2)\n',
    'def compute():\n    print(value)\n    value = 2\ncompute()\n',
    'def normalize(value):\n    return value\ndel normalize\nnormalize(2)\n',
    "raise NameError('deliberately raised', name='normalize')\ndef normalize(): pass\n",
    'normalize = None\nnormalize()\ndef normalize(): pass\n',
])
def test_other_failures_do_not_invent_definition_order(tmp_path, code):
    exit_code, record = run_program(tmp_path, code)
    assert exit_code == 1
    assert 'definition_order' not in record


def test_valid_deferred_reference_remains_valid(tmp_path):
    code = 'def transform(value):\n    return normalize(value)\ndef normalize(value):\n    return value\nassert transform(2) == 2\n'
    exit_code, record = run_program(tmp_path, code)
    assert exit_code == 0 and record == {}


def test_execution_repair_receives_order_diagnosis_and_preserves_first_code(tmp_path):
    original = "result = normalize(2)\ndef normalize(value):\n    return value\n"
    seen = []
    def generate(payload):
        seen.append(payload)
        if len(seen) == 1:
            return {'python_code': original}
        diagnosis = payload['execution_repair']['runtime_diagnostics']['definition_order']
        assert diagnosis['definition_line'] == 2
        assert diagnosis['module_execution_line'] == 1
        return {'python_code': "import json\nfrom pathlib import Path\n"
                "def normalize(value):\n    return value\nresult = normalize(2)\n"
                "Path('sandbox_result.json').write_text(json.dumps({'result': result}))\n"
                "Path('sandbox_summary.md').write_text('Result computed from the declared input.')\n"}
    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[], max_attempts=2)
    assert result['ok'] and result['result']['result'] == 2
    assert len(result['attempts']) == 2
    assert (tmp_path / 'execution_attempts/01/generated_code.py').read_text() == original.strip()
