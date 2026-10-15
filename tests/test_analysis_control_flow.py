import json
import textwrap

import pytest

from aso_agents.analysis_control_flow import unreachable_statement_lines
from aso_agents.analysis_execution import execute_analysis_program


@pytest.mark.parametrize("code,lines", [
    ("def f():\n    return 1\n    print('lost')\n", [3]),
    ("def f():\n    raise ValueError()\n    x = 3\n", [3]),
    ("for x in [1]:\n    break\n    print(x)\nprint('reachable')\n", [3]),
    ("for x in [1]:\n    continue\n    print(x)\n", [3]),
    ("def f(x):\n    if x:\n        return 1\n    else:\n        return 2\n    print(x)\n", [6]),
    ("def f(x):\n    if x:\n        return 1\n    return 2\n", []),
    ("def f():\n    try:\n        return 1\n    finally:\n        print('cleanup')\n", []),
    ("for x in []:\n    break\nelse:\n    print('empty')\nprint('done')\n", []),
])
def test_only_proven_unreachable_statements_are_rejected(code, lines):
    assert unreachable_statement_lines(code) == lines


def test_silent_lost_rows_enter_existing_repair_before_execution(tmp_path):
    bad = textwrap.dedent('''\
        from pathlib import Path
        import json
        values = [2, 4, 6]
        processed = []
        for value in values:
            doubled = value * 2
        def helper(value):
            return value
            processed.append(doubled)
        Path('sandbox_result.json').write_text(json.dumps({'processed': processed}))
        Path('sandbox_summary.md').write_text('Complete')
    ''')
    good = bad.replace("    doubled = value * 2\n", "    doubled = value * 2\n    processed.append(doubled)\n")
    good = good.replace("    return value\n    processed.append(doubled)", "    return value")
    requests = []

    def generate(request):
        requests.append(request)
        if len(requests) == 2:
            assert 'Unreachable' in request['execution_repair']['error']
            assert request['execution_repair']['previous_code'] == bad.strip()
            assert request['declared_inputs'] == {'values': [2, 4, 6]}
        return {'python_code': bad if len(requests) == 1 else good}

    result = execute_analysis_program(directory=tmp_path, generate=generate,
        payload={'declared_inputs': {'values': [2, 4, 6]}}, required_outputs=[], max_attempts=2)
    assert result['ok'] and len(result['attempts']) == 2
    assert result['result']['processed'] == [4, 8, 12]
    assert (tmp_path / 'execution_attempts/01/stdout.log').read_text() == ''
    assert 'Unreachable' in json.loads((tmp_path / 'execution_attempts/01/execution.json').read_text())['error']


def test_legitimate_empty_selection_is_not_a_control_flow_failure(tmp_path):
    code = """from pathlib import Path
import json
selected = [value for value in [1, 2, 3] if value > 10]
Path('sandbox_result.json').write_text(json.dumps({'selected': selected}))
Path('sandbox_summary.md').write_text('No values satisfy the stated filter.')
"""
    result = execute_analysis_program(directory=tmp_path, payload={},
        generate=lambda _: {'python_code': code}, required_outputs=[], max_attempts=1)
    assert result['ok'] and result['result']['selected'] == []
