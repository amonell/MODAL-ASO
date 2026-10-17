import textwrap

import pytest

from aso_agents.analysis_dependencies import validate_program_dependencies
from aso_agents.analysis_local_bindings import premature_local_helper_reads
from aso_agents.analysis_repair import apply_program_repair, code_digest


def test_local_helper_used_in_loop_before_definition_is_diagnosed():
    code = ('def main():\n'
            '    values = []\n'
            '    for value in [2, 3]:\n'
            '        values.append(transform(value))\n'
            '    def transform(value):\n'
            '        return value * 2\n'
            '    return values\n'
            'main()')
    assert premature_local_helper_reads(code) == [
        {'function': 'main', 'name': 'transform', 'read_line': 4, 'definition_line': 5}]
    with pytest.raises(UnboundLocalError):
        exec(code, {})
    with pytest.raises(ValueError, match='read line 4, local definition line 5'):
        validate_program_dependencies(code)


def test_dependency_feedback_includes_global_and_local_defects_together():
    code = 'def run():\n    missing_api()\n    helper()\n    def helper(): pass\nrun()'
    with pytest.raises(ValueError) as exc:
        validate_program_dependencies(code)
    assert 'missing_api' in str(exc.value) and 'helper in run' in str(exc.value)


@pytest.mark.parametrize('code', [
    'def run():\n    def helper(): return 4\n    return helper()\nassert run() == 4',
    'def run(): return helper()\ndef helper(): return 4\nassert run() == 4',
    'def run():\n    def helper(n): return helper(n-1) if n else 4\n    return helper(2)\nassert run() == 4',
    'def run():\n    def later(): return helper()\n    def helper(): return 4\n    return later()\nassert run() == 4',
    'def run():\n    later = lambda: helper()\n    def helper(): return 4\n    return later()\nassert run() == 4',
    'def run():\n    later = (helper() for _ in [1])\n    def helper(): return 4\n    return next(later)\nassert run() == 4',
    'def run(helper):\n    value = helper()\n    def helper(): return 5\n    return value\nassert run(lambda: 4) == 4',
    'def run():\n    helper = lambda: 4\n    value = helper()\n    def helper(): return 5\n    return value\nassert run() == 4',
    'def helper(): return 4\ndef run():\n    global helper\n    value = helper()\n    def helper(): return 5\n    return value\nassert run() == 4',
    'def run():\n    def assign():\n        nonlocal helper\n        helper = lambda: 4\n    assign()\n    value = helper()\n    def helper(): return 5\n    return value\nassert run() == 4',
    'def run():\n    from math import sqrt as helper\n    value = helper(16)\n    def helper(): return 5\n    return value\nassert run() == 4',
    'def run():\n    class helper:\n        def __call__(self): return 4\n    value = helper()()\n    def helper(): return 5\n    return value\nassert run() == 4',
    'def run():\n    for helper in [lambda: 4]:\n        value = helper()\n    def helper(): return 5\n    return value\nassert run() == 4',
    'def run():\n    match (lambda: 4):\n        case helper: value = helper()\n    def helper(): return 5\n    return value\nassert run() == 4',
    'def run():\n    values = [(helper := lambda: 4) for _ in [1]]\n    value = helper()\n    def helper(): return 5\n    return value\nassert run() == 4',
])
def test_valid_binding_and_deferred_use_controls_execute_unchanged(code):
    assert premature_local_helper_reads(code) == []
    validate_program_dependencies(code)
    exec(code, {})


def test_dynamic_namespace_excluded_from_static_claim():
    code = 'def run():\n    locals()\n    helper()\n    def helper(): pass'
    assert premature_local_helper_reads(code) == []


def test_local_function_reference_and_async_function_are_also_checked():
    code = ('async def run():\n    callback = helper\n'
            '    async def helper(): return 4\n    return await callback()')
    assert premature_local_helper_reads(code) == [
        {'function': 'run', 'name': 'helper', 'read_line': 2, 'definition_line': 3}]


def test_repair_must_resolve_new_local_helper_dependency():
    original = 'def run():\n    return 4\nvalue = run()'
    repair = {'base_sha256': code_digest(original), 'patches': [
        {'start_line': 2, 'end_line': 2, 'replacement':
         '    value = helper()\n    def helper(): return 4\n    return value'}]}
    with pytest.raises(ValueError, match='reads local helpers'):
        apply_program_repair({}, original, repair)
    repair['patches'][0]['replacement'] = (
        '    def helper(): return 4\n    value = helper()\n    return value')
    result = apply_program_repair({}, original, repair)
    namespace = {}
    exec(result['python_code'], namespace)
    assert namespace['value'] == 4


def test_actual_execution_reports_before_side_effect_then_repairs(tmp_path):
    from aso_agents.analysis_execution import execute_analysis_program

    output = tmp_path / 'results.csv'
    marker = tmp_path / 'execution-marker'
    bad = textwrap.dedent(f'''\
        from pathlib import Path
        Path({str(marker)!r}).write_text('executed')
        def run():
            values = [2, 3]
            rows = []
            for value in values:
                rows.append(str(transform(value)))
            def transform(value):
                return value * 2
            Path({str(output)!r}).write_text('value\\n' + '\\n'.join(rows) + '\\n')
        run()
        Path('sandbox_result.json').write_text('{{}}')
        Path('sandbox_summary.md').write_text('Both values were doubled.')
    ''')
    good = bad.replace('    values = [2, 3]',
                       '    def transform(value):\n        return value * 2\n    values = [2, 3]')
    good = good.replace('    def transform(value):\n        return value * 2\n    Path',
                        '    Path')
    requests = []

    def generate(request):
        requests.append(request)
        if len(requests) == 2:
            assert not marker.exists() and not output.exists()
            assert 'reads local helpers' in str(request)
        return {'python_code': bad if len(requests) == 1 else good,
                'analysis_plan': 'Double all input values', 'assumptions': [],
                'limitations': [], 'expected_outputs': [str(output)]}

    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[output], max_attempts=2)
    assert result['exit_code'] == 0 and len(requests) == 2
    assert output.read_text() == 'value\n4\n6\n'
    assert marker.read_text() == 'executed'
    assert len(result['attempts']) == 2
