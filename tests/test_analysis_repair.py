import pytest

from aso_agents.analysis_repair import apply_program_repair, code_digest


def test_line_repair_preserves_scientific_plan_and_untouched_program():
    code = "criterion = 0.5\nvalue = None\nprint(f'{value:.3f}')\n"
    plan = {"analysis_plan": "Use the requested threshold.", "python_code": code,
            "assumptions": ["Missing is unknown."]}
    repaired = apply_program_repair(plan, code, {"base_sha256": code_digest(code), "patches": [
        {"start_line": 3, "end_line": 3, "replacement": "print('unknown' if value is None else f'{value:.3f}')"}]})
    assert repaired['analysis_plan'] == plan['analysis_plan']
    assert repaired['assumptions'] == plan['assumptions']
    assert repaired['python_code'].splitlines()[:2] == code.splitlines()[:2]
    assert plan['python_code'] == code


@pytest.mark.parametrize('fault', ['hash', 'overlap', 'span', 'unchanged', 'syntax'])
def test_unsafe_or_ineffective_patch_is_rejected(fault):
    code = 'value = 1\nprint(value)'
    patch = {'base_sha256': code_digest(code), 'patches': [
        {'start_line': 1, 'end_line': 1, 'replacement': 'value = 2'}]}
    if fault == 'hash': patch['base_sha256'] = 'wrong'
    if fault == 'overlap': patch['patches'].append(dict(patch['patches'][0]))
    if fault == 'span': patch['patches'][0]['end_line'] = 9
    if fault == 'unchanged': patch['patches'][0]['replacement'] = 'value = 1'
    if fault == 'syntax': patch['patches'][0]['replacement'] = 'value = ('
    with pytest.raises((ValueError, SyntaxError)):
        apply_program_repair({}, code, patch)
