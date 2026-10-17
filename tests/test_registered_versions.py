import pytest

from aso_agents.analysis_helpers import shared_analysis_helpers, validate_shared_helper_bindings
from aso_agents.registered_versions import RegisteredVersions


def test_old_unregistered_observations_are_not_superseded():
    scope = ('sample-a', 'partition-one')
    registry = RegisteredVersions([(scope, 3), (scope, 7)])
    assert registry.latest(scope) == 7
    assert [registry.classify(scope, n) for n in (1, 3, 5, 7, 9)] == [
        'unregistered', 'superseded', 'unregistered', 'current', 'unregistered']


def test_latest_registration_does_not_depend_on_observed_rows():
    scope = ('job', 0)
    registry = RegisteredVersions([(scope, 0), (scope, 4)])
    observed_versions = [0, 0, 0]
    assert registry.latest(scope) == 4
    assert [registry.classify(scope, v) for v in observed_versions] == ['superseded'] * 3


def test_all_scope_dimensions_participate_and_unknown_scope_is_explicit():
    first, second = ('item', 'human', 0, 'left'), ('item', 'human', 1, 'left')
    registry = RegisteredVersions([(first, 1), (first, 2), (second, 1)])
    assert registry.classify(first, 1) == 'superseded'
    assert registry.classify(second, 1) == 'current'
    assert registry.classify(second, 2) == 'unregistered'
    for scope in [('other', 'human', 0, 'left'), ('item', 'mouse', 0, 'left'),
                  ('item', 'human', 0, 'right')]:
        assert registry.latest(scope) is None
        assert registry.classify(scope, 1) == 'unregistered'


def test_registry_order_duplicates_and_generators_do_not_change_selection():
    scope = ('x',)
    rows = [(scope, -2), (scope, 10), (scope, 2), (scope, 10)]
    for source in [rows, rows[::-1], (r for r in rows)]:
        registry = RegisteredVersions(source)
        assert registry.latest(scope) == 10
        assert [registry.classify(scope, v) for v in (-2, 2, 10)] == ['superseded', 'superseded', 'current']
    empty = RegisteredVersions([])
    assert empty.latest(scope) is None and empty.classify(scope, 0) == 'unregistered'


@pytest.mark.parametrize('scope', [(), 'name', (None,), ('a', float('nan')), ('a', [])])
def test_unknown_or_incomplete_scope_keys_are_rejected(scope):
    with pytest.raises((TypeError, ValueError)):
        RegisteredVersions([(scope, 1)])
    registry = RegisteredVersions([])
    with pytest.raises((TypeError, ValueError)):
        registry.classify(scope, 1)


@pytest.mark.parametrize('version', [None, True, 1.0, '1'])
def test_integer_versions_are_explicit_not_inferred_or_lexicographic(version):
    with pytest.raises(TypeError):
        RegisteredVersions([(('scope',), version)])
    with pytest.raises(TypeError):
        RegisteredVersions([]).classify(('scope',), version)


def test_native_import_binding_and_real_generated_analysis(tmp_path):
    from aso_agents.analysis_execution import execute_analysis_program
    assert any(h['name'] == 'RegisteredVersions' for h in shared_analysis_helpers())
    with pytest.raises(ValueError, match='shared analysis helper'):
        validate_shared_helper_bindings('class RegisteredVersions: pass')

    def generate(payload):
        contract = payload['execution_contract']
        return {'python_code': (
            'from aso_agents.registered_versions import RegisteredVersions\n'
            'from pathlib import Path\nimport json\n'
            "registry = RegisteredVersions([(('a', 'left'), 2), (('a', 'left'), 4), (('a', 'right'), 1)])\n"
            "states = [registry.classify(('a', 'left'), v) for v in [1, 2, 4]]\n"
            f"Path({contract['result_json']!r}).write_text(json.dumps({{'states': states}}))\n"
            f"Path({contract['summary_markdown']!r}).write_text('Checked exact version membership.')\n"
        )}

    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[], max_attempts=1)
    assert result['ok'], result
    assert result['result']['states'] == ['unregistered', 'superseded', 'current']
