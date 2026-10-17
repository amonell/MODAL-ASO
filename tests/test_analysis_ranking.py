from decimal import Decimal
from fractions import Fraction
import json

import pytest

from aso_agents.analysis_helpers import shared_analysis_helpers, validate_shared_helper_bindings
from aso_agents.analysis_ranking import dense_ranks


def test_dense_ranks_preserve_input_rows_and_advance_once_per_distinct_key():
    keys = [(8, 2), (3, 9), (8, 2), (3, 1), (9, 0), (3, 1)]
    assert dense_ranks(keys) == [3, 2, 3, 1, 4, 1]
    assert dense_ranks(keys, reverse=True) == [2, 3, 2, 4, 1, 4]
    assert keys == [(8, 2), (3, 9), (8, 2), (3, 1), (9, 0), (3, 1)]


def test_exact_keys_retain_mathematical_ties_without_float_conversion():
    keys = [(Fraction(2, 7) + Fraction(3, 7), Decimal('0.10') + Decimal('0.20')),
            (Fraction(5, 7), Decimal('0.30')),
            (Fraction(5, 7), Decimal('0.3000000000000000000000000001'))]
    assert dense_ranks(keys) == [1, 1, 2]
    huge = Fraction(10 ** 500, 3)
    assert dense_ranks([huge, huge - 1]) == [2, 1]


def test_empty_singleton_all_tied_and_generator_keys():
    assert dense_ranks([]) == []
    assert dense_ranks([7]) == [1]
    assert dense_ranks(['same', 'same']) == [1, 1]
    assert dense_ranks(x for x in [4, 1, 4, 2]) == [3, 1, 3, 2]


@pytest.mark.parametrize('key', [None, (1, None), float('nan'), float('inf'),
                                 Decimal('NaN'), (1, Decimal('-Infinity'))])
def test_unknown_or_nonfinite_keys_are_never_ranked(key):
    with pytest.raises((TypeError, ValueError), match='known'):
        dense_ranks([key])


def test_incomparable_keys_and_non_boolean_direction_are_rejected():
    with pytest.raises(TypeError, match='comparable'):
        dense_ranks([1, '1'])
    with pytest.raises(TypeError, match='Boolean'):
        dense_ranks([1, 2], reverse='false')


def test_native_helper_binding_and_real_analysis_execution(tmp_path):
    from aso_agents.analysis_execution import execute_analysis_program
    assert any(spec['name'] == 'dense_ranks' for spec in shared_analysis_helpers())
    with pytest.raises(ValueError, match='shared analysis helper'):
        validate_shared_helper_bindings('def dense_ranks(keys): return list(range(len(keys)))')

    def generate(payload):
        contract = payload['execution_contract']
        return {'python_code': (
            'from aso_agents.analysis_ranking import dense_ranks\n'
            'from pathlib import Path\nimport json\n'
            'ranks = dense_ranks([(9, 1), (3, 1), (3, 1), (7, 4)])\n'
            f"Path({contract['result_json']!r}).write_text(json.dumps({{'ranks': ranks}}))\n"
            f"Path({contract['summary_markdown']!r}).write_text('Computed requested dense ranks.')\n"
        )}

    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[], max_attempts=1)
    assert result['ok'], result
    assert result['result']['ranks'] == [3, 1, 1, 2]
