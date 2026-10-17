import json
import random

import pytest

from aso_agents.analysis_helpers import shared_analysis_helpers, validate_shared_helper_bindings
from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.mapping_evidence import scan_antisense_sites


def brute_force(aso, reference, budget):
    complement = dict(zip('ACGTU', 'TGCAA'))
    query = ''.join(complement[base] for base in aso.upper()[::-1])
    reference = reference.upper().replace('U', 'T')
    rows = []
    for offset in range(len(reference) - len(query) + 1):
        target = reference[offset:offset + len(query)]
        positions = [i + 1 for i, (a, b) in enumerate(zip(target, query)) if a != b]
        if len(positions) <= budget:
            rows.append((offset + 1, offset + len(query), target, positions))
    return rows


def check_scan(aso, reference, budget):
    result = scan_antisense_sites(antisense_5to3=aso, reference_5to3=reference,
                                 max_mismatches=budget)
    expected = brute_force(aso, reference, budget)
    assert [(r['target_start_1based'], r['target_end_1based'], r['target_5to3'],
             r['mismatch_positions_in_target_sequence_1based']) for r in result] == expected
    original = aso.upper().replace('U', 'T')
    for row in result:
        assert row['antisense_5to3'] == original
        assert row['coordinate_system'] == '1based_inclusive'
        assert row['search_mismatch_budget'] == budget
        assert row['mismatch_count'] == len(row['mismatch_details_in_target_orientation'])
        assert row['match_count'] + row['mismatch_count'] == len(original)
        for mismatch in row['mismatch_details_in_target_orientation']:
            target_index = mismatch['target_position_1based'] - 1
            aso_index = len(original) - target_index - 1
            assert mismatch['supplied_sequence_position_1based'] == aso_index + 1
            assert mismatch['supplied_sequence_base'] == original[aso_index]
            assert mismatch['reference_base'] == row['target_5to3'][target_index]
            assert mismatch['aligned_query_base'] == row['aligned_query_5to3'][target_index]
    return result


def test_nonpalindromic_aso_is_complemented_once_and_same_strand_is_not_a_hit():
    assert len(check_scan('ACGATTCG', 'CGAATCGT', 0)) == 1
    assert check_scan('ACGATTCG', 'ACGATTCG', 0) == []


@pytest.mark.parametrize('budget', [0, 1, 2, 8, 10])
def test_every_hit_matches_independent_hamming_scan(budget):
    check_scan('ACGATTCG', 'CGAATCGTACGAATCATCGAATCGT', budget)


def test_overlapping_and_repeated_sites_are_not_capped_or_deduplicated():
    result = check_scan('TTTT', 'A' * 19, 0)
    assert [r['target_start_1based'] for r in result] == list(range(1, 17))


def test_one_substitution_preserves_target_and_physical_aso_base_identities():
    result = check_scan('ACGATTCG', 'CGAGTCGT', 1)
    assert result[0]['mismatch_positions_in_target_sequence_1based'] == [4]
    assert result[0]['mismatch_positions_in_supplied_sequence_1based'] == [5]
    assert result[0]['mismatch_details_in_target_orientation'] == [{
        'target_position_1based': 4, 'reference_base': 'G', 'aligned_query_base': 'A',
        'supplied_sequence_position_1based': 5, 'supplied_sequence_base': 'T',
        'expected_supplied_sequence_base': 'C'}]


def test_rna_lowercase_and_short_nonempty_references():
    assert len(check_scan('acgauucg', 'cgaaucgu', 0)) == 1
    assert check_scan('ACGATTCG', 'ACG', 1) == []


def test_diverse_lengths_and_budgets_against_independent_brute_force():
    rng = random.Random(38293)
    for _ in range(100):
        length = rng.randint(1, 12)
        aso = ''.join(rng.choices('ACGT', k=length))
        reference = ''.join(rng.choices('ACGT', k=rng.randint(1, 64)))
        check_scan(aso, reference, rng.randint(0, length))


@pytest.mark.parametrize('bad', ['', 'ACGN', 'AC-G', ' ACGA', 'A[mC]GT', None, 123])
@pytest.mark.parametrize('field', ['antisense_5to3', 'reference_5to3'])
def test_unresolved_input_is_an_error_not_a_zero_site_result(bad, field):
    args = {'antisense_5to3': 'ACGA', 'reference_5to3': 'TCGT', 'max_mismatches': 1}
    args[field] = bad
    with pytest.raises((TypeError, ValueError)):
        scan_antisense_sites(**args)


@pytest.mark.parametrize('budget', [None, True, False, -1, 1.5, '1'])
def test_budget_requires_explicit_nonnegative_integer(budget):
    with pytest.raises(ValueError):
        scan_antisense_sites(antisense_5to3='ACGA', reference_5to3='TCGT', max_mismatches=budget)


def test_api_requires_named_sequence_roles_and_explicit_budget():
    with pytest.raises(TypeError):
        scan_antisense_sites('ACGA', 'TCGT', 0)
    with pytest.raises(TypeError):
        scan_antisense_sites(antisense_5to3='ACGA', reference_5to3='TCGT')


def test_published_native_helper_cannot_be_replaced_by_a_generated_lookalike():
    helper = next(h for h in shared_analysis_helpers() if h['name'] == 'scan_antisense_sites')
    assert helper['module'] == 'aso_agents.mapping_evidence'
    validate_shared_helper_bindings('from aso_agents.mapping_evidence import scan_antisense_sites')
    with pytest.raises(ValueError, match='Import the native API'):
        validate_shared_helper_bindings('def scan_antisense_sites(**kwargs): return []')


def test_actual_analysis_runtime_imports_the_scanner_and_preserves_every_site(tmp_path):
    inputs = {'aso': 'ACGATTCG', 'reference': 'CGAATCGTACGAATCATCGAATCGT'}
    (tmp_path / 'declared_inputs.json').write_text(json.dumps(inputs))

    def generate(payload):
        contract = payload['execution_contract']
        assert any(h['name'] == 'scan_antisense_sites' for h in contract['shared_analysis_helpers'])
        return {'python_code': '\n'.join([
            'import json', 'from pathlib import Path',
            'from aso_agents.mapping_evidence import scan_antisense_sites',
            "data=json.loads(Path('declared_inputs.json').read_text())",
            "sites=scan_antisense_sites(antisense_5to3=data['aso'], reference_5to3=data['reference'], max_mismatches=1)",
            f"Path({contract['result_json']!r}).write_text(json.dumps({{'sites':sites}}))",
            f"Path({contract['summary_markdown']!r}).write_text('All full-length antisense sites at the requested substitution budget.')",
        ])}

    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[], max_attempts=1)
    assert result['ok'], result
    rows = result['result']['sites']
    assert [(r['target_start_1based'], r['target_end_1based'], r['target_5to3'],
             r['mismatch_positions_in_target_sequence_1based']) for r in rows] == brute_force(
                 inputs['aso'], inputs['reference'], 1)
