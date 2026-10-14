from copy import deepcopy

from aso_agents.chemical_position_evidence import (chemical_position_evidence,
                                                    supplied_gapmer_architecture)
from aso_agents.evaluation_evidence import attach_measurement_references
from aso_agents.evaluation_response_audit import (chemical_position_audit_packets,
                                                  correct_native_chemical_position_claims,
                                                  filter_chemical_position_findings,
                                                  native_chemical_position_findings,
                                                  pointer_value)


def test_supplied_gapmer_regions_follow_numeric_descriptor_and_recorded_positions():
    row = {'name': 'A', 'input_row_id': 'row_a', 'sequence': 'A' * 20,
           'chemistry': 'full-PS 5-10-5 2′-MOE/DNA gapmer',
           'near_match_candidates': [
               {'mismatch_count': 4, 'mismatch_positions_in_supplied_sequence_1based': [1, 5, 6, 15]},
               {'mismatch_count': 2, 'mismatch_positions_in_supplied_sequence_1based': [16, 20]},
           ]}
    original = deepcopy(row)
    evidence = chemical_position_evidence(row)
    assert row == original
    assert evidence['position_ranges_1based_inclusive'] == {
        '5prime_wing': [1, 5], 'dna_gap': [6, 15], '3prime_wing': [16, 20]}
    assert evidence['recorded_near_match_mismatch_positions'] == [
        {'position_1based': p, 'region': region}
        for p, region in [(1, '5prime_wing'), (5, '5prime_wing'),
                          (6, 'dna_gap'), (15, 'dna_gap'),
                          (16, '3prime_wing'), (20, '3prime_wing')]]
    summary = attach_measurement_references({'provided_asos': [row]})
    assert summary['provided_asos'][0]['chemical_position_evidence'] == evidence


def test_ambiguous_or_inconsistent_chemistry_is_not_interpreted():
    sequence = 'A' * 20
    for chemistry in ('5-10-5 or 4-12-4 DNA gapmer', '5-10-5 MOE steric',
                      'full-PS DNA gapmer', 'full-PS 5-9-5 DNA gapmer'):
        assert supplied_gapmer_architecture(chemistry, sequence) is None
        assert chemical_position_evidence({'chemistry': chemistry, 'sequence': sequence}) is None
    assert supplied_gapmer_architecture('5-10-5', sequence).length == 20


def test_position_packet_binds_one_assessment_to_its_source_row():
    rows = [
        {'name': 'same', 'input_row_id': 'row_a', 'sequence': 'A' * 20,
         'chemistry': '5-10-5', 'near_match_candidates': [
             {'mismatch_count': 1, 'mismatch_positions_in_supplied_sequence_1based': [9]}]},
        {'name': 'same', 'input_row_id': 'row_b', 'sequence': 'C' * 20,
         'chemistry': '5-10-5', 'near_match_candidates': [
             {'mismatch_count': 1, 'mismatch_positions_in_supplied_sequence_1based': [3]}]},
    ]
    source = {'evaluation_summary': attach_measurement_references({'provided_asos': rows})}
    response = {'evaluation_assessments': [
        {'input_row_id': 'row_a', 'mechanistic_plausibility': 'Mismatch 9 lies in a MOE wing.'},
        {'input_row_id': 'row_b', 'mechanistic_plausibility': 'Mismatch 3 lies in a wing.'}]}
    packets = chemical_position_audit_packets(source, response)
    assert len(packets) == 2
    assert packets[0]['draft_narrative'][0]['response_path'] == '/evaluation_assessments/0/mechanistic_plausibility'
    assert packets[1]['draft_narrative'][0]['response_path'] == '/evaluation_assessments/1/mechanistic_plausibility'
    for packet in packets:
        assert all(pointer_value(source, item['source_pointer']) == item['recorded_value']
                   for item in packet['position_evidence'])


def test_model_position_findings_must_contain_a_native_verified_contradiction():
    row = {'name': 'A', 'input_row_id': 'row_a', 'sequence': 'A' * 20,
           'chemistry': '5-10-5', 'near_match_candidates': [
               {'mismatch_count': 1, 'mismatch_positions_in_supplied_sequence_1based': [9]}]}
    source = {'evaluation_summary': attach_measurement_references({'provided_asos': [row]})}
    pointer = '/evaluation_summary/provided_asos/0/chemical_position_evidence'
    path = '/evaluation_assessments/0/mechanistic_plausibility'
    wrong = 'The recorded mismatch at supplied-ASO position 9 lies in a MOE wing.'
    correct = 'The recorded mismatch at supplied-ASO position 9 lies in the DNA gap.'
    for sentence, keep in ((wrong, True), (correct, False)):
        response = {'evaluation_assessments': [
            {'name': 'A', 'input_row_id': 'row_a', 'mechanistic_plausibility': sentence}]}
        finding = {'response_path': path, 'quote': sentence,
                   'kind': 'contradicted_fact', 'evidence_paths': [pointer],
                   'reason': 'Model supplied reason is not trusted by this check.'}
        accepted, rejected = filter_chemical_position_findings(
            {'findings': [finding]}, response, source)
        assert bool(accepted) == keep
        assert bool(rejected) != keep
    response['evaluation_assessments'][0]['mechanistic_plausibility'] = (
        'Position 9 is in the DNA gap, not a wing.')
    finding['quote'] = response['evaluation_assessments'][0]['mechanistic_plausibility']
    assert filter_chemical_position_findings({'findings': [finding]}, response, source)[0] == []


def test_multiple_explicit_mismatches_are_checked_as_one_conservative_claim():
    row = {'name': 'B', 'input_row_id': 'row_b', 'sequence': 'A' * 20,
           'chemistry': 'full-PS 5-10-5 2-MOE/DNA gapmer',
           'near_match_candidates': [
               {'mismatch_count': 2,
                'mismatch_positions_in_supplied_sequence_1based': [5, 16]},
               {'mismatch_count': 2,
                'mismatch_positions_in_supplied_sequence_1based': [5, 9]}]}
    source = {'evaluation_summary': attach_measurement_references({'provided_asos': [row]})}
    base = {'response_path': '/evaluation_assessments/0/mechanistic_plausibility',
            'kind': 'contradicted_fact',
            'evidence_paths': ['/evaluation_summary/provided_asos/0/chemical_position_evidence'],
            'reason': 'Model judgment is independently verified.'}
    for sentence, expected in (
        ('If that duplex forms inside the DNA gap (mismatches at oligo positions 5 and 16 in the supplied 5-to-3 sequence), RNase H is possible.', True),
        ('Mismatches at oligo positions 5 and 16 are in MOE wings.', False),
        ('Mismatches at oligo positions 5 and 9 are both in the DNA gap.', False),
    ):
        response = {'evaluation_assessments': [
            {'name': 'B', 'input_row_id': 'row_b',
             'mechanistic_plausibility': sentence}]}
        finding = {**base, 'quote': sentence}
        accepted, rejected = filter_chemical_position_findings(
            {'findings': [finding]}, response, source)
        assert bool(accepted) == expected
        assert bool(rejected) != expected


def test_position_only_quote_uses_only_adjacent_original_draft_region():
    row = {'name': 'C', 'input_row_id': 'row_c', 'sequence': 'A' * 20,
           'chemistry': 'full-PS 5-10-5 2-MOE/DNA gapmer',
           'near_match_candidates': [
               {'mismatch_count': 2,
                'mismatch_positions_in_supplied_sequence_1based': [5, 16]}]}
    source = {'evaluation_summary': attach_measurement_references({'provided_asos': [row]})}
    quote = 'mismatches at oligo positions 5 and 16 in the supplied 5-to-3 sequence'
    base = {'response_path': '/evaluation_assessments/0/mechanistic_plausibility',
            'quote': quote, 'kind': 'contradicted_fact',
            'evidence_paths': ['/evaluation_summary/provided_asos/0/chemical_position_evidence'],
            'reason': 'Untrusted model reason'}
    for prefix, expected in (
        ('If a duplex forms inside the DNA gap (', True),
        ('If a duplex forms inside the MOE wings (', False),
        ('The DNA gap is elsewhere. Mismatches occur (', False),
        ('If a duplex forms not inside the DNA gap (', False),
    ):
        response = {'evaluation_assessments': [
            {'name': 'C', 'input_row_id': 'row_c',
             'mechanistic_plausibility': prefix + quote + ').'}]}
        accepted, rejected = filter_chemical_position_findings(
            {'findings': [base]}, response, source)
        assert bool(accepted) == expected
        assert bool(rejected) != expected


def test_native_check_finds_explicit_multi_position_error_when_model_returns_nothing():
    row = {'name': 'C', 'input_row_id': 'row_c', 'sequence': 'A' * 20,
           'chemistry': 'full-PS 5-10-5 2-MOE/DNA gapmer',
           'near_match_candidates': [
               {'mismatch_count': 2,
                'mismatch_positions_in_supplied_sequence_1based': [5, 16]}]}
    source = {'evaluation_summary': attach_measurement_references({'provided_asos': [row]})}
    wrong = ('A retained two-mismatch hypothesis is not measured activity. '
             'If that duplex forms inside the DNA gap (mismatches at oligo '
             'positions 5 and 16 in the supplied 5-to-3 sequence), RNase H '
             'activity is possible but unmeasured.')
    response = {'evaluation_assessments': [
        {'name': 'C', 'input_row_id': 'row_c',
         'mechanistic_plausibility': wrong}]}
    findings = native_chemical_position_findings(source, response)
    assert len(findings) == 1
    assert findings[0]['quote'].startswith('If that duplex forms inside the DNA gap')
    assert findings[0]['evidence_paths'] == [
        '/evaluation_summary/provided_asos/0/chemical_position_evidence']
    response['evaluation_assessments'][0]['mechanistic_plausibility'] = wrong.replace(
        'inside the DNA gap (mismatches', 'inside the MOE wings (mismatches')
    assert native_chemical_position_findings(source, response) == []


def test_native_correction_changes_only_wrong_sentence_and_its_markdown_view():
    row = {'name': 'C', 'input_row_id': 'row_c', 'sequence': 'A' * 20,
           'chemistry': 'full-PS 5-10-5 2-MOE/DNA gapmer',
           'near_match_candidates': [
               {'mismatch_count': 2,
                'mismatch_positions_in_supplied_sequence_1based': [5, 16]}]}
    source = {'evaluation_summary': attach_measurement_references({'provided_asos': [row]})}
    wrong = ('If that duplex forms inside the DNA gap (mismatches at oligo '
             'positions 5 and 16 in the supplied 5-to-3 sequence), RNase H '
             'activity is possible but unmeasured.')
    field = 'The alignment is a hypothesis. ' + wrong + ' No knockdown was measured.'
    response = {'evaluation_assessments': [
        {'name': 'C', 'input_row_id': 'row_c', 'mechanistic_plausibility': field,
         'uncertainty': 'No knockdown was measured.'}],
        'markdown': '## C\n\n' + field}
    original = deepcopy(response)
    corrected, receipts = correct_native_chemical_position_claims(source, response)
    assert response == original
    assert len(receipts) == 1
    assessment = corrected['evaluation_assessments'][0]
    assert assessment['uncertainty'] == original['evaluation_assessments'][0]['uncertainty']
    assert assessment['mechanistic_plausibility'].startswith('The alignment is a hypothesis. ')
    assert assessment['mechanistic_plausibility'].endswith('. No knockdown was measured.')
    assert '5 (5′ wing), 16 (3′ wing)' in assessment['mechanistic_plausibility']
    assert 'inside the DNA gap' not in assessment['mechanistic_plausibility']
    assert 'inside the DNA gap' not in corrected['markdown']
    assert native_chemical_position_findings(source, corrected) == []
    assert correct_native_chemical_position_claims(source, corrected) == (corrected, [])


def test_native_position_check_does_not_turn_region_mentions_into_location_claims():
    row = {'name': 'C', 'input_row_id': 'row_c', 'sequence': 'A' * 20,
           'chemistry': 'full-PS 5-10-5 DNA gapmer',
           'near_match_candidates': [{
               'mismatch_count': 2,
               'mismatch_positions_in_supplied_sequence_1based': [5, 16]}]}
    source = {'evaluation_summary': attach_measurement_references({'provided_asos': [row]})}
    for sentence in (
        'Mismatch positions 5 and 16 flank the DNA gap.',
        'The DNA gap remains intact, and mismatch positions 5 and 16 are recorded.',
        'If the duplex forms across the DNA gap (mismatches at positions 5 and 16), cleavage is plausible.',
        'The DNA gap remains intact, and mismatch positions 5 and 16 occur in the wings.',
    ):
        response = {'evaluation_assessments': [
            {'name': 'C', 'input_row_id': 'row_c',
             'mechanistic_plausibility': sentence}]}
        assert native_chemical_position_findings(source, response) == [], sentence
        assert correct_native_chemical_position_claims(source, response) == (response, [])
