from copy import deepcopy
import json

from aso_agents.evaluation_response_audit import response_audit_context, orientation_audit_packets, pointer_value
from aso_agents.response_context import render_response_context, decode_response_context


def test_index_records_exact_row_scopes_without_removing_complete_evidence():
    source = {'evaluation_summary': {'provided_asos': [
        {'input_row_id': 'row_a', 'evidence_inventory': {'feature_evidence_availability': {
            'requested_site_register': {'status': 'not_recorded'},
            'independent_reference_junction_geometry': {'status': 'recorded'}}},
         'requested_feature_membership': [{'identity_resolution': {'status': 'unresolved'}}],
         'near_match_search': {'genome_wide_search': False}, 'all_alignment_records': [1, 2, 3]},
        {'input_row_id': 'row_b', 'evidence_inventory': {'feature_evidence_availability': {
            'requested_site_register': {'status': 'recorded'},
            'independent_reference_junction_geometry': {'status': 'not_recorded'}}}},
    ], 'off_target_count_scope': {'search': 'full reference', 'count': 'filtered hits'},
       'measurement_evidence': {'duplicate': {'value': {'near_match_search': {'original': True}}}}}}
    response = {'answer_summary': 'Activity is unmeasured.'}
    before = deepcopy((source, response))
    context = response_audit_context(source, response)
    assert (context['source'], context['draft_response']) == before
    assert (source, response) == before
    entries = {entry['source_pointer']: entry['recorded_value'] for entry in context['evidence_scope_index']}
    for path, value in entries.items():
        assert pointer_value(source, path) == value
    assert entries['/evaluation_summary/provided_asos/0/evidence_inventory/feature_evidence_availability']['requested_site_register']['status'] == 'not_recorded'
    assert entries['/evaluation_summary/provided_asos/1/evidence_inventory/feature_evidence_availability']['requested_site_register']['status'] == 'recorded'
    assert '/evaluation_summary/off_target_count_scope' in entries
    assert all('/measurement_evidence/' not in path for path in entries)
    assert context['source']['evaluation_summary']['measurement_evidence'] == source['evaluation_summary']['measurement_evidence']
    rendered = json.loads(render_response_context(context))
    decoded = decode_response_context(rendered) if rendered.get('context_encoding') else rendered
    assert decoded == context


def test_index_escapes_source_keys_and_preserves_numbering_provenance():
    source = {'panel/name': [{'identity_resolution': {'status': 'resolved_reference_ordinal',
        'biological_label_verified': False, 'numbering_convention': 'one_based_transcript_ordinal'}}]}
    context = response_audit_context(source, {'answer_summary': 'Reference membership is separate from biological aliases.'})
    assert context['evidence_scope_index'][0]['source_pointer'] == '/panel~1name/0/identity_resolution'
    assert pointer_value(source, context['evidence_scope_index'][0]['source_pointer']) == context['evidence_scope_index'][0]['recorded_value']


def test_context_is_detached_and_does_not_invent_missing_scope_metadata():
    source = {'arbitrary_records': [{'value': [3]}]}
    response = {'answer_summary': 'Unknown.'}
    context = response_audit_context(source, response)
    assert context['evidence_scope_index'] == []
    context['source']['arbitrary_records'][0]['value'].append(7)
    context['draft_response']['answer_summary'] = 'Changed.'
    assert source == {'arbitrary_records': [{'value': [3]}]}
    assert response['answer_summary'] == 'Unknown.'


def test_scope_index_and_complete_source_are_separate_exact_copies():
    source = {'off_target_count_scope': {'categories': ['exact']}}
    context = response_audit_context(source, {})
    context['evidence_scope_index'][0]['recorded_value']['categories'].append('different')
    assert context['source'] == source == {'off_target_count_scope': {'categories': ['exact']}}


def test_orientation_index_exposes_original_and_conditional_molecule_separately():
    source = {'evaluation_summary': {'provided_asos': [{
        'input_row_id': 'row_a', 'orientation_interpretation': {
            'input_sequence': 'GCCT', 'evaluated_molecule_sequence': 'GCCT',
            'match_orientation': 'input_sequence', 'matched_target_sequence': 'GCCT',
            'original_molecule_exact_binding': False, 'likely_reversed_entry': True,
            'conditional_reverse_complement_sequence': 'AGGC',
            'sequence_role_confirmation': {'answers': ['large private history']}}}]}}
    context = response_audit_context(source, {'answer_summary': 'Conditional site is exact.'})
    entries = {entry['source_pointer']: entry['recorded_value']
               for entry in context['evidence_scope_index']}
    prefix = '/evaluation_summary/provided_asos/0/orientation_interpretation/'
    assert entries[prefix + 'matched_target_sequence'] == 'GCCT'
    assert entries[prefix + 'conditional_reverse_complement_sequence'] == 'AGGC'
    assert entries[prefix + 'original_molecule_exact_binding'] is False
    assert prefix + 'sequence_role_confirmation' not in entries
    assert all(pointer_value(source, path) == value for path, value in entries.items())


def test_focused_orientation_packet_uses_exact_native_pointers_and_bound_draft():
    source = {'request_text': 'Evaluate the written molecule and its separate orientation.',
              'evaluation_summary': {'provided_asos': [{
                  'name': 'test', 'input_row_id': 'row_1', 'sequence': 'GCCT',
                  'orientation_interpretation': {
                      'input_sequence': 'GCCT', 'matched_target_sequence': 'GCCT',
                      'match_orientation': 'input_sequence',
                      'conditional_reverse_complement_sequence': 'AGGC',
                      'original_molecule_exact_binding': False}}]}}
    response = {'answer_summary': 'The alternative is exact.',
                'evaluation_assessments': [{'name': 'test', 'input_row_id': 'row_1',
                    'mechanistic_plausibility': 'The alternative has no exact site.',
                    'ranking_rationale': 'The written molecule lacks exact binding.',
                    'selectivity_limitations': 'Unknown.'}]}
    packets = orientation_audit_packets(source, response)
    assert len(packets) == 1
    packet = packets[0]
    assert packet['verified_sequence_relation']['conditional_alternative_reverse_complement_equals_recorded_rna_window']
    assert {item['response_path'] for item in packet['draft_narrative']} == {
        '/evaluation_assessments/0/mechanistic_plausibility',
        '/evaluation_assessments/0/ranking_rationale'}
    assert all(pointer_value(source, entry['source_pointer']) == entry['recorded_value']
               for entry in packet['orientation_evidence'])
    source['evaluation_summary']['provided_asos'][0]['orientation_interpretation']['matched_target_sequence'] = 'AAAA'
    assert orientation_audit_packets(source, response) == []


def test_focused_orientation_packet_checks_cross_row_ranking_claim():
    source = {'request_text': 'Compare these two molecules and their orientations.',
              'evaluation_summary': {
                  'provided_asos': [{'name': 'A', 'input_row_id': 'row_a', 'sequence': 'AGGC'}],
                  'panel_comparison_overview': [
                      {'name': 'A', 'input_row_id': 'row_a', 'sequence': 'AGGC'},
                      {'name': 'B', 'input_row_id': 'row_b', 'sequence': 'GCCT',
                       'orientation_interpretation': {
                           'input_sequence': 'GCCT', 'matched_target_sequence': 'GCCT',
                           'match_orientation': 'input_sequence',
                           'conditional_reverse_complement_sequence': 'AGGC',
                           'original_molecule_exact_binding': False,
                           'interpretation_note': 'The alternative is exact at this RNA window.'}}]}}
    response = {'evaluation_assessments': [{'name': 'A', 'input_row_id': 'row_a',
                 'ranking_rationale': 'B and its reverse-complement alternative have no exact site.'}]}
    packets = orientation_audit_packets(source, response)
    assert len(packets) == 1
    assert packets[0]['orientation_subject']['source'] == 'panel_comparison_overview'
    assert packets[0]['draft_narrative'][0]['response_path'] == '/evaluation_assessments/0/ranking_rationale'
    assert all(pointer_value(source, entry['source_pointer']) == entry['recorded_value']
               for entry in packets[0]['orientation_evidence'])


def test_cross_row_orientation_packet_does_not_rebind_generic_assessment():
    source = {'evaluation_summary': {
        'provided_asos': [{'name': 'Current', 'input_row_id': 'row_current'}],
        'panel_comparison_overview': [
            {'name': 'Current', 'input_row_id': 'row_current'},
            {'name': 'Sibling', 'input_row_id': 'row_sibling',
             'sequence': 'GCCT', 'orientation_interpretation': {
                 'input_sequence': 'GCCT', 'matched_target_sequence': 'GCCT',
                 'match_orientation': 'input_sequence',
                 'conditional_reverse_complement_sequence': 'AGGC',
                 'original_molecule_exact_binding': False}}]}}
    response = {'evaluation_assessments': [{
        'name': 'Current', 'input_row_id': 'row_current',
        'ranking_rationale': 'No exact antisense alignment for the submitted molecule is recorded.'}]}
    assert orientation_audit_packets(source, response) == []
    response['evaluation_assessments'][0]['ranking_rationale'] = (
        'Sibling and its reverse-complement alternative have no exact site.')
    packets = orientation_audit_packets(source, response)
    assert len(packets) == 1
    assert packets[0]['orientation_subject']['input_row_id'] == 'row_sibling'
