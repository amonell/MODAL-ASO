from copy import deepcopy

import pytest

from aso_agents.sequence_orientation import interpret_orientation, preferred_binding_hits
from aso_agents.evaluation_report import evaluation_markdown


def row():
    return {'name':'sample', 'sequence':'AACCGGTA', 'reverse_complement_sequence':'TACCGGTT',
            'mapping_hits':[{'match_orientation':'input_sequence','matched_target_sequence':'AACCGGTA'}]}


def test_same_strand_scan_requires_confirmation_without_substitution():
    entry = row()
    before = deepcopy(entry)
    result = interpret_orientation(entry)
    assert result['requires_confirmation']
    assert result['evaluated_molecule_sequence'] == 'AACCGGTA'
    assert result['conditional_reverse_complement_sequence'] == 'TACCGGTT'
    assert result['provided_sequence_role'] == 'unknown'
    assert not result['original_molecule_exact_binding']
    assert result['likely_reversed_entry'] is True
    assert 'Likely reversed entry' in result['interpretation_note']
    assert preferred_binding_hits(entry, entry['mapping_hits']) == []
    assert entry == before


@pytest.mark.parametrize('role', ['antisense_oligo_sequence','unknown'])
def test_original_or_uncertain_reply_never_authorizes_reverse_complement(role):
    entry = {**row(), 'sequence_role_confirmation':{'role':role}}
    result = interpret_orientation(entry)
    assert not result['requires_confirmation']
    assert result['evaluated_molecule_sequence'] == entry['sequence']
    assert not preferred_binding_hits(entry, entry['mapping_hits'])
    if role == 'antisense_oligo_sequence':
        assert not result['likely_reversed_entry']


def test_declared_physical_aso_does_not_trigger_orientation_clarification():
    entry = {**row(), 'declared_sequence_role': 'antisense_oligo_sequence'}
    result = interpret_orientation(entry)
    assert not result['requires_confirmation']
    assert not result['likely_reversed_entry']
    assert result['evaluated_molecule_sequence'] == entry['sequence']
    assert result['conditional_reverse_complement_sequence'] == entry['reverse_complement_sequence']


def test_explicit_target_confirmation_allows_complement_and_preserves_input():
    entry = {**row(), 'sequence_role_confirmation':{'role':'target_site_sequence'}}
    result = interpret_orientation(entry)
    assert result['input_sequence'] == entry['sequence']
    assert result['evaluated_molecule_sequence'] == entry['reverse_complement_sequence']
    assert preferred_binding_hits(entry, entry['mapping_hits']) == entry['mapping_hits']
    assert result['likely_reversed_entry'] is False


def test_antisense_hit_has_priority_even_if_sense_match_appears_first():
    entry = row()
    hit = {'match_orientation':'reverse_complement','matched_target_sequence':'TACCGGTT'}
    entry['mapping_hits'].append(hit)
    result = interpret_orientation(entry)
    assert not result['requires_confirmation']
    assert result['matched_target_sequence'] == 'TACCGGTT'
    assert result['synthesized_antisense_sequence'] == entry['sequence']
    assert preferred_binding_hits(entry, entry['mapping_hits']) == [hit]
    assert result['likely_reversed_entry'] is False
    assert len(entry['mapping_hits']) == 2


def test_palindrome_does_not_require_confirmation():
    entry = {'sequence':'ACGT','reverse_complement_sequence':'ACGT',
             'mapping_hits':[{'match_orientation':'input_sequence','matched_target_sequence':'ACGT'}]}
    result = interpret_orientation(entry)
    assert not result['requires_confirmation']
    assert result['original_molecule_exact_binding']


def test_evaluation_markdown_respects_confirmed_physical_molecule():
    entry = row()
    entry['input_row_id'] = 'second_submitted_oligo'
    entry['name'] = 'second_submitted_oligo'
    entry['sequence_role_confirmation'] = {'role': 'antisense_oligo_sequence'}
    entry['orientation_interpretation'] = interpret_orientation(entry)
    result = evaluation_markdown(
        {'answer_summary': 'Orientation audit',
         'evaluation_assessments': [{'input_row_id': 'second_submitted_oligo',
                                     'name': 'second_submitted_oligo'}]},
        {'provided_asos': [entry]},
    )
    assert 'Likely reversed entry: the supplied molecule has no exact antisense' not in result
    assert 'Likely reversed entry' not in result


@pytest.mark.parametrize('chemistry,length,wing', [('5-10-5',20,5),('3-10-3',16,3),('steric-hindrance',18,18)])
def test_idt_encoding_has_all_requested_ps_linkages_and_sugars(chemistry,length,wing):
    import run_goal_intake as cli
    encoded = cli._apply_idt_configuration(('ACGT' * 5)[:length], chemistry)
    units = encoded.split('*')
    assert len(units) == length and all(units)
    for index, unit in enumerate(units):
        is_moe = chemistry == 'steric-hindrance' or index < wing or index >= length - wing
        assert ('2MOEr' in unit) == is_moe
    assert not encoded.endswith('*') and not encoded.startswith('*')
