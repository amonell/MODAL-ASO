from copy import deepcopy

from aso_agents.response_assembly import assembly_source


def test_historical_component_partition_cannot_change_current_delivery_membership():
    rows = [{'input_row_id': 'alpha', 'name': 'Alpha'},
            {'input_row_id': 'beta', 'name': 'Beta'},
            {'input_row_id': 'gamma', 'name': 'Gamma'}]
    payload = {'evaluation_summary': {'provided_asos': rows}}
    response = {'evaluation_assessments': deepcopy(rows),
                'answer_summary': 'All supplied observations are assessed.'}
    expected = assembly_source(payload, response)
    for groups in [[rows], [[row] for row in rows], [rows[:2], rows[2:]]]:
        draft = deepcopy(response)
        draft['original_component_responses'] = [
            {'evaluation_assessments': deepcopy(group),
             'answer_summary': 'This component assesses only its own input subset.'}
            for group in groups]
        before = deepcopy(draft)
        assert assembly_source(payload, draft) == expected
        assert draft == before
        assert expected['delivery_scope'] == {
            'assessment_count': 3, 'assessed_input_row_ids': ['alpha', 'beta', 'gamma']}
