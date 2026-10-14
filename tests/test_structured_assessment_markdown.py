from copy import deepcopy

from aso_agents.response_delivery import normalize_structured_assessment_markdown
from aso_agents.response_contracts import response_contract_errors


def assessment(row):
    return {
        'input_row_id': row, 'name': 'observation ' + row,
        'mechanistic_plausibility': 'Mechanism supported for ' + row,
        'selectivity_limitations': 'Screening unresolved for ' + row,
        'uncertainty': 'Activity unmeasured for ' + row,
        'ranking_rationale': 'No measured rank for ' + row,
        'evidence_refs': ['evidence/' + row],
    }


def test_lossless_markdown_view_of_complete_structured_assessments():
    original = {'answer_summary': 'Two submitted observations were assessed.',
                'markdown': '', 'evaluation_assessments': [assessment('row-A'), assessment('row-B')],
                'limitations': ['No measured potency.']}
    untouched = deepcopy(original)
    result = normalize_structured_assessment_markdown(original)
    assert original == untouched
    assert not response_contract_errors(result, {'task_type': 'build_composite_report'})
    for row in original['evaluation_assessments']:
        assert row['input_row_id'] in result['markdown']
        for key in ('mechanistic_plausibility', 'selectivity_limitations', 'uncertainty', 'ranking_rationale'):
            assert row[key] in result['markdown']
    assert 'No measured potency.' in result['markdown']
    assert result['evaluation_assessments'] == original['evaluation_assessments']
    assert result['native_delivery_normalization']['assessment_text_changed'] is False


def test_incomplete_or_ambiguous_assessments_are_not_repaired():
    original = {'answer_summary': 'Some assessment.', 'markdown': '',
                'evaluation_assessments': [assessment('row-A'), assessment('row-A')]}
    assert normalize_structured_assessment_markdown(original) is original
    assert response_contract_errors(original, {'task_type': 'build_composite_report'})
    original['evaluation_assessments'][1]['input_row_id'] = 'row-B'
    del original['evaluation_assessments'][1]['uncertainty']
    assert normalize_structured_assessment_markdown(original) is original


def test_existing_markdown_and_missing_summary_are_preserved():
    original = {'answer_summary': 'Summary', 'markdown': 'Full report',
                'evaluation_assessments': [assessment('row-A')]}
    assert normalize_structured_assessment_markdown(original) is original
    original['markdown'] = ''
    original['answer_summary'] = ''
    assert normalize_structured_assessment_markdown(original) is original


def test_keyed_assessments_render_only_when_keys_match_row_identity():
    original = {'answer_summary': 'Two submitted observations.', 'markdown': '',
                'evaluation_assessments': {'row-A': assessment('row-A'), 'row-B': assessment('row-B')}}
    result = normalize_structured_assessment_markdown(original)
    assert 'row-A' in result['markdown'] and 'row-B' in result['markdown']
    assert result['evaluation_assessments'] == original['evaluation_assessments']
    original['evaluation_assessments']['row-B']['input_row_id'] = 'different-row'
    assert normalize_structured_assessment_markdown(original) is original


def test_unbound_context_citations_cannot_be_rendered_or_accepted():
    original = {'answer_summary': 'Observation reviewed.', 'markdown': '',
                'evaluation_assessments': [assessment('row-A')]}
    original['evaluation_assessments'][0]['ranking_rationale'] = 'Rank based on {{E00081}}.'
    assert normalize_structured_assessment_markdown(original) is original
    assert 'Unresolved citation placeholder in the response.' in response_contract_errors(
        {**original, 'markdown': 'Rank based on {{E00081}}.'},
        {'task_type': 'build_composite_report'})
