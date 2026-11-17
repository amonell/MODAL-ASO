"""Missing biological inputs remain visible before full goal synthesis."""
import asyncio
from copy import deepcopy
import json

import pytest

from aso_agents import pipeline_tools as pipeline


def planned(context):
    return {'status': 'ready', 'questions': [], 'reason': None, 'request_envelope': {
        'schema_version': 'v3', 'request_mode': 'single_task', 'request_summary': 'Design gapmer ASOs',
        'request_classifications': ['design_new_asos'], 'tasks': [{
            'task_id': 'design', 'task_type': 'design_new_asos', 'label': 'Design gapmer ASOs',
            'native': True, 'can_run_in_parallel': False, 'context': deepcopy(context)}],
        'task_dependencies': [], 'final_outputs': ['ASO candidates'], 'response_mode': 'analysis',
        'requires_execution': True, 'assumptions': []}}


@pytest.mark.parametrize('goal_present', [False, True])
def test_missing_species_and_context_are_both_visible_before_synthesis(goal_present):
    context = {'gene_symbol': 'GENE1'}
    if goal_present:
        context['structured_goal'] = {'gene_symbol': 'GENE1'}
    envelope = planned(context)['request_envelope']
    before = deepcopy(envelope)
    entry = pipeline._compact_request_execution_validation(envelope)['tasks']['design']
    assert 'target_context_missing' in entry['issues']
    assert 'context.structured_goal.species' in entry['repairable_issues']
    assert envelope == before


def test_species_supplied_in_inherited_context_is_already_available():
    envelope = planned({'gene_symbol': 'GENE1'})['request_envelope']
    envelope['default_context'] = {'species': 'mouse', 'target_context': 'neurons'}
    entry = pipeline._compact_request_execution_validation(envelope)['tasks']['design']
    assert 'context.structured_goal.species' not in entry['repairable_issues']
    assert 'target_context_missing' not in entry['issues']


def test_confirmed_inputs_are_asked_together_before_expensive_graph_checks(monkeypatch):
    draft = planned({'gene_symbol': 'GENE1'})
    draft.update(status='need_clarification', questions=['Which biological context should be used?'],
                 reason='Missing biological context.')
    observed = []

    async def plan(_prompt):
        observed.append('plan')
        return deepcopy(draft)

    async def review(**kwargs):
        observed.append('review')
        body = json.loads(kwargs['prompt'])
        entry = body['platform_execution_validation']['tasks']['design']
        assert 'target_context_missing' in entry['issues']
        assert 'context.structured_goal.species' in entry['repairable_issues']
        return {'blocking_question_indexes': [0], 'additional_blocking_questions': [{
            'missing_field': 'context.structured_goal.species', 'question': 'Which organism should be used?',
            'reason': 'Organism is absent from the request and available evidence.'}],
            'reason': 'Both biological context and organism are needed.'}

    async def premature(*_args, **_kwargs):
        raise AssertionError('Confirmed missing input should reach the user before graph review or synthesis.')

    monkeypatch.setattr(pipeline, '_run_request_workbench_agent_once', plan)
    monkeypatch.setattr(pipeline, '_run_schema_validated_agent_once', review)
    monkeypatch.setattr(pipeline, '_run_request_graph_review_agent_once', premature)
    monkeypatch.setattr(pipeline, '_synthesize_missing_design_goals', premature)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once('Design GENE1 gapmers.'))
    assert result['status'] == 'need_clarification'
    assert result['questions'] == ['Which biological context should be used?', 'Which organism should be used?']
    assert result['request_envelope'] == draft['request_envelope']
    assert observed == ['plan', 'review']


def test_incomplete_model_fields_do_not_automatically_create_user_questions(monkeypatch):
    draft = planned({'gene_symbol': 'GENE1'})
    request = 'Design human GENE1 gapmers in neurons.'

    async def review(**kwargs):
        body = json.loads(kwargs['prompt'])
        assert body['request_evidence'] == request
        assert body['platform_execution_validation']['tasks']['design']['repairable_issues']
        return {'blocking_question_indexes': [], 'additional_blocking_questions': [],
                'reason': 'The original request supplies the inputs; synthesis can place them in the goal.'}

    monkeypatch.setattr(pipeline, '_run_schema_validated_agent_once', review)
    result = asyncio.run(pipeline._review_request_clarifications_if_needed(request, draft))
    assert result['status'] == 'ready' and result['questions'] == []
    assert result['request_envelope'] == draft['request_envelope']
