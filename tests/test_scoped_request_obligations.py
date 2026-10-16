"""Original user counts survive supporting tasks and scoped child execution."""
import asyncio
from copy import deepcopy
import json

import pytest

import run_goal_intake as intake
from aso_agents import pipeline_tools as p


REQUEST = 'Design five distinct ASOs; return at least three for each nominated interval. Also analyze overlaps and write a report.'
AUDIT = {'candidate_count': {'count': 5, 'kind': 'exact', 'scope': 'global',
                             'quote': 'Design five distinct ASOs'},
         'per_site_count': {'count': 3, 'kind': 'minimum',
                            'quote': 'at least three for each nominated interval'}}


def draft():
    goal = {'candidate_minimum': 1, 'candidate_limit': None, 'design_programs': [
        {'program_id': 'panel', 'modality': 'steric', 'intent': 'binding',
         'objective': 'Screen nominated intervals'}]}
    design = {'task_id': 'design', 'task_type': 'design_new_asos',
              'context': {'structured_goal': deepcopy(goal)}}
    others = [{'task_id': 'analysis', 'task_type': 'sandbox_analysis',
               'context': {'analysis_instruction': 'Analyze overlaps', 'candidate_limit': 99}},
              {'task_id': 'report', 'task_type': 'build_composite_report',
               'context': {'candidate_limit': 88}}]
    return {'status': 'ready', 'structured_goal': goal,
            'request_envelope': {'tasks': [design, *others]}}


def stub_auditor(monkeypatch, calls):
    async def respond(agent, *, input, **kwargs):
        calls.append(json.loads(input))
        return type('Response', (), {'final_output': json.dumps(AUDIT)})()
    monkeypatch.setattr(intake.Runner, 'run', respond)


def test_single_design_with_supporting_tasks_is_audited_and_only_design_is_changed(monkeypatch):
    payload = draft(); other_tasks = deepcopy(payload['request_envelope']['tasks'][1:]); calls=[]
    stub_auditor(monkeypatch, calls)
    result = asyncio.run(intake._audit_request_obligations(payload, REQUEST, []))
    assert calls == [{'original_request': REQUEST, 'clarification_history': []}]
    goal = result['structured_goal']
    assert (goal['candidate_minimum'], goal['candidate_limit'], goal['min_asos_per_site']) == (5, 5, 3)
    assert 'candidate_limit_per_site' not in goal
    nested = result['request_envelope']['tasks'][0]['context']['structured_goal']
    assert (nested['candidate_limit'], nested['min_asos_per_site']) == (5, 3)
    assert result['request_envelope']['tasks'][1:] == other_tasks
    assert result['request_obligation_audit']['candidate_count'] == AUDIT['candidate_count']


def test_global_count_is_not_copied_into_multiple_independent_design_tasks(monkeypatch):
    payload = draft()
    second = deepcopy(payload['request_envelope']['tasks'][0]); second['task_id'] = 'different_design'
    payload['request_envelope']['tasks'].append(second)
    before = deepcopy(payload)
    async def forbidden(*args, **kwargs):
        pytest.fail('Cannot assign one global panel count independently to two design tasks')
    monkeypatch.setattr(intake.Runner, 'run', forbidden)
    assert asyncio.run(intake._audit_request_obligations(payload, REQUEST, [])) == before


def test_original_evidence_survives_nested_scope_and_later_user_answers():
    parent = {'request': REQUEST, 'canonical_request_envelope': draft()['request_envelope'],
              'clarification_qa_history': [{'question': 'Chemistry?', 'answer': 'Fully modified'}]}
    task = parent['canonical_request_envelope']['tasks'][0]
    evidence = p._scoped_obligation_evidence(parent, task)
    assert evidence['original_request'] == REQUEST and evidence['qa_history'][0]['answer'] == 'Fully modified'
    child = {'request': 'Generated JSON contains candidate_minimum: 1',
             'request_envelope_preseed': {'tasks': [task]}, 'request_obligation_evidence': evidence}
    history = [{'question': 'Additional constraint?', 'answer': 'No repeated molecules'}]
    request, answers = p._obligation_audit_source(child, child['request'], history)
    assert request == REQUEST and len(answers) == 2
    child['canonical_request_envelope'] = {'tasks': [task]}
    assert p._scoped_obligation_evidence(child, task) == evidence
    assert parent['clarification_qa_history'] == evidence['qa_history']


def test_unrelated_or_ambiguous_task_cannot_inherit_global_evidence():
    parent = {'request': REQUEST, 'canonical_request_envelope': draft()['request_envelope']}
    task = parent['canonical_request_envelope']['tasks'][0]
    assert p._scoped_obligation_evidence(parent, {'task_id': 'wrong'}) == {}
    evidence = p._scoped_obligation_evidence(parent, task)
    child = {'request_envelope_preseed': {'tasks': [{**task, 'task_id': 'wrong'}]},
             'request_obligation_evidence': evidence}
    assert p._obligation_audit_source(child, 'scoped', []) == ('scoped', [])
    parent['canonical_request_envelope']['tasks'].append({**task, 'task_id': 'second'})
    assert p._scoped_obligation_evidence(parent, task) == {}


def test_native_fast_path_updates_canonical_child_preseed(monkeypatch, tmp_path):
    goal = {'gene_symbol': 'GENE1', 'species': 'human', 'target_context': 'cells',
            'intent': 'binding', **draft()['structured_goal']}
    envelope = p._build_request_envelope(request_text=REQUEST, request_id='obligations', structured_goal=goal)
    task_id = envelope['tasks'][0]['task_id']
    monkeypatch.setattr(p, '_request_parse_output_from_envelope_preseed',
                        lambda *a, **kw: {'status': 'ready', 'questions': [], 'request_envelope': deepcopy(envelope)})
    monkeypatch.setattr(p, '_repair_request_envelope_if_needed',
                        lambda **kw: {'status': 'ready', 'request_envelope': kw['request_envelope']})
    calls=[]; stub_auditor(monkeypatch,calls)
    run_dir=tmp_path/'run';run_dir.mkdir()
    state_path=tmp_path/'state.json'
    state_path.write_text(json.dumps({'version':'2','status':'initialized','phase':'initialized',
        'request':REQUEST,'run_dir':str(run_dir),'flags':{'skip_ensembl_check':True},'last_error':None,
        'pending_questions':[],'clarification':{'isoform_scope_confirmed':False},
        'clarification_qa_history':[],'force_rerun_stages':[],'events':[],'artifacts':{}}))
    result=p.tool_parse_goal_request(str(state_path));assert result['status']=='ok',result
    state=p._load_state(result['state_path'])
    canonical=state['canonical_request_envelope']['tasks'][0]['context']['structured_goal']
    assert (canonical['candidate_limit'],canonical['min_asos_per_site'])==(5,3)
    runtime_task=next(t for t in state['request_envelope']['tasks'] if t['task_id']==task_id)
    preseed=p._task_scoped_child_preseed_envelope(state,state['request_envelope'],runtime_task)
    inherited=preseed['tasks'][0]['context']['structured_goal']
    assert inherited['candidate_limit']==5 and inherited['min_asos_per_site']==3
    assert calls[0]['original_request']==REQUEST
