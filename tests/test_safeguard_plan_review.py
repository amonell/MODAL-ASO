"""The model reviews biological relevance; execution preserves measured policy."""
from copy import deepcopy
import importlib
import json

import pytest
import run_goal_intake as runtime
from aso_agents import safeguard_review as review
from aso_agents.splice_recommendations import safeguard_priority
from test_target_exon_safeguards import exon_goal, guards_for


def case(position=440, **changes):
    primary = {'label':'acceptor@400', 'site':'acceptor', 'position':400, 'chrom':'chr2', 'strand':'+'}
    candidate = {'label':f'donor@{position}', 'site':'donor', 'position':position, 'chrom':'chr2', 'strand':'+'}
    return review.review_case(primary, {**candidate, **changes}, 'keep', 'Distinct reference junction',
                              locked=changes.get('required', False))


def decision(item, policy='protect_both', **changes):
    roles = {'protect_both':('uncertain','unwanted'), 'limit_increase':('unwanted','intended'),
             'limit_decrease':('intended','unwanted'), 'observe':('acceptable','acceptable'),
             'keep_explicit':('uncertain','uncertain')}
    increase, decrease = roles[policy]
    return {'case_id':item['case_id'], 'candidate_label':item['candidate_label'], 'policy':policy, 'increase_role':increase,
            'decrease_role':decrease, 'rationale':'Preserve this distinct reference junction.', **changes}


def reviewer(payload, traces=None):
    decisions = []
    for item in payload['cases']:
        policy = 'keep_explicit' if item['locked'] else item['proposed_policy']
        decisions.append(decision(item, policy))
    return {'decisions':decisions}


@pytest.mark.parametrize('strand', ['+','-'])
@pytest.mark.parametrize('effect,bound', [('inclusion','max_decrease'), ('skipping','max_increase')])
def test_every_automatic_case_gets_review_with_current_objective(monkeypatch, strand, effect, bound):
    payloads = []
    def respond(payload, traces=None):
        payloads.append(deepcopy(payload))
        return reviewer(payload)
    monkeypatch.setattr(review, '_review_batch', respond)
    goal = exon_goal(strand, effect)
    guards = guards_for(goal)
    audit = goal['safeguard_selection_audit']['model_review']
    assert audit['status'] == 'completed' and audit['case_count'] == audit['reviewed_count']
    cases = [item for payload in payloads for item in payload['cases']]
    assert len(cases) == 6  # Two routes, each with the partner and both neighboring sites.
    assert all(payload['context']['job']['desired_effect'] == effect for payload in payloads)
    assert all(payload['context']['selected_references'] for payload in payloads)
    paired = [item for item in cases if item['paired_exon']]
    assert len(paired) == 2
    partner = runtime._canonical_site_label(goal['splice_site_plan'][1])
    assert next(g for g in guards if g['label'] == partner)[bound] == .05
    assert all(g['review']['status'] == 'reviewed' for g in guards)


def test_model_can_change_automatic_direction_and_reason(monkeypatch):
    def respond(payload, traces=None):
        return {'decisions':[decision(c, 'limit_increase', rationale='The requested outcome favors a decrease here.')
                             for c in payload['cases']]}
    monkeypatch.setattr(review, '_review_batch', respond)
    guards = guards_for(exon_goal())
    assert all(g.get('max_increase') == .05 and 'max_decrease' not in g for g in guards)
    assert all(g['note'] == 'The requested outcome favors a decrease here.' for g in guards)


def test_review_receives_paired_geometry_even_when_objectives_conflict(monkeypatch):
    from aso_agents.splice_safeguards import paired_exon_guard, target_exon_pairs
    goal = exon_goal()
    primary, partner = goal['splice_site_plan']
    candidate = {**partner, 'label':'donor@440', 'conflict_between_objectives':True}
    assert 'paired_exon_objective' not in paired_exon_guard(candidate, primary, target_exon_pairs(goal,[primary]))
    # A locked constraint keeps the pair visible to the reviewer while execution
    # still honors its explicit bounds.
    captured = []
    def respond(payload, traces=None):
        captured.extend(payload['cases'])
        return reviewer(payload)
    monkeypatch.setattr(review, '_review_batch', respond)
    guards_for(goal, candidates=[{**candidate, 'required':True, 'max_abs_delta':.02}])
    reviewed = next(c for c in captured if c['primary_label']=='acceptor@400' and c['candidate_label']=='donor@440')
    assert reviewed['paired_exon'] and reviewed['locked']


def test_unresolved_requested_numbering_is_explicit_in_review_context(monkeypatch):
    goal = exon_goal()
    goal['routing_job'].pop('exon_index')
    goal['routing_job'].update(exon_label=7, transcript_id='UNRESOLVED.1')
    monkeypatch.setattr(review, '_review_batch', reviewer)
    guards_for(goal)
    context = goal['safeguard_selection_audit']['model_review']['context']
    assert context['exon_objectives'][0]['identity']['status']=='unresolved'


def test_cached_review_is_replaced_for_current_job(monkeypatch):
    monkeypatch.setattr(review, '_review_batch', reviewer)
    goal = exon_goal()
    mapping = runtime._build_safeguard_candidates_by_target(goal['splice_site_plan'], [], goal)
    cached = deepcopy(mapping['acceptor@400'])
    before = deepcopy(cached)
    next_goal = exon_goal(effect='skipping')
    guards = guards_for(next_goal, candidates=cached)
    partner = next(g for g in guards if g['label'] == 'donor@440')
    assert partner['review']['policy'] == 'limit_increase'
    assert partner['max_increase'] == .05 and 'max_decrease' not in partner
    assert cached == before


def test_explicit_caps_and_preservation_are_reviewed_and_immutable(monkeypatch):
    from test_preservation_measurement_scope import fixture
    goal = fixture()
    captured = []
    def respond(payload, traces=None):
        captured.extend(deepcopy(payload['cases']))
        return reviewer(payload)
    monkeypatch.setattr(review, '_review_batch', respond)
    mapping = runtime._build_safeguard_candidates_by_target(goal['splice_site_plan'],
        [{'label':'donor@230','max_abs_delta':.2,'required':True}], goal, {'donor@250'})
    guards = runtime._default_safeguards_from_candidates(mapping['donor@250'])
    required = {g['label']:g for g in guards if g.get('required')}
    assert set(required) == {'acceptor@201','donor@230'}
    assert required['donor@230']['max_abs_delta'] == .2
    assert all(g['review']['policy'] == 'keep_explicit' for g in required.values())
    reviewed = {c['candidate_label']:c for c in captured if c['primary_label']=='donor@250'}
    assert all(reviewed[label]['locked'] for label in required)
    assert reviewed['donor@230']['explicit_bounds'] == {'max_abs_delta':.2}


@pytest.mark.parametrize('mutation', [
    {'case_id':'invented'}, {'case_id':[]}, {'candidate_label':'acceptor@400'}, {'policy':'ignore'}, {'policy':[]},
    {'increase_role':'harmless'}, {'increase_role':[]}, {'rationale':''}, {'rationale':' '},
    {'rationale':'x'*601}, {'max_abs_delta':.8}, {'required':False},
    {'policy':'keep_explicit'}, {'policy':'observe'},
])
def test_invalid_model_decision_is_repaired_then_retains_default(monkeypatch, mutation):
    calls = []
    def invalid(payload, traces=None):
        calls.append(payload)
        return {'decisions':[{**decision(payload['cases'][0]), **mutation}]}
    monkeypatch.setattr(review, '_review_batch', invalid)
    item = case()
    result = review.review_plan({}, [item])
    assert result['status'] == 'unavailable' and result['unreviewed_case_ids'] == [item['case_id']]
    assert len(calls) == 2 and calls[1]['corrections_needed']


@pytest.mark.parametrize('policy', ['observe','limit_increase','limit_decrease','protect_both'])
def test_model_cannot_weaken_explicit_policy(monkeypatch, policy):
    monkeypatch.setattr(review, '_review_batch', lambda payload, traces=None:
                        {'decisions':[decision(payload['cases'][0], policy)]})
    goal = exon_goal()
    guards = guards_for(goal, candidates=[{'label':'donor@440','max_abs_delta':.012,'required':True}])
    guard = next(g for g in guards if g['label']=='donor@440')
    assert guard['max_abs_delta'] == .012 and guard['required'] is True
    assert guard['review']['status'] == 'unavailable'


def test_duplicate_response_and_unknown_identity_cannot_pass():
    item = case()
    for response in ({'decisions':[decision(item),decision(item)]}, {'decisions':[], 'extra':True}):
        accepted, errors = review._validate(response, [item])
        assert not accepted and item['case_id'] in errors
    unresolved = review.review_case({'label':'acceptor@400'}, {'label':'donor@440'}, 'keep', 'Unresolved identity')
    accepted, errors = review._validate({'decisions':[decision(unresolved, 'observe')]}, [unresolved])
    assert not accepted and errors
    accepted, errors = review._validate({'decisions':[decision(unresolved)]}, [unresolved])
    assert accepted and not errors


def test_partial_response_repairs_only_missing_cases(monkeypatch):
    items = [case(),case(740)]
    calls = []
    def respond(payload, traces=None):
        calls.append(deepcopy(payload))
        return {'decisions':[decision(payload['cases'][0])]}
    monkeypatch.setattr(review, '_review_batch', respond)
    result = review.review_plan({}, items)
    assert result['status']=='completed' and result['reviewed_count']==2
    assert calls[1]['cases'] == [items[1]]


def test_large_plan_batches_every_case_and_preserves_successes_on_outage(monkeypatch):
    items = [case(440+i) for i in range(35)]
    calls = []
    def respond(payload, traces=None):
        calls.append(deepcopy(payload))
        if len(calls)==2:
            raise ConnectionError('provider detail should stay out of report')
        return reviewer(payload)
    monkeypatch.setattr(review, '_review_batch', respond)
    result = review.review_plan({}, items)
    assert [len(c['cases']) for c in calls] == [16,16,3]
    assert result['status']=='partial' and result['reviewed_count']==19
    assert len(result['unreviewed_case_ids']) == 16
    assert result['batches'][1]['attempts'] == [{'error_type':'ConnectionError'}]


def test_oversized_case_is_disclosed_without_losing_other_cases(monkeypatch):
    item = case(); item['proposal_reason'] = 'x'*review.MAX_BATCH_CHARACTERS
    monkeypatch.setattr(review, '_review_batch', reviewer)
    result = review.review_plan({}, [item,case(740)])
    assert result['status']=='partial' and result['reviewed_count']==1
    assert result['oversized_case_ids'] == [item['case_id']]


def test_observation_keeps_measurements_without_filtering_or_rank_penalty(tmp_path):
    router = importlib.import_module('aso_agents.splice_strategy_router_agent')
    rows = [{'Name':name, 'Sequence':'ACGT', 'delta_names':'acceptor@400,donor@440',
             'delta_deletion':f'{primary},{neighbor}', 'delta_replacement':f'{primary},{neighbor}',
             'shared_context_label':'donor@440', 'shared_context_abs_delta':abs(neighbor),
             'shared_context_delta':neighbor} for name,primary,neighbor in
             [('strong',.8,-.9),('weak',.6,.01)]]
    source = tmp_path/'scores.json'; source.write_text(json.dumps({'rows':rows}))
    output = tmp_path/'route.json'
    guard = {'label':'donor@440','policy':'observe', 'review':{'status':'reviewed','policy':'observe'}}
    args = dict(json_path=str(source), primary_label='acceptor@400', primary_direction='increase',
                safeguards_json=json.dumps([guard]), min_count=1, max_results=0, output_path=str(output))
    result = router.auto_threshold_constrained_site_metrics(**args)
    assert not result.get('error')
    data = json.loads(output.read_text())
    assert [r['Name'] for r in data['rows']]==['strong','weak']
    for row in data['rows']:
        assert row['safeguards_status']=='monitoring_only'
        assert row['safeguards_passed'] is None
        assert row['safeguard_priority']['status']=='monitoring_only'
        assert row['safeguard_priority']['rank_key']==[2,0]
        assert row['safeguard_report'][0]['status']=='observed'
    assert data['rows'][0]['safeguard_report'][0]['values']==[-.9,-.9]
    import csv
    csv_path,error = runtime._write_strategy_route_csv(str(output))
    assert not error
    with open(csv_path) as stream:
        exported = list(csv.DictReader(stream))
    assert all(r['safeguards_status']=='monitoring_only' for r in exported)
    assert all(r['safeguards_thresholds']=='observe' for r in exported)
    assert exported[0]['safeguards_deletion_values']=='-0.9'
    assert not router._shared_context_blocked('donor@440',-.9,.9,configured_guard_labels={'donor@440'})
    for conflicting in ({'required':True},{'max_abs_delta':.2}):
        args['safeguards_json']=json.dumps([{**guard,**conflicting}])
        assert 'safeguard_observation_conflicts' in router.auto_threshold_constrained_site_metrics(**args)['error']


def test_observation_and_missing_active_measurement_have_distinct_priority():
    result = safeguard_priority([{'policy':'observe','passes':None}, {'passes':None}])
    assert result['status']=='unmeasured' and result['unmeasured_count']==1
    assert result['guard_count']==1 and result['observation_count']==1


def test_fallback_visible_in_candidate_assessment_and_shortlist():
    from aso_agents.candidate_assessment import assess_candidates
    from aso_agents.report_assessment import prediction_note, render_shortlist_markdown
    row = {'Sequence':'ACGTACGT', 'safeguard_report':[{'review':{'status':'unavailable'},'passes':True}]}
    assessed = assess_candidates([row], {})[0]['candidate_assessment']
    assert 'splice_safeguard_review' in assessed['unavailable_evidence']
    record = {'assessment':assessed, 'safeguard_status':'passed'}
    assert prediction_note(record)=='Safeguard review is incomplete; the recorded default checks were used.'
    markdown = render_shortlist_markdown({'records':[record]}, 1)
    assert 'biological review of the splice safeguards' in markdown
    assert 'Safeguard review is incomplete' in markdown
    from aso_agents.readable_report import _sections
    from aso_agents.report_orientation import build_orientation
    assert prediction_note(record) in build_orientation(_sections(markdown))['notices']


def test_legacy_request_context_reaches_model(monkeypatch):
    goal = exon_goal()
    goal['splice_strategy_analysis']={'context_payload':{'user_request':'Include the selected exon.'}}
    monkeypatch.setattr(review, '_review_batch', reviewer)
    guards_for(goal)
    assert goal['safeguard_selection_audit']['model_review']['context']['request']=='Include the selected exon.'
