import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools as p
from aso_agents.response_artifacts import ResponseArtifacts, report_configuration
from aso_agents.evaluation_response_audit import apply_bound_narrative_patch
from aso_agents.evaluation_evidence import row_evidence_inventory
from aso_agents.alignment_identity import recorded_window_identity
from aso_agents.workbench_response_agent import (workbench_response_agent,
    evaluation_evidence_audit_agent, panel_comparison_audit_agent)
from test_evaluation_response_audit import source_and_response, finding


def test_patch_cannot_mutate_an_unflagged_path_or_input():
    source, response = source_and_response()
    before = deepcopy((source, response))
    key = finding()['response_path']
    good = {'corrections': {key: 'A retained antisense hypothesis is present; activity remains unmeasured.'}}
    result = apply_bound_narrative_patch(response, good, [finding()], source)
    assert result['evaluation_assessments'][0]['uncertainty'] == response['evaluation_assessments'][0]['uncertainty']
    for bad in ({'corrections': {}}, {'corrections': {key: None}},
                {'corrections': {key: good['corrections'][key], '/evaluation_assessments/0/input_row_id': 'different'}},
                {'corrections': good['corrections'], 'answer_summary': 'rewritten'}):
        with pytest.raises(ValueError):
            apply_bound_narrative_patch(response, bad, [finding()], source)
    assert (source, response) == before


def test_complete_drafts_and_components_survive_failures_and_invalidate_on_any_input(tmp_path):
    source, response = source_and_response()
    config = {'model': 'selected', 'effort': 'max', 'implementation': 'sha-A'}
    archive = ResponseArtifacts(tmp_path, source, config)
    draft = {'text': 'a' * 17000, 'ending': 'not truncated'}
    archive.record('draft', draft)
    archive.accept(response)
    assert json.loads((archive.attempt / 'draft.json').read_text()) == draft
    repeated = ResponseArtifacts(tmp_path, source, config)
    assert repeated.attempt != archive.attempt and repeated.accepted() == response
    changed = deepcopy(source); changed['request_text'] = 'New request'
    variants = [(changed, config), (source, {**config, 'effort': 'low'}),
                (source, {**config, 'model': 'different'}), (source, {**config, 'implementation': 'sha-B'})]
    for payload, settings in variants:
        assert ResponseArtifacts(tmp_path, payload, settings).accepted() is None
    saved = archive.root / 'accepted.json'
    altered = json.loads(saved.read_text()); altered['response']['answer_summary'] = 'Tampered'
    saved.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='provenance'):
        repeated.accepted()


def test_configuration_never_serializes_credentials(monkeypatch):
    from aso_agents import llm_provider
    from dataclasses import replace
    # Construct a real config shape; the secret is a sentinel, never a real key.
    config = llm_provider.ProviderConfig(name='openrouter', api_key='secret-sentinel',
        default_headers={'Authorization': 'header-sentinel'}, base_url='https://example.invalid',
        continuation_mode='stateless', provider_preferences={}, model_override='neutral/model')
    monkeypatch.setattr(llm_provider, 'get_provider_config', lambda: config)
    settings = report_configuration([workbench_response_agent])
    wire = json.dumps(settings)
    assert 'secret-sentinel' not in wire and 'header-sentinel' not in wire
    monkeypatch.setattr(llm_provider, 'get_provider_config', lambda: replace(config, reasoning_effort_override='max'))
    assert report_configuration([workbench_response_agent]) != settings


def test_native_completed_component_is_reused_without_recalling_model(monkeypatch, tmp_path):
    from aso_agents import response_artifacts
    source, response = source_and_response(); count = []
    monkeypatch.setattr(response_artifacts, 'report_configuration', lambda agents: {'model': 'neutral', 'effort': 'max'})
    async def run(agent, **kwargs):
        count.append(agent.name)
        value = {'findings': []} if agent.name == evaluation_evidence_audit_agent.name else response
        return SimpleNamespace(final_output=json.dumps(value))
    monkeypatch.setattr(p.Runner, 'run', run)
    first = asyncio.run(p._run_workbench_response_batch_once(source, artifact_dir=tmp_path))
    second = asyncio.run(p._run_workbench_response_batch_once(source, artifact_dir=tmp_path))
    assert first == second and len(count) == 2
    assert list(tmp_path.rglob('round-1-audit-call-1.json'))
    changed = deepcopy(source); changed['request_text'] = 'A changed request'
    asyncio.run(p._run_workbench_response_batch_once(changed, artifact_dir=tmp_path))
    assert len(count) == 4


def test_failed_report_records_terminal_state_and_cannot_reenter_analysis(monkeypatch, tmp_path):
    state = {'status': 'request_parsed_ready', 'phase': 'request_parsed_ready',
             'request': 'Evaluate the supplied observation.', 'workflow_name': 'evaluate_existing_aso_set',
             'current_task_id': 't1', 'run_dir': str(tmp_path),
             'request_envelope': {'schema_version': '3', 'tasks': [
                 {'task_id': 't1', 'task_type': 'evaluate_existing_aso_set'}]},
             'task_statuses': {'t1': {'status': 'in_progress'}}}
    state_path = str(tmp_path / 'state.json')
    prepared = {'status': 'ok', 'state': state, 'state_path': state_path, 'run_dir': str(tmp_path),
                'task_id': 't1', 'task_request_envelope': {}, 'task_context': {}}
    real_prepare = p._prepare_v2_task_execution
    monkeypatch.setattr(p, '_prepare_v2_task_execution', lambda **kwargs: prepared)
    monkeypatch.setattr(p, '_task_scoped_response_metadata', lambda **kwargs: {'request_envelope': {}})
    analysis = []
    payload = {'provided_asos': []}
    monkeypatch.setattr(p, '_evaluate_existing_asos_payload', lambda **kwargs: analysis.append(True) or payload)
    monkeypatch.setattr(p, '_should_finalize_current_task', lambda *args: True)
    def fail(*args, **kwargs):
        raise RuntimeError('workbench_response_contract_failed: neutral evidence conflict')
    monkeypatch.setattr(p, '_run_workbench_response_sync', fail)
    snapshots = []
    monkeypatch.setattr(p, '_save_state', lambda path, value, **kwargs: snapshots.append(deepcopy(value)))
    result = p.tool_evaluate_existing_aso_set(state_path)
    assert result['status'] == 'fatal_error' and len(analysis) == 1
    assert snapshots[-1]['status'] == 'failed' and snapshots[-1]['last_error']['retryable'] is False
    assert state['task_statuses']['t1']['status'] == 'failed'
    assert json.loads(Path(result['computed_evaluation_json']).read_text()) == payload
    monkeypatch.setattr(p, '_prepare_v2_task_execution', real_prepare)
    monkeypatch.setattr(p, '_load_state', lambda path: deepcopy(state))
    blocked = p.tool_evaluate_existing_aso_set(state_path)
    assert blocked['status'] == 'fatal_error' and len(analysis) == 1
    planning = p._derive_next_actions(state)
    assert planning['terminal_status'] == 'fatal'
    assert not any(action.get('tool') == 'tool_evaluate_existing_aso_set' for action in planning['next_actions'])


@pytest.mark.parametrize('strand', [1, -1, '1', '-1', '+', '-'])
def test_exact_mapping_producer_normalizes_native_strand_and_establishes_identity(monkeypatch, strand):
    import run_goal_intake as runtime
    sign = '-' if str(strand) in ('-', '-1') else '+'
    transcript = {'transcript_id': 'TX_NEUTRAL.2', 'strand': strand, 'chrom': 'chr7',
                  'exons_tx': [{'index': 1, 'start': 101, 'end': 108, 'cdna_start': 1, 'cdna_end': 8}]}
    monkeypatch.setattr(p, '_mapping_conservation_summary', lambda **kwargs: {})
    hits = p._build_sequence_mapping_hits(entry={'sequence': 'ACGT'}, isoform_transcripts={'test': transcript},
        cdna_map={'test': 'ACGTAAAA'}, runtime=runtime, species='human', preferred_transcript_id='TX_NEUTRAL.2', genome=None)
    assert len(hits) == 1 and hits[0]['strand'] == sign
    hits[0]['rna_context'] = 'mature_spliced_cdna'
    identity = recorded_window_identity(hits[0], 'exact_mapping')
    assert identity is not None and identity['strand'] == sign
    assert identity['ordered_genomic_blocks_1based_inclusive'] == [
        {'chrom': '7', 'start': 105 if sign == '-' else 101, 'end': 108 if sign == '-' else 104}]


def test_distinct_junction_windows_without_spans_are_never_deduplicated(monkeypatch):
    runtime = SimpleNamespace(_map_cdna_range_to_genomic_segments=lambda start,end,entry:
        [('7',start,start+1), ('7',start+100,start+101)], _continuous_genomic_span_from_segments=lambda segments: None)
    hits = p._build_sequence_mapping_hits(entry={'sequence': 'ACGT'}, isoform_transcripts={'a': {'transcript_id': 'X', 'strand': '+'}},
        cdna_map={'a': 'ACGTACGT'}, runtime=runtime, species='human', preferred_transcript_id='X', genome=None)
    assert [hit['cdna_start'] for hit in hits] == [1,5]


def test_resolved_membership_geometry_remains_available_without_site_register():
    row = {'requested_feature_membership': [{'requested_feature_identity_resolved': True,
        'chrom': '7', 'start': 101, 'end': 108, 'coordinate_system': '1based_inclusive',
        'identity_resolution': {'status': 'resolved_reference_ordinal', 'biological_label_verified': False},
        'observations': [{'match_orientation': 'input_sequence', 'overlap_bp': 4}]}]}
    availability = row_evidence_inventory(row)['feature_evidence_availability']
    assert availability['requested_site_register']['status'] == 'not_recorded'
    feature = availability['requested_feature_membership']['features'][0]
    assert feature['recorded_membership_observation_count'] == 1 and feature['requested_feature_identity_resolved']
    assert feature['identity_resolution']['biological_label_verified'] is False


def test_one_failed_component_does_not_discard_accepted_neighbors(monkeypatch, tmp_path):
    from aso_agents import response_artifacts, response_context
    from aso_agents.response_assembly import assembly_audit_agent
    from test_evaluation_response_batches import panel, answer
    from collections import Counter
    payload = panel(count=2)
    counts = Counter(); failing = [True]
    monkeypatch.setattr(response_artifacts, 'report_configuration', lambda agents: {'model': 'neutral'})
    async def run(agent, **kwargs):
        wire = json.loads(kwargs['input'])
        decoded = response_context.decode_response_context(wire) if wire.get('context_encoding') else wire
        if agent.name in {evaluation_evidence_audit_agent.name, panel_comparison_audit_agent.name,
                          assembly_audit_agent.name}:
            value = {'findings': []}
        else:
            source = decoded.get('source', decoded)
            row = source['evaluation_summary']['provided_asos'][0]
            counts[row['input_row_id']] += 1
            value = {'error': 'simulated_component_failure'} if failing[0] and row['input_row_id'] == 'row_1' else answer(row)
        return SimpleNamespace(final_output=json.dumps(value))
    monkeypatch.setattr(p.Runner, 'run', run)
    with pytest.raises(RuntimeError, match='component_failed'):
        asyncio.run(p._run_workbench_response_agent_once(payload, artifact_dir=tmp_path))
    assert counts == {'row_0': 1, 'row_1': 3}
    assert len(list(tmp_path.rglob('accepted.json'))) == 1
    assert len(list(tmp_path.rglob('failure.json'))) == 1
    failing[0] = False
    result = asyncio.run(p._run_workbench_response_agent_once(payload, artifact_dir=tmp_path))
    assert counts == {'row_0': 1, 'row_1': 4}
    assert [row['input_row_id'] for row in result['evaluation_assessments']] == ['row_0', 'row_1']


def test_distinct_narrative_fields_do_not_exhaust_each_others_repairs(monkeypatch):
    from aso_agents.workbench_response_agent import evaluation_evidence_repair_agent
    from test_evaluation_response_audit import decode
    source, original = source_and_response()
    fields = ['selectivity_limitations', 'uncertainty', 'mechanistic_plausibility', 'ranking_rationale']
    audited, patches = [], []
    async def run(agent, **kwargs):
        context = decode(kwargs['input'])
        if agent.name == evaluation_evidence_audit_agent.name:
            draft = context['draft_response']; audited.append(deepcopy(draft))
            pending = [field for field in fields if draft['evaluation_assessments'][0][field] != 'Activity remains unmeasured.']
            answer = {'findings': []}
            if pending:
                field = pending[0]
                answer['findings'] = [{**finding(), 'response_path': '/evaluation_assessments/0/'+field,
                    'quote': draft['evaluation_assessments'][0][field]}]
        elif agent.name == evaluation_evidence_repair_agent.name:
            paths = context['allowed_narrative_repair_paths']; patches.append(paths)
            answer = {'corrections': {path:'Activity remains unmeasured.' for path in paths}}
        else:
            answer = original
        return SimpleNamespace(final_output=json.dumps(answer))
    monkeypatch.setattr(p.Runner, 'run', run)
    result = asyncio.run(p._run_workbench_response_batch_once(source))
    assert len(patches) == 4 and len(audited) == 5
    assert result['native_consistency_review']['attempts'] == 5
    assert result['answer_summary'] == original['answer_summary']
    assert all(len(paths)==1 for paths in patches)


def test_no_progress_on_a_field_is_withheld_and_reaudited_without_full_recomposition(monkeypatch):
    from aso_agents.workbench_response_agent import evaluation_evidence_repair_agent
    from test_evaluation_response_audit import decode
    source, original = source_and_response(); counts = {}
    async def run(agent, **kwargs):
        context = decode(kwargs['input'])
        counts[agent.name] = counts.get(agent.name,0)+1
        if agent.name == evaluation_evidence_audit_agent.name:
            value = {'findings': [] if 'Assessment withheld for this narrative field'
                     in context['draft_response']['evaluation_assessments'][0]['selectivity_limitations']
                     else [finding()]}
        elif agent.name == evaluation_evidence_repair_agent.name:
            value = {'corrections': {finding()['response_path']: original['evaluation_assessments'][0]['selectivity_limitations']}}
        else:value = original
        return SimpleNamespace(final_output=json.dumps(value))
    monkeypatch.setattr(p.Runner, 'run', run)
    result = asyncio.run(p._run_workbench_response_batch_once(source))
    assert counts[workbench_response_agent.name] == 1
    assert counts[evaluation_evidence_repair_agent.name] == 3
    assert counts[evaluation_evidence_audit_agent.name] == 2
    assert 'Assessment withheld for this narrative field' in result['evaluation_assessments'][0]['selectivity_limitations']


def test_retained_draft_is_reaudited_without_regenerating_or_trusting_old_review(monkeypatch, tmp_path):
    from aso_agents import response_artifacts
    from test_evaluation_response_audit import decode
    source, response = source_and_response(); response['native_consistency_review']={'status':'old'}
    response.pop('native_consistency_review')
    seen=[]
    monkeypatch.setattr(response_artifacts,'report_configuration',lambda agents:{'model':'neutral'})
    async def run(agent, **kwargs):
        assert agent.name == evaluation_evidence_audit_agent.name
        seen.append(decode(kwargs['input'])['draft_response'])
        return SimpleNamespace(final_output=json.dumps({'findings':[]}))
    monkeypatch.setattr(p.Runner,'run',run)
    result=asyncio.run(p._run_workbench_response_batch_once(source,artifact_dir=tmp_path,initial_response=response))
    assert seen == [response]
    assert result['native_consistency_review']['status']=='no_factual_contradiction_flagged'


def test_retained_draft_selection_requires_exact_evidence_model_and_instructions(tmp_path):
    from aso_agents.report_recovery import retained_report_drafts
    from aso_agents.response_artifacts import digest
    source,response=source_and_response()
    config={'provider':{'name':'test','effort':'max'},'agents':[{'instructions':'original'}], 'implementation':'v1'}
    archive=ResponseArtifacts(tmp_path,source,config);archive.accept(response)
    current={**config,'implementation':'v2'}
    drafts=retained_report_drafts(source,tmp_path,current)
    assert drafts[digest(source)]['draft']==response
    changed=deepcopy(source);changed['request_text']='changed'
    with pytest.raises(ValueError,match='evidence-matched'):
        retained_report_drafts(changed,tmp_path,current)
    for changed in ({**current,'provider':{'name':'test','effort':'none'}},{**current,'agents':[{'instructions':'different'}]}):
        with pytest.raises(ValueError,match='model or instructions'):
            retained_report_drafts(source,tmp_path,changed)


@pytest.mark.parametrize('change', ['assembly', 'recovery', 'science', 'agent', 'provider',
                                  'source', 'tampered_response', 'missing_review', 'wrong_review'])
def test_assembly_only_recovery_reuses_only_attested_unchanged_scientific_components(tmp_path, change):
    from aso_agents.report_recovery import retained_accepted_components
    from aso_agents.response_context import evaluation_response_batches
    from aso_agents.response_artifacts import digest
    from aso_agents.evaluation_response_audit import response_audit_receipt
    from test_evaluation_response_batches import panel, answer
    payload = panel(count=2)
    config = {'provider': {'model': 'neutral', 'effort': 'max'}, 'agents': ['unchanged'],
        'implementation': {'pipeline_tools.py': 'science-v1', 'response_context.py': 'partition-v1',
                           'response_assembly.py': 'assembly-v1', 'report_recovery.py': 'recovery-v1'}}
    archives = []
    for batch in evaluation_response_batches(payload):
        response = answer(batch['evaluation_summary']['provided_asos'][0])
        response['native_consistency_review'] = response_audit_receipt(response, batch, attempts=1, model='neutral')
        if change == 'missing_review':
            response.pop('native_consistency_review')
        elif change == 'wrong_review':
            response['native_consistency_review']['source_sha256'] = 'wrong'
        archive = ResponseArtifacts(tmp_path, batch, config); archive.accept(response); archives.append(archive)
    current = deepcopy(config)
    if change in ('assembly', 'recovery', 'science'):
        name = {'assembly': 'response_assembly.py', 'recovery': 'report_recovery.py', 'science': 'pipeline_tools.py'}[change]
        current['implementation'][name] = 'changed'
    elif change == 'agent': current['agents'] = ['different']
    elif change == 'provider': current['provider']['effort'] = 'none'
    elif change == 'source': payload['request_text'] = 'A different request'
    elif change == 'tampered_response':
        path = archives[0].root / 'accepted.json'; saved = json.loads(path.read_text())
        saved['response']['answer_summary'] = 'Tampered'; path.write_text(json.dumps(saved))
    if change in ('assembly', 'recovery'):
        result = retained_accepted_components(payload, tmp_path, current)
        assert len(result) == 2
        for batch in evaluation_response_batches(payload):
            assert result[digest(batch)]['response']['native_consistency_review']['source_sha256'] == digest(batch)
    else:
        with pytest.raises(ValueError):
            retained_accepted_components(payload, tmp_path, current)


def test_recovery_repairs_known_unresolved_findings_before_any_new_audit(monkeypatch):
    from aso_agents.workbench_response_agent import evaluation_evidence_repair_agent
    from test_evaluation_response_audit import decode
    source,original=source_and_response();calls=[]
    corrected='A retained antisense hypothesis exists; activity is unmeasured.'
    async def run(agent,**kwargs):
        calls.append(agent.name);context=decode(kwargs['input'])
        if agent.name==evaluation_evidence_repair_agent.name:
            assert context['evidence_consistency_findings'][0]['quote']==finding()['quote']
            value={'corrections':{finding()['response_path']:corrected}}
        else:
            assert agent.name==evaluation_evidence_audit_agent.name
            assert context['draft_response']['evaluation_assessments'][0]['selectivity_limitations']==corrected
            value={'findings':[]}
        return SimpleNamespace(final_output=json.dumps(value))
    monkeypatch.setattr(p.Runner,'run',run)
    result=asyncio.run(p._run_workbench_response_batch_once(source,initial_response=original,initial_findings=[finding()]))
    assert calls==[evaluation_evidence_repair_agent.name,evaluation_evidence_audit_agent.name]
    assert result['evaluation_assessments'][0]['selectivity_limitations']==corrected


def test_saved_failure_findings_are_not_lost_during_draft_selection(tmp_path):
    from aso_agents.report_recovery import retained_report_drafts
    from aso_agents.response_artifacts import digest
    source,response=source_and_response();config={'provider':{},'agents':[]}
    archive=ResponseArtifacts(tmp_path,source,config)
    archive.record('round-4-audited-draft',response);archive.record('round-4-audit',{'findings':[finding()]})
    archive.record('failure',{'message':'repair exhausted'})
    selected=retained_report_drafts(source,tmp_path,config)[digest(source)]
    assert selected['unresolved_findings']==[finding()]
    assert selected['draft']==response
