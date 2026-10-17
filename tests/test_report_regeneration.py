import asyncio
from copy import deepcopy
import json

import pytest

from aso_agents import pipeline_tools as p
from aso_agents import report_recovery as recovery
from aso_agents import evaluation_evidence


def fixture_inputs(tmp_path):
    computed = {'rows': [], 'input_row_count': 0, 'limitations': ['Neutral empty fixture.']}
    compact = p._compact_existing_aso_evaluation_for_response(deepcopy(computed))
    source = {'task_type': 'evaluate_existing_aso_set', 'evaluation_summary': compact,
              'request_text': 'Summarize recorded evidence only.'}
    source_path, computed_path = tmp_path/'input.json', tmp_path/'computed.json'
    source_path.write_text(json.dumps(source)); computed_path.write_text(json.dumps(computed))
    return dict(response_input_path=source_path, computed_evaluation_path=computed_path,
                previous_history=tmp_path/'old-history', output_run_dir=tmp_path/'new-output'), source, computed


def test_regeneration_uses_unchanged_evidence_and_fresh_native_components(tmp_path, monkeypatch):
    args, source, computed = fixture_inputs(tmp_path)
    seen = []
    def no_retention(*args, **kwargs):
        raise AssertionError('Previous report drafts must not be read or accepted')
    monkeypatch.setattr(recovery, 'retained_report_drafts', no_retention)
    monkeypatch.setattr(recovery, 'retained_accepted_components', no_retention)
    async def native(payload, **kwargs):
        seen.append((payload, kwargs))
        return {'answer_summary': 'No observations recorded.', 'evaluation_assessments': []}
    monkeypatch.setattr(p, '_run_workbench_response_agent_once', native)
    monkeypatch.setattr(p, '_write_v2_response_outputs', lambda **kwargs: {'captured': kwargs['response_payload']})
    result = asyncio.run(recovery.recover_existing_aso_report(**args, retain_drafts=False))
    assert len(seen) == 1 and seen[0][0] == source
    assert seen[0][1]['initial_drafts'] == {} and seen[0][1]['initial_findings'] == {}
    final = result['artifacts']['captured']
    assert final['evaluation_summary'] == computed
    assert final['report_recovery']['components_regenerated'] is True
    assert final['report_recovery']['scientific_analysis_recomputed'] is False
    assert final['report_recovery']['accepted_scientific_reviews_reused'] is False
    assert final['report_recovery']['retained_drafts_reaudited'] is False
    assert not args['previous_history'].exists()
    assert json.loads(args['response_input_path'].read_text()) == source
    assert json.loads(args['computed_evaluation_path'].read_text()) == computed


def test_regeneration_still_rejects_changed_scientific_evidence(tmp_path):
    args, source, computed = fixture_inputs(tmp_path)
    source['evaluation_summary']['unexpected_new_evidence'] = True
    args['response_input_path'].write_text(json.dumps(source))
    with pytest.raises(ValueError, match='does not reproduce'):
        asyncio.run(recovery.recover_existing_aso_report(**args, retain_drafts=False))
    assert not args['output_run_dir'].exists()


def test_assembly_only_cannot_bypass_retained_review_validation(tmp_path):
    args, _, _ = fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match='requires retained accepted'):
        asyncio.run(recovery.recover_existing_aso_report(**args, assembly_only=True, retain_drafts=False))
    assert not args['output_run_dir'].exists()
