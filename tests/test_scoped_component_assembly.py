import asyncio
from copy import deepcopy
import json

import pytest

from aso_agents import response_context
from aso_agents.response_artifacts import digest
from aso_agents.response_assembly import review_assembled_response, scoped_component_response


def reviewed_fixture(monkeypatch):
    original_batches = response_context.evaluation_response_batches
    monkeypatch.setattr(response_context, 'evaluation_response_batches',
        lambda payload: original_batches(payload, target_context_bytes=1))
    rows = [{'input_row_id': str(index), 'name': 'same [label] <name>',
             'sequence': sequence, 'mapping_hits': [], 'near_match_candidates': []}
            for index, sequence in enumerate(['ACGT', 'TGCA'], 1)]
    payload = {'task_type': 'evaluate_existing_aso_set', 'request_text': 'Assess both synthetic observations.',
               'evaluation_summary': {'provided_asos': rows, 'provided_aso_count': 2}}
    parts = []
    for batch in response_context.evaluation_response_batches(payload):
        row = batch['evaluation_summary']['provided_asos'][0]
        part = {'answer_summary': 'This assessment covers one observation; activity is unmeasured.',
                'markdown': 'The target search and exact gene screen have separate scopes.',
                'assumptions': ['Only the supplied reference contexts are available here.'],
                'limitations': ['No near matches are retained for this observation.'],
                'suggested_followups': ['Measure activity for this observation.'],
                'opaque_original_field': {'retained': [True, None]},
                'evidence_refs': [],
                'evaluation_assessments': [{key: row[key] for key in ['input_row_id', 'name']}]}
        part['evaluation_assessments'][0].update(mechanistic_plausibility='Unmeasured.',
            uncertainty='No experimental data.', ranking_rationale='No potency order is established.')
        part['native_consistency_review'] = {'status': 'no_factual_contradiction_flagged',
            'source_sha256': digest(batch), 'draft_sha256': digest(part)}
        parts.append(part)
    return payload, response_context.merge_evaluation_responses(parts, payload)


def test_labels_preserve_all_sentences_fields_and_observation_scope(monkeypatch):
    payload, response = reviewed_fixture(monkeypatch)
    before = deepcopy((payload, response))
    result = scoped_component_response(payload, response)
    assert (payload, response) == before
    assert result['evaluation_assessments'] == response['evaluation_assessments']
    assert result['original_component_responses'] == response['original_component_responses']
    assert 'all 2 supplied observations' in result['answer_summary']
    for index, part in enumerate(response['original_component_responses'], 1):
        label = f'Observation {index} (same \\[label\\] &lt;name&gt;)'
        for field in ['answer_summary', 'markdown']:
            assert f'### {label}\n\n' + part[field] in result[field]
        for field in ['assumptions', 'limitations', 'suggested_followups']:
            assert f'**{label}:** ' + part[field][0] in result[field]
            assert len(result[field]) == 2  # Same words retain both original scopes.


@pytest.mark.parametrize('fault', ['receipt_missing', 'draft_changed', 'source_changed',
    'receipt_status', 'duplicate_component', 'missing_component', 'assembled_science_changed'])
def test_incomplete_or_mismatched_scientific_attestation_fails_closed(monkeypatch, fault):
    payload, response = reviewed_fixture(monkeypatch)
    first = response['original_component_responses'][0]
    if fault == 'receipt_missing': first.pop('native_consistency_review')
    if fault == 'draft_changed': first['answer_summary'] = 'A new, unaudited conclusion.'
    if fault == 'source_changed': payload['evaluation_summary']['provided_asos'][0]['sequence'] = 'AAAA'
    if fault == 'receipt_status': first['native_consistency_review']['status'] = 'pending'
    if fault == 'duplicate_component': response['original_component_responses'][1] = deepcopy(first)
    if fault == 'missing_component': response['original_component_responses'].pop()
    if fault == 'assembled_science_changed': response['evaluation_assessments'][0]['ranking_rationale'] = 'Invented efficacy.'
    with pytest.raises(ValueError):
        scoped_component_response(payload, response)


def test_component_order_does_not_change_scopes_or_assessment_order(monkeypatch):
    payload, response = reviewed_fixture(monkeypatch)
    expected = scoped_component_response(payload, response)
    response['original_component_responses'].reverse()
    result = scoped_component_response(payload, response)
    assert result['original_component_responses'] == response['original_component_responses']
    for field in ['answer_summary', 'markdown', 'limitations', 'evaluation_assessments']:
        assert result[field] == expected[field]


def test_unreviewed_components_do_not_take_the_scoped_acceptance_path(monkeypatch):
    payload, response = reviewed_fixture(monkeypatch)
    for part in response['original_component_responses']:
        part.pop('native_consistency_review')
    assert scoped_component_response(payload, response) is None


def test_archived_scoped_assembly_never_recalls_or_claims_a_native_review(monkeypatch, tmp_path):
    from aso_agents import pipeline_tools as p
    from aso_agents.report_recovery import retained_assembled_report
    payload, response = reviewed_fixture(monkeypatch)
    async def unexpected(**kwargs):
        raise AssertionError('No additional scientific or assembly model call is permitted.')
    monkeypatch.setattr(p, '_run_schema_validated_agent_once', unexpected)
    result = asyncio.run(review_assembled_response(payload, response, artifact_dir=tmp_path/'assembly'))
    receipt = result['native_assembly_review']
    assert receipt['status'] == 'component_scopes_preserved'
    assert receipt['native_model_calls'] == 0 and receipt['scientific_component_count'] == 2
    assert receipt['accepted_draft_sha256'] == digest({k: v for k, v in result.items() if k != 'native_assembly_review'})
    assert asyncio.run(review_assembled_response(payload, response, artifact_dir=tmp_path/'assembly')) == result
    assert retained_assembled_report(response, tmp_path) is None
    saved = list((tmp_path/'assembly').glob('components/*/accepted.json'))
    assert len(saved) == 1 and json.loads(saved[0].read_text())['response'] == result
