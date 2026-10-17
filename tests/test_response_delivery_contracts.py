import asyncio
import json
from types import SimpleNamespace

import pytest
from aso_agents.workbench_response_agent import evaluation_evidence_audit_agent

from aso_agents import pipeline_tools
from aso_agents.response_contracts import response_contract_errors


def response(text):
    return {'evaluation_assessments': [{'name': 'neutral_probe',
                                      'mechanistic_plausibility': text}]}


def payload():
    return {'task_type': 'evaluate_existing_aso_set', 'evaluation_summary': {
        'provided_asos': [{'name': 'neutral_probe'}],
        'measurement_evidence': {'R001_near_match_pre_mrna_reverse_complement': {
            'name': 'neutral_probe', 'input_row_id': None,
            'field': 'near_match_pre_mrna_reverse_complement',
            'value': {'minimum_recorded_mismatches': 7, 'maximum_recorded_mismatches': 7,
                      'rna_context': 'pre_mrna', 'match_orientation': 'reverse_complement'}}}}}


@pytest.mark.parametrize('text', [
    'The 1/2/3-mismatch bins were not searched.',
    'Searches allowing one, two or three substitutions were not performed.',
    'No three-mismatch screening results are available.',
    'The search tolerance was three mismatches; this does not measure binding.',
    'Retained hypotheses: {{R001_near_match_pre_mrna_reverse_complement}}.',
    # These require a factual verdict from the mandatory evidence audit,
    # rather than acceptance or rejection by grammatical pattern matching.
    'All retained hypotheses have 8-9 mismatches.',
    'The pre-mRNA hit has one mismatch.',
    'The molecule has 0 mismatches.',
    'The alignment has two genomic blocks.',
])
def test_structural_validation_does_not_assign_truth_from_wording(text):
    assert not response_contract_errors(response(text), payload())


def test_unknown_reference_is_still_a_delivery_error():
    assert any('Unknown evaluation measurement reference' in error
               for error in response_contract_errors(response('See {{UNKNOWN}}.'), payload()))


def test_serialization_repair_receives_original_draft(monkeypatch):
    from aso_agents.response_context import decode_response_context
    prompts = []
    invalid = response('Retained evidence: {{UNKNOWN}}.')
    repaired = response('Retained evidence: {{R001_near_match_pre_mrna_reverse_complement}}.')
    invalid['answer_summary'] = repaired['answer_summary'] = 'Activity remains unresolved.'

    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            return SimpleNamespace(final_output=json.dumps({'findings': []}))
        prompts.append(kwargs['input'])
        return SimpleNamespace(final_output=json.dumps(invalid if len(prompts) == 1 else repaired))

    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    result = asyncio.run(pipeline_tools._run_workbench_response_agent_once(payload()))
    assert result.pop('native_consistency_review')['attempts'] == 1
    assert result == repaired and len(prompts) == 2
    wire = json.loads(prompts[1])
    repair = decode_response_context(wire) if wire.get('context_encoding') else wire
    assert repair['preceding_response'] == invalid
    assert 'Unknown evaluation measurement reference' in repair['contract_errors'][0]
