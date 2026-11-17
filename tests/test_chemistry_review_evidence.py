"""Review the same chemical identity across typed and compiled representations."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_chemistry import (MIN_NATIVE_GAPMER_DNA_GAP_NT, encode_idt_oligo,
                           native_chemistry_contract, require_modality_chemistry)
from aso_agents import pipeline_tools as pipeline
from aso_agents.chemistry_agent import chemistry_agent
from aso_agents.chemistry_contracts import ChemistryResolutionError, compile_chemistry_proposal


def proposal(kind, sugar, explicit=False):
    pmo = sugar == 'MORPHOLINO'
    return {'recommendation': {
        'architecture': {'kind': kind, 'sugar': sugar,
                         'backbone': 'PHOSPHORODIAMIDATE' if pmo else 'PS',
                         'wing_5prime': 5 if kind == 'gapmer' else 0,
                         'dna_gap': 10 if kind == 'gapmer' else 0,
                         'wing_3prime': 5 if kind == 'gapmer' else 0,
                         'linkages': ['PS'] * 19 if explicit else None},
        'length': 20, 'cytosine_methylation': 'profile_default',
        'stereochemistry': None, 'terminal_modifications': [], 'conjugates': [],
        'rationale': ('Linkages=null specifies the phosphorodiamidate backbone.' if pmo else
                      'Linkages are supplied explicitly as PS.' if explicit else
                      'Linkages=null uses the full-PS representation. The DNA-gap support rule is platform-specific.')},
        'unsupported_reason': None}


@pytest.mark.parametrize('kind,sugar,explicit', [
    ('gapmer', '2MOE', False), ('gapmer', '2MOE', True),
    ('gapmer', '2OME', False), ('gapmer', '2OME', True),
    ('steric_blocker', '2MOE', False), ('steric_blocker', '2OME', True),
    ('steric_blocker', 'MORPHOLINO', False),
])
def test_review_and_adjudication_retain_original_and_compiled_identity(monkeypatch, kind, sugar, explicit):
    selected = proposal(kind, sugar, explicit)
    original = deepcopy(selected)
    calls = []

    async def run(agent, *, input, **_kwargs):
        body = json.loads(input)
        calls.append(agent.name)
        assert body['native_chemistry_contract'] == native_chemistry_contract()
        if agent.name == 'ASO Chemistry Agent':
            return SimpleNamespace(final_output=json.dumps(selected))
        assert body['selection_proposal'] == original
        links = body['chemistry']['recommendation']['linkage_specification']
        assert links == (None if sugar == 'MORPHOLINO' else ['PS'] * 19)
        if agent.name == 'Chemistry Mechanism Consistency Check':
            return SimpleNamespace(final_output=json.dumps({'issues': [{
                'claim': selected['recommendation']['rationale'],
                'reason': 'The serialization differs between the proposal and compiled record.'}]}))
        assert agent.name == 'Chemistry Finding Adjudicator'
        return SimpleNamespace(final_output=json.dumps({'decisions': [{
            'finding_index': 0, 'decision': 'dismiss',
            'reason': 'The supplied representation contract and original proposal establish the same physical backbone.'}]}))

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    goal = {'intent': 'knockdown' if kind == 'gapmer' else 'binding',
            'modality': 'gapmer' if kind == 'gapmer' else 'steric', 'aso_length_nt': 20}
    result = pipeline._run_chemistry_agent_for_goal(chemistry_agent, goal)
    expected = pipeline._canonicalize_chemistry_payload(compile_chemistry_proposal(original))
    assert result['recommendation'] == expected['recommendation']
    assert result['mechanism_review']['review_evidence']['selection_proposal'] == original
    assert len(calls) == 3
    assert selected == original
    if sugar != 'MORPHOLINO':
        rec = result['recommendation']
        assert encode_idt_oligo('ACGT' * 5, rec['CHEMISTRY'], linkages=rec['linkage_specification']).count('*') == 19


@pytest.mark.parametrize('prefix', ['', '2ome:'])
def test_reviewed_native_minimum_is_the_enforced_support_boundary(prefix):
    minimum = native_chemistry_contract()['native_gapmer_minimum_dna_gap_nt']
    assert minimum == MIN_NATIVE_GAPMER_DNA_GAP_NT
    require_modality_chemistry('gapmer', f'{prefix}5-{minimum}-5')
    with pytest.raises(ValueError, match='supported DNA gap'):
        require_modality_chemistry('gapmer', f'{prefix}5-{minimum - 1}-5')


def test_real_mechanism_issue_still_reaches_bounded_repair_with_both_representations(monkeypatch):
    selected = proposal('gapmer', '2MOE')
    selected['recommendation']['rationale'] = 'Gapmer chemistry guarantees isoform-selective knockdown.'
    selections = []

    async def run(agent, *, input, **_kwargs):
        body = json.loads(input)
        if agent.name == 'ASO Chemistry Agent':
            if selections:
                feedback = body['output_contract_feedback']
                assert json.loads(feedback['previous_response']) == selected
                assert feedback['compiled_recommendation']['recommendation']['linkage_specification'] == ['PS'] * 19
                assert 'selectivity' in ' '.join(feedback['validation_errors'])
            selections.append(body)
            return SimpleNamespace(final_output=json.dumps(selected))
        assert body['selection_proposal'] == selected
        if agent.name == 'Chemistry Mechanism Consistency Check':
            return SimpleNamespace(final_output=json.dumps({'issues': [{
                'claim': selected['recommendation']['rationale'],
                'reason': 'The chemistry alone does not establish isoform selectivity.'}]}))
        return SimpleNamespace(final_output=json.dumps({'decisions': [{
            'finding_index': 0, 'decision': 'uphold', 'reason': 'Target-specific binding evidence is required.'}]}))

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    with pytest.raises(ChemistryResolutionError, match='chemistry_contract_unresolved'):
        pipeline._run_chemistry_agent_for_goal(chemistry_agent, {'modality': 'gapmer', 'intent': 'knockdown'})
    assert len(selections) == 3


def test_invalid_physical_identity_never_reaches_semantic_review(monkeypatch):
    selected = proposal('gapmer', '2MOE', explicit=True)
    selected['recommendation']['architecture']['linkages'][-1] = 'PO'
    calls = []

    async def run(agent, **_kwargs):
        calls.append(agent.name)
        assert agent.name == 'ASO Chemistry Agent'
        return SimpleNamespace(final_output=json.dumps(selected))

    monkeypatch.setattr(pipeline.Runner, 'run', run)
    with pytest.raises(ChemistryResolutionError, match='Mixed PS/PO'):
        pipeline._run_chemistry_agent_for_goal(chemistry_agent, {'modality': 'gapmer'})
    assert len(calls) == 3
