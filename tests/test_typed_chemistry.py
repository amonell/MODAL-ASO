import json
from types import SimpleNamespace

import pytest

from aso_chemistry import encode_idt_oligo, architecture_label, chemistry_components, require_chemistry_length
from aso_agents.chemistry_contracts import compile_chemistry_proposal, chemistry_contract_errors
from aso_agents import pipeline_tools


def proposal(kind='steric_blocker', sugar='2OME', length=22, counts=(0, 0, 0)):
    return {'recommendation': {'architecture': {'kind': kind, 'sugar': sugar, 'backbone': 'PS',
        **dict(zip(('wing_5prime', 'dna_gap', 'wing_3prime'), counts))}, 'length': length,
        'rationale': 'The requested chemical identity is retained; activity is unmeasured.'},
        'unsupported_reason': None}


@pytest.mark.parametrize('sugar,label', [('2MOE', 'steric-hindrance'), ('2OME', 'steric-2ome-ps')])
def test_steric_sugars_remain_distinct_and_have_no_dna_gap(sugar, label):
    compiled = compile_chemistry_proposal(proposal(sugar=sugar))
    assert compiled['recommendation']['CHEMISTRY'] == label
    assert compiled['recommendation']['gap_chemistry'] == 'none'
    assert not chemistry_contract_errors(compiled, 22)
    canonical = pipeline_tools._canonicalize_chemistry_payload(compiled)
    assert canonical['recommendation']['wing_chemistry'] == sugar
    assert pipeline_tools._chemistry_platform(label) == 'steric_blocker'


def test_ome_synthesis_uses_rna_bases_and_all_internucleotide_ps_links():
    # IDT's documented mA/mC/mG/mU notation, not an MOE substitution.
    assert encode_idt_oligo('ACGT', 'steric-2ome-ps') == 'mA*mC*mG*mU'
    assert encode_idt_oligo('ACGT', 'steric-hindrance') == '/52MOErA/*/i2MOErC/*/i2MOErG/*/32MOErT/'


def test_pmo_retains_morpholino_phosphorodiamidate_identity_without_ps_notation():
    raw = proposal('steric_blocker', 'MORPHOLINO', 25)
    raw['recommendation']['architecture']['backbone'] = 'PHOSPHORODIAMIDATE'
    compiled = compile_chemistry_proposal(raw)
    assert compiled['recommendation']['CHEMISTRY'] == 'steric-pmo'
    assert not chemistry_contract_errors(compiled, 25)
    assert chemistry_components('steric-pmo') == {
        'platform': 'steric_blocker', 'wing_chemistry': 'MORPHOLINO',
        'gap_chemistry': 'none', 'backbone': 'PHOSPHORODIAMIDATE',
    }
    assert encode_idt_oligo('ACGT' * 6 + 'A', 'steric-pmo') == ''
    assert pipeline_tools._canonicalize_chemistry_payload(compiled)['recommendation']['backbone'] == 'PHOSPHORODIAMIDATE'
    with pytest.raises(ValueError):
        architecture_label({'kind': 'steric_blocker', 'sugar': 'MORPHOLINO',
            'backbone': 'PS', 'wing_5prime': 0, 'dna_gap': 0, 'wing_3prime': 0})
    with pytest.raises(ValueError):
        architecture_label({'kind': 'gapmer', 'sugar': 'MORPHOLINO',
            'backbone': 'PHOSPHORODIAMIDATE', 'wing_5prime': 5, 'dna_gap': 10,
            'wing_3prime': 5})


def test_gapmer_counts_compile_without_model_authored_profile_or_platform_strings():
    compiled = compile_chemistry_proposal(proposal('gapmer', '2MOE', 19, (4, 11, 4)))
    assert compiled['recommendation']['CHEMISTRY'] == '4-11-4'
    assert compiled['recommendation']['platform'] == 'gapmer'
    assert not chemistry_contract_errors(compiled, 19)
    assert require_chemistry_length('4-11-4', 19).dna_gap == 11
    with pytest.raises(ValueError, match='requires 19'):
        require_chemistry_length('4-11-4', 20)


def test_ome_gapmer_encoder_preserves_the_dna_gap():
    compiled = compile_chemistry_proposal(proposal('gapmer', '2OME', 6, (2, 2, 2)))
    label = compiled['recommendation']['CHEMISTRY']
    assert label == '2ome:2-2-2'
    assert encode_idt_oligo('ACGTAC', label) == 'mA*mC*G*T*mA*mC'
    assert not chemistry_contract_errors(compiled, 6)


@pytest.mark.parametrize('mutation', [dict(sugar='LNA'), dict(backbone='PO'), dict(dna_gap=3)])
def test_unrepresented_or_conflicting_components_are_never_silently_substituted(mutation):
    value = proposal()['recommendation']['architecture']
    value.update(mutation)
    with pytest.raises(ValueError):
        architecture_label(value)
    with pytest.raises(ValueError, match='unsupported_chemistry'):
        compile_chemistry_proposal({'recommendation': None, 'unsupported_reason': 'Requested component unavailable.'})


def test_native_chemistry_selection_compiles_typed_output_before_validation(monkeypatch):
    async def run(agent, *, input, **kwargs):
        if agent.name == 'Chemistry Mechanism Consistency Check':
            return SimpleNamespace(final_output=json.dumps({'issues': []}))
        assert 'output_schema' in json.loads(input)
        return SimpleNamespace(final_output=json.dumps(proposal()))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    from aso_agents.chemistry_agent import chemistry_agent
    selected = pipeline_tools._run_chemistry_agent_for_goal(chemistry_agent, {'intent': 'binding', 'aso_length_nt': 22})
    assert selected['recommendation']['CHEMISTRY'] == 'steric-2ome-ps'
    assert selected['mechanism_review']['independent_validation'] is False


def test_chemistry_nonissue_can_be_dismissed_without_rewriting_the_proposal(monkeypatch):
    calls = []
    raw = proposal(sugar='2MOE')
    rationale = raw['recommendation']['rationale']
    async def run(agent, *, input, **kwargs):
        calls.append(agent.name)
        if agent.name == 'Chemistry Mechanism Consistency Check':
            return SimpleNamespace(final_output=json.dumps({'issues': [{'claim': rationale,
                'reason': 'The claim is consistent with the request; no contradiction was identified.'}]}))
        if agent.name == 'Chemistry Finding Adjudicator':
            return SimpleNamespace(final_output=json.dumps({'decisions': [{'finding_index': 0,
                'decision': 'dismiss', 'reason': 'The cited observation identifies no contradiction.'}]}))
        return SimpleNamespace(final_output=json.dumps(raw))
    monkeypatch.setattr(pipeline_tools.Runner, 'run', run)
    from aso_agents.chemistry_agent import chemistry_agent
    selected = pipeline_tools._run_chemistry_agent_for_goal(chemistry_agent, {'intent': 'binding', 'aso_length_nt': 22})
    assert calls.count('ASO Chemistry Agent') == 1
    assert selected['recommendation']['rationale'] == rationale
    assert selected['mechanism_review']['finding_adjudication']['decisions'][0]['decision'] == 'dismiss'
