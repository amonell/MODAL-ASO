from copy import deepcopy

import pytest

from aso_agents.splice_plan_report import render_splice_plan


def fixture():
    context = {"isoform_objectives": [{"isoform_label": "REF.4", "desired_abundance": "increase"}],
               "preserved_isoforms": ["OTHER.2"],
               "steric_site_candidates": [{"canonical_label": "donor@812", "site_kind": "donor",
                   "exon_indices": {"REF.4": 3}, "uniqueness": "unique",
                   "reference_scope": {"scope": "selected_design_transcripts",
                       "compared_transcript_ids": ["REF.4"], "present_in": ["REF.4"],
                       "unresolved_transcript_ids": []}}]}
    priorities = [{"rank": 1, "bundle": [{"label": "donor@812", "desired_direction": "increase"}],
                   "confidence": "high", "primary_effect": "Proven increase by masking donor",
                   "rationale": "Unique to this transcript across the transcriptome",
                   "safeguards": [{"label": "acceptor@721", "guard": "neutral"}]}]
    return context, priorities


def test_plan_reports_actual_reference_scope_without_promoting_model_claims():
    context, priorities = fixture()
    before = deepcopy((context, priorities))
    report = render_splice_plan(context, priorities)
    assert "donor@812 (increase)" in report
    assert "1 matching transcript(s) among 1 inspected" in report
    assert "outside-catalog specificity remains unknown" in report
    assert "Preserve: OTHER.2" in report and "acceptor@721 (neutral)" in report
    assert "Annotation exon ordinals: REF.4: 3" in report
    assert "Proven increase" not in report and "Unique to this transcript" not in report
    assert "does not establish that directly masking" in report
    assert (context, priorities) == before


def test_shared_and_unresolved_catalog_evidence_remain_visible():
    context, priorities = fixture()
    scope = context['steric_site_candidates'][0]['reference_scope']
    scope.update(compared_transcript_ids=['REF.4', 'OTHER.2'],
                 present_in=['REF.4', 'OTHER.2'], unresolved_transcript_ids=['UNKNOWN.1'])
    report = render_splice_plan(context, priorities)
    assert '2 matching transcript(s) among 2 inspected' in report
    assert 'Matching references: REF.4, OTHER.2' in report
    assert 'Unresolved references: UNKNOWN.1' in report


@pytest.mark.parametrize('field,value', [('label','unknown@1'), ('desired_direction','proven')])
def test_unbound_model_priority_is_not_a_delivered_reference_claim(field, value):
    context, priorities = fixture()
    priorities[0]['bundle'][0][field] = value
    with pytest.raises(ValueError, match='Unresolved splice priority'):
        render_splice_plan(context, priorities)


def test_native_strategy_keeps_identifier_and_ranker_and_uses_typed_final_rendering(monkeypatch):
    import aso_agents.splice_strategy_agent as strategy
    context, priorities = fixture()
    calls = []
    def invoke(agent, payload, **kwargs):
        calls.append(agent.name)
        if agent is strategy._target_identifier_agent:
            return {'targets':[{'label':'donor@812','desired_direction':'increase'}]}
        return {'ranked_targets': priorities}
    monkeypatch.setattr(strategy, '_call_agent_json', invoke)
    report, trace = strategy.generate_splice_strategy_report('request context', context)
    assert len(calls) == 2
    assert 'donor@812 (increase)' in report
    assert trace[-1]['kind'] == 'reference_bound_plan_rendering'
