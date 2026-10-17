from copy import deepcopy

import pytest

from aso_agents.response_assembly import apply_scope_edits, scope_finding_errors


@pytest.mark.parametrize('kind', ['ambiguous_evidence_scope', 'unsupported_factual_claim'])
def test_valid_nonbinding_observation_does_not_require_edit_authorization_scope(kind):
    response = {'suggested_followups': ['Inspect the recorded count for this observation.']}
    source = {'evidence_scope': {'record_count': 2}, 'delivery_scope': {'assessment_count': 2}}
    finding = {'response_path': '/suggested_followups/0',
               'quote': 'the recorded count for this observation',
               'kind': kind, 'scope_kind': 'component_scope',
               'evidence_paths': ['/evidence_scope/record_count'],
               'reason': 'The suggested follow-up has an ambiguous observation scope.'}
    before = deepcopy(response)
    assert not scope_finding_errors({'findings': [finding]}, response, source)
    with pytest.raises(ValueError, match='Only a source-bound factual contradiction'):
        apply_scope_edits(response, {'replacements': {'0': 'a different count'}}, [finding], source)
    assert response == before
    factual = {**finding, 'kind': 'contradicted_fact'}
    assert scope_finding_errors({'findings': [factual]}, response, source)
    for invalid in [{**finding, 'evidence_paths': ['/evidence_scope/absent']},
                    {**finding, 'quote': 'a quote that is not in the draft'}]:
        assert scope_finding_errors({'findings': [invalid]}, response, source)
