from copy import deepcopy

from aso_agents.evaluation_evidence import unsupported_measurement_reference_findings
from aso_agents.evaluation_response_audit import (apply_bound_narrative_patch,
                                                  response_audit_repair_findings)


def test_unsupported_citation_is_bound_to_one_assessment_and_source_row():
    source = {'evaluation_summary': {
        'measurement_evidence': {'E001': {'value': 'recorded evidence'}},
        'provided_asos': [
            {'input_row_id': 'row-A', 'name': 'A', 'knockdown_proxy_rank': 1},
            {'input_row_id': 'row-B', 'name': 'B', 'knockdown_proxy_rank': 7},
        ]}}
    response = {'evaluation_assessments': [
        {'input_row_id': 'row-A', 'name': 'A', 'ranking_rationale': 'Supported {{E001}}.'},
        {'input_row_id': 'row-B', 'name': 'B', 'ranking_rationale': 'Claim {{knockdown_proxy_rank: 7}} needs repair.'},
    ]}
    original = deepcopy(response)
    findings = unsupported_measurement_reference_findings(response, source)
    assert len(findings) == 1
    assert findings[0]['response_path'] == '/evaluation_assessments/1/ranking_rationale'
    assert findings[0]['evidence_paths'] == ['/evaluation_summary/provided_asos/1']
    bound = response_audit_repair_findings({'findings': findings}, response, source)
    corrected = apply_bound_narrative_patch(response, {'corrections': {
        '/evaluation_assessments/1/ranking_rationale': 'Claim requires evidence-scoped interpretation.'}}, bound, source)
    assert response == original
    assert corrected['evaluation_assessments'][0] == response['evaluation_assessments'][0]
    assert corrected['evaluation_assessments'][1]['ranking_rationale'] != response['evaluation_assessments'][1]['ranking_rationale']


def test_no_scoped_repair_without_exact_row_identity():
    source = {'evaluation_summary': {'measurement_evidence': {},
              'provided_asos': [{'input_row_id': 'row-A'}]}}
    response = {'evaluation_assessments': [{'input_row_id': 'unknown',
                'ranking_rationale': 'Claim {{not-a-token}}.'}]}
    assert unsupported_measurement_reference_findings(response, source) == []
