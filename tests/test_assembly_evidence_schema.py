from copy import deepcopy

import pytest

from aso_agents.pipeline_tools import _json_schema_validation_errors
from aso_agents.response_assembly import assembly_findings_schema, source_evidence_pointers
from aso_agents.evaluation_response_audit import pointer_value


def test_exact_source_citation_schema_includes_nested_values_without_guessing_paths():
    source = {'evidence_scope': {'observations': [
        {'recorded_inventory': {'feature_evidence_availability': {
            'count_scope': 'Source records, not biological events.',
            'count': 0, 'unresolved': None, 'empty': [],
        }}},
        {'recorded_inventory': {}},
    ]}, 'escaped/~key': {'': False}}
    before = deepcopy(source)
    pointers = source_evidence_pointers(source)
    assert len(pointers) == len(set(pointers))
    assert all(pointer_value(source, path) is not ... for path in pointers)
    assert '/escaped~1~0key/' in pointers
    assert '/evidence_scope/observations/1/recorded_inventory' in pointers
    assert source == before
    response = {'answer_summary': 'Counts denote biological events.'}
    schema = assembly_findings_schema(response, {'/answer_summary': response['answer_summary']}, source)
    allowed = schema['properties']['findings']['items']['properties']['evidence_paths']['items']['enum']
    assert allowed == pointers
    correct = '/evidence_scope/observations/0/recorded_inventory/feature_evidence_availability/count_scope'
    finding = {'response_path': '/answer_summary', 'quote': response['answer_summary'],
               'kind': 'contradicted_fact', 'evidence_paths': [correct],
               'reason': 'The recorded count unit differs.', 'scope_kind': 'retained_evidence_availability'}
    assert not _json_schema_validation_errors({'findings': [finding]}, schema)
    for invalid in [
        '/source' + correct,
        '/evidence_scope/observations/0/recorded_inventory/count_scope',
        correct.replace('/0/', '/1/'),
        correct.replace('/0/', '/00/'),
    ]:
        finding['evidence_paths'] = [invalid]
        assert _json_schema_validation_errors({'findings': [finding]}, schema)
    assert source == before  # No citation repair or cross-row fallback.


@pytest.mark.parametrize('rows', [1, 10, 100])
def test_source_pointer_inventory_scales_with_metadata_nodes(rows):
    source = {'observations': [{'id': i, 'value': None, 'scope': 'recorded'} for i in range(rows)]}
    assert len(source_evidence_pointers(source)) == 1 + rows * 4


def test_schema_enumeration_does_not_replace_source_relevance_validation():
    from aso_agents.response_assembly import scope_finding_errors
    source = {'request_scope': {'observation_count': 2}, 'delivery_scope': {'assessment_count': 2}}
    response = {'answer_summary': 'One observation was assessed.'}
    finding = {'response_path': '/answer_summary', 'quote': response['answer_summary'],
               'kind': 'contradicted_fact', 'scope_kind': 'assessment_membership',
               'reason': 'Deliberately wrong scope register.',
               'evidence_paths': ['/request_scope/observation_count']}
    schema = assembly_findings_schema(response, {'/answer_summary': response['answer_summary']}, source)
    assert not _json_schema_validation_errors({'findings': [finding]}, schema)
    assert scope_finding_errors({'findings': [finding]}, response, source)
