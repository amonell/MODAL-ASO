from copy import deepcopy
import json

import pytest

from aso_agents.report_recovery import retained_assembled_report
from aso_agents.response_artifacts import ResponseArtifacts, digest


def archive(tmp_path, *, science_change=False):
    original = {'answer_summary': 'Only one observation was assessed.',
        'evaluation_assessments': [{'input_row_id': 'row_a', 'mechanistic_plausibility': 'Unmeasured.',
                                   'ranking_rationale': 'No potency ranking.'}],
        'original_component_responses': [{'unchanged': 'scientific response'}]}
    candidate = deepcopy(original)
    candidate['answer_summary'] = 'The supplied observation is assessed; activity is unmeasured.'
    if science_change:
        candidate['evaluation_assessments'][0]['mechanistic_plausibility'] = 'Invented measured activity.'
    source = {'assembly_source': {'request_scope': {'observation_count': 1}}, 'assembled_response': original}
    saved = ResponseArtifacts(tmp_path/'assembly', source, {'model': 'neutral'})
    candidate['native_assembly_review'] = {'status': 'no_request_scope_contradiction_flagged',
        'source_sha256': digest(source['assembly_source']), 'original_draft_sha256': digest(original),
        'accepted_draft_sha256': digest(candidate)}
    saved.accept(candidate)
    return original, saved


def test_retained_assembly_is_only_a_draft_with_identical_science(tmp_path):
    original, saved = archive(tmp_path)
    retained = retained_assembled_report(original, tmp_path)
    assert retained['draft']['answer_summary'] != original['answer_summary']
    assert retained['draft']['evaluation_assessments'] == original['evaluation_assessments']
    assert 'native_assembly_review' not in retained['draft']
    changed = deepcopy(original); changed['original_component_responses'][0]['unchanged'] = 'different science'
    assert retained_assembled_report(changed, tmp_path) is None


def test_retained_assembly_rejects_scientific_changes_even_with_consistent_hashes(tmp_path):
    original, _ = archive(tmp_path, science_change=True)
    with pytest.raises(ValueError, match='scientific components'):
        retained_assembled_report(original, tmp_path)


@pytest.mark.parametrize('fault', ['response', 'receipt', 'component_key'])
def test_retained_assembly_rejects_broken_provenance(tmp_path, fault):
    original, saved = archive(tmp_path)
    path = saved.root/'accepted.json'; data = json.loads(path.read_text())
    if fault == 'response': data['response']['answer_summary'] = 'Tampered'
    if fault == 'component_key': data['component_key'] = 'wrong'
    if fault == 'receipt':
        data['response']['native_assembly_review']['source_sha256'] = 'wrong'
        data['response_sha256'] = digest(data['response'])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        retained_assembled_report(original, tmp_path)
