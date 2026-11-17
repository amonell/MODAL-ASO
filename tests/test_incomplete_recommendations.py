from copy import deepcopy
import io
import json

from aso_agents import shortlist_review
from aso_agents.candidate_assessment import write_assessment_artifacts
from aso_agents.recommendation_status import completion_review
from aso_agents.report_recommendations import recommendations_html
from aso_agents.report_explorer_data import project_records
from run_goal_intake import _CliNarrator
from test_shortlist_review import pool, run, model


def test_rejected_five_candidate_draft_retains_four_provisional_records(tmp_path, monkeypatch):
    rows = pool(8)
    original = deepcopy(rows)
    def response(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'rank':
            result['suggested_size'] = 5
        return result, metadata
    monkeypatch.setattr(shortlist_review, '_model_call', response)
    def failed_explanation(catalog, ids, *args, **kwargs):
        assert len(ids) == 5
        raise shortlist_review.ReviewUnavailable('explanation_evidence_review_incomplete', details={
            'stage': 'explanation_audit', 'draft_molecule_count': 5,
            'issues': [{'candidate_id': ids[0], 'problem': 'The recorded count unit is genomic positions.'}]})
    monkeypatch.setattr(shortlist_review, '_explain', failed_explanation)
    selected, _, review = run(rows, tmp_path)
    assert rows == original and len(selected) == 4
    assert review['status'] == 'unavailable'
    assert review['draft_molecule_count'] == 5 and review['provisional_molecule_count'] == 4
    paths = write_assessment_artifacts(rows, selected, {}, tmp_path, review=review)
    shortlist = json.loads((tmp_path/'recommended_shortlist.json').read_text())
    assert shortlist['status'] == 'provisional' and shortlist['count'] == 4
    assert shortlist['requested_count'] is None
    assert len(shortlist['records']) == 4
    status = completion_review(tmp_path, paths)
    assert '4 provisional ASOs' in status['completion_notice']
    assert status['draft_molecule_count'] == 5
    data = project_records(selected)
    data['recommendation_review'] = review
    for index, row in enumerate(data['records'], 1):
        row['panel_rank'] = index
        row['recommendation'] = {'why_chosen': 'Rejected draft text'}
    before = deepcopy(data)
    rendered = recommendations_html(data)
    assert data == before
    assert rendered.count('data-recommendation-pick=') == 4
    assert '5-ASO review draft remains unapproved' in rendered
    assert 'Add provisional panel to basket' in rendered
    assert 'Rejected draft text' not in rendered
    assert '<dl></dl>' not in rendered
    output = io.StringIO()
    _CliNarrator(stream=output, enable_stream=False, enable_spinner=False).show_completion({
        'status':'run_complete','run_dir':str(tmp_path),'artifacts':paths})
    assert 'Recommendation review remains incomplete.' in output.getvalue()
    assert 'The run is complete.' not in output.getvalue()
    assert 'Provisional panel' in (tmp_path/'recommended_shortlist.md').read_text()


def test_failed_audit_diagnostics_survive_without_approving_a_draft(tmp_path, monkeypatch):
    def broken_audit(kind, payload, schema):
        result, metadata = model(kind, payload, schema)
        if kind == 'audit':
            result['field_reviews'].pop()
        return result, metadata
    monkeypatch.setattr(shortlist_review, '_model_call', broken_audit)
    _, _, review = run(pool(), tmp_path, {'candidate_limit': 2})
    assert review['status'] == 'unavailable'
    assert review['failure_details']['stage'] == 'audit'
    assert review['draft_molecule_count'] == 2
    assert review['failure_details']['validation_errors']
    assert review.get('recommendations') is None


def test_intentionally_skipped_review_is_visible_without_a_failure_claim(tmp_path):
    directory = tmp_path/'artifacts/candidates'
    directory.mkdir(parents=True)
    (directory/'recommended_shortlist.json').write_text(json.dumps({
        'review':{'status':'skipped'},'records':[{}]}))
    status = completion_review(tmp_path)
    assert 'skipped at your request' in status['completion_notice']
    assert '1 provisional ASO is' in status['completion_notice']
