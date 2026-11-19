"""Recommendation coverage describes the actual level of evidence review."""
from aso_agents.report_recommendations import recommendations_html
from test_report_recommendation_browser import recommendation_data


def test_query_review_distinguishes_survey_comparison_and_complete_evidence():
    data = recommendation_data(3)
    data['recommendation_review'].update(
        review_coverage={'indexed_molecule_count': 4057, 'matched_molecule_count': 100,
                         'compared_molecule_count': 64, 'fully_reviewed_molecule_count': 12},
        selection_process={'summary': 'Candidates were compared across recorded activity and target coverage.',
            'steps': [{'rationale': 'Retain activity predictions under the requested conditions.',
                       'matched_count': 100, 'unknown_count': 15},
                      {'rationale': '<script>unsafe()</script>'}]})
    rendered = recommendations_html(data)
    assert '3 recommended from 4,057 eligible ASOs.' in rendered
    assert '4,057 ASOs reviewed' not in rendered
    assert 'ASOs individually compared</dt><dd>64' in rendered
    assert 'ASOs with complete evidence reviewed</dt><dd>12' in rendered
    assert '100 ASOs matched; 15 ASOs had unavailable measurements.' in rendered
    assert 'How these ASOs were chosen' in rendered
    assert '&lt;script&gt;unsafe()&lt;/script&gt;' in rendered
    assert '<script>unsafe()' not in rendered


def test_legacy_and_provisional_receipts_keep_their_original_coverage_meaning():
    data = recommendation_data(3)
    assert '3 recommended from 3 ASOs reviewed.' in recommendations_html(data)
    data['recommendation_review'].update(status='unavailable',
        review_coverage={'indexed_molecule_count': 4057},
        selection_process={'summary': 'Unapproved proposal'})
    rendered = recommendations_html(data)
    assert 'Provisional panel' in rendered
    assert 'Unapproved proposal' not in rendered
    assert 'ASOs reviewed' not in rendered


def test_new_run_projection_carries_saved_review_coverage_into_html(tmp_path):
    import json
    from aso_agents.report_explorer_data import project_run
    from aso_agents.readable_report import basic_document
    from test_report_explorer import run_fixture
    run_fixture(tmp_path)
    path = tmp_path/'artifacts/candidates/recommended_shortlist.json'
    saved = json.loads(path.read_text())
    saved['review'] = {'status': 'completed', 'summary': 'A complementary research panel.',
        'review_strategy': 'query_guided',
        'review_coverage': {'indexed_molecule_count': 4057, 'matched_molecule_count': 100,
            'compared_molecule_count': 64, 'fully_reviewed_molecule_count': 12},
        'selection_process': {'summary': 'Compared recorded activity and binding positions.', 'steps': []}}
    path.write_text(json.dumps(saved))
    data = project_run(tmp_path)
    assert data['recommendation_review']['review_coverage'] == saved['review']['review_coverage']
    rendered = basic_document('Test', '# Results', candidate_data=data)
    assert '1 recommended from 4,057 eligible ASOs.' in rendered
    assert 'ASOs individually compared</dt><dd>64' in rendered
