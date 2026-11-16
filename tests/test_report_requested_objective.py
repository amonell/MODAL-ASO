"""Requested goals survive approach selection and incomplete candidate coverage."""
from copy import deepcopy
from html.parser import HTMLParser
import json
import os
import shutil
import subprocess

import pytest

from aso_agents.readable_report import basic_document
from aso_agents.report_explorer import explorer_html
from aso_agents.report_explorer_data import project_run, project_records
from test_report_explorer import run_fixture, save


OBJECTIVE = (
    "Reduce inclusion of the selected poison exon to increase productive mRNA by "
    "tiling 18-nt fully 2'-MOE/PS steric-blocking ASOs across the exon and both of its splice junctions."
)


class GoalText(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.active = False
        self.text = ''
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        if dict(attrs).get('data-role') == 'objective':
            self.active = True

    def handle_data(self, data):
        if self.active:
            self.text += data

    def handle_endtag(self, tag):
        if tag == 'p':
            self.active = False


def two_routes(root):
    goal, raw, pool = run_fixture(root)
    goal['design_programs'][0].update(objective=OBJECTIVE, desired_delta='decrease', desired_effect='skipping')
    for row in [*raw, *pool]:
        row['splice_site_deltas'] = [*row['splice_site_deltas'], {
            'canonical_label': 'donor@120', 'site_kind': 'donor',
            'donor': {'chrom': 'chr1', 'position': 120},
            'delta_donor_deletion': -.1, 'delta_donor_replacement': -.3,
        }]
    pool.append({**deepcopy(pool[0]), 'route_label': 'donor@120'})
    save(root, 'artifacts/intake/structured_goal.json', goal)
    save(root, 'artifacts/candidates/candidate_assessments.json', {'rows': pool})
    save(root, 'artifacts/design/steric_jobs/j/scored_tiles.json', {'rows': raw})
    return goal


def test_run_goal_retains_the_complete_biological_objective_above_splice_approaches(tmp_path):
    two_routes(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    data = project_run(tmp_path)
    assert len(OBJECTIVE) > 150
    assert data['objective'] == OBJECTIVE
    assert [c['objective'] for c in data['contexts']] == [
        'Decrease use of the acceptor at 110 for TEST.',
        'Decrease use of the donor at 120 for TEST.',
    ]
    assert GoalText(explorer_html(data)).text == OBJECTIVE
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_all_requested_programs_remain_visible_even_when_one_has_no_candidates(tmp_path):
    goal = two_routes(tmp_path)
    second = 'Block translation of a separate transcript while preserving the productive isoform.'
    goal['design_programs'].append({'program_id': 'unfulfilled', 'objective': second, 'modality': 'steric'})
    save(tmp_path, 'artifacts/intake/structured_goal.json', goal)
    data = project_run(tmp_path)
    assert {c['program'] for c in data['contexts']} == {'p'}
    assert data['objective'] == OBJECTIVE + '\n\n' + second
    assert GoalText(explorer_html(data)).text == data['objective']


@pytest.mark.parametrize('child', [False, True])
def test_missing_program_prose_uses_the_saved_request_with_child_task_scope(tmp_path, child):
    root = tmp_path / 'child_runs' / 'task' if child else tmp_path
    goal, _, _ = run_fixture(root)
    request = 'Evaluate the exon-skipping and intron-retention designs for the requested transcript.'
    label = 'Evaluate the intron-retention designs.'
    save(root, 'artifacts/intake/goal_intake.json', {
        'request': request,
        'request_envelope': {'tasks': [{'label': label}]},
    })
    data = project_run(root)
    assert data['objective'] == (label if child else request)
    assert GoalText(explorer_html(data)).text == data['objective']


def test_request_text_is_rendered_as_text_and_evaluation_scope_is_preserved():
    objective = 'Compare <candidate> & <control> under the supplied experimental conditions.'
    data = project_records([{'antisense_5to3': 'ACGT'}], objective=objective)
    document = explorer_html(data)
    assert GoalText(document).text == objective
    assert '<candidate>' not in document and '&lt;candidate&gt;' in document


def test_browser_switching_approaches_keeps_the_requested_goal_visible(tmp_path):
    chrome = shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:
        pytest.skip('A local Chromium browser is required for this interaction check.')
    two_routes(tmp_path)
    data = project_run(tmp_path)
    document = basic_document('Requested objective check', '', candidate_data=data)
    check = '''<script>
    window.addEventListener('load', () => {
      const root = document.querySelector('.report-explorer');
      const goal = root.querySelector('[data-role="objective"]');
      const select = root.querySelector('[data-role="context"]');
      const first = goal.textContent;
      const stable = [], labels = [], counts = [];
      for (const option of select.options) {
        select.value = option.value;
        select.dispatchEvent(new Event('change', {bubbles:true}));
        stable.push(goal.textContent === first);
        labels.push([...root.querySelectorAll('[data-approach-context]')]
          .filter(el => !el.hidden).map(el => el.textContent).join(' '));
        counts.push(root.querySelector('[data-role="count"]').textContent);
      }
      document.body.dataset.requestedGoalStable = String(stable.length === 3 && stable.every(Boolean));
      document.body.dataset.approachChanged = String(labels[0] !== labels[1]);
      document.body.dataset.candidateScopeChanged = String(counts[0] !== counts[2]);
    });
    </script>'''
    path = tmp_path / 'report.html'
    path.write_text(document.replace('</body>', check + '</body>'))
    environment = {k: v for k, v in os.environ.items() if not any(word in k for word in ('API_KEY', 'TOKEN', 'SECRET'))}
    environment.update(TMPDIR=os.environ.get('TMPDIR', '/tmp'), XDG_CACHE_HOME=str(tmp_path / 'cache'))
    result = subprocess.run([chrome, '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
        '--disable-background-networking', '--no-first-run', '--user-data-dir=' + str(tmp_path / 'chrome'),
        '--virtual-time-budget=1000', '--dump-dom', path.as_uri()],
        capture_output=True, text=True, env=environment, check=True, timeout=30)
    assert GoalText(result.stdout).text == OBJECTIVE
    assert 'data-requested-goal-stable="true"' in result.stdout
    assert 'data-approach-changed="true"' in result.stdout
    assert 'data-candidate-scope-changed="true"' in result.stdout
