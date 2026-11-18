"""GUI compatibility with shared completion, output and installation contracts."""
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import os
import sys
import urllib.error
import urllib.request

import pytest

from aso_agents.gui.artifacts import artifact_server, register
from aso_agents.gui.config import Settings, model_family
from aso_agents.gui.diagnostics import redact_credentials
from aso_agents.gui.presentation import result_message
from aso_agents.gui.worker import plain_terminal_text, run


@pytest.mark.parametrize('name,expected', [
    ('gpt-5.2-2025-12-11', 'gpt-5.2'), ('openai/gpt-5-mini', 'gpt-5-mini'),
    ('gpt-5.2-pro', None), ('gpt-5.2-2025-99-99', None), ('gpt-5-nano-2025-08-07', 'gpt-5-nano')])
def test_model_family_uses_exact_names_and_valid_dates(name, expected):
    assert model_family(name) == expected


def test_plain_text_and_credentials_preserve_surrounding_message():
    assert plain_terminal_text('before \x1b[31mred\x1b[0m after') == 'before red after'
    assert redact_credentials('Auth: bEaReR other-secret; key="sk-key123".', '') == 'Auth: bEaReR [redacted] key="[redacted]".'


@pytest.mark.parametrize('status', ['unavailable', 'pending_review', 'skipped'])
def test_incomplete_review_stays_visible_in_gui(status):
    message = result_message({'status': 'run_complete', 'recommendation_review': {
        'status': status, 'panel_count': 4}})
    assert '4 provisional ASOs' in message
    assert 'Analysis completed.' not in message


@pytest.mark.parametrize('answer,installed', [('yes', True), ('no', False), ('maybe', False)])
def test_shared_controller_events_support_explicit_model_installation(monkeypatch, answer, installed):
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    events, answers = Queue(), Queue()
    answers.put([{'question': 'Install OligoAI now? Answer yes or no.', 'answer': answer}])
    def controller(*, argv, clarification_resolver, result_callback, event_callback):
        assert argv == ['--', 'Evaluate supplied ASOs']
        event_callback({'type': 'progress', 'message': 'Reviewing 83 ASOs'})
        assert event_callback({'type': 'model_installation', 'model': 'OligoAI',
                               'message': 'The local model is missing.'}) is installed
        result_callback({'status': 'run_complete'})
    monkeypatch.setitem(sys.modules, 'run_goal_intake', SimpleNamespace(main=controller))
    run(Settings('openrouter', 'test-key'), 'Evaluate supplied ASOs', events, answers)
    saved = []
    while not events.empty():
        saved.append(events.get_nowait())
    assert ('message', 'Reviewing 83 ASOs') in saved
    assert any(kind == 'questions' and value['questions'] == ['Install OligoAI now? Answer yes or no.'] for kind, value in saved)
    assert saved[-1] == ('done', None)


@pytest.fixture
def output_server(tmp_path):
    run_dir = tmp_path / 'run';run_dir.mkdir()
    (run_dir / 'report.html').write_text('<!doctype html><title>ASO report</title><h1>Results</h1><a href="artifacts/table.csv">Download ASOs</a>')
    (run_dir / 'artifacts').mkdir()
    (run_dir / 'artifacts/table.csv').write_text('ASO,score\n1,0.81234567890123\n')
    (run_dir / 'genome.fa').symlink_to('/etc/passwd')
    server = artifact_server(run_dir)
    yield server, 'http://' + server.authority + '/' + server.token + '/'
    server.shutdown();server.server_close()


def test_report_assets_stream_with_path_and_host_boundaries(output_server):
    server, base = output_server
    with urllib.request.urlopen(base + 'report.html') as response:
        assert b'Download ASOs' in response.read()
        assert response.headers['Referrer-Policy'] == 'no-referrer'
    with urllib.request.urlopen(base + 'artifacts/table.csv') as response:
        assert response.headers['Content-Disposition'] == 'attachment'
        assert b'0.81234567890123' in response.read()
    for relative in ['../report.html', 'genome.fa', '%2fetc/passwd']:
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + relative)
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen('http://' + server.authority + '/wrong/report.html')
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(urllib.request.Request(base + 'report.html', headers={'Host': 'another.example'}))


def test_many_reports_share_one_bounded_server(output_server, tmp_path):
    server, _ = output_server
    for i in range(100):
        register(server, tmp_path / str(i))
    assert len(server.roots) == 50
    token = register(server, tmp_path / '99')
    assert register(server, tmp_path / '99') == token


def test_embedded_report_relative_download_works_in_browser(output_server):
    import shutil
    playwright = pytest.importorskip('playwright.sync_api')
    executable = shutil.which('google-chrome') or shutil.which('chromium')
    if not executable:
        pytest.skip('Local browser unavailable')
    _, base = output_server
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
        page = browser.new_page(accept_downloads=True)
        page.set_content('<iframe title="Report" src="' + base + 'report.html"></iframe>')
        with page.expect_download() as download:
            page.frame_locator('iframe').get_by_role('link', name='Download ASOs').click()
        assert Path(download.value.path()).read_text() == 'ASO,score\n1,0.81234567890123\n'
        browser.close()


def test_large_file_lists_are_searchable_and_paginated(tmp_path):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    for i in range(101):
        (tmp_path / f'candidate-{i:03}.csv').write_text('ASO,score\n')
    app = Path(__file__).parents[1] / 'aso_agents/gui/app.py'
    ui = AppTest.from_file(str(app))
    ui.session_state['validated'] = Settings('openai', 'test-key')
    ui.session_state['messages'] = []
    ui.session_state['result'] = {'status': 'run_complete', 'run_dir': str(tmp_path)}
    ui.run(timeout=20)
    assert not ui.exception and len(ui.get('download_button')) == 50
    ui.selectbox[0].select(3).run()
    assert not ui.exception and len(ui.get('download_button')) == 1
    ui.text_input[0].set_value('100.csv').run()
    assert not ui.exception and len(ui.get('download_button')) == 1
