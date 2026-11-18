"""GUI compatibility with shared completion, output and installation contracts."""
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import os
import sys

import pytest

from aso_agents.gui import artifacts
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
    assert any(kind == 'questions' and value['questions'] == ['Install OligoAI now? Answer yes or no.'] for kind, value in saved)
    assert saved[-1] == ('done', None)


@pytest.fixture
def output_server(tmp_path):
    pytest.importorskip("starlette")
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    (tmp_path / "report.html").write_text('<a href="table.csv">Download</a>')
    (tmp_path / "table.csv").write_text("ASO,score\n1,0.81234567890123\n")
    (tmp_path / ".local-config").write_text("private")
    (tmp_path / "reference.fa").symlink_to(tmp_path / ".local-config")
    prefix = artifacts.route_prefix("/preview/")
    token = artifacts.register(tmp_path)
    app = Starlette(routes=[Route(prefix + "/{token}/{relative:path}", artifacts.serve_artifact)])
    with TestClient(app, base_url="https://forwarded.example") as client:
        yield client, f"{prefix}/{token}/", tmp_path


def test_report_and_artifacts_share_forwarded_origin(output_server):
    client, base, run_dir = output_server
    frame = artifacts.report_frame(run_dir, "/preview/")
    assert f'src="{base}report.html"' in frame
    assert "127.0.0.1" not in frame
    report = client.get(base + "report.html")
    assert report.status_code == 200 and "table.csv" in report.text
    assert "sandbox allow-scripts" in report.headers["Content-Security-Policy"]
    csv = client.get(base + "table.csv")
    assert csv.text == "ASO,score\n1,0.81234567890123\n"
    assert csv.headers["Content-Disposition"].startswith("attachment")
    for relative in ("reference.fa", ".local-config", "%2e%2e/table.csv", "%2ftable.csv"):
        assert client.get(base + relative).status_code == 404
    assert client.get("/preview/_modal_aso_artifacts/unknown/report.html").status_code == 404


def test_report_registration_is_bounded_and_reuses_existing_token(tmp_path, monkeypatch):
    from collections import OrderedDict
    monkeypatch.setattr(artifacts, "_roots", OrderedDict())
    first = artifacts.register(tmp_path / "first")
    for index in range(51):
        token = artifacts.register(tmp_path / str(index))
    assert len(artifacts._roots) == 50
    assert first not in artifacts._roots
    assert artifacts.register(tmp_path / "50") == token


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
    next(item for item in ui.text_input if item.label == 'Find a file').set_value('100.csv').run()
    assert not ui.exception and len(ui.get('download_button')) == 1
