"""Optional Streamlit UI tests; run with the gui extra installed."""
from pathlib import Path
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from aso_agents.gui.config import SetupError

APP = Path(__file__).parents[1] / "aso_agents" / "gui" / "app.py"


def app():
    return AppTest.from_file(str(APP)).run(timeout=15)


def test_start_page_gates_chat_and_provider_switch_invalidates_models(monkeypatch):
    monkeypatch.setattr("aso_agents.gui.config.list_models", lambda provider, key: ["openai/gpt-5.2"])
    ui = app()
    assert not ui.exception
    assert not ui.chat_input
    assert ui.button[0].disabled
    ui.text_input[0].set_value("test-key").run()
    assert ui.selectbox[1].value == "deepseek/deepseek-v4.1-flash-1017"
    ui.button[0].click().run()
    assert ui.selectbox[1].options == ["deepseek/deepseek-v4.1-flash-1017", "openai/gpt-5.2"]
    assert ui.selectbox[1].value == "deepseek/deepseek-v4.1-flash-1017"
    ui.selectbox[1].select("openai/gpt-5.2").run()
    assert "xhigh" in ui.selectbox[2].options
    assert ui.selectbox[2].value == "xhigh"
    ui.selectbox[2].select("low").run()
    assert ui.selectbox[2].value == "low"
    ui.selectbox[0].select("openai").run()
    assert len(ui.selectbox) == 1
    assert not ui.chat_input


def test_validation_failure_keeps_setup_open(monkeypatch):
    monkeypatch.setattr("aso_agents.gui.config.list_models", lambda *args: ["openai/gpt-5.2"])
    def fail(*args):
        raise SetupError("API key", "Authentication failed.")
    monkeypatch.setattr("aso_agents.gui.config.validate", fail)
    ui = app()
    ui.text_input[0].set_value("bad-key").run()
    ui.button[0].click().run()
    ui.button[1].click().run()
    assert "API key" in ui.error[0].value
    assert not ui.chat_input
    assert not ui.exception


def test_success_opens_chat_and_results_tabs(monkeypatch):
    monkeypatch.setattr("aso_agents.gui.config.list_models", lambda *args: ["openai/gpt-5.2"])
    monkeypatch.setattr("aso_agents.gui.config.validate", lambda *args: None)
    ui = app()
    ui.session_state["messages"] = [("user", "Earlier request")]
    ui.text_input[0].set_value("test-key").run()
    ui.button[0].click().run()
    ui.button[1].click().run()
    assert not ui.exception
    assert len(ui.chat_input) == 1
    assert [tab.label for tab in ui.tabs] == ["Chat", "Results", "Files"]
    assert "test-key" not in repr(ui.session_state["validated"])
    assert ("user", "Earlier request") in ui.session_state["messages"]


def test_chat_delegates_request_and_collects_clarification(monkeypatch):
    from queue import Queue
    from types import SimpleNamespace as NS
    from aso_agents.gui.config import Settings
    events, answers = Queue(), Queue()
    process = NS(is_alive=lambda: True, exitcode=None)
    calls = []
    def start(settings, request):
        calls.append((settings, request))
        events.put(("questions", {"questions": ["How many ASOs?"]}))
        return process, events, answers
    monkeypatch.setattr("aso_agents.gui.worker.start", start)
    ui = AppTest.from_file(str(APP))
    ui.session_state["validated"] = Settings("openai", "test")
    ui.session_state["messages"] = []
    ui.run(timeout=15)
    ui.chat_input[0].set_value("Design ten ASOs").run()
    assert calls[0][1] == "Design ten ASOs"
    assert ui.chat_input[0].disabled
    assert not any('class="aso-typing"' in item.value for item in ui.markdown)
    ui.text_input[0].set_value("10").run()
    next(button for button in ui.button if button.label == "Send answers").click().run()
    assert answers.get_nowait() == [{"question": "How many ASOs?", "answer": "10"}]
    assert any('class="aso-typing"' in item.value for item in ui.markdown)
    events.put(("message", "I am checking the target transcript."))
    ui.run()
    assert any("checking the target transcript" in item.value for item in ui.markdown)
    events.put(("result", {"status": "run_complete"}))
    events.put(("done", None))
    ui.run()
    assert not any('class="aso-typing"' in item.value for item in ui.markdown)
    assert not ui.chat_input[0].disabled
    assert not ui.exception


def test_saved_report_and_markdown_are_read_only(tmp_path):
    from aso_agents.gui.config import Settings
    report = tmp_path / "report.html"
    markdown = tmp_path / "report.md"
    report.write_text("<html><body>Existing scientific report</body></html>", encoding="utf-8")
    markdown.write_text("# Saved findings\nEvidence unchanged.", encoding="utf-8")
    before = {path: path.read_bytes() for path in [report, markdown]}
    ui = AppTest.from_file(str(APP))
    ui.session_state["validated"] = Settings("openai", "test")
    ui.session_state["messages"] = []
    ui.session_state["result"] = {"run_dir": str(tmp_path)}
    ui.run(timeout=15)
    assert not ui.selectbox
    assert len(ui.get("download_button")) == 2
    assert any("report.html" == item.value for item in ui.markdown)
    assert any("report.md" == item.value for item in ui.markdown)
    assert not ui.exception
    assert any("Saved findings" in item.value for item in ui.markdown)
    assert {path: path.read_bytes() for path in before} == before


def test_live_activity_remains_in_session_after_completion():
    from queue import Queue
    from types import SimpleNamespace as NS
    from aso_agents.gui.config import Settings
    events, answers = Queue(), Queue()
    ui = AppTest.from_file(str(APP))
    ui.session_state["validated"] = Settings("openai", "test")
    ui.session_state["messages"] = [("user", "Design ASOs")]
    ui.session_state["show_activity"] = True
    ui.session_state["job"] = (NS(is_alive=lambda: True, exitcode=None), events, answers)
    events.put(("activity", "Starting: parse goal request."))
    ui.run(timeout=15)
    assert any("parse goal request" in item.value for item in ui.caption)
    events.put(("result", {"status": "run_complete"}))
    events.put(("done", None))
    ui.run()
    ui.run()
    assert ("activity", "Starting: parse goal request.") in ui.session_state["messages"]
    assert any("parse goal request" in item.value for item in ui.caption)
    assert not ui.exception


def test_files_list_includes_large_outputs_without_preloading(tmp_path, monkeypatch):
    from aso_agents.gui.config import Settings
    output = tmp_path / "large_output.bin"
    with output.open("wb") as stream:
        stream.truncate(26 * 1024 * 1024)
    original = Path.read_bytes
    def guarded_read(path):
        if path == output:
            raise AssertionError("Download must be deferred until clicked")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    ui = AppTest.from_file(str(APP))
    ui.session_state["validated"] = Settings("openai", "test")
    ui.session_state["messages"] = []
    ui.session_state["result"] = {"run_dir": str(tmp_path)}
    ui.run(timeout=15)
    assert not ui.exception
    assert not ui.selectbox
    assert len(ui.get("download_button")) == 1
    assert any(item.value == "large_output.bin" for item in ui.markdown)


def test_gui_keeps_returned_results_and_expands_error_details(tmp_path):
    from queue import Queue
    from types import SimpleNamespace as NS
    from aso_agents.gui.config import Settings
    events, answers = Queue(), Queue()
    events.put(("result", {"status": "run_complete", "run_dir": str(tmp_path)}))
    events.put(("error", "The GUI worker encountered TypeError: display failure"))
    events.put(("diagnostic", {"type": "TypeError", "message": "display failure", "traceback": "TypeError: display failure"}))
    events.put(("done", None))
    ui = AppTest.from_file(str(APP))
    ui.session_state["validated"] = Settings("openai", "test")
    ui.session_state["messages"] = []
    ui.session_state["job"] = (NS(is_alive=lambda: True, exitcode=None), events, answers)
    ui.run(timeout=15)
    assert not ui.exception
    assert ui.session_state["result"]["status"] == "run_complete"
    assert any(item.label == "Technical details · TypeError" for item in ui.expander)
    assert any("TypeError: display failure" in item.value for item in ui.code)


def test_commentary_is_visible_while_technical_activity_is_hidden():
    from aso_agents.gui.config import Settings
    ui = AppTest.from_file(str(APP))
    ui.session_state["validated"] = Settings("openai", "test")
    ui.session_state["messages"] = [
        ("assistant", "I'm resolving the transcripts that matter for this request."),
        ("activity", "Requesting a model response."),
    ]
    ui.run(timeout=15)
    assert any("resolving the transcripts" in item.value for item in ui.markdown)
    assert not any("Requesting a model response" in item.value for item in ui.caption)
    ui.checkbox[0].check().run()
    assert any("Requesting a model response" in item.value for item in ui.caption)
    ui.checkbox[0].uncheck().run()
    assert any("resolving the transcripts" in item.value for item in ui.markdown)
    assert not ui.exception
