"""Process-isolated bridge to the unchanged CLI controller."""
import multiprocessing
import os
import io
import re
from contextlib import redirect_stdout, redirect_stderr


class ChatStream(io.TextIOBase):
    """Collect terminal narration into complete chat lines inside the worker."""
    def __init__(self, events, api_key):
        self.events, self.api_key, self.pending = events, api_key, ""
        self.warning_lines = []

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.emit(line)
        return len(text)

    def emit(self, line):
        raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        line = raw.strip()
        if self.api_key:
            line = line.replace(self.api_key, "[redacted]")
        warning_header = "Pydantic serializer warnings" in line
        warning_value = "PydanticSerializationUnexpectedValue" in line
        serializer_source = bool(re.match(r"return self\.__pydantic_serializer__\.(to_python|to_json)\(", line))
        if warning_header:
            self.flush_warning()
        if warning_header or warning_value or serializer_source or (self.warning_lines and (not line or raw[:1].isspace())):
            self.warning_lines.append(line)
            if serializer_source:
                self.flush_warning()
            return
        self.flush_warning()
        # The controller reports this same record through result_callback.
        # Its machine-readable terminal export is not a conversational reply.
        if line.startswith(("AGENTIC_RUN_RESULT:", "[TOOL CALL]", "[TOOL RESULT]")):
            return
        if line:
            self.events.put(("message", line))

    def flush_warning(self):
        if self.warning_lines:
            warning = "\n".join(self.warning_lines)
            self.events.put(("diagnostic", {"type": "SDK serialization warning", "severity": "warning",
                                          "message": warning, "traceback": warning}))
            self.warning_lines = []

    def finish(self):
        self.emit(self.pending)
        self.pending = ""
        self.flush_warning()


def run(settings, request, events, answers):
    from aso_agents.gui.config import environment_for, api_error
    from aso_agents.gui.feedback import runner_feedback
    result_received = False
    try:
        # This process has its own environment. Preserve inherited CLI settings
        # and apply only explicit choices through existing CLI variables.
        os.environ.update(environment_for(settings))
        # Same no-argument runtime bootstrap as cli._run, before agent imports.
        from aso_agents.runtime_binaries import refresh_runtime_path
        refresh_runtime_path()
        os.environ.setdefault("MODAL_ASO_SKIP_RUNTIME_INSTALL", "1")
        from aso_agents.llm_provider import get_provider_config
        get_provider_config()
        import run_goal_intake as controller
        main = controller.main
        import inspect
        try:
            inspect.signature(main).bind(argv=[], clarification_resolver=None, result_callback=None)
        except TypeError:
            events.put(("error", "The GUI adapter is incompatible with this controller version. Update aso_agents/gui/worker.py; the analysis engine does not need a GUI-specific change."))
            return

        def clarify(*, questions, context):
            events.put(("questions", {"questions": questions, "context": context}))
            return answers.get()

        def result_callback(result):
            nonlocal result_received
            result_received = True
            events.put(("result", result))

        stream = ChatStream(events, settings.api_key)
        try:
            events.put(("activity", "Starting the analysis controller."))
            with redirect_stdout(stream), redirect_stderr(stream), runner_feedback(
                events, commentary_for_tool=getattr(controller, "_cli_message_for_tool", None)
            ):
                main(argv=["--", request], clarification_resolver=clarify,
                     result_callback=result_callback)
        finally:
            stream.finish()
    except SystemExit as exc:
        # Failed-run records already describe the cause. An exit code is not
        # a network exception, and must not replace that cause in the chat.
        if not result_received and exc.code not in (None, 0):
            events.put(("error", "The analysis controller stopped before returning a result. Review the preceding diagnostics."))
    except Exception as exc:
        from aso_agents.gui.diagnostics import diagnostic
        details = diagnostic(exc, settings.api_key)
        if getattr(exc, "status_code", None) is not None or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}:
            error = api_error(exc)
            message = f"Analysis stopped. {error.field}: {error}"
        else:
            message = f"The GUI worker encountered {details['type']}: {details['message']}\n\nOpen Technical details below for the traceback. Any returned results remain available."
        events.put(("error", message))
        events.put(("diagnostic", details))
    finally:
        events.put(("done", None))


def start(settings, request):
    context = multiprocessing.get_context("spawn")
    events, answers = context.Queue(), context.Queue()
    process = context.Process(target=run, args=(settings, request, events, answers), daemon=False)
    process.start()
    return process, events, answers
