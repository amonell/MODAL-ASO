"""Process-isolated bridge to the unchanged CLI controller."""
import multiprocessing
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from aso_agents.gui.presentation import transport_error_message


def plain_terminal_text(text):
    """Drop ANSI CSI control sequences while preserving their surrounding text."""
    parts = text.split("\x1b[")
    result = [parts[0]]
    for part in parts[1:]:
        end = next((i for i, character in enumerate(part) if "@" <= character <= "~"), None)
        if end is not None:
            result.append(part[end + 1:])
    return "".join(result)


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
        raw = plain_terminal_text(line)
        line = raw.strip()
        if self.api_key:
            line = line.replace(self.api_key, "[redacted]")
        warning_header = "Pydantic serializer warnings" in line
        warning_value = "PydanticSerializationUnexpectedValue" in line
        serializer_source = line.startswith(("return self.__pydantic_serializer__.to_python(",
                                             "return self.__pydantic_serializer__.to_json("))
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
        # The GUI owns run timing. Suppress the terminal's periodic heartbeat.
        if line.startswith("Still running: ") and line.endswith(" elapsed)."):
            return
        if line:
            self.events.put(("message", transport_error_message(line) or line))

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


def run(settings, request, events, answers, options=()):
    from aso_agents.cli import _run
    from aso_agents.gui.config import environment_for, api_error, credential_name
    from aso_agents.gui.feedback import activity_message
    result_received = False
    api_key = settings.api_key.strip() or os.environ.get(credential_name(settings.provider), "").strip()
    try:
        os.environ.update(environment_for(settings))

        def clarify(*, questions, context, can_pause=True):
            events.put(("questions", {"questions": questions, "context": context, "can_pause": can_pause}))
            return answers.get()

        def result_callback(result):
            nonlocal result_received
            result_received = True
            events.put(("result", result))

        offered_installations = set()

        def event_callback(event):
            if event.get("type") == "model_installation":
                model = str(event.get("model") or "the model")
                installation = (model, event.get("root"))
                if installation in offered_installations or "--raw-cli" in options:
                    return False
                offered_installations.add(installation)
                responses = clarify(questions=[f"Install {model} now? Answer yes or no."],
                                    context={"why_this_matters": event.get("message", "")}, can_pause=False)
                return bool(responses and str(responses[0].get("answer", "")).strip().lower()
                            in {"yes", "y"})
            if message := activity_message(event):
                events.put(("activity", message))

        stream = ChatStream(events, api_key)
        argv = [*options, "--", request] if request else list(options)
        try:
            events.put(("activity", "Starting the analysis controller."))
            with redirect_stdout(stream), redirect_stderr(stream):
                _run(argv, clarification_resolver=clarify, result_callback=result_callback,
                     event_callback=event_callback)
        finally:
            stream.finish()
    except SystemExit as exc:
        # Failed-run records already describe the cause. An exit code is not
        # a network exception, and must not replace that cause in the chat.
        if not result_received and exc.code not in (None, 0):
            from aso_agents.gui.diagnostics import redact_credentials
            detail = exc.code if isinstance(exc.code, str) else "The analysis stopped before returning a result. See the preceding messages."
            detail = redact_credentials(detail, api_key)
            events.put(("error", transport_error_message(detail) or detail))
    except Exception as exc:
        from aso_agents.gui.diagnostics import diagnostic
        details = diagnostic(exc, api_key)
        if getattr(exc, "status_code", None) is not None or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}:
            error = api_error(exc)
            message = f"Analysis stopped. {error.field}: {error}"
        elif isinstance(exc, (OSError, ValueError, RuntimeError)):
            # The terminal entry point displays these shared execution errors directly.
            message = transport_error_message(details["message"]) or details["message"]
        else:
            message = f"Analysis stopped with {details['type']}: {details['message']}\n\nOpen Technical details for the traceback."
        events.put(("error", message))
        events.put(("diagnostic", details))
    finally:
        events.put(("done", None))


def start(settings, request, options=()):
    context = multiprocessing.get_context("spawn")
    events, answers = context.Queue(), context.Queue()
    process = context.Process(target=run, args=(settings, request, events, answers, tuple(options)), daemon=False)
    process.start()
    return process, events, answers
