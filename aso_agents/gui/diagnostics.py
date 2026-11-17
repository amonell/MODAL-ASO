"""Redacted GUI-worker diagnostics; never capture local variable values."""
import re
import traceback


def diagnostic(exc, api_key):
    def redact(value):
        text = str(value)
        if api_key:
            text = text.replace(api_key, "[redacted]")
        text = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [redacted]", text)
        text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", text)
        return text

    return {"type": type(exc).__name__,
            "message": redact(exc)[:2000] or "No exception message was provided.",
            "traceback": redact("".join(traceback.TracebackException.from_exception(
                exc, capture_locals=False).format()))}
