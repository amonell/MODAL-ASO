"""Redacted GUI-worker diagnostics; never capture local variable values."""
import traceback


def redact_credentials(text, api_key):
    if api_key:
        text = text.replace(api_key, "[redacted]")
    result = []
    i = 0
    while i < len(text):
        boundary = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        if boundary and text[i:i + 6].lower() == "bearer" and text[i + 6:i + 7].isspace():
            j = i + 6
            while j < len(text) and text[j].isspace():
                j += 1
            end = j
            while end < len(text) and not text[end].isspace() and text[end] not in "\"'<>":
                end += 1
            result.append(text[i:j] + "[redacted]")
            i = end
        elif boundary and text.startswith("sk-", i):
            end = i + 3
            while end < len(text) and (text[end].isascii() and text[end].isalnum() or text[end] in "_-"):
                end += 1
            result.append("[redacted]")
            i = end
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def diagnostic(exc, api_key):
    def redact(value):
        return redact_credentials(str(value), api_key)

    return {"type": type(exc).__name__,
            "message": redact(exc)[:2000] or "No exception message was provided.",
            "traceback": redact("".join(traceback.TracebackException.from_exception(
                exc, capture_locals=False).format()))}
