"""Scoped UI adapters. Restore every hook when the GUI worker finishes."""
from contextlib import contextmanager


def activity_message(event):
    kind = event.get("type")
    tool = str(event.get("tool") or "analysis step").removeprefix("tool_").replace("_", " ")
    if kind == "tool_call":
        return f"Starting: {tool}."
    if kind == "tool_result":
        result = event.get("result")
        failed = isinstance(result, dict) and (bool(result.get("error")) or result.get("status") in {"failed", "fatal", "error"})
        return f"Finished: {tool}" + (" — an error was reported." if failed else ".")
    if kind == "final_output":
        # Internal planner output can contain deliberation or machine JSON.
        # Report the event, never expose the raw final_output content here.
        return "Model response received."
    return None


@contextmanager
def runner_feedback(events, commentary_for_tool=None):
    """Observe existing events while forwarding original callbacks unchanged."""
    from agents import Runner
    original = Runner._emit_event
    counts = {}
    last_commentary = None

    def emit(callback, event):
        nonlocal last_commentary
        try:
            if event.get("type") == "tool_call" and callable(commentary_for_tool):
                tool = str(event.get("tool") or "")
                counts[tool] = counts.get(tool, 0) + 1
                commentary = commentary_for_tool(tool, counts[tool])
                if commentary and commentary != last_commentary:
                    events.put(("message", commentary))
                    last_commentary = commentary
            message = activity_message(event)
            if message:
                events.put(("activity", message))
        except Exception:
            pass  # Display feedback must not change execution.
        return original(callback, event)

    Runner._emit_event = staticmethod(emit)
    try:
        yield
    finally:
        Runner._emit_event = staticmethod(original)
