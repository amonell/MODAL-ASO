"""Format shared runner events for the UI without including request data."""


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
