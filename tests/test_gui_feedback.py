"""The shared events can be described without including their request data."""
from aso_agents.gui.feedback import activity_message


def test_events_keep_private_content_out_of_ui_messages():
    assert activity_message({
        "type": "tool_call", "tool": "tool_parse_goal_request", "arguments": {"private": "request body"},
    }) == "Starting: parse goal request."
    assert activity_message({
        "type": "tool_result", "tool": "tool_parse_goal_request", "result": {"status": "ready", "secret": "not displayed"},
    }) == "Finished: parse goal request."
    assert activity_message({"type": "final_output", "content": "private data"}) == "Model response received."
