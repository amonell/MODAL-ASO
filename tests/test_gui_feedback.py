"""UI observation must leave request and response processing unchanged."""
from queue import Queue
from aso_agents.gui.feedback import runner_feedback


def test_progress_observation_does_not_patch_sdk_serialization():
    from openai._models import BaseModel as SDKModel
    from pydantic import BaseModel
    methods = (SDKModel.model_dump, SDKModel.model_dump_json, BaseModel.model_dump)
    with runner_feedback(Queue()):
        assert (SDKModel.model_dump, SDKModel.model_dump_json, BaseModel.model_dump) == methods
    assert (SDKModel.model_dump, SDKModel.model_dump_json, BaseModel.model_dump) == methods


def test_live_events_preserve_callback_and_do_not_expose_private_content():
    from agents import Runner
    original = Runner._emit_event
    events, received = Queue(), []
    payloads = [
        {"type": "tool_call", "tool": "tool_parse_goal_request", "arguments": {"private": "request body"}},
        {"type": "tool_result", "tool": "tool_parse_goal_request", "result": {"status": "ready", "secret": "not displayed"}},
        {"type": "final_output", "content": "private planner deliberation"},
    ]
    with runner_feedback(events):
        for payload in payloads:
            Runner._emit_event(received.append, payload)
    assert received == payloads
    assert all(a is b for a, b in zip(received, payloads))
    updates = [events.get_nowait()[1] for _ in payloads]
    assert updates == ["Starting: parse goal request.", "Finished: parse goal request.", "Model response received."]
    assert Runner._emit_event is original


def test_existing_cli_commentary_is_relayed_without_changing_events():
    from agents import Runner
    events, received, calls = Queue(), [], []
    def formatter(tool, count):
        calls.append((tool, count))
        return "I'm resolving the transcripts that matter for this request."
    payload = {"type": "tool_call", "tool": "tool_resolve_isoforms_and_sites", "arguments": {"secret": "private"}}
    with runner_feedback(events, commentary_for_tool=formatter):
        Runner._emit_event(received.append, payload)
        Runner._emit_event(received.append, payload)
    output = []
    while not events.empty():
        output.append(events.get_nowait())
    assert calls == [("tool_resolve_isoforms_and_sites", 1), ("tool_resolve_isoforms_and_sites", 2)]
    assert [item for item in output if item[0] == "message"] == [("message", "I'm resolving the transcripts that matter for this request.")]
    assert received == [payload, payload]
    assert "private" not in str(output)
