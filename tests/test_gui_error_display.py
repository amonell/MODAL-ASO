"""Backend validation syntax must remain literal in chat."""
from aso_agents.gui.presentation import literal_block, result_message


def test_json_schema_error_paths_are_inside_literal_code_block():
    detail = "invalid_structured_output: $: 'status' is a required property; $: '$schema' was unexpected"
    message = result_message({"status": "failed", "failed_stage": "intake", "error": {"message": detail}})
    assert "```text\n" + detail + "\n```" in message
    assert "required data format" in message


def test_backticks_in_error_cannot_break_out_of_code_block():
    text = "Unexpected ``` and $schema"
    assert literal_block(text) == "````text\n" + text + "\n````"
