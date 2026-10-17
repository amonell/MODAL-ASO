"""Schema retries explain the actual rejected nested object."""

from aso_agents.pipeline_tools import (
    REQUEST_WORKBENCH_OUTPUT_SCHEMA,
    _structured_output_retry_prompt,
)


def test_nested_task_rejection_shows_task_contract_not_only_root_keys():
    prompt = _structured_output_retry_prompt(
        "Public request", [
            "$.request_envelope.tasks[0]: Additional properties are not allowed ('chemistry' was unexpected)"
        ], REQUEST_WORKBENCH_OUTPUT_SCHEMA,
        {"status": "ready", "request_envelope": {"tasks": [{"chemistry": "full-PS"}]}},
    )
    assert '"path": "$.request_envelope.tasks[0]"' in prompt
    assert '"allowed_keys"' in prompt
    assert '"context"' in prompt
    assert "chemistry' was unexpected" in prompt
    assert "Public request" in prompt


def test_nested_design_program_rejection_shows_local_allowed_keys():
    prompt = _structured_output_retry_prompt(
        "Public request", [
            "$.request_envelope.tasks[0].context.structured_goal.design_programs[0]: Additional properties are not allowed ('constraints' was unexpected)"
        ], REQUEST_WORKBENCH_OUTPUT_SCHEMA,
    )
    assert '"path": "$.request_envelope.tasks[0].context.structured_goal.design_programs[0]"' in prompt
    assert '"program_id"' in prompt
    assert '"modality"' in prompt
    assert '"allowed_keys"' in prompt
    assert "correct that nested object" in prompt


def test_unknown_path_does_not_invent_nested_contract():
    prompt = _structured_output_retry_prompt(
        "Public request", ["$.not_in_schema[0]: invalid"],
        REQUEST_WORKBENCH_OUTPUT_SCHEMA,
    )
    assert "Contracts at the rejected paths" in prompt
    assert "Contracts at the rejected paths (not a replacement for the full schema):\n[]" in prompt
