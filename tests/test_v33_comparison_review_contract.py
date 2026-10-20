"""The independent reviewer must see the native submitted-set capability."""

from aso_agents.pipeline_tools import _request_graph_operation_coverage_review_prompt


def test_operation_reviewer_receives_submitted_set_and_target_relative_boundaries():
    envelope = {
        "tasks": [
            {"task_id": "evaluate", "task_type": "evaluate_existing_aso_set", "context": {}},
            {"task_id": "compare", "task_type": "compare_aso_sets", "context": {},
             "depends_on": ["evaluate"]},
        ],
    }
    prompt = _request_graph_operation_coverage_review_prompt(
        "Compare two submitted ASO sets after transcript-relative evaluation.", envelope,
    )
    assert "compares distinct submitted physical ASO sets as well as generated sets" in prompt
    assert "it does not require newly designed molecules" in prompt
    assert '"task_type":"compare_aso_sets"' in prompt
