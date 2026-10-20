"""Child requests preserve their full inputs without repeating compiled goals."""
from copy import deepcopy
import json

import pytest

from aso_agents import pipeline_tools


def _decode_section(request, heading):
    start = request.index(heading) + len(heading)
    start += len(request[start:]) - len(request[start:].lstrip())
    return json.JSONDecoder().raw_decode(request, start)[0]


@pytest.mark.parametrize("modality,intent", [
    ("gapmer", "knockdown"),
    ("steric", "splice_switching"),
    ("steric", "translation_modulation"),
])
def test_child_request_preserves_complete_goal_once(modality, intent):
    goal = {
        "gene_symbol": "EXAMPLE", "species": "human",
        "intent": intent,
        "design_programs": [{
            "program_id": "program_from_parent", "modality": modality,
            "intent": intent, "aso_lengths": [17, 21, 25],
            "region_constraints": [{
                "source": "user_artifact", "artifact_ref": "/input/regions.bed",
                "overlap_policy": "contains_feature", "notes": "preserve_every_field",
            }],
        }],
        "artifact_refs": [{"path": "/input/regions.bed"}],
    }
    task = {"task_id": "design", "task_type": "design_new_asos",
            "label": "Scoped design", "context": {"structured_goal": goal},
            "inputs": [{"path": "/input/regions.bed"}],
            "outputs": [{"path": "/output/library.csv"}]}
    envelope = {"tasks": [task]}
    before = deepcopy(envelope)

    request = pipeline_tools._build_scoped_design_request_text(
        "Execute this one design.", task, envelope)
    decoded = _decode_section(request, "Scoped design task:\n")

    assert decoded["context"]["structured_goal"] == goal
    assert decoded["inputs"] == task["inputs"]
    assert decoded["outputs"] == task["outputs"]
    assert request.count('"program_id": "program_from_parent"') == 1
    assert request.count('"notes": "preserve_every_field"') == 1
    assert envelope == before


def test_child_request_retains_sources_dependencies_and_user_answers():
    goal = {"gene_symbol": "EXAMPLE", "species": "human",
            "intent": "knockdown", "target_scope": "preserve_scope"}
    task = {"task_id": "design", "task_type": "design_new_asos",
            "context": {"structured_goal": goal}}
    artifacts = {"entries": [{"path": "/input/source.json",
                              "excerpt": "Exact source text and unresolved limitations."}]}
    dependencies = [{"upstream_task_id": "read_reference",
                     "output_payload": {"coordinate_convention": "1-based inclusive",
                                        "limitations": ["finite supplied references"]}}]
    history = [{"question": "Which model should be used?",
                "answer": "Use the general reference baseline."}]

    request = pipeline_tools._build_scoped_design_request_text(
        "Design from the supplied data.", task, {"tasks": [task]}, history,
        dependencies, artifacts)

    assert _decode_section(request, "Scoped design task:\n")["context"]["structured_goal"] == goal
    assert _decode_section(request, "Indexed direct-artifact context (authoritative source-backed task input):\n") == artifacts
    assert _decode_section(request, "Completed upstream dependency results (authoritative source-backed execution inputs):\n") == dependencies
    assert history[0]["answer"] in request
    assert request.count('"target_scope": "preserve_scope"') == 1


def test_child_request_without_goal_remains_scoped():
    task = {"task_id": "design", "task_type": "design_new_asos",
            "context": {"gene_symbol": "EXAMPLE", "species": "human"}}
    request = pipeline_tools._build_scoped_design_request_text(
        "Execute this task.", task, {"tasks": [task]})

    decoded = _decode_section(request, "Scoped design task:\n")
    assert decoded["task_id"] == "design"
    assert decoded["context"]["gene_symbol"] == "EXAMPLE"
    assert "Run only this one design request." in request
    assert pipeline_tools._user_authored_request_text(request) == ""
