"""Available reference data, not molecule metadata, sets analysis boundaries."""

import asyncio
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline


@pytest.fixture
def supplied_analysis(tmp_path):
    molecules = tmp_path / "observations.csv"
    molecules.write_text("id,sequence\none,ACGTTAGC\n")
    reference = tmp_path / "reference.fa"
    reference.write_text(">ENST00000000001.7\nGCTAACGTACGTTAGC\n")
    coordinates = tmp_path / "coordinates.csv"
    coordinates.write_text("start1,end1,genomic_start1,strand\n1,16,101,-\n")
    return {
        "status": "ready", "questions": [], "reason": None,
        "request_envelope": {
            "tasks": [{
                "task_id": "audit", "task_type": "sandbox_analysis", "native": False,
                "context": {
                    "provided_aso_files": [{"path": str(molecules)}],
                    "transcript_id": "ENST00000000001.7",
                    "artifact_refs": [{"path": str(reference)}, {"path": str(coordinates)}],
                    "summary": "Search the supplied finite reference and project all matches.",
                },
            }],
        },
    }


@pytest.mark.parametrize("molecule_key", ["provided_asos", "provided_aso_files"])
@pytest.mark.parametrize("target_metadata", [
    {"target_context": "match against the supplied finite reference"},
    {"transcript_id": "ENST00000000001.7"},
    {"evaluation_targets": [{"transcript_id": "ENST00000000001.7"}]},
])
def test_available_reference_analysis_accepts_target_metadata(
    supplied_analysis, molecule_key, target_metadata,
):
    envelope = supplied_analysis["request_envelope"]
    context = envelope["tasks"][0]["context"]
    files = context.pop("provided_aso_files")
    context.pop("transcript_id")
    context[molecule_key] = (
        [{"name": "one", "sequence": "ACGTTAGC"}]
        if molecule_key == "provided_asos" else files
    )
    context.update(deepcopy(target_metadata))
    assert pipeline._executor_capability_issues(
        envelope, request_evidence="Audit ENST00000000001.7 using the supplied files.",
    ) == []


@pytest.mark.parametrize("required", [True, False])
def test_explicit_external_retrieval_boundary_is_preserved(supplied_analysis, required):
    envelope = supplied_analysis["request_envelope"]
    envelope["tasks"][0]["context"]["web_retrieval"] = {"required": required}
    issues = pipeline._executor_capability_issues(
        envelope, request_evidence="Audit ENST00000000001.7.",
    )
    assert [issue["category"] for issue in issues] == (["wrong_task_type"] if required else [])


@pytest.mark.parametrize("review_error", [None, "unresolved_reference"])
def test_supplied_reference_plan_still_requires_both_independent_reviews(
    supplied_analysis, monkeypatch, review_error,
):
    proposals, reviews, finalized = [], [], []

    async def plan(prompt):
        proposals.append(prompt)
        return deepcopy(supplied_analysis)

    async def clarification(_prompt, proposed):
        return proposed

    async def review(prompt):
        reviews.append(prompt)
        if review_error:
            return {"error": review_error, "reason": "Required reference not resolved."}
        return {"status": "valid", "issues": []}

    async def finalize(_prompt, proposed):
        finalized.append(proposed)
        return proposed

    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", clarification)
    monkeypatch.setattr(pipeline, "_run_request_graph_review_agent_once", review)
    monkeypatch.setattr(pipeline, "_finalize_verified_request_plan", finalize)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once(
        "Audit ENST00000000001.7 using the supplied finite reference and coordinate map.",
    ))
    assert len(proposals) == 1
    if review_error:
        assert result["error"] == review_error
        assert len(reviews) == 1
        assert finalized == []
    else:
        assert result == supplied_analysis
        assert len(reviews) == 2
        assert finalized == [result]

