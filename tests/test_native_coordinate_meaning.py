from copy import deepcopy

import pytest

from aso_agents.coordinate_contract import coordinate_execution_summary


def goal_for(row, *, convention="bed0_half_open", space="genomic"):
    return {"design_programs": [{"program_id": "branch", "label": "branch",
        "modality": "gapmer", "intent": "knockdown", "desired_delta": "decrease",
        "objective": "decrease the supplied target", "length": 20,
        "region_constraints": [{"source_type": "explicit_rows", "operator": "union",
            "coordinate_space": space, "coordinate_convention": convention,
            "aso_overlap_requirement": "full_containment",
            "intervals_bed0_half_open": [row]}]}]}


@pytest.mark.parametrize("start,end,strand,space", [
    (0, 1, "+", "genomic"), (10, 30, "-", "genomic"),
    (37, 116, "+", "cdna"), (100004, 100119, "-", "genomic"),
])
def test_typed_interval_meaning_preserves_covered_bases(start, end, strand, space):
    row = {"chrom": "TX.2" if space == "cdna" else "chr7",
           "start": start, "end": end, "strand": strand}
    # Source-coordinate metadata cannot reinterpret an explicitly typed BED field.
    goal = goal_for(row, convention="one_based_inclusive", space=space)
    before = deepcopy(goal)
    constraint = coordinate_execution_summary(goal)["constraints"][0]
    meaning = constraint["typed_interval_meaning"][0]
    assert meaning["status"] == "interpreted"
    assert meaning["reference"] == row["chrom"]
    assert meaning["strand"] == strand
    assert meaning["bed0_half_open"] == {"start": start, "end": end}
    assert meaning["one_based_inclusive"] == {"start": start + 1, "end": end}
    assert meaning["length_nt"] == end - start
    assert meaning["one_based_inclusive"]["end"] - meaning["one_based_inclusive"]["start"] + 1 == meaning["length_nt"]
    assert constraint["recorded_coordinate_convention"] == "one_based_inclusive"
    assert goal == before


@pytest.mark.parametrize("row", [
    {"chrom": "chr7", "start": 0, "end": 0},
    {"chrom": "chr7", "start": -1, "end": 5},
    {"chrom": "chr7", "start": True, "end": 5},
    {"chrom": "chr7", "start": "0", "end": 5},
    {"chrom": "chr7", "start": 0},
    {"start": 0, "end": 5},
])
def test_invalid_typed_intervals_are_not_given_invented_bounds(row):
    meaning = coordinate_execution_summary(goal_for(row))["constraints"][0]["typed_interval_meaning"][0]
    assert meaning["status"] == "invalid_typed_interval"
    assert "one_based_inclusive" not in meaning
    assert "length_nt" not in meaning


def test_untyped_rows_are_not_silently_interpreted_as_typed_bed():
    goal = goal_for({"chrom": "chr7", "start": 10, "end": 30})
    constraint = goal["design_programs"][0]["region_constraints"][0]
    constraint.pop("intervals_bed0_half_open")
    constraint["rows"] = [["chr7", 11, 30]]
    constraint["coordinate_convention"] = "one_based_inclusive"
    summary = coordinate_execution_summary(goal)["constraints"][0]
    assert summary["typed_interval_meaning"] == []
    assert summary["rows"] == [["chr7", 11, 30]]


def test_existing_post_repair_review_sees_actual_endpoint_and_length(monkeypatch):
    from aso_agents import pipeline_tools as p
    request = "Use 20-nt gapmers inside chr7:101-179, one-based inclusive, to decrease the target."
    goal = goal_for({"chrom": "chr7", "start": 100, "end": 179, "strand": "-"})
    envelope = {"schema_version": "v3", "request_text": request, "tasks": [{
        "task_id": "design", "task_type": "design_new_asos", "context": {
            "target_context": "reviewed cells", "structured_goal": goal}}]}
    original = deepcopy(envelope)
    pointer = "/tasks/0/context/structured_goal/design_programs/0/region_constraints/0/intervals_bed0_half_open/0/end"
    seen = []

    async def audit(prompt):
        payload = p._goal_audit_protocol_payload(prompt)
        seen.append(payload)
        meaning = payload["coordinate_execution"][0]["constraints"][0]["typed_interval_meaning"][0]
        assert payload["evidence_sources"][0]["text"] == request
        if len(seen) == 1:
            assert meaning["one_based_inclusive"] == {"start": 101, "end": 179}
            assert meaning["length_nt"] == 79
            value = 180  # Simulate a bad model repair; a reason is not evidence.
        elif len(seen) == 2:
            assert meaning["one_based_inclusive"] == {"start": 101, "end": 180}
            assert meaning["length_nt"] == 80
            change = next(x for x in payload["latest_repair_transition"]["changes"] if x["path"] == pointer)
            assert change["before"]["value"] == 179 and change["after"]["value"] == 180
            value = 179
        else:
            assert meaning["one_based_inclusive"] == {"start": 101, "end": 179}
            assert meaning["length_nt"] == 79
            return dict(status="valid", patches=[], goal_replacements=[], field_dispositions=[], unresolved=[])
        return dict(status="repairable", patches=[{
            "op": "replace", "path": pointer, "value": value, "evidence_refs": ["request"],
            "evidence_quote": request, "reason": "Preserve the requested inclusive endpoint."}],
            goal_replacements=[], field_dispositions=[], unresolved=[])

    monkeypatch.setattr(p, "_run_goal_contract_audit_agent_once", audit)
    result = p._repair_request_envelope_if_needed(request_text=request, run_dir="", structured_goal={},
        request_envelope=envelope, qa_history=[], semantic_audit=True)
    assert result["status"] == "ready", result
    # The existing loop also runs the region and coordinate focused audits.
    assert len(seen) == 5
    final = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert final["design_programs"][0]["region_constraints"][0]["intervals_bed0_half_open"][0]["end"] == 179
    assert envelope == original
