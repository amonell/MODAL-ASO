from copy import deepcopy

import pytest

from aso_agents import pipeline_tools


@pytest.mark.parametrize("field,correct,incorrect", [
    ("intent", "knockdown", "splice_modulation"),
    ("length", 18, 24),
])
def test_existing_verification_sees_actual_repair_not_just_claim(
    monkeypatch, field, correct, incorrect,
):
    request = "Use an 18-nt gapmer to decrease the supplied target in reviewed cells."
    program = dict(program_id="branch", label="branch", modality="gapmer",
                   intent="knockdown", objective="decrease the supplied target",
                   desired_delta="decrease", length=18)
    envelope = {"schema_version": "v3", "request_text": request, "tasks": [{
        "task_id": "design", "task_type": "design_new_asos", "context": {
            "target_context": "reviewed cells", "structured_goal": {
                "target_context": "reviewed cells", "design_programs": [program],
            },
        },
    }]}
    original = deepcopy(envelope)
    pointer = "/tasks/0/context/structured_goal/design_programs/0/" + field
    reason = "Preserve the requested value: " + str(correct)
    seen = []

    async def audit(prompt):
        payload = pipeline_tools._goal_audit_protocol_payload(prompt)
        seen.append(payload)
        if len(seen) == 1:
            assert not payload.get("latest_repair_transition")
            value = incorrect
        else:
            transition = payload.get("latest_repair_transition")
            assert transition, "Existing verification needs actual before/after repair evidence"
            change = next(c for c in transition["changes"] if c["path"] == pointer)
            assert change["before"] == {"present": True, "value": correct if len(seen) == 2 else incorrect}
            assert change["after"] == {"present": True, "value": incorrect if len(seen) == 2 else correct}
            assert transition["applied_repairs"][0]["reason"] == reason
            assert transition["applied_repairs"][0]["evidence_quote"] in request
            assert len(transition["applied_repairs"]) == 1
            if len(seen) == 3:
                return dict(status="valid", patches=[], goal_replacements=[],
                            field_dispositions=[], unresolved=[])
            value = correct
        patches = [{
            "op": "replace", "path": pointer, "value": value,
            "evidence_refs": ["request"], "evidence_quote": request,
            "reason": reason,
        }]
        if len(seen) == 1:
            # Neither a rejected identity edit nor an ineffective reassertion
            # may be presented as an accepted change to the next reviewer.
            patches.extend([
                dict(patches[0], path="/request_text", value="unsupported request"),
                dict(patches[0], path=pointer.rsplit("/", 1)[0] + "/label", value="branch"),
            ])
        return dict(status="repairable", patches=patches,
                    goal_replacements=[], field_dispositions=[], unresolved=[])

    monkeypatch.setattr(pipeline_tools, "_run_goal_contract_audit_agent_once", audit)
    result = pipeline_tools._repair_request_envelope_if_needed(
        request_text=request, run_dir="", structured_goal={},
        request_envelope=envelope, qa_history=[], semantic_audit=True,
    )
    assert result["status"] == "ready", result
    assert len(seen) == 3  # The existing apply/verify loop, with no added audit call.
    assert result["request_envelope"]["tasks"][0]["context"]["structured_goal"]["design_programs"][0][field] == correct
    assert envelope == original


def test_changes_preserve_presence_types_and_pointer_escaping():
    from aso_agents.goal_repair_verification import envelope_changes

    before = {"a/b~c": None, "keep": False, "number": 0, "removed": None}
    after = {"a/b~c": False, "keep": False, "number": False, "added": None}
    changes = {c["path"]: c for c in envelope_changes(before, after)}
    assert set(changes) == {"/a~1b~0c", "/number", "/removed", "/added"}
    assert changes["/removed"]["before"] == {"present": True, "value": None}
    assert changes["/removed"]["after"] == {"present": False}
    assert changes["/added"]["before"] == {"present": False}
    assert changes["/added"]["after"] == {"present": True, "value": None}
    assert changes["/number"]["before"]["value"] is not False
    assert changes["/number"]["after"]["value"] is False


@pytest.mark.parametrize("after", [[{"label": "second"}],
    [{"label": "first"}, {"label": "second"}, {"label": "third"}]],
)
def test_resized_arrays_report_collection_without_false_element_attribution(after):
    from aso_agents.goal_repair_verification import envelope_changes

    before = {"items": [{"label": "first"}, {"label": "second"}]}
    result = envelope_changes(before, {"items": after})
    assert result == [{"path": "/items", "before": {"present": True, "value": before["items"]},
                       "after": {"present": True, "value": after}}]
    after[0]["label"] = "mutated later"
    assert result[0]["after"]["value"][0]["label"] != "mutated later"


def test_same_length_arrays_report_actual_changes_and_relocation():
    from aso_agents.goal_repair_verification import envelope_changes

    before = {"tasks": [{"old": {"count": 3}, "stable": "keep"}]}
    after = {"tasks": [{"context": {"count": 3}, "stable": "keep"}]}
    result = {c["path"]: c for c in envelope_changes(before, after)}
    assert set(result) == {"/tasks/0/old", "/tasks/0/context"}
    assert result["/tasks/0/old"]["after"] == {"present": False}
    assert result["/tasks/0/context"]["after"] == {"present": True, "value": {"count": 3}}
    assert envelope_changes(after, deepcopy(after)) == []
