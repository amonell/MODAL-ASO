from aso_agents.pipeline_tools import _coalesce_goal_audit_patches


def test_identical_audit_writes_are_applied_once():
    write = {"op": "add", "path": "/tasks/0/context/structured_goal/constraints/gc_content",
             "value": {"min": 0.35, "max": 0.65}, "evidence_refs": ["request"]}
    result, rejected = _coalesce_goal_audit_patches([write] * 400)
    assert result == [write]
    assert rejected == []


def test_conflicting_audit_writes_do_not_silently_choose_a_value():
    first = {"op": "add", "path": "/tasks/0/context/structured_goal/constraints/gc_content",
             "value": {"min": 0, "max": 1}}
    second = {**first, "value": {"min": 0.35, "max": 0.65}}
    independent = {"op": "add", "path": "/tasks/0/context/structured_goal/chemistry",
                   "value": "fully 2'-MOE"}
    result, rejected = _coalesce_goal_audit_patches([first, second, independent])
    assert result == [independent]
    assert rejected == [{"path": first["path"],
                         "reason": "conflicting_operations_for_same_destination",
                         "proposal_count": 2}]


def test_distinct_array_slots_remain_distinct_operations():
    patches = [{"op": "remove", "path": "/tasks/0/rows/0"},
               {"op": "remove", "path": "/tasks/0/rows/1"}]
    result, rejected = _coalesce_goal_audit_patches(patches)
    assert result == patches and not rejected


def test_identical_value_prefers_a_verifiable_evidence_citation():
    value = {"op": "add", "path": "/tasks/0/context/structured_goal/chemistry",
             "value": "full PS"}
    unverified = {**value, "evidence_refs": ["request"], "evidence_quote": "unquoted"}
    verified = {**value, "evidence_refs": ["request"], "evidence_quote": "full PS"}
    result, rejected = _coalesce_goal_audit_patches(
        [unverified, verified], evidence_sources=[{"evidence_id": "request", "text": "Use full PS chemistry"}])
    assert result == [verified] and not rejected
