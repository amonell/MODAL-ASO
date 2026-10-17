"""Quoted role quotas survive intake serialization without invented site counts."""
from copy import deepcopy
from types import SimpleNamespace
import asyncio
import json

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.role_selection import select_ordered_roles
from aso_agents.run_layout import resolve_run_path


QUOTE = "Choose one for Left, then one for Right, by ascending GC percentage; do not reuse sequences."
SITE_QUOTE = "Also retain at least two candidates for each physical site."
SITE_FIELDS = ("candidate_limit_per_site", "min_asos_per_site", "per_site_candidate_limit")


def role_goal():
    return {
        "candidate_minimum": 2, "candidate_limit": 2,
        "design_programs": [
            {"program_id": key, "label": label, "modality": "steric", "intent": "binding"}
            for key, label in (("left_pool", "Left"), ("right_pool", "Right"))
        ],
        "candidate_role_selection": {
            "reuse_policy": "distinct_sequences",
            "roles": [
                {"role_id": label, "program_id": key, "count": 1,
                 "criteria": [{"metric": "gc_percent", "direction": "minimize"}]}
                for key, label in (("left_pool", "Left"), ("right_pool", "Right"))
            ],
        },
    }


def accepted_audit(goal):
    return {"status": "accepted", "candidate_count": None, "per_site_count": None,
            "candidate_role_ranking": {"status": "applied", "quote": QUOTE,
                                       "contract": deepcopy(goal["candidate_role_selection"])}}


def nested_envelope(goal):
    stale = {**deepcopy(goal), **{key: 7 for key in SITE_FIELDS}}
    return {**deepcopy(stale), "structured_goal": deepcopy(stale),
            "default_context": {**deepcopy(stale), "structured_goal": deepcopy(stale)},
            "tasks": [{"task_type": "design_new_asos", **deepcopy(stale),
                       "structured_goal": deepcopy(stale),
                       "context": {**deepcopy(stale), "structured_goal": deepcopy(stale)}}]}


def goal_containers(envelope):
    task = envelope["tasks"][0]
    direct = [envelope, envelope["default_context"], task, task["context"]]
    return direct + [entry["structured_goal"] for entry in direct]


@pytest.mark.parametrize("validated", [False, True])
def test_persisted_role_audit_prevents_stale_counts_and_allows_selection(tmp_path, validated):
    goal = role_goal()
    if validated:
        goal["goal_schema_validation"] = {"ok": True}
    envelope = nested_envelope(goal)
    final = {"structured_goal": deepcopy(goal), "request_envelope": deepcopy(envelope),
             "request_obligation_audit": accepted_audit(goal),
             **{key: 7 for key in SITE_FIELDS}}
    payload = {"request": QUOTE, "request_envelope": deepcopy(envelope)}
    pipeline._save_goal_payload(str(tmp_path), payload, goal, final)
    saved = json.loads(resolve_run_path(tmp_path, "structured_goal.json").read_text())
    intake = pipeline._load_goal_intake_payload(str(tmp_path))
    for container in [saved, intake["final_payload"], *goal_containers(intake["request_envelope"])]:
        assert all(key not in container for key in SITE_FIELDS)
    assert saved["candidate_role_selection"] == role_goal()["candidate_role_selection"]
    assert (saved["candidate_minimum"], saved["candidate_limit"]) == (2, 2)
    rows = [{"Synthesis": seq, "IDT": "chemistry", "program_id": program,
             "binding_intervals_bed0_half_open": [["chr1", start, start + len(seq)]]}
            for seq, program, start in [("AATTCC", "left_pool", 0),
                                         ("AATTCC", "right_pool", 20),
                                         ("CCGGAT", "right_pool", 40)]]
    selected, receipt = select_ordered_roles(rows, saved,
        lambda row, **_: row["binding_intervals_bed0_half_open"])
    assert receipt["status"] == "complete"
    assert [row["Synthesis"] for row in selected] == ["AATTCC", "CCGGAT"]
    assert len(rows) == 3


@pytest.mark.parametrize("physical_scope", [None, "per_site_count", "candidate_count"])
def test_intake_audit_synchronizes_nested_goals_without_erasing_explicit_sites(monkeypatch, physical_scope):
    import run_goal_intake as intake
    goal = {**role_goal(), **{key: 7 for key in SITE_FIELDS}}
    contract = goal["candidate_role_selection"]
    role_audit = {"quote": QUOTE, "reuse_policy": contract["reuse_policy"],
                  "roles": [{key: deepcopy(value) for key, value in role.items() if key != "program_id"}
                            for role in contract["roles"]]}
    audit = {"candidate_role_ranking": role_audit, "candidate_panel_ranking": None}
    if physical_scope:
        audit[physical_scope] = {"count": 2, "kind": "minimum", "quote": SITE_QUOTE}
        if physical_scope == "candidate_count":
            audit[physical_scope]["scope"] = "per_site"
    async def run(*args, **kwargs):
        return SimpleNamespace(final_output=json.dumps(audit))
    monkeypatch.setattr(intake.Runner, "run", run)
    draft = {"status": "ready", "structured_goal": goal, "request_envelope": nested_envelope(goal)}
    result = asyncio.run(intake._audit_request_obligations(draft, QUOTE + " " + SITE_QUOTE, []))
    assert result["status"] == "ready"
    for container in [result, result["structured_goal"], *goal_containers(result["request_envelope"])]:
        if physical_scope:
            assert container.get("min_asos_per_site") == 2
        else:
            assert all(key not in container for key in SITE_FIELDS)
        assert container["candidate_role_selection"] == role_goal()["candidate_role_selection"]


@pytest.mark.parametrize("reason", ["no_audit", "unresolved_audit", "unapplied_roles",
                                   "different_contract", "explicit_sites", "per_site_candidate_count"])
def test_persistence_preserves_counts_without_applicable_authoritative_role_audit(tmp_path, reason):
    goal = {**role_goal(), **{key: 7 for key in SITE_FIELDS}}
    audit = accepted_audit(goal)
    if reason == "no_audit":
        audit = None
    elif reason == "unresolved_audit":
        audit["status"] = "unresolved"
    elif reason == "unapplied_roles":
        audit["candidate_role_ranking"]["status"] = "unresolved"
    elif reason == "different_contract":
        audit["candidate_role_ranking"]["contract"]["roles"][0]["count"] = 2
    elif reason == "explicit_sites":
        audit["per_site_count"] = {"count": 7, "kind": "exact", "quote": SITE_QUOTE}
    else:
        audit["candidate_count"] = {"scope": "per_site", "count": 7, "kind": "exact", "quote": SITE_QUOTE}
    envelope = nested_envelope(goal)
    pipeline._save_goal_payload(str(tmp_path), {"request": QUOTE, "request_envelope": envelope}, goal,
                               {"request_envelope": deepcopy(envelope), "request_obligation_audit": audit})
    saved = json.loads(resolve_run_path(tmp_path, "structured_goal.json").read_text())
    assert {key: saved[key] for key in SITE_FIELDS} == {key: 7 for key in SITE_FIELDS}
    assert saved["candidate_role_selection"] == role_goal()["candidate_role_selection"]
