from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import run_goal_intake as native
from aso_agents import pipeline_tools as pipeline
from aso_agents.supplied_reference_role import role_question, resolved_roles, apply_role_answers


def annotation(path, variants=True):
    return {"snapgene_path": str(path), "sequence_sha256": "sequence-identity",
            "variant_derivation_attempted": True,
            "patient_variants": [{"chrom": "chr2", "pos": 123, "ref": "A", "alt": "G"}] if variants else []}


def test_default_false_does_not_silently_discard_detected_differences(tmp_path):
    goal = {"use_supplied_sequence_variants": False}
    question = role_question(goal, [annotation(tmp_path/"custom.gb")], "Increase this supplied isoform")
    assert question["kind"] == "supplied_reference_background"
    assert len(question["questions"]) == 1
    assert not role_question({}, [annotation(tmp_path/"normal.gb", False)], "Increase this isoform")


@pytest.mark.parametrize("quote,asked", [("Annotations only please", False), ("invented annotation choice", True)])
def test_explicit_annotation_only_needs_actual_user_evidence(tmp_path, quote, asked):
    path = tmp_path/"custom.gb"
    goal = {"supplied_sequence_roles": [{"artifact_ref_path": str(path), "role": "annotation_only", "source_quote": quote}]}
    assert bool(role_question(goal, [annotation(path)], "Annotations only please")) is asked


def test_legacy_explicit_positive_background_remains_supported(tmp_path):
    assert not role_question({"use_supplied_sequence_variants": True}, [annotation(tmp_path/"custom.gb")])


def test_multiple_files_keep_independent_roles_and_sequence_identity(tmp_path):
    annotations = [annotation(tmp_path/"one.gb"), annotation(tmp_path/"two.gb")]
    goal = {}
    question = role_question(goal, annotations)
    answers = [{"question": q, "answer": a} for q, a in zip(question["questions"], ["design background", "annotation only"])]
    apply_role_answers(goal, answers)
    assert list(resolved_roles(goal, annotations, "design background\nannotation only").values()) == ["design_background", "annotation_only"]
    assert role_question(goal, annotations, "design background\nannotation only") is None
    annotations[0]["sequence_sha256"] = "changed"
    goal["use_supplied_sequence_variants"] = True
    assert len(role_question(goal, annotations, "design background\nannotation only")["questions"]) == 1


def test_unclear_answer_cannot_partially_apply_other_files(tmp_path):
    goal = {}
    question = role_question(goal, [annotation(tmp_path/"one.gb"), annotation(tmp_path/"two.gb")])
    before = deepcopy(goal)
    with pytest.raises(ValueError, match="ambiguous"):
        apply_role_answers(goal, [{"question": q, "answer": a} for q, a in zip(question["questions"], ["design background", "yes"])])
    assert goal == before


def prepare_fixture(tmp_path, monkeypatch):
    path = tmp_path/"construct.gb"
    path.write_text("supplied file")
    goal = {"species": "human", "gene_symbol": "NEUTRAL", "use_supplied_sequence_variants": False,
            "evaluation_targets": [{"target_type": "isoform", "objective_type": "primary_outcome", "label": "custom",
                                    "desired_delta": "increase", "artifact_ref_path": str(path)}]}
    calls, builds = [], []
    def ingest(path, **kwargs):
        calls.append(kwargs)
        result = annotation(path)
        result["sequence_sha256"] = native.hashlib.sha256(native.Path(path).read_bytes()).hexdigest()
        return result
    monkeypatch.setattr(native, "ingest_snapgene_file", ingest)
    monkeypatch.setattr(native, "_build_snapgene_transcript_entry", lambda a,l: {"transcript_id": "SNAPGENE::custom", "exons_tx": [{"start": 10, "end": 30}], "chrom": "chr2", "strand": "+"})
    monkeypatch.setattr(native, "_apply_patient_reference_from_snapgene_annotations", lambda g,f,r,a: builds.append(deepcopy(a)))
    return path, goal, calls, builds


def test_discovery_pauses_before_background_build_and_resume_reuses_annotation(tmp_path, monkeypatch):
    path, goal, calls, builds = prepare_fixture(tmp_path, monkeypatch)
    first = native._prepare_snapgene_objectives(goal, {}, str(tmp_path))
    assert first["clarification_request"] and not builds
    assert calls[0]["derive_patient_variants"] is True
    apply_role_answers(goal, [{"question": first["clarification_request"]["questions"][0], "answer": "design background"}])
    second = native._prepare_snapgene_objectives(goal, {}, str(tmp_path), user_request="design background")
    assert not second["clarification_request"] and len(builds) == 1 and len(calls) == 1
    assert builds[0][0]["patient_variants"]
    path.write_text("changed supplied file")
    third = native._prepare_snapgene_objectives(goal, {}, str(tmp_path), user_request="design background")
    assert third["clarification_request"] and len(calls) == 2 and len(builds) == 1


def test_annotation_only_does_not_build_background(tmp_path, monkeypatch):
    _, goal, calls, builds = prepare_fixture(tmp_path, monkeypatch)
    first = native._prepare_snapgene_objectives(goal, {}, str(tmp_path))
    apply_role_answers(goal, [{"question": first["clarification_request"]["questions"][0], "answer": "annotation only"}])
    assert not native._prepare_snapgene_objectives(goal, {}, str(tmp_path), user_request="annotation only")["clarification_request"]
    assert not builds and len(calls) == 1
    assert goal["snapgene_annotation"]["patient_variants"]


def test_native_resolution_guard_and_answer_restart(tmp_path, monkeypatch):
    path, goal, calls, builds = prepare_fixture(tmp_path, monkeypatch)
    state_path = tmp_path/"state.json"
    state = {"run_dir": str(tmp_path), "request": "Increase the supplied isoform", "status": "goal_schema_ready", "phase": "goal_schema_ready", "force_rerun_stages": []}
    state_path.write_text(json.dumps(state))
    context = {"payload": {}, "structured_goal": goal, "final_payload": {"structured_goal": goal}}
    monkeypatch.setattr(pipeline, "provider_credentials_available", lambda: False)
    monkeypatch.setattr(pipeline, "_prepare_execution_stage", lambda **kw: {"status": "ok", "state": pipeline._load_state(str(state_path)), "run_dir": str(tmp_path)})
    monkeypatch.setattr(pipeline, "_load_goal_context_from_run_dir", lambda _: context)
    monkeypatch.setattr(pipeline, "_save_goal_payload", lambda *args: None)
    monkeypatch.setattr(pipeline, "_goal_runtime_module", lambda: SimpleNamespace(_prepare_snapgene_objectives=native._prepare_snapgene_objectives))
    result = pipeline.tool_resolve_isoforms_and_sites(str(state_path))
    assert result["needs_user_input"] and not builds
    assert pipeline._derive_next_actions(pipeline._load_state(str(state_path)))["clarification_request"]["kind"] == "supplied_reference_background"
    pipeline.submit_clarification_answers(str(state_path), ["yes"])
    assert pipeline.apply_clarification_to_goal(str(state_path))["needs_user_input"]
    pipeline.submit_clarification_answers(str(state_path), ["design background"])
    result = pipeline.apply_clarification_to_goal(str(state_path))
    assert result["restart_stage"] == "resolution"
    saved = pipeline._load_state(str(state_path))
    assert saved["request"] == state["request"]
    assert saved["force_rerun_stages"] == ["resolution"]
    assert saved["clarification_qa_history"][-1]["kind"] == "supplied_reference_background"
    assert goal["supplied_sequence_roles"][0]["role"] == "design_background"
    assert pipeline.apply_clarification_to_goal(str(state_path))["already_applied"] is True
