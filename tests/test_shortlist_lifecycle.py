import hashlib
import json

import pytest

from aso_agents import pipeline_tools as pipeline


def test_late_answer_preserves_science_history_and_resumes_only_report(tmp_path,monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    science = run / "unchanged_scored_tiles.json"
    science.write_text('{"rows":[{"computed":true}]}')
    before = hashlib.sha256(science.read_bytes()).hexdigest()
    state_path = tmp_path / "state.json"
    state = {"run_dir":str(run),"request":"Design candidates on the specified transcript.",
             "status":"gapmer_scoring_ready","phase":"gapmer_scoring_ready",
             "force_rerun_stages":[],"clarification_qa_history":[{"question":"Which reference?","answer":"Use the supplied reference."}]}
    state_path.write_text(json.dumps(state))
    goal = {"gene_symbol":"TEST", "species":"human", "intent":"knockdown"}
    context = {"payload":{},"structured_goal":goal,"final_payload":{"structured_goal":goal}}
    monkeypatch.setattr(pipeline,"_prepare_execution_stage",lambda **kwargs:{"status":"ok","state":pipeline._load_state(str(state_path)),"run_dir":str(run)})
    monkeypatch.setattr(pipeline,"_load_goal_context_from_run_dir",lambda _:context)
    monkeypatch.setattr(pipeline,"_collect_generated_candidate_rows",lambda *args:[{"Sequence":"ACGT"}])
    monkeypatch.setattr(pipeline,"_save_goal_payload",lambda *args:None)
    monkeypatch.setattr(pipeline,"_run_clarification_request_scope_sync",lambda *args:pytest.fail("Size answer must not launch a new model/intake pass"))
    question = pipeline.tool_build_report_payload(str(state_path))
    assert question["needs_user_input"] and question["clarification_kind"] == "shortlist_size"
    pipeline.submit_clarification_answers(str(state_path),["7"])
    result = pipeline.apply_clarification_to_goal(str(state_path))
    assert result["scientific_recomputation"] is False
    assert goal["candidate_limit"] == goal["candidate_minimum"] == 7
    saved = pipeline._load_state(str(state_path))
    assert saved["status"] == "shortlist_size_ready"
    assert saved["force_rerun_stages"] == []
    assert len(saved["clarification_qa_history"]) == 2
    assert saved["clarification_qa_history"][0]["answer"] == "Use the supplied reference."
    assert context["payload"]["qa_history"][-1]["answer"] == "7"
    assert context["payload"]["qa_history"][-1]["request_effect"] == "augment"
    assert pipeline.apply_clarification_to_goal(str(state_path))["already_applied"] is True
    assert len(pipeline._load_state(str(state_path))["clarification_qa_history"]) == 2
    next_actions = pipeline._derive_next_actions(saved)
    assert "tool_build_report_payload" in json.dumps(next_actions)
    assert "tool_run_steric_tiling" not in json.dumps(next_actions)
    assert hashlib.sha256(science.read_bytes()).hexdigest() == before


def test_refseq_program_resolves_exact_xref_before_structure_fetch(monkeypatch):
    import run_goal_intake as native
    import importlib
    chooser = importlib.import_module("aso_agents.transcript_chooser_agent")
    from aso_agents.declared_reference_catalog import complete_target_references
    from test_declared_reference_catalog import structure
    reference = "NM_123456.7"
    goal = {"gene_id":"ENSG00000000001","species":"human",
            "design_programs":[{"program_id":"one","transcript_id":reference}]}
    monkeypatch.setattr(chooser,"resolve_external_transcript_accession",lambda *args,**kwargs:{
        "transcript_id":"ENST00000000002.3","requested_accession":reference,
        "method":"local_gencode_refseq_xref"})
    seen=[]
    def fetch(identity):
        seen.append(identity)
        return structure(identity)
    monkeypatch.setattr(native,"_fetch_transcript_structure",fetch)
    assert complete_target_references(native,goal)
    assert seen == ["ENST00000000002.3"]
    record = goal["isoform_transcripts"][reference]
    assert record["transcript_id"] == "ENST00000000002.3"
    assert record["metadata"]["reference_accession_resolution"]["exact_accession_verified"] is True
    assert complete_target_references(native,goal)
    assert len(seen) == 1


@pytest.mark.parametrize("kind", ["shortlist_size", "supplied_reference_background"])
def test_parent_routes_selection_answer_to_only_the_question_owning_child(tmp_path, monkeypatch, kind):
    from aso_agents.shortlist_clarification import size_question
    question = size_question({},True)["questions"][0] if kind == "shortlist_size" else "Use this file as design background or annotation only?"
    answer = "6" if kind == "shortlist_size" else "design background"
    parent = {"run_dir":str(tmp_path), "current_task_id":"one", "status":"needs_user_input",
              "phase":"task_waiting_for_clarification", "task_artifacts":{},
              "task_validation":{},"task_statuses":{},"blocked_task_ids":["one","two"],
              "clarification_queue":[{"task_id":"one"},{"task_id":"two"}],
              "clarification":{"kind":kind,"pending_apply":True,
                               "last_answers":[{"question":question,"answer":answer}]}}
    parent["request_envelope"] = {"request_mode":"multi_task", "response_mode":"composite",
        "tasks":[{"task_id":task,"task_type":"design_new_asos","label":task,
                  "context":{"structured_goal":{"gene_symbol":"TEST", "species":"human", "intent":"knockdown",
                                                    "target_context":"general human baseline"}}}
                 for task in ("one","two")]}
    for task in ("one","two"):
        path = tmp_path/f"{task}.json"
        path.write_text(json.dumps({"clarification":{"kind":kind},"pending_questions":[question]}))
        parent["task_artifacts"][task] = {"child_state_path":str(path),"child_status":"needs_user_input",
                                          "child_pending_questions":[question]}
        parent["task_validation"][task] = {"status":"blocked","issues":["child_design_clarification_needed"]}
        parent["task_statuses"][task] = {"status":"blocked"}
    state_path = tmp_path/"parent.json"
    state_path.write_text(json.dumps(parent))
    monkeypatch.setattr(pipeline,"_load_goal_context_from_run_dir",lambda *args:pytest.fail("Child count cannot alter parent/global goal"))
    result = pipeline.apply_clarification_to_goal(str(state_path))
    assert result["resumed_child_task_ids"] == ["one"]
    saved = pipeline._load_state(str(state_path))
    assert saved["task_artifacts"]["one"]["child_resume_answers"][0]["answer"] == answer
    assert saved["task_artifacts"]["two"] == parent["task_artifacts"]["two"]
    assert saved["blocked_task_ids"] == ["two"]
    assert saved["clarification_qa_history"][-1]["task_id"] == "one"
    assert saved["status"] == "request_parsed_ready"


def test_child_cannot_borrow_a_sibling_shortlist_preference():
    seen = []
    def resolver(payload):
        seen.append(payload)
        return {"status":"unresolved","answers":[],"unresolved_questions":payload["child_questions"]}
    pipeline._resolve_child_clarification_from_inherited_context(parent_request="Two design panels",
        parent_qa_history=[{"task_id":"one","question":"How many?","answer":"6"},
                           {"question":"Which organism?","answer":"Human"}],
        task={"task_id":"two"},task_context={},child_questions=["How many?"],resolver=resolver)
    assert seen[0]["parent_clarification_history"] == [{"question":"Which organism?","answer":"Human"}]


def test_unverified_refseq_version_never_falls_through_to_ensembl_structure(monkeypatch):
    import run_goal_intake as native
    import importlib
    chooser = importlib.import_module("aso_agents.transcript_chooser_agent")
    from aso_agents.declared_reference_catalog import complete_target_references
    goal = {"species":"human","design_programs":[{"transcript_id":"NM_123456.7"}]}
    monkeypatch.setattr(chooser,"resolve_external_transcript_accession",lambda *args,**kwargs:{"transcript_id":"ENST00000000002"})
    monkeypatch.setattr(chooser,"_get_ensembl_json",lambda *args:(None,[{"display_id":"NM_123456.8"}]))
    monkeypatch.setattr(native,"_fetch_transcript_structure",lambda *args:pytest.fail("Unverified version reached structure lookup"))
    with pytest.raises(ValueError,match="exact_accession_unverified"):
        complete_target_references(native,goal)


def test_native_intron_plan_and_regions_use_both_actual_boundaries():
    import run_goal_intake as native
    from test_intron_events import inputs
    program,catalog = inputs()
    for index, exon in enumerate(catalog["ref"]["exons_tx"], 1):
        exon.update(index=index, contains_coding=True)
    catalog["ref"]["sites"] = native._extract_splice_sites({"exons_tx":catalog["ref"]["exons_tx"],"seq_region_name":"7","strand":1})
    goal = {"gene_symbol":"TEST","species":"human","intent":"splice_modulation",
            "design_programs":[program],"isoform_transcripts":catalog}
    plan,_ = native._compute_splice_site_plan([],catalog,goal)
    goal["splice_site_plan"] = plan
    selected = native._collect_program_splice_plan_entries(goal,program)
    assert {(row["site"],row["position"]) for row in selected} == {("donor",150),("acceptor",301)}
    assert {row["delta_direction"] for row in selected} == {"decrease"}
    rows,meta = native._build_steric_program_site_window_rows(goal,program)
    assert rows and len(rows) == 2
