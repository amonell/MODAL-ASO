"""Repairs must see their actual input and be able to reach the owning boundary."""
import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
import pytest

from aso_agents import pipeline_tools as pipeline


def design_goal():
    return {"gene_symbol": "NEUTRAL", "species": "human", "target_context": "cultured cells",
            "intent": "knockdown", "candidate_minimum": 0, "candidate_limit": 5,
            "species_targets": [{"species": "human"}, {"species": "mouse"}],
            "design_programs": [{"program_id": "p", "label": "p", "modality": "gapmer",
                                 "intent": "knockdown", "objective": "reduce RNA", "desired_delta": "decrease"}]}


def envelope(goal=None):
    context = {"species": "human and mouse", "target_context": "cultured cells"}
    if goal is not None:
        context["structured_goal"] = goal
    return {"schema_version": "v3", "request_mode": "single_task", "request_summary": "Conserved design",
            "request_classifications": ["design_new_asos"],
            "tasks": [{"task_id": "design", "task_type": "design_new_asos", "label": "Design",
                       "native": True, "can_run_in_parallel": False, "context": context}],
            "task_dependencies": [], "final_outputs": ["Candidate panel"],
            "response_mode": "report", "requires_execution": True, "assumptions": []}


def test_semantic_audit_add_to_existing_leaf_uses_json_patch_replace_semantics():
    goal = design_goal()
    goal["design_programs"][0]["transcript_id"] = "ENST000001.1"
    protocol = {
        "current_request_envelope": envelope(goal),
        "design_goal_locations": [{"json_pointer": "/tasks/0/context/structured_goal"}],
    }
    existing = "/tasks/0/context/structured_goal/design_programs/0/transcript_id"
    missing = "/tasks/0/context/structured_goal/design_programs/0/exon_label"
    audit = {"patches": [
        {"op": "add", "path": existing, "value": "ENST000002.1"},
        {"op": "add", "path": missing, "value": 2},
    ]}
    result = pipeline._canonicalize_existing_member_additions(audit, protocol)
    assert [patch["op"] for patch in result["patches"]] == ["replace", "add"]
    assert audit["patches"][0]["op"] == "add"
    protocol["canonical_contract_issues"] = [{"json_pointer": existing}]
    assert pipeline._canonicalize_existing_member_additions(audit, protocol) == audit


def test_semantic_audit_replace_of_absent_optional_member_uses_add():
    from jsonschema import Draft202012Validator

    goal = design_goal()
    goal['design_programs'][0]['region_constraints'] = [{
        'constraint_id': 'site_window', 'source_type': 'splice_relative_window',
        'aso_overlap_requirement': 'full_containment',
        'splice_relative_window': {'anchor_site': 'acceptor',
                                   'reference_transcript_id': 'ENST000001.1',
                                   'relative_start_nt': -30, 'relative_end_nt': 15},
    }]
    current = envelope(goal)
    protocol = {
        'current_request_envelope': current,
        'design_goal_locations': [{'json_pointer': '/tasks/0/context/structured_goal'}],
    }
    path = '/tasks/0/context/structured_goal/design_programs/0/region_constraints/0/operator'
    patch = {'op': 'replace', 'path': path, 'value': 'union',
             'evidence_refs': ['request'], 'evidence_quote': 'a requested region',
             'reason': 'Preserve one requested region.'}
    prompt = 'Audit.\n\n' + json.dumps({**protocol, 'allowed_nonempty_channels': ['patches']})
    schema = pipeline._goal_contract_audit_schema_for_prompt(prompt)
    validator = Draft202012Validator(schema['properties']['patches']['items'])
    assert list(validator.iter_errors(patch))
    normalized = pipeline._canonicalize_existing_member_additions({'patches': [patch]}, protocol)
    assert normalized['patches'][0]['op'] == 'add'
    assert not list(validator.iter_errors(normalized['patches'][0]))
    assert patch['op'] == 'replace'

    unknown = {**patch, 'path': path + '/unknown'}
    assert pipeline._canonicalize_existing_member_additions({'patches': [unknown]}, protocol)['patches'][0]['op'] == 'replace'
    protocol['canonical_contract_issues'] = [{'json_pointer': path}]
    assert pipeline._canonicalize_existing_member_additions({'patches': [patch]}, protocol)['patches'][0]['op'] == 'replace'


def test_semantic_audit_null_replace_of_optional_member_uses_remove_only_when_offered():
    goal = design_goal()
    goal['target_region'] = 'exploratory'
    current = envelope(goal)
    optional = '/tasks/0/context/structured_goal/target_region'
    required = '/tasks/0/context/structured_goal/candidate_minimum'
    protocol = {
        'current_request_envelope': current,
        'design_goal_locations': [{'json_pointer': '/tasks/0/context/structured_goal'}],
        'required_goal_fields': pipeline._synthesized_design_goal_schema()['required'],
    }
    patches = [
        {'op': 'replace', 'path': optional, 'value': None},
        {'op': 'replace', 'path': required, 'value': None},
        {'op': 'replace', 'path': optional + '/unknown', 'value': None},
    ]
    normalized = pipeline._canonicalize_existing_member_additions({'patches': patches}, protocol)
    assert [patch['op'] for patch in normalized['patches']] == ['remove', 'replace', 'replace']
    assert [patch['op'] for patch in patches] == ['replace'] * 3


def test_future_output_refs_do_not_block_generated_artifact_analysis(tmp_path):
    current = envelope(design_goal())
    future_csv = str(tmp_path / 'new_panel.csv')
    user_input = str(tmp_path / 'missing_user_input.csv')
    current['request_text'] = 'Design a panel and summarize its generated screening evidence.'
    current['tasks'][0]['context']['structured_goal']['artifact_refs'] = [
        {'artifact_type': 'output_csv', 'path': future_csv},
        {'artifact_type': 'user_input', 'path': user_input},
    ]
    current['tasks'].append({
        'task_id': 'analysis', 'task_type': 'analyze_artifact', 'label': 'Summarize generated evidence',
        'native': True, 'can_run_in_parallel': False,
        'context': {'artifact_refs': [{'artifact_type': 'design_output',
                                      'path': 'generated by design'}]},
        'depends_on': ['design'],
    })
    current['task_dependencies'] = [{'upstream_task_id': 'design',
                                     'downstream_task_id': 'analysis'}]
    current['final_outputs'] = [{'kind': 'design_panel_csv', 'task_id': 'design',
                                 'path': future_csv, 'format': 'csv'}]
    pipeline._discard_unmaterialized_output_refs(current)
    assert current['tasks'][0]['context']['structured_goal']['artifact_refs'] == [
        {'artifact_type': 'user_input', 'path': user_input}]
    assert current['tasks'][1]['context']['artifact_refs'] == []
    analysis = pipeline._request_validation_summary(current)['tasks']['analysis']
    assert analysis['status'] == 'runnable'
    assert analysis['issues'] == []

    # A missing user-supplied file is still a blocker even alongside generated work.
    current['tasks'][1]['context']['artifact_refs'] = [
        {'artifact_type': 'user_input', 'path': user_input}]
    pipeline._discard_unmaterialized_output_refs(current)
    assert 'artifact_refs_unavailable' in pipeline._request_validation_summary(current)['tasks']['analysis']['issues']


def test_species_disagreement_reaches_evidence_repair_without_becoming_an_automatic_override(monkeypatch):
    current = envelope(design_goal())
    payload = {"status": "ready", "questions": [], "reason": None, "request_envelope": current}
    assert pipeline._request_workbench_contract_errors(payload) == []
    validation = pipeline._request_validation_summary(current)
    record = next(r for r in pipeline._goal_contract_issue_records(current, validation)
                  if r["validation_issue"] == "context.species")
    assert record["json_pointer"] == "/tasks/0/context/species"
    assert record["required_operation"] == "replace"
    assert current["tasks"][0]["context"]["species"] == "human and mouse"
    calls = []
    async def audit(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            assert '"json_pointer": "/tasks/0/context/species"' in prompt
            return {"status": "repairable", "patches": [{"op": "replace", "path": record["json_pointer"],
                "value": "human", "evidence_refs": ["request"], "evidence_quote": "human anchor",
                "reason": "Use the explicitly supplied anchor while retaining both targets."}], "unresolved": []}
        return {"status": "valid", "patches": [], "unresolved": []}
    monkeypatch.setattr(pipeline, "_run_goal_contract_audit_agent_once", audit)
    result = pipeline._repair_request_envelope_if_needed(request_text="Use a human anchor and mouse comparator in cultured cells.",
        run_dir="", structured_goal={}, request_envelope=current, qa_history=[], semantic_audit=True)
    assert result["status"] == "ready", result
    repaired = result["request_envelope"]["tasks"][0]["context"]
    assert repaired["species"] == repaired["structured_goal"]["species"] == "human"
    assert len(repaired["structured_goal"]["species_targets"]) == 2
    assert len(calls) == 2
    assert "context.species" not in pipeline._request_envelope_validation_issues(result["request_envelope"])


def test_missing_goal_stays_an_open_repair_obligation_after_a_valid_verdict(monkeypatch):
    current = envelope()
    current["tasks"][0]["context"]["species"] = "human"
    calls = []
    async def audit(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            assert '"prior_rejected_patches"' in prompt
            assert '"json_pointer": "/tasks/0/context/structured_goal"' in prompt
            return {"status": "repairable", "patches": [], "goal_replacements": [{"task_index": 0,
                "task_id": "design", "structured_goal": design_goal(), "evidence_refs": ["request"],
                "evidence_quote": "Conserved design", "reason": "Compile the supplied design intent."}], "unresolved": []}
        return {"status": "valid", "patches": [], "unresolved": []}
    monkeypatch.setattr(pipeline, "_run_goal_contract_audit_agent_once", audit)
    result = pipeline._repair_request_envelope_if_needed(request_text="Conserved design in human and mouse cultured cells.",
        run_dir="", structured_goal={}, request_envelope=current, qa_history=[], semantic_audit=True)
    assert result["status"] == "ready", result
    assert len(calls) == 3
    assert result["request_envelope"]["tasks"][0]["context"]["structured_goal"]["design_programs"]


def test_schema_retry_receives_the_rejected_values_and_original_request(monkeypatch):
    schema = {"type": "object", "properties": {"minimum": {"type": "integer"}, "kind": {"const": "panel"}},
              "required": ["minimum", "kind"], "additionalProperties": False}
    calls = []
    async def run(agent, *, input, max_steps):
        calls.append(input)
        if len(calls) == 1:
            return SimpleNamespace(final_output='{"minimum":0}')
        rejected = input.split("Rejected response:\n", 1)[1].split("\n\n", 1)[0]
        assert json.loads(rejected) == {"minimum": 0}
        assert "Empty results are acceptable" in input
        return SimpleNamespace(final_output='{"minimum":0,"kind":"panel"}')
    monkeypatch.setattr(pipeline.Runner, "run", run)
    result = asyncio.run(pipeline._run_schema_validated_agent_once(agent=object(),
        prompt="Empty results are acceptable", schema=schema, error_prefix="neutral", max_attempts=2))
    assert result == {"minimum": 0, "kind": "panel"}


def test_declared_reference_directory_contents_reach_only_the_selected_task(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "arbitrary-annotation.gtf").write_text('7\tsource\texon\t11\t30\t.\t+\t.\ttranscript_id "TX_A";\n')
    (inputs / "settings.json").write_text('{"length":21,"maximum":0}')
    (inputs / ".private.txt").write_text("HIDDEN_VALUE")
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("SIBLING_ONLY")
    (inputs / "escape.txt").symlink_to(sibling)
    current = envelope(design_goal())
    current["tasks"][0]["context"]["artifact_refs"] = [str(inputs)]
    other = deepcopy(current["tasks"][0])
    other["task_id"] = "other"
    other["context"]["artifact_refs"] = [str(sibling)]
    current["tasks"].append(other)
    prompt = pipeline._design_goal_synthesis_prompt(request_evidence="Use supplied references",
        request_envelope=current, task_index=0)
    evidence = json.loads(prompt.split("\n\n", 1)[1])["declared_artifact_evidence"]
    files = json.loads(evidence[0]["text"])["key_file_excerpts"]
    assert "TX_A" in files["arbitrary-annotation.gtf"]
    assert json.loads(files["settings.json"]) == {"length": 21, "maximum": 0}
    assert "HIDDEN_VALUE" not in prompt and "SIBLING_ONLY" not in prompt


def test_annotation_preview_is_bounded_and_marks_partial_evidence(tmp_path):
    reference = tmp_path / "annotation.gff3"
    reference.write_text("a" * 10000)
    preview = pipeline._artifact_excerpt(reference, max_chars=32)
    assert preview == "a" * 32 + "\n... <truncated>"


def test_sandbox_receives_scalar_json_contents_in_addition_to_the_table_index(tmp_path):
    source = tmp_path / "arbitrary-contract.json"
    source.write_text('{"length":21,"maximum":0,"unknown":null}')
    declared = pipeline._build_sandbox_declared_inputs({}, {"artifact_refs": [str(source)]})
    entry = declared["artifact_contents"][0]
    assert entry["path"] == str(source) and entry["exists"]
    assert json.loads(entry["excerpt"]) == {"length": 21, "maximum": 0, "unknown": None}


def test_clarification_boundary_can_inspect_the_already_supplied_file(monkeypatch, tmp_path):
    source = tmp_path / "supplied-table.csv"
    source.write_text("id,value\na,0\nb,NA\n")
    current = envelope(design_goal())
    current["tasks"][0]["context"]["artifact_refs"] = [str(source)]
    async def run(agent, *, input, max_steps):
        data = json.loads(input)
        evidence = json.loads(data["declared_artifact_evidence"][0]["text"])
        assert evidence["excerpt"] == source.read_text()
        return SimpleNamespace(final_output=json.dumps({"blocking_question_indexes": [],
            "additional_blocking_questions": [], "reason": "The table has been supplied."}))
    monkeypatch.setattr(pipeline.Runner, "run", run)
    result = asyncio.run(pipeline._review_request_clarifications_if_needed("Use the table", {
        "status": "need_clarification", "questions": ["Where are the table contents?"],
        "reason": "Input data missing", "request_envelope": current}))
    assert result["status"] == "ready" and result["questions"] == []


def audit_prompt(current, required):
    return pipeline._goal_audit_prompt(request_envelope=current, validation_issues=[], issue_records=[],
        evidence_sources=[{"evidence_id": "request", "kind": "request", "text": "Document an empty result when needed."}],
        semantic_audit=True, required_goal_fields=required)


@pytest.mark.parametrize("anchors,expected", [
    (["acceptor"], "acceptor"),
    (["donor"], "donor"),
    (["acceptor", "donor"], "both"),
])
def test_splice_audit_patch_schema_uses_each_programs_own_window_anchors(anchors, expected):
    from jsonschema import Draft202012Validator
    current = envelope(design_goal())
    program = current["tasks"][0]["context"]["structured_goal"]["design_programs"][0]
    program.update(intent="splice_modulation", site_kind="both", region_constraints=[
        {"splice_relative_window": {"anchor_site": anchor}}
        for anchor in anchors
    ])
    pointer = "/tasks/0/context/structured_goal/design_programs/0/site_kind"
    prompt = pipeline._goal_audit_prompt(
        request_envelope=current,
        validation_issues=["context.structured_goal.design_programs[0].site_kind"],
        issue_records=[{"task_id": "design", "validation_issue": "context.structured_goal.design_programs[0].site_kind",
                        "json_pointer": pointer, "required_operation": "replace",
                        "expected_schema": {"type": "string", "enum": ["acceptor", "donor", "both"]}}],
        evidence_sources=[{"evidence_id": "request", "kind": "request", "text": "Use the named splice windows."}],
        semantic_audit=True,
    )
    schema = pipeline._goal_contract_audit_schema_for_prompt(prompt)
    patch_schema = schema["properties"]["patches"]["items"]
    patch = {"op": "replace", "path": pointer, "value": expected,
             "evidence_refs": ["request"], "evidence_quote": "Use the named splice windows.",
             "reason": "Match the program's own window anchors."}
    assert not list(Draft202012Validator(patch_schema).iter_errors(patch))
    for other in {"acceptor", "donor", "both"} - {expected}:
        assert list(Draft202012Validator(patch_schema).iter_errors({**patch, "value": other}))


def test_fresh_audit_schema_cannot_delete_the_producer_required_count_fields():
    from jsonschema import Draft202012Validator
    current = envelope(design_goal())
    required = pipeline._synthesized_design_goal_schema()["required"]
    prompt = audit_prompt(current, required)
    protocol = pipeline._goal_audit_protocol_payload(prompt)
    assert set(protocol["canonical_requirement_catalog"]["context.structured_goal"]) == set(required)
    schema = pipeline._goal_contract_audit_schema_for_prompt(prompt)
    validator = Draft202012Validator(schema["properties"]["patches"]["items"])
    patch = {"op": "remove", "path": "/tasks/0/context/structured_goal/candidate_minimum", "value": None,
             "evidence_refs": ["request"], "evidence_quote": "empty result", "reason": "Remove an optional field"}
    assert list(validator.iter_errors(patch))
    patch.update(path="/tasks/0/context/structured_goal/candidate_limit")
    assert list(validator.iter_errors(patch))
    # Evidence-based corrections remain possible; only silent deletion is prohibited.
    patch.update(op="replace", path="/tasks/0/context/structured_goal/candidate_minimum", value=2)
    assert not list(validator.iter_errors(patch))
    assert "/tasks/0/context/structured_goal/candidate_minimum" in pipeline._canonical_optional_remove_paths(current)


def test_new_goal_replacements_obey_the_same_required_field_contract():
    from jsonschema import Draft202012Validator
    required = pipeline._synthesized_design_goal_schema()["required"]
    schema = pipeline._goal_contract_audit_schema_for_prompt(audit_prompt(envelope(), required))
    validator = Draft202012Validator(schema["properties"]["goal_replacements"]["items"]["properties"]["structured_goal"])
    candidate = design_goal()
    assert not list(validator.iter_errors(candidate))
    del candidate["candidate_minimum"]
    assert list(validator.iter_errors(candidate))


def test_required_field_loss_is_repaired_before_audited_envelope_can_execute(monkeypatch):
    current = envelope(design_goal())
    current["tasks"][0]["context"]["species"] = "human"
    calls = []
    async def audit(prompt):
        calls.append(prompt)
        pointer = "/tasks/0/context/structured_goal/candidate_minimum"
        if len(calls) == 3:
            return {"status": "valid", "patches": [], "unresolved": []}
        if len(calls) == 2:
            assert '"json_pointer": "' + pointer + '"' in prompt
            assert '"required_operation": "add"' in prompt
        return {"status": "repairable", "patches": [{"op": "remove" if len(calls) == 1 else "add",
            "path": pointer, "value": None if len(calls) == 1 else 0, "evidence_refs": ["request"],
            "evidence_quote": "empty result", "reason": "Preserve the requested empty-result option"}], "unresolved": []}
    monkeypatch.setattr(pipeline, "_run_goal_contract_audit_agent_once", audit)
    result = pipeline._repair_request_envelope_if_needed(request_text="An empty result is acceptable in cultured cells.",
        run_dir="", structured_goal={}, request_envelope=current, qa_history=[], semantic_audit=True,
        required_goal_fields=pipeline._synthesized_design_goal_schema()["required"])
    assert result["status"] == "ready", result
    assert len(calls) == 3
    assert result["request_envelope"]["tasks"][0]["context"]["structured_goal"]["candidate_minimum"] == 0


def test_absent_field_removal_is_normalized_before_phase_path_validation(monkeypatch):
    current = envelope(design_goal())
    calls = []
    raw = {"status": "repairable", "patches": [{"op": "remove",
        "path": "/tasks/0/context/structured_goal/design_programs/0/notes", "value": None,
        "evidence_refs": ["request"], "evidence_span_id": "request.span.0", "reason": "Remove unsupported notes"}],
        "goal_replacements": [], "field_dispositions": [], "unresolved": []}
    async def run(agent, *, input, max_steps):
        calls.append(input)
        return SimpleNamespace(final_output=json.dumps(raw))
    monkeypatch.setattr(pipeline.Runner, "run", run)
    result = asyncio.run(pipeline._run_goal_contract_audit_agent_once(audit_prompt(current, [])))
    assert result["status"] == "valid" and result["patches"] == [], result
    assert len(calls) == 1 and len(raw["patches"]) == 1


def test_effective_illegal_patch_is_still_rejected_at_the_same_boundary(monkeypatch):
    current = envelope(design_goal())
    async def run(agent, *, input, max_steps):
        return SimpleNamespace(final_output=json.dumps({"status": "repairable", "patches": [{"op": "remove",
            "path": "/tasks/0/context/structured_goal/design_programs", "value": None,
            "evidence_refs": ["request"], "evidence_span_id": "request.span.0", "reason": "Delete required programs"}],
            "goal_replacements": [], "field_dispositions": [], "unresolved": []}))
    monkeypatch.setattr(pipeline.Runner, "run", run)
    result = asyncio.run(pipeline._run_goal_contract_audit_agent_once(audit_prompt(current, [])))
    assert result["error"] == "invalid_structured_output"
    assert current["tasks"][0]["context"]["structured_goal"]["design_programs"]


def test_malformed_patch_entries_cannot_be_discarded_as_noops(monkeypatch):
    async def run(agent, *, input, max_steps):
        return SimpleNamespace(final_output=json.dumps({"status": "repairable", "patches": ["remove some field"],
            "goal_replacements": [], "field_dispositions": [], "unresolved": []}))
    monkeypatch.setattr(pipeline.Runner, "run", run)
    result = asyncio.run(pipeline._run_goal_contract_audit_agent_once(audit_prompt(envelope(design_goal()), [])))
    assert result["error"] == "invalid_structured_output"


def clarification_candidate():
    return {"status": "needs_clarification", "contradiction_record_indexes": [],
        "record_findings": [{"record_index": 0, "identity_requirement": "A target mapping is missing",
            "requirement_kind": "positive_mapping", "status": "not_verifiable", "evidence_assessment": "Mapping unavailable."}],
        "question": "Which supplied reference defines the intended target?", "reason": "Target identity remains unresolved."}


def test_clarification_index_summary_is_derived_without_changing_any_finding():
    candidate = clarification_candidate()
    original = deepcopy(candidate)
    result = pipeline._recover_resolved_target_consistency_clarification(candidate,
        [{"record_index": 0}], pipeline._resolved_target_consistency_schema(1))
    assert result == {**candidate, "contradiction_record_indexes": [0]}
    assert candidate == original
    assert result["status"] == "needs_clarification"


@pytest.mark.parametrize("mutation", ["question", "coverage", "index", "status"])
def test_summary_recovery_cannot_supply_missing_scientific_or_clarification_content(mutation):
    candidate = clarification_candidate()
    if mutation == "question": candidate["question"] = None
    if mutation == "coverage": candidate["record_findings"] = []
    if mutation == "index": candidate["record_findings"][0]["record_index"] = 3
    if mutation == "status": candidate["status"] = "consistent"
    result = pipeline._recover_resolved_target_consistency_clarification(candidate,
        [{"record_index": 0}], pipeline._resolved_target_consistency_schema(1))
    assert result is None
