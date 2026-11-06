from aso_agents import pipeline_tools as pipeline
from aso_agents.splice_execution_meaning import splice_execution_meaning


def test_audit_exposes_execution_that_contradicts_a_correct_prose_label():
    goal = {"design_programs": [{"program_id": "reduce_skipping", "intent": "splice_modulation",
            "objective": "Reduce skipping of exon 6.", "desired_effect": "skipping", "desired_delta": "decrease", "exon_label": 6}]}
    envelope = {"tasks": [{"task_id": "design", "task_type": "design_new_asos", "context": {"structured_goal": goal}}]}
    prompt = pipeline._goal_audit_prompt(request_envelope=envelope, validation_issues=[], issue_records=[],
        evidence_sources=[{"evidence_id": "request", "text": "Reduce skipping of exon 6."}], semantic_audit=True)
    protocol = pipeline._goal_audit_protocol_payload(prompt)
    summary = protocol["splice_execution_meaning"][0]
    assert summary["goal_pointer"] == "/tasks/0/context/structured_goal"
    assert summary["effects"][0]["execution_meaning"] == "Promote exon skipping; decrease exon inclusion."
    assert summary["effects"][0]["pointer"] == "/design_programs/0"
    assert "reducing skipping means promoting inclusion" in protocol["splice_effect_semantics"]


def test_isoform_abundance_and_exon_usage_keep_their_distinct_meanings():
    summary = splice_execution_meaning({"splice_isoform_objectives": [
        {"objective_scope": "specific_isoform", "isoform_label": "skipped product", "desired_abundance": "increase", "desired_effect": "skipping"},
        {"objective_scope": "specific_isoform", "isoform_label": "included product", "desired_abundance": "decrease", "desired_effect": "inclusion"},
        {"objective_scope": "default_exon_context", "exon_label": 6, "desired_effect": "inclusion", "desired_abundance": "increase"}]})
    assert [row["execution_meaning"] for row in summary] == [
        "Increase the abundance of the identified isoform.",
        "Decrease the abundance of the identified isoform.",
        "Promote exon inclusion; decrease exon skipping."]


def test_prose_alone_has_no_executable_splice_meaning():
    assert splice_execution_meaning({"design_programs": [
        {"intent": "splice_modulation", "objective": "Increase skipping", "label": "retain", "notes": "include"},
        {"intent": "binding", "objective": "Skip the long report"}]}) == []
