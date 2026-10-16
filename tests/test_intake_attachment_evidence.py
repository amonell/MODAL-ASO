import asyncio
import hashlib
import json
from pathlib import Path

from aso_agents import pipeline_tools as pipeline
from aso_agents.request_artifact_evidence import collect_request_artifacts, append_request_artifact_evidence


def test_complete_small_table_is_available_before_first_planner_and_question_review(tmp_path, monkeypatch):
    table = tmp_path / "molecules.csv"
    raw = b"name,sequence,chemistry\r\na,AACCGGTT,steric\r\nb,AACCGGTT,gapmer\r\n"
    table.write_bytes(raw)
    request = f"Evaluate every molecule in {table}. Return {tmp_path / 'result.csv'}."
    calls = []
    async def plan(prompt):
        calls.append(prompt)
        assert 'a,AACCGGTT,steric' in prompt and 'b,AACCGGTT,gapmer' in prompt
        assert hashlib.sha256(raw).hexdigest() in prompt
        return {"status": "need_clarification", "questions": ["Which tissue?"], "request_envelope": None}
    async def review(prompt, result):
        assert prompt == calls[0]
        assert request in prompt
        return {"error": "test_boundary_reached"}
    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    monkeypatch.setattr(pipeline, "_review_request_clarifications_if_needed", review)
    result = asyncio.run(pipeline._run_verified_request_workbench_agent_once(request))
    assert result == {"error": "test_boundary_reached"}
    assert len(calls) == 1 and table.read_bytes() == raw


def test_only_original_user_paths_are_read_not_generated_proposal_paths(tmp_path, monkeypatch):
    supplied = tmp_path / "supplied.txt"
    proposed = tmp_path / "unrelated.txt"
    supplied.write_text("available user data")
    proposed.write_text("not authorized by original request")
    original = f"Analyze {supplied}"
    prompt = original + f"\nGenerated draft claims source {proposed}"
    async def plan(value):
        assert 'available user data' in value
        assert 'not authorized by original request' not in value
        return {"error": "test_boundary_reached"}
    monkeypatch.setattr(pipeline, "_run_request_workbench_agent_once", plan)
    asyncio.run(pipeline._run_verified_request_workbench_agent_once(prompt, request_evidence=original))


def test_quoted_spaced_path_and_duplicate_reference_keep_one_source(tmp_path):
    table = tmp_path / "my table.tsv"
    table.write_text("name\tsequence\na\tACGT\n")
    context = collect_request_artifacts(pipeline, f'Analyze "{table}" and `{table}`.')
    assert len(context['entries']) == 1
    assert context['entries'][0]['declared_path'] == str(table)
    assert context['entries'][0]['preview_truncated'] is False


def test_long_file_is_prefix_bound_and_does_not_claim_full_hash_or_completeness(tmp_path):
    path = tmp_path / 'large.fasta'
    raw = b'>reference\n' + b'ACGT' * 10000
    path.write_bytes(raw)
    item = collect_request_artifacts(pipeline, str(path))['entries'][0]
    assert item['preview_truncated'] is True
    assert 'file_sha256' not in item
    assert item['preview_bytes_sha256'] == hashlib.sha256(raw[:16000]).hexdigest()
    assert len(item['content'].encode()) == 16000


def test_preview_cap_and_missing_destination_are_explicit(tmp_path):
    files = [tmp_path / f'input{i}.txt' for i in range(10)]
    for p in files:
        p.write_text(p.name)
    request = ' '.join(str(p) for p in files) + f" Return {tmp_path / 'new-output.csv'}."
    context = collect_request_artifacts(pipeline, request)
    assert len(context['entries']) == 8
    assert context['omitted_existing_references'] == 2
    assert all(Path(item['resolved_path']).exists() for item in context['entries'])


def test_directory_preview_never_reads_undeclared_children(tmp_path):
    (tmp_path / 'secret.txt').write_text('THIS CONTENT MUST NOT BE READ')
    context = collect_request_artifacts(pipeline, str(tmp_path))
    assert context['entries'][0]['names'] == ['secret.txt']
    assert 'THIS CONTENT' not in json.dumps(context)


def test_corrupt_workbook_is_available_but_unreadable_not_empty_or_a_crash(tmp_path):
    path = tmp_path / 'bad.xlsx'
    path.write_text('not an archive')
    item = collect_request_artifacts(pipeline, str(path))['entries'][0]
    assert item['exists'] is True and item['content_status'] == 'unreadable'
    assert item['error'] and 'content' not in item


def test_workbook_preserves_text_and_formulas_as_bounded_preview(tmp_path):
    from openpyxl import Workbook
    path = tmp_path / 'input.xlsx'
    book = Workbook()
    sheet = book.active
    sheet.append(['name', 'sequence', 'formula'])
    sheet.append(['oligo1', 'AACCGGTT', '=1+1'])
    book.save(path)
    item = collect_request_artifacts(pipeline, str(path))['entries'][0]
    assert item['workbook_preview']['sheets'][0]['rows'][1] == ['oligo1', 'AACCGGTT', '=1+1']
    assert item['preview_truncated'] is True


def test_no_local_inputs_leaves_prompt_unchanged(tmp_path):
    prompt = f"Design ASOs and write {tmp_path / 'future.csv'}."
    assert append_request_artifact_evidence(pipeline, prompt, prompt) == prompt
