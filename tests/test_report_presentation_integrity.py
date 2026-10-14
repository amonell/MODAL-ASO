"""Scientific presentation invariants independent of any target or benchmark."""
import json
import os
import shutil
import subprocess
from pathlib import Path

from aso_agents.report_presentation import (
    append_requested_downloads, display_request, final_candidate_panel, inline_markdown, link_child_report_excerpts, local_report_artifact, markdown_html,
)
from aso_agents.run_layout import canonical_run_path


def put(root, name, value):
    path = canonical_run_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_links_are_real_portable_and_do_not_follow_another_run(tmp_path):
    root = tmp_path / 'run with spaces'; root.mkdir()
    artifact = root / 'data & evidence.csv'; artifact.write_text('id,value\na,0\n')
    before = artifact.read_bytes()
    page = markdown_html('[Download](<data & evidence.csv>) and `data & evidence.csv`.', root)
    assert page.count('href="data%20%26%20evidence.csv"') == 2
    assert 'file unavailable' not in page
    assert artifact.read_bytes() == before
    assert 'href=' not in markdown_html('[Missing](no.csv)', root)
    assert 'file unavailable' in markdown_html('[Missing](no.csv)', root)
    other = tmp_path / 'different'; other.mkdir(); (other / artifact.name).write_text('wrong')
    assert local_report_artifact(root, str(other / artifact.name)) is None
    assert local_report_artifact(root, '../different/data & evidence.csv') is None
    simple = root/'reference.json'; simple.write_text('{}')
    linked = markdown_html('Full structures: `'+str(simple)+'`.', root)
    assert 'href="reference.json"' in linked and str(tmp_path) not in linked
    bare_root = tmp_path/'unspaced'; bare_root.mkdir(); (bare_root/'reference.json').write_text('{}')
    assert 'href="reference.json"' in markdown_html('Full structures: '+str(bare_root/'reference.json')+'.', bare_root)


def test_markdown_preserves_chemical_notation_fences_and_zero():
    value = '/52MOErA/*A*T*/32MOErG/'
    result = markdown_html(f'# Chemistry\n\n`{value}`\n\n```text\n{value}\n<script>alert(1)</script>\n```\n\n| ID | Value |\n|---|---:|\n| a | 0 |\n')
    assert value in result and '<em>' not in result
    assert '<script>alert(1)</script>' not in result
    assert '&lt;script&gt;' in result
    assert 'class="table-scroll"' in result and '<td>0</td>' in result
    assert '```' not in result
    assert 'href=' not in inline_markdown('[bad](javascript:alert)')


def test_embedded_fragment_keeps_its_own_evidence_links(tmp_path):
    from aso_agents.report_presentation import read_markdown_fragment
    root = tmp_path / 'run with spaces'
    child = root / 'child_runs' / 'arm' / 'artifacts'
    child.mkdir(parents=True)
    (root / 'evidence.csv').write_text('unrelated top-level record')
    (child / 'evidence.csv').write_text('actual scoped evidence')
    (tmp_path / 'private.csv').write_text('outside the report bundle')
    fragment = child / 'panel.md'
    fragment.write_text('[Evidence](evidence.csv) [Missing](missing.csv) '
                        '[Outside](../../../../private.csv) [Anchor](#section) '
                        '[External](https://example.org) `literal [link](evidence.csv)`')
    rebased = read_markdown_fragment(fragment, root)
    assert '[Evidence](<child_runs/arm/artifacts/evidence.csv>)' in rebased
    assert '[Missing](<child_runs/arm/artifacts/missing.csv>)' in rebased
    assert 'Outside (file unavailable)' in rebased
    assert '[Anchor](#section)' in rebased
    assert '[External](https://example.org)' in rebased
    assert '`literal [link](evidence.csv)`' in rebased
    page = markdown_html(rebased, root)
    assert 'href="child_runs/arm/artifacts/evidence.csv"' in page
    assert 'href="evidence.csv"' not in page


def test_child_report_shows_typed_scope_without_execution_instructions(tmp_path):
    from aso_agents.report_explainer import _request_text_from_payload
    run = tmp_path/'parent'/'child_runs'/'arm'; run.mkdir(parents=True)
    raw = 'INTERNAL_EXECUTION_INSTRUCTIONS_AND_SIBLING_CONTEXT'
    payload = {'request': raw, 'request_envelope': {'tasks': [{'task_id': 'arm', 'label': 'Evaluate the requested panel.'}]}}
    assert _request_text_from_payload(payload, str(run)) == 'Evaluate the requested panel.'
    assert display_request(tmp_path/'standalone', payload['request_envelope'], raw) == raw


def test_declared_deliverables_survive_handoff_without_copying_unrelated_files(tmp_path):
    root = tmp_path/'run'; root.mkdir()
    output = tmp_path/'output'; output.mkdir()
    (output/'result.csv').write_text('id,score\na,0\n')
    (output/'unrelated.txt').write_text('Not part of the requested delivery.')
    specs = [{'path': '/output/result.csv', 'label': 'Complete results'}, {'path': '/output/missing.csv', 'label': 'Missing output'}]
    md = append_requested_downloads('# Results\n', root, specs, delivery_root=output)
    files = list((root/'artifacts/deliverables').rglob('*.csv'))
    assert len(files) == 1 and files[0].read_bytes() == (output/'result.csv').read_bytes()
    assert not list((root/'artifacts/deliverables').rglob('unrelated.txt'))
    assert 'Missing output: file unavailable' in md
    (output/'result.csv').unlink()
    assert append_requested_downloads(md, root, specs, delivery_root=output) == md
    assert 'href=' in markdown_html(md, root)


def test_bundled_evaluation_report_preserves_its_relative_evidence_assets(tmp_path):
    source = tmp_path/'exports'; source.mkdir()
    report = source/'evaluation.md'; report.write_text('[Evidence](evaluation.md.assets/version/evidence.html)')
    asset = source/'evaluation.md.assets/version/evidence.html'; asset.parent.mkdir(parents=True); asset.write_text('Exact recorded evidence.')
    root = tmp_path/'run'; root.mkdir()
    md = append_requested_downloads('# Results', root, [{'path': str(report)}])
    bundled = next((root/'artifacts/deliverables').rglob('evaluation.md'))
    assert bundled.read_bytes() == report.read_bytes()
    assert (bundled.parent/'evaluation.md.assets/version/evidence.html').read_bytes() == asset.read_bytes()
    assert append_requested_downloads(md, root, [{'path': str(report)}]) == md


def test_selected_panel_never_uses_target_or_unattested_rc_as_oligo(tmp_path):
    put(tmp_path, 'generated_aso_set.json', {'rows': [
        {'sequence': 'TARGET_ONLY', 'reverse_complement_sequence': 'UNATTESTED_RC', 'selection_rank': 0},
        {'sequence': 'TARGET', 'synthesized_antisense_sequence': 'PHYSICAL_OLIGO', 'gc_fraction': 0,
         'selection_rank': 1, 'job_id': 'separate_scope', 'molecule_specification': {'architecture': '5-10-5'}},
    ]})
    page = final_candidate_panel(tmp_path)
    assert 'PHYSICAL_OLIGO' in page
    assert 'TARGET_ONLY' not in page and 'UNATTESTED_RC' not in page
    assert '0</td>' in page and '1 (separate_scope)' in page
    assert '2 delivered native records' in page
    separate = final_candidate_panel(tmp_path, request_envelope={'final_outputs': [{'path': '/output/final.csv'}]})
    assert '<h2>Native design-stage panel</h2>' in separate
    assert '<h2>Final selected panel</h2>' not in separate
    assert 'separately requested deliverables define the final selection' in separate


def test_basic_and_composite_html_are_readable_and_portable(tmp_path):
    from aso_agents import pipeline_tools as pipeline
    child = tmp_path / 'parent' / 'child_runs' / 'arm with spaces'
    child.mkdir(parents=True)
    (child / 'report.html').write_text('<!doctype html><title>Child</title>')
    (child / 'report.md').write_text('# Child\n')
    response = {'completed_task_context': {'arm': {'task_type': 'design_new_asos', 'task_id': 'arm',
                'child_run_dir': '/old/server/parent/child_runs/arm with spaces'}}}
    page = pipeline._composite_html_document(title='Two strategies', body_markdown='# Results', response_payload=response, run_dir=str(child.parent.parent))
    assert "src='child_runs/arm%20with%20spaces/report.html'" in page
    assert 'srcdoc=' not in page and 'file://' not in page
    assert 'name=\'viewport\'' in page
    basic = pipeline._basic_html_document('Evaluation', '| ID | Value |\n|---|---|\n| a | 0 |')
    assert 'class="table-scroll"' in basic and "name='viewport'" in basic
    response['completed_task_context']['arm']['report_excerpt'] = 'TRUNCATED_INTERNAL_EXECUTION_EXCERPT'
    parent = link_child_report_excerpts('# Comparison\n\nTRUNCATED_INTERNAL_EXECUTION_EXCERPT', response, str(child.parent.parent))
    assert 'TRUNCATED_INTERNAL_EXECUTION_EXCERPT' not in parent
    assert 'child_runs/arm%20with%20spaces/report.md' in parent


def test_native_html_projects_saved_oligoai_and_does_not_require_the_network(tmp_path):
    from aso_agents.html_report import build_goal_intake_html_payload, render_goal_intake_html
    evaluation = {'schema_version': 'modal-aso-oligoai-evaluation/1.0', 'candidates': [
        {'candidate_id': 'candidate_1', 'antisense_5to3': 'ACGT', 'contexts': [], 'status': 'not_predicted', 'reason': 'Context unavailable.'}],
        'candidate_count': 1, 'predicted_count': 0}
    put(tmp_path, 'generated_aso_set.json', {'rows': [], 'oligoai_evaluation': evaluation})
    payload = build_goal_intake_html_payload(run_dir=str(tmp_path), core_payload={'meta': {}, 'targets': [],
        'llm_rationale': {'agent_trace': [{'private': 'PRIVATE_ENGINEERING_TRACE'}]}},
        report_manifest={}, structured_goal={}, final_payload={}, qa_history=[], report_markdown='# Results\n\nPartial evidence.', report_explainer={})
    assert payload['oligoai_evaluations'] == [evaluation]
    page = render_goal_intake_html(payload)
    assert 'Context unavailable.' in page
    assert 'fonts.googleapis.com' not in page and 'fonts.gstatic.com' not in page
    assert 'id="results"' in page and 'data-tab="results"' in page
    assert '["Run directory", meta.run_dir]' not in page
    assert 'PRIVATE_ENGINEERING_TRACE' not in page


def test_recipient_readable_views_remain_bound_to_exact_molecules(tmp_path):
    from aso_recipient_report import attest_record, write_recipient_report, validate_recipient_report
    from aso_chemistry import encode_idt_oligo
    seq = 'TACAAATTATGCCTGGAATA'
    row = attest_record('neutral', seq, '5-10-5', encode_idt_oligo(seq, '5-10-5'),
                        linkages=['PS']*19, assumption_status='explicit_request', native_rank=1)
    root = tmp_path/'recipient'
    write_recipient_report(root, [row], title='Chemistry', context='Computed candidates.', provenance={})
    assert validate_recipient_report(root)['records_checked'] == 1
    page = (root/'report.html').read_text()
    assert 'href="candidates.xlsx"' in page and '<table>' in page
    assert 'Not available' not in page
    # Changing a visible sequence while keeping the embedded JSON is detected.
    (root/'report.md').write_text((root/'report.md').read_text().replace('| '+seq+' |', '| WRONG |', 1))
    import pytest
    with pytest.raises(ValueError, match='Readable Markdown'):
        validate_recipient_report(root)


def test_actual_browser_sorting_and_denied_clipboard_keep_evidence_honest(tmp_path):
    import pytest
    chrome = shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:
        pytest.skip('A local Chromium browser is required for this interaction check.')
    from aso_agents.html_report import render_goal_intake_html
    page = render_goal_intake_html({})
    check = '''<script>
      const table = document.createElement('table');
      table.innerHTML='<thead><tr><th>Value</th></tr></thead><tbody><tr><td>N/A</td></tr><tr><td>0</td></tr><tr><td>-1</td></tr></tbody>';
      document.body.append(table);
      sortTable(table,0,'number');document.body.dataset.sortAsc=[...table.tBodies[0].rows].map(r=>r.innerText).join(',');
      sortTable(table,0,'number');document.body.dataset.sortDesc=[...table.tBodies[0].rows].map(r=>r.innerText).join(',');
      Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async()=>{throw Error('Denied');}}});
      document.execCommand=()=>false;
      const button=document.createElement('button');document.body.append(button);
      copyToClipboard('ACGT',button).then(()=>{
        document.body.dataset.copyFallback=String(button.innerText==='Select sequence and copy' && button.nextElementSibling.value==='ACGT');
      });
    </script>'''
    path = tmp_path/'controls.html'; path.write_text(page.replace('</body>', check+'</body>'))
    environment = {key: value for key, value in os.environ.items() if not any(word in key for word in ['API_KEY', 'TOKEN', 'SECRET'])}
    # Chrome's local IPC socket cannot use an arbitrarily long pytest node path.
    environment.update(TMPDIR=os.environ.get('TMPDIR', '/tmp'), XDG_CACHE_HOME=str(tmp_path/'cache'))
    result = subprocess.run([chrome, '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
        '--disable-background-networking', '--no-first-run', '--user-data-dir='+str(tmp_path/'chrome'),
        '--virtual-time-budget=1000', '--dump-dom', path.as_uri()], capture_output=True, text=True, env=environment, check=True, timeout=30)
    assert 'data-sort-asc="-1,0,N/A"' in result.stdout
    assert 'data-sort-desc="0,-1,N/A"' in result.stdout
    assert 'data-copy-fallback="true"' in result.stdout
